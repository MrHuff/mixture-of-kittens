#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <tuple>
#include <vector>

using namespace kittens;

namespace fp4_dispatch {

static constexpr int TILE_ROWS = 128;
static constexpr int TILE_COLS = 128;
static constexpr int DEFAULT_PULL_COLS = 640;
static constexpr int COMBINE_ROWS = 16;
static constexpr int DEFAULT_COMBINE_COLS = 0;
static constexpr int MAX_EP_SIZE = 64;

template<int PULL_COLS>
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = TILE_ROWS;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        TILE_ROWS * PULL_COLS * static_cast<int>(sizeof(bf16)) + 2048;
    static_assert(DYNAMIC_SHARED_MEMORY <= MAX_SHARED_MEMORY - 1024);
};

struct globals {
    std::array<bf16 *, MAX_EP_SIZE> x_ptrs;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    uint8_t *output;
    uint8_t *scales;
    const float *global_scale;
    int ep_size;
    int schedule_capacity;
    int row_block_start;
    int num_row_blocks;
    int hidden_size;
    int topk;
    int num_comm_sms;
    int pull_cols;

    __host__ inline dim3 grid() const {
        const int tasks = ((hidden_size + pull_cols - 1) / pull_cols)
                        * num_row_blocks;
        return dim3(std::min(tasks, num_comm_sms));
    }
};

template<int COMBINE_COLS>
struct combine_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = TILE_ROWS;
    static constexpr int PIPE_DEPTH =
        (MAX_SHARED_MEMORY - 1024)
        / (COMBINE_ROWS * COMBINE_COLS * static_cast<int>(sizeof(bf16)));
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
    static_assert(COMBINE_COLS > 0 && COMBINE_COLS % TILE_COLS == 0);
    static_assert(PIPE_DEPTH > 0);
};

struct combine_globals {
    bf16 *input;
    std::array<bf16 *, MAX_EP_SIZE> output_ptrs;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    int ep_size;
    int row_start;
    int num_rows;
    int hidden_size;
    int output_rows;
    int num_comm_sms;
    int combine_cols;
    int pipe_depth;

    __host__ inline dim3 grid() const {
        const int tiles = ((hidden_size + combine_cols - 1) / combine_cols)
                        * (num_rows / COMBINE_ROWS);
        const int tasks = (tiles + pipe_depth - 1) / pipe_depth;
        return dim3(std::min(tasks, num_comm_sms));
    }
};

static __device__ __forceinline__ uint8_t e8m0_ceil(float value) {
    if (value <= 1.0e-38f)
        return 0;
    const uint32_t bits = __float_as_uint(value);
    uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xff);
    if ((bits & 0x7fffff) != 0 && exponent < 0xfe)
        ++exponent;
    return exponent;
}

static __device__ __forceinline__ uint8_t fp8_bits(fp8e4m3 value) {
    return std::bit_cast<uint8_t>(value);
}

template <bool NVFP4>
static __device__ __forceinline__ void quantize_row(
    const bf16 *input,
    uint8_t *output,
    uint8_t *scales,
    float global_decode_scale
) {
    constexpr int BLOCK_SIZE = NVFP4 ? 16 : 32;
    constexpr int NUM_BLOCKS = TILE_COLS / BLOCK_SIZE;
    uint32_t packed_words[TILE_COLS / 8];
    uint8_t scale_bytes[NUM_BLOCKS];

    #pragma unroll
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        float block_amax = 0.0f;
        #pragma unroll
        for (int element = 0; element < BLOCK_SIZE; ++element)
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(input[block * BLOCK_SIZE + element])));

        float coefficient;
        if constexpr (NVFP4) {
            const float encode_scale = global_decode_scale > 0.0f
                                     ? 1.0f / global_decode_scale
                                     : 1.0f;
            const float multiplier_value = block_amax > 1.0e-9f
                                         ? 6.0f / (block_amax * encode_scale)
                                         : 448.0f;
            const fp8e4m3 multiplier = static_cast<fp8e4m3>(fminf(multiplier_value, FLT_MAX));
            const float decoded_multiplier = static_cast<float>(multiplier);
            const fp8e4m3 decode_scale = static_cast<fp8e4m3>(1.0f / decoded_multiplier);
            scale_bytes[block] = fp8_bits(decode_scale);
            coefficient = decoded_multiplier * encode_scale;
        } else {
            const uint8_t decode_scale = e8m0_ceil(block_amax);
            scale_bytes[block] = decode_scale;
            coefficient = decode_scale == 0
                        ? 1.0f
                        : ldexpf(6.0f, 127 - static_cast<int>(decode_scale));
        }

        #pragma unroll
        for (int group = 0; group < BLOCK_SIZE / 8; ++group) {
            uint32_t packed = 0;
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int offset = block * BLOCK_SIZE + group * 8 + pair * 2;
                const float2 values = {
                    __bfloat162float(input[offset]) * coefficient,
                    __bfloat162float(input[offset + 1]) * coefficient,
                };
                const uint8_t fp4x2 = static_cast<uint8_t>(
                    __nv_cvt_float2_to_fp4x2(values, __NV_E2M1, cudaRoundNearest));
                packed |= static_cast<uint32_t>(fp4x2) << (pair * 8);
            }
            packed_words[block * (BLOCK_SIZE / 8) + group] = packed;
        }
    }

    #pragma unroll
    for (int index = 0; index < TILE_COLS / 32; ++index) {
        reinterpret_cast<uint4 *>(output)[index] = uint4{
            packed_words[index * 4],
            packed_words[index * 4 + 1],
            packed_words[index * 4 + 2],
            packed_words[index * 4 + 3],
        };
    }

    const int row = threadIdx.x;
    const int swizzled_offset = (row % 32) * 16 + (row / 32) * 4;
    if constexpr (NVFP4) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint32_t packed_scales = static_cast<uint32_t>(scale_bytes[half * 4])
                                         | static_cast<uint32_t>(scale_bytes[half * 4 + 1]) << 8
                                         | static_cast<uint32_t>(scale_bytes[half * 4 + 2]) << 16
                                         | static_cast<uint32_t>(scale_bytes[half * 4 + 3]) << 24;
            *reinterpret_cast<uint32_t *>(scales + half * 512 + swizzled_offset) = packed_scales;
        }
    } else {
        const uint32_t packed_scales = static_cast<uint32_t>(scale_bytes[0])
                                     | static_cast<uint32_t>(scale_bytes[1]) << 8
                                     | static_cast<uint32_t>(scale_bytes[2]) << 16
                                     | static_cast<uint32_t>(scale_bytes[3]) << 24;
        *reinterpret_cast<uint32_t *>(scales + swizzled_offset) = packed_scales;
    }
}

template <bool NVFP4, int PULL_COLS>
static __device__ void kernel(const globals &g) {
    const int tid = threadIdx.x;
    extern __shared__ int shared_storage[];
    const uint64_t shared_base =
        (reinterpret_cast<uint64_t>(&shared_storage[0]) + 1023) & ~uint64_t(1023);
    auto &input_rows = *reinterpret_cast<bf16 (*)[TILE_ROWS][PULL_COLS]>(shared_base);
    __shared__ semaphore inputs_arrived;
    const int col_blocks = (g.hidden_size + PULL_COLS - 1) / PULL_COLS;
    const int total_tasks = g.num_row_blocks * col_blocks;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_block = g.row_block_start + task / col_blocks;
        const int row_offset = row_block * TILE_ROWS;
        if (row_offset >= g.num_tokens[0])
            break;

        const int global_row = row_offset + tid;
        const int col_block = task % col_blocks;
        const int col_offset = col_block * PULL_COLS;
        const int chunk_cols = min(PULL_COLS, g.hidden_size - col_offset);
        const int chunk_bytes = chunk_cols * static_cast<int>(sizeof(bf16));
        const int peer_rank = g.schedule_peer_rank[global_row];
        const int peer_token_idx = g.schedule_peer_token_idx[global_row];
        const bool valid = peer_rank >= 0 && peer_rank < g.ep_size;
        const int num_valid = __syncthreads_count(valid);

        if (tid == 0) {
            init_semaphore(inputs_arrived, 0, 1);
            tma::expect_bytes(inputs_arrived, num_valid * chunk_bytes);
        }
        __syncthreads();

        if (valid) {
            const size_t source_row = static_cast<size_t>(peer_token_idx / g.topk);
            bf16 *source = g.x_ptrs[peer_rank]
                         + source_row * g.hidden_size + col_offset;
            tma::load_async(input_rows[tid], source, chunk_bytes, inputs_arrived);
        } else {
            auto *words = reinterpret_cast<float4 *>(input_rows[tid]);
            #pragma unroll
            for (int index = 0; index < PULL_COLS * static_cast<int>(sizeof(bf16)) /
                                    static_cast<int>(sizeof(float4)); ++index)
                words[index] = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }

        wait(inputs_arrived, 0);
        __syncthreads();

        const int scale_tiles_per_row_block = NVFP4 ? g.hidden_size / 64
                                                     : g.hidden_size / 128;
        const float global_decode_scale = NVFP4 ? g.global_scale[0] : 1.0f;
        const int num_subtiles = chunk_cols / TILE_COLS;
        #pragma unroll
        for (int subtile = 0; subtile < PULL_COLS / TILE_COLS; ++subtile) {
            if (subtile < num_subtiles) {
                const int quant_col_block = col_block * (PULL_COLS / TILE_COLS)
                                          + subtile;
                uint8_t *output = g.output
                                + static_cast<size_t>(global_row) * (g.hidden_size / 2)
                                + quant_col_block * (TILE_COLS / 2);
                const int scale_col_block = NVFP4 ? quant_col_block * 2
                                                   : quant_col_block;
                uint8_t *scales = g.scales
                                + (static_cast<size_t>(row_block) * scale_tiles_per_row_block
                                   + scale_col_block) * 512;
                quantize_row<NVFP4>(
                    input_rows[tid] + subtile * TILE_COLS,
                    output,
                    scales,
                    global_decode_scale
                );
            }
        }
        __syncthreads();
    }
}

template<int COMBINE_COLS>
static __device__ void combine_kernel(const combine_globals &g) {
    constexpr int COMBINE_PIPE_DEPTH =
        combine_config<COMBINE_COLS>::PIPE_DEPTH;
    const int tid = threadIdx.x;
    const bool is_worker = tid < COMBINE_ROWS;
    extern __shared__ int shared_storage[];
    const uint64_t shared_base =
        (reinterpret_cast<uint64_t>(&shared_storage[0]) + 1023) & ~uint64_t(1023);
    auto &rows = *reinterpret_cast<
        bf16 (*)[COMBINE_PIPE_DEPTH][COMBINE_ROWS][COMBINE_COLS]
    >(shared_base);
    __shared__ semaphore inputs_arrived[COMBINE_PIPE_DEPTH];
    const int col_blocks = (g.hidden_size + COMBINE_COLS - 1) / COMBINE_COLS;
    const int total_tiles = (g.num_rows / COMBINE_ROWS) * col_blocks;

    for (int first_tile = blockIdx.x * COMBINE_PIPE_DEPTH;
         first_tile < total_tiles;
         first_tile += gridDim.x * COMBINE_PIPE_DEPTH) {
        int global_row[COMBINE_PIPE_DEPTH];
        int col_offset[COMBINE_PIPE_DEPTH];
        int chunk_bytes[COMBINE_PIPE_DEPTH];
        int peer_rank[COMBINE_PIPE_DEPTH];
        int peer_token_idx[COMBINE_PIPE_DEPTH];
        int num_valid[COMBINE_PIPE_DEPTH];

        #pragma unroll
        for (int stage = 0; stage < COMBINE_PIPE_DEPTH; ++stage) {
            const int tile = first_tile + stage;
            const bool valid_tile = tile < total_tiles;
            const int row_block = valid_tile ? tile / col_blocks : 0;
            const int col_block = valid_tile ? tile % col_blocks : 0;
            global_row[stage] = g.row_start + row_block * COMBINE_ROWS + tid;
            col_offset[stage] = col_block * COMBINE_COLS;
            const int chunk_cols = valid_tile
                                 ? min(COMBINE_COLS, g.hidden_size - col_offset[stage])
                                 : 0;
            chunk_bytes[stage] = chunk_cols * static_cast<int>(sizeof(bf16));
            peer_rank[stage] = valid_tile && is_worker
                             ? g.schedule_peer_rank[global_row[stage]] : -1;
            peer_token_idx[stage] = valid_tile && is_worker
                                  ? g.schedule_peer_token_idx[global_row[stage]] : -1;
            const bool valid = valid_tile && is_worker
                            && global_row[stage] < g.num_tokens[0]
                            && peer_rank[stage] >= 0
                            && peer_rank[stage] < g.ep_size
                            && peer_token_idx[stage] >= 0
                            && peer_token_idx[stage] < g.output_rows;
            num_valid[stage] = __syncthreads_count(valid);
            if (!valid) {
                peer_rank[stage] = -1;
                peer_token_idx[stage] = -1;
            }
        }

        if (tid == 0) {
            #pragma unroll
            for (int stage = 0; stage < COMBINE_PIPE_DEPTH; ++stage) {
                init_semaphore(inputs_arrived[stage], 0, 1);
                tma::expect_bytes(
                    inputs_arrived[stage],
                    num_valid[stage] * chunk_bytes[stage]
                );
            }
        }
        __syncthreads();

        #pragma unroll
        for (int stage = 0; stage < COMBINE_PIPE_DEPTH; ++stage) {
            if (peer_rank[stage] >= 0 && peer_rank[stage] < g.ep_size) {
                bf16 *source = g.input
                             + static_cast<size_t>(global_row[stage]) * g.hidden_size
                             + col_offset[stage];
                tma::load_async(
                    rows[stage][tid],
                    source,
                    chunk_bytes[stage],
                    inputs_arrived[stage]
                );
            }
        }

        #pragma unroll
        for (int stage = 0; stage < COMBINE_PIPE_DEPTH; ++stage) {
            wait(inputs_arrived[stage], 0);
            if (peer_rank[stage] >= 0 && peer_rank[stage] < g.ep_size) {
                bf16 *destination = g.output_ptrs[peer_rank[stage]]
                                  + static_cast<size_t>(peer_token_idx[stage])
                                    * g.hidden_size
                                  + col_offset[stage];
                tma::store_async(
                    destination, rows[stage][tid], chunk_bytes[stage]
                );
            }
        }
        tma::store_async_read_wait();
        __syncthreads();
    }
}

static __host__ inline void validate_inputs(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    int topk,
    int num_comm_sms
) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2,
                "x must be contiguous CUDA BF16 [tokens, hidden]");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % TILE_COLS == 0,
                "x hidden size must be positive and divisible by 128");
    TORCH_CHECK(x_ptrs.size() >= 2 && x_ptrs.size() <= MAX_EP_SIZE,
                "x_ptrs must contain between 2 and 64 peers");
    TORCH_CHECK(schedule_peer_rank.is_cuda() && schedule_peer_rank.is_contiguous()
                && schedule_peer_rank.scalar_type() == at::kInt && schedule_peer_rank.dim() == 1,
                "schedule_peer_rank must be contiguous CUDA int32 [capacity]");
    TORCH_CHECK(schedule_peer_token_idx.is_cuda() && schedule_peer_token_idx.is_contiguous()
                && schedule_peer_token_idx.scalar_type() == at::kInt && schedule_peer_token_idx.dim() == 1
                && schedule_peer_token_idx.numel() == schedule_peer_rank.numel(),
                "schedule_peer_token_idx must match schedule_peer_rank");
    TORCH_CHECK(schedule_peer_rank.numel() > 0 && schedule_peer_rank.numel() % TILE_ROWS == 0,
                "schedule capacity must be positive and divisible by 128");
    TORCH_CHECK(num_tokens.is_cuda() && num_tokens.is_contiguous()
                && num_tokens.scalar_type() == at::kInt && num_tokens.numel() == 1,
                "num_tokens must be contiguous CUDA int32 [1]");
    TORCH_CHECK(schedule_peer_rank.device() == x.device()
                && schedule_peer_token_idx.device() == x.device()
                && num_tokens.device() == x.device(),
                "dispatch tensors must be on the same CUDA device");
    TORCH_CHECK(topk > 0, "topk must be positive");
    TORCH_CHECK(num_comm_sms > 0, "num_comm_sms must be positive");
}

template <bool NVFP4>
static __host__ void launch_into(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &global_scale,
    at::Tensor &output,
    at::Tensor &scales,
    int64_t row_start,
    int64_t num_rows,
    int topk,
    int num_comm_sms,
    int pull_cols = DEFAULT_PULL_COLS
) {
    validate_inputs(x, x_ptrs, schedule_peer_rank, schedule_peer_token_idx,
                    num_tokens, topk, num_comm_sms);
    if constexpr (NVFP4) {
        TORCH_CHECK(global_scale.is_cuda() && global_scale.is_contiguous()
                    && global_scale.scalar_type() == at::kFloat && global_scale.numel() == 1
                    && global_scale.device() == x.device(),
                    "global_scale must be contiguous CUDA float32 [1]");
    }

    const int64_t capacity = schedule_peer_rank.numel();
    const int64_t hidden = x.size(1);
    TORCH_CHECK(row_start >= 0 && row_start % TILE_ROWS == 0,
                "row_start must be non-negative and divisible by 128");
    TORCH_CHECK(num_rows > 0 && num_rows % TILE_ROWS == 0,
                "num_rows must be positive and divisible by 128");
    TORCH_CHECK(row_start + num_rows <= capacity,
                "dispatch row range exceeds schedule capacity");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous()
                && output.scalar_type() == at::ScalarType::Float4_e2m1fn_x2
                && output.sizes() == at::IntArrayRef({capacity, hidden / 2})
                && output.device() == x.device(),
                "output must be contiguous CUDA FP4 [capacity, hidden / 2]");
    if constexpr (NVFP4) {
        TORCH_CHECK(scales.is_cuda() && scales.is_contiguous()
                    && scales.scalar_type() == at::ScalarType::Float8_e4m3fn
                    && scales.sizes() == at::IntArrayRef({capacity / 128, hidden / 64, 512})
                    && scales.device() == x.device(),
                    "NVFP4 scales must have production GEMM layout");
    } else {
        TORCH_CHECK(scales.is_cuda() && scales.is_contiguous()
                    && scales.scalar_type() == at::kByte
                    && scales.sizes() == at::IntArrayRef({capacity / 128, hidden / 128, 32, 16})
                    && scales.device() == x.device(),
                    "MXFP4 scales must have production GEMM layout");
    }

    globals g{};
    for (size_t peer = 0; peer < x_ptrs.size(); ++peer)
        g.x_ptrs[peer] = reinterpret_cast<bf16 *>(x_ptrs[peer]);
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.output = reinterpret_cast<uint8_t *>(output.data_ptr());
    g.scales = reinterpret_cast<uint8_t *>(scales.data_ptr());
    g.global_scale = NVFP4 ? global_scale.data_ptr<float>() : nullptr;
    g.ep_size = static_cast<int>(x_ptrs.size());
    g.schedule_capacity = static_cast<int>(capacity);
    g.row_block_start = static_cast<int>(row_start / TILE_ROWS);
    g.num_row_blocks = static_cast<int>(num_rows / TILE_ROWS);
    g.hidden_size = static_cast<int>(hidden);
    g.topk = topk;
    g.num_comm_sms = num_comm_sms;
    g.pull_cols = pull_cols;
    switch (pull_cols) {
        case 512:
            kittens::py::launch_kernel<config<512>, globals, kernel<NVFP4, 512>>(g);
            break;
        case 640:
            kittens::py::launch_kernel<config<640>, globals, kernel<NVFP4, 640>>(g);
            break;
        case 768:
            kittens::py::launch_kernel<config<768>, globals, kernel<NVFP4, 768>>(g);
            break;
        case 896:
            kittens::py::launch_kernel<config<896>, globals, kernel<NVFP4, 896>>(g);
            break;
        default:
            TORCH_CHECK(false, "pull_cols must be one of 512, 640, 768, or 896");
    }
}

template <bool NVFP4>
static __host__ std::tuple<at::Tensor, at::Tensor> launch(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &global_scale,
    int topk,
    int num_comm_sms
) {
    const int64_t capacity = schedule_peer_rank.numel();
    const int64_t hidden = x.size(1);
    auto output = at::empty(
        {capacity, hidden / 2},
        x.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    at::Tensor scales;
    if constexpr (NVFP4) {
        scales = at::empty(
            {capacity / 128, hidden / 64, 512},
            x.options().dtype(at::ScalarType::Float8_e4m3fn));
    } else {
        scales = at::empty(
            {capacity / 128, hidden / 128, 32, 16},
            x.options().dtype(at::kByte));
    }
    launch_into<NVFP4>(x, x_ptrs, schedule_peer_rank, schedule_peer_token_idx,
                       num_tokens, global_scale, output, scales, 0, capacity,
                       topk, num_comm_sms);
    return {output, scales};
}

static __host__ inline std::tuple<at::Tensor, at::Tensor> dispatch_mxfp4(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    int topk,
    int num_comm_sms
) {
    return launch<false>(x, x_ptrs, schedule_peer_rank,
                         schedule_peer_token_idx, num_tokens, at::Tensor{},
                         topk, num_comm_sms);
}

static __host__ inline std::tuple<at::Tensor, at::Tensor> dispatch_nvfp4(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &global_scale,
    int topk,
    int num_comm_sms
) {
    return launch<true>(x, x_ptrs, schedule_peer_rank,
                        schedule_peer_token_idx, num_tokens, global_scale,
                        topk, num_comm_sms);
}

static __host__ inline void dispatch_mxfp4_into(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    at::Tensor &output,
    at::Tensor &scales,
    int64_t row_start,
    int64_t num_rows,
    int topk,
    int num_comm_sms,
    int pull_cols = DEFAULT_PULL_COLS
) {
    launch_into<false>(x, x_ptrs, schedule_peer_rank, schedule_peer_token_idx,
                       num_tokens, at::Tensor{}, output, scales, row_start,
                       num_rows, topk, num_comm_sms, pull_cols);
}

static __host__ inline void dispatch_nvfp4_into(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &global_scale,
    at::Tensor &output,
    at::Tensor &scales,
    int64_t row_start,
    int64_t num_rows,
    int topk,
    int num_comm_sms,
    int pull_cols = DEFAULT_PULL_COLS
) {
    launch_into<true>(x, x_ptrs, schedule_peer_rank, schedule_peer_token_idx,
                      num_tokens, global_scale, output, scales, row_start,
                      num_rows, topk, num_comm_sms, pull_cols);
}

static __host__ inline void combine_bf16_into(
    const at::Tensor &input,
    at::Tensor &local_output,
    const std::vector<int64_t> &output_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    int64_t row_start,
    int64_t num_rows,
    int num_comm_sms,
    int combine_cols = DEFAULT_COMBINE_COLS
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous()
                && input.scalar_type() == at::kBFloat16 && input.dim() == 2
                && input.size(1) > 0 && input.size(1) % 256 == 0,
                "combine input must be contiguous CUDA BF16 [capacity, hidden]");
    TORCH_CHECK(local_output.is_cuda() && local_output.is_contiguous()
                && local_output.scalar_type() == at::kBFloat16 && local_output.dim() == 2
                && local_output.size(0) > 0 && local_output.size(1) == input.size(1)
                && local_output.device() == input.device(),
                "local combine output must be contiguous CUDA BF16 [routes, hidden]");
    TORCH_CHECK(output_ptrs.size() == 2 || output_ptrs.size() == 4
                || output_ptrs.size() == 8 || output_ptrs.size() == 16
                || output_ptrs.size() == 32 || output_ptrs.size() == 64,
                "combine output_ptrs must contain 2, 4, 8, 16, 32, or 64 peers");
    TORCH_CHECK(std::all_of(output_ptrs.begin(), output_ptrs.end(),
                            [](int64_t pointer) { return pointer > 0; }),
                "combine output_ptrs must be positive");
    TORCH_CHECK(schedule_peer_rank.is_cuda() && schedule_peer_rank.is_contiguous()
                && schedule_peer_rank.scalar_type() == at::kInt && schedule_peer_rank.dim() == 1,
                "schedule_peer_rank must be contiguous CUDA int32 [capacity]");
    TORCH_CHECK(schedule_peer_token_idx.is_cuda() && schedule_peer_token_idx.is_contiguous()
                && schedule_peer_token_idx.scalar_type() == at::kInt
                && schedule_peer_token_idx.sizes() == schedule_peer_rank.sizes(),
                "schedule_peer_token_idx must match schedule_peer_rank");
    TORCH_CHECK(num_tokens.is_cuda() && num_tokens.is_contiguous()
                && num_tokens.scalar_type() == at::kInt && num_tokens.numel() == 1,
                "num_tokens must be contiguous CUDA int32 [1]");
    TORCH_CHECK(schedule_peer_rank.device() == input.device()
                && schedule_peer_token_idx.device() == input.device()
                && num_tokens.device() == input.device(),
                "combine tensors must be on the same CUDA device");
    const int64_t capacity = schedule_peer_rank.numel();
    TORCH_CHECK(capacity > 0 && capacity % COMBINE_ROWS == 0,
                "combine schedule capacity must be positive and divisible by 16");
    TORCH_CHECK(row_start >= 0 && row_start % COMBINE_ROWS == 0,
                "combine row_start must be non-negative and divisible by 16");
    TORCH_CHECK(num_rows > 0 && num_rows % COMBINE_ROWS == 0
                && row_start + num_rows <= capacity,
                "combine num_rows must be 32-aligned and within capacity");
    TORCH_CHECK(row_start + num_rows <= input.size(0),
                "combine row range exceeds input rows");
    TORCH_CHECK(num_comm_sms > 0, "combine num_comm_sms must be positive");
    TORCH_CHECK(combine_cols == 0 || combine_cols == 512 || combine_cols == 640
                || combine_cols == 768 || combine_cols == 1024
                || combine_cols == 1280 || combine_cols == 2560,
                "combine_cols must be 0, 512, 640, 768, 1024, 1280, or 2560");

    const int selected_combine_cols = combine_cols == 0
        ? (input.size(1) % 2560 == 0 ? 2560
           : input.size(1) % 1280 == 0 ? 1280
           : 1024)
        : combine_cols;

    combine_globals g{};
    g.input = reinterpret_cast<bf16 *>(input.data_ptr());
    for (size_t peer = 0; peer < output_ptrs.size(); ++peer)
        g.output_ptrs[peer] = reinterpret_cast<bf16 *>(output_ptrs[peer]);
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.ep_size = static_cast<int>(output_ptrs.size());
    g.row_start = static_cast<int>(row_start);
    g.num_rows = static_cast<int>(num_rows);
    g.hidden_size = static_cast<int>(input.size(1));
    g.output_rows = static_cast<int>(local_output.size(0));
    g.num_comm_sms = num_comm_sms;
    g.combine_cols = selected_combine_cols;
    auto launch = [&]<int COMBINE_COLS>() {
        g.pipe_depth = combine_config<COMBINE_COLS>::PIPE_DEPTH;
        kittens::py::launch_kernel<
            combine_config<COMBINE_COLS>,
            combine_globals,
            combine_kernel<COMBINE_COLS>
        >(g);
    };
    switch (selected_combine_cols) {
        case 512: launch.template operator()<512>(); break;
        case 640: launch.template operator()<640>(); break;
        case 768: launch.template operator()<768>(); break;
        case 1024: launch.template operator()<1024>(); break;
        case 1280: launch.template operator()<1280>(); break;
        case 2560: launch.template operator()<2560>(); break;
    }
}

} // namespace fp4_dispatch
