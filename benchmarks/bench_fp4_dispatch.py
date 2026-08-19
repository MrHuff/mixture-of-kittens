import argparse
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils import init_distributed, median_rank_max_latency
from mok import functional, ops


_DEEPSEEK_ROOT = Path(__file__).resolve().parents[2]
_FP4_ROOT = Path(os.environ.get("FP4_MATMUL_ROOT", _DEEPSEEK_ROOT / "fp4_matmul"))
_MXFP4_GEMM_MAX_BATCHES = 40
_NVFP4_GEMM_MAX_BATCHES = 40


def _load_production_quantizers():
    mxfp4_dir = _FP4_ROOT / "TK_quantisation" / "mxfp4_v4"
    nvfp4_dir = _FP4_ROOT / "TK_quantisation" / "nvfp4_v5"
    for path in (mxfp4_dir, nvfp4_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"production quantizer directory not found: {path}")
        sys.path.insert(0, str(path))

    import _tk_quant_v5
    import mxfp4_quant_v4

    return mxfp4_quant_v4, _tk_quant_v5


def _load_fp4_gemms():
    mxfp4_dir = _FP4_ROOT / "ThunderKittens" / "kernels" / "gemm" / "mxfp4_gb200"
    nvfp4_dir = _FP4_ROOT / "ThunderKittens" / "kernels" / "gemm" / "nvfp4_b200"
    for path in (mxfp4_dir, nvfp4_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"FP4 GEMM directory not found: {path}")

    sys.path.insert(0, str(mxfp4_dir))
    mxfp4_gemm = importlib.import_module("_C_mx")
    sys.path.insert(0, str(nvfp4_dir))
    nvfp4_gemm = importlib.import_module("_C")
    return mxfp4_gemm, nvfp4_gemm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark MoK-style fused FP4 pull dispatch on EP2."
    )
    parser.add_argument("--tokens", type=int, default=8192, help="local tokens per EP rank")
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--experts", type=int, default=72)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument(
        "--expert-hidden",
        type=int,
        default=0,
        help="routed expert hidden size; zero skips routed-MLP integration",
    )
    parser.add_argument("--comm-sms", type=int, default=148)
    parser.add_argument(
        "--overlap-comm-sms",
        type=int,
        default=48,
        help="communication SMs after the first streamed group; zero reuses --comm-sms",
    )
    parser.add_argument(
        "--combine-sms",
        type=int,
        default=136,
        help="communication SMs used to push routed BF16 outputs back to source ranks",
    )
    parser.add_argument(
        "--sweep-combine-sms",
        default="",
        help="optional comma-separated push-SM values for integrated pipeline timing",
    )
    parser.add_argument(
        "--combine-cols",
        type=int,
        choices=(0, 512, 640, 768, 1024, 1280, 2560),
        default=0,
        help="BF16 route-push columns handled by each combine tile; zero selects automatically",
    )
    parser.add_argument(
        "--sweep-combine-cols",
        default="",
        help="optional comma-separated route-push combine tile widths",
    )
    parser.add_argument(
        "--sweep-comm-sms",
        default="",
        help="optional comma-separated communication-SM values for raw-kernel timing",
    )
    parser.add_argument(
        "--sweep-mxfp4-gemm-configs",
        default="",
        help="optional comma-separated MXFP4 batched-GEMM config IDs",
    )
    parser.add_argument(
        "--sweep-nvfp4-gemm-configs",
        default="",
        help="optional comma-separated NVFP4 pitched batched-GEMM config IDs",
    )
    parser.add_argument(
        "--mxfp4-pull-cols",
        type=int,
        choices=(0, 512, 640, 768, 896),
        default=0,
        help="MXFP4 dispatch pull width; zero uses the production default",
    )
    parser.add_argument(
        "--sweep-mxfp4-pull-cols",
        default="",
        help="optional comma-separated MXFP4 dispatch pull widths",
    )
    parser.add_argument(
        "--epilogue-tokens-per-cta",
        type=int,
        choices=(1, 2, 4, 8),
        default=2,
        help="tokens grouped in each full-MoE epilogue CTA",
    )
    parser.add_argument(
        "--sweep-epilogue-tokens-per-cta",
        default="",
        help="optional comma-separated epilogue token group sizes",
    )
    parser.add_argument(
        "--epilogue-cols-per-cta",
        type=int,
        choices=(0, 1024, 1280, 2560),
        default=0,
        help="hidden columns covered by each full-MoE epilogue CTA; zero selects automatically",
    )
    parser.add_argument(
        "--sweep-epilogue-cols-per-cta",
        default="",
        help="optional comma-separated epilogue column tile sizes",
    )
    parser.add_argument("--correctness-repeats", type=int, default=3)
    parser.add_argument(
        "--pipeline-groups",
        default="2",
        help="comma-separated expert-group counts for dispatch/compute overlap",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--benchmark-filter",
        default="",
        help="only time variants whose names contain this substring",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="bracket timed variants with cudaProfilerStart/Stop",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if world_size != 2:
        raise ValueError(f"this prototype benchmark requires EP2, got EP{world_size}")
    if args.tokens < 512 or args.tokens % 256:
        raise ValueError("--tokens must be at least 512 and divisible by 256")
    if args.hidden <= 0 or args.hidden % 128:
        raise ValueError("--hidden must be positive and divisible by 128")
    if args.experts <= 0 or args.experts % world_size:
        raise ValueError("--experts must be positive and divisible by EP size")
    if not 0 < args.topk <= args.experts:
        raise ValueError("--topk must be in [1, experts]")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.comm_sms <= 0:
        raise ValueError("--comm-sms must be positive")
    if args.overlap_comm_sms < 0:
        raise ValueError("--overlap-comm-sms must be non-negative")
    if args.combine_sms <= 0:
        raise ValueError("--combine-sms must be positive")
    if args.correctness_repeats < 2:
        raise ValueError("--correctness-repeats must be at least 2")
    if args.expert_hidden < 0 or args.expert_hidden % 128:
        raise ValueError("--expert-hidden must be zero or a positive multiple of 128")


def _comm_sms_sweep(args: argparse.Namespace) -> list[int]:
    values = [args.comm_sms]
    if args.sweep_comm_sms:
        try:
            values.extend(int(value) for value in args.sweep_comm_sms.split(","))
        except ValueError as error:
            raise ValueError("--sweep-comm-sms must be comma-separated integers") from error
    if any(value <= 0 for value in values):
        raise ValueError("all communication-SM sweep values must be positive")
    return list(dict.fromkeys(values))


def _pipeline_group_sweep(args: argparse.Namespace) -> list[int]:
    try:
        values = [int(value) for value in args.pipeline_groups.split(",")]
    except ValueError as error:
        raise ValueError("--pipeline-groups must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("all pipeline group counts must be positive")
    return list(dict.fromkeys(values))


def _combine_sms_sweep(args: argparse.Namespace) -> list[int]:
    values = [args.combine_sms]
    if args.sweep_combine_sms:
        try:
            values.extend(
                int(value) for value in args.sweep_combine_sms.split(",")
            )
        except ValueError as error:
            raise ValueError(
                "--sweep-combine-sms must be comma-separated integers"
            ) from error
    if any(value <= 0 for value in values):
        raise ValueError("all combine-SM sweep values must be positive")
    return list(dict.fromkeys(values))


def _combine_cols_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_combine_cols:
        return []
    try:
        values = [int(value) for value in args.sweep_combine_cols.split(",")]
    except ValueError as error:
        raise ValueError(
            "--sweep-combine-cols must be comma-separated integers"
        ) from error
    supported = {512, 640, 768, 1024, 1280, 2560}
    if any(value not in supported for value in values):
        raise ValueError(
            "combine widths must be 512, 640, 768, 1024, 1280, or 2560"
        )
    return list(dict.fromkeys(values))


def _mxfp4_gemm_config_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_mxfp4_gemm_configs:
        return []
    try:
        values = [
            int(value) for value in args.sweep_mxfp4_gemm_configs.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--sweep-mxfp4-gemm-configs must be comma-separated integers"
        ) from error
    if any(value < 0 or value > 9 for value in values):
        raise ValueError("MXFP4 batched-GEMM config IDs must be in [0, 9]")
    return list(dict.fromkeys(values))


def _nvfp4_gemm_config_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_nvfp4_gemm_configs:
        return []
    try:
        values = [
            int(value) for value in args.sweep_nvfp4_gemm_configs.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--sweep-nvfp4-gemm-configs must be comma-separated integers"
        ) from error
    if any(value < 0 or value > 12 for value in values):
        raise ValueError("NVFP4 pitched batched-GEMM config IDs must be in [0, 12]")
    return list(dict.fromkeys(values))


def _mxfp4_pull_cols_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_mxfp4_pull_cols:
        return []
    try:
        values = [
            int(value) for value in args.sweep_mxfp4_pull_cols.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--sweep-mxfp4-pull-cols must be comma-separated integers"
        ) from error
    supported = {512, 640, 768, 896}
    if any(value not in supported for value in values):
        raise ValueError(
            "MXFP4 dispatch pull widths must be 512, 640, 768, or 896"
        )
    return list(dict.fromkeys(values))


def _epilogue_tokens_per_cta_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_epilogue_tokens_per_cta:
        return []
    try:
        values = [
            int(value)
            for value in args.sweep_epilogue_tokens_per_cta.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--sweep-epilogue-tokens-per-cta must be comma-separated integers"
        ) from error
    supported = {1, 2, 4, 8}
    if any(value not in supported for value in values):
        raise ValueError("epilogue tokens per CTA must be 1, 2, 4, or 8")
    return list(dict.fromkeys(values))


def _epilogue_cols_per_cta_sweep(args: argparse.Namespace) -> list[int]:
    if not args.sweep_epilogue_cols_per_cta:
        return []
    try:
        values = [
            int(value)
            for value in args.sweep_epilogue_cols_per_cta.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--sweep-epilogue-cols-per-cta must be comma-separated integers"
        ) from error
    supported = {1024, 1280, 2560}
    if any(value not in supported for value in values):
        raise ValueError("epilogue columns per CTA must be 1024, 1280, or 2560")
    return list(dict.fromkeys(values))


def _make_inputs(
    rank: int,
    device: torch.device,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device, generator=generator
    )
    logits = torch.randn(tokens, experts, device=device, generator=generator)
    top_logits, top_experts = torch.topk(logits, topk, dim=1)
    router_weights = torch.softmax(top_logits, dim=1)
    return x, top_experts, router_weights


def _materialize_reference(
    x: torch.Tensor,
    schedule: functional.MoKSchedule,
    topk: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    world_size = dist.get_world_size()
    tokens, hidden = x.shape
    x_all = torch.empty(
        world_size * tokens, hidden, dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(x_all, x)
    x_all = x_all.view(world_size, tokens, hidden)

    num_tokens = int(schedule.num_tokens.item())
    peer = schedule.peer_rank[:num_tokens].to(torch.long)
    token = torch.div(
        schedule.peer_token_idx[:num_tokens], topk, rounding_mode="floor"
    ).to(torch.long)
    valid = peer >= 0
    packed = torch.zeros(
        schedule.peer_rank.numel(), hidden, dtype=x.dtype, device=x.device
    )
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    packed[positions] = x_all[peer[valid], token[valid]]
    return packed, num_tokens, valid


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _pack_e2m1_rne(values: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=values.device,
    )
    magnitudes = values.abs().float()
    distances = (magnitudes.unsqueeze(-1) - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    codes = torch.arange(8, dtype=torch.int64, device=values.device)
    even_codes = torch.where(
        candidates & ((codes & 1) == 0), codes, 8
    ).amin(dim=-1)
    magnitude_codes = torch.where(
        even_codes < 8, even_codes, distances.argmin(dim=-1)
    )
    signed_codes = magnitude_codes | ((values < 0).to(torch.int64) << 3)
    return (
        signed_codes[..., 0::2] | (signed_codes[..., 1::2] << 4)
    ).to(torch.uint8)


def _reference_mxfp4_bytes(x: torch.Tensor) -> torch.Tensor:
    rows, hidden_size = x.shape
    blocks = x.float().view(rows, hidden_size // 32, 32)
    block_amax = blocks.abs().amax(dim=-1)
    bits = block_amax.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xff).to(torch.int32)
    exponent += ((bits & 0x7fffff) != 0).to(torch.int32) * (exponent < 0xfe)
    exponent = torch.where(block_amax <= 1.0e-38, 0, exponent)
    coefficient = torch.ldexp(
        torch.full_like(block_amax, 6.0), 127 - exponent
    )
    coefficient = torch.where(exponent == 0, 1.0, coefficient)
    return _pack_e2m1_rne(blocks * coefficient.unsqueeze(-1)).view(
        rows, hidden_size // 2
    )


def _reference_nvfp4_bytes(
    x: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    blocks = x.float().view(rows, hidden_size // 16, 16)
    block_amax = blocks.abs().amax(dim=-1)
    encode_scale = torch.where(
        global_decode_scale > 0,
        global_decode_scale.reciprocal(),
        torch.ones_like(global_decode_scale),
    )
    multiplier = torch.where(
        block_amax > 1.0e-9,
        6.0 / (block_amax * encode_scale),
        448.0,
    )
    decoded_multiplier = multiplier.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    coefficient = decoded_multiplier * encode_scale
    return _pack_e2m1_rne(blocks * coefficient.unsqueeze(-1)).view(
        rows, hidden_size // 2
    )


def _assert_exact(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    row_metadata: dict[str, torch.Tensor] | None = None,
) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: metadata mismatch: actual={actual.dtype} {tuple(actual.shape)}, "
            f"expected={expected.dtype} {tuple(expected.shape)}"
        )
    actual_bytes = _as_bytes(actual)
    expected_bytes = _as_bytes(expected)
    mismatch = actual_bytes != expected_bytes
    if mismatch.any().item():
        count = int(mismatch.sum().item())
        first = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0].item())
        actual_sample = actual_bytes.flatten()[first:first + 8].tolist()
        expected_sample = expected_bytes.flatten()[first:first + 8].tolist()
        row_details = ""
        if mismatch.ndim >= 2:
            mismatch_rows = mismatch.reshape(mismatch.shape[0], -1).sum(dim=1)
            row_indices = torch.nonzero(mismatch_rows, as_tuple=False).flatten()[:8]
            row_details = (
                f", rows={row_indices.tolist()}, "
                f"row_counts={mismatch_rows[row_indices].tolist()}"
            )
            if row_metadata is not None:
                for metadata_name, metadata in row_metadata.items():
                    row_details += (
                        f", {metadata_name}={metadata[row_indices].tolist()}"
                    )
        raise AssertionError(
            f"{name}: {count}/{mismatch.numel()} bytes differ; first={first}, "
            f"actual={actual_sample}, expected={expected_sample}{row_details}"
        )


def _time(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    dist.barrier(async_op=True).block_current_stream()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in events]
    dist.barrier()
    return median_rank_max_latency(samples, device)


def _print_result(name: str, latency_ms: float, tokens: int) -> None:
    tokens_per_second = tokens * 1000.0 / latency_ms
    print(f"{name:30s} {latency_ms:8.3f} ms  {tokens_per_second:12,.0f} tok/s/rank")


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error_rms = (actual.float() - expected.float()).square().mean().sqrt()
    reference_rms = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return float((error_rms / reference_rms).item())


def main() -> None:
    args = _parse_args()
    rank, world_size, device = init_distributed()
    _validate_args(args, world_size)
    comm_sms_sweep = _comm_sms_sweep(args)
    pipeline_group_sweep = _pipeline_group_sweep(args)
    combine_sms_sweep = _combine_sms_sweep(args)
    combine_cols_sweep = _combine_cols_sweep(args)
    mxfp4_gemm_config_sweep = _mxfp4_gemm_config_sweep(args)
    nvfp4_gemm_config_sweep = _nvfp4_gemm_config_sweep(args)
    mxfp4_pull_cols_sweep = _mxfp4_pull_cols_sweep(args)
    epilogue_tokens_per_cta_sweep = _epilogue_tokens_per_cta_sweep(args)
    epilogue_cols_per_cta_sweep = _epilogue_cols_per_cta_sweep(args)
    mxfp4_quant, nvfp4_quant = _load_production_quantizers()

    min_schedule_factor = (
        (args.experts // world_size) * 256 + args.tokens * args.topk - 1
    ) // (args.tokens * args.topk)
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=max(
            0.5, min_schedule_factor / world_size
        ),
    )
    bf16_control_config = functional.MoKConfig(
        fwd_num_comm_sms=24,
        bwd_num_comm_sms=28,
        minibatch_size=4096,
        macrobatch_size=131072,
        schedule_capacity_multiplier=config.schedule_capacity_multiplier,
    )
    mxfp8_control_config = functional.MoKConfig(
        fwd_num_comm_sms=36,
        bwd_num_comm_sms=36,
        minibatch_size=4096,
        macrobatch_size=131072,
        schedule_capacity_multiplier=config.schedule_capacity_multiplier,
    )
    x, top_experts, router_weights = _make_inputs(
        rank,
        device,
        args.tokens,
        args.hidden,
        args.experts,
        args.topk,
        args.seed,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=args.tokens,
        hidden_size=args.hidden,
        topk=args.topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=args.experts // world_size,
    )
    original_peer_rank = schedule.peer_rank.clone()
    original_peer_token_idx = schedule.peer_token_idx.clone()
    reference, num_tokens, valid_rows = _materialize_reference(
        x, schedule, args.topk
    )
    num_valid_tokens = int(valid_rows.sum().item())
    valid_schedule_rows = torch.nonzero(valid_rows, as_tuple=False).flatten()
    source_routes = schedule.peer_token_idx[valid_schedule_rows]
    data_row_metadata = {
        "schedule_rows": valid_schedule_rows,
        "peers": schedule.peer_rank[valid_schedule_rows],
        "source_tokens": torch.div(
            source_routes, args.topk, rounding_mode="floor"
        ),
    }
    sample_count = min(1024, num_valid_tokens)
    sample_indices = (
        torch.arange(sample_count, device=device, dtype=torch.int64)
        * num_valid_tokens
        // sample_count
    )
    sample_schedule_rows = valid_schedule_rows[sample_indices]
    sample_reference = reference[sample_schedule_rows]
    sample_row_metadata = {
        name: values[sample_indices] for name, values in data_row_metadata.items()
    }
    initialized_rows = min(
        reference.shape[0], ((num_tokens + 127) // 128) * 128
    )
    initialized_scale_tiles = initialized_rows // 128

    expected_mxfp4_bytes = _reference_mxfp4_bytes(sample_reference)
    expected_mxfp4 = mxfp4_quant.mxfp4_quantize_for_gemm(reference, 1)
    actual_mxfp4 = functional.dispatch_fp4(
        workspace, schedule, x, "mxfp4", num_comm_sms=args.comm_sms
    )
    for repeat in range(1, args.correctness_repeats):
        repeated_mxfp4 = functional.dispatch_fp4(
            workspace, schedule, x, "mxfp4", num_comm_sms=args.comm_sms
        )
        _assert_exact(
            f"MXFP4 repeat {repeat} scales",
            repeated_mxfp4[1][:initialized_scale_tiles],
            actual_mxfp4[1][:initialized_scale_tiles],
        )
        _assert_exact(
            f"MXFP4 repeat {repeat} data",
            _as_bytes(repeated_mxfp4[0])[:num_tokens][valid_rows],
            _as_bytes(actual_mxfp4[0])[:num_tokens][valid_rows],
            row_metadata=data_row_metadata,
        )
    _assert_exact(
        "MXFP4 scales",
        actual_mxfp4[1][:initialized_scale_tiles],
        expected_mxfp4[1][:initialized_scale_tiles],
    )
    _assert_exact(
        "MXFP4 data",
        _as_bytes(actual_mxfp4[0])[sample_schedule_rows],
        expected_mxfp4_bytes,
        row_metadata=sample_row_metadata,
    )
    _assert_exact("schedule peer ranks", schedule.peer_rank, original_peer_rank)
    _assert_exact(
        "schedule peer token indices",
        schedule.peer_token_idx,
        original_peer_token_idx,
    )

    expected_nvfp4 = nvfp4_quant.tk_quantize_for_gemm(reference, False, True)
    nvfp4_global_scale = (
        reference.float().abs().amax().reshape(1) / 2688.0
    )
    _assert_exact("NVFP4 global scale", nvfp4_global_scale, expected_nvfp4[4])
    expected_nvfp4_bytes = _reference_nvfp4_bytes(
        sample_reference, nvfp4_global_scale
    )
    actual_nvfp4 = functional.dispatch_fp4(
        workspace,
        schedule,
        x,
        "nvfp4",
        global_scale=nvfp4_global_scale,
        num_comm_sms=args.comm_sms,
    )
    for repeat in range(1, args.correctness_repeats):
        repeated_nvfp4 = functional.dispatch_fp4(
            workspace,
            schedule,
            x,
            "nvfp4",
            global_scale=nvfp4_global_scale,
            num_comm_sms=args.comm_sms,
        )
        _assert_exact(
            f"NVFP4 repeat {repeat} scales",
            repeated_nvfp4[1][:initialized_scale_tiles],
            actual_nvfp4[1][:initialized_scale_tiles],
        )
        _assert_exact(
            f"NVFP4 repeat {repeat} data",
            _as_bytes(repeated_nvfp4[0])[:num_tokens][valid_rows],
            _as_bytes(actual_nvfp4[0])[:num_tokens][valid_rows],
            row_metadata=data_row_metadata,
        )
    _assert_exact(
        "NVFP4 data",
        _as_bytes(actual_nvfp4[0])[sample_schedule_rows],
        expected_nvfp4_bytes,
        row_metadata=sample_row_metadata,
    )
    _assert_exact(
        "NVFP4 scales",
        actual_nvfp4[1][:initialized_scale_tiles],
        expected_nvfp4[1][:initialized_scale_tiles],
    )

    gemm_variants: list[tuple[str, Callable[[], object]]] = []
    gemm_relative_rms: tuple[float, float] | None = None
    routed_mlp_relative_rms: tuple[float, float] | None = None
    mxfp4_shared_mlp_relative_rms: float | None = None
    mxfp4_shared_full_moe_relative_rms: float | None = None
    nvfp4_shared_mlp_relative_rms: float | None = None
    nvfp4_shared_full_moe_relative_rms: float | None = None
    nvfp4_constant_routed_mlp_relative_rms: float | None = None
    bf16_routed_mlp_relative_rms: float | None = None
    pipeline_relative_rms: dict[int, tuple[float, float]] = {}
    full_moe_relative_rms: tuple[float, float] | None = None
    combine_sample_count = 0
    if args.expert_hidden:
        mxfp4_gemm, nvfp4_gemm = _load_fp4_gemms()
        local_experts = args.experts // world_size
        gate_up_size = 2 * args.expert_hidden
        expert_rows = [int(value) for value in schedule.tokens_per_expert.tolist()]
        expert_starts: list[int] = []
        offset = 0
        for rows in expert_rows:
            expert_starts.append(offset)
            offset += rows
        active_experts = [
            expert for expert, rows in enumerate(expert_rows) if rows > 0
        ]
        if not active_experts:
            raise AssertionError("expected at least one active local expert")

        weight_generator = torch.Generator(device=device).manual_seed(
            args.seed + 10_000 + rank
        )
        gate_up_weights = torch.randn(
            local_experts,
            gate_up_size,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        ) * 0.02
        gate_up_weights_flat = gate_up_weights.view(
            local_experts * gate_up_size, args.hidden
        )
        gate_up_weights_t = gate_up_weights.transpose(1, 2).contiguous()
        gate_up_mxfp4_weights = mxfp4_quant.mxfp4_quantize_for_gemm(
            gate_up_weights_flat, 1
        )
        gate_up_nvfp4_weights = nvfp4_quant.tk_quantize_for_gemm(
            gate_up_weights_flat, False, True
        )
        down_weights = torch.randn(
            local_experts,
            args.hidden,
            args.expert_hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        ) * 0.02
        down_weights_flat = down_weights.view(
            local_experts * args.hidden, args.expert_hidden
        )
        down_weights_t = down_weights.transpose(1, 2).contiguous()
        down_mxfp4_weights = mxfp4_quant.mxfp4_quantize_for_gemm(
            down_weights_flat, 1
        )
        down_nvfp4_weights = nvfp4_quant.tk_quantize_for_gemm(
            down_weights_flat, False, True
        )
        shared_weight_generator = torch.Generator(device=device).manual_seed(
            args.seed + 20_000
        )
        shared_gate_up_weights = torch.randn(
            gate_up_size,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_weight_generator,
        ) * 0.02
        shared_down_weights = torch.randn(
            args.hidden,
            args.expert_hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_weight_generator,
        ) * 0.02
        shared_gate_up_mxfp4_weights = mxfp4_quant.mxfp4_quantize_for_gemm(
            shared_gate_up_weights, 1
        )
        shared_down_mxfp4_weights = mxfp4_quant.mxfp4_quantize_for_gemm(
            shared_down_weights, 1
        )
        shared_x_mxfp4 = mxfp4_quant.mxfp4_quantize_for_gemm(x, 1)
        shared_gate_up_mxfp4 = torch.empty(
            args.tokens,
            gate_up_size,
            dtype=torch.bfloat16,
            device=device,
        )
        shared_hidden_mxfp4 = torch.empty(
            args.tokens,
            args.expert_hidden // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=device,
        )
        shared_hidden_mxfp4_scales = torch.empty(
            args.tokens // 128,
            args.expert_hidden // 128,
            32,
            16,
            dtype=torch.uint8,
            device=device,
        )
        shared_output_mxfp4 = torch.empty(
            args.tokens,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        shared_mxfp4_gate_plan = mxfp4_gemm.MXFP4GemmPlan(
            shared_x_mxfp4[0],
            shared_x_mxfp4[1],
            shared_gate_up_mxfp4_weights[0],
            shared_gate_up_mxfp4_weights[1],
            shared_gate_up_mxfp4,
        )
        shared_mxfp4_down_plan = mxfp4_gemm.MXFP4GemmPlan(
            shared_hidden_mxfp4,
            shared_hidden_mxfp4_scales,
            shared_down_mxfp4_weights[0],
            shared_down_mxfp4_weights[1],
            shared_output_mxfp4,
        )
        shared_gate_up_nvfp4_weights = nvfp4_quant.tk_quantize_for_gemm(
            shared_gate_up_weights, False, True
        )
        shared_down_nvfp4_weights = nvfp4_quant.tk_quantize_for_gemm(
            shared_down_weights, False, True
        )
        shared_x_nvfp4 = (
            torch.empty(
                args.tokens,
                args.hidden // 2,
                dtype=torch.float4_e2m1fn_x2,
                device=device,
            ),
            torch.empty(
                args.tokens // 128,
                args.hidden // 64,
                512,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(1, dtype=torch.float32, device=device),
        )
        shared_x_nvfp4_sync = torch.empty(1, dtype=torch.int32, device=device)
        shared_gate_up_nvfp4 = torch.empty_like(shared_gate_up_mxfp4)
        shared_hidden_nvfp4 = (
            torch.empty(
                args.tokens,
                args.expert_hidden // 2,
                dtype=torch.float4_e2m1fn_x2,
                device=device,
            ),
            torch.empty(
                args.tokens // 128,
                args.expert_hidden // 64,
                512,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(1, dtype=torch.float32, device=device),
        )
        shared_hidden_nvfp4_sync = torch.empty(
            1, dtype=torch.int32, device=device
        )
        shared_output_nvfp4 = torch.empty_like(shared_output_mxfp4)
        shared_nvfp4_gate_plan = nvfp4_gemm.NVFP4BatchedGemmPlan(
            [shared_x_nvfp4[0]],
            [shared_x_nvfp4[1]],
            [shared_x_nvfp4[2]],
            [shared_gate_up_nvfp4_weights[0]],
            [shared_gate_up_nvfp4_weights[1]],
            [shared_gate_up_nvfp4_weights[4]],
            [shared_gate_up_nvfp4],
        )
        shared_nvfp4_down_plan = nvfp4_gemm.NVFP4BatchedGemmPlan(
            [shared_hidden_nvfp4[0]],
            [shared_hidden_nvfp4[1]],
            [shared_hidden_nvfp4[2]],
            [shared_down_nvfp4_weights[0]],
            [shared_down_nvfp4_weights[1]],
            [shared_down_nvfp4_weights[4]],
            [shared_output_nvfp4],
        )
        shared_gate_weights = shared_gate_up_weights[
            :args.expert_hidden
        ].contiguous()
        shared_up_weights = shared_gate_up_weights[
            args.expert_hidden:
        ].contiguous()
        routed_gate_weights = gate_up_weights[
            :, :args.expert_hidden
        ].contiguous()
        routed_up_weights = gate_up_weights[
            :, args.expert_hidden:
        ].contiguous()
        routed_gate_mxfp8 = ops.mxfp8_quantize(
            routed_gate_weights, True, True
        )
        routed_up_mxfp8 = ops.mxfp8_quantize(
            routed_up_weights, True, True
        )
        routed_down_mxfp8 = ops.mxfp8_quantize(
            down_weights, True, True
        )
        gate_up_mxfp4 = torch.empty(
            num_tokens, gate_up_size, dtype=torch.bfloat16, device=device
        )
        gate_up_nvfp4 = torch.empty_like(gate_up_mxfp4)
        routed_mxfp4 = torch.empty(
            num_tokens, args.hidden, dtype=torch.bfloat16, device=device
        )
        routed_nvfp4 = torch.empty_like(routed_mxfp4)
        hidden_mxfp4 = torch.empty(
            num_tokens,
            args.expert_hidden // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=device,
        )
        hidden_mxfp4_scales = torch.empty(
            num_tokens // 128,
            args.expert_hidden // 128,
            32,
            16,
            dtype=torch.uint8,
            device=device,
        )

        active_starts = [expert_starts[expert] for expert in active_experts]
        active_rows = [expert_rows[expert] for expert in active_experts]
        zeros = [0] * len(active_experts)
        gate_up_sizes = [gate_up_size] * len(active_experts)
        hidden_sizes = [args.hidden] * len(active_experts)
        weight_starts = [expert * gate_up_size for expert in active_experts]
        expert_offsets = torch.cumsum(
            schedule.tokens_per_expert, dim=0, dtype=torch.int32
        )

        pipeline_specs: dict[int, list[tuple[int, int, int, int]]] = {}
        for requested_groups in pipeline_group_sweep:
            target_groups = min(requested_groups, len(active_experts))
            experts_per_group = (
                len(active_experts) + target_groups - 1
            ) // target_groups
            groups = []
            for expert_begin in range(0, len(active_experts), experts_per_group):
                expert_end = min(
                    expert_begin + experts_per_group, len(active_experts)
                )
                row_start = active_starts[expert_begin]
                row_end = (
                    active_starts[expert_end - 1] + active_rows[expert_end - 1]
                )
                groups.append(
                    (expert_begin, expert_end, row_start, row_end - row_start)
                )
            pipeline_specs[len(groups)] = groups

        def run_mxfp4_gate_up(
            quantized: tuple[torch.Tensor, torch.Tensor],
            expert_begin: int = 0,
            expert_end: int | None = None,
            config_id: int = -1,
        ) -> torch.Tensor:
            if expert_end is None:
                expert_end = len(active_experts)
            for batch_start in range(
                expert_begin, expert_end, _MXFP4_GEMM_MAX_BATCHES
            ):
                batch_end = min(
                    batch_start + _MXFP4_GEMM_MAX_BATCHES,
                    expert_end,
                )
                batch = slice(batch_start, batch_end)
                mxfp4_gemm.mxfp4_batched_gemm_slices(
                    quantized[0],
                    quantized[1],
                    gate_up_mxfp4_weights[0],
                    gate_up_mxfp4_weights[1],
                    gate_up_mxfp4,
                    active_starts[batch],
                    zeros[batch],
                    weight_starts[batch],
                    zeros[batch],
                    active_starts[batch],
                    zeros[batch],
                    active_rows[batch],
                    gate_up_sizes[batch],
                    hidden_sizes[batch],
                    config_id,
                )
            return gate_up_mxfp4

        nv_weight_fp4 = [
            gate_up_nvfp4_weights[0].narrow(
                0, expert * gate_up_size, gate_up_size
            )
            for expert in active_experts
        ]
        nv_weight_scales = [
            gate_up_nvfp4_weights[1].narrow(
                0, expert * gate_up_size // 128, gate_up_size // 128
            )
            for expert in active_experts
        ]
        nv_weight_global_scales = [
            gate_up_nvfp4_weights[4] for _ in active_experts
        ]
        nv_outputs = [
            gate_up_nvfp4.narrow(0, start, rows)
            for start, rows in zip(active_starts, active_rows, strict=True)
        ]

        def nvfp4_activation_views(
            quantized: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            return (
                [
                    quantized[0].narrow(0, start, rows)
                    for start, rows in zip(
                        active_starts, active_rows, strict=True
                    )
                ],
                [
                    quantized[1].narrow(0, start // 128, rows // 128)
                    for start, rows in zip(
                        active_starts, active_rows, strict=True
                    )
                ],
            )

        def run_nvfp4_gate_up(
            quantized: tuple[torch.Tensor, torch.Tensor],
            activation_views: tuple[
                list[torch.Tensor], list[torch.Tensor]
            ] | None = None,
            expert_begin: int = 0,
            expert_end: int | None = None,
            config_id: int = -1,
            flatten_batches: bool = False,
            cached_plan: object | None = None,
        ) -> torch.Tensor:
            if cached_plan is not None:
                if expert_begin != 0 or (
                    expert_end is not None and expert_end != len(active_experts)
                ):
                    raise ValueError("cached NVFP4 plan requires the full expert range")
                cached_plan.run()
                return gate_up_nvfp4
            if activation_views is None:
                activation_views = nvfp4_activation_views(quantized)
            if expert_end is None:
                expert_end = len(active_experts)
            activation_fp4, activation_scales = activation_views
            activation_global_scales = [
                nvfp4_global_scale for _ in active_experts
            ]
            for batch_start in range(
                expert_begin, expert_end, _NVFP4_GEMM_MAX_BATCHES
            ):
                batch_end = min(
                    batch_start + _NVFP4_GEMM_MAX_BATCHES,
                    expert_end,
                )
                batch = slice(batch_start, batch_end)
                operands = (
                    activation_fp4[batch],
                    activation_scales[batch],
                    activation_global_scales[batch],
                    nv_weight_fp4[batch],
                    nv_weight_scales[batch],
                    nv_weight_global_scales[batch],
                    nv_outputs[batch],
                )
                if config_id < 0:
                    nvfp4_gemm.nvfp4_batched_gemm(*operands)
                else:
                    nvfp4_gemm.nvfp4_batched_gemm_pitched(
                        *operands,
                        flatten_batches,
                        config_id,
                    )
            return gate_up_nvfp4

        down_weight_starts = [
            expert * args.hidden for expert in active_experts
        ]
        down_output_sizes = [args.hidden] * len(active_experts)
        expert_hidden_sizes = [args.expert_hidden] * len(active_experts)

        def run_mxfp4_down(
            quantized: tuple[torch.Tensor, ...],
            expert_begin: int = 0,
            expert_end: int | None = None,
            row_base: int = 0,
            config_id: int = -1,
        ) -> torch.Tensor:
            if expert_end is None:
                expert_end = len(active_experts)
            for batch_start in range(
                expert_begin, expert_end, _MXFP4_GEMM_MAX_BATCHES
            ):
                batch_end = min(
                    batch_start + _MXFP4_GEMM_MAX_BATCHES,
                    expert_end,
                )
                batch = slice(batch_start, batch_end)
                activation_starts = [
                    start - row_base for start in active_starts[batch]
                ]
                mxfp4_gemm.mxfp4_batched_gemm_slices(
                    quantized[0],
                    quantized[1],
                    down_mxfp4_weights[0],
                    down_mxfp4_weights[1],
                    routed_mxfp4,
                    activation_starts,
                    zeros[batch],
                    down_weight_starts[batch],
                    zeros[batch],
                    active_starts[batch],
                    zeros[batch],
                    active_rows[batch],
                    down_output_sizes[batch],
                    expert_hidden_sizes[batch],
                    config_id,
                )
            return routed_mxfp4

        nv_down_weight_fp4 = [
            down_nvfp4_weights[0].narrow(
                0, expert * args.hidden, args.hidden
            )
            for expert in active_experts
        ]
        nv_down_weight_scales = [
            down_nvfp4_weights[1].narrow(
                0, expert * args.hidden // 128, args.hidden // 128
            )
            for expert in active_experts
        ]
        nv_down_weight_global_scales = [
            down_nvfp4_weights[4] for _ in active_experts
        ]
        nv_down_outputs = [
            routed_nvfp4.narrow(0, start, rows)
            for start, rows in zip(active_starts, active_rows, strict=True)
        ]

        def run_nvfp4_down(
            quantized: tuple[torch.Tensor, ...],
            expert_begin: int = 0,
            expert_end: int | None = None,
            row_base: int = 0,
            config_id: int = -1,
            flatten_batches: bool = False,
        ) -> torch.Tensor:
            if expert_end is None:
                expert_end = len(active_experts)
            activation_fp4 = [
                quantized[0].narrow(0, active_starts[index] - row_base, active_rows[index])
                for index in range(expert_begin, expert_end)
            ]
            activation_scales = [
                quantized[1].narrow(
                    0,
                    (active_starts[index] - row_base) // 128,
                    active_rows[index] // 128,
                )
                for index in range(expert_begin, expert_end)
            ]
            activation_global_scales = [
                quantized[4] for _ in range(expert_begin, expert_end)
            ]
            for batch_start in range(
                expert_begin, expert_end, _NVFP4_GEMM_MAX_BATCHES
            ):
                batch_end = min(
                    batch_start + _NVFP4_GEMM_MAX_BATCHES,
                    expert_end,
                )
                batch = slice(batch_start - expert_begin, batch_end - expert_begin)
                output_batch = slice(batch_start, batch_end)
                operands = (
                    activation_fp4[batch],
                    activation_scales[batch],
                    activation_global_scales[batch],
                    nv_down_weight_fp4[output_batch],
                    nv_down_weight_scales[output_batch],
                    nv_down_weight_global_scales[output_batch],
                    nv_down_outputs[output_batch],
                )
                if config_id < 0:
                    nvfp4_gemm.nvfp4_batched_gemm(*operands)
                else:
                    nvfp4_gemm.nvfp4_batched_gemm_pitched(
                        *operands,
                        flatten_batches,
                        config_id,
                    )
            return routed_nvfp4

        def run_mxfp4_routed_mlp(
            quantized: tuple[torch.Tensor, torch.Tensor],
            expert_begin: int = 0,
            expert_end: int | None = None,
            row_start: int = 0,
            row_count: int | None = None,
        ) -> torch.Tensor:
            if expert_end is None:
                expert_end = len(active_experts)
            if row_count is None:
                row_count = num_tokens - row_start
            run_mxfp4_gate_up(quantized, expert_begin, expert_end)
            hidden_data = hidden_mxfp4.narrow(0, row_start, row_count)
            hidden_scales = hidden_mxfp4_scales.narrow(
                0, row_start // 128, row_count // 128
            )
            mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided_launch_inplace(
                gate_up_mxfp4.narrow(0, row_start, row_count),
                args.expert_hidden,
                args.expert_hidden,
                hidden_data,
                hidden_scales,
                1,
            )
            return run_mxfp4_down(
                (hidden_data, hidden_scales),
                expert_begin,
                expert_end,
                row_start,
            )

        def run_nvfp4_routed_mlp(
            quantized: tuple[torch.Tensor, torch.Tensor],
            expert_begin: int = 0,
            expert_end: int | None = None,
            row_start: int = 0,
            row_count: int | None = None,
            use_constant_scale: bool = False,
            cached_gate_plan: object | None = None,
            cached_down_plan: object | None = None,
        ) -> torch.Tensor:
            if expert_end is None:
                expert_end = len(active_experts)
            if row_count is None:
                row_count = num_tokens - row_start
            run_nvfp4_gate_up(
                quantized,
                expert_begin=expert_begin,
                expert_end=expert_end,
                cached_plan=cached_gate_plan,
            )
            if cached_down_plan is not None:
                if expert_begin != 0 or expert_end != len(active_experts):
                    raise ValueError(
                        "cached NVFP4 down plan requires the full expert range"
                    )
                nvfp4_quant.tk_silu_quantize_for_gemm_constant_scale_into(
                    gate_up_nvfp4.narrow(0, row_start, row_count),
                    args.expert_hidden,
                    cached_hidden_nvfp4[0],
                    cached_hidden_nvfp4[1],
                    cached_hidden_nvfp4[2],
                    cached_hidden_nvfp4_sync,
                )
                cached_down_plan.run()
                return routed_nvfp4
            hidden_quantized = nvfp4_quant.tk_silu_quantize_for_gemm(
                gate_up_nvfp4.narrow(0, row_start, row_count),
                args.expert_hidden,
                False,
                use_constant_scale,
            )
            return run_nvfp4_down(
                hidden_quantized, expert_begin, expert_end, row_start
            )

        def run_bf16_routed_mlp() -> torch.Tensor:
            gate_up = torch._grouped_mm(
                reference[:num_tokens], gate_up_weights_t, offs=expert_offsets
            )
            hidden = (
                F.silu(gate_up[:, :args.expert_hidden])
                * gate_up[:, args.expert_hidden:]
            )
            return torch._grouped_mm(hidden, down_weights_t, offs=expert_offsets)

        def run_bf16_shared_mlp() -> torch.Tensor:
            gate_up = F.linear(x, shared_gate_up_weights)
            hidden = (
                F.silu(gate_up[:, :args.expert_hidden])
                * gate_up[:, args.expert_hidden:]
            )
            return F.linear(hidden, shared_down_weights)

        def run_mxfp4_shared_mlp() -> torch.Tensor:
            mxfp4_quant.mxfp4_quantize_for_gemm_launch_inplace(
                x,
                shared_x_mxfp4[0],
                shared_x_mxfp4[1],
                1,
            )
            shared_mxfp4_gate_plan.run()
            mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided_launch_inplace(
                shared_gate_up_mxfp4,
                args.expert_hidden,
                args.expert_hidden,
                shared_hidden_mxfp4,
                shared_hidden_mxfp4_scales,
                1,
            )
            shared_mxfp4_down_plan.run()
            return shared_output_mxfp4

        def run_nvfp4_shared_mlp() -> torch.Tensor:
            nvfp4_quant.tk_quantize_for_gemm_constant_scale_into(
                x,
                shared_x_nvfp4[0],
                shared_x_nvfp4[1],
                shared_x_nvfp4[2],
                shared_x_nvfp4_sync,
            )
            shared_nvfp4_gate_plan.run()
            nvfp4_quant.tk_silu_quantize_for_gemm_constant_scale_into(
                shared_gate_up_nvfp4,
                args.expert_hidden,
                shared_hidden_nvfp4[0],
                shared_hidden_nvfp4[1],
                shared_hidden_nvfp4[2],
                shared_hidden_nvfp4_sync,
            )
            shared_nvfp4_down_plan.run()
            return shared_output_nvfp4

        shared_x_mxfp4_reference = mxfp4_quant.mxfp4_quantize_for_gemm(x, 1)
        mxfp4_quant.mxfp4_quantize_for_gemm_launch_inplace(
            x,
            shared_x_mxfp4[0],
            shared_x_mxfp4[1],
            1,
        )
        _assert_exact(
            "MXFP4 in-place input quant data",
            _as_bytes(shared_x_mxfp4[0]),
            _as_bytes(shared_x_mxfp4_reference[0]),
        )
        _assert_exact(
            "MXFP4 in-place input quant scales",
            shared_x_mxfp4[1],
            shared_x_mxfp4_reference[1],
        )
        mxfp4_gemm.mxfp4_gemm(
            shared_x_mxfp4[0],
            shared_x_mxfp4[1],
            shared_gate_up_mxfp4_weights[0],
            shared_gate_up_mxfp4_weights[1],
            shared_gate_up_mxfp4,
        )
        shared_gate_sample = shared_gate_up_mxfp4[:128].clone()
        shared_mxfp4_gate_plan.run()
        _assert_exact(
            "MXFP4 cached shared gate plan",
            shared_gate_up_mxfp4[:128],
            shared_gate_sample,
        )

        shared_x_nvfp4_reference = (
            nvfp4_quant.tk_quantize_for_gemm_constant_scale(x, False)
        )
        nvfp4_quant.tk_quantize_for_gemm_constant_scale_into(
            x,
            shared_x_nvfp4[0],
            shared_x_nvfp4[1],
            shared_x_nvfp4[2],
            shared_x_nvfp4_sync,
        )
        _assert_exact(
            "NVFP4 in-place input quant data",
            _as_bytes(shared_x_nvfp4[0]),
            _as_bytes(shared_x_nvfp4_reference[0]),
        )
        _assert_exact(
            "NVFP4 in-place input quant scales",
            shared_x_nvfp4[1],
            shared_x_nvfp4_reference[1],
        )
        _assert_exact(
            "NVFP4 in-place input global scale",
            shared_x_nvfp4[2],
            shared_x_nvfp4_reference[4],
        )
        del (
            shared_x_mxfp4_reference,
            shared_gate_sample,
            shared_x_nvfp4_reference,
        )

        def run_bf16_mok_forward() -> torch.Tensor:
            output, _ = functional.forward(
                bf16_control_config,
                workspace,
                schedule,
                x,
                router_weights,
                shared_gate_weights,
                shared_up_weights,
                shared_down_weights,
                routed_gate_weights,
                routed_up_weights,
                down_weights,
            )
            return output

        def run_mxfp8_mok_forward() -> torch.Tensor:
            output, _ = functional.forward(
                mxfp8_control_config,
                workspace,
                schedule,
                x,
                router_weights,
                shared_gate_weights,
                shared_up_weights,
                shared_down_weights,
                routed_gate_mxfp8[:2],
                routed_up_mxfp8[:2],
                routed_down_mxfp8[:2],
            )
            return output

        pipelined_mxfp4 = (
            torch.empty_like(actual_mxfp4[0]),
            torch.empty_like(actual_mxfp4[1]),
        )
        pipelined_nvfp4 = (
            torch.empty_like(actual_nvfp4[0]),
            torch.empty_like(actual_nvfp4[1]),
        )
        pipelined_nvfp4_views = nvfp4_activation_views(pipelined_nvfp4)
        pipelined_nvfp4_gate_plan = nvfp4_gemm.NVFP4BatchedGemmPlan(
            pipelined_nvfp4_views[0],
            pipelined_nvfp4_views[1],
            [nvfp4_global_scale for _ in active_experts],
            nv_weight_fp4,
            nv_weight_scales,
            nv_weight_global_scales,
            nv_outputs,
        )
        cached_hidden_nvfp4 = (
            torch.empty(
                num_tokens,
                args.expert_hidden // 2,
                dtype=torch.float4_e2m1fn_x2,
                device=device,
            ),
            torch.empty(
                num_tokens // 128,
                args.expert_hidden // 64,
                512,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(1, dtype=torch.float32, device=device),
        )
        cached_hidden_nvfp4_sync = torch.empty(
            1, dtype=torch.int32, device=device
        )
        cached_hidden_activation_fp4 = [
            cached_hidden_nvfp4[0].narrow(0, start, rows)
            for start, rows in zip(active_starts, active_rows, strict=True)
        ]
        cached_hidden_activation_scales = [
            cached_hidden_nvfp4[1].narrow(0, start // 128, rows // 128)
            for start, rows in zip(active_starts, active_rows, strict=True)
        ]
        cached_nvfp4_down_plan = nvfp4_gemm.NVFP4BatchedGemmPlan(
            cached_hidden_activation_fp4,
            cached_hidden_activation_scales,
            [cached_hidden_nvfp4[2] for _ in active_experts],
            nv_down_weight_fp4,
            nv_down_weight_scales,
            nv_down_weight_global_scales,
            nv_down_outputs,
        )
        comm_stream = torch.cuda.Stream(device=device)
        combine_stream = torch.cuda.Stream(device=device)
        shared_stream = torch.cuda.Stream(device=device)
        shared_done = torch.cuda.Event()
        pull_ready_events = {
            group_count: [
                torch.cuda.Event() for _ in groups
            ]
            for group_count, groups in pipeline_specs.items()
        }
        compute_ready_events = {
            group_count: [
                torch.cuda.Event() for _ in groups
            ]
            for group_count, groups in pipeline_specs.items()
        }
        combine_done_events = {
            group_count: [
                torch.cuda.Event() for _ in groups
            ]
            for group_count, groups in pipeline_specs.items()
        }

        overlap_comm_sms = args.overlap_comm_sms or args.comm_sms
        mok_c = importlib.import_module("mok._C")

        def dispatch_mxfp4_range(
            row_start: int,
            row_count: int,
            num_comm_sms: int = args.comm_sms,
            pull_cols: int | None = None,
        ) -> None:
            if pull_cols is None:
                pull_cols = args.mxfp4_pull_cols
            dispatch_args = (
                workspace.x_buffer,
                workspace.x_buffer_ptrs,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                pipelined_mxfp4[0],
                pipelined_mxfp4[1],
                row_start,
                row_count,
                workspace.topk,
                num_comm_sms,
            )
            if pull_cols:
                mok_c.dispatch_mxfp4_into(*dispatch_args, pull_cols)
            else:
                ops.dispatch_mxfp4_into(*dispatch_args)

        def dispatch_nvfp4_range(
            row_start: int,
            row_count: int,
            num_comm_sms: int = args.comm_sms,
        ) -> None:
            ops.dispatch_nvfp4_into(
                workspace.x_buffer,
                workspace.x_buffer_ptrs,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                nvfp4_global_scale,
                pipelined_nvfp4[0],
                pipelined_nvfp4[1],
                row_start,
                row_count,
                workspace.topk,
                num_comm_sms,
            )

        def combine_range(
            routed_output: torch.Tensor,
            row_start: int,
            row_count: int,
            num_comm_sms: int = args.combine_sms,
            combine_cols: int = args.combine_cols,
        ) -> None:
            ops.combine_bf16_into(
                routed_output,
                workspace.combine_buffer,
                workspace.combine_buffer_ptrs,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                row_start,
                row_count,
                num_comm_sms,
                combine_cols,
            )

        def finish_combine() -> torch.Tensor:
            ops.barrier_all(
                workspace.barrier_buffer,
                workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr,
                workspace.barrier_target,
            )
            return workspace.combine_buffer

        def run_mxfp4_serial_into(
            pull_cols: int | None = None,
        ) -> torch.Tensor:
            dispatch_mxfp4_range(0, num_tokens, pull_cols=pull_cols)
            return run_mxfp4_routed_mlp(pipelined_mxfp4)

        def run_nvfp4_serial_into(
            use_constant_scale: bool = False,
            use_cached_gate: bool = False,
            use_cached_down: bool = False,
        ) -> torch.Tensor:
            dispatch_nvfp4_range(0, num_tokens)
            return run_nvfp4_routed_mlp(
                pipelined_nvfp4,
                use_constant_scale=use_constant_scale,
                cached_gate_plan=(
                    pipelined_nvfp4_gate_plan if use_cached_gate else None
                ),
                cached_down_plan=(
                    cached_nvfp4_down_plan if use_cached_down else None
                ),
            )

        def run_combine_only(
            routed_output: torch.Tensor,
            combine_sms: int = args.combine_sms,
            combine_cols: int = args.combine_cols,
        ) -> torch.Tensor:
            combine_range(
                routed_output, 0, num_tokens, combine_sms, combine_cols
            )
            return finish_combine()

        def run_mxfp4_serial_with_combine(
            combine_sms: int = args.combine_sms,
            pull_cols: int | None = None,
            combine_cols: int = args.combine_cols,
        ) -> torch.Tensor:
            run_mxfp4_serial_into(pull_cols)
            return run_combine_only(routed_mxfp4, combine_sms, combine_cols)

        def run_nvfp4_serial_with_combine(
            combine_sms: int = args.combine_sms,
            combine_cols: int = args.combine_cols,
            use_constant_scale: bool = False,
            use_cached_gate: bool = False,
            use_cached_down: bool = False,
        ) -> torch.Tensor:
            run_nvfp4_serial_into(
                use_constant_scale,
                use_cached_gate,
                use_cached_down,
            )
            return run_combine_only(routed_nvfp4, combine_sms, combine_cols)

        def run_mxfp4_streamed(
            group_count: int,
            push_output: bool = False,
            combine_sms: int = args.combine_sms,
        ) -> torch.Tensor:
            current_stream = torch.cuda.current_stream(device)
            comm_stream.wait_stream(current_stream)
            if push_output:
                combine_stream.wait_stream(current_stream)
            events = zip(
                pull_ready_events[group_count],
                compute_ready_events[group_count],
                combine_done_events[group_count],
                pipeline_specs[group_count],
                strict=True,
            )
            for group_index, event_group in enumerate(events):
                pull_ready, compute_ready, combine_done, group = event_group
                expert_begin, expert_end, row_start, row_count = group
                with torch.cuda.stream(comm_stream):
                    dispatch_mxfp4_range(
                        row_start,
                        row_count,
                        args.comm_sms if group_index == 0 else overlap_comm_sms,
                    )
                    pull_ready.record()
                current_stream.wait_event(pull_ready)
                run_mxfp4_routed_mlp(
                    pipelined_mxfp4,
                    expert_begin,
                    expert_end,
                    row_start,
                    row_count,
                )
                if push_output:
                    compute_ready.record(current_stream)
                    with torch.cuda.stream(combine_stream):
                        combine_stream.wait_event(compute_ready)
                        combine_range(
                            routed_mxfp4, row_start, row_count, combine_sms
                        )
                        combine_done.record()
            if not push_output:
                return routed_mxfp4
            for combine_done in combine_done_events[group_count]:
                current_stream.wait_event(combine_done)
            return finish_combine()

        def run_nvfp4_streamed(
            group_count: int,
            push_output: bool = False,
            combine_sms: int = args.combine_sms,
        ) -> torch.Tensor:
            current_stream = torch.cuda.current_stream(device)
            comm_stream.wait_stream(current_stream)
            if push_output:
                combine_stream.wait_stream(current_stream)
            events = zip(
                pull_ready_events[group_count],
                compute_ready_events[group_count],
                combine_done_events[group_count],
                pipeline_specs[group_count],
                strict=True,
            )
            for group_index, event_group in enumerate(events):
                pull_ready, compute_ready, combine_done, group = event_group
                expert_begin, expert_end, row_start, row_count = group
                with torch.cuda.stream(comm_stream):
                    dispatch_nvfp4_range(
                        row_start,
                        row_count,
                        args.comm_sms if group_index == 0 else overlap_comm_sms,
                    )
                    pull_ready.record()
                current_stream.wait_event(pull_ready)
                run_nvfp4_routed_mlp(
                    pipelined_nvfp4,
                    expert_begin,
                    expert_end,
                    row_start,
                    row_count,
                )
                if push_output:
                    compute_ready.record(current_stream)
                    with torch.cuda.stream(combine_stream):
                        combine_stream.wait_event(compute_ready)
                        combine_range(
                            routed_nvfp4, row_start, row_count, combine_sms
                        )
                        combine_done.record()
            if not push_output:
                return routed_nvfp4
            for combine_done in combine_done_events[group_count]:
                current_stream.wait_event(combine_done)
            return finish_combine()

        def run_full_moe(
            run_routed: Callable[[int], torch.Tensor],
            combine_sms: int = args.combine_sms,
            launch_shared_early: bool = True,
            epilogue_tokens_per_cta: int = args.epilogue_tokens_per_cta,
            epilogue_cols_per_cta: int = args.epilogue_cols_per_cta,
            run_shared: Callable[[], torch.Tensor] = run_bf16_shared_mlp,
        ) -> torch.Tensor:
            current_stream = torch.cuda.current_stream(device)
            if launch_shared_early:
                shared_stream.wait_stream(current_stream)
                with torch.cuda.stream(shared_stream):
                    shared_output = run_shared()
                    shared_done.record()
            workspace.x_buffer.copy_(x)
            ops.barrier_all(
                workspace.barrier_buffer,
                workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr,
                workspace.barrier_target,
            )
            if not launch_shared_early:
                shared_stream.wait_stream(current_stream)
                with torch.cuda.stream(shared_stream):
                    shared_output = run_shared()
                    shared_done.record()
            run_routed(combine_sms)
            current_stream.wait_event(shared_done)
            return ops.fwd_epilogue(
                shared_output,
                workspace.combine_buffer,
                router_weights,
                epilogue_tokens_per_cta,
                epilogue_cols_per_cta,
            )

        def run_mxfp4_full_moe(
            combine_sms: int = args.combine_sms,
            pull_cols: int | None = None,
            epilogue_tokens_per_cta: int = args.epilogue_tokens_per_cta,
            epilogue_cols_per_cta: int = args.epilogue_cols_per_cta,
            combine_cols: int = args.combine_cols,
            use_mxfp4_shared: bool = False,
        ) -> torch.Tensor:
            return run_full_moe(
                lambda sms: run_mxfp4_serial_with_combine(
                    sms, pull_cols, combine_cols
                ),
                combine_sms,
                epilogue_tokens_per_cta=epilogue_tokens_per_cta,
                epilogue_cols_per_cta=epilogue_cols_per_cta,
                run_shared=(
                    run_mxfp4_shared_mlp
                    if use_mxfp4_shared
                    else run_bf16_shared_mlp
                ),
            )

        def run_nvfp4_full_moe(
            combine_sms: int = args.combine_sms,
            epilogue_tokens_per_cta: int = args.epilogue_tokens_per_cta,
            epilogue_cols_per_cta: int = args.epilogue_cols_per_cta,
            combine_cols: int = args.combine_cols,
            launch_shared_early: bool = False,
            use_constant_scale: bool = False,
            use_cached_gate: bool = False,
            use_cached_down: bool = False,
            use_nvfp4_shared: bool = False,
        ) -> torch.Tensor:
            return run_full_moe(
                lambda sms: run_nvfp4_serial_with_combine(
                    sms,
                    combine_cols,
                    use_constant_scale,
                    use_cached_gate,
                    use_cached_down,
                ),
                combine_sms,
                launch_shared_early=launch_shared_early,
                epilogue_tokens_per_cta=epilogue_tokens_per_cta,
                epilogue_cols_per_cta=epilogue_cols_per_cta,
                run_shared=(
                    run_nvfp4_shared_mlp
                    if use_nvfp4_shared
                    else run_bf16_shared_mlp
                ),
            )

        validation_groups = pipeline_specs[max(pipeline_specs)]
        for _, _, row_start, row_count in validation_groups:
            dispatch_mxfp4_range(row_start, row_count)
        _assert_exact(
            "MXFP4 row-range dispatch data",
            _as_bytes(pipelined_mxfp4[0])[:num_tokens],
            _as_bytes(actual_mxfp4[0])[:num_tokens],
        )
        _assert_exact(
            "MXFP4 row-range dispatch scales",
            pipelined_mxfp4[1][:num_tokens // 128],
            actual_mxfp4[1][:num_tokens // 128],
        )
        for pull_cols in mxfp4_pull_cols_sweep:
            dispatch_mxfp4_range(
                0, num_tokens, pull_cols=pull_cols
            )
            _assert_exact(
                f"MXFP4 {pull_cols}-column dispatch data",
                _as_bytes(pipelined_mxfp4[0])[:num_tokens],
                _as_bytes(actual_mxfp4[0])[:num_tokens],
            )
            _assert_exact(
                f"MXFP4 {pull_cols}-column dispatch scales",
                pipelined_mxfp4[1][:num_tokens // 128],
                actual_mxfp4[1][:num_tokens // 128],
            )
        for _, _, row_start, row_count in validation_groups:
            dispatch_nvfp4_range(row_start, row_count)
        _assert_exact(
            "NVFP4 row-range dispatch data",
            _as_bytes(pipelined_nvfp4[0])[:num_tokens],
            _as_bytes(actual_nvfp4[0])[:num_tokens],
        )
        _assert_exact(
            "NVFP4 row-range dispatch scales",
            pipelined_nvfp4[1][:num_tokens // 128],
            actual_nvfp4[1][:num_tokens // 128],
        )

        actual_nvfp4_views = nvfp4_activation_views(actual_nvfp4)
        run_mxfp4_gate_up(actual_mxfp4)
        run_nvfp4_gate_up(actual_nvfp4, actual_nvfp4_views)
        gemm_sample_rows_list: list[torch.Tensor] = []
        for expert in active_experts[:8]:
            start = expert_starts[expert]
            rows = expert_rows[expert]
            local_valid = torch.nonzero(
                valid_rows[start:start + rows], as_tuple=False
            ).flatten()[:2]
            if local_valid.numel():
                gemm_sample_rows_list.append(local_valid + start)
        if not gemm_sample_rows_list:
            raise AssertionError("expected valid rows in active expert segments")
        gemm_sample_rows = torch.cat(gemm_sample_rows_list)
        gemm_expected_parts = []
        routed_mlp_expected_parts = []
        sample_offset = 0
        for expert in active_experts[:8]:
            start = expert_starts[expert]
            rows = expert_rows[expert]
            local_valid = torch.nonzero(
                valid_rows[start:start + rows], as_tuple=False
            ).flatten()[:2]
            num_samples = local_valid.numel()
            if num_samples:
                rows_for_expert = gemm_sample_rows[
                    sample_offset:sample_offset + num_samples
                ]
                gate_up_expected = (
                    reference[rows_for_expert].float()
                    @ gate_up_weights[expert].float().T
                )
                gemm_expected_parts.append(gate_up_expected)
                hidden_expected = (
                    F.silu(gate_up_expected[:, :args.expert_hidden])
                    * gate_up_expected[:, args.expert_hidden:]
                )
                routed_mlp_expected_parts.append(
                    hidden_expected @ down_weights[expert].float().T
                )
                sample_offset += num_samples
        gemm_expected = torch.cat(gemm_expected_parts)
        routed_mlp_expected = torch.cat(routed_mlp_expected_parts)
        mxfp4_sample = gate_up_mxfp4[gemm_sample_rows].clone()
        nvfp4_sample = gate_up_nvfp4[gemm_sample_rows].clone()
        gemm_relative_rms = (
            _relative_rms(mxfp4_sample, gemm_expected),
            _relative_rms(nvfp4_sample, gemm_expected),
        )
        if max(gemm_relative_rms) >= 0.5:
            raise AssertionError(
                f"FP4 gate/up relative RMS error too large: {gemm_relative_rms}"
            )
        run_mxfp4_gate_up(actual_mxfp4)
        run_nvfp4_gate_up(actual_nvfp4, actual_nvfp4_views)
        _assert_exact(
            "MXFP4 gate/up repeat", gate_up_mxfp4[gemm_sample_rows], mxfp4_sample
        )
        _assert_exact(
            "NVFP4 gate/up repeat", gate_up_nvfp4[gemm_sample_rows], nvfp4_sample
        )
        dispatch_nvfp4_range(0, num_tokens)
        run_nvfp4_gate_up(
            pipelined_nvfp4,
            cached_plan=pipelined_nvfp4_gate_plan,
        )
        _assert_exact(
            "NVFP4 cached-plan gate/up",
            gate_up_nvfp4[gemm_sample_rows],
            nvfp4_sample,
        )
        hidden_mxfp4_rowcol = (
            mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_and_col_strided(
                gate_up_mxfp4,
                args.expert_hidden,
                args.expert_hidden,
                1,
            )
        )
        hidden_mxfp4_row = (
            mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided(
                gate_up_mxfp4,
                args.expert_hidden,
                args.expert_hidden,
                1,
            )
        )
        _assert_exact(
            "MXFP4 row-only fused SwiGLU data",
            _as_bytes(hidden_mxfp4_row[0]),
            _as_bytes(hidden_mxfp4_rowcol[0]),
        )
        _assert_exact(
            "MXFP4 row-only fused SwiGLU scales",
            hidden_mxfp4_row[1],
            hidden_mxfp4_rowcol[1],
        )
        hidden_nvfp4_rowcol = nvfp4_quant.tk_silu_quantize_for_gemm(
            gate_up_nvfp4, args.expert_hidden, True
        )
        hidden_nvfp4_row = nvfp4_quant.tk_silu_quantize_for_gemm(
            gate_up_nvfp4, args.expert_hidden, False
        )
        _assert_exact(
            "NVFP4 row-only fused SwiGLU data",
            _as_bytes(hidden_nvfp4_row[0]),
            _as_bytes(hidden_nvfp4_rowcol[0]),
        )
        _assert_exact(
            "NVFP4 row-only fused SwiGLU scales",
            hidden_nvfp4_row[1],
            hidden_nvfp4_rowcol[1],
        )
        _assert_exact(
            "NVFP4 row-only fused SwiGLU global scale",
            hidden_nvfp4_row[4],
            hidden_nvfp4_rowcol[4],
        )
        run_mxfp4_routed_mlp(actual_mxfp4)
        run_nvfp4_routed_mlp(actual_nvfp4)
        mxfp4_routed_sample = routed_mxfp4[gemm_sample_rows].clone()
        nvfp4_routed_sample = routed_nvfp4[gemm_sample_rows].clone()
        routed_mlp_relative_rms = (
            _relative_rms(mxfp4_routed_sample, routed_mlp_expected),
            _relative_rms(nvfp4_routed_sample, routed_mlp_expected),
        )
        if max(routed_mlp_relative_rms) >= 0.7:
            raise AssertionError(
                "FP4 routed MLP relative RMS error too large: "
                f"{routed_mlp_relative_rms}"
            )
        run_nvfp4_routed_mlp(actual_nvfp4, use_constant_scale=True)
        nvfp4_constant_routed_sample = routed_nvfp4[gemm_sample_rows].clone()
        nvfp4_constant_routed_mlp_relative_rms = _relative_rms(
            nvfp4_constant_routed_sample,
            routed_mlp_expected,
        )
        if nvfp4_constant_routed_mlp_relative_rms >= 0.7:
            raise AssertionError(
                "NVFP4 constant-scale routed MLP relative RMS error too large: "
                f"{nvfp4_constant_routed_mlp_relative_rms}"
            )
        run_nvfp4_routed_mlp(
            actual_nvfp4,
            use_constant_scale=True,
            cached_down_plan=cached_nvfp4_down_plan,
        )
        _assert_exact(
            "NVFP4 cached-plan down",
            routed_nvfp4[gemm_sample_rows],
            nvfp4_constant_routed_sample,
        )
        bf16_routed_sample = run_bf16_routed_mlp()[gemm_sample_rows].clone()
        bf16_routed_mlp_relative_rms = _relative_rms(
            bf16_routed_sample, routed_mlp_expected
        )
        if bf16_routed_mlp_relative_rms >= 0.05:
            raise AssertionError(
                "BF16 routed MLP relative RMS error too large: "
                f"{bf16_routed_mlp_relative_rms}"
            )
        run_mxfp4_routed_mlp(actual_mxfp4)
        run_nvfp4_routed_mlp(actual_nvfp4)
        _assert_exact(
            "MXFP4 routed MLP repeat",
            routed_mxfp4[gemm_sample_rows],
            mxfp4_routed_sample,
        )
        _assert_exact(
            "NVFP4 routed MLP repeat",
            routed_nvfp4[gemm_sample_rows],
            nvfp4_routed_sample,
        )
        for group_count in pipeline_specs:
            run_mxfp4_streamed(group_count)
            streamed_mxfp4_sample = routed_mxfp4[gemm_sample_rows].clone()
            _assert_exact(
                f"MXFP4 {group_count}-group streamed routed MLP",
                streamed_mxfp4_sample,
                mxfp4_routed_sample,
            )
            run_nvfp4_streamed(group_count)
            streamed_nvfp4_sample = routed_nvfp4[gemm_sample_rows].clone()
            streamed_rms = (
                _relative_rms(streamed_mxfp4_sample, routed_mlp_expected),
                _relative_rms(streamed_nvfp4_sample, routed_mlp_expected),
            )
            if max(streamed_rms) >= 0.7:
                raise AssertionError(
                    f"FP4 {group_count}-group streamed MLP RMS too large: "
                    f"{streamed_rms}"
                )
            pipeline_relative_rms[group_count] = streamed_rms

        combine_sample_count = min(64, workspace.num_local_tokens * workspace.topk)
        combine_sample_routes = (
            torch.arange(combine_sample_count, device=device, dtype=torch.int64)
            * (workspace.num_local_tokens * workspace.topk)
            // combine_sample_count
        )
        route_to_sample = torch.full(
            (workspace.num_local_tokens * workspace.topk,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        route_to_sample[combine_sample_routes] = torch.arange(
            combine_sample_count, device=device, dtype=torch.int64
        )

        def expected_combine_samples(
            routed_output: torch.Tensor,
        ) -> torch.Tensor:
            contributions = torch.zeros(
                world_size,
                combine_sample_count,
                args.hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            counts = torch.zeros(
                world_size,
                combine_sample_count,
                dtype=torch.int32,
                device=device,
            )
            peers = schedule.peer_rank[:num_tokens].to(torch.int64)
            routes = schedule.peer_token_idx[:num_tokens].to(torch.int64)
            valid_schedule = (
                (peers >= 0)
                & (routes >= 0)
                & (routes < route_to_sample.numel())
            )
            schedule_rows = torch.nonzero(
                valid_schedule, as_tuple=False
            ).flatten()
            sample_slots = route_to_sample[routes[schedule_rows]]
            sampled = sample_slots >= 0
            schedule_rows = schedule_rows[sampled]
            sample_slots = sample_slots[sampled]
            destinations = (
                peers[schedule_rows] * combine_sample_count + sample_slots
            )
            contributions.view(-1, args.hidden).index_copy_(
                0, destinations, routed_output[schedule_rows]
            )
            counts.view(-1).index_add_(
                0,
                destinations,
                torch.ones_like(destinations, dtype=torch.int32),
            )
            dist.all_reduce(contributions)
            dist.all_reduce(counts)
            if not torch.all(counts == 1).item():
                raise AssertionError(
                    "each sampled source route must receive exactly one expert output"
                )
            return contributions[rank]

        combine_validation_groups = min(pipeline_specs)
        workspace.combine_buffer.fill_(float("nan"))
        dist.barrier(async_op=True).block_current_stream()
        run_mxfp4_streamed(combine_validation_groups, push_output=True)
        mxfp4_combine_sample = workspace.combine_buffer[
            combine_sample_routes
        ].clone()
        mxfp4_expected_combine = expected_combine_samples(routed_mxfp4)
        _assert_exact(
            "MXFP4 streamed push combine",
            mxfp4_combine_sample,
            mxfp4_expected_combine,
        )
        for combine_cols in combine_cols_sweep:
            workspace.combine_buffer.fill_(float("nan"))
            dist.barrier(async_op=True).block_current_stream()
            run_combine_only(
                routed_mxfp4,
                args.combine_sms,
                combine_cols,
            )
            _assert_exact(
                f"MXFP4 {combine_cols}-column route push",
                workspace.combine_buffer[combine_sample_routes],
                mxfp4_expected_combine,
            )

        workspace.combine_buffer.fill_(float("nan"))
        dist.barrier(async_op=True).block_current_stream()
        run_nvfp4_streamed(combine_validation_groups, push_output=True)
        _assert_exact(
            "NVFP4 streamed push combine",
            workspace.combine_buffer[combine_sample_routes],
            expected_combine_samples(routed_nvfp4),
        )

        def full_moe_reference() -> torch.Tensor:
            shared_output = run_bf16_shared_mlp()
            routed_output = workspace.combine_buffer.view(
                args.tokens, args.topk, args.hidden
            )
            return (
                shared_output.float()
                + (
                    routed_output.float()
                    * router_weights.unsqueeze(2)
                ).sum(dim=1)
            ).to(torch.bfloat16)

        mxfp4_shared_output = run_mxfp4_shared_mlp()
        mxfp4_shared_mlp_relative_rms = _relative_rms(
            mxfp4_shared_output, run_bf16_shared_mlp()
        )
        if mxfp4_shared_mlp_relative_rms >= 0.7:
            raise AssertionError(
                "MXFP4 shared MLP relative RMS error too large: "
                f"{mxfp4_shared_mlp_relative_rms}"
            )
        nvfp4_shared_output = run_nvfp4_shared_mlp()
        nvfp4_shared_mlp_relative_rms = _relative_rms(
            nvfp4_shared_output, run_bf16_shared_mlp()
        )
        if nvfp4_shared_mlp_relative_rms >= 0.7:
            raise AssertionError(
                "NVFP4 shared MLP relative RMS error too large: "
                f"{nvfp4_shared_mlp_relative_rms}"
            )

        mxfp4_full_output = run_mxfp4_full_moe()
        mxfp4_full_rms = _relative_rms(
            mxfp4_full_output, full_moe_reference()
        )
        nvfp4_full_output = run_nvfp4_full_moe()
        nvfp4_full_rms = _relative_rms(
            nvfp4_full_output, full_moe_reference()
        )
        full_moe_relative_rms = (mxfp4_full_rms, nvfp4_full_rms)
        if max(full_moe_relative_rms) >= 0.01:
            raise AssertionError(
                "FP4 full MoE assembly relative RMS error too large: "
                f"{full_moe_relative_rms}"
            )
        mxfp4_shared_full_output = run_mxfp4_full_moe(
            use_mxfp4_shared=True
        )
        mxfp4_shared_full_moe_relative_rms = _relative_rms(
            mxfp4_shared_full_output, full_moe_reference()
        )
        if mxfp4_shared_full_moe_relative_rms >= 0.7:
            raise AssertionError(
                "MXFP4 shared full-MoE relative RMS error too large: "
                f"{mxfp4_shared_full_moe_relative_rms}"
            )
        nvfp4_shared_full_output = run_nvfp4_full_moe(
            launch_shared_early=True,
            use_constant_scale=True,
            use_cached_gate=True,
            use_cached_down=True,
            use_nvfp4_shared=True,
        )
        nvfp4_shared_full_moe_relative_rms = _relative_rms(
            nvfp4_shared_full_output, full_moe_reference()
        )
        if nvfp4_shared_full_moe_relative_rms >= 0.7:
            raise AssertionError(
                "NVFP4 shared full-MoE relative RMS error too large: "
                f"{nvfp4_shared_full_moe_relative_rms}"
            )

        epilogue_shared_output = run_bf16_shared_mlp()
        epilogue_reference = ops.fwd_epilogue(
            epilogue_shared_output,
            workspace.combine_buffer,
            router_weights,
            args.epilogue_tokens_per_cta,
            args.epilogue_cols_per_cta,
        )
        for tokens_per_cta in epilogue_tokens_per_cta_sweep:
            _assert_exact(
                f"Epilogue {tokens_per_cta} tokens/CTA",
                ops.fwd_epilogue(
                    epilogue_shared_output,
                    workspace.combine_buffer,
                    router_weights,
                    tokens_per_cta,
                    args.epilogue_cols_per_cta,
                ),
                epilogue_reference,
            )
        for cols_per_cta in epilogue_cols_per_cta_sweep:
            _assert_exact(
                f"Epilogue {cols_per_cta} columns/CTA",
                ops.fwd_epilogue(
                    epilogue_shared_output,
                    workspace.combine_buffer,
                    router_weights,
                    args.epilogue_tokens_per_cta,
                    cols_per_cta,
                ),
                epilogue_reference,
            )

        def run_mxfp4_pipeline() -> torch.Tensor:
            quantized = functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "mxfp4",
                num_comm_sms=args.comm_sms,
            )
            return run_mxfp4_gate_up(quantized)

        def run_nvfp4_pipeline() -> torch.Tensor:
            quantized = functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "nvfp4",
                global_scale=nvfp4_global_scale,
                num_comm_sms=args.comm_sms,
            )
            return run_nvfp4_gate_up(quantized)

        def run_mxfp4_routed_pipeline() -> torch.Tensor:
            quantized = functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "mxfp4",
                num_comm_sms=args.comm_sms,
            )
            return run_mxfp4_routed_mlp(quantized)

        def run_nvfp4_routed_pipeline() -> torch.Tensor:
            quantized = functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "nvfp4",
                global_scale=nvfp4_global_scale,
                num_comm_sms=args.comm_sms,
            )
            return run_nvfp4_routed_mlp(quantized)

        if mxfp4_gemm_config_sweep:
            run_mxfp4_gate_up(actual_mxfp4)
            mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided_launch_inplace(
                gate_up_mxfp4,
                args.expert_hidden,
                args.expert_hidden,
                hidden_mxfp4,
                hidden_mxfp4_scales,
                1,
            )
            sweep_hidden_mxfp4 = (hidden_mxfp4, hidden_mxfp4_scales)
            for config_id in mxfp4_gemm_config_sweep:
                gemm_variants.extend(
                    (
                        (
                            f"MXFP4 gate/up GEMM config {config_id}",
                            lambda config_id=config_id: run_mxfp4_gate_up(
                                actual_mxfp4, config_id=config_id
                            ),
                        ),
                        (
                            f"MXFP4 down GEMM config {config_id}",
                            lambda config_id=config_id: run_mxfp4_down(
                                sweep_hidden_mxfp4, config_id=config_id
                            ),
                        ),
                        (
                            f"MXFP4 shared gate/up config {config_id}",
                            lambda config_id=config_id: mxfp4_gemm.mxfp4_gemm_config(
                                shared_x_mxfp4[0],
                                shared_x_mxfp4[1],
                                shared_gate_up_mxfp4_weights[0],
                                shared_gate_up_mxfp4_weights[1],
                                shared_gate_up_mxfp4,
                                config_id,
                            ),
                        ),
                        (
                            f"MXFP4 shared down config {config_id}",
                            lambda config_id=config_id: mxfp4_gemm.mxfp4_gemm_config(
                                shared_hidden_mxfp4,
                                shared_hidden_mxfp4_scales,
                                shared_down_mxfp4_weights[0],
                                shared_down_mxfp4_weights[1],
                                shared_output_mxfp4,
                                config_id,
                            ),
                        ),
                    )
                )

        for config_id in nvfp4_gemm_config_sweep:
            for flatten_batches in (False, True):
                schedule_name = "flat" if flatten_batches else "z-grid"
                gemm_variants.extend(
                    (
                        (
                            f"NVFP4 pitched gate/up c{config_id} {schedule_name}",
                            lambda config_id=config_id,
                            flatten_batches=flatten_batches: run_nvfp4_gate_up(
                                actual_nvfp4,
                                config_id=config_id,
                                flatten_batches=flatten_batches,
                            ),
                        ),
                        (
                            f"NVFP4 pitched down c{config_id} {schedule_name}",
                            lambda config_id=config_id,
                            flatten_batches=flatten_batches: run_nvfp4_down(
                                hidden_nvfp4_row,
                                config_id=config_id,
                                flatten_batches=flatten_batches,
                            ),
                        ),
                    )
                )

        gemm_variants.append(
            (
                "MXFP4 row-only SwiGLU quant",
                lambda: mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided_launch_inplace(
                    gate_up_mxfp4,
                    args.expert_hidden,
                    args.expert_hidden,
                    hidden_mxfp4,
                    hidden_mxfp4_scales,
                    1,
                ),
            )
        )
        gemm_variants.extend(
            (
                (
                    "NVFP4 row+col SwiGLU quant",
                    lambda: nvfp4_quant.tk_silu_quantize_for_gemm(
                        gate_up_nvfp4, args.expert_hidden, True
                    ),
                ),
                (
                    "NVFP4 row-only SwiGLU quant",
                    lambda: nvfp4_quant.tk_silu_quantize_for_gemm(
                        gate_up_nvfp4, args.expert_hidden, False
                    ),
                ),
                (
                    "NVFP4 constant-scale SwiGLU quant",
                    lambda: nvfp4_quant.tk_silu_quantize_for_gemm(
                        gate_up_nvfp4,
                        args.expert_hidden,
                        False,
                        True,
                    ),
                ),
            )
        )

        for tokens_per_cta in epilogue_tokens_per_cta_sweep:
            gemm_variants.extend(
                (
                    (
                        f"Isolated epilogue ({tokens_per_cta} tokens/CTA)",
                        lambda tokens_per_cta=tokens_per_cta: ops.fwd_epilogue(
                            epilogue_shared_output,
                            workspace.combine_buffer,
                            router_weights,
                            tokens_per_cta,
                            args.epilogue_cols_per_cta,
                        ),
                    ),
                    (
                        f"MXFP4 full MoE epilogue {tokens_per_cta} tokens/CTA",
                        lambda tokens_per_cta=tokens_per_cta: run_mxfp4_full_moe(
                            args.combine_sms,
                            None,
                            tokens_per_cta,
                        ),
                    ),
                )
            )

        for cols_per_cta in epilogue_cols_per_cta_sweep:
            gemm_variants.extend(
                (
                    (
                        f"Isolated epilogue ({cols_per_cta} columns/CTA)",
                        lambda cols_per_cta=cols_per_cta: ops.fwd_epilogue(
                            epilogue_shared_output,
                            workspace.combine_buffer,
                            router_weights,
                            args.epilogue_tokens_per_cta,
                            cols_per_cta,
                        ),
                    ),
                    (
                        f"MXFP4 full MoE epilogue {cols_per_cta} columns/CTA",
                        lambda cols_per_cta=cols_per_cta: run_mxfp4_full_moe(
                            args.combine_sms,
                            None,
                            args.epilogue_tokens_per_cta,
                            cols_per_cta,
                        ),
                    ),
                )
            )

        gemm_variants.extend(
            (
                (
                    "BF16 full MoE forward (prebuilt schedule)",
                    run_bf16_mok_forward,
                ),
                (
                    "MXFP8 full MoE forward (prebuilt schedule)",
                    run_mxfp8_mok_forward,
                ),
                ("BF16 routed expert MLP", run_bf16_routed_mlp),
                ("BF16 shared expert MLP", run_bf16_shared_mlp),
                ("MXFP4 shared expert MLP", run_mxfp4_shared_mlp),
                (
                    "MXFP4 shared input quant",
                    lambda: mxfp4_quant.mxfp4_quantize_for_gemm_launch_inplace(
                        x,
                        shared_x_mxfp4[0],
                        shared_x_mxfp4[1],
                        1,
                    ),
                ),
                (
                    "MXFP4 shared gate/up GEMM",
                    shared_mxfp4_gate_plan.run,
                ),
                (
                    "MXFP4 shared SwiGLU quant",
                    lambda: mxfp4_quant.mxfp4_fused_silu_mul_quantize_row_strided_launch_inplace(
                        shared_gate_up_mxfp4,
                        args.expert_hidden,
                        args.expert_hidden,
                        shared_hidden_mxfp4,
                        shared_hidden_mxfp4_scales,
                        1,
                    ),
                ),
                (
                    "MXFP4 shared down GEMM",
                    shared_mxfp4_down_plan.run,
                ),
                ("NVFP4 shared expert MLP", run_nvfp4_shared_mlp),
                (
                    "NVFP4 shared input quant",
                    lambda: nvfp4_quant.tk_quantize_for_gemm_constant_scale_into(
                        x,
                        shared_x_nvfp4[0],
                        shared_x_nvfp4[1],
                        shared_x_nvfp4[2],
                        shared_x_nvfp4_sync,
                    ),
                ),
                ("NVFP4 shared gate/up GEMM", shared_nvfp4_gate_plan.run),
                (
                    "NVFP4 shared SwiGLU quant",
                    lambda: nvfp4_quant.tk_silu_quantize_for_gemm_constant_scale_into(
                        shared_gate_up_nvfp4,
                        args.expert_hidden,
                        shared_hidden_nvfp4[0],
                        shared_hidden_nvfp4[1],
                        shared_hidden_nvfp4[2],
                        shared_hidden_nvfp4_sync,
                    ),
                ),
                ("NVFP4 shared down GEMM", shared_nvfp4_down_plan.run),
                ("MXFP4 gate/up GEMM", lambda: run_mxfp4_gate_up(actual_mxfp4)),
                ("MXFP4 dispatch + gate/up", run_mxfp4_pipeline),
                (
                    "NVFP4 gate/up GEMM",
                    lambda: run_nvfp4_gate_up(actual_nvfp4, actual_nvfp4_views),
                ),
                (
                    "NVFP4 cached gate/up GEMM",
                    pipelined_nvfp4_gate_plan.run,
                ),
                ("NVFP4 dispatch + gate/up", run_nvfp4_pipeline),
                (
                    "MXFP4 routed expert MLP",
                    lambda: run_mxfp4_routed_mlp(actual_mxfp4),
                ),
                ("MXFP4 dispatch + routed MLP", run_mxfp4_routed_pipeline),
                (
                    "NVFP4 routed expert MLP",
                    lambda: run_nvfp4_routed_mlp(actual_nvfp4),
                ),
                ("NVFP4 dispatch + routed MLP", run_nvfp4_routed_pipeline),
            )
        )
        gemm_variants.extend(
            (
                ("MXFP4 serial pull + routed MLP", run_mxfp4_serial_into),
                ("NVFP4 serial pull + routed MLP", run_nvfp4_serial_into),
            )
        )
        for group_count in pipeline_specs:
            gemm_variants.extend(
                (
                    (
                        f"MXFP4 streamed MLP ({group_count} groups, "
                        f"{args.comm_sms}/{overlap_comm_sms} SM)",
                        lambda group_count=group_count: run_mxfp4_streamed(
                            group_count
                        ),
                    ),
                    (
                        f"NVFP4 streamed MLP ({group_count} groups, "
                        f"{args.comm_sms}/{overlap_comm_sms} SM)",
                        lambda group_count=group_count: run_nvfp4_streamed(
                            group_count
                        ),
                    ),
                )
            )
        for pull_cols in mxfp4_pull_cols_sweep:
            gemm_variants.append(
                (
                    f"MXFP4 full MoE pull-cols {pull_cols}",
                    lambda pull_cols=pull_cols: run_mxfp4_full_moe(
                        args.combine_sms, pull_cols
                    ),
                )
            )
        for combine_cols in combine_cols_sweep:
            gemm_variants.extend(
                (
                    (
                        f"BF16 route push combine-cols {combine_cols}",
                        lambda combine_cols=combine_cols: run_combine_only(
                            routed_mxfp4,
                            args.combine_sms,
                            combine_cols,
                        ),
                    ),
                    (
                        f"MXFP4 full MoE combine-cols {combine_cols}",
                        lambda combine_cols=combine_cols: run_mxfp4_full_moe(
                            combine_cols=combine_cols
                        ),
                    ),
                )
            )
        for combine_sms in combine_sms_sweep:
            gemm_variants.extend(
                (
                    (
                        f"BF16 route push + barrier ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_combine_only(
                            routed_mxfp4, combine_sms
                        ),
                    ),
                    (
                        f"MXFP4 serial pull + MLP + push ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: (
                            run_mxfp4_serial_with_combine(combine_sms)
                        ),
                    ),
                    (
                        f"NVFP4 serial pull + MLP + push ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: (
                            run_nvfp4_serial_with_combine(combine_sms)
                        ),
                    ),
                    (
                        f"MXFP4 full MoE forward ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: (
                            run_mxfp4_full_moe(combine_sms)
                        ),
                    ),
                    (
                        f"MXFP4 full MoE FP4 shared ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_mxfp4_full_moe(
                            combine_sms,
                            use_mxfp4_shared=True,
                        ),
                    ),
                    (
                        f"NVFP4 full MoE forward ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: (
                            run_nvfp4_full_moe(combine_sms)
                        ),
                    ),
                    (
                        f"NVFP4 full MoE shared-early ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_nvfp4_full_moe(
                            combine_sms,
                            launch_shared_early=True,
                        ),
                    ),
                    (
                        f"NVFP4 full MoE constant-scale ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_nvfp4_full_moe(
                            combine_sms,
                            launch_shared_early=True,
                            use_constant_scale=True,
                        ),
                    ),
                    (
                        f"NVFP4 full MoE cached-gate ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_nvfp4_full_moe(
                            combine_sms,
                            launch_shared_early=True,
                            use_constant_scale=True,
                            use_cached_gate=True,
                        ),
                    ),
                    (
                        f"NVFP4 full MoE cached-plans ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_nvfp4_full_moe(
                            combine_sms,
                            launch_shared_early=True,
                            use_constant_scale=True,
                            use_cached_gate=True,
                            use_cached_down=True,
                        ),
                    ),
                    (
                        f"NVFP4 full MoE all-FP4 ({combine_sms} SM)",
                        lambda combine_sms=combine_sms: run_nvfp4_full_moe(
                            combine_sms,
                            launch_shared_early=True,
                            use_constant_scale=True,
                            use_cached_gate=True,
                            use_cached_down=True,
                            use_nvfp4_shared=True,
                        ),
                    ),
                )
            )
            for group_count in pipeline_specs:
                gemm_variants.extend(
                    (
                        (
                            f"MXFP4 streamed MLP + push ({group_count} groups, "
                            f"{args.comm_sms}/{overlap_comm_sms}/{combine_sms} SM)",
                            lambda group_count=group_count, combine_sms=combine_sms: (
                                run_mxfp4_streamed(
                                    group_count,
                                    push_output=True,
                                    combine_sms=combine_sms,
                                )
                            ),
                        ),
                        (
                            f"NVFP4 streamed MLP + push ({group_count} groups, "
                            f"{args.comm_sms}/{overlap_comm_sms}/{combine_sms} SM)",
                            lambda group_count=group_count, combine_sms=combine_sms: (
                                run_nvfp4_streamed(
                                    group_count,
                                    push_output=True,
                                    combine_sms=combine_sms,
                                )
                            ),
                        ),
                    )
                )
    dist.barrier()

    workspace.x_buffer.copy_(x)
    ops.barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )

    variants = [
        (
            "MXFP4 fused dispatch E2E",
            lambda: functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "mxfp4",
                num_comm_sms=args.comm_sms,
            ),
        ),
        (
            "MXFP4 local quant ceiling",
            lambda: mxfp4_quant.mxfp4_quantize_for_gemm(reference, 1),
        ),
        (
            "NVFP4 fused dispatch E2E",
            lambda: functional.dispatch_fp4(
                workspace,
                schedule,
                x,
                "nvfp4",
                global_scale=nvfp4_global_scale,
                num_comm_sms=args.comm_sms,
            ),
        ),
        (
            "NVFP4 local quant ceiling",
            lambda: nvfp4_quant.tk_quantize_for_gemm(reference, False, True),
        ),
    ]
    variants.extend(gemm_variants)
    for comm_sms in comm_sms_sweep:
        variants.extend(
            (
                (
                    f"MXFP4 pull+quant ({comm_sms} SM)",
                    lambda comm_sms=comm_sms: ops.dispatch_mxfp4(
                        workspace.x_buffer,
                        workspace.x_buffer_ptrs,
                        schedule.peer_rank,
                        schedule.peer_token_idx,
                        schedule.num_tokens,
                        workspace.topk,
                        comm_sms,
                    ),
                ),
                (
                    f"NVFP4 pull+quant ({comm_sms} SM)",
                    lambda comm_sms=comm_sms: ops.dispatch_nvfp4(
                        workspace.x_buffer,
                        workspace.x_buffer_ptrs,
                        schedule.peer_rank,
                        schedule.peer_token_idx,
                        schedule.num_tokens,
                        nvfp4_global_scale,
                        workspace.topk,
                        comm_sms,
                    ),
                ),
            )
        )

    if rank == 0:
        print(
            f"EP{world_size} tokens/rank={args.tokens} routed_rows/rank={num_valid_tokens} "
            f"padded_rows/rank={num_tokens} "
            f"experts={args.experts} topk={args.topk} H={args.hidden} "
            f"comm_sms={args.comm_sms} "
            f"overlap_comm_sms={overlap_comm_sms if args.expert_hidden else 'n/a'} "
            f"combine_sms={args.combine_sms if args.expert_hidden else 'n/a'} "
            f"combine_cols={(args.combine_cols or 'auto') if args.expert_hidden else 'n/a'} "
            f"mxfp4_pull_cols={args.mxfp4_pull_cols or 'default'} "
            f"epilogue_tokens_per_cta={args.epilogue_tokens_per_cta} "
            f"epilogue_cols_per_cta={args.epilogue_cols_per_cta or 'auto'}"
        )
        print(
            f"Bit-exact on {sample_count} sampled rows against an independent "
            "Torch FP4 reference: PASS"
        )
        if gemm_relative_rms is not None:
            print(
                f"Gate/up GEMM sampled relative RMS vs BF16: "
                f"MXFP4={gemm_relative_rms[0]:.4f}, "
                f"NVFP4={gemm_relative_rms[1]:.4f}"
            )
        if routed_mlp_relative_rms is not None:
            print(
                f"Routed MLP sampled relative RMS vs BF16: "
                f"MXFP4={routed_mlp_relative_rms[0]:.4f}, "
                f"NVFP4={routed_mlp_relative_rms[1]:.4f}"
            )
        if nvfp4_constant_routed_mlp_relative_rms is not None:
            print(
                "NVFP4 constant-scale routed MLP relative RMS vs BF16: "
                f"{nvfp4_constant_routed_mlp_relative_rms:.4f}"
            )
        if bf16_routed_mlp_relative_rms is not None:
            print(
                "BF16 routed MLP sampled relative RMS vs FP32 reference: "
                f"{bf16_routed_mlp_relative_rms:.4f}"
            )
        if mxfp4_shared_mlp_relative_rms is not None:
            print(
                "MXFP4 shared MLP relative RMS vs BF16: "
                f"{mxfp4_shared_mlp_relative_rms:.4f}"
            )
        if nvfp4_shared_mlp_relative_rms is not None:
            print(
                "NVFP4 shared MLP relative RMS vs BF16: "
                f"{nvfp4_shared_mlp_relative_rms:.4f}"
            )
        for group_count, relative_rms in pipeline_relative_rms.items():
            print(
                f"Streamed {group_count}-group MLP sampled relative RMS vs BF16: "
                f"MXFP4={relative_rms[0]:.4f}, "
                f"NVFP4={relative_rms[1]:.4f}"
            )
        if combine_sample_count:
            print(
                f"MXFP4/NVFP4 streamed push combine bit-exact on "
                f"{combine_sample_count} sampled source routes: PASS"
            )
        if full_moe_relative_rms is not None:
            print(
                "Full MoE assembly relative RMS vs Torch epilogue: "
                f"MXFP4={full_moe_relative_rms[0]:.6f}, "
                f"NVFP4={full_moe_relative_rms[1]:.6f}"
            )
        if mxfp4_shared_full_moe_relative_rms is not None:
            print(
                "MXFP4-shared full MoE relative RMS vs BF16 shared: "
                f"{mxfp4_shared_full_moe_relative_rms:.4f}"
            )
        if nvfp4_shared_full_moe_relative_rms is not None:
            print(
                "NVFP4-shared full MoE relative RMS vs BF16 shared: "
                f"{nvfp4_shared_full_moe_relative_rms:.4f}"
            )
    if args.benchmark_filter:
        variants = [
            variant for variant in variants
            if args.benchmark_filter in variant[0]
        ]
        if not variants:
            raise ValueError(
                f"no benchmark names contain {args.benchmark_filter!r}"
            )
    if args.cuda_profiler_range:
        torch.cuda.profiler.start()
    for name, fn in variants:
        latency_ms = _time(fn, device, args.warmup, args.iters)
        if rank == 0:
            _print_result(name, latency_ms, args.tokens)
    if args.cuda_profiler_range:
        torch.cuda.profiler.stop()

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
