#!/usr/bin/env python3
"""Benchmark the FLA 0.5.2 Triton KDA forward implementation on one B200."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import inspect
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence


HEADS = 96
KEY_DIM = 128
VALUE_DIM = 128
CHUNK_SIZE = 64
SEQUENCE_LENGTHS = (1024, 8192, 16384, 65536)
ATK_CASE_ID_START = 250
SEED_BASE = 20260812
EXPECTED_FLA_VERSION = "0.5.2"
B200_BF16_DENSE_PEAK_TFLOPS = 2250.0
A5_US = {
    250: 576.645,
    251: 631.411,
    252: 4284.247,
    253: 4773.215,
    254: 8501.267,
    255: 9520.606,
    256: 33911.582,
    257: 38118.852,
}


@dataclass(frozen=True)
class Case:
    case_id: int
    total_tokens: int
    disable_recompute: bool
    seed: int
    a5_us: float

    @property
    def recompute_enabled(self) -> bool:
        return not self.disable_recompute

    @property
    def chunk_count(self) -> int:
        return self.total_tokens // CHUNK_SIZE

    @property
    def case_key(self) -> str:
        enabled = str(self.recompute_enabled).lower()
        return (
            f"ascend950_h96_t{self.total_tokens}_c64_dense_"
            f"recompute_{enabled}"
        )


def build_cases() -> tuple[Case, ...]:
    cases = []
    for sequence_index, total_tokens in enumerate(SEQUENCE_LENGTHS):
        for disable_recompute in (False, True):
            case_id = (
                ATK_CASE_ID_START
                + sequence_index * 2
                + int(disable_recompute)
            )
            cases.append(
                Case(
                    case_id=case_id,
                    total_tokens=total_tokens,
                    disable_recompute=disable_recompute,
                    seed=SEED_BASE + sequence_index,
                    a5_us=A5_US[case_id],
                )
            )
    result = tuple(cases)
    if tuple(case.case_id for case in result) != tuple(range(250, 258)):
        raise RuntimeError("B200 Triton matrix must contain case IDs 250-257")
    return result


CASES = build_cases()
CASE_BY_ID = {str(case.case_id): case for case in CASES}
CASE_BY_KEY = {case.case_key: case for case in CASES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the eight dense KDA cases with the installed FLA Triton "
            "implementation and report the mean of 10 CUDA-event timings."
        )
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--expected-fla-version",
        default=EXPECTED_FLA_VERSION,
        help="installed flash-linear-attention version required by the run",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=B200_BF16_DENSE_PEAK_TFLOPS,
        help="single-GPU dense BF16 peak used for MFU",
    )
    parser.add_argument("--allow-non-b200", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def select_cases(value: str) -> list[Case]:
    if value == "all":
        return list(CASES)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    selectors = {**CASE_BY_ID, **CASE_BY_KEY}
    unknown = sorted(set(requested).difference(selectors))
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return [selectors[item] for item in requested]


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / "kda-b200-triton" / f"run_{timestamp}_{os.getpid()}"


def _distribution_metadata():
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - FLA 0.5.2 requires newer Python.
        import importlib_metadata as metadata
    return metadata


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def load_installed_fla(expected_version: str):
    metadata = _distribution_metadata()
    try:
        distribution = metadata.distribution("flash-linear-attention")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "flash-linear-attention is not installed in the current Python environment"
        ) from exc
    if distribution.version != expected_version:
        raise RuntimeError(
            "unexpected flash-linear-attention version: "
            f"expected {expected_version}, got {distribution.version}"
        )

    distribution_root = Path(distribution.locate_file("")).resolve()
    installed_fla_root = Path(distribution.locate_file("fla")).resolve()
    expected_source = installed_fla_root / "ops" / "kda" / "chunk_fwd.py"
    if not expected_source.is_file():
        raise RuntimeError(
            "installed flash-linear-attention does not contain "
            f"{expected_source}"
        )

    # This repository also has a minimal top-level `fla` package. Put the pip
    # distribution first so the benchmark cannot silently import that package.
    sys.path.insert(0, str(distribution_root))
    for module_name in tuple(sys.modules):
        if module_name == "fla" or module_name.startswith("fla."):
            del sys.modules[module_name]

    module = importlib.import_module("fla.ops.kda.chunk_fwd")
    operation = getattr(module, "chunk_kda_fwd", None)
    if not callable(operation):
        raise RuntimeError("installed FLA chunk_kda_fwd is not callable")
    source = inspect.getsourcefile(operation)
    if source is None or not _is_within(Path(source).resolve(), installed_fla_root):
        raise RuntimeError(
            "chunk_kda_fwd was not imported from the installed "
            "flash-linear-attention distribution"
        )

    required_parameters = {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "state_v_first",
        "chunk_size",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "A_log",
        "dt_bias",
        "disable_recompute",
        "return_intermediate_states",
    }
    signature = inspect.signature(operation)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        missing = sorted(required_parameters.difference(signature.parameters))
        if missing:
            raise RuntimeError(
                "installed FLA chunk_kda_fwd has an incompatible signature; "
                f"missing parameters: {', '.join(missing)}"
            )
    return operation, distribution.version, Path(source).resolve()


def normal_quantized(
    torch,
    shape: Sequence[int],
    generator,
    original_dtype,
    device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    sigmoid: bool = False,
    l2_normalize: bool = False,
):
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = value.mul(std).add(mean)
    if sigmoid:
        value = torch.sigmoid(value)
    if l2_normalize:
        value = value * torch.rsqrt(
            value.square().sum(dim=-1, keepdim=True) + 1e-6
        )
    return value.to(original_dtype).to(device)


def make_inputs(torch, case: Case, device) -> dict:
    shape = (1, case.total_tokens, HEADS, KEY_DIM)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)
    return {
        "q": normal_quantized(
            torch,
            shape,
            generator,
            torch.bfloat16,
            device,
            std=0.05,
            l2_normalize=True,
        ),
        "k": normal_quantized(
            torch,
            shape,
            generator,
            torch.bfloat16,
            device,
            std=0.05,
            l2_normalize=True,
        ),
        "v": normal_quantized(
            torch, shape, generator, torch.bfloat16, device, std=0.05
        ),
        "g": normal_quantized(
            torch, shape, generator, torch.float32, device, std=1.25
        ),
        "beta": normal_quantized(
            torch,
            (1, case.total_tokens, HEADS),
            generator,
            torch.bfloat16,
            device,
            mean=1.5,
            std=0.35,
            sigmoid=True,
        ),
        "A_log": normal_quantized(
            torch,
            (HEADS,),
            generator,
            torch.float32,
            device,
            std=0.12,
        ),
        "dt_bias": normal_quantized(
            torch,
            (HEADS * KEY_DIM,),
            generator,
            torch.float32,
            device,
            mean=-3.0,
            std=1.65,
        ),
    }


def invoke(operation: Callable, inputs: dict, case: Case):
    return operation(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=KEY_DIM**-0.5,
        initial_state=None,
        output_final_state=False,
        state_v_first=True,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        chunk_size=CHUNK_SIZE,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        disable_recompute=case.disable_recompute,
        return_intermediate_states=False,
    )


def validate_outputs(torch, outputs, case: Case) -> None:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 12:
        raise RuntimeError("FLA chunk_kda_fwd must return 12 values")
    output = outputs[0]
    expected_shape = (1, case.total_tokens, HEADS, VALUE_DIM)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected output shape: expected {expected_shape}, got {tuple(output.shape)}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("attention output contains NaN or Inf")


def kda_flops(case: Case) -> float:
    chunk = CHUNK_SIZE
    per_head_chunk = (
        6 * chunk * chunk * KEY_DIM
        + 4 * chunk * chunk * VALUE_DIM
        + 6 * chunk * KEY_DIM * VALUE_DIM
        + chunk * (chunk - 1) * (chunk + 1) / 3
    )
    return HEADS * case.chunk_count * per_head_chunk


def summarize_timings(
    case: Case, timings_us: Sequence[float], peak_tflops: float
) -> dict:
    if not timings_us:
        raise ValueError("at least one timing is required")
    mean_us = sum(timings_us) / len(timings_us)
    effective_tflops = kda_flops(case) / (mean_us * 1e-6) / 1e12
    return {
        "mean_us": mean_us,
        "median_us": statistics.median(timings_us),
        "min_us": min(timings_us),
        "max_us": max(timings_us),
        "stddev_us": statistics.pstdev(timings_us),
        "effective_tflops": effective_tflops,
        "mfu_percent": effective_tflops / peak_tflops * 100.0,
        "a5_over_b200": case.a5_us / mean_us,
    }


def benchmark_case(
    torch,
    operation: Callable,
    case: Case,
    device,
    warmup: int,
    runs: int,
    peak_tflops: float,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    inputs = make_inputs(torch, case, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    outputs = None
    for warmup_index in range(warmup):
        outputs = invoke(operation, inputs, case)
        torch.cuda.synchronize(device)
        print(
            f"case={case.case_id} warmup={warmup_index + 1}/{warmup}",
            flush=True,
        )
        del outputs
        outputs = None

    timings_us = []
    for run_index in range(runs):
        if outputs is not None:
            del outputs
            outputs = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = invoke(operation, inputs, case)
        end.record()
        end.synchronize()
        elapsed_us = float(start.elapsed_time(end) * 1000.0)
        timings_us.append(elapsed_us)
        print(
            f"case={case.case_id} run={run_index + 1:02d}/{runs} "
            f"elapsed_us={elapsed_us:.3f}",
            flush=True,
        )

    validate_outputs(torch, outputs, case)
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    summary = summarize_timings(case, timings_us, peak_tflops)
    del outputs
    del inputs
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return {
        **asdict(case),
        "case_key": case.case_key,
        "layout_mode": "dense",
        "sequence_count": 1,
        "chunk_count": case.chunk_count,
        "recompute_enabled": case.recompute_enabled,
        "status": "PASS",
        "warmup": warmup,
        "runs": runs,
        "timings_us": timings_us,
        "peak_memory_gib": peak_memory_bytes / 2**30,
        "total_flops": kda_flops(case),
        **summary,
        "error": "",
    }


def error_result(case: Case, warmup: int, runs: int, error: Exception) -> dict:
    return {
        **asdict(case),
        "case_key": case.case_key,
        "layout_mode": "dense",
        "sequence_count": 1,
        "chunk_count": case.chunk_count,
        "recompute_enabled": case.recompute_enabled,
        "status": "ERROR",
        "warmup": warmup,
        "runs": runs,
        "timings_us": [],
        "peak_memory_gib": None,
        "total_flops": kda_flops(case),
        "mean_us": None,
        "median_us": None,
        "min_us": None,
        "max_us": None,
        "stddev_us": None,
        "effective_tflops": None,
        "mfu_percent": None,
        "a5_over_b200": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _number(value: Optional[float], digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def markdown_report(metadata: dict, results: Sequence[dict]) -> str:
    lines = [
        "# FLA Triton KDA performance on B200",
        "",
        f"- GPU: `{metadata['gpu_name']}`",
        f"- flash-linear-attention: `{metadata['fla_version']}`",
        f"- PyTorch: `{metadata['torch_version']}`",
        f"- Triton: `{metadata['triton_version']}`",
        f"- warmup runs per case: `{metadata['warmup']}`",
        f"- measured runs per case: `{metadata['runs']}`",
        f"- BF16 dense peak for MFU: `{metadata['peak_tflops']:.1f} TFLOPS`",
        "",
        "| Case | total tokens | layout mode | sequence count | chunk count | "
        "recompute enabled | status | B200 mean us | B200 MFU | A5 us | "
        "A5/B200 |",
        "| ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        values = dict(result)
        values["mean_us_text"] = _number(result["mean_us"])
        values["mfu_text"] = (
            ""
            if result["mfu_percent"] is None
            else f"{result['mfu_percent']:.2f}%"
        )
        values["speedup_text"] = (
            ""
            if result["a5_over_b200"] is None
            else f"{result['a5_over_b200']:.3f}x"
        )
        lines.append(
            "| {case_id} | {total_tokens} | dense | 1 | {chunk_count} | "
            "{recompute_enabled} | {status} | {mean_us_text} | {mfu_text} | "
            "{a5_us:.3f} | {speedup_text} |".format(**values)
        )
    errors = [result for result in results if result["status"] != "PASS"]
    if errors:
        lines.extend(["", "## Errors", ""])
        for result in errors:
            lines.append(f"- case {result['case_id']}: `{result['error']}`")
    return "\n".join(lines) + "\n"


def write_reports(output_dir: Path, metadata: dict, results: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {"metadata": metadata, "results": list(results)}
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    summary_fields = [
        "case_id",
        "case_key",
        "total_tokens",
        "layout_mode",
        "sequence_count",
        "chunk_count",
        "recompute_enabled",
        "disable_recompute",
        "status",
        "warmup",
        "runs",
        "mean_us",
        "median_us",
        "min_us",
        "max_us",
        "stddev_us",
        "effective_tflops",
        "mfu_percent",
        "peak_memory_gib",
        "a5_us",
        "a5_over_b200",
        "error",
    ]
    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in summary_fields})

    with (output_dir / "timings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("case_id", "run", "elapsed_us")
        )
        writer.writeheader()
        for result in results:
            for run, elapsed_us in enumerate(result["timings_us"], start=1):
                writer.writerow(
                    {
                        "case_id": result["case_id"],
                        "run": run,
                        "elapsed_us": elapsed_us,
                    }
                )

    report = markdown_report(metadata, results)
    (output_dir / "results.md").write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"Results: {output_dir.resolve()}", flush=True)


def package_version(name: str) -> str:
    metadata = _distribution_metadata()
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    args = parse_args()
    cases = select_cases(args.cases)
    if args.list_cases:
        for case in cases:
            print(
                f"{case.case_id} {case.case_key} T={case.total_tokens} "
                f"chunks={case.chunk_count} "
                f"recompute_enabled={case.recompute_enabled}"
            )
        return 0
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be nonnegative and --runs must be positive")
    if args.peak_tflops <= 0:
        raise ValueError("--peak-tflops must be positive")

    # FLA 0.5.2 can dispatch to optional non-Triton backends. These variables
    # force its default Triton implementation before the package is imported.
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    os.environ["FLA_TILELANG"] = "0"
    os.environ["FLA_FLASH_KDA"] = "0"

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "B200" not in gpu_name.upper() and not args.allow_non_b200:
        raise RuntimeError(
            f"expected an NVIDIA B200, got {gpu_name!r}; "
            "use --allow-non-b200 only for intentional non-B200 runs"
        )

    operation, fla_version, fla_source = load_installed_fla(
        args.expected_fla_version
    )
    output_dir = args.output_dir or default_output_dir()
    metadata = {
        "gpu_name": gpu_name,
        "device": args.device,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": package_version("triton"),
        "fla_version": fla_version,
        "fla_source": "fla/ops/kda/chunk_fwd.py",
        "fla_distribution_location_verified": True,
        "warmup": args.warmup,
        "runs": args.runs,
        "peak_tflops": args.peak_tflops,
        "timing_scope": "complete low-level FLA chunk_kda_fwd via CUDA events",
        "backend_env": {
            "FLA_DISABLE_BACKEND_DISPATCH": "1",
            "FLA_TILELANG": "0",
            "FLA_FLASH_KDA": "0",
        },
    }
    print(
        f"GPU={gpu_name} flash-linear-attention={fla_version} "
        f"source={fla_source}",
        flush=True,
    )

    results = []
    with torch.inference_mode():
        for case in cases:
            print(
                f"\n[{case.case_id}] T={case.total_tokens} "
                f"recompute_enabled={case.recompute_enabled}",
                flush=True,
            )
            try:
                result = benchmark_case(
                    torch,
                    operation,
                    case,
                    device,
                    args.warmup,
                    args.runs,
                    args.peak_tflops,
                )
            except Exception as exc:  # Keep later cases available after an OOM.
                print(f"case={case.case_id} ERROR: {exc}", file=sys.stderr)
                result = error_result(case, args.warmup, args.runs, exc)
                gc.collect()
                torch.cuda.empty_cache()
            results.append(result)

    write_reports(output_dir, metadata, results)
    return 0 if all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
