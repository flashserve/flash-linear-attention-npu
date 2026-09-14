"""Build the root wheel and print its exact installation command."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fla_npu_artifacts import get_wheel_filename  # noqa: E402


def _resolve_output_dir(value: str) -> Path:
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    return output_dir.resolve()


def _install_command(wheel_path: Path) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m pip install "
        "--force-reinstall --no-cache-dir --no-deps "
        f"{shlex.quote(str(wheel_path))}"
    )


def _collect_build_args(args: argparse.Namespace) -> str:
    parts = list(args.build_args)
    env_args = os.getenv("FLA_NPU_BUILD_ARGS", "").strip()
    if env_args:
        parts.insert(0, env_args)
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_values(build_args: str, option: str) -> list:
    """Extract the comma-separated values of an option from the forwarded args."""
    values: list[str] = []
    tokens = build_args.split()
    for i, token in enumerate(tokens):
        if token == option:
            if i + 1 < len(tokens):
                values.extend(tokens[i + 1].split(","))
            continue
        if token.startswith(f"{option}="):
            values.extend(token.split("=", 1)[1].split(","))
    return values


def _drop_option(build_args: str, option: str) -> str:
    """Remove all occurrences of an option (space-separated or = spelling)."""
    tokens = build_args.split()
    filtered: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == option:
            i += 2
            continue
        if token.startswith(f"{option}="):
            i += 1
            continue
        filtered.append(token)
        i += 1
    return " ".join(filtered)


def _native_build_args(args: argparse.Namespace) -> list:
    """Map 一键编包的原生 -g / --sanitizer / --oom 选项到 asc_opc 合法值。

    - -g / --debug   -> ccec_g（kernel 调试信息）
    - --sanitizer    -> sanitizer（asc_opc 内存越界插桩，CANN 9.1.0 起合法）
    - --oom          -> oom（kernel 侧 OOM 检查）

    值通过 build.sh --bisheng_flags 或 --op_debug_config 传递，最终由
    ascendc_bin_param_build.py 拼成 asc_opc 的 --op_debug_config=<values>。
    CANN 9.1.0 的 asc_opc 合法值表为
    (oom, dump_cce, dump_bin, dump_loc, ccec_O0, ccec_g, check_flag, sanitizer)，
    因此 --sanitizer 必须映射为 sanitizer，不能使用更高版本才识别的
    check_flag_sanitizer。
    """
    configs = []
    if args.debug:
        configs.append("ccec_g")
    if args.sanitizer:
        configs.append("sanitizer")
    if args.oom:
        configs.append("oom")
    return configs


def _assemble_build_args(args: argparse.Namespace) -> str:
    build_args = _collect_build_args(args)
    native = _native_build_args(args)
    if not native:
        return build_args
    # 原生选项优先：将 --bisheng_flags 与 --op_debug_config 中已经存在的
    # 用户显式值全部取出，与原生值合并去重后只传一次，保留 dump_cce 等
    # 用户显式配置（review 之前的问题：直接丢弃用户的配置）。
    option = "--bisheng_flags"
    existing = (
        _extract_values(build_args, "--bisheng_flags")
        + _extract_values(build_args, "--op_debug_config")
    )
    build_args = _drop_option(build_args, "--bisheng_flags")
    build_args = _drop_option(build_args, "--op_debug_config")
    merged = list(dict.fromkeys(native + existing))
    # build.sh 只识别 --bisheng_flags=<values> 的等号写法，不能用空格分隔。
    tail = f"{option}={','.join(merged)}" if merged else ""
    return f"{build_args} {tail}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        default="dist",
        help="wheel output directory relative to the repository root (default: dist)",
    )
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help=(
            "add kernel debug info (asc_opc -g). Equivalent to passing "
            "--op_debug_config ccec_g to build.sh."
        ),
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help=(
            "enable Ascend kernel memory sanitizer support for mssanitizer. "
            "Maps to sanitizer (asc_opc --op_debug_config=sanitizer), which "
            "instruments kernels to detect memory errors. Runtime detection is "
            "done by mssanitizer via LD_PRELOAD injection "
            "(libmssanitizer_injection.so). Requires the Ascend toolkit's "
            "mssanitizer debug environment when running."
        ),
    )
    parser.add_argument(
        "--oom",
        action="store_true",
        help=(
            "enable kernel-side OOM debug. Maps to oom (asc_opc "
            "--op_debug_config=oom) for build.sh."
        ),
    )
    parser.add_argument(
        "--build-args",
        action="append",
        default=[],
        metavar="ARGS",
        help=(
            "extra arguments forwarded to build.sh (e.g. "
            "--build-args='-O3'). build.sh parses option values "
            "space-separated, so do not use '=' between an option and its "
            "value. May be repeated or space-separated within one value. "
            "Also honored via the FLA_NPU_BUILD_ARGS environment variable."
        ),
    )
    args = parser.parse_args()

    wheel_dir = _resolve_output_dir(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / get_wheel_filename(REPO_ROOT)

    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        ".",
        "-w",
        str(wheel_dir),
    ]

    env = os.environ.copy()
    build_args = _assemble_build_args(args)
    if build_args:
        env["FLA_NPU_BUILD_ARGS"] = build_args
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)

    if not wheel_path.is_file():
        raise RuntimeError(f"Expected wheel was not produced: {wheel_path}")

    print(f"[fla-npu build] Wheel: {wheel_path}", flush=True)
    print(f"[fla-npu build] Install command:", flush=True)
    print(_install_command(wheel_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
