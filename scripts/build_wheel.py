"""Build the root wheel and print its exact installation command.

默认给"一键编包"加一条进度条：编包是一串阶段（pip 读元数据 → 环境预检 →
生成算子 OPP run 包 → 安装 run 包 → 组装 wheel），编译算子阶段占绝大部分
时间。TTY 下渲染单行进度条，非 TTY 下退化为里程碑行，``--no-progress`` 可
完全关闭。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Lowest torch whose stable headers/symbols the launcher was verified against.
# Built against 2.9 headers, loaded and run under 2.7.1 (241: py3.10 +
# torch 2.7.1.post5 + torch_npu 2.7.1.post5, full Ascend950 scenario set).
STABLE_ABI_MIN_TORCH = "2.7.1"

# 一键编包的阶段划分与权重。权重和实际耗时大致同量级：编译算子占大头。
_PROGRESS_STAGES: tuple[tuple[str, float], ...] = (
    ("准备构建", 0.03),
    ("环境预检", 0.05),
    ("编译算子", 0.72),
    ("打包 OPP", 0.08),
    ("组装 wheel", 0.09),
    ("收尾", 0.03),
)

_PROGRESS_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# build.sh 的整个主流程 stdout 会被 `} | gawk '{print strftime(...), $0}'` 加上
# `[YYYY-MM-DD HH:MM:SS] ` 前缀，因此 make 的 `[ NN%]` 行不总在行首，这里按
# 行内匹配。
_MAKE_PROGRESS_RE = re.compile(r"\[\s*(\d{1,3})%\]")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _stage_index(name: str) -> int:
    for index, (label, _) in enumerate(_PROGRESS_STAGES):
        if label == name:
            return index
    raise KeyError(name)


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _BuildProgress:
    """一键编包进度条。

    进度按阶段权重推进；编译阶段内优先用 make 输出的 ``[ NN%]`` 换算百分比
    （make 在同一阶段会为多个 target 反复从 0% 开始，这里只取历史最大值，
    保证进度单调不回退），没有该信号时用计时器与最近一条构建日志表示仍在
    活动。

    - TTY：渲染单行进度条；输出子进程原始日志前先清掉该行，日志本身不受
      影响；
    - 非 TTY（CI 日志）：只按阶段打印里程碑行，不写 ANSI 控制符；
    - ``--no-progress`` 或 ``FLA_NPU_BUILD_PROGRESS=0``：完全关闭。
    """

    BAR_WIDTH = 24

    def __init__(self, enabled: bool, announce: bool, stream=None) -> None:
        self._bar_enabled = enabled
        self._announce = announce
        self._stream = stream if stream is not None else sys.stdout
        self._stage = 0
        self._fraction = 0.0
        self._started = time.monotonic()
        self._frame = 0
        self._last_draw = 0.0
        self._line_open = False
        self._announced = -1

    # ---- 内部渲染 ----------------------------------------------------

    def _overall(self) -> float:
        done = sum(weight for _, weight in _PROGRESS_STAGES[: self._stage])
        done += _PROGRESS_STAGES[self._stage][1] * self._fraction
        return min(done, 1.0)

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def _clear_line(self) -> None:
        if self._bar_enabled and self._line_open:
            self._write("\r\x1b[2K")
            self._line_open = False

    def _announce_stage(self) -> None:
        if not self._announce or self._announced >= self._stage:
            return
        self._announced = self._stage
        label = _PROGRESS_STAGES[self._stage][0]
        self._clear_line()
        self._write(
            f"[fla-npu build] [{self._stage + 1}/{len(_PROGRESS_STAGES)}] "
            f"{label}\n"
        )

    def _draw(self, force: bool = False) -> None:
        if not self._bar_enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < 0.1:
            return
        self._last_draw = now
        ratio = self._overall()
        filled = int(round(ratio * self.BAR_WIDTH))
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        spinner = _PROGRESS_SPINNER[self._frame % len(_PROGRESS_SPINNER)]
        label = _PROGRESS_STAGES[self._stage][0]
        self._write(
            f"\r\x1b[2K[fla-npu build] {bar} {ratio * 100:3.0f}%"
            f"  {label}  {spinner}  {_format_elapsed(now - self._started)}"
        )
        self._line_open = True

    # ---- 阶段推进 ----------------------------------------------------

    def enter(self, stage: str, fraction: float = 0.0) -> None:
        """进入阶段；同名或更早的阶段只更新阶段内进度，不回退。"""

        index = _stage_index(stage)
        if index > self._stage:
            self._stage = index
            self._fraction = min(max(fraction, 0.0), 1.0)
        elif index == self._stage:
            self.set_fraction(fraction)
        self._announce_stage()
        self._draw(force=True)

    def set_fraction(self, fraction: float) -> None:
        """设置阶段内完成比例，只增不减。"""

        self._fraction = max(self._fraction, min(max(fraction, 0.0), 1.0))
        self._draw()

    def _note_start(self, command: str) -> None:
        if "build.sh" in command:
            self.enter("编译算子")
        elif "--install-path" in command:
            self.enter("打包 OPP", 0.2)
        elif "prepare_offline_bundle" in command or "build_stable" in command:
            self.enter("组装 wheel", 0.3)
        elif "bdist_wheel" in command:
            self.enter("组装 wheel", 0.6)
        else:
            self.enter("准备构建")

    def _note_finish(self, command: str) -> None:
        if "build.sh" in command:
            self.enter("打包 OPP")
        elif "--install-path" in command:
            self.enter("打包 OPP", 0.8)
        elif "prepare_offline_bundle" in command or "build_stable" in command:
            self.enter("组装 wheel", 0.6)
        else:
            self.enter("收尾")

    def feed(self, line: str) -> None:
        """根据子进程输出推进进度。"""

        text = _ANSI_ESCAPE_RE.sub("", line).strip()
        if not text:
            return
        match = _MAKE_PROGRESS_RE.search(text)
        if match:
            self.enter("编译算子", int(match.group(1)) / 100.0)
            return
        if "[fla-npu build] START" in text:
            self._note_start(text.split("START", 1)[1])
            return
        if "[fla-npu build] DONE" in text or "[fla-npu build] FAILED" in text:
            self._note_finish(text.split(":", 1)[-1])
            return
        if "Embedded OPP staged at" in text:
            self.enter("组装 wheel", 0.5)
            return
        if text.startswith("[fla-npu build] staged "):
            self.enter("组装 wheel", 0.4)
            return
        if text.startswith("[fla-npu build] pinned "):
            self.enter("收尾")
            return
        if "Info: cmake config" in text:
            self.enter("编译算子")
            return
        if "FLA_NPU_SOC=" in text:
            self.enter("环境预检", 0.8)
            return
        if "Building wheel for" in text:
            self.enter("环境预检")
            return
        if "Preparing metadata" in text or "Processing ./" in text:
            self.enter("准备构建")

    # ---- 对外接口 ----------------------------------------------------

    def message(self, text: str) -> None:
        """打印一条属于本脚本自己的日志，保持进度条不被打断。"""

        self._clear_line()
        self._write(text + "\n")
        self._draw()

    def echo(self, line: str) -> None:
        """原样输出子进程日志，并据此推进进度。"""

        self._clear_line()
        self._write(line)
        self.feed(line)
        self._draw()

    def tick(self) -> None:
        """子进程暂时没有输出时刷新一次动画。"""

        self._frame += 1
        self._draw()

    def close(self, success: bool) -> None:
        if success:
            self._stage = len(_PROGRESS_STAGES) - 1
            self._fraction = 1.0
        if self._bar_enabled:
            self._draw(force=True)
        self._clear_line()
        if self._bar_enabled or self._announce:
            state = "完成" if success else "失败"
            self._write(
                f"[fla-npu build] 一键编包{state}，用时 "
                f"{_format_elapsed(time.monotonic() - self._started)}\n"
            )


def _progress_enabled(args: argparse.Namespace) -> bool:
    if args.no_progress:
        return False
    if os.getenv("FLA_NPU_BUILD_PROGRESS", "").upper() in {
            "0", "FALSE", "NO", "OFF"}:
        return False
    if os.getenv("TERM", "") == "dumb":
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _run_with_progress(command: list, env: dict, progress: _BuildProgress) -> int:
    """运行子进程，逐行回显日志并驱动进度条。"""

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout = process.stdout
    if stdout is None:  # pragma: no cover - Popen 已要求管道
        raise RuntimeError("failed to capture the build log")

    lines: "queue.Queue[str | None]" = queue.Queue()

    def _pump() -> None:
        try:
            for line in stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=_pump, name="fla-npu-build-log", daemon=True)
    reader.start()
    while True:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            progress.tick()
            continue
        if line is None:
            break
        progress.echo(line)
    reader.join(timeout=1.0)
    return process.wait()


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


def _prepare_abi_free_launcher() -> None:
    """Stage the ABI-free launcher the wheel will carry.

    ``pip wheel`` builds in a temporary copy of the project, so preparing the
    package directory here (before the wheel is built) is what actually decides
    what ships: pure Python plus ``libfla_npu_stable.so``, with no CPython ABI
    and no libtorch C++ ABI.  A pure-ctypes wheel is
    ``FLA_NPU_BUILD_STABLE_ABI=0``.
    """

    package_dir = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu"
    if not package_dir.is_dir():
        return
    if os.getenv("FLA_NPU_BUILD_STABLE_ABI", "TRUE").upper() in {
            "0", "FALSE", "NO", "OFF"}:
        return
    builder = (REPO_ROOT / "torch_custom" / "fla_npu" / "csrc"
               / "build_stable.py")
    if not builder.is_file():
        # 老分支没有 Stable-ABI 源码（csrc/），此时按纯 ctypes wheel 处理，
        # 不要因为缺少构建脚本直接失败。
        print("[fla-npu build] stable-ABI launcher sources not found; "
              "building a pure-ctypes wheel", flush=True)
        return
    target = package_dir / "libfla_npu_stable.so"
    subprocess.run([sys.executable, str(builder), "--no-debug-probe",
                    "--out", str(target)], check=True)
    print(f"[fla-npu build] staged {target.name} ({target.stat().st_size} bytes)",
          flush=True)


def _inject_runtime_pins(wheel_path: Path) -> None:
    """Add Requires-Dist pins to a wheel that carries a compiled launcher.

    pyproject.toml owns ``[project]`` metadata, so ``install_requires`` in
    setup.py is ignored; the pins have to be injected into the produced wheel.

    The Stable-ABI wheel (``libfla_npu_stable.so``) only needs the
    ``aoti_torch_*`` runtime symbols, which exist from 2.7.1 on, so it declares a
    *lower bound*: one wheel then serves every torch/torch_npu above it.
    """

    with zipfile.ZipFile(wheel_path) as archive:
        infos = archive.infolist()
        blobs = {info.filename: archive.read(info.filename) for info in infos}

    has_stable = any(name.endswith("libfla_npu_stable.so") for name in blobs)
    if has_stable:
        pins = [f"torch>={STABLE_ABI_MIN_TORCH}",
                f"torch_npu>={STABLE_ABI_MIN_TORCH}"]
    else:
        return  # pure-ctypes wheel: nothing to declare
    if not pins:
        return
    meta_name = next(name for name in blobs
                     if name.endswith(".dist-info/METADATA"))
    meta = blobs[meta_name].decode("utf-8")
    if any(f"Requires-Dist: {pin}" in meta for pin in pins):
        return
    lines = meta.splitlines()
    insert_at = len(lines)
    for index, line in enumerate(lines):
        if line.startswith("Requires-Dist:"):
            insert_at = index + 1
    lines[insert_at:insert_at] = [f"Requires-Dist: {pin}" for pin in pins]
    blobs[meta_name] = ("\n".join(lines) + "\n").encode("utf-8")

    record_name = next(name for name in blobs if name.endswith(".dist-info/RECORD"))
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(blobs[meta_name]).digest()).rstrip(b"=").decode()
    size = len(blobs[meta_name])
    record = [
        f"{meta_name},sha256={digest},{size}"
        if line.startswith(meta_name + ",") else line
        for line in blobs[record_name].decode("utf-8").splitlines()
    ]
    blobs[record_name] = ("\n".join(record) + "\n").encode("utf-8")

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, blobs[info.filename])
    print(f"[fla-npu build] pinned {', '.join(pins)} into the wheel metadata",
          flush=True)


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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "disable the build progress bar and the per-stage milestone "
            "lines. Also honored via FLA_NPU_BUILD_PROGRESS=0."
        ),
    )
    args = parser.parse_args()

    wheel_dir = _resolve_output_dir(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    progress = _BuildProgress(
        enabled=_progress_enabled(args),
        announce=not args.no_progress,
    )
    progress.message(
        f"[fla-npu build] 一键编包开始（soc="
        f"{os.getenv('FLA_NPU_SOC', '<default>')}, wheel 目录 {wheel_dir}）"
    )
    _prepare_abi_free_launcher()
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
    returncode = _run_with_progress(command, env, progress)
    progress.close(returncode == 0)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)

    # The wheel is tagged for the host platform and the build tag carries the
    # SoC, so resolve the actual file instead of predicting the name.
    wheel_files = sorted(wheel_dir.glob("flash_linear_attention_npu-*.whl"))
    if not wheel_files:
        raise RuntimeError(f"Expected wheel was not produced under {wheel_dir}")
    wheel_path = wheel_files[-1]

    _inject_runtime_pins(wheel_path)

    print(f"[fla-npu build] Wheel: {wheel_path}", flush=True)
    print(f"[fla-npu build] Install command:", flush=True)
    print(_install_command(wheel_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
