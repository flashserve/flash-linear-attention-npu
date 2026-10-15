"""解释器启动时把 wheel 内嵌 OPP（含叠加层 overlay vendor）写入 ASCEND_CUSTOM_OPP_PATH（issue #429）。

通过随 wheel 安装的 fla_npu_opp_env.pth 在 site 初始化阶段执行，早于任何用户代码
（包括 import torch_npu），使 CANN 初始化前环境变量已就绪，避免“CANN 先初始化、
fla_npu 后设置环境变量来不及”的问题。设置 FLA_NPU_DISABLE_PTH=1/TRUE/YES/ON 可跳过。

叠加层（overlay）vendor 排在基础 vendor 之前：kernel/tiling 的查找按这个顺序进行，
因此 overlay 里的算子覆盖基础包，overlay 未提供的算子自动回落到基础包。
可用 FLA_NPU_OVERLAY_VENDOR 指定（os.pathsep 分隔，靠前者优先）。
"""

import os
import re
import warnings
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDORS_DIR = _PACKAGE_DIR / "fla_npu" / "opp" / "vendors"
_BASE_VENDOR = "fla_npu_transformer"
_DEFAULT_DELTA_VENDOR = "fla_npu_overlay_transformer"
_OVERLAY_VENDOR_ENV = "FLA_NPU_OVERLAY_VENDOR"
_OVERLAY_VENDOR_PREFIX = "fla_npu_overlay_"
_OVERLAY_VENDOR_NAME_RE = re.compile(
    rf"{re.escape(_OVERLAY_VENDOR_PREFIX)}[A-Za-z0-9][A-Za-z0-9_.-]*\Z"
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").upper() in {"1", "TRUE", "YES", "ON"}


def _prepend(name: str, value) -> None:
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    value = str(value)
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def _overlay_names() -> list[str]:
    """Overlay vendor names, highest priority first.

    **Must stay in sync with ``fla_npu.overlay_vendor_names()``.**  This hook runs
    during site initialisation, i.e. *before* any user code and before CANN
    initialises; if it only honoured the env var or the default name, a
    custom-named overlay would be missing from ``ASCEND_CUSTOM_OPP_PATH`` exactly
    when CANN discovers its kernels/tiling, and importing ``fla_npu`` later
    (which does derive the full order) would be too late.
    """

    raw = os.environ.get(_OVERLAY_VENDOR_ENV)
    if raw is not None:
        names: list[str] = []
        for candidate in raw.split(os.pathsep):
            name = candidate.strip()
            if not name:
                continue
            if not _OVERLAY_VENDOR_NAME_RE.fullmatch(name):
                warnings.warn(
                    f"fla_npu: ignoring invalid {_OVERLAY_VENDOR_ENV} entry {name!r}; "
                    f"overlay names must match {_OVERLAY_VENDOR_PREFIX}<name>.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if name not in names:
                names.append(name)
        return names

    names: list[str] = []

    def _add(candidate: str) -> None:
        name = candidate.strip()
        if (not name or name == _BASE_VENDOR or name in names
                or not _OVERLAY_VENDOR_NAME_RE.fullmatch(name)):
            return
        if (_VENDORS_DIR / name).is_dir():
            names.append(name)

    try:
        for line in (_VENDORS_DIR / "config.ini").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("load_priority="):
                for entry in line.split("=", 1)[1].split(","):
                    _add(entry)
                break
    except OSError:
        pass
    try:
        for child in sorted(_VENDORS_DIR.iterdir()):
            if child.is_dir():
                _add(child.name)
    except OSError:
        pass
    return names


def _vendor_dirs() -> list[Path]:
    """Overlay vendors first (highest priority), base vendor last."""

    names = _overlay_names() or [_DEFAULT_DELTA_VENDOR]
    dirs: list[Path] = []
    vendors_root = _VENDORS_DIR.resolve()
    for name in [*names, _BASE_VENDOR]:
        vendor_dir = _VENDORS_DIR / name
        if not vendor_dir.is_dir():
            continue
        resolved = vendor_dir.resolve()
        try:
            resolved.relative_to(vendors_root)
        except ValueError:
            continue
        if resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _setup() -> None:
    if _env_flag("FLA_NPU_DISABLE_PTH"):
        return
    vendor_dirs = _vendor_dirs()
    # 只处理 wheel 内嵌 OPP；skeleton（未 overlay）时静默跳过。
    if not vendor_dirs:
        return
    # 反向 prepend，最终顺序为：overlay、overlay/op_api/lib、base、base/op_api/lib…
    for vendor_dir in reversed(vendor_dirs):
        _prepend("ASCEND_CUSTOM_OPP_PATH", vendor_dir / "op_api" / "lib")
        _prepend("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    os.environ.setdefault("FLA_NPU_OPP_PATH", str(_VENDORS_DIR.parent))


_setup()
