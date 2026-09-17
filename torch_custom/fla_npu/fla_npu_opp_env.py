"""解释器启动时把 wheel 内嵌 OPP 写入 ASCEND_CUSTOM_OPP_PATH（issue #429）。

通过随 wheel 安装的 fla_npu_opp_env.pth 在 site 初始化阶段执行，早于任何用户代码
（包括 import torch_npu），使 CANN 初始化前环境变量已就绪，避免“CANN 先初始化、
fla_npu 后设置环境变量来不及”的问题。设置 FLA_NPU_DISABLE_PTH=1/TRUE/YES/ON 可跳过。
"""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PACKAGE_DIR / "fla_npu" / "opp" / "vendors" / "fla_npu_transformer"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").upper() in {"1", "TRUE", "YES", "ON"}


def _prepend(name: str, value) -> None:
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    value = str(value)
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def _setup() -> None:
    if _env_flag("FLA_NPU_DISABLE_PTH"):
        return
    # 只处理 wheel 内嵌 OPP；skeleton（未 overlay）时静默跳过。
    if not _VENDOR_DIR.is_dir():
        return
    _prepend("ASCEND_CUSTOM_OPP_PATH", _VENDOR_DIR / "op_api" / "lib")
    _prepend("ASCEND_CUSTOM_OPP_PATH", _VENDOR_DIR)


_setup()
