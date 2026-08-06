"""Opt-in Python startup hook for model-transparent KDA forward replacement."""

import os


if os.environ.get("FLA_NPU_ENABLE_TRITON_ASCEND_KDA_ADAPTER") == "1":
    from fla_npu.adapters import install_triton_ascend_kda_adapter

    install_triton_ascend_kda_adapter()
