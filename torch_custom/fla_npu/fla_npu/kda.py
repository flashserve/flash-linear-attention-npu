# Copyright (c) 2026 Tianjin University, Ltd.
#
# KDA operators are registered by the native custom_aclnn_extension_lib through
# npu_custom.yaml.  This module is intentionally kept as a no-op compatibility
# import for older fla_npu package entrypoints.


def register_kda_ops() -> None:
    return None
