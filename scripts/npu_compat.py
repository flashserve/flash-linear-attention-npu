"""Canonical runtime/version compatibility tables.

Single source of truth shared by:
- scripts/check_npu_env.py (build/runtime preflight)
- the PyPI import guard embedded into tiered wheels at build time
  (generated torch_custom/fla_npu/fla_npu/_compat.py)

Keep the public promise aligned with the README: CANN >= 8.5.2, torch >= 2.6,
torch_npu >= TORCH_NPU_GDN_FIX_MINIMUMS[torch], triton-ascend >= 3.2.1 on CANN
9.x (9.0.0+).
"""

from __future__ import annotations

# CANN minimum per product tier: 950 (a5) requires CANN >= 9.0.0, while
# 910b/910_93 (a2/a3) follow the README promise of >= 8.5.2.
MIN_CANN_BY_TIER = {"a2": "8.5.2", "a3": "8.5.2", "a5": "9.0.0"}
MIN_CANN = MIN_CANN_BY_TIER["a2"]
MIN_TORCH = "2.6.0"
MIN_TRITON_ASCEND = "3.2.0"
MIN_TRITON_ASCEND_A5 = "3.2.1"
# CANN 9.x (9.0.0+) requires triton-ascend >= 3.2.1: 3.2.0 fails to JIT-compile
# triton/backends/ascend/npu_utils.cpp on CANN 9.1.0.
MIN_TRITON_ASCEND_CANN9 = "3.2.1"
# Minimum torch_npu post/fix releases required per torch version (GDN fixes).
TORCH_NPU_GDN_FIX_MINIMUMS = {
    "2.7.1": "2.7.1.post5",
    "2.8.0": "2.8.0.post5",
    "2.9.0": "2.9.0.post3",
    "2.10.0": "2.10.0.post2",
    "2.11.0": "2.11.0rc3",
    "2.12.0": "2.12.0rc1",
}
MIN_TORCH_NPU_FUTURE_FIX_FAMILY = "2.13.0"
# (CANN public version prefix, torch version) combos validated on real NPUs
# before the v1 PyPI release. Combos inside the promised range but not listed
# here trigger a runtime warning (supported-but-not-yet-validated).
VALIDATED_COMBOS = [
    ("9.1.0", "2.7.1"),
    ("9.1.0", "2.8.0"),
    ("9.1.0", "2.9.0"),
    ("9.1.0", "2.10.0"),
    ("9.1.0", "2.11.0"),
    ("9.1.0", "2.12.0"),
]
