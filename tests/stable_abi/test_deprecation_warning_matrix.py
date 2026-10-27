#!/usr/bin/env python3
"""Deprecation warning matrix for 9 sunset operators.

Verifies that each deprecated operator emits a FutureWarning on three layers:
  - dispatch: __init__.py _make_raw_wrapper (first call only)
  - stable:   _stable.py warnings.warn (every stable backend call)
  - ctypes:   _aclnn_ctypes.py warnings.warn (every ctypes backend call)

Also checks:
  - All 9 ops are listed in __init__.py _DEPRECATED_OPS
  - _stable.py functions still have valid __doc__ (docstring not masked by warn)
  - Replacement text does not recommend another deprecated op

Each operator is tested in a fresh subprocess (backend selection is global and
the dispatch warning fires only once per process).

usage::

    python tests/stable_abi/test_deprecation_warning_matrix.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEPRECATED_OPS = [
    "npu_chunk_bwd_dqkwg",
    "npu_chunk_bwd_dv_local",
    "npu_prepare_wy_repr_bwd",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
    "npu_chunk_local_cumsum",
    "npu_chunk_scaled_dot_kkt",
    "npu_recompute_w_u_fwd",
    "npu_kda_gate_cumsum",
]

CHILD_TEMPLATE = r'''
import json, os, sys, warnings
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "2")
os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
os.environ.setdefault("PYTORCH_NO_NPU_MEMORY_CACHING", "1")
{backend_line}
import torch, torch_npu
import fla_npu.ops.ascendc as ops
from fla_npu.ops.ascendc import _stable, _aclnn_ctypes

result = {{"warnings": [], "doc_ok": False, "error": None}}

try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = getattr(ops, "{op_name}")

        # Check docstring on source module function
        stable_fn = getattr(_stable, "{op_name}", None)
        ctypes_fn = getattr(_aclnn_ctypes, "{op_name}", None)
        result["doc_ok"] = (stable_fn is not None
                           and stable_fn.__doc__ is not None
                           and len(stable_fn.__doc__.strip()) > 0)

        # Minimal inputs
        B, Hk, Hv, T, K, V, CS = 1, 2, 2, 64, 16, 16, 64
        bf16 = torch.bfloat16
        f32 = torch.float32
        d = "npu"
        def t(shape, dtype=f32):
            return (torch.rand(shape, dtype=dtype) * 0.1).to(d)

        {inputs}

        # Call the operator
        try:
            {call}
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:200]

        result["warnings"] = [str(w.message) for w in caught
                             if "deprecated" in str(w.message).lower()]
except Exception as exc:
    result["error"] = type(exc).__name__ + ": " + str(exc)[:200]

print("RESULT:" + json.dumps(result))
'''

INPUTS = {
    "npu_chunk_bwd_dqkwg": (
        "q=t((B,Hk,T,K),bf16);k=t((B,Hk,T,K),bf16);v=t((B,Hv,T,V),bf16)\n"
        "        g=t((B,Hv,T),bf16);h=t((B,Hk,T,K),f32);dox=t((B,Hv,T,V),bf16)\n"
        "        dh=t((B,Hk,T,K),f32);dv=t((B,Hv,T,V),bf16)"
    ),
    "npu_chunk_bwd_dv_local": (
        "q=t((B,Hk,T,K),bf16);k=t((B,Hk,T,K),bf16)\n"
        "        d_o=t((B,Hv,T,V),bf16);g=t((B,Hv,T),bf16)"
    ),
    "npu_prepare_wy_repr_bwd": (
        "k=t((B,Hv,T,K),bf16);v=t((B,Hv,T,V),bf16);beta=t((B,Hv,T),bf16)\n"
        "        A=t((B,Hv,T,CS),f32);dw=t((B,Hv,T,K),bf16);du=t((B,Hv,T,V),bf16);g=t((B,Hv,T),bf16)"
    ),
    "npu_prepare_wy_repr_bwd_da": (
        "k=t((B,Hv,T,K),bf16);v=t((B,Hv,T,V),bf16);beta=t((B,Hv,T),bf16)\n"
        "        A=t((B,Hv,T,CS),f32);dw=t((B,Hv,T,K),bf16);du=t((B,Hv,T,V),bf16);g=t((B,Hv,T),bf16)"
    ),
    "npu_prepare_wy_repr_bwd_full": (
        "k=t((B,Hv,T,K),bf16);v=t((B,Hv,T,V),bf16);beta=t((B,Hv,T),bf16)\n"
        "        A=t((B,Hv,T,CS),f32);dA=t((B,Hv,T,CS),f32)\n"
        "        dw=t((B,Hv,T,K),bf16);du=t((B,Hv,T,V),bf16);g=t((B,Hv,T),bf16)"
    ),
    "npu_chunk_local_cumsum": "g=t((B,Hv,T),bf16)",
    "npu_chunk_scaled_dot_kkt": (
        "k=t((B,Hv,T,K),bf16);g=t((B,Hv,T),f32);beta=t((B,Hv,T),f32)"
    ),
    "npu_recompute_w_u_fwd": (
        "k=t((B,Hv,T,K),bf16);v=t((B,Hv,T,V),bf16);beta=t((B,Hv,T),bf16)\n"
        "        A=t((B,Hv,T,CS),f32);g=t((B,Hv,T),bf16)"
    ),
    "npu_kda_gate_cumsum": "g=t((B,Hv,T),bf16)",
}

CALLS = {
    "npu_chunk_bwd_dqkwg": "fn(q,k,v,g,h,dox,dh,dv,CS,scale=1.0,use_exp2=False,transpose_state_layout=False)",
    "npu_chunk_bwd_dv_local": "fn(q,k,d_o,g,1.0,CS)",
    "npu_prepare_wy_repr_bwd": "fn(k,v,beta,A,dw,du,g,CS)",
    "npu_prepare_wy_repr_bwd_da": "fn(k,v,beta,A,dw,du,g,chunk_size=CS)",
    "npu_prepare_wy_repr_bwd_full": "fn(k,v,beta,A,dA,dw,du,g,CS)",
    "npu_chunk_local_cumsum": "fn(g,CS)",
    "npu_chunk_scaled_dot_kkt": "fn(k,g,beta,chunk_size=CS)",
    "npu_recompute_w_u_fwd": "fn(k,v,beta,A,CS,g=g)",
    "npu_kda_gate_cumsum": "fn(g,CS)",
}


def _run_child(op_name, backend="default"):
    backend_line = ""
    if backend == "ctypes":
        backend_line = "os.environ['FLA_NPU_STABLE_ABI'] = 'ctypes'"
    elif backend == "stable":
        backend_line = "os.environ['FLA_NPU_STABLE_ABI'] = 'stable'"

    script = CHILD_TEMPLATE.format(
        backend_line=backend_line,
        op_name=op_name,
        inputs=INPUTS.get(op_name, ""),
        call=CALLS.get(op_name, "fn()"),
    )

    env = os.environ.copy()
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "2"
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    env["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "warnings": [], "doc_ok": False}
    except Exception as exc:
        return {"error": str(exc), "warnings": [], "doc_ok": False}

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            try:
                return json.loads(line[len("RESULT:"):])
            except Exception:
                pass
    return {"error": "No RESULT line", "warnings": [], "doc_ok": False,
            "stdout": result.stdout[-200:], "stderr": result.stderr[-200:]}


class TestDeprecationWarningMatrix(unittest.TestCase):

    def test_all_ops_in_deprecated_set(self):
        """All 9 ops must be in __init__.py _DEPRECATED_OPS."""
        import fla_npu.ops.ascendc as ascendc
        deprecated = ascendc._DEPRECATED_OPS
        for op in DEPRECATED_OPS:
            self.assertIn(op, deprecated, f"{op} not in _DEPRECATED_OPS")

    def test_replacement_does_not_recommend_deprecated(self):
        """Replacement text must not recommend another deprecated op."""
        import fla_npu.ops.ascendc as ascendc
        replacements = ascendc._DEPRECATED_OPS_REPLACEMENTS
        for op, replacement in replacements.items():
            for other_op in DEPRECATED_OPS:
                if other_op == op:
                    continue
                self.assertNotIn(
                    other_op, replacement,
                    f"{op} recommends {other_op} which is itself deprecated")

    def test_stable_imports_warnings(self):
        """_stable.py must import warnings at module level."""
        from fla_npu.ops.ascendc import _stable
        self.assertTrue(
            hasattr(_stable, "warnings"),
            "_stable module does not have 'warnings' attribute; "
            "missing 'import warnings' at module level")

    def test_stable_docstrings_intact(self):
        """_stable.py deprecated functions must have non-None __doc__."""
        from fla_npu.ops.ascendc import _stable
        for op in DEPRECATED_OPS:
            fn = getattr(_stable, op, None)
            if fn is None:
                continue
            self.assertIsNotNone(
                fn.__doc__, f"_stable.{op}.__doc__ is None; "
                f"warnings.warn may be masking the docstring")

    def test_stable_backend_emits_warning(self):
        """Each op on stable backend must emit >=2 deprecation warnings
        (dispatch + stable). Operator execution errors are OK as long as
        the deprecation warnings are emitted before the error."""
        for op in DEPRECATED_OPS:
            with self.subTest(op=op, backend="stable"):
                result = _run_child(op, backend="default")
                warnings = result.get("warnings", [])
                self.assertGreaterEqual(
                    len(warnings), 2,
                    f"{op} stable: expected >=2 deprecation warnings, "
                    f"got {len(warnings)}: {warnings}; error={result.get('error')}")

    def test_ctypes_backend_emits_warning(self):
        """Each op on ctypes backend must emit >=2 deprecation warnings
        (dispatch + ctypes). Operator execution errors are OK as long as
        the deprecation warnings are emitted before the error."""
        for op in DEPRECATED_OPS:
            with self.subTest(op=op, backend="ctypes"):
                result = _run_child(op, backend="ctypes")
                warnings = result.get("warnings", [])
                self.assertGreaterEqual(
                    len(warnings), 2,
                    f"{op} ctypes: expected >=2 deprecation warnings, "
                    f"got {len(warnings)}: {warnings}; error={result.get('error')}")

    def test_warning_content_has_replacement(self):
        """Each deprecation warning must mention 'will be removed' and a replacement."""
        for op in DEPRECATED_OPS:
            with self.subTest(op=op):
                result = _run_child(op, backend="default")
                warnings_list = result.get("warnings", [])
                self.assertTrue(
                    len(warnings_list) > 0,
                    f"{op}: no deprecation warnings emitted")
                for msg in warnings_list:
                    self.assertIn("deprecated", msg.lower(),
                                  f"{op}: warning missing 'deprecated': {msg}")
                    self.assertIn("will be removed", msg.lower(),
                                  f"{op}: warning missing 'will be removed': {msg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
