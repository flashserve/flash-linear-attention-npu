"""Customer-facing compatibility tests: thin launcher vs ctypes.

Valid inputs must be bit-identical and share the same public signature and
mutation contract. Invalid-input error kinds are reported, not asserted equal,
because the thin fast path intentionally skips the full Python validation.
"""
from __future__ import annotations

import inspect
import unittest

import pytest
import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def _have_thin() -> bool:
    try:
        import fla_npu._C_thin  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _npu_available(), reason="NPU is required"),
    pytest.mark.skipif(not _have_thin(), reason="thin launcher extension not built"),
]


def make_case(batch, tokens_per_request, contiguous, heads=(8, 16), dim=128):
    nk, nv = heads
    lengths = [tokens_per_request] * batch
    total = sum(lengths)
    gap = 0 if contiguous else 16384
    offset = 0 if contiguous else 12288
    inner = nv * dim * dim
    block_stride = inner + gap
    state_blocks = total + 1

    def make_state():
        backing = torch.empty(
            state_blocks * block_stride * 4, dtype=torch.int8, device="npu")
        typed = backing.view(torch.float32)
        state = torch.as_strided(
            typed,
            size=(state_blocks, nv, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset,
        )
        state.zero_()
        return state, backing

    def norm(t):
        return torch.nn.functional.normalize(t, p=2, dim=-1)

    query = norm(torch.randn(total, nk, dim, device="npu")).to(torch.bfloat16)
    key = norm(torch.randn(total, nk, dim, device="npu")).to(torch.bfloat16)
    value = torch.randn(total, nv, dim, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(total, nv, dtype=torch.bfloat16, device="npu")
    g = torch.rand(total, nv, dtype=torch.float32, device="npu")
    actual_seq_lengths = torch.tensor(
        [0, *lengths], dtype=torch.int32, device="npu")
    ssm_state_indices = torch.arange(total, dtype=torch.int32, device="npu")
    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "g": g,
        "scale": dim ** -0.5,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "make_state": make_state,
    }


def call_ctypes(op, inputs, state):
    return op(
        inputs["query"], inputs["key"], inputs["value"], state,
        beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None)


def call_thin(inputs, state):
    import fla_npu._C_thin as thin_ext

    stream = int(torch.npu.current_stream().npu_stream)
    return thin_ext.npu_recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state,
        inputs["beta"], inputs["scale"], inputs["actual_seq_lengths"],
        inputs["ssm_state_indices"], None, inputs["g"], None, stream)


class TestCustomerCompat(unittest.TestCase):
    def test_public_signature_matches_ctypes(self):
        from fla_npu.ops.ascendc import _aclnn_ctypes, _thin

        ctypes_sig = inspect.signature(
            _aclnn_ctypes.npu_recurrent_gated_delta_rule)
        thin_sig = inspect.signature(_thin.npu_recurrent_gated_delta_rule)
        # Same parameter names and defaults; stream is internal to _thin.
        self.assertEqual(list(ctypes_sig.parameters), list(thin_sig.parameters))
        for name, p in ctypes_sig.parameters.items():
            self.assertEqual(p.default, thin_sig.parameters[name].default)

    def test_parity_layouts_and_tokens(self):
        from fla_npu.ops.ascendc import _aclnn_ctypes

        ctypes_op = _aclnn_ctypes.npu_recurrent_gated_delta_rule
        for batch, tpr, contiguous in (
            (16, 1, True),
            (64, 1, False),
            (16, 4, False),
        ):
            with self.subTest(batch=batch, tpr=tpr, contiguous=contiguous):
                torch.manual_seed(1024)
                inputs = make_case(batch, tpr, contiguous)
                s1, _ = inputs["make_state"]()
                s2, _ = inputs["make_state"]()
                o1 = call_ctypes(ctypes_op, inputs, s1)
                o2 = call_thin(inputs, s2)
                torch.npu.synchronize()
                self.assertEqual(
                    float((o1.float() - o2.float()).abs().max()), 0.0)
                self.assertEqual(
                    float((s1.float() - s2.float()).abs().max()), 0.0)

    def test_mutation_version_and_grad_guard(self):
        from fla_npu.ops.ascendc import _thin, _wrap_mutable_direct_op

        inputs = make_case(8, 1, False)
        wrapped = _wrap_mutable_direct_op(
            "npu_recurrent_gated_delta_rule",
            _thin.npu_recurrent_gated_delta_rule)
        state, backing = inputs["make_state"]()
        before = state._version
        wrapped(
            inputs["query"], inputs["key"], inputs["value"], state,
            beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"])
        torch.npu.synchronize()
        self.assertGreater(state._version, before)

        inputs2 = make_case(8, 1, False)
        state2, _ = inputs2["make_state"]()
        state2.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            wrapped(
                inputs2["query"], inputs2["key"], inputs2["value"], state2,
                beta=inputs2["beta"], g=inputs2["g"], scale=inputs2["scale"],
                actual_seq_lengths=inputs2["actual_seq_lengths"],
                ssm_state_indices=inputs2["ssm_state_indices"])

    def test_invalid_input_error_kind_report(self):
        """Report error kinds for both paths (kinds may differ by design)."""
        from fla_npu.ops.ascendc import _aclnn_ctypes, _thin

        inputs = make_case(4, 1, False)
        state, _ = inputs["make_state"]()

        def kind(fn):
            try:
                fn()
                return "no-error"
            except Exception as exc:  # noqa: BLE001
                return type(exc).__name__

        bad_query = inputs["query"].to(torch.float32)
        ctypes_err = kind(lambda: _aclnn_ctypes.npu_recurrent_gated_delta_rule(
            bad_query, inputs["key"], inputs["value"], state,
            beta=inputs["beta"], g=inputs["g"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"]))
        thin_err = kind(lambda: _thin.npu_recurrent_gated_delta_rule(
            bad_query, inputs["key"], inputs["value"], state,
            beta=inputs["beta"], g=inputs["g"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"]))
        missing_g_err = kind(lambda: _thin.npu_recurrent_gated_delta_rule(
            inputs["query"], inputs["key"], inputs["value"], state,
            beta=inputs["beta"], g=None, gk=None,
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"]))
        print("error kinds: ctypes_bad_dtype=%s thin_bad_dtype=%s "
              "thin_missing_g=%s" % (ctypes_err, thin_err, missing_g_err))

    def test_determinism_and_stream_order(self):
        from fla_npu.ops.ascendc import _aclnn_ctypes

        ctypes_op = _aclnn_ctypes.npu_recurrent_gated_delta_rule
        inputs = make_case(32, 1, False)
        outs = []
        for _ in range(3):
            s, _ = inputs["make_state"]()
            o = call_thin(inputs, s)
            outs.append(o)
        torch.npu.synchronize()
        for o in outs[1:]:
            self.assertEqual(
                float((outs[0].float() - o.float()).abs().max()), 0.0)

        # Ordering: thin op must complete before a following dependent kernel.
        s, _ = inputs["make_state"]()
        o = call_thin(inputs, s)
        consumed = (o.float() * 2.0).sum()
        torch.npu.synchronize()
        self.assertTrue(bool(torch.isfinite(consumed)))


if __name__ == "__main__":
    unittest.main()
