"""Small independent-oracle checks for three Stable public entries.

FastGELU forward reuses the existing FP32 [-3, 3] case, CPU formula and 1e-3
comparison in torch_custom/fla_npu/test/test_npu_fast_gelu_custom.py. No general
backward error tolerance is established there, so backward is deliberately
limited to x=0: the CPU derivative is exactly 1/2 and binary-exact FP32 upstream
gradients require no tolerance. This is an origin-point contract check, not a
claim of general backward numerical coverage.

KDA gate reuses _quick_case, _gate_cumsum_reference and the 2e-3 standalone-gate
comparison from diagnose_npu_chunk_kda_cu_seqlens.py. The two original quick-case
seeds/layouts/chunk sizes are unchanged. Neither the oracle nor its input domain
or tolerance is redefined here.

Run in the installed NPU environment, optionally through
scripts/verify_stable_only_results.py record. Missing NPU support fails the
standalone command rather than producing a passing run with no operator calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _source_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StableElementwiseContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the packaged OPP before torch_npu initializes CANN.
        from fla_npu.ops import ascendc
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("An available Ascend NPU is required for these contract tests")
        cls.torch, cls.ascendc = torch, ascendc
        cls.device = torch.device("npu:" + os.environ.get("TEST_DEVICE_ID", "0"))
        torch.npu.set_device(cls.device)
        torch.npu.config.allow_internal_format = False
        cls.gelu_reference = _source_module(
            "stable_elementwise_gelu_reference",
            "torch_custom/fla_npu/test/test_npu_fast_gelu_custom.py")
        cls.gate_reference = _source_module(
            "stable_elementwise_gate_reference",
            "torch_custom/fla_npu/test/diagnose_npu_chunk_kda_cu_seqlens.py")

    def test_fast_gelu_forward_existing_fp32_case(self):
        torch = self.torch
        values = torch.linspace(-3, 3, steps=3 * 16 * 32, dtype=torch.float32).reshape(3, 16, 32)
        expected = self.gelu_reference.fast_gelu_ref(values)
        actual = self.ascendc.npu_fast_gelu_custom(values.to(self.device))
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.gelu_reference.TestFastGelu.assertTensorClose(self, actual, expected)

    def test_fast_gelu_backward_exact_origin_derivative(self):
        torch = self.torch
        values = torch.zeros((3, 16, 32), dtype=torch.float32, requires_grad=True)
        upstream = ((torch.arange(values.numel()) % 7) - 3).to(torch.float32).reshape(values.shape)
        self.gelu_reference.fast_gelu_ref(values).backward(upstream)
        expected = values.grad.detach()
        # Independent analytic check of the CPU autograd oracle at the origin.
        self.assertTrue(torch.equal(expected, upstream * 0.5))
        actual = self.ascendc.npu_fast_gelu_custom_backward(
            upstream.to(self.device), values.detach().to(self.device))
        self.assertEqual(actual.dtype, upstream.dtype)
        self.assertEqual(actual.shape, values.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.equal(actual.detach().cpu(), expected),
                        "FastGELU backward at x=0 must return upstream/2 exactly")

    def test_kda_gate_cumsum_existing_quick_cases(self):
        torch = self.torch
        reference = self.gate_reference
        for chunk_size, seed, rank3 in ((64, 20260801, True), (128, 20260802, False)):
            with self.subTest(chunk_size=chunk_size, seed=seed, rank3=rank3):
                case = reference._quick_case(chunk_size, seed)
                gate = reference._to_head_major(case["g"], rank3)
                cu = tuple(int(value) for value in case["cu_seqlens"].tolist())
                expected = reference._gate_cumsum_reference(
                    gate, case["A_log"], case["dt_bias"], cu,
                    chunk_size, case["safe_gate"], case["lower_bound"])
                actual = self.ascendc.npu_kda_gate_cumsum(
                    gate.to(self.device), chunk_size,
                    A_log=case["A_log"].to(self.device),
                    dt_bias=case["dt_bias"].to(self.device), cu_seqlens=cu,
                    use_gate_in_kernel=True, safe_gate=case["safe_gate"],
                    lower_bound=case["lower_bound"])
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.shape, gate.shape)
                self.assertTrue(torch.isfinite(actual).all().item())
                reference._assert_close("standalone gate cumsum", actual, expected,
                                        rtol=2e-3, atol=2e-3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    args = parser.parse_args()
    os.environ["TEST_DEVICE_ID"] = str(args.device)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StableElementwiseContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
