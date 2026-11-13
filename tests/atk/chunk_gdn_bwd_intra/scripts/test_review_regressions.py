"""PR 1176 的参数拒绝与绝对 head 映射回归，无需 NPU。"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from smoke_operator import main, parse_args, rejection_matches


def rejection_message(hv=17, hk=1, arch=2201, limit=16, code=161002):
    # 对齐 aclnn 参数校验层回退后的 OP_LOGE(errno, fmt) 直报格式。
    return (
        f"fla_npu(stable): aclnnChunkGdnBwdIntraGetWorkspaceSize failed: {code}. "
        f"Error message: Parameter hv/hk of aclnnChunkGdnBwdIntra has incorrect value "
        f"HV={hv}, HK={hk} (G={hv // hk if hk else 0}). "
        f"It should be an integer ratio in [1, {limit}] on the current SoC arch."
    )


class RejectionTests(unittest.TestCase):
    def assert_parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                parse_args(argv)
        self.assertEqual(caught.exception.code, 2)

    def test_a2_group17_reaches_operator_configuration(self):
        args = parse_args(["--group", "17", "--expect-reject", "--expected-arch", "2201"])
        self.assertEqual((args.hk, args.hv), (1, 17))

    def test_a5_group5_reaches_operator_configuration(self):
        args = parse_args(["--group", "5", "--expect-reject", "--expected-arch", "3510"])
        self.assertEqual((args.hk, args.hv), (1, 5))
        self.assertTrue(rejection_matches(rejection_message(5, 1, 3510, 4), 5, 1, 3510))

    def test_nondivisible_heads_are_preserved(self):
        args = parse_args(["--hk", "2", "--hv", "5", "--expect-reject", "--expected-arch", "2201"])
        self.assertEqual((args.hk, args.hv), (2, 5))
        self.assertTrue(rejection_matches(rejection_message(5, 2), 5, 2, 2201))

    def test_positive_mode_retains_constraints(self):
        self.assert_parse_error(["--group", "17"])
        self.assert_parse_error(["--hk", "2", "--hv", "5"])
        self.assertEqual(parse_args(["--group", "16"]).hv, 16)

    def test_negative_mode_requires_independent_invalid_expectation(self):
        self.assert_parse_error(["--group", "17", "--expect-reject"])
        self.assert_parse_error(["--group", "16", "--expect-reject", "--expected-arch", "2201"])
        self.assert_parse_error(["--hk", "0", "--expect-reject", "--expected-arch", "2201"])

    def test_detailed_parameter_rejection_matches(self):
        self.assertTrue(rejection_matches(rejection_message(), 17, 1, 2201))

    def test_wrong_values_architecture_range_and_error_are_rejected(self):
        messages = (
            rejection_message(hv=18),
            rejection_message(hk=2),
            rejection_message(code=561000),
            rejection_message(arch=3510, limit=4),
            rejection_message().replace("incorrect value", "wrong value"),
            rejection_message().replace("current SoC arch", "this SoC"),
            "device synchronize failed: HV=17, HK=1",
            "unsupported hv/hk",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertFalse(rejection_matches(message, 17, 1, 2201))


class OperatorCallTests(unittest.TestCase):
    def run_negative(self, exception, expected_exit):
        inputs = {name: object() for name in ("q", "k", "v", "g", "beta", "A", "d_o", "scale", "chunk_size", "use_exp2")}
        build = Mock(return_value=inputs)
        operator = Mock(side_effect=exception)
        torch = types.ModuleType("torch")
        torch.device = Mock(return_value="npu")
        torch.npu = Mock()
        executor = types.ModuleType("executor_chunk_gdn_bwd_intra")
        executor.build_inputs, executor._cpu_ref = build, Mock()
        ops = types.ModuleType("fla_npu.ops")
        ops.ascendc = types.SimpleNamespace(chunk_gdn_bwd_intra=operator)
        modules = {"torch": torch, "executor_chunk_gdn_bwd_intra": executor, "fla_npu.ops": ops}
        argv = ["smoke_operator.py", "--group", "17", "--expect-reject", "--expected-arch", "2201"]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), patch.dict(os.environ), patch.object(sys, "path", sys.path.copy()):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, expected_exit)
            self.assertEqual(os.environ["FLA_NPU_STABLE_LAUNCH"], "inline")
        self.assertEqual(build.call_args.args[0]["HV"], 17)
        operator.assert_called_once()
        torch.npu.synchronize.assert_not_called()

    def test_group17_calls_operator_and_accepts_only_parameter_report(self):
        self.run_negative(RuntimeError(rejection_message()), 0)

    def test_unrelated_operator_error_fails(self):
        self.run_negative(RuntimeError("device error: HV=17, HK=1"), 3)

    def test_operator_returning_normally_fails(self):
        self.run_negative(None, 2)


class KernelDispatchTests(unittest.TestCase):
    def test_a5_shared_runtime_key_is_compile_time_isolated(self):
        root = Path(__file__).resolve().parents[4]
        kernel = root / "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gdn_bwd_intra/op_kernel"
        source = (kernel / "chunk_gdn_bwd_intra.cpp").read_text(encoding="utf-8")
        table = (kernel / "chunk_gdn_bwd_intra_tiling_key.h").read_text(encoding="utf-8")
        self.assertIn("CHUNK_GDN_BWD_INTRA_TPL_MAIN(STRATEGY, 0)", table)
        dispatch = source[source.index("__aicore__ inline void DispatchKernel("):]
        a5 = dispatch.split("#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)", 1)[1].split("#else", 1)[0]
        self.assertNotIn("static_assert(G_KEY != 0", a5)
        guard = a5.index("if constexpr (G_KEY != CHUNK_GDN_BWD_INTRA_G_RUNTIME)")
        self.assertLess(guard, a5.index("ChunkGdnBwdIntraCube<MainT, G_KEY>"))
        self.assertLess(guard, a5.index("ChunkGdnBwdIntraVector<MainT, GateT, BetaT, G_KEY>"))


class HeadMappingTests(unittest.TestCase):
    def test_runtime_group_unaligned_slice(self):
        group, begin, r = 5, 4, 1
        self.assertEqual(max(0, (begin + r) // group * group - begin), 1)
        self.assertNotEqual(r - r % group, 1)

    def test_leader_formula_matches_first_head_for_all_supported_groups(self):
        for group in range(1, 17):
            cg = 3 if group == 3 else 4
            for hk_count in range(1, 65):
                hv_count = hk_count * group
                for begin in range(0, hv_count, cg):
                    valid = min(cg, hv_count - begin)
                    heads = [(begin + r) // group for r in range(valid)]
                    for r, head in enumerate(heads):
                        leader = max(0, (begin + r) // group * group - begin)
                        self.assertEqual(leader, heads.index(head))
                    count = (begin + valid - 1) // group - begin // group + 1
                    self.assertEqual(count, len(set(heads)))


if __name__ == "__main__":
    unittest.main()
