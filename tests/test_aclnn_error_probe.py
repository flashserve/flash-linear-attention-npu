"""Raw negative ABI helper must never become an execution backend."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parent / "atk/common/aclnn_error_probe.py"
spec = importlib.util.spec_from_file_location("negative_abi_probe", PATH)
PROBE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROBE)


class RawAbiProbeTest(unittest.TestCase):
    def invoke(self, result, *, workspace=0, executor=0):
        api, context = mock.Mock(), mock.Mock()
        def call(*args):
            args[-2]._obj.value = workspace
            args[-1]._obj.value = executor
            return result
        function = mock.Mock(side_effect=call)
        api.symbol.return_value = function
        api._recent_error_message.return_value = "EZ0004 invalid output"
        with mock.patch.object(PROBE, "runtime", return_value=api), \
                mock.patch.object(PROBE, "_CallContext", return_value=context):
            try:
                PROBE.reject_workspace("aclnnTest", lambda ctx: [None])
            finally:
                api.symbol.assert_called_once_with("aclnnTestGetWorkspaceSize")
                context.destroy.assert_called_once()
                self.assertIsNone(function.call_args.args[0])

    def test_rejection_keeps_code_and_message_and_cleans_descriptors(self):
        with self.assertRaisesRegex(RuntimeError, "aclnnStatus=161001: EZ0004"):
            self.invoke(161001)

    def test_success_never_launches(self):
        with self.assertRaisesRegex(AssertionError, "no kernel was launched"):
            self.invoke(0)

    def test_rejected_executor_is_a_failure(self):
        with self.assertRaisesRegex(AssertionError, "executor or workspace"):
            self.invoke(161001, executor=42)

    def test_rejected_workspace_is_a_failure(self):
        with self.assertRaisesRegex(AssertionError, "executor or workspace"):
            self.invoke(161001, workspace=64)

    def test_build_args_failure_still_releases_descriptors(self):
        context = mock.Mock()
        with mock.patch.object(PROBE, "runtime"), \
                mock.patch.object(PROBE, "_CallContext", return_value=context):
            with self.assertRaisesRegex(ValueError, "descriptor"):
                PROBE.reject_workspace("aclnnTest", mock.Mock(side_effect=ValueError("descriptor")))
        context.destroy.assert_called_once()


class PreprocessNegativeCpuContractTest(unittest.TestCase):
    def test_all_frozen_negative_cases_accept_actual_flat_atk_attributes(self):
        folder = PATH.parent.parent / "pre_process_bwd_kernel_merged"
        source = folder / "executor_pre_process_bwd_kernel_merged.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"_placeholder_output", "_negative_expected_code", "run_cpu"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in names]
        self.assertEqual({node.name for node in functions}, names)
        torch = mock.Mock()
        torch.float32 = "float32"
        namespace = {"torch": torch, "InputDataset": object,
                     "_ACLNN_CODES": {"ACLNN_ERR_PARAM_INVALID": 161001,
                                      "ACLNN_ERR_PARAM_NULLPTR": 161002},
                     "OP_NAME": "pre_process_bwd_kernel_merged",
                     "build_inputs": mock.Mock(side_effect=AssertionError("negative CPU must not build invalid tensors")),
                     "_call_op": mock.Mock(side_effect=AssertionError("negative CPU must not call reference"))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        cases = json.loads((folder / "atk_pre_process_bwd_kernel_merged.json").read_text())
        negatives = [case for case in cases if case.get("expected_error_msg")]
        self.assertEqual(len(negatives), 13)
        for case in negatives:
            with self.subTest(case=case["id"]):
                # This is the scalar channel that survives ATK before_call.
                spec = {item["name"]: item["range_values"] for item in case["inputs"]
                        if item.get("type") == "attr"}
                self.assertNotIn("shape", spec)
                torch.zeros.reset_mock()
                output = namespace["run_cpu"](spec, object())
                torch.zeros.assert_called_once_with(
                    (int(spec["Hv"]), int(spec["K"]), int(spec["V"]) + int(spec["K"])),
                    dtype="float32")
                self.assertEqual(output, (torch.zeros.return_value,))
        namespace["build_inputs"].assert_not_called()
        namespace["_call_op"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
