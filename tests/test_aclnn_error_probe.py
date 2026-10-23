"""Raw negative ABI helper must never become an execution backend."""
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
