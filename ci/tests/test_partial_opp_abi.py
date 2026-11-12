import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "test_ci_op_abi", ROOT / "torch_custom/fla_npu/fla_npu/_op_abi.py"
)
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)


def vendor(path, name="Demo", parameter="const aclTensor *input"):
    include = path / abi.INCLUDE
    include.mkdir(parents=True)
    (path / abi.LIBRARY).parent.mkdir(parents=True)
    (path / abi.LIBRARY).write_bytes(b"fixture library; exports supplied below")
    (include / "aclnn_demo.h").write_text(
        f"aclnnStatus aclnn{name}GetWorkspaceSize({parameter}, "
        "uint64_t *workspaceSize, aclOpExecutor **executor);\n"
        f"aclnnStatus aclnn{name}(void *workspace, uint64_t workspaceSize, "
        "aclOpExecutor *executor, aclrtStream stream);\n",
        encoding="utf-8",
    )
    with mock.patch.object(abi, "exported_symbols", return_value={
        f"aclnn{name}", f"aclnn{name}GetWorkspaceSize"
    }):
        abi.write_manifest(path)
    return path


class PartialOppAbiTests(unittest.TestCase):
    def check_vendors(self, *, base_name="Demo", overlay_parameter="const aclTensor *input",
                      legacy_base=False, corrupt_base=False, corrupt_overlay=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "fla_npu"
            package.mkdir()
            # Neither the runtime nor its native libraries may be initialized.
            (package / "__init__.py").write_text("raise RuntimeError('runtime imported')\n")
            base = vendor(package / "opp/vendors/fla_npu_transformer", name=base_name)
            overlay = vendor(root / "overlay", parameter=overlay_parameter)
            if legacy_base:
                (base / abi.MANIFEST_NAME).unlink()
            if corrupt_base:
                (base / abi.LIBRARY).write_bytes(b"stale base manifest")
            if corrupt_overlay:
                (overlay / abi.MANIFEST_NAME).unlink()
            return subprocess.run(
                [sys.executable, str(ROOT / "ci/check_partial_opp_abi.py"),
                 "--overlay", str(overlay)],
                cwd=root, env=dict(os.environ, PYTHONPATH=str(root), FLA_NPU_DISABLE_PTH="1"),
                text=True, capture_output=True,
            )

    def test_matching_base_keeps_incremental_build_without_loading_runtime(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                result = self.check_vendors(legacy_base=legacy)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ABI matches", result.stdout)

    def test_missing_interface_requests_full_build(self):
        result = self.check_vendors(base_name="OldDemo")
        self.assertEqual(result.returncode, 10, result.stderr)
        self.assertIn("aclnnDemoGetWorkspaceSize: not declared by the base wheel", result.stderr)

    def test_changed_parameter_requests_full_build(self):
        result = self.check_vendors(overlay_parameter="float input")
        self.assertEqual(result.returncode, 10, result.stderr)
        self.assertIn("overlay=['float'", result.stderr)

    def test_invalid_base_requests_full_build(self):
        result = self.check_vendors(corrupt_base=True)
        self.assertEqual(result.returncode, 10, result.stderr)
        self.assertIn("Stale or invalid ABI manifest", result.stderr)

    def test_invalid_overlay_fails_without_requesting_full_build(self):
        result = self.check_vendors(corrupt_overlay=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Cannot validate the built partial OPP", result.stderr)


if __name__ == "__main__":
    unittest.main()
