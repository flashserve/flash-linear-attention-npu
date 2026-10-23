#!/usr/bin/env python3
"""The removed legacy extension must not be importable or shipped."""
from pathlib import Path
import unittest


class RemovedLegacyBackendTest(unittest.TestCase):
    def test_legacy_api_and_payload_are_absent(self):
        import fla_npu
        from fla_npu.ops import ascendc
        root = Path(fla_npu.__file__).resolve().parent
        for name in ("load_legacy_torch_ops", "is_legacy_torch_ops_loaded"):
            self.assertFalse(hasattr(fla_npu, name), name)
        self.assertFalse(hasattr(ascendc, "install_torch_npu_ops_compat"))
        self.assertEqual(list(root.rglob("custom_aclnn_extension_lib*.so")), [])
        for name in ("_runtime.py", "_aclnn_ctypes.py"):
            self.assertFalse((root / "ops" / "ascendc" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
