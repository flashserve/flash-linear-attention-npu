#!/usr/bin/env python3
"""The Python OPP installer must merge ``load_priority`` instead of replacing it.

CANN resolves several custom vendors that share one ``opp/vendors`` root through
the ``load_priority`` list in ``vendors/config.ini``.  ``install_opp`` rewrote
that file with its own vendor name only, so every vendor already registered
there (DrivingSDK's ``customize``, for example) silently dropped out of the list
and its operators could no longer be discovered.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_OPP = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu" / "install_opp.py"
VENDOR_NAME = "fla_npu_transformer"
OTHER_VENDOR = "customize"


def _load_install_opp():
    spec = importlib.util.spec_from_file_location("_fla_npu_install_opp", INSTALL_OPP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VendorsConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.install_opp = _load_install_opp()

    def _vendors_root_with_config(self, root, config_text):
        vendors_root = Path(root) / "vendors"
        vendors_root.mkdir(parents=True, exist_ok=True)
        (vendors_root / "config.ini").write_text(config_text, encoding="utf-8")
        return vendors_root

    def _run_write_vendors_config(self, vendors_root, names=(VENDOR_NAME,)):
        vendor_dirs = []
        for name in names:
            vendor_dir = vendors_root / name
            vendor_dir.mkdir(parents=True, exist_ok=True)
            vendor_dirs.append(vendor_dir)
        self.install_opp._write_vendors_config(vendors_root, vendor_dirs)
        return (vendors_root / "config.ini").read_text(encoding="utf-8")

    def test_existing_vendor_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendors_root = self._vendors_root_with_config(
                temp_dir, f"load_priority={OTHER_VENDOR}\n"
            )
            self.assertEqual(
                self._run_write_vendors_config(vendors_root),
                f"load_priority={VENDOR_NAME},{OTHER_VENDOR}\n",
            )

    def test_existing_vendors_keep_their_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendors_root = self._vendors_root_with_config(
                temp_dir, f"load_priority={OTHER_VENDOR},other_vendor\n"
            )
            self.assertEqual(
                self._run_write_vendors_config(vendors_root),
                f"load_priority={VENDOR_NAME},{OTHER_VENDOR},other_vendor\n",
            )

    def test_repeated_install_does_not_duplicate_vendors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendors_root = self._vendors_root_with_config(
                temp_dir, f"load_priority={OTHER_VENDOR}\n"
            )
            first = self._run_write_vendors_config(vendors_root)
            second = self._run_write_vendors_config(vendors_root)
            self.assertEqual(first, second)
            self.assertEqual(second, f"load_priority={VENDOR_NAME},{OTHER_VENDOR}\n")

    def test_missing_config_still_registers_own_vendor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendors_root = Path(temp_dir) / "vendors"
            vendors_root.mkdir(parents=True)
            self.assertEqual(
                self._run_write_vendors_config(vendors_root),
                f"load_priority={VENDOR_NAME}\n",
            )

    def test_install_opp_keeps_vendors_already_present_in_target_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_opp = root / "package" / "opp"
            vendor_src = package_opp / "vendors" / VENDOR_NAME / "op_api" / "lib"
            vendor_src.mkdir(parents=True)
            (vendor_src / "libcust_opapi.so").write_bytes(b"\x7fELF")

            install_root = root / "target"
            vendors_root = self._vendors_root_with_config(
                install_root, f"load_priority={OTHER_VENDOR}\n"
            )

            original_package_opp_root = self.install_opp.PACKAGE_OPP_ROOT
            self.install_opp.PACKAGE_OPP_ROOT = package_opp
            try:
                self.install_opp.install_opp(install_root)
            finally:
                self.install_opp.PACKAGE_OPP_ROOT = original_package_opp_root

            self.assertEqual(
                (vendors_root / "config.ini").read_text(encoding="utf-8"),
                f"load_priority={VENDOR_NAME},{OTHER_VENDOR}\n",
            )


if __name__ == "__main__":
    unittest.main()
