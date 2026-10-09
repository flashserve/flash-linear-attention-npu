#!/usr/bin/env python3
"""The installed-OPP manifest check has to compare installed paths only.

``bdist_wheel`` moves the whole payload under ``<dist>-<version>.data/<scheme>/``
when a distribution reports ``root_is_pure == False`` without extension modules,
and pip maps that back onto the same site root.  The archive layout is therefore
not part of the comparison: the check has to fold the scheme directory away and
keep comparing the installed tree byte for byte.
"""

import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_install_workflows.py"
DIST_INFO = "flash_linear_attention_npu_a2-1.0.dist-info"

OPP_FILES = {
    "fla_npu/opp/vendors/config.ini": b"load_priority=fla_npu_transformer\n",
    "fla_npu/opp/vendors/fla_npu_transformer/README.txt": b"readme\n",
    "fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash": b"export A=1\n",
    "fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so": b"\x7fELF",
}


def _write_wheel(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{DIST_INFO}/METADATA",
            "Metadata-Version: 2.1\nName: flash-linear-attention-npu-a2\nVersion: 1.0\n",
        )
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr(f"{DIST_INFO}/RECORD", f"{DIST_INFO}/RECORD,,\n")


def _data_layout_entries(entries):
    return {
        f"flash_linear_attention_npu_a2-1.0.data/purelib/{name}": payload
        for name, payload in entries.items()
    }


def _write_installed_tree(site_root, entries):
    for name, payload in entries.items():
        path = site_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


class InstalledOppManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = runpy.run_path(str(CHECKER))

    def _assert_layout_accepted(self, wheel_entries):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wheel = root / "flash_linear_attention_npu_a2-1.0-py3-none-any.whl"
            _write_wheel(wheel, wheel_entries)
            site_root = root / "site-packages"
            _write_installed_tree(site_root, OPP_FILES)
            package_dir = site_root / "fla_npu"

            self.assertEqual(
                self.checker["_manifest_from_wheel"](wheel),
                self.checker["_manifest_from_directory"](package_dir),
            )
            self.checker["_assert_manifest_matches_wheel"](package_dir, wheel)

    def _failure_for(self, wheel_entries, installed_entries):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wheel = root / "flash_linear_attention_npu_a2-1.0-py3-none-any.whl"
            _write_wheel(wheel, wheel_entries)
            site_root = root / "site-packages"
            _write_installed_tree(site_root, installed_entries)
            with self.assertRaises(AssertionError) as context:
                self.checker["_assert_manifest_matches_wheel"](
                    site_root / "fla_npu", wheel
                )
            return str(context.exception)

    def test_root_layout_matches_installed_tree(self):
        self._assert_layout_accepted(OPP_FILES)

    def test_data_purelib_layout_matches_installed_tree(self):
        self._assert_layout_accepted(_data_layout_entries(OPP_FILES))

    def test_data_platlib_layout_matches_installed_tree(self):
        self._assert_layout_accepted(
            {
                name.replace(".data/purelib/", ".data/platlib/"): payload
                for name, payload in _data_layout_entries(OPP_FILES).items()
            }
        )

    def test_dot_slash_prefixed_entries_match_installed_tree(self):
        self._assert_layout_accepted(
            {f"./{name}": payload for name, payload in OPP_FILES.items()}
        )

    def test_payload_outside_the_known_schemes_is_still_reported(self):
        wheel_entries = {
            name.replace(".data/purelib/", ".data/data/"): payload
            for name, payload in _data_layout_entries(OPP_FILES).items()
        }
        message = self._failure_for(wheel_entries, OPP_FILES)
        # Only ``purelib``/``platlib`` map onto the site root; any other scheme
        # keeps the payload outside the installed OPP tree and must not pass.
        self.assertIn("missing=[]", message)
        self.assertIn("unexpected=['fla_npu/opp/vendors/config.ini'", message)

    def test_missing_installed_opp_file_is_reported(self):
        installed = dict(OPP_FILES)
        installed.pop("fla_npu/opp/vendors/config.ini")
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("missing=['fla_npu/opp/vendors/config.ini']", message)

    def test_extra_installed_opp_file_is_still_reported(self):
        installed = dict(OPP_FILES)
        installed["fla_npu/opp/vendors/fla_npu_transformer/extra.bin"] = b"x"
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("unexpected=['fla_npu/opp/vendors/fla_npu_transformer/extra.bin']", message)

    def test_changed_installed_opp_file_is_still_reported(self):
        installed = dict(OPP_FILES)
        installed["fla_npu/opp/vendors/config.ini"] = b"load_priority=other\n"
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("changed=['fla_npu/opp/vendors/config.ini']", message)


if __name__ == "__main__":
    unittest.main()
