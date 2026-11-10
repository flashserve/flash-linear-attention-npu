"""A fast base must come from validated package metadata, never a library count."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_wheel_environment import (
    _create_runtime_package, _load_runtime, _write_aclnn_header,
)


class VerifiedBasePackageTest(unittest.TestCase):
    def load_fixture(self, mutate=None, manifest=True, expected_fast=None):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, vendor, library = _create_runtime_package(Path(tmp))
            if manifest:
                _write_aclnn_header(vendor, "int64_t")
            if mutate:
                mutate(vendor, library)
            with mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": "/fake/cann"}, clear=True), \
                    mock.patch("ctypes.CDLL", return_value=mock.sentinel.library):
                loaded = _load_runtime(runtime)
                fast = manifest if expected_fast is None else expected_fast
                self.assertEqual(loaded["_VERIFIED_BASE_OPAPI"], str(library.resolve()) if fast else None)
                # Loading the cached package again must not repeat binary hashing.
                with mock.patch("hashlib.sha256", side_effect=AssertionError("repeated base hash")):
                    loaded["load_ascendc_opapi_libraries"]()

    def test_valid_base_is_registered(self):
        self.load_fixture()

    def test_legacy_base_remains_unregistered(self):
        self.load_fixture(manifest=False)

    def test_changed_binary_cannot_enable_fast_lookup(self):
        with self.assertWarnsRegex(RuntimeWarning, "using strict ACLNN"):
            self.load_fixture(lambda vendor, library: library.write_bytes(b"modified"), expected_fast=False)

    def test_changed_header_cannot_enable_fast_lookup(self):
        def change_header(vendor, library):
            header = vendor / "op_api/include/aclnnop/aclnn_demo.h"
            header.write_text(header.read_text() + "\n// modified\n")
        with self.assertWarnsRegex(RuntimeWarning, "using strict ACLNN"):
            self.load_fixture(change_header, expected_fast=False)

    def test_partial_pair_evidence_keeps_strict_lookup(self):
        def remove_launch(vendor, library):
            manifest = vendor / "abi_manifest.json"
            data = json.loads(manifest.read_text())
            del data["contracts"]["aclnnDemo"]
            manifest.write_text(json.dumps(data))
        with self.assertWarnsRegex(RuntimeWarning, "using strict ACLNN"):
            self.load_fixture(remove_launch, expected_fast=False)


if __name__ == "__main__":
    unittest.main()
