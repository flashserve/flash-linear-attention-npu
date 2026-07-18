# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the LICENSE.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


FLA_NPU_INIT_PATH = Path(__file__).resolve().parents[1] / "fla_npu" / "__init__.py"
SPEC = importlib.util.spec_from_file_location("fla_npu_test_init", FLA_NPU_INIT_PATH)
FLA_NPU = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FLA_NPU)


class AscendCOpApiLoaderTest(unittest.TestCase):
    def setUp(self):
        FLA_NPU._ASCENDC_OPAPI_LIBRARIES = None

    def _vendor_dir(self, root: Path, custom: bytes, alias: bytes) -> Path:
        vendor_dir = root / "vendors" / "fla_npu_transformer"
        op_api_dir = vendor_dir / "op_api" / "lib"
        op_api_dir.mkdir(parents=True)
        (op_api_dir / "libcust_opapi.so").write_bytes(custom)
        (op_api_dir / "libopapi.so").write_bytes(alias)
        return vendor_dir

    def test_loads_compatibility_shim_without_loading_custom_twice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = self._vendor_dir(Path(temp_dir), b"custom", b"stale")
            loaded = []

            def fake_load(path):
                loaded.append(Path(path).name)
                return Path(path).name

            with mock.patch.object(FLA_NPU, "_prepare_embedded_opp", return_value=vendor_dir):
                with mock.patch.object(FLA_NPU, "_load_shared_library_required", side_effect=fake_load):
                    libraries = FLA_NPU.load_ascendc_opapi_libraries()

        self.assertEqual(loaded, ["libcust_opapi.so", "libopapi.so"])
        self.assertEqual(libraries, ["libcust_opapi.so", "libopapi.so"])

    def test_falls_back_to_custom_library_when_shim_is_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = self._vendor_dir(Path(temp_dir), b"same", b"same")
            (vendor_dir / "op_api" / "lib" / "libopapi.so").unlink()
            loaded = []

            def fake_load(path):
                loaded.append(Path(path).name)
                return Path(path).name

            with mock.patch.object(FLA_NPU, "_prepare_embedded_opp", return_value=vendor_dir):
                with mock.patch.object(FLA_NPU, "_load_shared_library_required", side_effect=fake_load):
                    libraries = FLA_NPU.load_ascendc_opapi_libraries()

        self.assertEqual(loaded, ["libcust_opapi.so"])
        self.assertEqual(libraries, ["libcust_opapi.so"])


if __name__ == "__main__":
    unittest.main()
