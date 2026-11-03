"""Verify catalog-derived error capabilities and private host include scope."""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_host"
CODES = {"EZ0008", "EZ0009", "EZ0010", "EZ0013", "EZ0017", "EZ0018", "EZ0019",
         "EZ0021", "EZ0024", "EZ0025", "EZ0026", "EZ0027", "EZ0037"}
LEGACY = {"EZ0002", "EZ0003", "EZ0004", "EZ0006", "EZ0007"}


@unittest.skipUnless(shutil.which("cmake") and any(shutil.which(x) for x in ("c++", "g++", "cl")),
                     "CMake and a C++ compiler are required")
class PrepareErrorCatalogTest(unittest.TestCase):
    def configure(self, native_codes):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "opbase/src/op_common/log/log.cpp"
            catalog.parent.mkdir(parents=True)
            records = [{"ErrCode": code} for code in sorted(LEGACY | set(native_codes))]
            catalog.write_text('const char *catalog = R"json(' +
                               json.dumps({"error_info_list": records}, indent=2) +
                               ')json";\n', encoding="utf-8")
            project = r'''
cmake_minimum_required(VERSION 3.18)
project(PrepareCatalogTest LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(OPHOST_NAME prepare_catalog_test)
set(OPBASE_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/opbase")
file(WRITE "${CMAKE_BINARY_DIR}/unrelated.cpp" "void unrelated() {}\n")
function(add_op_to_compiled_list)
endfunction()
function(add_ops_compile_options)
endfunction()
function(add_modules_sources)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT
        "@HOST@/op_api/aclnn_chunk_kda_fwd_prepare.cpp"
        "${CMAKE_BINARY_DIR}/unrelated.cpp")
    add_library(${OPHOST_NAME}_infer_obj OBJECT
        "@HOST@/chunk_kda_fwd_prepare_infershape.cpp")
    add_library(${OPHOST_NAME}_tiling_obj OBJECT
        "@HOST@/chunk_kda_fwd_prepare_tiling.cpp")
endfunction()
add_subdirectory("@HOST@" host)
'''.replace("@HOST@", HOST.as_posix())
            (root / "CMakeLists.txt").write_text(project, encoding="utf-8")
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            generated = root / "build/host/prepare_error_catalog/chunk_kda_fwd_prepare_error_catalog.h"
            text = generated.read_text(encoding="utf-8")
            observed = {code.upper(): value == "true" for code, value in
                        re.findall(r"inline constexpr bool kHas(Ez\d+) = (true|false);", text)}
            self.assertEqual(observed, {code: code in native_codes for code in CODES})
            self.assertNotIn("@PREPARE", text)
            commands = json.loads((root / "build/compile_commands.json").read_text(encoding="utf-8"))
            own_sources = {"aclnn_chunk_kda_fwd_prepare.cpp", "chunk_kda_fwd_prepare_infershape.cpp",
                           "chunk_kda_fwd_prepare_tiling.cpp"}
            config_dir = str(generated.parent)
            for item in commands:
                self.assertEqual(config_dir in item["command"],
                                 Path(item["file"]).name in own_sources, item["file"])

    def test_legacy_catalog_disables_unregistered_native_codes(self):
        self.configure(set())

    def test_native_catalog_enables_its_registered_codes(self):
        self.configure(CODES)

    def test_partial_catalog_selects_capability_per_code(self):
        self.configure(CODES - {"EZ0008"})


if __name__ == "__main__":
    unittest.main()