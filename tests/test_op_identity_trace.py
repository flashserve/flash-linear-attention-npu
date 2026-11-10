"""Coverage, code-generation and zero-cost-off checks for entry diagnostics."""
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
GEN = runpy.run_path(str(ROOT/"scripts/util/instrument_aclnn_trace.py"))
KERNEL = re.compile(r"__global__\s+__aicore__\s+void\s+(\w+)\s*\([^;{}]*\)\s*\{")


class IdentityTraceTest(unittest.TestCase):
    def test_all_kernel_compile_options_accumulate(self):
        parser = runpy.run_path(str(ROOT/"cmake/scripts/util/opdesc_parser.py"))
        ops = [SimpleNamespace(op_type="Demo", custom_all_compile_options={}),
               SimpleNamespace(op_type="Other", custom_all_compile_options={})]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"custom_compile_options.ini").write_text(
                "ALL,,-DNOT_DYNAMIC_COMPILE;-g\n"
                "ALL,,-I/example/include\n"
                "ALL,,-DFLA_NPU_OP_TRACE=1;-DASCENDC_DUMP=1\n")
            parser["_get_op_custom_options"](ops, directory)
        expected = ["-DNOT_DYNAMIC_COMPILE", "-g", "-I/example/include",
                    "-DFLA_NPU_OP_TRACE=1", "-DASCENDC_DUMP=1"]
        for op in ops:
            self.assertEqual(op.custom_all_compile_options[parser["SOC_ALL"]], expected)
        self.assertIsNot(ops[0].custom_all_compile_options, ops[1].custom_all_compile_options)

    def test_all_handwritten_entries_have_markers(self):
        host = kernel = 0
        for path in (ROOT/"fla/ops/ascendc").rglob("*"):
            if path.suffix not in (".cpp", ".h"):
                continue
            text = path.read_text(encoding="utf-8")
            if "op_api" in path.parts and path.suffix == ".cpp":
                for entry in GEN["ENTRY"].finditer(text):
                    host += 1
                    self.assertTrue(text[entry.end():].lstrip().startswith("FLA_ACLNN_TRACE();"), str(path))
            if "op_kernel" in path.parts:
                for entry in KERNEL.finditer(text):
                    kernel += 1
                    self.assertTrue(text[entry.end():].lstrip().startswith("FLA_KERNEL_TRACE("), str(path))
        self.assertGreater(host, 0)
        self.assertGreater(kernel, 0)

    def test_generated_overloads_and_idempotence(self):
        source = '''aclnnStatus aclnnDemoGetWorkspaceSize(const aclTensor* x, uint64_t* w, aclOpExecutor** e) { return 0; }
aclnnStatus aclnnDemo(void* w, uint64_t n, aclOpExecutor* e, aclrtStream s) { return 0; }
aclnnStatus aclnnDemoTensorGetWorkspaceSize(const aclTensor* x, uint64_t* w, aclOpExecutor** e) { return 0; }
'''
        result = GEN["instrument"](source)
        self.assertEqual(result.count("FLA_ACLNN_TRACE();"), 3)
        self.assertEqual(GEN["instrument"](result), result)

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++"), "Linux compiler required")
    def test_disabled_macros_produce_identical_o3_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"sample.cpp"
            code = '#include "fla_aclnn_trace.h"\n#include "fla_kernel_trace.h"\nextern "C" int sample(int x) { %s return x*3+1; }\n'
            results = []
            for marker in ("", 'FLA_ACLNN_TRACE(); FLA_KERNEL_TRACE("sample");'):
                source.write_text(code % marker)
                subprocess.run(["g++", "-O3", "-S", "-I"+str(ROOT/"common/include"),
                                "-I"+str(ROOT/"fla/ops/ascendc/common"), str(source), "-o", str(root/"sample.s")], check=True)
                results.append((root/"sample.s").read_bytes())
            self.assertEqual(*results)

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++"), "Linux compiler required")
    def test_enabled_host_marker_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"sample.cpp"
            source.write_text('#include "fla_aclnn_trace.h"\nint main(){FLA_ACLNN_TRACE();}\n')
            subprocess.run(["g++", "-O3", "-DFLA_NPU_OP_TRACE=1", "-I"+str(ROOT/"common/include"), str(source), "-o", str(root/"sample")], check=True)
            result = subprocess.run([str(root/"sample")], text=True, capture_output=True, check=True)
            self.assertIn("[FLA][ACLNN] main", result.stderr)


if __name__ == "__main__":
    unittest.main()
