"""Exercise the real resolver with tiny host libraries, without CANN or NPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "torch_custom/fla_npu/csrc"


@unittest.skipUnless(os.name == "posix" and shutil.which("g++"), "Linux host compiler required")
class SymbolSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for name, source in {
            "custom": 'extern "C" int aclnnOwned() { return 7; }',
            "empty": 'extern "C" int unrelated() { return 1; }',
            "opapi": 'extern "C" int aclnnOwned() { return 9; } extern "C" int aclHelper() { return 4; }',
        }.items():
            path = cls.root / (name + ".cpp")
            path.write_text(source, encoding="utf-8")
            subprocess.run(["g++", "-shared", "-fPIC", str(path), "-o", str(cls.root / ("lib" + name + ".so"))], check=True)
        source = cls.root / "probe.cpp"
        source.write_text('#include "binding.h"\n#include <iostream>\n'
            'int main(int argc, char** argv) { try { bind_experimental_symbols(argv[2], '
            'std::string(argv[1]) == "aclnnOwned" ? std::initializer_list<const char*>{"aclnnOwned"} : std::initializer_list<const char*>{}); '
            'auto fn = reinterpret_cast<int(*)()>('
            'fla_npu_stable::Runtime::instance().symbol(argv[1])); std::cout << fn(); return 0; }'
            'catch (const std::exception& e) { std::cerr << e.what(); return 2; } }', encoding="utf-8")
        cls.probe = cls.root / "probe"
        subprocess.run(["g++", "-std=c++17", "-I", str(ROOT / "experimental/torch_custom"), "-I", str(CSRC / "include"),
            str(source), str(CSRC / "src/runtime.cpp"), "-ldl", "-o", str(cls.probe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_symbol(self, custom, name):
        env = dict(os.environ, LD_LIBRARY_PATH=str(self.root),
            FLA_NPU_EXPERIMENTAL_OP_API_LIB=str(self.root / custom),
            FLA_NPU_OP_API_LIB=str(self.root / "libcustom.so"),
            FLA_NPU_OP_API_LIBS=str(self.root / "libcustom.so"))
        return subprocess.run([str(self.probe), name, str(self.root / custom)], env=env, capture_output=True, text=True)

    def test_owned_symbol_uses_only_selected_experimental_library(self):
        result = self.run_symbol("libcustom.so", "aclnnOwned")
        self.assertEqual((result.returncode, result.stdout), (0, "7"), result.stderr)

    def test_missing_owned_symbol_never_falls_back_to_base_or_cann(self):
        result = self.run_symbol("libempty.so", "aclnnOwned")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing owned symbol", result.stderr)

    def test_sdk_helper_can_resolve_from_cann(self):
        result = self.run_symbol("libempty.so", "aclHelper")
        self.assertEqual((result.returncode, result.stdout), (0, "4"), result.stderr)
