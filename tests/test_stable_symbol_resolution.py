"""Exercise the production C++ resolver against real Linux ELF dependencies."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "torch_custom/fla_npu/csrc"


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++"), "Linux ELF compiler required")
class SymbolResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.lib("libdependency.so", "aclnnDemo", 900, extra='extern "C" int aclHelper(){return 901;}')
        cls.lib("libopapi.so", "aclnnDemo", 800)
        cls.lib("base.so", "aclnnDemo", 100)
        cls.lib("overlay.so", "aclnnDemo", 200, dependency=True)
        cls.lib("new.so", "aclnnNew", 300, dependency=True)
        cls.lib("partial.so", "aclnnDemo", 400, dependency=True, phase="workspace")
        cls.lib("partial_launch.so", "aclnnDemo", 500, phase="launch")
        for vendor in ("vendor_a", "vendor_b", "split_cann"):
            (cls.root / vendor).mkdir()
        cls.lib("vendor_a/libcust_opapi.so", "aclnnNew", 300, dependency=True)
        cls.lib("vendor_b/libcust_opapi.so", "aclnnDemo", 100)
        cls.lib("split_cann/libopapi.so", "aclnnDemo", 800, dependency=True, phase="launch")
        (cls.root / "base-link.so").symlink_to(cls.root / "base.so")
        driver = cls.root / "driver.cpp"
        driver.write_text(r'''
#include "stable/runtime.h"
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
extern "C" int fla_npu_stable_set_verified_base(const char*);
extern "C" int __real_dladdr1(const void*, Dl_info*, void**, int);
extern "C" int __real_dlinfo(void*, int, void*);
int owner_checks = 0;
extern "C" int __wrap_dladdr1(const void* p, Dl_info* i, void** e, int f) {
  ++owner_checks; return __real_dladdr1(p, i, e, f);
}
extern "C" int __wrap_dlinfo(void* p, int r, void* i) {
  ++owner_checks; return __real_dlinfo(p, r, i);
}
int main(int argc,char** argv) {
  try {
    if(argc>3 && std::string(argv[3])=="1") dlopen("libopapi.so",RTLD_LOCAL|RTLD_NOW);
    auto& rt=fla_npu_stable::Runtime::instance();
    const std::string mode(argv[1]);
    if(mode=="late") {
      rt.init();
      if(fla_npu_stable_set_verified_base(argv[4])==0) return 7;
      std::cout<<"rejected"; return 0;
    }
    if(argc>4 && *argv[4]) {
      if(fla_npu_stable_set_verified_base(argv[4])!=0 ||
         fla_npu_stable_set_verified_base(argv[4])!=0) return 8;
    }
    using Fn=int(*)();
    if(mode.find("pair")!=std::string::npos) {
      auto pair=rt.op_api(argv[2]);
      auto again=rt.op_api(argv[2]);
      if(pair.get_workspace!=again.get_workspace || pair.launch!=again.launch) return 4;
      std::cout<<reinterpret_cast<Fn>(pair.get_workspace)()<<","<<reinterpret_cast<Fn>(pair.launch)();
    } else {
      void* address=rt.symbol(argv[2]);
      if(address!=rt.symbol(argv[2])) return 5;
      std::cout<<reinterpret_cast<Fn>(address)();
    }
    if(mode.find("no_owner")!=std::string::npos && owner_checks!=0) return 6;
  } catch(const std::exception& e) {std::cerr<<e.what();return 2;}
}
''')
        subprocess.run(["g++", "-O3", "-std=c++17", "-pthread", "-I"+str(CSRC/"include"),
                        str(CSRC/"src/runtime.cpp"), str(driver), "-ldl",
                        "-Wl,--wrap=dladdr1,--wrap=dlinfo", "-o", str(cls.root/"driver")], check=True)

    @classmethod
    def lib(cls, name, symbol, value, *, dependency=False, phase="both", extra=""):
        source = cls.root / (name + ".cpp")
        source.write_text(
            (f'extern "C" int {symbol}GetWorkspaceSize(){{return {value};}}\n' if phase != "launch" else "")
            + (f'extern "C" int {symbol}(){{return {value+1};}}\n' if phase != "workspace" else "") + extra)
        command = ["g++", "-shared", "-fPIC", str(source), "-o", str(cls.root/name)]
        if dependency:
            command += ["-Wl,--no-as-needed", "-L"+str(cls.root), "-ldependency", "-Wl,-rpath,$ORIGIN"]
        subprocess.run(command, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, libs, symbol="aclnnDemo", mode="pair", preload=False, cann_dir=None,
                  verified_base=None):
        search = str(self.root) if cann_dir is None else str(self.root/cann_dir)+":"+str(self.root)
        env = dict(os.environ, LD_LIBRARY_PATH=search,
                   FLA_NPU_OP_API_LIBS=":".join(str(self.root/name) for name in libs))
        return subprocess.run([str(self.root/"driver"), mode, symbol, "1" if preload else "0",
                               str(self.root/verified_base) if verified_base else ""],
                              env=env, text=True, capture_output=True)

    def test_dependency_cannot_shadow_base(self):
        for preload in (False, True):
            with self.subTest(preload=preload):
                result = self.run_probe(["new.so", "base.so"], preload=preload)
                self.assertEqual((result.returncode, result.stdout), (0, "100,101"), result.stderr)

    def test_single_symbol_also_skips_dependency(self):
        result = self.run_probe(["new.so", "base.so"], mode="single")
        self.assertEqual((result.returncode, result.stdout), (0, "101"), result.stderr)

    def test_overlay_wins_when_it_defines_both(self):
        result = self.run_probe(["overlay.so", "base.so"])
        self.assertEqual((result.returncode, result.stdout), (0, "200,201"), result.stderr)

    def test_new_interface_needs_no_base_entry(self):
        result = self.run_probe(["new.so", "base.so"], "aclnnNew")
        self.assertEqual((result.returncode, result.stdout), (0, "300,301"), result.stderr)

    def test_cann_fallback_skips_custom_dependencies(self):
        result = self.run_probe(["new.so"])
        self.assertEqual((result.returncode, result.stdout), (0, "800,801"), result.stderr)

    def test_partial_custom_pair_fails_instead_of_mixing(self):
        for name in ("partial.so", "partial_launch.so"):
            with self.subTest(name=name):
                result = self.run_probe([name, "base.so"])
                self.assertEqual(result.returncode, 2)
                self.assertIn("incomplete custom ACLNN pair", result.stderr)

    def test_non_aclnn_dependency_resolution_is_preserved(self):
        result = self.run_probe(["new.so", "base.so"], "aclHelper", "single")
        self.assertEqual((result.returncode, result.stdout), (0, "901"), result.stderr)

    def test_symlink_provider_identity(self):
        result = self.run_probe(["new.so", "base-link.so"])
        self.assertEqual((result.returncode, result.stdout), (0, "100,101"), result.stderr)

    def test_same_basename_does_not_confuse_provider_identity(self):
        result = self.run_probe(["vendor_a/libcust_opapi.so", "vendor_b/libcust_opapi.so"])
        self.assertEqual((result.returncode, result.stdout), (0, "100,101"), result.stderr)

    def test_cann_pair_from_different_providers_is_rejected(self):
        result = self.run_probe(["new.so"], cann_dir="split_cann")
        self.assertEqual(result.returncode, 2)
        self.assertIn("CANN ACLNN phases have different providers", result.stderr)

    def test_missing_pair_reports_error(self):
        result = self.run_probe(["base.so"], "aclnnAbsent")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing CANN ACLNN pair", result.stderr)

    def test_verified_base_has_no_ownership_queries(self):
        for mode, expected in (("no_owner_pair", "100,101"), ("no_owner_single", "101")):
            with self.subTest(mode=mode):
                result = self.run_probe(["base.so"], mode=mode, verified_base="base.so")
                self.assertEqual((result.returncode, result.stdout), (0, expected), result.stderr)

    def test_overlay_checks_are_preserved_with_verified_base(self):
        for overlay, expected in (("new.so", "100,101"), ("overlay.so", "200,201")):
            with self.subTest(overlay=overlay):
                result = self.run_probe([overlay, "base.so"], verified_base="base.so")
                self.assertEqual((result.returncode, result.stdout), (0, expected), result.stderr)
        result = self.run_probe(["partial.so", "base.so"], verified_base="base.so")
        self.assertEqual(result.returncode, 2)
        self.assertIn("incomplete custom ACLNN pair", result.stderr)

    def test_single_unverified_library_keeps_strict_checks(self):
        result = self.run_probe(["partial.so"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("incomplete custom ACLNN pair", result.stderr)

    def test_verified_registration_requires_matching_terminal_path(self):
        # Registering another file must not exempt this partial library.
        result = self.run_probe(["partial.so"], verified_base="base.so")
        self.assertEqual(result.returncode, 2)
        # Even a registered path cannot bypass a later vendor via dependencies.
        result = self.run_probe(["new.so", "base.so"], verified_base="new.so")
        self.assertEqual((result.returncode, result.stdout), (0, "100,101"), result.stderr)
        # A repeated/aliased terminal handle must not exempt an earlier entry.
        result = self.run_probe(["new.so", "base.so", "new.so"], verified_base="new.so")
        self.assertEqual((result.returncode, result.stdout), (0, "100,101"), result.stderr)

    def test_registration_after_initialization_is_rejected(self):
        result = self.run_probe(["base.so"], mode="late", verified_base="base.so")
        self.assertEqual((result.returncode, result.stdout), (0, "rejected"), result.stderr)


if __name__ == "__main__":
    unittest.main()
