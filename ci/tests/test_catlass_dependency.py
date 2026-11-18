"""Guard the single pinned Catlass dependency used by every SoC build."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "https://gitcode.com/cann/catlass.git"
PINNED_REVISION = "490ae05852f7fad7e5e2ed13b305a80f4d9d91f9"


class CatlassDependencyTest(unittest.TestCase):
    def test_build_and_ci_use_the_single_official_revision(self):
        files = (
            ROOT / "cmake/third_party/catlass.cmake",
            ROOT / "ci/Dockerfile",
            ROOT / "ci/Dockerfile.ascend950",
        )
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertIn(OFFICIAL_REPOSITORY, text)
                self.assertIn(PINNED_REVISION, text)
                self.assertEqual(
                    re.findall(r"\b[0-9a-f]{40}\b", text)[-1],
                    PINNED_REVISION,
                )

    def test_cmake_revision_cannot_be_overridden_by_stale_cache(self):
        path = ROOT / "cmake/third_party/catlass.cmake"
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r'set\(CATLASS_VENDOR_GIT_TAG\s+"([0-9a-f]{40})"\s*\)',
            text,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), PINNED_REVISION)
        self.assertNotIn("CACHE", match.group(0))

    @unittest.skipUnless(shutil.which("cmake"), "CMake is required")
    def test_compat_headers_preserve_vendor_and_all_includes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / "vendor"
            block = vendor / "catlass/gemm/block"
            block.mkdir(parents=True)
            original = (
                'auto shape = MakeShape(rows, cols);\n'
                'auto strideInPage = MakeStride(cols, Int<1>{});\n'
                'auto alreadyQualified = tla::MakeShape(rows, cols);\n'
            )
            headers = ("block_mmad_hstu_qk_split_row_tla.hpp",
                       "block_mmad_hstu_pv_split_row_tla.hpp")
            for name in headers:
                (block / name).write_text(original, encoding="utf-8")
            umbrella = '#include "catlass/gemm/block/block_mmad_flash_attention_pv.hpp"\n'
            (block / "block_mmad.hpp").write_text(umbrella, encoding="utf-8")
            (vendor / "tla").mkdir()
            (vendor / "tla/layout.hpp").write_text("// retained\n", encoding="utf-8")
            helper = (ROOT / "cmake/third_party/catlass_compat.cmake").as_posix()
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.16)\n'
                'project(catlass_compat_check NONE)\n'
                f'include("{helper}")\n'
                'fla_prepare_catlass_headers("${CMAKE_SOURCE_DIR}/vendor" headers)\n',
                encoding="utf-8",
            )
            output = root / "build/_deps/catlass_fla/include"
            for iteration in range(2):
                subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build")],
                               check=True, capture_output=True, text=True)
                for name in headers:
                    self.assertEqual((block / name).read_text(), original)
                    patched = (output / "catlass/gemm/block" / name).read_text()
                    self.assertIn("auto shape = tla::MakeShape(", patched)
                    self.assertIn("auto strideInPage = tla::MakeStride(", patched)
                    self.assertIn("auto alreadyQualified = tla::MakeShape(", patched)
                    self.assertNotIn("tla::tla::", patched)
                self.assertEqual((output / "catlass/gemm/block/block_mmad.hpp").read_text(), umbrella)
                self.assertEqual((output / "tla/layout.hpp").read_text(), "// retained\n")
                self.assertFalse((output / "stale.hpp").exists())
                if iteration == 0:
                    (output / "stale.hpp").write_text("// stale generated header\n")


if __name__ == "__main__":
    unittest.main()
