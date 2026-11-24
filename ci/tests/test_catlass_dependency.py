"""Guard the single pinned Catlass dependency used by every SoC build."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "https://gitcode.com/cann/catlass.git"
PINNED_REVISION = "3a6b66970ea1469fb566eb5f50afa9a469606e92"


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

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("git"), "requires CMake and Git")
    def test_cached_revision_is_shared_by_all_socs_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fla-catlass-") as temporary:
            root = Path(temporary)
            cache = root / "third_party/catlass"
            header = cache / "include/catlass/catlass.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("// fixture\n", encoding="utf-8")

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(cache), *args], text=True, encoding="utf-8", stderr=subprocess.STDOUT,
                ).strip()

            git("init", "--quiet")
            git("add", "include")
            git("-c", "user.name=Catlass test", "-c", "user.email=test@example.invalid",
                "commit", "--quiet", "-m", "fixture")
            revision = git("rev-parse", "HEAD")
            production = (ROOT / "cmake/third_party/catlass.cmake").read_text(encoding="utf-8")
            module = root / "catlass.cmake"
            module.write_text(production.replace(PINNED_REVISION, revision), encoding="utf-8")
            script = root / "check.cmake"
            script.write_text(
                'cmake_minimum_required(VERSION 3.16)\n'
                'include("${CMAKE_CURRENT_LIST_DIR}/catlass.cmake")\n'
                'if(NOT CATLASS_VENDOR_GIT_TAG STREQUAL EXPECTED_REVISION)\n'
                '  message(FATAL_ERROR "SoC changed the pinned revision")\n'
                'endif()\n', encoding="utf-8",
            )

            def configure(soc):
                return subprocess.run([
                    "cmake", f"-DCANN_3RD_LIB_PATH={cache.parent.as_posix()}",
                    f"-DASCEND_COMPUTE_UNIT={soc}", f"-DEXPECTED_REVISION={revision}",
                    "-P", str(script),
                ], text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            for soc in ("ascend310p", "ascend910b", "ascend910_93", "ascend950", "ascend960dt",
                        "mc62cm12a", "kirinx90", "kirin9030",
                        "Ascend960Dt;", "ascend950;ASCEND960DT"):
                with self.subTest(soc=soc):
                    result = configure(soc)
                    self.assertEqual(result.returncode, 0, result.stdout)

            module.write_text(production, encoding="utf-8")
            result = configure("ascend950")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Cached Catlass revision", result.stdout)
            self.assertIn("does not match", " ".join(result.stdout.split()))
            self.assertEqual(git("rev-parse", "HEAD"), revision)


if __name__ == "__main__":
    unittest.main()
