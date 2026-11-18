"""Guard the single pinned Catlass dependency used by every SoC build."""

from pathlib import Path
import re
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


if __name__ == "__main__":
    unittest.main()
