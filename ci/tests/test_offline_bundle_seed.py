#!/usr/bin/env python3
"""The image's third_party seed must be able to satisfy the offline bundle.

``scripts/tools/prepare_offline_bundle.py`` hard-fails when any archive in its
manifest is missing from the cache, and the release gates a wheel on that bundle
(``check_pypi_wheel.py --require-offline-bundle``).  The cache is seeded from
``/opt/fla-ci/third_party`` inside the CI image, so an archive the image forgets
to prewarm cannot be recovered at build time: ``cmake/third_party/gtest.cmake``
only fetches its archive when gtest is missing, and the base image already ships
gtest.  The result is a release that fails at the very end of a long build.

Keep the image seed and the bundle manifest in step.
"""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_BUNDLE_PATH = REPO_ROOT / "scripts" / "tools" / "prepare_offline_bundle.py"
PREWARM_DOCKERFILES = (
    REPO_ROOT / "ci" / "Dockerfile",
    REPO_ROOT / "ci" / "Dockerfile.ascend950",
)
SEED_DIR = "/opt/fla-ci/third_party"


def _load_bundle_manifest():
    spec = importlib.util.spec_from_file_location(
        "fla_npu_prepare_offline_bundle", PREPARE_BUNDLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfflineBundleSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = _load_bundle_manifest()

    def test_prepare_bundle_lists_archives(self):
        # Guard the guard: an empty manifest would make the checks below vacuous.
        self.assertIn("googletest-1.14.0.tar.gz", self.manifest._ARCHIVES)
        self.assertIn("opbase", [name for name, _, _, _ in self.manifest._COMPONENTS])

    def test_every_prewarm_dockerfile_seeds_the_whole_archive_set(self):
        for dockerfile in PREWARM_DOCKERFILES:
            with self.subTest(dockerfile=dockerfile.name):
                text = dockerfile.read_text(encoding="utf-8")
                self.assertIn(
                    "/opt/fla-ci/third_party/pkg",
                    text,
                    f"{dockerfile.name}: no third_party prewarm block",
                )
                # Destinations are written as ".../<archive>; \" so stop at the
                # shell separator rather than taking the rest of the line.
                seeded = set(
                    re.findall(
                        rf"{re.escape(SEED_DIR)}/pkg/([^\s;\\]+)", text
                    )
                )
                missing = sorted(set(self.manifest._ARCHIVES) - seeded)
                self.assertEqual(
                    missing,
                    [],
                    f"{dockerfile.name}: the image seed omits {missing}, so a wheel "
                    "built from it cannot carry a complete offline bundle",
                )

    def test_every_prewarm_dockerfile_clones_the_source_components(self):
        cloned = {
            name
            for name, _, _, _ in self.manifest._COMPONENTS
            if name in {"opbase", "catlass"}
        }
        for dockerfile in PREWARM_DOCKERFILES:
            with self.subTest(dockerfile=dockerfile.name):
                text = dockerfile.read_text(encoding="utf-8")
                for name in sorted(cloned):
                    self.assertIn(
                        f"{SEED_DIR}/{name}",
                        text,
                        f"{dockerfile.name}: the image seed does not clone {name}",
                    )


if __name__ == "__main__":
    unittest.main()
