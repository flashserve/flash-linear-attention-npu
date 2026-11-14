"""Exercise dependency seeding across a Git revision that relocates headers."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PrepareCiCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "ci").mkdir(parents=True)
        shutil.copy2(ROOT / "ci/prepare_ci_cache.sh", self.repo / "ci/prepare_ci_cache.sh")
        self.seed = self.root / "seed"
        self.opbase = self.seed / "opbase"
        self.opbase.mkdir(parents=True)
        self.git(self.opbase, "init", "-q")
        self.git(self.opbase, "config", "user.name", "CI test")
        self.git(self.opbase, "config", "user.email", "ci-test@example.invalid")
        old_header = self.opbase / "pkg_inc/op_common/log/log.h"
        old_header.parent.mkdir(parents=True)
        old_header.write_text("old header\n")
        self.git(self.opbase, "add", ".")
        self.git(self.opbase, "commit", "-qm", "old layout")
        self.old = self.git(self.opbase, "rev-parse", "HEAD")
        new_header = self.opbase / "include/op_common/log/log.h"
        new_header.parent.mkdir(parents=True)
        self.git(self.opbase, "mv", str(old_header), str(new_header))
        new_header.write_text("new header\n")
        self.git(self.opbase, "commit", "-qam", "new layout")
        self.new = self.git(self.opbase, "rev-parse", "HEAD")
        self.git(self.opbase, "checkout", "-q", self.old)
        self.cache = self.root / "cache"
        self.env = dict(os.environ, CI_THIRD_PARTY_DIR=str(self.cache),
                        CI_THIRD_PARTY_SEED_DIR=str(self.seed),
                        CI_FORCE_CLEAN_CACHE="false", CI_SEED_THIRD_PARTY="true")

    @staticmethod
    def git(path, *args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    def prepare(self, **env):
        return subprocess.run(["bash", str(self.repo / "ci/prepare_ci_cache.sh")],
                              env=dict(self.env, **env), text=True, capture_output=True, check=True)

    def test_reseeding_does_not_restore_headers_removed_by_checkout(self):
        self.prepare()
        cached = self.cache / "opbase"
        self.git(cached, "checkout", "-q", self.new)
        (self.seed / "pkg").mkdir()
        (self.seed / "pkg/new.tar.gz").write_text("archive")
        result = self.prepare()
        self.assertIn("Build inputs unchanged", result.stdout)
        self.assertEqual(self.git(cached, "rev-parse", "HEAD"), self.new)
        self.assertEqual(self.git(cached, "status", "--porcelain"), "")
        self.assertFalse((cached / "pkg_inc/op_common/log/log.h").exists())
        self.assertEqual((cached / "include/op_common/log/log.h").read_text(), "new header\n")
        self.assertEqual((self.cache / "pkg/new.tar.gz").read_text(), "archive")

    def test_partial_opbase_is_replaced_before_seeding(self):
        self.prepare()
        cached = self.cache / "opbase"
        shutil.rmtree(cached / ".git")
        (cached / "stale.h").write_text("partial checkout")
        self.prepare()
        self.assertTrue((cached / ".git").is_dir())
        self.assertFalse((cached / "stale.h").exists())
        self.assertEqual(self.git(cached, "status", "--porcelain"), "")

    def test_force_clean_removes_existing_pollution(self):
        self.prepare()
        (self.cache / "opbase/stale.h").write_text("untracked")
        self.prepare(CI_FORCE_CLEAN_CACHE="true")
        self.assertFalse((self.cache / "opbase/stale.h").exists())
        self.assertEqual(self.git(self.cache / "opbase", "status", "--porcelain"), "")

    def test_images_seed_the_selected_opbase_revision(self):
        cmake = (ROOT / "cmake/third_party/opbase.cmake").read_text()
        revision = re.search(r"set\(OPBASE_TAG_ID\s+([0-9a-f]{40})\)", cmake).group(1)
        for name in ("Dockerfile", "Dockerfile.ascend950"):
            with self.subTest(dockerfile=name):
                text = (ROOT / "ci" / name).read_text()
                self.assertRegex(text, r'clone_at "https://gitcode.com/cann/opbase.git"\s*\\\s*' + revision)


if __name__ == "__main__":
    unittest.main()
