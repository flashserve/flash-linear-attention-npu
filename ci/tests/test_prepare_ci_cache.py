"""Exercise real Git dependency resolution without network or production caches."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from ci.prepare_ci_dependencies import read_dependencies


ROOT = Path(__file__).resolve().parents[2]


class PrepareCiCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fla-ci-deps-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "ci").mkdir(parents=True)
        (self.repo / "cmake/third_party").mkdir(parents=True)
        for name in ("prepare_ci_cache.sh", "prepare_ci_dependencies.py"):
            shutil.copy2(ROOT / "ci" / name, self.repo / "ci" / name)
        self.seed = self.root / "seed"
        self.cache = self.root / "cache"
        self.origins = {}
        self.revisions = {}
        for name in ("catlass", "opbase"):
            origin = self.root / "origins" / name
            origin.mkdir(parents=True)
            self.git(origin, "init", "-q")
            self.git(origin, "config", "user.name", "CI test")
            self.git(origin, "config", "user.email", "ci-test@example.invalid")
            marker = origin / "include/catlass/catlass.hpp"
            marker.parent.mkdir(parents=True)
            marker.write_text("old header\n")
            old_header = origin / "old-layout.h"
            old_header.write_text("old layout\n")
            self.git(origin, "add", ".")
            self.git(origin, "commit", "-qm", "old revision")
            old = self.git(origin, "rev-parse", "HEAD")
            marker.write_text("new header\n")
            self.git(origin, "mv", "old-layout.h", "new-layout.h")
            self.git(origin, "commit", "-qam", "new revision")
            new = self.git(origin, "rev-parse", "HEAD")
            self.origins[name] = origin
            self.revisions[name] = (old, new)
            shutil.copytree(origin, self.seed / name)
            self.git(self.seed / name, "checkout", "-q", old)
            self.pin(name, old)
        self.env = dict(os.environ, CI_THIRD_PARTY_DIR=self.cache.as_posix(),
                        CI_THIRD_PARTY_SEED_DIR=self.seed.as_posix(),
                        CI_FORCE_CLEAN_CACHE="false", CI_SEED_THIRD_PARTY="true")

    @staticmethod
    def git(path, *args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    def pin(self, name, revision, repository=None):
        repository = repository or self.origins[name].as_uri()
        if name == "catlass":
            text = (f'set(CATLASS_VENDOR_GIT_REPOSITORY "{repository}")\n'
                    f'set(CATLASS_VENDOR_GIT_TAG "{revision}")\n')
        else:
            text = (f"set(OPBASE_TAG_ID {revision})\n"
                    f'FetchContent_Declare(opbase\n  GIT_REPOSITORY "{repository}"\n'
                    f"  GIT_TAG {revision})\n")
        (self.repo / f"cmake/third_party/{name}.cmake").write_text(text)

    def prepare(self, check=True, **env):
        result = subprocess.run(["bash", (self.repo / "ci/prepare_ci_cache.sh").as_posix()],
                                env=dict(self.env, **env), text=True, capture_output=True,
                                timeout=90)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def assert_revision(self, name, revision):
        cached = self.cache / name
        self.assertEqual(self.git(cached, "rev-parse", "HEAD"), revision)
        self.assertEqual(self.git(cached, "status", "--porcelain"), "")
        return cached

    def disable_origins(self):
        for origin in self.origins.values():
            origin.rename(origin.with_name(origin.name + "-unavailable"))

    def test_matching_seed_works_without_network(self):
        self.disable_origins()
        result = self.prepare()
        self.assertIn("seed matches", result.stdout)
        for name, (old, _) in self.revisions.items():
            self.assert_revision(name, old)

    def test_changed_pin_fetches_new_revision_instead_of_restoring_old_seed(self):
        self.prepare()
        for name, (_, new) in self.revisions.items():
            self.pin(name, new)
        result = self.prepare()
        for name, (old, new) in self.revisions.items():
            cached = self.assert_revision(name, new)
            self.assertFalse((cached / "old-layout.h").exists())
            self.assertTrue((cached / "new-layout.h").exists())
            self.assertEqual(self.git(self.seed / name, "rev-parse", "HEAD"), old)
            self.assertEqual(self.git(self.seed / name, "status", "--porcelain"), "")
        self.assertIn("Build inputs changed", result.stdout)
        self.assertIn("no matching cache/seed; fetching", result.stdout)

    def test_matching_cache_ignores_old_seed_and_unavailable_network(self):
        for name, (_, new) in self.revisions.items():
            self.pin(name, new)
        self.prepare()
        self.disable_origins()
        (self.seed / "pkg").mkdir()
        (self.seed / "pkg/new.tar.gz").write_text("archive")
        result = self.prepare()
        self.assertIn("Build inputs unchanged", result.stdout)
        self.assertIn("reuse cache", result.stdout)
        for name, (_, new) in self.revisions.items():
            cached = self.assert_revision(name, new)
            self.assertFalse((cached / "old-layout.h").exists())
        self.assertEqual((self.cache / "pkg/new.tar.gz").read_text(), "archive")

    def test_cache_revision_is_checked_even_when_build_signature_is_unchanged(self):
        self.prepare()
        for name, (_, new) in self.revisions.items():
            self.git(self.cache / name, "checkout", "-q", new)
        result = self.prepare()
        self.assertIn("Build inputs unchanged", result.stdout)
        for name, (old, _) in self.revisions.items():
            cached = self.assert_revision(name, old)
            self.assertFalse((cached / "new-layout.h").exists())

    def test_newer_seed_does_not_restore_headers_after_legacy_checkout(self):
        for name, (_, new) in self.revisions.items():
            self.git(self.seed / name, "checkout", "-q", new)
        self.prepare()
        self.disable_origins()
        self.prepare()
        for name, (old, _) in self.revisions.items():
            cached = self.assert_revision(name, old)
            self.assertFalse((cached / "new-layout.h").exists())
            self.assertTrue((cached / "old-layout.h").exists())

    def test_partial_checkouts_without_git_are_replaced(self):
        self.prepare()
        for name in self.revisions:
            shutil.rmtree(self.cache / name / ".git")
            (self.cache / name / "stale.h").write_text("partial checkout")
        self.prepare()
        for name, (old, _) in self.revisions.items():
            cached = self.assert_revision(name, old)
            self.assertFalse((cached / "stale.h").exists())

    def test_dirty_cache_is_not_reused(self):
        self.prepare()
        (self.cache / "catlass/include/catlass/catlass.hpp").write_text("modified")
        (self.cache / "opbase/stale.h").write_text("untracked")
        self.prepare()
        for name, (old, _) in self.revisions.items():
            self.assert_revision(name, old)

    def test_dirty_seed_is_not_copied(self):
        (self.seed / "catlass/include/catlass/catlass.hpp").write_text("modified")
        (self.seed / "opbase/stale.h").write_text("untracked")
        result = self.prepare()
        self.assertIn("fetching", result.stdout)
        for name, (old, _) in self.revisions.items():
            self.assert_revision(name, old)
        self.assertTrue((self.seed / "opbase/stale.h").exists())

    def test_missing_catlass_marker_is_repaired(self):
        self.prepare()
        (self.cache / "catlass/include/catlass/catlass.hpp").unlink()
        self.prepare()
        self.assertTrue((self.cache / "catlass/include/catlass/catlass.hpp").is_file())

    def test_force_clean_removes_existing_pollution(self):
        self.prepare()
        (self.cache / "opbase/stale.h").write_text("untracked")
        self.prepare(CI_FORCE_CLEAN_CACHE="true")
        self.assert_revision("opbase", self.revisions["opbase"][0])

    def test_seeding_disabled_fetches_pinned_revisions(self):
        result = self.prepare(CI_SEED_THIRD_PARTY="false")
        self.assertIn("fetching", result.stdout)
        self.assertNotIn("seed matches", result.stdout)
        for name, (old, _) in self.revisions.items():
            self.assert_revision(name, old)

    def test_missing_seed_directory_fetches_pinned_revisions(self):
        self.prepare(CI_THIRD_PARTY_SEED_DIR=(self.root / "missing-seed").as_posix())
        for name, (old, _) in self.revisions.items():
            self.assert_revision(name, old)

    def test_fetch_failure_stops_instead_of_falling_back_to_old_seed(self):
        for name, (_, new) in self.revisions.items():
            self.pin(name, new)
        self.disable_origins()
        result = self.prepare(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dependency preparation failed", result.stderr)
        self.assertFalse((self.cache / "catlass").exists())
        self.assertEqual(list(self.cache.glob(".catlass-*")), [])

    def test_missing_remote_commit_stops_instead_of_using_another_revision(self):
        self.pin("catlass", "0" * 40)
        result = self.prepare(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.cache / "catlass").exists())

    def test_non_pinned_revision_is_rejected(self):
        self.pin("catlass", "main")
        result = self.prepare(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full pinned Git commit SHA", result.stderr)

    def test_legacy_catlass_declaration_uses_its_own_pin(self):
        old, new = self.revisions["catlass"]
        for revision in (old, new):
            (self.repo / "cmake/third_party/catlass.cmake").write_text(
                f'set(CATLASS_VENDOR_GIT_TAG\n "{revision}"\n'
                ' CACHE STRING "Catlass vendor git tag (fixed revision)")\n'
                'FetchContent_Declare(catlass_vendor\n'
                f' GIT_REPOSITORY {self.origins["catlass"].as_uri()}\n'
                f' GIT_TAG {revision})\n')
            self.prepare()
            self.assert_revision("catlass", revision)

    def test_reads_current_checkout_dependency_declarations(self):
        dependencies = read_dependencies(ROOT)
        self.assertEqual([item.name for item in dependencies], ["catlass", "opbase"])
        for item in dependencies:
            self.assertRegex(item.revision, r"^[0-9a-f]{40}$")
            self.assertEqual(item.repository, f"https://gitcode.com/cann/{item.name}.git")

    def test_images_seed_the_selected_opbase_revision(self):
        cmake = (ROOT / "cmake/third_party/opbase.cmake").read_text(encoding="utf-8")
        revision = re.search(r"set\(OPBASE_TAG_ID\s+([0-9a-f]{40})\)", cmake).group(1)
        for name in ("Dockerfile", "Dockerfile.ascend950"):
            with self.subTest(dockerfile=name):
                text = (ROOT / "ci" / name).read_text(encoding="utf-8")
                self.assertRegex(text, r'clone_at "https://gitcode.com/cann/opbase.git"\s*\\\s*' + revision)


if __name__ == "__main__":
    unittest.main()
