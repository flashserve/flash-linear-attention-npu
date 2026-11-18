import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import prepare_publish_dependencies as prep
import make_packages as pkg


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.out = self.root / "out"
        (self.repo / "cmake/third_party").mkdir(parents=True)
        (self.repo / "scripts/tools").mkdir(parents=True)
        self.pins = {}
        for name in ("catlass", "opbase"):
            origin = self.root / name
            origin.mkdir()
            git(origin, "init", "-q")
            git(origin, "config", "user.email", "test@localhost")
            git(origin, "config", "user.name", "test")
            marker = origin / ("include/catlass/catlass.hpp" if name == "catlass" else "marker.txt")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("fixture")
            git(origin, "add", ".")
            git(origin, "commit", "-qm", "fixture")
            self.pins[name] = git(origin, "rev-parse", "HEAD")
        (self.repo / "cmake/third_party/catlass.cmake").write_text(
            f'set(CATLASS_VENDOR_GIT_REPOSITORY "{self.root / "catlass"}")\n'
            f'set(CATLASS_VENDOR_GIT_TAG "{self.pins["catlass"]}" CACHE STRING "pin")\n')
        (self.repo / "cmake/third_party/opbase.cmake").write_text(
            f'set(OPBASE_TAG_ID {self.pins["opbase"]})\nGIT_REPOSITORY "{self.root / "opbase"}"\n')
        self.archive = self.root / "lib-1.tar.gz"
        with tarfile.open(self.archive, "w:gz") as bundle:
            info = tarfile.TarInfo("lib/header.h")
            info.size = 5
            bundle.addfile(info, io.BytesIO(b"hello"))
        (self.repo / "scripts/tools/third_lib_download.py").write_text(f"my_urls = {[self.archive.as_uri()]!r}")
        (self.repo / "scripts/tools/prepare_offline_bundle.py").write_text(f"_ARCHIVES = {[self.archive.name]!r}")
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@localhost")
        git(self.repo, "config", "user.name", "test")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "fixture")

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, fresh=False, seed=None):
        return prep.prepare(self.repo, self.out, seed, fresh)

    def test_complete_set_reused_and_fresh_mode_downloads_once(self):
        first = self.run_prepare()
        self.assertEqual(first["archives"][0]["source"], "download")
        with patch.object(prep.compat, "fetch", side_effect=AssertionError("unexpected Git fetch")), \
             patch.object(prep.urllib.request, "urlopen", side_effect=AssertionError("unexpected download")):
            second = self.run_prepare()
        self.assertTrue(all(x["source"] == "cache" for x in second["git"] + second["archives"]))
        with patch.object(prep.compat, "fetch", wraps=prep.compat.fetch) as fetch, \
             patch.object(prep.urllib.request, "urlopen", wraps=prep.urllib.request.urlopen) as download:
            third = self.run_prepare(fresh=True)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(download.call_count, 1)
        self.assertTrue(all(x["source"] == "download" for x in third["git"] + third["archives"]))

    def test_wrong_git_seed_and_corrupt_archive_replaced(self):
        seed = self.root / "seed"
        seed.mkdir()
        shutil.copytree(self.root / "catlass", seed / "catlass")
        git(seed / "catlass", "commit", "--allow-empty", "-qm", "wrong revision")
        self.run_prepare(seed=seed)
        cached = self.repo / "third_party/pkg" / self.archive.name
        cached.write_bytes(b"corrupt")
        result = self.run_prepare(seed=seed)
        self.assertEqual(result["archives"][0]["source"], "download")
        self.assertEqual(git(self.repo / "third_party/catlass", "rev-parse", "HEAD"), self.pins["catlass"])

    def test_partial_failure_keeps_previous_complete_set(self):
        self.run_prepare()
        before = (self.repo / "third_party" / prep.REPORT).read_bytes()
        with patch.object(prep.urllib.request, "urlopen", side_effect=OSError("offline")), \
             patch.object(prep.time, "sleep"):
            with self.assertRaises(OSError):
                self.run_prepare(fresh=True)
        self.assertEqual((self.repo / "third_party" / prep.REPORT).read_bytes(), before)

    def test_old_opbase_declaration_uses_toolkit_api(self):
        path = self.repo / "cmake/third_party/opbase.cmake"
        new = "f" * 40
        path.write_text(f"set(OPBASE_LEGACY_TAG_ID {self.pins['opbase']})\n"
                        f"set(OPBASE_CONST_API_TAG_ID {new})\n"
                        f'GIT_REPOSITORY "{self.root / "opbase"}"\n')
        toolkit = self.root / "cann"
        header = toolkit / (prep.platform.machine() + "-linux/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h")
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": str(toolkit)}):
            self.assertEqual(prep.git_requirements(self.repo, prep.compat)[1].revision, self.pins["opbase"])
            header.parent.mkdir(parents=True)
            header.write_text("int64_t GetBlockDim() const;")
            self.assertEqual(prep.git_requirements(self.repo, prep.compat)[1].revision, new)

    def test_normal_entrypoints_have_one_preparation_step(self):
        for fresh in ("0", "1"):
            with patch.dict(os.environ, {"FLA_PKG_NO_CACHE": fresh, "CI_SEED_THIRD_PARTY": "true"}):
                command = pkg.dependency_preparation_command()
                self.assertEqual(command.count("prepare_publish_dependencies.py"), 1)
                self.assertEqual("--fresh" in command, fresh == "1")
                remote = pkg.x86_build_script("/tmp/job/repo", "/tmp/out", "ascend950", "image", "ctr", "")
                self.assertIn(command, remote)
                self.assertNotIn("fresh_dependencies.py", remote)
                self.assertNotIn("prepare_ci_cache.sh", remote)

    def test_next_soc_clears_all_previous_build_outputs(self):
        for name in ("build", "build_out", "output", "third_party"):
            (self.repo / name).mkdir()
            (self.repo / name / "old-platform-result").touch()
        pkg.clean_build_state(self.repo)
        for name in ("build", "build_out", "output", "third_party"):
            self.assertFalse((self.repo / name).exists())


if __name__ == "__main__":
    unittest.main()
