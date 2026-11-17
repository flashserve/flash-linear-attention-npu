"""Exercise Catlass selection and cached sources without network access."""

from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "https://gitcode.com/cann/catlass.git"
OLD_REVISION = "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
NEW_REVISION = "86acf1db511ee727d6b2995dd1b41f7f4f9221a8"
OLD_ARCH_HASH = "f9f730ef9dee785d90fb4a824421a3d514eb44d333abc7ec29efc44e40aa44a1"
NEW_ARCH_HASH = "ae634315dbedfa1f0b66c808caf2763573b801dd0ddba52f2611e1e5a237bca8"


class CatlassDependencyTest(unittest.TestCase):
    def test_a2_a5_images_cache_the_old_revision(self):
        files = (
            ROOT / "ci/Dockerfile",
            ROOT / "ci/Dockerfile.ascend950",
        )
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertIn(OFFICIAL_REPOSITORY, text)
                self.assertIn(OLD_REVISION, text)
                self.assertNotIn(NEW_REVISION, text)

@unittest.skipUnless(shutil.which("cmake") and shutil.which("git"), "requires CMake and Git")
class CatlassConfigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fla-catlass-")
        cls.root = Path(cls.temporary.name)
        cls.origin = cls.root / "origin"
        cls.origin.mkdir()
        cls.git(cls.origin, "init", "--quiet")
        cls.git(cls.origin, "config", "user.name", "Catlass test")
        cls.git(cls.origin, "config", "user.email", "test@example.invalid")
        cls.git(cls.origin, "config", "core.autocrlf", "false")
        include = cls.origin / "include/catlass"
        (include / "arch").mkdir(parents=True)
        (include / "catlass.hpp").write_text("// test header\n", encoding="utf-8")
        cls.revisions, cls.headers = {}, {}
        for label in ("old", "new"):
            data = f"// {label} architecture definitions\n".encode()
            (include / "arch/arch.hpp").write_bytes(data)
            cls.headers[label] = data
            cls.git(cls.origin, "add", "include")
            cls.git(cls.origin, "commit", "--quiet", "-m", label)
            cls.revisions[label] = cls.git(cls.origin, "rev-parse", "HEAD")
        # Run the production module against two tiny local commits.
        cls.module = (ROOT / "cmake/third_party/catlass.cmake").read_text(encoding="utf-8")
        for original, replacement in (
            (OLD_REVISION, cls.revisions["old"]), (NEW_REVISION, cls.revisions["new"]),
            (OLD_ARCH_HASH, hashlib.sha256(cls.headers["old"]).hexdigest()),
            (NEW_ARCH_HASH, hashlib.sha256(cls.headers["new"]).hexdigest()),
            (OFFICIAL_REPOSITORY, cls.origin.as_uri()),
        ):
            if original not in cls.module:
                raise AssertionError(f"Missing production pin: {original}")
            cls.module = cls.module.replace(original, replacement)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def git(directory, *args):
        result = subprocess.run(
            ["git", "-C", str(directory), *args], text=True, encoding="utf-8", check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.cache = self.case / "third_party"
        self.source = self.case / "source"
        self.source.mkdir()
        (self.source / "catlass.cmake").write_text(self.module, encoding="utf-8")
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.16)\nproject(catlass_selection NONE)\n'
            'include("${CMAKE_CURRENT_LIST_DIR}/catlass.cmake")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${CATLASS_VENDOR_GIT_TAG}")\n',
            encoding="utf-8",
        )

    def configure(self, soc, *, success=True, extra=()):
        command = ["cmake", "-S", str(self.source), "-B", str(self.case / "build")]
        if shutil.which("ninja"):
            command += ["-G", "Ninja"]
        command += [f"-DCANN_3RD_LIB_PATH={self.cache.as_posix()}", f"-DASCEND_COMPUTE_UNIT={soc}", *extra]
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
            return (self.case / "build/selected.txt").read_text(encoding="utf-8")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def clone_cache(self, *, shallow=False):
        self.cache.mkdir()
        args = ["clone", "--quiet"] + (["--depth", "1"] if shallow else [])
        self.git(self.case, *args, self.origin.as_uri(), str(self.cache / "catlass"))

    def offline_cache(self, revision):
        include = self.cache / "catlass/include/catlass"
        (include / "arch").mkdir(parents=True)
        (include / "catlass.hpp").write_text("// offline header\n", encoding="utf-8")
        (include / "arch/arch.hpp").write_bytes(self.headers[revision])

    def test_case_insensitive_selection_and_reconfiguration(self):
        for soc, expected in (
            ("ascend950", "old"), ("ascend960dt", "new"), ("ASCEND960DT", "new"),
            ("Ascend960Dt", "new"), ("ASCEND950", "old"),
            ("ascend910b;ascend910_93", "old"), ("ascend310p", "old"),
        ):
            with self.subTest(soc=soc):
                self.assertEqual(self.configure(soc), self.revisions[expected])
                self.assertEqual(self.git(self.cache / "catlass", "rev-parse", "HEAD"), self.revisions[expected])

    def test_stale_cmake_cache_cannot_override_selection(self):
        self.assertEqual(self.configure("ascend950", extra=(
            f"-DCATLASS_VENDOR_GIT_TAG={self.revisions['new']}",
        )), self.revisions["old"])

    def test_missing_revision_is_fetched(self):
        self.clone_cache(shallow=True)
        self.assertEqual(self.configure("ascend950"), self.revisions["old"])

    def test_disconnected_missing_revision_is_reported(self):
        self.clone_cache(shallow=True)
        output = self.configure("ascend950", success=False, extra=("-DFETCHCONTENT_FULLY_DISCONNECTED=ON",))
        self.assertIn("not cached for offline use", " ".join(output.split()))

    def test_dirty_cache_is_preserved(self):
        self.clone_cache()
        header = self.cache / "catlass/include/catlass/catlass.hpp"
        header.write_text("// local changes\n", encoding="utf-8")
        self.assertIn("local changes", self.configure("ascend950", success=False))
        self.assertEqual(header.read_text(encoding="utf-8"), "// local changes\n")
        self.assertEqual(self.git(self.cache / "catlass", "rev-parse", "HEAD"), self.revisions["new"])

    def test_matching_offline_bundle(self):
        self.offline_cache("old")
        self.assertEqual(self.configure("ASCEND950"), self.revisions["old"])

    def test_wrong_offline_bundle_is_rejected(self):
        self.offline_cache("new")
        self.assertIn("Offline Catlass headers do not match", self.configure("ascend950", success=False))

    def test_mixed_revision_soc_list_is_rejected(self):
        for soc in ("ASCEND960DT;ascend950", "ascend950,Ascend960Dt"):
            with self.subTest(soc=soc):
                self.assertIn("Build ascend960dt separately", self.configure(soc, success=False))


if __name__ == "__main__":
    unittest.main()
