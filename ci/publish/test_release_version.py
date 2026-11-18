import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import make_packages as driver


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'fla').mkdir()
        self.file = self.root/'fla/__init__.py'

    def test_matching_dev0_changes_only_version_and_is_idempotent(self):
        before = '__version__ = "26.10.0.dev0"  # marker\nOTHER = 1\n'
        self.file.write_text(before)
        self.assertEqual(driver.prepare_release_version(self.root, '26.10.0'), '26.10.0.dev0')
        self.assertEqual(self.file.read_text(), before.replace('26.10.0.dev0', '26.10.0'))
        self.assertEqual(driver.prepare_release_version(self.root, '26.10.0'), '26.10.0')

    def test_reject_other_release_prerelease_and_ambiguous_markers(self):
        for version in ('26.9.1.dev0', '26.11.0.dev0', '26.10.0.dev1', '26.10.0rc1', '26.10.0+foo'):
            with self.subTest(version=version):
                before = f"__version__ = '{version}'\n"
                self.file.write_text(before)
                with self.assertRaises(RuntimeError):
                    driver.prepare_release_version(self.root, '26.10.0')
                self.assertEqual(self.file.read_text(), before)
        self.file.write_text('__version__ = "26.10.0.dev0"\n__version__ = "26.10.0.dev0"\n')
        with self.assertRaises(RuntimeError):
            driver.prepare_release_version(self.root, '26.10.0')

    def test_reject_nonformal_target(self):
        self.file.write_text('__version__ = "26.10.0.dev0"\n')
        with self.assertRaises(RuntimeError):
            driver.prepare_release_version(self.root, '26.10.0.dev0')


if __name__ == '__main__':
    unittest.main()
