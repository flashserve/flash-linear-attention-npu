import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import make_packages as pkg


class SharedImageBuildTests(unittest.TestCase):
    def exercise(self, build_rc=0, cleanup_fails=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, wheels = root / 'repo', root / 'wheels'
            repo.mkdir()
            calls, build_calls = [], []

            def docker(argv, **kwargs):
                calls.append(argv)
                if argv[:3] == ['docker', 'image', 'inspect']:
                    return subprocess.CompletedProcess(argv, 0, json.dumps([{'Id': 'sha256:immutable-ci-id'}]), '')
                if cleanup_fails:
                    raise subprocess.CalledProcessError(1, argv)
                return subprocess.CompletedProcess(argv, 0)

            def build(argv, *args, **kwargs):
                build_calls.append(argv)
                if not build_rc:
                    (wheels / 'flash_linear_attention_npu_a5-0.dev20261113+main.7065b0c-py3-none-manylinux_2_34_aarch64.whl').touch()
                return build_rc

            with patch.object(pkg.platform, 'machine', return_value='aarch64'), \
                 patch.object(pkg, 'docker_image', return_value='shared-ci:mutable'), \
                 patch.object(pkg, 'clean_build_state'), \
                 patch.object(pkg, 'wheel_check_version_args', return_value=''), \
                 patch.object(pkg.subprocess, 'run', side_effect=docker), \
                 patch.object(pkg, 'run_logged', side_effect=build):
                if cleanup_fails:
                    with self.assertRaises(subprocess.CalledProcessError):
                        pkg.build_local(repo, 'ascend950', wheels, root / 'log', arch='arm')
                elif build_rc:
                    with self.assertRaises(RuntimeError):
                        pkg.build_local(repo, 'ascend950', wheels, root / 'log', arch='arm')
                else:
                    result = pkg.build_local(repo, 'ascend950', wheels, root / 'log', arch='arm')
                    self.assertEqual(len(result), 1)
            self.assertIn('sha256:immutable-ci-id', build_calls[0])
            self.assertNotIn('shared-ci:mutable', build_calls[0])
            self.assertEqual(build_calls[0][build_calls[0].index('--user') + 1], '0:0')
            self.assertEqual(calls[-1][-2:], ['/workspace/repo', '/workspace/pkg-out'])
            self.assertIn('-Rh', calls[-1])
            self.assertIn('sha256:immutable-ci-id', calls[-1])

    def test_success_restores_ownership_using_pinned_image(self):
        self.exercise()

    def test_compile_failure_still_restores_ownership(self):
        self.exercise(build_rc=1)

    def test_ownership_failure_is_not_reported_as_success(self):
        self.exercise(cleanup_fails=True)


if __name__ == '__main__':
    unittest.main()
