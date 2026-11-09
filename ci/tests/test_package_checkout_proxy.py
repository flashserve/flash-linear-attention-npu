#!/usr/bin/env python3
"""Check the package workflow's direct checkout and proxy fallback."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/package.yml"


class PackageCheckoutProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["build"]
        cls.steps = cls.job["steps"]
        cls.direct, cls.proxy, cls.retry = cls.steps[:3]

    def test_direct_checkout_then_conditional_proxy_retry(self):
        self.assertEqual(self.direct["uses"], "actions/checkout@v4")
        self.assertTrue(self.direct["continue-on-error"])
        self.assertEqual(self.direct["env"]["no_proxy"], "*")
        for name in ("http_proxy", "https_proxy", "all_proxy",
                     "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
            self.assertEqual(self.direct["env"][name], "")
        condition = f"steps.{self.direct['id']}.outcome == 'failure'"
        self.assertEqual(self.proxy["if"], condition)
        self.assertEqual(self.retry["if"], condition)
        self.assertEqual(self.retry["uses"], self.direct["uses"])
        self.assertEqual(self.retry["with"], self.direct["with"])
        self.assertNotIn("continue-on-error", self.retry)
        self.assertNotIn("env", self.retry)  # Use the proxy saved in GITHUB_ENV.
        for step in (self.direct, self.retry):
            self.assertGreater(step["timeout-minutes"], 0)
        self.assertEqual(self.steps[3]["id"], "build")

    def run_proxy(self, directory, script):
        proxy_file = directory / "proxy.sh"
        if script is not None:
            proxy_file.write_text(script, encoding="utf-8")
        env_file = directory / "github-env"
        env = {
            **os.environ,
            "CI_PROXY_SCRIPT": str(proxy_file), "GITHUB_ENV": str(env_file),
            "http_proxy": "http://inherited.invalid", "NO_PROXY": "*",
        }
        result = subprocess.run(
            ["bash", "-c", self.proxy["run"]], env=env,
            capture_output=True, text=True, timeout=10,
        )
        persisted = dict(line.split("=", 1) for line in
                         env_file.read_text().splitlines()) if env_file.exists() else {}
        return result, persisted

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux Bash required")
    def test_proxyon_persists_runner_proxy_for_the_next_checkout(self):
        self.assertEqual(self.job["env"]["CI_PROXY_SCRIPT"],
                         "/home/huangjunzhe/ci/vpn-proxy.sh")
        for upper in (False, True):
            with self.subTest(upper=upper), tempfile.TemporaryDirectory() as temporary:
                http = "HTTP_PROXY" if upper else "http_proxy"
                https = "HTTPS_PROXY" if upper else "https_proxy"
                script = (f"proxyon() {{ export {http}=http://127.0.0.1:17890 "
                          f"{https}=http://127.0.0.1:17890 PROXY_HTTP_PORT=17890; }}\n")
                result, values = self.run_proxy(Path(temporary), script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["http_proxy"], "http://127.0.0.1:17890")
                self.assertEqual(values["https_proxy"], "http://127.0.0.1:17890")
                self.assertEqual(values["PROXY_HTTP_PORT"], "17890")
                self.assertEqual(values["no_proxy"], "")  # Clear direct-mode '*'.
                self.assertEqual(len(values), len({name.casefold() for name in values}))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux Bash required")
    def test_proxy_setup_failures_are_not_reported_as_checkout_success(self):
        for script in (None, "proxyon() { return 1; }\n",
                       "proxyon() { export http_proxy=$'bad\\nvalue'; }\n"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as temporary:
                result, _ = self.run_proxy(Path(temporary), script)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
