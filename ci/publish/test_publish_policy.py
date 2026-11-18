"""Offline policy tests: no network calls, Docker builds, or real uploads."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import daily_branches
import publish_from_manifest as publisher
import make_packages as driver


class PublishPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, arches, omit=None, version="26.10.0+main.devabcdef0"):
        results = []
        for soc, package in publisher.SOC_TO_PACKAGE.items():
            for arch in arches:
                if (soc, arch) == omit:
                    continue
                name = (f"{package}-{version}-py3-none-manylinux_2_34_"
                        f"{publisher.ARCH_TO_WHEEL[arch]}.whl")
                wheel = self.root / "wheels" / f"{soc}-{arch}" / name
                wheel.parent.mkdir(parents=True, exist_ok=True)
                wheel.write_bytes(b"offline-policy-fixture")
                results.append({"soc": soc, "arch": arch,
                                "status": "success", "wheels": [name]})
        data = {"arches": arches, "results": results, "head_sha": "a" * 40,
                "branch": "main", "build_type": "daily", "interrupted": ""}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, data

    def test_arm_all_platforms_pass_independently(self):
        path, _ = self.fixture(["arm"])
        _, wheels = publisher.validate_build(path)
        self.assertEqual(len(wheels), len(publisher.SOC_TO_PACKAGE))

    def test_x86_all_platforms_pass_independently(self):
        path, _ = self.fixture(["x86"])
        _, wheels = publisher.validate_build(path)
        self.assertEqual({w["arch"] for w in wheels}, {"x86"})

    def test_missing_platform_blocks(self):
        path, _ = self.fixture(["arm"], omit=("ascend910_93", "arm"))
        with self.assertRaisesRegex(RuntimeError, "缺少组合"):
            publisher.validate_build(path)

    def test_combined_manifest_requires_each_arch_complete(self):
        path, _ = self.fixture(["arm", "x86"], omit=("ascend950", "x86"))
        with self.assertRaisesRegex(RuntimeError, "缺少组合"):
            publisher.validate_build(path)

    def test_future_platform_in_registry_is_required(self):
        path, _ = self.fixture(["x86"])
        with patch.dict(publisher.SOC_TO_PACKAGE, {"test-future-soc": "test_a6"}):
            with self.assertRaisesRegex(RuntimeError, "test-future-soc/x86"):
                publisher.validate_build(path)

    def test_daily_cannot_be_relabelled_as_release(self):
        path, _ = self.fixture(["arm"])
        manifest, wheels = publisher.validate_build(path)
        args = argparse.Namespace(branch="main", build_type="release",
                                  release_version="26.10.0")
        with self.assertRaisesRegex(RuntimeError, "包类型不一致"):
            publisher.validate_identity(args, manifest, wheels)

    def test_exact_formal_release_version_passes(self):
        path, manifest = self.fixture(["arm"], version="26.10.0")
        manifest.update(build_type="release", release_version="26.10.0",
                        branch="v26.10.0")
        _, wheels = publisher.validate_build(path)
        args = argparse.Namespace(branch="v26.10.0", build_type="release",
                                  release_version="26.10.0")
        publisher.validate_identity(args, manifest, wheels)

    def test_history_daily_tag_is_not_formal(self):
        self.assertFalse(daily_branches.is_formal_release(
            {"tag_name": "v26.10.0-20261028-020000"}, "v26.10.0"))

    def test_draft_and_prerelease_do_not_stop_daily(self):
        for flag in ("draft", "prerelease"):
            self.assertFalse(daily_branches.is_formal_release(
                {"tag_name": "v26.10.0-release", flag: True}, "v26.10.0"))

    def test_published_branch_is_removed_even_if_explicitly_selected(self):
        def lookup(repo, tag, api):
            return {"tag_name": tag} if tag == "v26.9.1-release" else None
        with patch.object(daily_branches, "get_release", side_effect=lookup):
            branches = daily_branches.select_daily_branches(
                ["main", "v26.9.1", "v26.10.0"], "owner/repo", "https://example.test")
        self.assertEqual(branches, ["main", "v26.10.0"])

    def test_network_failure_is_not_treated_as_unreleased(self):
        with patch.object(daily_branches, "get_release", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                daily_branches.select_daily_branches(
                    ["v26.10.0"], "owner/repo", "https://example.test")

    def test_other_arch_release_description_is_preserved(self):
        arm = "<!-- fla-npu-arch:arm -->ARM<!-- /fla-npu-arch:arm -->"
        x86 = "<!-- fla-npu-arch:x86 -->X86<!-- /fla-npu-arch:x86 -->"
        merged = publisher.merge_release_body(arm, x86)
        self.assertIn(arm, merged)
        self.assertIn(x86, merged)
        self.assertEqual(publisher.merge_release_body(merged, x86).count("X86"), 1)

    def test_all_six_obs_paths_use_hyphenated_project_names(self):
        expected = {
            f"{publisher.PREFIX_ROOT}/flash-linear-attention-npu-a2",
            f"{publisher.PREFIX_ROOT}/flash-linear-attention-npu-a3",
            f"{publisher.PREFIX_ROOT}/flash-linear-attention-npu-a5",
            f"{publisher.PREFIX_ROOT}/nightly/flash-linear-attention-npu-a2",
            f"{publisher.PREFIX_ROOT}/nightly/flash-linear-attention-npu-a3",
            f"{publisher.PREFIX_ROOT}/nightly/flash-linear-attention-npu-a5",
        }
        actual = {
            publisher.obs_prefix(build_type, package)
            for build_type in ("daily", "release")
            for package in publisher.SOC_TO_PACKAGE.values()
        }
        self.assertEqual(actual, expected)

    def test_daily_has_only_obs_nightly_destination(self):
        self.assertEqual(publisher.publish_destinations("daily"),
                         ["obs-nightly"])

    def test_release_keeps_obs_and_github_release_destinations(self):
        self.assertEqual(publisher.publish_destinations("release"),
                         ["obs-release", "github-release"])

    def test_daily_rejects_github_release_arguments(self):
        argv = [
            "publish_from_manifest.py",
            "--manifest", str(self.root / "manifest.json"),
            "--build-type", "daily",
            "--branch", "main",
            "--gh-tag", "forbidden-daily-tag",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            publisher.parse_args()

    def test_build_jobs_are_forced_to_64(self):
        with patch.dict(driver.os.environ,
                        {"FLA_NPU_BUILD_ARGS": "-O3 -j256 --oom"}):
            driver.enforce_build_job_limit()
            self.assertEqual(driver.os.environ["FLA_NPU_BUILD_ARGS"],
                             "-O3 --oom -j64")

    def test_remote_x86_container_is_limited_to_64_cpus(self):
        script = driver.x86_build_script(
            "/repo", "/out", "ascend950", "image", "container", "")
        self.assertIn("docker run --rm --cpus 64", script)

    def test_incremental_daily_publish_uses_soc_nightly_prefix(self):
        wheel_dir = self.root / "wheels" / "ascend910b-arm"
        wheel_dir.mkdir(parents=True)
        (self.root / "obs_publish.py").write_text("# fixture\n",
                                                   encoding="utf-8")
        completed = argparse.Namespace(returncode=0)
        with patch.dict(driver.os.environ, {
                "FLA_PKG_INCREMENTAL_OBS": "1",
                "FLA_PKG_BUILD_TYPE": "daily"}), \
                patch.object(driver, "PACKAGE_DIR", self.root), \
                patch.object(driver.subprocess, "run",
                             return_value=completed) as run:
            result = driver.publish_incremental_obs(
                "ascend910b", "arm", wheel_dir, self.root)
        self.assertEqual(run.call_count, 2)
        command = run.call_args_list[0].args[0]
        latest_command = run.call_args_list[1].args[0]
        self.assertEqual(result["status"], "success")
        self.assertIn("--sync", command)
        prefix = command[command.index("--prefix") + 1]
        self.assertEqual(
            prefix,
            "pypi/packages/ascend/repos/pypi/nightly/"
            "flash-linear-attention-npu-a2")
        self.assertEqual(latest_command[latest_command.index('--prefix') + 1], prefix + '/latest')

    def test_incremental_publish_is_disabled_for_release(self):
        with patch.dict(driver.os.environ, {
                "FLA_PKG_INCREMENTAL_OBS": "1",
                "FLA_PKG_BUILD_TYPE": "release"}), \
                patch.object(driver.subprocess, "run") as run:
            result = driver.publish_incremental_obs(
                "ascend910b", "arm", self.root, self.root)
        self.assertEqual(result["status"], "disabled")
        run.assert_not_called()

    def test_local_x86_uses_ci_user_and_preserves_release_container_settings(self):
        wheel_dir = self.root / "local-x86-out"
        wheel_dir.mkdir()
        wheel = wheel_dir / "flash_linear_attention_npu_a2-26.10.0-py3-none-manylinux_2_34_x86_64.whl"
        wheel.write_bytes(b"fixture")
        with patch.object(driver.platform, "machine", return_value="x86_64"), \
                patch.object(driver.os, "getuid", return_value=1000, create=True), \
                patch.object(driver.os, "getgid", return_value=1000, create=True), \
                patch.object(driver, "clean_build_state"), \
                patch.object(driver.subprocess, "run") as inspect, \
                patch.object(driver, "image_digest", return_value="test-image"), \
                patch.object(driver, "run_logged", return_value=0) as run, \
                patch.dict(driver.os.environ, {"FLA_PKG_BUILD_TYPE": "release",
                    "FLA_NPU_DISABLE_LOCAL_VERSION": "TRUE",
                    "FLA_NPU_BUILD_OFFLINE_BUNDLE": "TRUE"}):
            inspect.return_value.returncode = 0
            inspect.return_value.stdout = json.dumps([{"Id": "sha256:test-ci-image"}])
            result = driver.build_local(self.root, "ascend910b", wheel_dir,
                                        self.root / "local.log", arch="x86")
        self.assertEqual(result, [wheel])
        command = run.call_args.args[0]
        self.assertIn("--user", command)
        self.assertEqual(command[command.index("--user") + 1], "0:0")
        self.assertIn("sha256:test-ci-image", command)
        cleanup = inspect.call_args.args[0]
        self.assertIn("-Rh", cleanup)
        self.assertIn("1000:1000", cleanup)
        self.assertEqual(command[command.index("--cpus") + 1], "64")
        self.assertIn("HOME=/tmp", command)
        self.assertIn("FLA_NPU_DISABLE_LOCAL_VERSION=TRUE", command)
        self.assertNotIn("--allow-local-version", command[-1])


if __name__ == "__main__":
    unittest.main()
