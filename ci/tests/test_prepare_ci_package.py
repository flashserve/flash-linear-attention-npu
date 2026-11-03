"""Package preparation behavior without CANN, pip network access or NPU builds."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_ci_package.py"
SPEC = importlib.util.spec_from_file_location("ci_prepare_package", SCRIPT)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)
TAG = next(iter(PACKAGE.parse_wheel_filename(
    "flash_linear_attention_npu_a2-0.dev20261102+main.1234567-py3-none-manylinux_2_34_aarch64.whl")[3]))


def wheel_fixture(directory, *, tier="a2", version="0.dev20261102+main.1234567",
                  soc="ascend910b", arch="aarch64", metadata_version=None):
    directory.mkdir(parents=True, exist_ok=True)
    distribution = f"flash_linear_attention_npu_{tier}"
    path = directory / f"{distribution}-{version}-py3-none-manylinux_2_34_{arch}.whl"
    vendor = "fla_npu/opp/vendors/fla_npu_transformer"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fla_npu/libfla_npu_stable.so", b"launcher")
        archive.writestr(f"{vendor}/op_api/lib/libcust_opapi.so", b"complete-base")
        archive.writestr(f"{vendor}/op_impl/ai_core/tbe/kernel/{soc}/op/kernel.o", b"kernel")
        archive.writestr(f"{vendor}/op_impl/ai_core/tbe/config/{soc}/binary_info_config.json", "{}")
        archive.writestr(f"{distribution}-{version}.dist-info/METADATA",
                          f"Name: flash-linear-attention-npu-{tier}\nVersion: {metadata_version or version}\n")
    return path


class FakePreparer(PACKAGE.PackagePreparer):
    """Simulate build/install effects while exercising the real orchestration."""
    def __init__(self, repo, environ, fail_overlay=False):
        super().__init__(repo, environ)
        self.commands = []
        self.fail_overlay = fail_overlay
        self.package_dir = repo / "installed/fla_npu"

    def run(self, args, *, cwd=None):
        args = [str(value) for value in args]
        self.commands.append((args, cwd, dict(self.env)))
        if "download" in args:
            wheel_fixture(Path(args[args.index("--dest") + 1]))
        elif "scripts/build_wheel.py" in args:
            wheel_fixture(Path(args[args.index("--wheel-dir") + 1]))
        elif "--pkg" in args:
            output = self.repo / "build_out"
            output.mkdir(exist_ok=True)
            (output / "fla_npu_linux-aarch64.run").write_text("# simulated run\n")
        elif any(value.startswith("--install-path=") for value in args):
            staging = Path(next(value.split("=", 1)[1] for value in args
                                if value.startswith("--install-path=")))
            (staging / "vendors").mkdir(parents=True)
        elif "--overlay" in args:
            if self.fail_overlay:
                raise subprocess.CalledProcessError(1, args)
            (self.package_dir / "opp/vendors/fla_npu_overlay_transformer").mkdir()
        elif "setup.py" in args:
            wheel_fixture(Path(args[args.index("--dist-dir") + 1]))
        elif "install" in args and "pip" in args:
            opp = self.package_dir / "opp"
            if opp.exists():
                shutil.rmtree(opp)
            vendor = opp / "vendors/fla_npu_transformer"
            (vendor / "bin").mkdir(parents=True)
            (vendor / "bin/set_env.bash").write_text("# base env\n")
            if "torch-wheel" not in args[-1]:
                (vendor / "op_api/lib").mkdir(parents=True)
                (vendor / "op_api/lib/libcust_opapi.so").write_bytes(b"complete-base")
                (vendor / "untouched-operator").write_bytes(b"keep-me")


class DailySelectionTests(unittest.TestCase):
    def validate(self, wheel, base_ref="main", soc="ascend910b", tier="a2"):
        with patch.object(PACKAGE, "sys_tags", return_value=iter([TAG])), \
                patch.object(PACKAGE.platform, "machine", return_value="aarch64"):
            return PACKAGE.validate_base_wheel(wheel, package=f"flash-linear-attention-npu-{tier}",
                                               base_ref=base_ref, soc=soc)

    def test_accepts_matching_main_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.validate(wheel_fixture(Path(tmp))), "0.dev20261102+main.1234567")

    def test_maintenance_selects_its_own_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp), version="26.10.0.dev20261102+1234567")
            self.assertEqual(self.validate(wheel, base_ref="v26.10.0"), "26.10.0.dev20261102+1234567")
            with self.assertRaisesRegex(ValueError, "target branch main"):
                self.validate(wheel)

    def test_a5_uses_a5_x86_wheel_and_soc(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp), tier="a5", soc="ascend950", arch="x86_64")
            compatible = iter(PACKAGE.parse_wheel_filename(wheel.name)[3])
            with patch.object(PACKAGE, "sys_tags", return_value=compatible), \
                    patch.object(PACKAGE.platform, "machine", return_value="x86_64"):
                self.assertEqual(PACKAGE.validate_base_wheel(
                    wheel, package="flash-linear-attention-npu-a5", base_ref="main", soc="ascend950"),
                    "0.dev20261102+main.1234567")

    def test_release_is_not_a_daily_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp), version="26.10.0")
            with self.assertRaisesRegex(ValueError, "target branch"):
                self.validate(wheel, base_ref="v26.10.0")

    def test_main_cannot_masquerade_as_maintenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp), version="26.10.0.dev20261102+main.1234567")
            with self.assertRaisesRegex(ValueError, "Main Daily base cannot"):
                self.validate(wheel, base_ref="v26.10.0")

    def test_wrong_product_tier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Expected flash-linear-attention-npu-a5"):
                self.validate(wheel_fixture(Path(tmp)), soc="ascend950", tier="a5")

    def test_wrong_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "incompatible with Python"):
                self.validate(wheel_fixture(Path(tmp), arch="x86_64"))

    def test_a_prebuilt_soc_package_cannot_claim_a_universal_cpu_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = wheel_fixture(Path(tmp))
            wheel = original.with_name(original.name.replace("manylinux_2_34_aarch64", "any"))
            original.rename(wheel)
            with patch.object(PACKAGE, "sys_tags", return_value=iter(PACKAGE.parse_wheel_filename(wheel.name)[3])), \
                    self.assertRaisesRegex(ValueError, "incompatible with Python"):
                PACKAGE.validate_base_wheel(wheel, package="flash-linear-attention-npu-a2",
                                            base_ref="main", soc="ascend910b")

    def test_correct_filename_with_wrong_soc_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "wrong SOC"):
                self.validate(wheel_fixture(Path(tmp), soc="ascend950"))

    def test_metadata_and_filename_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "METADATA disagree"):
                self.validate(wheel_fixture(Path(tmp), metadata_version="0.dev20261101+main.1234567"))

    def test_unknown_target_branch_fails_closed(self):
        for branch in ("", "master", "feature/a", "v26.10.0-other"):
            with self.subTest(branch=branch), self.assertRaisesRegex(ValueError, "No Daily base mapping"):
                PACKAGE.branch_version(branch)


class PackagePreparationTests(unittest.TestCase):
    def prepare(self, root, kind="base", **extra):
        env = {"CI_BUILD_KIND": kind, "CI_BASE_REF": "main", "CI_SOC": "ascend910b", "CI_JOBS": "16"}
        if kind == "incremental":
            env["CI_BUILD_OPS"] = "chunk_fwd_o,chunk_fwd_h"
        env.update(extra)
        preparer = FakePreparer(root, env)
        with patch.object(PACKAGE, "installed_package", return_value=preparer.package_dir), \
                patch.object(PACKAGE, "sys_tags", return_value=iter([TAG])), \
                patch.object(PACKAGE.platform, "machine", return_value="aarch64"):
            preparer.prepare()
        return preparer

    def test_base_downloads_branch_daily_and_runs_no_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            preparer = self.prepare(Path(tmp))
            commands = [entry[0] for entry in preparer.commands]
            self.assertEqual(len(commands), 2)
            download = commands[0]
            self.assertIn("--no-index", download)
            self.assertIn("--no-cache-dir", download)
            self.assertIn("https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/flash-linear-attention-npu-a2/latest/", download)
            self.assertEqual(download[-1], "flash-linear-attention-npu-a2==0.*")
            self.assertIn("FLA_NPU_ENV=", preparer.env_file.read_text())
            self.assertIn("FLA_NPU_OVERLAY_VENDOR=''", preparer.env_file.read_text())
            self.assertEqual(json.loads(preparer.info_file.read_text())["kind"], "base")

    def test_empty_container_options_keep_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            preparer = self.prepare(Path(tmp), CI_BASE_WHEEL_INDEX_URL="", CI_JOBS="", CI_SOC="",
                                    CI_PACKAGE_ENV_FILE="", CI_PACKAGE_INFO_FILE="")
            download = preparer.commands[0][0]
            self.assertIn(PACKAGE.DAILY_INDEX + "flash-linear-attention-npu-a2/latest/", download)
            self.assertEqual(preparer.jobs, "16")
            self.assertEqual(preparer.soc, "ascend910b")
            self.assertEqual(preparer.env_file, Path(tmp) / ".ci-tmp/npu-ci-package-env.sh")
            self.assertEqual(preparer.info_file, Path(tmp) / ".ci-tmp/npu-ci-package.json")

    def test_full_builds_once_with_root_build_wheel_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            preparer = self.prepare(Path(tmp), kind="full", FLA_NPU_OPS="old_operator", CI_JOBS="128")
            commands = [entry[0] for entry in preparer.commands]
            self.assertEqual(commands[0], ["bash", "ci/prepare_ci_cache.sh"])
            self.assertEqual(commands[1][1], "scripts/build_wheel.py")
            self.assertFalse(any("download" in command for command in commands))
            for _command, _cwd, env in preparer.commands:
                self.assertNotIn("FLA_NPU_OPS", env)
                self.assertEqual(env["FLA_NPU_BUILD_ARGS"], "-j16")

    def test_incremental_only_builds_selected_ops_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            preparer = self.prepare(Path(tmp), kind="incremental")
            commands = [entry[0] for entry in preparer.commands]
            build = next(command for command in commands if "--pkg" in command)
            self.assertIn("--ops=chunk_fwd_o,chunk_fwd_h", build)
            self.assertIn("-j16", build)
            self.assertFalse(any("setup.py" in command or "scripts/build_wheel.py" in command for command in commands))
            extract = next(command for command in commands
                           if any(arg.startswith("--install-path=") for arg in command))
            self.assertNotIn("--install", extract)
            self.assertTrue(any("fla_npu_opp_manage" in command and "--overlay" in command for command in commands))
            vendor = preparer.package_dir / "opp/vendors/fla_npu_transformer"
            self.assertEqual((vendor / "untouched-operator").read_bytes(), b"keep-me")
            self.assertIn("FLA_NPU_OVERLAY_VENDOR=fla_npu_overlay_transformer", preparer.env_file.read_text())

    def test_torch_rebuild_checks_abi_and_preserves_complete_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            cann = Path(tmp) / "fake-cann"
            (cann / "include/aclnnop").mkdir(parents=True)
            (cann / "aarch64-linux/include/aclnnop").mkdir(parents=True)
            preparer = self.prepare(Path(tmp), kind="torch", ASCEND_HOME_PATH=str(cann))
            commands = [entry[0] for entry in preparer.commands]
            abi_index = next(index for index, command in enumerate(commands) if any("op_abi_validate.py" in arg for arg in command))
            build_index = next(index for index, command in enumerate(commands) if "setup.py" in command)
            self.assertLess(abi_index, build_index)
            self.assertIn(str(cann / "include/aclnnop"), commands[abi_index])
            self.assertIn(str(cann / "aarch64-linux/include/aclnnop"), commands[abi_index])
            self.assertFalse(any("--pkg" in command or "build.sh" in command for command in commands))
            self.assertTrue(any(any("finalize_wheel_opp.py" in arg for arg in command) for command in commands))
            vendor = preparer.package_dir / "opp/vendors/fla_npu_transformer"
            self.assertEqual((vendor / "untouched-operator").read_bytes(), b"keep-me")
            self.assertEqual((vendor / "op_api/lib/libcust_opapi.so").read_bytes(), b"complete-base")

    def test_failure_never_leaves_a_success_env_or_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"CI_BUILD_KIND": "incremental", "CI_BUILD_OPS": "chunk_fwd_o", "CI_BASE_REF": "main"}
            preparer = FakePreparer(root, env, fail_overlay=True)
            preparer.env_file.parent.mkdir(parents=True)
            preparer.env_file.write_text("old-success")
            with patch.object(PACKAGE, "installed_package", return_value=preparer.package_dir), \
                    patch.object(PACKAGE, "sys_tags", return_value=iter([TAG])), \
                    patch.object(PACKAGE.platform, "machine", return_value="aarch64"), \
                    self.assertRaises(subprocess.CalledProcessError):
                preparer.prepare()
            self.assertFalse(preparer.env_file.exists())
            self.assertFalse(preparer.info_file.exists())

    def test_pinned_wrong_branch_is_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            preparer = FakePreparer(Path(tmp), {"CI_BUILD_KIND": "base", "CI_BASE_REF": "main",
                                               "CI_BASE_WHEEL_VERSION": "26.10.0.dev20261102+1234567"})
            with self.assertRaisesRegex(ValueError, "target branch main"):
                preparer.download_base(Path(tmp) / "wheel")
            self.assertEqual(preparer.commands, [])

    def test_invalid_or_ambiguous_build_inputs_are_rejected(self):
        inputs = [{"CI_BUILD_KIND": "unknown"}, {"CI_BUILD_KIND": "incremental"},
                  {"CI_BUILD_KIND": "incremental", "CI_BUILD_OPS": "op;echo bad"},
                  {"CI_BUILD_KIND": "base", "CI_BUILD_OPS": "chunk_fwd_o"},
                  {"CI_JOBS": "0"}, {"CI_SOC": "unknown"}]
        for values in inputs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PACKAGE.PackagePreparer(Path("unused"), values)


if __name__ == "__main__":
    unittest.main()
