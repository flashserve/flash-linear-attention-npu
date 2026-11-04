"""Package preparation behavior without CANN, pip network access or NPU builds."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
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


def rewrite_wheel_paths(wheel, remap):
    """Keep payload bytes while modeling the paths a wheel installer will spread."""
    with zipfile.ZipFile(wheel) as archive:
        entries = [(entry.filename, archive.read(entry)) for entry in archive.infolist()]
    with zipfile.ZipFile(wheel, "w") as archive:
        for original, data in entries:
            for replacement in remap(original):
                archive.writestr(replacement, data)
    return wheel


def data_prefix(wheel, scheme="purelib"):
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
    return metadata[:-len(".dist-info/METADATA")] + f".data/{scheme}/"


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
        with patch.object(PACKAGE, "sys_tags", side_effect=lambda: iter([TAG])), \
                patch.object(PACKAGE.platform, "machine", return_value="aarch64"):
            return PACKAGE.validate_base_wheel(wheel, package=f"flash-linear-attention-npu-{tier}",
                                               base_ref=base_ref, soc=soc)

    def test_accepts_matching_main_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.validate(wheel_fixture(Path(tmp))), "0.dev20261102+main.1234567")

    def test_accepts_payload_in_its_own_purelib_or_platlib_scheme(self):
        for scheme in ("purelib", "platlib"):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as tmp:
                wheel = wheel_fixture(Path(tmp))
                prefix = data_prefix(wheel, scheme)
                rewrite_wheel_paths(wheel, lambda name: [prefix + name] if name.startswith("fla_npu/") else [name])
                self.assertEqual(self.validate(wheel), "0.dev20261102+main.1234567")

    def test_other_data_stems_or_non_library_schemes_cannot_supply_assets(self):
        for scheme in ("wrong-stem", "headers", "scripts", "data"):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as tmp:
                wheel = wheel_fixture(Path(tmp))
                prefix = ("other_distribution-0.dev20261102.data/purelib/" if scheme == "wrong-stem"
                          else data_prefix(wheel, scheme))
                rewrite_wheel_paths(wheel, lambda name: [prefix + name] if name.startswith("fla_npu/") else [name])
                error = "foreign data directory" if scheme == "wrong-stem" else "incomplete.*libfla_npu_stable.so"
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(wheel)

    def test_assets_scattered_across_invalid_prefixes_do_not_form_a_complete_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            valid_prefix = data_prefix(wheel)
            headers_prefix = data_prefix(wheel, "headers")

            def remap(name):
                if name.endswith("libfla_npu_stable.so"):
                    return [valid_prefix + name]
                if name.endswith("libcust_opapi.so"):
                    return ["other_distribution-0.dev20261102.data/headers/" + name]
                if "/config/" in name:
                    return [headers_prefix + name]
                return [name]

            rewrite_wheel_paths(wheel, remap)
            with self.assertRaises(ValueError) as error:
                self.validate(wheel)
            self.assertIn("op_api/lib/libcust_opapi.so", str(error.exception))
            self.assertIn("ascend910b config", str(error.exception))

    def test_data_layout_does_not_relax_missing_opapi_or_wrong_soc_checks(self):
        for missing in ("opapi", "soc"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                wheel = wheel_fixture(Path(tmp), soc="ascend950" if missing == "soc" else "ascend910b")
                prefix = data_prefix(wheel)

                def remap(name):
                    if missing == "opapi" and name.endswith("libcust_opapi.so"):
                        return []
                    return [prefix + name] if name.startswith("fla_npu/") else [name]

                rewrite_wheel_paths(wheel, remap)
                with self.assertRaisesRegex(ValueError, "incomplete|wrong SOC"):
                    self.validate(wheel)

    def test_root_and_data_duplicate_install_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            prefix = data_prefix(wheel)
            rewrite_wheel_paths(wheel, lambda name: [name, prefix + name]
                                if name.endswith("libcust_opapi.so") else [name])
            with self.assertRaisesRegex(ValueError, "conflicting.*libcust_opapi.so"):
                self.validate(wheel)

    def test_foreign_data_cannot_overwrite_an_otherwise_complete_valid_package(self):
        for layout in ("root", "purelib", "platlib"):
            for scheme in ("purelib", "platlib"):
                with self.subTest(layout=layout, foreign_scheme=scheme), tempfile.TemporaryDirectory() as tmp:
                    wheel = wheel_fixture(Path(tmp))
                    valid_prefix = "" if layout == "root" else data_prefix(wheel, layout)
                    foreign_prefix = f"other_distribution-0.dev20261102.data/{scheme}/"

                    def remap(name):
                        if not name.startswith("fla_npu/"):
                            return [name]
                        paths = [valid_prefix + name]
                        if name.endswith("libcust_opapi.so"):
                            paths.append(foreign_prefix + name)
                        return paths

                    rewrite_wheel_paths(wheel, remap)
                    with self.assertRaisesRegex(ValueError, "foreign data directory.*libcust_opapi.so"):
                        self.validate(wheel)

    def test_invalid_zip_aliases_cannot_overwrite_a_complete_valid_package(self):
        aliases = ("./fla_npu/libfla_npu_stable.so", "other/../fla_npu/libfla_npu_stable.so",
                   "/fla_npu/libfla_npu_stable.so", "fla_npu//libfla_npu_stable.so",
                   "fla_npu\\libfla_npu_stable.so", "C:/fla_npu/libfla_npu_stable.so",
                   "unrelated//extra.txt", "./fla_npu/", "fla_npu//")
        for alias in aliases:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                wheel = wheel_fixture(Path(tmp))
                with zipfile.ZipFile(wheel, "a") as archive:
                    entry = zipfile.ZipInfo(alias)
                    # Preserve raw backslashes even on Windows, where ZipInfo
                    # otherwise changes them before the bytes reach the ZIP.
                    entry.filename = alias
                    archive.writestr(entry, b"replacement")
                with self.assertRaisesRegex(ValueError, "invalid ZIP path"):
                    self.validate(wheel)

    def test_data_scheme_aliases_are_checked_before_install_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            prefix = data_prefix(wheel)
            alias = prefix + "./fla_npu/libfla_npu_stable.so"
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr(alias, b"replacement")
            with self.assertRaisesRegex(ValueError, "invalid ZIP path"):
                self.validate(wheel)

    def test_regular_directory_entries_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("fla_npu/", b"")
                archive.writestr(data_prefix(wheel) + "fla_npu/", b"")
            self.assertEqual(self.validate(wheel), "0.dev20261102+main.1234567")

    def test_purelib_platlib_conflicts_are_rejected_even_with_other_assets_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            purelib, platlib = data_prefix(wheel), data_prefix(wheel, "platlib")

            def remap(name):
                if name.endswith("libfla_npu_stable.so"):
                    return [purelib + name, platlib + name]
                if name.endswith("libcust_opapi.so"):
                    return []
                return [name]

            rewrite_wheel_paths(wheel, remap)
            with self.assertRaisesRegex(ValueError, "conflicting.*libfla_npu_stable.so"):
                self.validate(wheel)

    def test_repeated_zip_entry_is_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            with self.assertWarns(UserWarning):
                rewrite_wheel_paths(wheel, lambda name: [name, name]
                                    if name.endswith("libcust_opapi.so") else [name])
            with self.assertRaisesRegex(ValueError, "conflicting.*libcust_opapi.so"):
                self.validate(wheel)

    def test_directory_entry_cannot_count_as_soc_kernel_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            prefix = data_prefix(wheel)
            rewrite_wheel_paths(wheel, lambda name: [prefix + name + ("/" if "/kernel/" in name else "")]
                                if name.startswith("fla_npu/") else [name])
            with self.assertRaisesRegex(ValueError, "ascend910b kernel"):
                self.validate(wheel)

    def test_foreign_or_nested_metadata_directory_is_rejected(self):
        for replacement in ("other_distribution-0.dev20261102.dist-info/METADATA",
                            "nested/flash_linear_attention_npu_a2-0.dev20261102+main.1234567.dist-info/METADATA"):
            with self.subTest(metadata=replacement), tempfile.TemporaryDirectory() as tmp:
                wheel = wheel_fixture(Path(tmp))
                rewrite_wheel_paths(wheel, lambda name: [replacement] if name.endswith(".dist-info/METADATA") else [name])
                with self.assertRaisesRegex(ValueError, "METADATA"):
                    self.validate(wheel)

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

    def test_corrupt_payload_is_rejected_before_cache_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = wheel_fixture(Path(tmp))
            payload = wheel.read_bytes()
            wheel.write_bytes(payload.replace(b"complete-base", b"damaged--base", 1))
            with self.assertRaisesRegex((ValueError, zipfile.BadZipFile), "corrupt|CRC"):
                self.validate(wheel)

    def test_unknown_target_branch_fails_closed(self):
        for branch in ("", "master", "feature/a", "v26.10.0-other"):
            with self.subTest(branch=branch), self.assertRaisesRegex(ValueError, "No Daily base mapping"):
                PACKAGE.branch_version(branch)


class PackagePreparationTests(unittest.TestCase):
    def prepare(self, root, kind="base", **extra):
        env = {"CI_BUILD_KIND": kind, "CI_BASE_REF": "main", "CI_SOC": "ascend910b", "CI_JOBS": "16",
               "CI_PACKAGE_CACHE": str(root / "pkg-cache")}
        if kind == "incremental":
            env["CI_BUILD_OPS"] = "chunk_fwd_o,chunk_fwd_h"
        env.update(extra)
        preparer = FakePreparer(root, env)
        with patch.object(PACKAGE, "installed_package", return_value=preparer.package_dir), \
                patch.object(PACKAGE, "sys_tags", side_effect=lambda: iter([TAG])), \
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
            self.assertFalse(json.loads(preparer.info_file.read_text())["cache"]["hit"])

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
            preparer = self.prepare(Path(tmp), kind="full", FLA_NPU_OPS="old_operator", CI_JOBS="128",
                                    CI_UPDATE_PACKAGE="true")
            commands = [entry[0] for entry in preparer.commands]
            self.assertEqual(commands[0], ["bash", "ci/prepare_ci_cache.sh"])
            self.assertEqual(commands[1][1], "scripts/build_wheel.py")
            self.assertFalse(any("download" in command for command in commands))
            self.assertFalse((Path(tmp) / "pkg-cache").exists())
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
            env = {"CI_BUILD_KIND": "incremental", "CI_BUILD_OPS": "chunk_fwd_o", "CI_BASE_REF": "main",
                   "CI_PACKAGE_CACHE": str(root / "pkg-cache")}
            preparer = FakePreparer(root, env, fail_overlay=True)
            preparer.env_file.parent.mkdir(parents=True)
            preparer.env_file.write_text("old-success")
            with patch.object(PACKAGE, "installed_package", return_value=preparer.package_dir), \
                    patch.object(PACKAGE, "sys_tags", side_effect=lambda: iter([TAG])), \
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
                  {"CI_JOBS": "0"}, {"CI_SOC": "unknown"}, {"CI_UPDATE_PACKAGE": "yes"}]
        for values in inputs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PACKAGE.PackagePreparer(Path("unused"), values)


class DailyCacheTests(unittest.TestCase):
    NOW = datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.environ = {"CI_BUILD_KIND": "base", "CI_BASE_REF": "main", "CI_SOC": "ascend910b",
                        "CI_PACKAGE_CACHE": str(self.root / "pkg-cache")}
        self.tags = patch.object(PACKAGE, "sys_tags", side_effect=lambda: iter([TAG]))
        self.machine = patch.object(PACKAGE.platform, "machine", return_value="aarch64")
        self.clock = patch.object(PACKAGE, "utc_now", return_value=self.NOW)
        self.tags.start()
        self.machine.start()
        self.now = self.clock.start()
        self.addCleanup(self.tags.stop)
        self.addCleanup(self.machine.stop)
        self.addCleanup(self.clock.stop)

    def cached(self, *, force=False, **extra):
        preparer = FakePreparer(self.root, dict(self.environ, **extra))
        wheel, version, url = preparer.cached_base(self.root / "run-wheel", force=force)
        self.assertTrue(wheel.is_file())
        self.assertEqual(version, "0.dev20261102+main.1234567")
        self.assertTrue(url.startswith("https://"))
        return preparer

    def manifests(self):
        return sorted((self.root / "pkg-cache").glob("*/manifest.json"))

    def test_same_day_reuses_a_verified_run_copy_without_network(self):
        first = self.cached()
        second = self.cached()
        self.assertEqual(len(first.commands), 1)
        self.assertEqual(second.commands, [])
        self.assertTrue(second.base_cache_info["hit"])
        manifest = json.loads(self.manifests()[0].read_text())
        self.assertEqual(manifest["refreshed_date"], "2026-11-03")
        # Pip/overlay activity in a run cannot modify the shared cached wheel.
        (self.root / "run-wheel" / manifest["wheel"]).write_bytes(b"modified by this run")
        self.assertEqual(self.cached().commands, [])

    def test_beijing_midnight_starts_a_new_daily_refresh(self):
        self.now.return_value = datetime(2026, 11, 3, 15, 59, tzinfo=timezone.utc)
        self.cached()
        self.now.return_value += timedelta(minutes=1)
        self.assertEqual(len(self.cached().commands), 1)
        self.assertEqual(json.loads(self.manifests()[0].read_text())["refreshed_date"], "2026-11-04")

    def test_download_crossing_midnight_cannot_count_as_the_next_day_refresh(self):
        self.now.return_value = datetime(2026, 11, 3, 15, 59, tzinfo=timezone.utc)
        preparer = FakePreparer(self.root, self.environ)
        download = preparer.download_base

        def cross_midnight(directory):
            wheel = download(directory)
            self.now.return_value += timedelta(minutes=2)
            return wheel

        with patch.object(preparer, "download_base", side_effect=cross_midnight):
            preparer.cached_base(self.root / "run-wheel")
        self.assertEqual(json.loads(self.manifests()[0].read_text())["refreshed_date"], "2026-11-03")
        self.assertEqual(len(self.cached().commands), 1)
        self.assertEqual(json.loads(self.manifests()[0].read_text())["refreshed_date"], "2026-11-04")

    def test_environment_and_explicit_updates_refresh_same_day(self):
        self.cached()
        self.now.return_value += timedelta(seconds=1)
        self.assertEqual(len(self.cached(CI_UPDATE_PACKAGE="true").commands), 1)
        self.now.return_value += timedelta(seconds=1)
        self.assertEqual(len(self.cached(force=True).commands), 1)
        self.assertEqual(len(self.manifests()), 1)

    def test_failed_daily_refresh_does_not_use_old_package_or_mark_today(self):
        self.cached()
        previous = self.manifests()[0].read_bytes()
        self.now.return_value += timedelta(days=1)
        preparer = FakePreparer(self.root, self.environ)
        with patch.object(preparer, "download_base", side_effect=RuntimeError("remote unavailable")), \
                self.assertRaisesRegex(RuntimeError, "remote unavailable"):
            preparer.cached_base(self.root / "failed-run")
        self.assertFalse((self.root / "failed-run").exists())
        self.assertEqual(self.manifests()[0].read_bytes(), previous)
        self.assertEqual(len(self.cached().commands), 1)

    def test_failed_forced_refresh_also_fails_instead_of_using_today_cache(self):
        self.cached()
        self.now.return_value += timedelta(seconds=1)
        preparer = FakePreparer(self.root, self.environ)
        with patch.object(preparer, "download_base", side_effect=RuntimeError("remote unavailable")), \
                self.assertRaisesRegex(RuntimeError, "remote unavailable"):
            preparer.cached_base(self.root / "failed-run", force=True)

    def test_corrupt_hash_metadata_or_wheel_is_repaired_before_use(self):
        for damage in ("hash", "metadata", "filename", "identity", "version", "date"):
            with self.subTest(damage=damage):
                self.cached()
                manifest_path = self.manifests()[0]
                manifest = json.loads(manifest_path.read_text())
                if damage == "hash":
                    wheel = manifest_path.parent / "wheels" / manifest["sha256"] / manifest["wheel"]
                    wheel.write_bytes(b"damaged wheel")
                elif damage == "metadata":
                    manifest_path.write_text("incomplete JSON")
                else:
                    if damage == "filename":
                        manifest["wheel"] = "../outside.whl"
                    elif damage == "identity":
                        manifest["identity"]["soc"] = "ascend950"
                    elif damage == "version":
                        manifest["version"] = "0.dev20261101+main.1234567"
                    elif damage == "date":
                        manifest["refreshed_date"] = "2026-11-04"
                    manifest_path.write_text(json.dumps(manifest))
                self.assertEqual(len(self.cached().commands), 1)
                self.assertEqual(self.cached().commands, [])

    def test_partial_download_cannot_replace_a_good_manifest(self):
        self.cached()
        manifest = self.manifests()[0]
        previous = manifest.read_bytes()
        self.now.return_value += timedelta(days=1)
        preparer = FakePreparer(self.root, self.environ)

        def incomplete(directory):
            (directory / "partial.whl").write_bytes(b"unfinished download")
            raise subprocess.CalledProcessError(1, ["pip", "download"])

        with patch.object(preparer, "download_base", side_effect=incomplete), \
                self.assertRaises(subprocess.CalledProcessError):
            preparer.cached_base(self.root / "failed-run")
        self.assertEqual(manifest.read_bytes(), previous)
        self.assertEqual(list(manifest.parent.glob(".download-*")), [])

    def test_target_branch_soc_cpu_python_tags_and_index_have_separate_keys(self):
        self.cached()
        # Isolate key selection from the fixture's intentionally fixed A2 wheel.
        original_validation = PACKAGE.validate_base_wheel
        with patch.object(PACKAGE, "validate_base_wheel", return_value="0.dev20261102+main.1234567"):
            cases = ({"CI_BASE_REF": "v26.10.0"}, {"CI_SOC": "ascend950"},
                     {"CI_BASE_WHEEL_INDEX_URL": "https://other.example/nightly/"},
                     {"CI_BASE_WHEEL_VERSION": "0.dev20261102+main.1234567"})
            for extra in cases:
                preparer = FakePreparer(self.root, dict(self.environ, **extra))
                preparer.cached_base(self.root / "run-wheel")
                self.assertEqual(len(preparer.commands), 1)
            with patch.object(PACKAGE.platform, "machine", return_value="x86_64"):
                self.assertEqual(len(self.cached().commands), 1)
            different_tag = next(iter(PACKAGE.parse_wheel_filename(
                "flash_linear_attention_npu_a2-0.dev20261102+main.1234567-cp313-cp313-manylinux_2_34_aarch64.whl")[3]))
            with patch.object(PACKAGE, "sys_tags", side_effect=lambda: iter([different_tag])):
                self.assertEqual(len(self.cached().commands), 1)
        self.assertEqual(len(self.manifests()), 7)
        self.assertIs(PACKAGE.validate_base_wheel, original_validation)

    def test_update_only_refreshes_and_reports_without_install_build_or_env(self):
        preparer = FakePreparer(self.root, self.environ)
        preparer.update_package()
        self.assertEqual(len(preparer.commands), 1)
        self.assertIn("download", preparer.commands[0][0])
        self.assertFalse(preparer.env_file.exists())
        info = json.loads(preparer.info_file.read_text())
        self.assertEqual(info["kind"], "update")
        self.assertEqual(info["base_ref"], "main")
        self.assertEqual(info["cache"]["refreshed_date"], "2026-11-03")
        self.assertEqual(info["base_version"], "0.dev20261102+main.1234567")

    def test_update_cli_uses_cache_only_entrypoint(self):
        with patch.object(PACKAGE, "PackagePreparer") as preparer:
            self.assertEqual(PACKAGE.main(["--update-package"]), 0)
        preparer.return_value.update_package.assert_called_once_with()
        preparer.return_value.prepare.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX setgid cache permission contract")
    def test_new_cache_files_share_group_access_without_changing_existing_root(self):
        cache_root = self.root / "pkg-cache"
        cache_root.mkdir(mode=0o2775)
        cache_root.chmod(0o2775)
        self.cached()
        manifest = self.manifests()[0]
        metadata = json.loads(manifest.read_text())
        wheel = manifest.parent / "wheels" / metadata["sha256"] / metadata["wheel"]
        for directory in (cache_root, manifest.parent, wheel.parent.parent, wheel.parent):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o2775)
        for file in (manifest, wheel, manifest.parent / ".lock"):
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o664)
            self.assertEqual(file.stat().st_gid, cache_root.stat().st_gid)
        cache_root.chmod(0o2755)
        self.cached()
        self.assertEqual(stat.S_IMODE(cache_root.stat().st_mode), 0o2755)

    def test_concurrent_processes_download_once_under_the_shared_lock(self):
        worker = self.root / "cache_worker.py"
        worker.write_text('''import importlib.util, pathlib, sys, time
spec = importlib.util.spec_from_file_location("package_tests", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.PACKAGE.sys_tags = lambda: iter([module.TAG])
module.PACKAGE.platform.machine = lambda: "aarch64"
root = pathlib.Path(sys.argv[2])
preparer = module.FakePreparer(root, {"CI_BUILD_KIND": "base", "CI_BASE_REF": "main",
    "CI_SOC": "ascend910b", "CI_PACKAGE_CACHE": str(root / "pkg-cache")})
original_run = preparer.run
def run(args, **kwargs):
    if "download" in args:
        with (root / "downloads.log").open("a") as log:
            log.write("download\\n")
        time.sleep(0.5)
    original_run(args, **kwargs)
preparer.run = run
preparer.cached_base(root / ("worker-" + sys.argv[3]))
''', encoding="utf-8")
        processes = [subprocess.Popen([sys.executable, str(worker), str(Path(__file__).resolve()),
                                       str(self.root), str(index)], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True) for index in range(2)]
        try:
            outputs = [process.communicate(timeout=20) for process in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        for process, (stdout, stderr) in zip(processes, outputs):
            self.assertEqual(process.returncode, 0, stdout + stderr)
        self.assertEqual((self.root / "downloads.log").read_text().splitlines(), ["download"])


if __name__ == "__main__":
    unittest.main()
