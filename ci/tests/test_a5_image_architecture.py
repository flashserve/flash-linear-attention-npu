"""Exercise the A5 Dockerfile's native wheel selection without Docker or downloads."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ci/Dockerfile.ascend950"


def docker_instructions():
    text = DOCKERFILE.read_text(encoding="utf-8")
    return [line for line in text.replace("\\\n", " ").splitlines() if line.startswith("RUN ")]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
class A5NativeWheelTests(unittest.TestCase):
    def run_install(self, arch="aarch64", **overrides):
        script = next(run[4:] for run in docker_instructions() if "install_wheel()" in run)
        wrapper = r'''
record() { printf '%s\n' "$*" >> "$WHEEL_COMMAND_LOG"; }
uname() { echo "$FIXTURE_CPU"; }
python3() { record python "$@"; return 0; }
curl() { record curl "$@"; return 0; }
sha256sum() { local line; IFS= read -r line; record sha256 "$line"; return "${FIXTURE_HASH_STATUS:-0}"; }
rm() { record rm "$@"; }
sleep() { return 0; }
'''
        args = dict(re.findall(r"^ARG ([A-Z0-9_]+)=(.*)$", DOCKERFILE.read_text(encoding="utf-8"), re.M))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "wheel-commands"
            env = dict(os.environ, **args)
            env.update(FIXTURE_CPU=arch, WHEEL_COMMAND_LOG=str(log), **overrides)
            completed = subprocess.run(["bash", "-c", wrapper + "\n" + script], env=env,
                                       text=True, capture_output=True, check=False)
            commands = log.read_text().splitlines() if log.exists() else []
        return completed, commands

    def test_native_cpu_selects_matching_pinned_framework_wheels(self):
        expected = {
            "x86_64": ("5e3dc83725581fa38b7b2e45c58692e30b2a3cde19191af54b675ffcac3840a6",
                       "acb39467b740c9dc24165b0002f3531521615d396d9386134689ce59221006c1"),
            "aarch64": ("ce2ddb880b0813fcc91a737f08fdd973a8115a74c64ccb34e9c09a7964b4d448",
                        "6ebd04ca1facecef7618c34fb4f300b486a5f728e887f8aa5dec1154c140b888"),
        }
        for arch, hashes in expected.items():
            with self.subTest(arch=arch):
                completed, commands = self.run_install(arch)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                downloads = [command for command in commands if command.startswith("curl ")]
                self.assertEqual(len(downloads), 2)
                self.assertIn(f"torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_{arch}.whl", downloads[0])
                self.assertIn(f"torch_npu-2.12.0-cp312-cp312-manylinux_2_28_{arch}.whl", downloads[1])
                checks = [command for command in commands if command.startswith("sha256 ")]
                self.assertEqual([command.split()[1] for command in checks], list(hashes))
                installs = [command for command in commands if command.startswith("python -m pip install ")]
                self.assertEqual(len(installs), 2)
                self.assertTrue(all(f"manylinux_2_28_{arch}.whl" in command for command in installs))
                self.assertTrue(all(commands.index(check) < commands.index(install)
                                    for check, install in zip(checks, installs)))

    def test_complete_native_source_overrides_are_verified(self):
        completed, commands = self.run_install(
            TORCH_WHL="torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl",
            TORCH_WHL_URL="https://controlled.example/torch.whl", TORCH_WHL_SHA256="a" * 64,
            TORCH_NPU_WHL="torch_npu-2.12.0-cp312-cp312-manylinux_2_28_aarch64.whl",
            TORCH_NPU_WHL_URL="https://controlled.example/torch_npu.whl", TORCH_NPU_WHL_SHA256="b" * 64)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(any("https://controlled.example/torch.whl" in command for command in commands))
        self.assertTrue(any(command.startswith("sha256 " + "a" * 64) for command in commands))
        self.assertTrue(any(command.startswith("sha256 " + "b" * 64) for command in commands))

    def test_partial_wrong_architecture_insecure_or_unverified_overrides_fail(self):
        valid = {"TORCH_WHL": "torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl",
                 "TORCH_WHL_URL": "https://controlled.example/torch.whl", "TORCH_WHL_SHA256": "a" * 64}
        cases = [{"TORCH_WHL_URL": valid["TORCH_WHL_URL"]},
                 dict(valid, TORCH_WHL=valid["TORCH_WHL"].replace("aarch64", "x86_64")),
                 dict(valid, TORCH_WHL=valid["TORCH_WHL"].replace("cp312", "cp311")),
                 dict(valid, TORCH_WHL_URL="http://controlled.example/torch.whl"),
                 dict(valid, TORCH_WHL_SHA256="bad")]
        for values in cases:
            with self.subTest(values=values):
                completed, commands = self.run_install(**values)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("complete filename/HTTPS URL/SHA256 triple", completed.stderr)
                self.assertFalse(any(command.startswith("curl ") for command in commands))

    def test_checksum_failure_stops_before_package_install(self):
        completed, commands = self.run_install(FIXTURE_HASH_STATUS="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(any(command.startswith("sha256 ") for command in commands))
        self.assertFalse(any(command.startswith("python -m pip install ") for command in commands))

    def test_a2_versions_or_unsupported_cpus_cannot_select_a5_wheels(self):
        for arch, options in (("s390x", {}), ("aarch64", {"TORCH_VERSION": "2.7.1"}),
                              ("aarch64", {"TORCH_NPU_VERSION": "2.7.1.post5"})):
            with self.subTest(arch=arch, options=options):
                completed, commands = self.run_install(arch, **options)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(any(command.startswith("curl ") for command in commands))

    def test_explicit_framework_skip_does_not_download_either_architecture(self):
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch):
                completed, commands = self.run_install(arch, INSTALL_TORCH_NPU="false")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
