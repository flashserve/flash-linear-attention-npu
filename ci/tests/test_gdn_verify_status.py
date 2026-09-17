#!/usr/bin/env python3
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_bash() -> str | None:
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    return shutil.which("bash")


BASH = _find_bash()


@unittest.skipUnless(BASH, "Bash is required")
class GdnVerifyStatusTest(unittest.TestCase):
    def _run_status_check(self, mutation: str = "") -> subprocess.CompletedProcess[str]:
        script = f"""
source ./gdn-verify.sh
CANN_MAJOR_MINOR=9.1
SKIPPED_COMPILE=false
SKIP_TEST=false
SKIP_EXAMPLE=false
WHL_OK=true
RUN_OK=true
TEST_STAGE_OK=true
EXAMPLE_OK=true
for soc in $(get_soc_list); do
    for type in "整包" "单算子"; do
        COMPILE_RESULTS["B_${{soc}}_${{type}}"]="OK"
    done
done
for op in "${{TEST_OPS[@]}}"; do
    TEST_RESULTS["$op"]="PASS"
done
{mutation}
verification_succeeded
"""
        return subprocess.run(
            [BASH, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

    def test_all_required_stages_pass(self):
        completed = self._run_status_check()
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_each_failed_stage_produces_nonzero_status(self):
        mutations = {
            "compile": 'COMPILE_RESULTS["B_ascend950_整包"]="FAIL"',
            "wheel": "WHL_OK=false",
            "install": "RUN_OK=false",
            "test process": "TEST_STAGE_OK=false",
            "test result": 'TEST_RESULTS["${TEST_OPS[0]}"]="FAIL"',
            "example": "EXAMPLE_OK=false",
        }
        for stage, mutation in mutations.items():
            with self.subTest(stage=stage):
                completed = self._run_status_check(mutation)
                self.assertNotEqual(completed.returncode, 0, completed.stderr)

    def test_last_legacy_operator_failure_produces_nonzero_status(self):
        completed = self._run_status_check(
            'TEST_RESULTS["chunk_scaled_dot_kkt"]="FAIL"'
        )
        self.assertNotEqual(completed.returncode, 0, completed.stderr)

    def test_skipped_stages_do_not_fail(self):
        completed = self._run_status_check(
            "SKIPPED_COMPILE=true; SKIP_TEST=true; SKIP_EXAMPLE=true; "
            "WHL_OK=false; RUN_OK=false; TEST_STAGE_OK=false; EXAMPLE_OK=false"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_skipped_compile_and_example_are_reported_as_skipped(self):
        script = """
source ./gdn-verify.sh
npu-smi() { echo 'Ascend950'; }
CANN_MAJOR_MINOR=9.1
SOC_FOR_INSTALL=ascend950
DEVICE_ID=0
SKIPPED_COMPILE=true
SKIP_TEST=true
SKIP_EXAMPLE=true
print_report
"""
        completed = subprocess.run(
            [BASH, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("编译通过: 0 /", completed.stdout)
        self.assertIn("flash_gated_delta_rule.py: (已跳过)", completed.stdout)

    def test_comma_separated_single_mode_checks_each_operator(self):
        selected = "causal_conv1d,gdn_fwd_o"
        completed = self._run_status_check(
            f'MODE=single; SINGLE_OP="{selected}"; TEST_RESULTS=(); '
            'TEST_RESULTS["causal_conv1d"]="PASS"; '
            'TEST_RESULTS["gdn_fwd_o"]="PASS"'
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        missing_result = self._run_status_check(
            f'MODE=single; SINGLE_OP="{selected}"; TEST_RESULTS=(); '
            'TEST_RESULTS["causal_conv1d"]="PASS"'
        )
        self.assertNotEqual(missing_result.returncode, 0, missing_result.stderr)

    def test_build_operator_names_map_to_legacy_test_names(self):
        selected = "chunk_fwd_o,chunk_gated_delta_rule_fwd_h"
        completed = self._run_status_check(
            f'MODE=single; SINGLE_OP="{selected}"; TEST_RESULTS=(); '
            'TEST_RESULTS["gdn_fwd_o"]="PASS"; '
            'TEST_RESULTS["gdn_fwd_h"]="PASS"'
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        missing_result = self._run_status_check(
            f'MODE=single; SINGLE_OP="{selected}"; TEST_RESULTS=(); '
            'TEST_RESULTS["gdn_fwd_o"]="PASS"'
        )
        self.assertNotEqual(missing_result.returncode, 0, missing_result.stderr)

    def test_requested_build_ops_are_normalized_to_one_csv_argument(self):
        script = """
source ./gdn-verify.sh
MODE=single
SINGLE_OP=' chunk_fwd_o, recurrent_kda '
normalized_requested_build_ops
"""
        completed = subprocess.run(
            [BASH, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "chunk_fwd_o,recurrent_kda")

    def test_operator_without_legacy_test_does_not_fail_legacy_stage(self):
        completed = self._run_status_check(
            'MODE=single; SINGLE_OP="recurrent_gated_delta_rule"; TEST_RESULTS=()'
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_argument_validation_rejects_missing_or_unknown_single_op(self):
        invalid_inputs = (
            'MODE=single; SINGLE_OP=""',
            'MODE=single; SINGLE_OP="not_a_real_op"',
            'MODE=invalid; SINGLE_OP=""',
        )
        for mutation in invalid_inputs:
            with self.subTest(mutation=mutation):
                script = f"source ./gdn-verify.sh; {mutation}; validate_arguments"
                completed = subprocess.run(
                    [BASH, "-c", script],
                    cwd=REPO_ROOT,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, completed.stderr)

    def test_known_operator_without_legacy_test_passes_argument_validation(self):
        for operator in (
            "recurrent_gated_delta_rule",
            "chunk_scaled_dot_kkt",
            "recurrent_kda",
        ):
            with self.subTest(operator=operator):
                script = (
                    'source ./gdn-verify.sh; MODE=single; '
                    f'SINGLE_OP="{operator}"; validate_arguments'
                )
                completed = subprocess.run(
                    [BASH, "-c", script],
                    cwd=REPO_ROOT,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_legacy_test_dry_run_selects_comma_separated_operators(self):
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env["TEST_LOG_DIR"] = Path(temp).as_posix()
            completed = subprocess.run(
                [
                    BASH,
                    "torch_custom/fla_npu/test/test.sh",
                    "--device",
                    "0",
                    "--op",
                    "causal_conv1d,gdn_fwd_o",
                    "--mode",
                    "dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("[DRY-RUN] causal_conv1d", completed.stdout)
        self.assertIn("[DRY-RUN] gdn_fwd_o", completed.stdout)
        self.assertNotIn("[DRY-RUN] chunk_bwd_dv_local", completed.stdout)

    def test_legacy_test_failure_prints_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            bin_dir = temp_path / "bin"
            log_dir = temp_path / "logs"
            bin_dir.mkdir()
            fake_timeout = bin_dir / "timeout"
            fake_timeout.write_text(
                "#!/bin/sh\n"
                "echo 'Traceback (most recent call last):'\n"
                "echo '  File \"case.py\", line 7, in <module>'\n"
                "echo 'RuntimeError: ACL kernel launch failed'\n"
                "i=1\n"
                "while [ $i -le 40 ]; do\n"
                "  echo \"[ERROR] repeated runtime marker $i\"\n"
                "  i=$((i + 1))\n"
                "done\n"
                "exit 1\n",
                encoding="utf-8",
            )
            fake_timeout.chmod(0o755)
            bash_bin_dir = bin_dir.as_posix()
            if os.name == "nt":
                drive, remainder = bash_bin_dir.split(":", 1)
                bash_bin_dir = f"/{drive.lower()}{remainder}"
            command = (
                f"export PATH={shlex.quote(bash_bin_dir)}:$PATH; "
                f"export TEST_LOG_DIR={shlex.quote(log_dir.as_posix())}; "
                "bash torch_custom/fla_npu/test/test.sh --device 0 "
                "--op causal_conv1d"
            )
            completed = subprocess.run(
                [BASH, "-c", command],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Traceback (most recent call last):", completed.stdout)
        self.assertIn("RuntimeError: ACL kernel launch failed", completed.stdout)

    def test_failure_excerpt_prefers_real_compiler_error(self):
        with tempfile.TemporaryDirectory() as temp:
            log_file = Path(temp) / "build.log"
            log_file.write_text(
                "src/noise.h:1: warning: harmless warning\n"
                "src/kernel.cpp:77:13: error: use of undeclared identifier 'memcpy'\n"
                "    memcpy(dst, src, size);\n"
                "    ^\n"
                + "".join(
                    f"make[{index}]: *** [kernel.o] Error 1\n"
                    for index in range(1, 41)
                )
                + "ninja: build stopped: subcommand failed.\n",
                encoding="utf-8",
            )
            command = (
                "source ./gdn-verify.sh; print_failure_excerpt "
                + shlex.quote(log_file.as_posix())
            )
            completed = subprocess.run(
                [BASH, "-c", command],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("error: use of undeclared identifier 'memcpy'", completed.stdout)
        self.assertIn("memcpy(dst, src, size)", completed.stdout)
        self.assertNotIn("make[40]", completed.stdout)


if __name__ == "__main__":
    unittest.main()
