#!/usr/bin/env python3
import importlib.util
import json
import shlex
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "summarize_npu_ci.py"
SPEC = importlib.util.spec_from_file_location("summarize_npu_ci", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def _case(name, status="passed", accuracy_status="passed", metrics=None, return_code=0):
    if metrics is None:
        metrics = (
            [
                {
                    "tensor": "o",
                    "finite": True,
                    "allclose": True,
                    "cosine_ok": True,
                }
            ]
            if accuracy_status == "passed"
            else []
        )
    return {
        "name": name,
        "status": status,
        "return_code": return_code,
        "accuracy_check": True,
        "accuracy_status": accuracy_status,
        "metrics": metrics,
    }


def _report(cases, passed, failed=0, not_run=0):
    return {
        "schema": "gdr-accuracy-report-v1",
        "complete": True,
        "metadata": {},
        "summary": {
            "total": len(cases),
            "passed": sum(case["status"] == "passed" for case in cases),
            "failed": sum(case["status"] == "failed" for case in cases),
            "not_run": sum(case["status"] == "not_run" for case in cases),
            "accuracy_total": passed + failed + not_run,
            "accuracy_passed": passed,
            "accuracy_failed": failed,
            "accuracy_not_run": not_run,
        },
        "cases": cases,
    }


class SummarizeNpuCiTest(unittest.TestCase):
    def _run(
        self,
        log,
        *,
        report=None,
        exit_code=0,
        require_report=False,
        platform="a5",
        soc="ascend950",
        ops="chunk_fwd_o",
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_path = root / "raw.log"
            report_path = root / "accuracy.json"
            json_path = root / "diagnostics.json"
            markdown_path = root / "summary.md"
            log_path.write_text(log, encoding="utf-8")
            if report is not None:
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False), encoding="utf-8"
                )
            argv = [
                "--platform",
                platform,
                "--soc",
                soc,
                "--mode",
                "quick",
                "--ops",
                ops,
                "--exit-code",
                str(exit_code),
                "--head-sha",
                "abc123",
                "--run-id",
                "42",
                "--run-attempt",
                "1",
                "--log-file",
                str(log_path),
                "--accuracy-report-file",
                str(report_path),
                "--json-out",
                str(json_path),
                "--markdown-out",
                str(markdown_path),
            ]
            if require_report:
                argv.append("--require-accuracy-report")
            return_code = summarizer.main(argv)
            payload_text = json_path.read_text(encoding="utf-8")
            markdown = markdown_path.read_text(encoding="utf-8")
            return return_code, json.loads(payload_text), payload_text, markdown

    def test_success_ignores_large_warning_volume(self):
        cases = [_case(f"case-{index}") for index in range(4)]
        warnings = "".join(
            f"/workspace/repo/file{index}.cpp:1: warning: harmless warning {index}\n"
            for index in range(5000)
        )
        return_code, payload, payload_text, markdown = self._run(
            warnings,
            report=_report(cases, passed=4),
            require_report=True,
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["accuracy"]["passed"], 4)
        self.assertEqual(payload["diagnostics"], {"compile": [], "runtime": [], "infrastructure": []})
        self.assertNotIn("warning", payload_text.lower())
        self.assertEqual(
            markdown,
            "### A5 / `ascend950`\n- 执行：通过\n- 精度：4/4\n",
        )

    def test_repeated_memcpy_compile_error_is_deduplicated_with_source_context(self):
        diagnostic = (
            "\x1b[31m/workspace/repo/fla/ops/kernel.h:210:9: error: use of "
            "undeclared identifier 'memcpy'\x1b[0m\n"
            "        memcpy(&lowerBound_, &bits, sizeof(lowerBound_));\n"
            "        ^\n"
            "1 error generated.\n"
        )
        return_code, payload, payload_text, markdown = self._run(
            diagnostic + diagnostic,
            exit_code=1,
        )

        self.assertEqual(return_code, 1)
        blocks = payload["diagnostics"]["compile"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["occurrences"], 2)
        joined = "\n".join(blocks[0]["lines"])
        self.assertIn("memcpy(&lowerBound_, &bits", joined)
        self.assertIn("^", joined)
        self.assertNotIn("\x1b", payload_text)
        self.assertNotIn("/workspace/repo", payload_text)
        self.assertIn("重复 2 次", markdown)

    def test_exported_github_log_prefixes_are_removed_before_deduplication(self):
        first = (
            "A5\tUNKNOWN STEP\t2026-09-11T03:59:17.1414853Z "
            "/workspace/repo/fla/ops/kernel.h:77:13: error: use of undeclared identifier 'memcpy'\n"
            "A5\tUNKNOWN STEP\t2026-09-11T03:59:17.1415806Z     memcpy(&value, &bits, 4);\n"
            "A5\tUNKNOWN STEP\t2026-09-11T03:59:17.1416146Z     ^\n"
        )
        second = first.replace("17.141", "18.222")
        command_echo = (
            "A5\tUNKNOWN STEP\t2026-09-11T03:58:41.6437454Z "
            "^[[36;1mraise RuntimeError('source only')^[[0m\n"
        )
        _, payload, payload_text, _ = self._run(
            command_echo + first + second,
            exit_code=1,
        )

        self.assertEqual(len(payload["diagnostics"]["compile"]), 1)
        self.assertEqual(payload["diagnostics"]["compile"][0]["occurrences"], 2)
        self.assertFalse(payload["diagnostics"]["runtime"])
        self.assertNotIn("UNKNOWN STEP", payload_text)
        self.assertIn("^", "\n".join(payload["diagnostics"]["compile"][0]["lines"]))

    def test_traceback_is_kept_as_runtime_diagnostic(self):
        log = """Traceback (most recent call last):
  File "/data/example-project/example.py", line 8, in <module>
    launch()
RuntimeError: ACL stream synchronize failed, error code 507011
"""
        return_code, payload, payload_text, _ = self._run(log, exit_code=1)

        self.assertEqual(return_code, 1)
        self.assertEqual(len(payload["diagnostics"]["runtime"]), 1)
        lines = "\n".join(payload["diagnostics"]["runtime"][0]["lines"])
        self.assertIn("Traceback", lines)
        self.assertIn("RuntimeError: ACL stream synchronize failed", lines)
        self.assertNotIn("/data/example-project", payload_text)

    def test_acl_error_and_exit_137_are_classified(self):
        log = "[ERROR] aclnn launch failed: ACL_ERROR_RT_DEVICE_TASK_ABORT\n"
        return_code, payload, _, markdown = self._run(log, exit_code=137)

        self.assertEqual(return_code, 1)
        self.assertTrue(payload["diagnostics"]["runtime"])
        infrastructure = payload["diagnostics"]["infrastructure"]
        self.assertTrue(
            any("137" in block["headline"] for block in infrastructure)
        )
        self.assertIn("执行错误、基础设施错误", markdown)

    def test_plain_error_marker_is_kept_as_actionable_runtime_diagnostic(self):
        log = "[ERROR] dtype must be half or bf16\n[FAIL] gdn_fwd_o\n"
        return_code, payload, _, _ = self._run(log, exit_code=1)

        self.assertEqual(return_code, 1)
        runtime = payload["diagnostics"]["runtime"]
        self.assertTrue(
            any("dtype must be half or bf16" in block["headline"] for block in runtime)
        )

    def test_wheel_and_dependency_check_failures_keep_their_root_cause(self):
        messages = [
            "[CI][ERROR] Standalone fla_npu wheel is not discoverable",
            "[CI][ERROR] Missing scoped wheel OPP files: libcust_opapi.so",
            "[CI][ERROR] Missing Python dependencies for Example ST: torch_npu: unavailable",
        ]
        for message in messages:
            with self.subTest(message=message):
                return_code, payload, _, _ = self._run(message + "\n", exit_code=1)
                self.assertEqual(return_code, 1)
                self.assertTrue(
                    any(message in block["headline"] for block in payload["diagnostics"]["runtime"])
                )

    def test_gdn_failure_marker_is_kept_as_runtime_diagnostic(self):
        log = "[FAIL] whl 编译安装失败\n[FAIL] GDN 验证存在未通过项。\n"
        return_code, payload, _, markdown = self._run(log, exit_code=1)

        self.assertEqual(return_code, 1)
        runtime = payload["diagnostics"]["runtime"]
        self.assertEqual(len(runtime), 2)
        self.assertIn("whl 编译安装失败", runtime[0]["headline"])
        self.assertIn("[FAIL] whl", markdown)

    def test_pure_compile_failure_does_not_add_runtime_marker_noise(self):
        log = """src/kernel.cpp:77:13: error: use of undeclared identifier 'memcpy'
[FAIL] ascend950 整包编译失败
[FAIL] GDN 验证存在未通过项。
"""
        return_code, payload, _, markdown = self._run(log, exit_code=1)

        self.assertEqual(return_code, 1)
        self.assertTrue(payload["diagnostics"]["compile"])
        self.assertEqual(payload["diagnostics"]["runtime"], [])
        self.assertNotIn("编译错误、执行错误", markdown)

    def test_compile_source_excerpt_and_cascading_link_errors_are_filtered(self):
        log = """src/kernel.cpp:77:13: error: use of undeclared identifier 'memcpy'
  284 |     "Compare input shape failed for input npu tensor"
            ^
ld.lld: error: cannot open build/kernel_meta/foo.o: No such file or directory
"""
        _, payload, _, _ = self._run(log, exit_code=1)

        self.assertEqual(len(payload["diagnostics"]["compile"]), 1)
        self.assertIn("memcpy", payload["diagnostics"]["compile"][0]["headline"])
        self.assertFalse(payload["diagnostics"]["runtime"])

    def test_compiler_variants_keep_actionable_context(self):
        log = """cc1plus: fatal error: missing_header.h: No such file or directory
compilation terminated.
CMake Error at CMakeLists.txt:42 (message):
  Required compiler component was not found
src/werror.cpp:9: warning: promoted warning [-Werror=unused-variable]
"""
        _, payload, _, _ = self._run(log, exit_code=1)

        joined = "\n".join(
            line
            for block in payload["diagnostics"]["compile"]
            for line in block["lines"]
        )
        self.assertIn("missing_header.h", joined)
        self.assertIn("Required compiler component", joined)
        self.assertIn("[-Werror=unused-variable]", joined)

    def test_same_error_at_different_locations_is_not_deduplicated(self):
        log = """src/first.cpp:7:3: error: invalid operands
  bad();
  ^
src/second.cpp:7:3: error: invalid operands
  bad();
  ^
"""
        _, payload, _, _ = self._run(log, exit_code=1)

        self.assertEqual(len(payload["diagnostics"]["compile"]), 2)

    def test_traceback_displaces_lower_priority_runtime_noise(self):
        noise = "".join(
            f"NPU device error: unique failure {index}\n" for index in range(4)
        )
        traceback = """Traceback (most recent call last):
  File "/workspace/repo/test.py", line 7, in <module>
    run_case()
RuntimeError: kernel launch failed
"""
        _, payload, _, _ = self._run(noise + traceback, exit_code=1)

        self.assertEqual(len(payload["diagnostics"]["runtime"]), 4)
        self.assertTrue(
            any(
                any("Traceback" in line for line in block["lines"])
                for block in payload["diagnostics"]["runtime"]
            )
        )

    def test_real_runtime_error_displaces_gdn_failure_markers(self):
        markers = "".join(f"[FAIL] stage-{index}\n" for index in range(6))
        traceback = """Traceback (most recent call last):
  File "/workspace/repo/test.py", line 7, in <module>
    run_case()
RuntimeError: ACL kernel launch failed
"""
        _, payload, payload_text, _ = self._run(markers + traceback, exit_code=1)

        runtime = payload["diagnostics"]["runtime"]
        self.assertEqual(len(runtime), 1)
        self.assertTrue(any("Traceback" in "\n".join(block["lines"]) for block in runtime))
        self.assertFalse(any(block["headline"].startswith("[FAIL]") for block in runtime))
        self.assertNotIn("_priority", payload_text)

    def test_accuracy_failure_lists_only_failed_cases_and_metrics(self):
        passed_metric = {
            "tensor": "o",
            "raw": "o: finite=True allclose=True cosine_ok=True bad_frac=0",
            "finite": True,
            "allclose": True,
            "cosine_ok": True,
        }
        failed_metric = {
            "tensor": "final_state",
            "raw": "final_state: finite=True allclose=False cosine_ok=False bad_frac=0.2",
            "finite": True,
            "allclose": False,
            "cosine_ok": False,
        }
        cases = [
            _case("passed-case", metrics=[passed_metric]),
            _case(
                "failed-case",
                status="failed",
                accuracy_status="failed",
                metrics=[passed_metric, failed_metric],
                return_code=1,
            ),
            _case("not-run-case", status="not_run", accuracy_status="not_run", return_code=None),
        ]
        return_code, payload, payload_text, markdown = self._run(
            "accuracy check failed\n",
            report=_report(cases, passed=1, failed=1, not_run=1),
            exit_code=1,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        failures = payload["accuracy"]["failures"]
        self.assertEqual([case["name"] for case in failures], ["failed-case", "not-run-case"])
        self.assertEqual(len(failures[0]["metrics"]), 1)
        self.assertEqual(failures[0]["metrics"][0]["output"], "final_state")
        self.assertFalse(failures[0]["metrics"][0]["details"]["allclose"])
        self.assertNotIn("passed-case", payload_text)
        command = payload["reproduction"][0]
        self.assertIn("CI_EXAMPLE_CASE_FILTER=failed-case,not-run-case", command)
        self.assertIn("CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK=true", command)
        self.assertIn("CI_RUN_SCOPED_WHEEL_INSTALL_CHECK=true", command)
        self.assertIn("ci/run_ci_container.sh", command)
        self.assertNotIn(str(Path.cwd()), command)
        self.assertIn("failed-case", markdown)
        self.assertNotIn("passed-case", markdown)

    def test_failed_metric_overrides_inconsistent_passed_case_summary(self):
        failed_metric = {
            "tensor": "o",
            "finite": True,
            "allclose": False,
            "cosine_ok": True,
            "tol": 0.005,
            "bad_frac": 0.1,
        }
        cases = [_case("inconsistent-case", metrics=[failed_metric])]
        return_code, payload, _, markdown = self._run(
            "accuracy check passed\n",
            report=_report(cases, passed=1),
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(payload["accuracy"]["status"], "failure")
        self.assertEqual(payload["accuracy"]["failures"][0]["name"], "inconsistent-case")
        self.assertIn("allclose=False", markdown)

    def test_non_finite_metric_values_are_strict_json_strings(self):
        failed_metric = {
            "tensor": "o",
            "finite": False,
            "allclose": False,
            "cosine_ok": False,
            "cosine": float("nan"),
            "max_abs": float("inf"),
        }
        cases = [
            _case(
                "non-finite-case",
                status="failed",
                accuracy_status="failed",
                metrics=[failed_metric],
                return_code=1,
            )
        ]
        return_code, payload, payload_text, markdown = self._run(
            "accuracy check failed\n",
            report=_report(cases, passed=0, failed=1),
            exit_code=1,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        details = payload["accuracy"]["failures"][0]["metrics"][0]["details"]
        self.assertEqual(details["cosine"], "nan")
        self.assertEqual(details["max_abs"], "inf")
        self.assertNotIn("NaN", payload_text)
        self.assertNotIn("Infinity", payload_text)
        self.assertIn("finite=False", markdown)

    def test_summary_is_checked_against_case_results(self):
        report = _report([], passed=0)
        report["summary"].update(
            {
                "accuracy_total": 4,
                "accuracy_passed": 4,
            }
        )
        return_code, payload, _, markdown = self._run(
            "CI completed\n",
            report=report,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(payload["accuracy"]["status"], "not_available")
        self.assertIn("Accuracy report is invalid", markdown)

    def test_summary_requires_explicit_zero_fields(self):
        report = _report([_case("valid")], passed=1)
        del report["summary"]["accuracy_failed"]

        return_code, payload, _, markdown = self._run(
            "CI completed\n",
            report=report,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(payload["accuracy"]["status"], "not_available")
        self.assertIn("Accuracy report is invalid", markdown)

    def test_case_status_must_match_return_code(self):
        reports = [
            _report([_case("passed-nonzero", return_code=1)], passed=1),
            _report(
                [_case("failed-zero", status="failed", accuracy_status="failed", return_code=0)],
                passed=0,
                failed=1,
            ),
            _report(
                [_case("not-run-nonnull", status="not_run", accuracy_status="not_run", return_code=1)],
                passed=0,
                not_run=1,
            ),
        ]

        for report in reports:
            with self.subTest(case=report["cases"][0]["name"]):
                return_code, payload, _, _ = self._run(
                    "CI completed\n",
                    report=report,
                    require_report=True,
                )
                self.assertEqual(return_code, 1)
                self.assertEqual(payload["accuracy"]["status"], "not_available")
                headline = payload["diagnostics"]["infrastructure"][0]["headline"]
                self.assertIn("return_code", headline)

    def test_non_object_case_and_metric_are_rejected(self):
        valid_report = _report([_case("valid")], passed=1)
        invalid_case_report = json.loads(json.dumps(valid_report))
        invalid_case_report["cases"] = [None]
        invalid_metric_report = json.loads(json.dumps(valid_report))
        invalid_metric_report["cases"][0]["metrics"] = [None]

        for report in (invalid_case_report, invalid_metric_report):
            with self.subTest(report=report):
                return_code, payload, _, _ = self._run(
                    "CI completed\n",
                    report=report,
                    require_report=True,
                )
                self.assertEqual(return_code, 1)
                self.assertEqual(payload["accuracy"]["status"], "not_available")

    def test_required_missing_accuracy_report_is_infrastructure_failure(self):
        return_code, payload, _, markdown = self._run(
            "CI completed without report\n",
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(payload["accuracy"]["status"], "not_available")
        self.assertTrue(payload["diagnostics"]["infrastructure"])
        self.assertIn("Required accuracy report", payload["diagnostics"]["infrastructure"][0]["headline"])
        self.assertIn("精度：未生成报告", markdown)

    def test_partial_accuracy_report_is_never_treated_as_success(self):
        report = _report([_case("completed-prefix")], passed=1)
        report["complete"] = False

        return_code, payload, _, markdown = self._run(
            "CI process interrupted\n",
            report=report,
            exit_code=1,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(payload["accuracy"]["status"], "not_available")
        self.assertNotIn("精度：1/1", markdown)
        self.assertIn("精度：未执行（执行阶段失败）", markdown)

    def test_missing_report_after_compile_failure_is_not_a_second_infrastructure_error(self):
        log = "fla/ops/kernel.h:7:3: error: invalid operands\n  bad();\n  ^\n"
        return_code, payload, _, markdown = self._run(
            log,
            exit_code=2,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertTrue(payload["diagnostics"]["compile"])
        self.assertFalse(payload["diagnostics"]["infrastructure"])
        self.assertIn("精度：未执行（执行阶段失败）", markdown)

    def test_sensitive_values_are_redacted(self):
        fake_token = "ghp_" + "a" * 26
        log = (
            "RuntimeError: failed on 10.23.45.67 in /data/example-project/private/run.py "
            "with /usr/include/private.h and /var/log/private.log and /mnt/ci/output.txt "
            "and /custom/location/private.txt "
            "via fd00::1, ::1, 127.0.0.1, and 169.254.10.20 as builduser@ci-host "
            "hostname=ci-runner-01 bare-root=/data on ci-a5-runner-07 "
            "GET https://cache.corp.local/artifact failed "
            f"token={fake_token}; password=test-password; CI_JOB_TOKEN=ci-job-value; "
            "AWS_SECRET_ACCESS_KEY=aws-secret-value; api_key=api-secret-value; "
            "access_token=access-secret-value; passwd=passwd-secret-value; "
            "config={\"token\": \"json secret value\", 'api_key':'single quoted value'}; "
            "escaped={\"token\":\"alpha\\\"omega\", 'api_key':'beta\\'theta'}; "
            "env={\"hostname\":\"buildbox\", \"runner\": \"private-box\", "
            "'worker':'build-worker', 'node': 'node-box'}; machine='machine-box'; "
            "tool --token demo-token-value --password=demo-password-value "
            "--api-key \"demo key value\" --aws-secret-access-key='aws cli value'; "
            "password: demo secret phrase; "
            "url=https://testuser:sample-auth@example.invalid/api\n"
        )
        _, _, payload_text, markdown = self._run(log, exit_code=1)
        combined = payload_text + markdown

        for secret in (
            "10.23.45.67",
            "/data/example-project",
            "/usr/include",
            "/var/log",
            "/mnt/ci",
            "/custom/location",
            "fd00::1",
            "::1",
            "127.0.0.1",
            "169.254.10.20",
            "builduser@ci-host",
            "ci-runner-01",
            "ci-a5-runner-07",
            "cache.corp.local",
            "/data",
            fake_token,
            "test-password",
            "ci-job-value",
            "aws-secret-value",
            "api-secret-value",
            "access-secret-value",
            "passwd-secret-value",
            "json secret value",
            "single quoted value",
            "alpha",
            "omega",
            "beta",
            "theta",
            "buildbox",
            "private-box",
            "build-worker",
            "node-box",
            "machine-box",
            "demo-token-value",
            "demo-password-value",
            "demo key value",
            "aws cli value",
            "demo secret phrase",
            "testuser:sample-auth",
        ):
            self.assertNotIn(secret, combined)
        self.assertIn("[PRIVATE_IP]", combined)
        self.assertIn("[REDACTED]", combined)

    def test_sanitizer_keeps_code_symbols_and_is_idempotent(self):
        code_diagnostics = (
            "1970 | const uint32_t token = task.begin + sourceRow;\n"
            "RuntimeError: token = 128 is outside sequence length\n"
            "23 | const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);\n"
            "RuntimeError: worker = 2 returned invalid result\n"
            "RuntimeError: host = tensor_host returned invalid result\n"
            "error: use of undeclared identifier worker_count\n"
            "RuntimeError: node_index out of range\n"
            "RuntimeError: host_tensor shape mismatch\n"
            'RuntimeError: comparison against "\\\\n" failed'
        )
        sanitized_code = summarizer.sanitize_text(code_diagnostics, 1000)

        self.assertIn("worker_count", sanitized_code)
        self.assertIn("node_index", sanitized_code)
        self.assertIn("host_tensor", sanitized_code)
        self.assertIn("\\\\n", sanitized_code)
        self.assertIn("token = task.begin + sourceRow", sanitized_code)
        self.assertIn("token = 128 is outside sequence length", sanitized_code)
        self.assertIn("node = dynamic_cast", sanitized_code)
        self.assertIn("worker = 2 returned invalid result", sanitized_code)
        self.assertIn("host = tensor_host returned invalid result", sanitized_code)

        sensitive = (
            'failed on ci-a5-runner-07; GET https://cache.corp.local/x failed; '
            'config={"token": "idempotent secret phrase"}; '
            'env={"hostname":"idempotent-buildbox", "runner":"idempotent-runner"}; '
            'tool --token [REDACTED] --api-key=[REDACTED]; '
            'array=[{"api_key": "another idempotent secret"}]'
        )
        sanitized_once = summarizer.sanitize_text(sensitive, 1000)
        sanitized_twice = summarizer.sanitize_text(sanitized_once, 1000)
        self.assertEqual(sanitized_once, sanitized_twice)
        self.assertNotIn("ci-a5-runner-07", sanitized_once)
        self.assertNotIn("cache.corp.local", sanitized_once)
        self.assertNotIn("idempotent secret phrase", sanitized_once)
        self.assertNotIn("another idempotent secret", sanitized_once)

    def test_report_fields_are_forced_to_one_line(self):
        failed_metric = {
            "tensor": "output\n### injected heading",
            "finite": True,
            "allclose": False,
            "cosine_ok": True,
            "tol": "0.005\n```",
        }
        cases = [
            _case(
                "@org/team <!-- case --> [`x`]\n### injected heading",
                status="failed",
                accuracy_status="failed",
                metrics=[failed_metric],
                return_code=1,
            )
        ]
        _, payload, _, markdown = self._run(
            "accuracy check failed\n",
            report=_report(cases, passed=0, failed=1),
            exit_code=1,
            require_report=True,
        )

        failure = payload["accuracy"]["failures"][0]
        self.assertNotIn("\n", failure["name"])
        self.assertNotIn("\n", failure["metrics"][0]["output"])
        self.assertNotIn("\n", payload["reproduction"][0])
        self.assertNotIn("\n### injected heading", markdown)
        prose = markdown.split("#### 复现", maxsplit=1)[0]
        self.assertNotIn("@org/team", prose)
        self.assertNotIn("<!--", prose)
        self.assertIn("_at_org/team", prose)

    def test_outputs_are_length_limited(self):
        log = "\n".join(
            f"RuntimeError: unique failure {index} " + "x" * 2000
            for index in range(1000)
        )
        cases = []
        for index in range(100):
            metrics = [
                {
                    "tensor": f"output-{metric_index}",
                    "finite": True,
                    "allclose": False,
                    "cosine_ok": True,
                    "raw": f"output-{metric_index}: allclose=False " + "y" * 2000,
                }
                for metric_index in range(20)
            ]
            cases.append(
                _case(
                    f"failed-case-{index}",
                    status="failed",
                    accuracy_status="failed",
                    metrics=metrics,
                    return_code=1,
                )
            )
        return_code, _, payload_text, markdown = self._run(
            log,
            report=_report(cases, passed=0, failed=100),
            exit_code=1,
            require_report=True,
        )

        self.assertEqual(return_code, 1)
        self.assertLessEqual(len(payload_text.encode("utf-8")), summarizer.MAX_JSON_BYTES)
        self.assertLessEqual(len(markdown), summarizer.MAX_MARKDOWN_CHARS)
        self.assertEqual(
            markdown.count("```text") + markdown.count("```bash"),
            markdown.count("\n```\n"),
        )

    def test_reproduction_command_never_truncates_a_shell_token(self):
        failed_cases = [
            {"name": f"case {index} 'quoted' " + "x" * 150}
            for index in range(10)
        ]
        command = summarizer.build_reproduction(
            "a5",
            "ascend950",
            "full",
            "operator 'quoted' " + "y" * 480,
            failed_cases,
        )[0]

        self.assertLessEqual(len(command), summarizer.MAX_REPRO_COMMAND_CHARS)
        tokens = shlex.split(command)
        self.assertEqual(tokens[-2:], ["bash", "ci/run_ci_container.sh"])
        self.assertNotIn("[truncated]", command)


if __name__ == "__main__":
    unittest.main()
