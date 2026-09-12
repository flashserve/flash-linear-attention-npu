#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


class WorkflowResultPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        cls.matrix_steps = workflow["jobs"]["ascend-npu"]["steps"]
        cls.finalize_steps = workflow["jobs"]["finalize"]["steps"]
        cls.finalize_script = next(
            step["with"]["script"]
            for step in cls.finalize_steps
            if step.get("uses") == "actions/github-script@v7"
        )

    def _run_finalize_script(
        self,
        malformed=False,
        malicious=False,
        inconsistent=None,
        incomplete=False,
        hidden_payload=None,
        invalid_return_code=False,
        matrix_result="success",
    ):
        sha = "a" * 40
        run_id = "42"
        run_attempt = "1"
        metric = {
            "tensor": "o",
            "finite": True,
            "allclose": True,
            "cosine_ok": True,
            "tol": 0.005,
            "cosine": 1.0,
            "cos_min": 0.999,
            "max_abs": 0.0,
            "bad_frac": 0.0,
        }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "accuracy-reports"
            reports.mkdir()
            for platform, soc in (("a2", "ascend910b"), ("a5", "ascend950")):
                metadata = {
                    "platform": platform,
                    "soc": soc,
                    "head_sha": sha,
                    "run_id": run_id,
                    "run_attempt": run_attempt,
                }
                result = {
                    "schema": "npu-ci-platform-result-v1",
                    "metadata": metadata,
                    "status": "success",
                }
                report = {
                    "schema": "gdr-accuracy-report-v1",
                    "complete": True,
                    "metadata": metadata,
                    "summary": {
                        "total": 1,
                        "passed": 1,
                        "failed": 0,
                        "not_run": 0,
                        "accuracy_total": 1,
                        "accuracy_passed": 1,
                        "accuracy_failed": 0,
                        "accuracy_not_run": 0,
                    },
                    "cases": [
                        {
                            "name": "case-1",
                            "status": "passed",
                            "return_code": 0,
                            "accuracy_check": True,
                            "accuracy_status": "passed",
                            "metrics": [metric],
                        }
                    ],
                }
                diagnostic = {
                    "schema": "npu-ci-diagnostics-v1",
                    "metadata": metadata,
                    "status": "success",
                    "execution": {"status": "success", "exit_code": 0},
                    "accuracy": {
                        "status": "success",
                        "total": 1,
                        "passed": 1,
                        "failed": 0,
                        "not_run": 0,
                        "failures": [],
                    },
                    "diagnostics": {
                        "compile": [],
                        "runtime": [],
                        "infrastructure": [],
                    },
                    "reproduction": [],
                }
                if malformed and platform == "a5":
                    report["cases"] = [None]
                    diagnostic["accuracy"]["failures"] = [None]
                if malicious and platform == "a5":
                    malicious_name = (
                        "@org/team ci-a5-runner-07 <!-- injected --> [`x`] "
                        'config={"token": "node json secret", '
                        '"api_key":"node escaped\\\"secret"} '
                        'env={"hostname":"node-buildbox", "runner":"node-private-box"} '
                        'tool --token node-cli-token --password="node cli password" '
                        'RuntimeError: token = 128 is outside sequence length '
                        'RuntimeError: worker = 2 returned invalid result '
                        'NodeProto *node = dynamic_cast<const NodeProto *>(op_src) '
                        "password: 'node phrase secret'"
                    )
                    malicious_tensor = (
                        "@org/tensor cache.corp.local worker_count node_index host_tensor <b>bad</b>"
                    )
                    result["status"] = "failure"
                    report["summary"].update(
                        accuracy_passed=0,
                        accuracy_failed=1,
                    )
                    report["cases"][0].update(
                        name=malicious_name,
                        accuracy_status="failed",
                        metrics=[{**metric, "tensor": malicious_tensor, "allclose": False}],
                    )
                    diagnostic.update(status="failure")
                    diagnostic["accuracy"].update(
                        status="failure",
                        passed=0,
                        failed=1,
                        failures=[
                            {
                                "name": malicious_name,
                                "status": "failed",
                                "return_code": 1,
                                "metrics": [
                                    {
                                        "output": malicious_tensor,
                                        "status": "failed",
                                        "details": {"allclose": False},
                                    }
                                ],
                            }
                        ],
                    )
                    diagnostic["diagnostics"]["runtime"] = [
                        {
                            "headline": "sanitizer regression fixture",
                            "lines": [
                                'env={"hostname":"node-buildbox", "runner":"node-private-box"}',
                                'tool --token node-cli-token --password="node cli password" password: node phrase secret;',
                                "RuntimeError: worker = 2 returned invalid result; RuntimeError: token = 128 is outside sequence length",
                                "23 | const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);",
                            ],
                            "occurrences": 1,
                        }
                    ]
                if inconsistent == "top" and platform == "a5":
                    diagnostic["execution"] = {"status": "failure", "exit_code": 1}
                if inconsistent == "accuracy" and platform == "a5":
                    result["status"] = "failure"
                    diagnostic["status"] = "failure"
                    diagnostic["accuracy"].update(
                        status="failure",
                        passed=0,
                        failed=1,
                    )
                if incomplete and platform == "a5":
                    report["complete"] = False
                if hidden_payload == "compile" and platform == "a5":
                    diagnostic["diagnostics"]["compile"] = [
                        {
                            "headline": "broken compiler input",
                            "lines": ["src/x.cpp:1: error: broken compiler input"],
                            "occurrences": 1,
                        }
                    ]
                if hidden_payload == "accuracy" and platform == "a5":
                    diagnostic["accuracy"]["failures"] = [
                        {
                            "name": "hidden-failure",
                            "status": "failed",
                            "return_code": 1,
                            "metrics": [
                                {
                                    "output": "o",
                                    "status": "failed",
                                    "details": {"allclose": False},
                                }
                            ],
                        }
                    ]
                if invalid_return_code and platform == "a5":
                    report["cases"][0]["return_code"] = 1
                for name, payload in (
                    (f"npu-ci-result-{platform}.json", result),
                    (f"gdr_accuracy_report-{platform}.json", report),
                    (f"npu-ci-diagnostics-{platform}.json", diagnostic),
                ):
                    (reports / name).write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )

            script = self.finalize_script
            replacements = {
                "${{ toJson(needs.prepare.outputs.ci_mode) }}": json.dumps("quick"),
                "${{ toJson(needs.prepare.outputs.ops) }}": json.dumps(""),
                "${{ toJson(needs.prepare.outputs.pr_number) }}": json.dumps("527"),
                "${{ toJson(needs.prepare.outputs.comment_id) }}": json.dumps("123"),
            }
            for source, replacement in replacements.items():
                script = script.replace(source, replacement)

            wrapper = """
globalThis.context = {
  serverUrl: 'https://github.com',
  repo: { owner: 'example', repo: 'repo' },
  runId: 42,
  runNumber: 7,
  actor: 'ci-user',
};
const record = (kind, payload) => require('fs').appendFileSync(
  process.env.RECORD_FILE,
  JSON.stringify({ kind, payload }) + '\\n',
  'utf8'
);
globalThis.github = { rest: {
  pulls: { get: async () => ({ data: { head: { sha: process.env.NPU_EXPECTED_HEAD_SHA } } }) },
  repos: { createCommitStatus: async (payload) => record('status', payload) },
  issues: {
    updateComment: async (payload) => require('fs').writeFileSync(process.env.COMMENT_FILE, payload.body, 'utf8'),
    createComment: async (payload) => require('fs').writeFileSync(process.env.COMMENT_FILE, payload.body, 'utf8'),
  },
} };
globalThis.core = {
  warning: (message) => record('warning', message),
  setFailed: (message) => record('failed', message),
};
(async () => {
""" + script + """
})().catch((error) => {
  console.error(error && error.stack || error);
  process.exitCode = 1;
});
"""
            comment_file = root / "comment.md"
            record_file = root / "records.jsonl"
            env = os.environ.copy()
            env.update(
                {
                    "COMMENT_FILE": str(comment_file),
                    "RECORD_FILE": str(record_file),
                    "NPU_EXPECTED_HEAD_SHA": sha,
                    "NPU_EXPECTED_RUN_ID": run_id,
                    "NPU_EXPECTED_RUN_ATTEMPT": run_attempt,
                    "NPU_MATRIX_RESULT": matrix_result,
                }
            )
            completed = subprocess.run(
                ["node"],
                input=wrapper,
                text=True,
                encoding="utf-8",
                cwd=root,
                env=env,
                capture_output=True,
                check=False,
            )
            records = [
                json.loads(line)
                for line in record_file.read_text(encoding="utf-8").splitlines()
            ]
            return completed, comment_file.read_text(encoding="utf-8"), records

    def test_container_output_is_grouped_and_summarized(self):
        run_script = next(
            step["run"]
            for step in self.matrix_steps
            if "bash ci/run_ci_container.sh" in step.get("run", "")
        )
        self.assertIn("::group::", run_script)
        self.assertIn('tee "$CI_RAW_LOG_FILE"', run_script)
        self.assertIn("summarize_npu_ci.py", run_script)
        self.assertIn("--require-accuracy-report", run_script)
        self.assertIn('cat "$CI_SUMMARY_FILE" >>"$GITHUB_STEP_SUMMARY"', run_script)
        self.assertIn('elif wait "$ci_pid"; then', run_script)

    def test_platform_artifact_contains_structured_diagnostics(self):
        upload = next(
            step
            for step in self.matrix_steps
            if step.get("uses") == "actions/upload-artifact@v4"
        )
        paths = upload["with"]["path"]
        self.assertIn("CI_ACCURACY_REPORT_FILE", paths)
        self.assertIn("CI_DIAGNOSTIC_FILE", paths)
        self.assertIn("CI_SUMMARY_FILE", paths)
        self.assertNotIn("CI_RAW_LOG_FILE", paths)

    def test_cleanup_runs_before_status_capture_and_upload(self):
        names = [step.get("name", "") for step in self.matrix_steps]
        cleanup = next(index for index, name in enumerate(names) if name.startswith("清理 "))
        record = next(index for index, name in enumerate(names) if name.startswith("记录 "))
        upload = next(index for index, name in enumerate(names) if name.startswith("上传 "))
        self.assertLess(cleanup, record)
        self.assertLess(record, upload)

    def test_pr_comment_only_expands_failed_diagnostics(self):
        script = self.finalize_script
        self.assertIn("npu-ci-diagnostics-v1", script)
        self.assertIn("platforms.filter((platform) => !platform.executionOk || !platform.accuracyOk)", script)
        self.assertIn("accuracy.failures", script)
        self.assertIn("sanitizeInline", script)
        self.assertIn("renderComment", script)
        self.assertIn("sanitizeInline(line, 2400)", script)
        self.assertIn("summary 与 cases 不一致", script)
        self.assertIn("metric ${metricIndex} 不是对象", script)
        self.assertIn("accuracy.failures.filter(isRecord).slice(0, 4)", script)
        self.assertIn("failure.metrics.filter(isRecord).slice(0, 3)", script)
        self.assertIn("失败判据", script)
        self.assertIn("finite", script)
        self.assertIn("return_code", script)
        self.assertNotIn("for (const caseItem of item.report.cases)", script)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_renders_compact_success(self):
        completed, comment, records = self._run_finalize_script()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["success", "success"])
        self.assertIn("NPU CI 通过", comment)
        self.assertIn("1/1", comment)
        self.assertNotIn("case-1", comment)
        self.assertLess(len(comment), 48000)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_fails_closed_for_malformed_items(self):
        completed, comment, records = self._run_finalize_script(malformed=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "failure"])
        self.assertTrue(any(item["kind"] == "failed" for item in records))
        self.assertIn("NPU CI 失败", comment)
        self.assertIn("不是对象", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_neutralizes_untrusted_markdown_and_mentions(self):
        completed, comment, records = self._run_finalize_script(malicious=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "failure"])
        self.assertNotIn("@org/team", comment)
        self.assertNotIn("@org/tensor", comment)
        self.assertNotIn("<!-- injected -->", comment)
        self.assertNotIn("ci-a5-runner-07", comment)
        self.assertNotIn("cache.corp.local", comment)
        self.assertNotIn("node json secret", comment)
        self.assertNotIn("node phrase secret", comment)
        self.assertNotIn("node-buildbox", comment)
        self.assertNotIn("node-private-box", comment)
        self.assertNotIn("node-cli-token", comment)
        self.assertNotIn("node cli password", comment)
        self.assertNotIn("node escaped", comment)
        self.assertIn("token = 128 is outside sequence length", comment)
        self.assertIn("worker = 2 returned invalid result", comment)
        self.assertIn("node = dynamic_cast", comment)
        self.assertIn("_at_org/team", comment)
        self.assertIn("worker_count", comment)
        self.assertIn("node_index", comment)
        self.assertIn("host_tensor", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_inconsistent_diagnostic_top_status(self):
        completed, comment, records = self._run_finalize_script(inconsistent="top")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "success"])
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_report_diagnostic_mismatch(self):
        completed, comment, records = self._run_finalize_script(inconsistent="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "failure"])
        self.assertIn("accuracy 与精度报告不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_explains_matrix_artifact_mismatch(self):
        completed, comment, records = self._run_finalize_script(matrix_result="failure")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "success"])
        self.assertIn("但双平台 artifact 均记录为通过", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_incomplete_accuracy_report(self):
        completed, comment, records = self._run_finalize_script(incomplete=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["success", "failure"])
        self.assertIn("精度报告未完整结束", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_compile_payload_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="compile")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "success"])
        self.assertIn("broken compiler input", comment)
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_accuracy_failure_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "failure"])
        self.assertIn("精度成功但包含失败用例", comment)
        self.assertIn("hidden-failure", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_case_return_code_mismatch(self):
        completed, comment, records = self._run_finalize_script(invalid_return_code=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [item["payload"]["state"] for item in records if item["kind"] == "status"]
        self.assertEqual(states, ["failure", "failure"])
        self.assertIn("执行状态与 return_code 不一致", comment)


if __name__ == "__main__":
    unittest.main()
