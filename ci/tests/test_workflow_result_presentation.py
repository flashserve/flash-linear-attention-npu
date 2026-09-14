#!/usr/bin/env python3
import hashlib
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
DEFAULT_STATUS_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "npu-ci-default-status.yml"
)
PUBLISH_SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "publish_npu_ci_status.js"
BRANCH_PROTECTION_PATH = REPO_ROOT / "scripts" / "github" / "apply_branch_protection.sh"
RUN_CHECKS_PATH = REPO_ROOT / "ci" / "run_checks.sh"
CONTRACT_TEST_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-contract-tests.yml"
STAGE_KEYS = (
    "environment-contracts",
    "opp-package",
    "standalone-layout",
    "torch-adapter",
    "gdr-example-st",
    "scoped-overlay",
)
STATUS_CONTEXTS = (
    "NPU CI / A2+A5 / 01 环境、wheel 与运行时契约",
    "NPU CI / A2+A5 / 02 全量 OPP 构建",
    "NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局",
    "NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配",
    "NPU CI / A2+A5 / 05 GDR Example/ST",
    "NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装",
    "NPU CI / A2+A5 / 07 报告与 commit 校验",
)
SCOPED_STATUS_CONTEXT = "NPU CI / A2+A5 / 定向诊断"
CONTRACT_TEST_CONTEXT = "CI 契约测试"


class WorkflowResultPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        cls.prepare_action = next(
            step
            for step in workflow["jobs"]["prepare"]["steps"]
            if step.get("id") == "prepare"
        )
        cls.prepare_script = cls.prepare_action["with"]["script"]
        cls.matrix_steps = workflow["jobs"]["ascend-npu"]["steps"]
        cls.finalize_steps = workflow["jobs"]["finalize"]["steps"]
        cls.finalize_action = next(
            step
            for step in cls.finalize_steps
            if step.get("uses") == "actions/github-script@v8"
        )
        cls.finalize_loader = cls.finalize_action["with"]["script"]
        cls.finalize_script = PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")

    def _run_finalize_script(
        self,
        malformed=False,
        malicious=False,
        inconsistent=None,
        incomplete=False,
        hidden_payload=None,
        invalid_return_code=False,
        matrix_result="success",
        stage_failure=None,
        invalid_stage_sequence=False,
        skipped_required_stage=False,
        missing_stage_report=False,
        accuracy_failure=False,
        missing_required_case=False,
        altered_required_contract=False,
        invalid_required_boundaries=False,
        missing_required_metric=False,
        duplicate_required_metric=False,
        altered_metric_threshold=False,
        ops="",
        expected_run_attempt="1",
        platform_attempts=None,
        nested_artifacts=False,
    ):
        sha = "a" * 40
        run_id = "42"
        run_attempt = expected_run_attempt
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
        accuracy_thresholds = {
            "output_tol": 5e-3,
            "grad_tol": 8e-3,
            "beta_grad_tol": 2e-2,
            "gate_grad_tol": 2e-2,
            "output_cos_min": 0.999,
            "grad_cos_min": 0.999,
            "beta_grad_cos_min": 0.99,
            "gate_grad_cos_min": 0.99,
        }
        all_accuracy_tensors = ["o", "dq", "dk", "dv", "dbeta", "dg"]
        required_cases = (
            {
                "name": "case1_current_default",
                "contract": {
                    "script": "examples/flash_gated_delta_rule.py",
                    "batch": 1,
                    "tokens": 4087,
                    "chunk_size": 64,
                    "query_heads": 32,
                    "value_heads": 32,
                    "key_dim": 128,
                    "value_dim": 128,
                    "dtype": "bf16",
                    "varlen": True,
                    "mean_len": 1024,
                    "gate_source": "g",
                    "gate_function": "logsigmoid",
                    "initial_state": "none",
                    "output_final_state": False,
                    "qk_l2norm": True,
                    "demo_model": False,
                    "conv_kernel": 4,
                    "seed": 20260630,
                    "scale": None,
                    "accuracy_tensors": ["o"],
                    "accuracy_thresholds": accuracy_thresholds,
                    "cu_seqlens": [
                        0,
                        2049,
                        3060,
                        3573,
                        3829,
                        3957,
                        4022,
                        4054,
                        4070,
                        4077,
                        4081,
                        4086,
                        4087,
                    ],
                },
            },
            {
                "name": "gdr_accuracy_dense_b2_t128_h2_d128_fp16",
                "contract": {
                    "script": "examples/flash_gated_delta_rule.py",
                    "batch": 2,
                    "tokens": 128,
                    "chunk_size": 64,
                    "query_heads": 2,
                    "value_heads": 2,
                    "key_dim": 128,
                    "value_dim": 128,
                    "dtype": "fp16",
                    "varlen": False,
                    "mean_len": 128,
                    "gate_source": "g",
                    "gate_function": "logsigmoid",
                    "initial_state": "none",
                    "output_final_state": False,
                    "qk_l2norm": False,
                    "demo_model": False,
                    "conv_kernel": 4,
                    "seed": 42,
                    "scale": 0.1,
                    "accuracy_tensors": all_accuracy_tensors,
                    "accuracy_thresholds": accuracy_thresholds,
                    "cu_seqlens": [],
                },
            },
            {
                "name": "gdr_accuracy_varlen_64_64_h2_d128_fp16",
                "contract": {
                    "script": "examples/flash_gated_delta_rule.py",
                    "batch": 1,
                    "tokens": 128,
                    "chunk_size": 64,
                    "query_heads": 2,
                    "value_heads": 2,
                    "key_dim": 128,
                    "value_dim": 128,
                    "dtype": "fp16",
                    "varlen": True,
                    "mean_len": 1024,
                    "gate_source": "g",
                    "gate_function": "logsigmoid",
                    "initial_state": "none",
                    "output_final_state": False,
                    "qk_l2norm": False,
                    "demo_model": False,
                    "conv_kernel": 4,
                    "seed": 43,
                    "scale": 0.1,
                    "accuracy_tensors": all_accuracy_tensors,
                    "accuracy_thresholds": accuracy_thresholds,
                    "cu_seqlens": [0, 64, 128],
                },
            },
            {
                "name": "gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16",
                "contract": {
                    "script": "examples/flash_gated_delta_rule.py",
                    "batch": 1,
                    "tokens": 3991,
                    "chunk_size": 64,
                    "query_heads": 2,
                    "value_heads": 2,
                    "key_dim": 128,
                    "value_dim": 128,
                    "dtype": "fp16",
                    "varlen": True,
                    "mean_len": 1024,
                    "gate_source": "g",
                    "gate_function": "logsigmoid",
                    "initial_state": "none",
                    "output_final_state": False,
                    "qk_l2norm": False,
                    "demo_model": False,
                    "conv_kernel": 4,
                    "seed": 44,
                    "scale": 0.1,
                    "accuracy_tensors": ["o"],
                    "accuracy_thresholds": accuracy_thresholds,
                    "cu_seqlens": [0, 1024, 2048, 3991],
                },
            },
        )

        def metrics_for_contract(contract):
            threshold_fields = {
                "o": ("output_tol", "output_cos_min"),
                "dq": ("grad_tol", "grad_cos_min"),
                "dk": ("grad_tol", "grad_cos_min"),
                "dv": ("grad_tol", "grad_cos_min"),
                "dbeta": ("beta_grad_tol", "beta_grad_cos_min"),
                "dg": ("gate_grad_tol", "gate_grad_cos_min"),
            }
            metrics = []
            for tensor in contract["accuracy_tensors"]:
                tol_field, cos_min_field = threshold_fields[tensor]
                metrics.append(
                    {
                        **metric,
                        "tensor": tensor,
                        "tol": accuracy_thresholds[tol_field],
                        "cos_min": accuracy_thresholds[cos_min_field],
                    }
                )
            return metrics

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "accuracy-reports"
            reports.mkdir()
            for platform, soc in (("a2", "ascend910b"), ("a5", "ascend950")):
                platform_attempt = str((platform_attempts or {}).get(platform, run_attempt))
                platform_reports = reports
                if nested_artifacts:
                    platform_reports = reports / (
                        f"npu-ci-result-{platform}-{run_id}-{platform_attempt}"
                    )
                    platform_reports.mkdir()
                metadata = {
                    "platform": platform,
                    "soc": soc,
                    "head_sha": sha,
                    "run_id": run_id,
                    "run_attempt": platform_attempt,
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
                        "total": 4,
                        "passed": 4,
                        "failed": 0,
                        "not_run": 0,
                        "accuracy_total": 4,
                        "accuracy_passed": 4,
                        "accuracy_failed": 0,
                        "accuracy_not_run": 0,
                    },
                    "cases": [
                        {
                            "name": case["name"],
                            "status": "passed",
                            "return_code": 0,
                            "accuracy_check": True,
                            "accuracy_status": "passed",
                            "contract": dict(case["contract"]),
                            "metrics": metrics_for_contract(case["contract"]),
                        }
                        for case in required_cases
                    ],
                }
                diagnostic = {
                    "schema": "npu-ci-diagnostics-v1",
                    "metadata": metadata,
                    "status": "success",
                    "execution": {"status": "success", "exit_code": 0},
                    "accuracy": {
                        "status": "success",
                        "total": 4,
                        "passed": 4,
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
                stage_report = {
                    "schema": "npu-ci-stage-report-v1",
                    "complete": True,
                    "metadata": metadata,
                    "status": "success",
                    "stages": {
                        stage: {"status": "success", "exit_code": 0, "reason": ""}
                        for stage in STAGE_KEYS
                    },
                }
                if stage_failure and platform == "a5":
                    failed_index = STAGE_KEYS.index(stage_failure)
                    stage_report["status"] = "failure"
                    stage_report["stages"][stage_failure] = {
                        "status": "failure",
                        "exit_code": 1,
                        "reason": f"{stage_failure} failed",
                    }
                    for skipped_stage in STAGE_KEYS[failed_index + 1 :]:
                        stage_report["stages"][skipped_stage] = {
                            "status": "skipped",
                            "exit_code": None,
                            "reason": f"prerequisite {stage_failure} failed",
                        }
                    result["status"] = "failure"
                    diagnostic.update(status="failure")
                    diagnostic["execution"] = {"status": "failure", "exit_code": 1}
                    diagnostic["diagnostics"]["compile"] = [
                        {
                            "headline": "stage failed",
                            "lines": [f"{stage_failure}: error: stage failed"],
                            "occurrences": 1,
                        }
                    ]
                    diagnostic["reproduction"] = [
                        f"CI_STAGE={stage_failure} CI_MODE=quick bash ci/run_ci_container.sh"
                    ]
                if invalid_stage_sequence and platform == "a5":
                    stage_report["status"] = "failure"
                    stage_report["stages"]["environment-contracts"] = {
                        "status": "failure",
                        "exit_code": 1,
                        "reason": "environment contracts failed",
                    }
                if skipped_required_stage and platform == "a5":
                    stage_report["stages"]["standalone-layout"] = {
                        "status": "skipped",
                        "exit_code": None,
                        "reason": "stage is disabled by the current CI configuration",
                    }
                if accuracy_failure and platform == "a5":
                    result["status"] = "failure"
                    report["summary"].update(accuracy_passed=3, accuracy_failed=1)
                    report["cases"][0].update(
                        accuracy_status="failed",
                        metrics=[{**metric, "allclose": False, "max_abs": 0.125}],
                    )
                    diagnostic.update(status="failure")
                    diagnostic["execution"] = {"status": "failure", "exit_code": 1}
                    diagnostic["accuracy"].update(
                        status="failure",
                        passed=3,
                        failed=1,
                        failures=[
                            {
                                "name": "case1_current_default",
                                "status": "failed",
                                "return_code": 1,
                                "metrics": [
                                    {
                                        "output": "o",
                                        "status": "failed",
                                        "details": {
                                            "finite": True,
                                            "allclose": False,
                                            "cosine_ok": True,
                                            "tol": 0.005,
                                            "max_abs": 0.125,
                                        },
                                    }
                                ],
                            }
                        ],
                    )
                    diagnostic["reproduction"] = [
                        "CI_STAGE=gdr-example-st CI_MODE=quick bash ci/run_ci_container.sh"
                    ]
                    failed_index = STAGE_KEYS.index("gdr-example-st")
                    stage_report["status"] = "failure"
                    stage_report["stages"]["gdr-example-st"] = {
                        "status": "failure",
                        "exit_code": 1,
                        "reason": "accuracy check failed",
                    }
                    for skipped_stage in STAGE_KEYS[failed_index + 1 :]:
                        stage_report["stages"][skipped_stage] = {
                            "status": "skipped",
                            "exit_code": None,
                            "reason": "prerequisite gdr-example-st failed",
                        }
                if missing_required_case and platform == "a5":
                    report["cases"] = report["cases"][1:]
                    report["summary"].update(
                        total=3,
                        passed=3,
                        accuracy_total=3,
                        accuracy_passed=3,
                    )
                    diagnostic["accuracy"].update(total=3, passed=3)
                if altered_required_contract and platform == "a5":
                    report["cases"][0]["contract"]["tokens"] = 4086
                if invalid_required_boundaries and platform == "a5":
                    report["cases"][3]["contract"]["cu_seqlens"] = [0, 0, 0, 3991]
                if missing_required_metric and platform == "a5":
                    report["cases"][1]["metrics"].pop()
                if duplicate_required_metric and platform == "a5":
                    report["cases"][1]["metrics"].append(
                        dict(report["cases"][1]["metrics"][0])
                    )
                if altered_metric_threshold and platform == "a5":
                    report["cases"][1]["metrics"][0]["tol"] = 1.0
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
                        accuracy_passed=3,
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
                        passed=3,
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
                if ops:
                    diagnostic["accuracy"] = {
                        "status": "not_available",
                        "total": 0,
                        "passed": 0,
                        "failed": 0,
                        "not_run": 0,
                        "failures": [],
                    }
                    for skipped_stage in STAGE_KEYS[2:]:
                        stage_report["stages"][skipped_stage] = {
                            "status": "skipped",
                            "exit_code": None,
                            "reason": "not selected by CI_STAGE=opp-package",
                        }
                for name, payload in (
                    (f"npu-ci-result-{platform}.json", result),
                    (f"gdr_accuracy_report-{platform}.json", report),
                    (f"npu-ci-diagnostics-{platform}.json", diagnostic),
                    (f"npu-ci-stages-{platform}.json", stage_report),
                ):
                    if (
                        missing_stage_report
                        and platform == "a5"
                        and name.startswith("npu-ci-stages-")
                    ):
                        continue
                    if ops and name.startswith("gdr_accuracy_report-"):
                        continue
                    (platform_reports / name).write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )

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
  const publishNpuCiStatus = require(process.env.NPU_PUBLISH_SCRIPT_PATH);
  await publishNpuCiStatus({
    github: globalThis.github,
    context: globalThis.context,
    core: globalThis.core,
  });
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
                    "NPU_PUBLISH_SCRIPT_PATH": str(PUBLISH_SCRIPT_PATH),
                    "NPU_CI_MODE": "quick",
                    "NPU_OPS": ops,
                    "NPU_PR_NUMBER": "527",
                    "NPU_COMMENT_ID": "123",
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

    def _run_prepare_script(
        self,
        statuses,
        *,
        mode="quick",
        ops="",
        workflow_run_status="completed",
        run_attempt="1",
    ):
        sha = "a" * 40
        wrapper = """
const fs = require('fs');
const record = (kind, payload) => fs.appendFileSync(
  process.env.RECORD_FILE,
  JSON.stringify({ kind, payload }) + '\\n',
  'utf8'
);
const statusEndpoint = async () => {};
const commentEndpoint = async () => {};
globalThis.context = {
  actor: 'ci-admin',
  eventName: 'workflow_dispatch',
  payload: { inputs: {
    pr_number: '527',
    ci_mode: process.env.PREPARE_MODE,
    ops: process.env.PREPARE_OPS,
  } },
  repo: { owner: 'example', repo: 'repo' },
  runId: 99,
  runNumber: 12,
  serverUrl: 'https://github.com',
};
globalThis.github = {
  paginate: async (endpoint) => endpoint === statusEndpoint
    ? JSON.parse(process.env.PREPARE_STATUSES)
    : [],
  rest: {
    actions: {
      getWorkflowRun: async () => ({
        data: { status: process.env.PREPARE_WORKFLOW_RUN_STATUS },
      }),
    },
    issues: {
      createComment: async (payload) => {
        record('comment', payload);
        return { data: { id: 123 } };
      },
      listComments: commentEndpoint,
      updateComment: async (payload) => record('comment', payload),
    },
    pulls: {
      get: async () => ({ data: {
        draft: false,
        base: { ref: 'main' },
        head: { sha: process.env.PREPARE_HEAD_SHA, repo: { full_name: 'example/repo' } },
      } }),
    },
    repos: {
      createCommitStatus: async (payload) => record('status', payload),
      getBranch: async () => ({ data: { commit: { sha: 'c'.repeat(40) } } }),
      getCollaboratorPermissionLevel: async () => ({ data: { permission: 'admin' } }),
      listCommitStatusesForRef: statusEndpoint,
    },
  },
};
globalThis.core = {
  notice: (message) => record('notice', message),
  setFailed: (message) => record('failed', message),
  setOutput: (name, value) => record('output', { name, value }),
  warning: (message) => record('warning', message),
};
(async () => {
""" + self.prepare_script + """
})().catch((error) => {
  console.error(error && error.stack || error);
  process.exitCode = 1;
});
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record_file = root / "records.jsonl"
            env = os.environ.copy()
            env.update(
                {
                    "NPU_CI_WORKFLOW_SHA": "b" * 40,
                    "PREPARE_HEAD_SHA": sha,
                    "PREPARE_MODE": mode,
                    "PREPARE_OPS": ops,
                    "NPU_CI_RUN_ATTEMPT": run_attempt,
                    "PREPARE_WORKFLOW_RUN_STATUS": workflow_run_status,
                    "PREPARE_STATUSES": json.dumps(statuses),
                    "RECORD_FILE": str(record_file),
                }
            )
            completed = subprocess.run(
                ["node"],
                input=wrapper,
                text=True,
                encoding="utf-8",
                env=env,
                capture_output=True,
                check=False,
            )
            records = [
                json.loads(line)
                for line in record_file.read_text(encoding="utf-8").splitlines()
            ]
            return completed, records

    def _assert_statuses(self, records, expected_states):
        statuses = [item["payload"] for item in records if item["kind"] == "status"]
        self.assertEqual([item["context"] for item in statuses], list(STATUS_CONTEXTS))
        self.assertEqual([item["state"] for item in statuses], expected_states)
        self.assertNotIn("pending", [item["state"] for item in statuses])

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
        self.assertIn("--require-stage-report", run_script)
        self.assertIn('cat "$CI_SUMMARY_FILE" >>"$GITHUB_STEP_SUMMARY"', run_script)
        self.assertIn('elif wait "$ci_pid"; then', run_script)

    def test_status_contexts_are_consistent_across_workflows_and_protection(self):
        sources = (
            WORKFLOW_PATH.read_text(encoding="utf-8"),
            DEFAULT_STATUS_WORKFLOW_PATH.read_text(encoding="utf-8"),
            self.finalize_script,
            BRANCH_PROTECTION_PATH.read_text(encoding="utf-8"),
        )
        for status_context in STATUS_CONTEXTS:
            for source in sources:
                self.assertIn(status_context, source)
        for retired_context in (
            "NPU CI / A2+A5 " + "手动验证",
            "NPU CI / A2+A5 " + "精度检查",
        ):
            for source in sources:
                self.assertNotIn(retired_context, source)
        self.assertIn(CONTRACT_TEST_CONTEXT, CONTRACT_TEST_WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.assertIn(CONTRACT_TEST_CONTEXT, BRANCH_PROTECTION_PATH.read_text(encoding="utf-8"))
        self.assertIn(CONTRACT_TEST_CONTEXT, (REPO_ROOT / "docs" / "repository-rules.md").read_text(encoding="utf-8"))

    def test_stage_order_and_reproduction_selector_are_explicit(self):
        script = RUN_CHECKS_PATH.read_text(encoding="utf-8")
        positions = [script.index(f"run_pipeline_stage \\\n    {stage}") for stage in STAGE_KEYS]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("prerequisite stage ${failed_stage} failed", script)
        self.assertIn("CI_STAGE=%q", script)
        self.assertIn("CI_OPS=%q", script)
        self.assertIn("CI_IMAGE=%q CI_DOCKERFILE=%q CI_REQUIRE_PRELOADED_IMAGE=%q", script)
        self.assertIn('image="fla-npu-ci:9.1.0-950"', script)
        self.assertIn('dockerfile="ci/Dockerfile.ascend950"', script)
        self.assertIn("manage_npu_ci_stage_report.py", script)
        self.assertLess(
            script.index('manage_npu_ci_stage_report.py --output "$stage_report_file" init'),
            script.index('ci_tmpdir="$(select_ci_tmpdir)"'),
        )
        self.assertIn("finalize_bootstrap_failure", script)
        self.assertIn("quick|full)", script)
        self.assertNotIn("bash gdn-verify.sh", script)

    def test_ci_contract_tests_are_wired_with_python_and_node(self):
        run_checks = RUN_CHECKS_PATH.read_text(encoding="utf-8")
        workflow = CONTRACT_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")
        discovery = "python3 -m unittest discover -s ci/tests -p 'test_*.py' -b"
        self.assertIn(discovery, run_checks)
        self.assertIn("bash ci/tests/test_run_checks_stage_pipeline.sh", run_checks)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertIn("actions/setup-node@v6", workflow)
        self.assertIn("node-version: '24'", workflow)
        self.assertIn("python -m unittest discover -s ci/tests -p 'test_*.py' -b", workflow)
        self.assertIn("bash ci/tests/test_run_checks_stage_pipeline.sh", workflow)

    def test_scoped_request_selects_only_opp_package_stage(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        matrix_env = workflow["jobs"]["ascend-npu"]["env"]
        self.assertEqual(
            matrix_env["CI_STAGE"],
            "${{ needs.prepare.outputs.ops == '' && 'all' || 'opp-package' }}",
        )

    def test_prepare_and_publisher_share_request_scope_signature(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("normalizedOps ? 'scoped' : 'all'", workflow)
        self.assertIn("(${ciMode},${requestKey},A2+A5)", workflow)
        self.assertIn("allowing a replacement run", workflow)
        self.assertIn("const latestExecutionCi = finalCiStatuses[0]", workflow)
        self.assertIn("latestExecutionMatchesRequest", workflow)
        self.assertIn("allContextsPassedInSameRun", workflow)
        self.assertIn("latestStatus.target_url === latestExecutionCi.target_url", workflow)
        self.assertNotIn("finalCiStatuses.find", workflow)
        self.assertIn("normalizedOps ? 'scoped' : 'all'", self.finalize_script)
        self.assertIn("(${mode},${requestKey},A2+A5)", self.finalize_script)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_does_not_reuse_final_success_with_failed_or_missing_sibling(self):
        target_url = "https://github.com/example/repo/actions/runs/88"

        def statuses_with(*, failed_context=None, missing_context=None):
            return [
                {
                    "id": index + 1,
                    "context": context_name,
                    "state": "failure" if context_name == failed_context else "success",
                    "description": f"result (quick,all,A2+A5)",
                    "target_url": target_url,
                    "created_at": f"2026-09-14T00:00:{index:02d}Z",
                }
                for index, context_name in enumerate(STATUS_CONTEXTS)
                if context_name != missing_context
            ]

        variants = {
            "failed": statuses_with(failed_context=STATUS_CONTEXTS[2]),
            "missing": statuses_with(missing_context=STATUS_CONTEXTS[2]),
        }
        for name, statuses in variants.items():
            with self.subTest(name=name):
                completed, records = self._run_prepare_script(statuses)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                outputs = {
                    item["payload"]["name"]: item["payload"]["value"]
                    for item in records
                    if item["kind"] == "output"
                }
                self.assertEqual(outputs["should_run"], "true")
                pending = [item for item in records if item["kind"] == "status"]
                self.assertEqual(len(pending), 7)
                self.assertTrue(all(item["payload"]["state"] == "pending" for item in pending))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_reuses_only_complete_same_run_success(self):
        target_url = "https://github.com/example/repo/actions/runs/88"
        statuses = [
            {
                "id": index + 1,
                "context": context_name,
                "state": "success",
                "description": "result (quick,all,A2+A5)",
                "target_url": target_url,
                "created_at": f"2026-09-14T00:00:{index:02d}Z",
            }
            for index, context_name in enumerate(STATUS_CONTEXTS)
        ]

        completed, records = self._run_prepare_script(statuses)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = {
            item["payload"]["name"]: item["payload"]["value"]
            for item in records
            if item["kind"] == "output"
        }
        self.assertEqual(outputs["should_run"], "false")
        self.assertFalse(any(item["kind"] == "status" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_does_not_reuse_older_scoped_success(self):
        op_a = "chunk_fwd_o"
        op_b = "chunk_fwd_h"
        key_a = hashlib.sha256(op_a.encode()).hexdigest()[:10]
        key_b = hashlib.sha256(op_b.encode()).hexdigest()[:10]
        statuses = [
            {
                "id": 1,
                "context": SCOPED_STATUS_CONTEXT,
                "state": "success",
                "description": f"result (quick,{key_a},A2+A5)",
                "target_url": "https://github.com/example/repo/actions/runs/80",
                "created_at": "2026-09-14T00:00:00Z",
            },
            {
                "id": 2,
                "context": SCOPED_STATUS_CONTEXT,
                "state": "failure",
                "description": f"result (quick,{key_b},A2+A5)",
                "target_url": "https://github.com/example/repo/actions/runs/81",
                "created_at": "2026-09-14T00:00:01Z",
            },
        ]

        completed, records = self._run_prepare_script(statuses, ops=op_a)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = {
            item["payload"]["name"]: item["payload"]["value"]
            for item in records
            if item["kind"] == "output"
        }
        self.assertEqual(outputs["should_run"], "true")
        pending = [item for item in records if item["kind"] == "status"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["payload"]["context"], SCOPED_STATUS_CONTEXT)

    def test_prepare_does_not_treat_current_rerun_as_an_existing_active_run(self):
        current_run_pending = [
            {
                "id": 1,
                "context": STATUS_CONTEXTS[-1],
                "state": "pending",
                "description": "等待执行 (quick,all,A2+A5)",
                "target_url": "https://github.com/example/repo/actions/runs/99",
                "created_at": "2026-09-14T00:00:00Z",
            }
        ]
        completed, records = self._run_prepare_script(
            current_run_pending,
            workflow_run_status="in_progress",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = {
            item["payload"]["name"]: item["payload"]["value"]
            for item in records
            if item["kind"] == "output"
        }
        self.assertEqual(outputs["should_run"], "true")
        pending = [item for item in records if item["kind"] == "status"]
        self.assertEqual(len(pending), len(STATUS_CONTEXTS))

    def test_prepare_rerun_does_not_reuse_previous_attempt_success(self):
        target_url = "https://github.com/example/repo/actions/runs/99"
        previous_success = [
            {
                "id": index + 1,
                "context": context_name,
                "state": "success",
                "description": "通过 (quick,all,A2+A5)",
                "target_url": target_url,
                "created_at": f"2026-09-14T00:00:{index:02d}Z",
            }
            for index, context_name in enumerate(STATUS_CONTEXTS)
        ]
        completed, records = self._run_prepare_script(
            previous_success,
            run_attempt="2",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = {
            item["payload"]["name"]: item["payload"]["value"]
            for item in records
            if item["kind"] == "output"
        }
        self.assertEqual(outputs["should_run"], "true")
        pending = [item for item in records if item["kind"] == "status"]
        self.assertEqual(len(pending), len(STATUS_CONTEXTS))

    def test_publisher_writes_final_status_after_stage_statuses(self):
        script = self.finalize_script
        stage_publish = script.index("const stagePublishResults = await Promise.allSettled")
        final_publish = script.index("await publishCommitStatus(finalStatusPayload)")
        self.assertLess(stage_publish, final_publish)

    def test_finalize_has_terminal_status_fallback(self):
        fallback = next(
            step
            for step in self.finalize_steps
            if step.get("name") == "状态发布异常时收敛全部分项"
        )
        self.assertIn("statuses_published != 'true'", fallback["if"])
        for status_context in STATUS_CONTEXTS:
            self.assertIn(status_context, fallback["with"]["script"])
        self.assertIn("state: 'failure'", fallback["with"]["script"])

    def test_platform_artifact_contains_structured_diagnostics(self):
        upload = next(
            step
            for step in self.matrix_steps
            if step.get("uses") == "actions/upload-artifact@v6"
        )
        paths = upload["with"]["path"]
        self.assertIn("CI_ACCURACY_REPORT_FILE", paths)
        self.assertIn("CI_DIAGNOSTIC_FILE", paths)
        self.assertIn("CI_STAGE_REPORT_FILE", paths)
        self.assertIn("CI_SUMMARY_FILE", paths)
        self.assertNotIn("CI_RAW_LOG_FILE", paths)

    def test_partial_rerun_uses_latest_available_attempt_per_platform(self):
        download = next(
            step
            for step in self.finalize_steps
            if step.get("uses") == "actions/download-artifact@v6"
        )
        self.assertEqual(
            download["with"]["pattern"],
            "npu-ci-result-*-${{ github.run_id }}-*",
        )
        self.assertFalse(download["with"]["merge-multiple"])
        completed, _, records = self._run_finalize_script(
            expected_run_attempt="2",
            platform_attempts={"a2": "1", "a5": "2"},
            nested_artifacts=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * len(STATUS_CONTEXTS))

    def test_cleanup_runs_before_status_capture_and_upload(self):
        names = [step.get("name", "") for step in self.matrix_steps]
        cleanup = next(index for index, name in enumerate(names) if name.startswith("清理 "))
        record = next(index for index, name in enumerate(names) if name.startswith("记录 "))
        upload = next(index for index, name in enumerate(names) if name.startswith("上传 "))
        self.assertLess(cleanup, record)
        self.assertLess(record, upload)

    def test_finalize_loader_uses_trusted_workflow_revision(self):
        checkout = next(
            step
            for step in self.finalize_steps
            if step.get("name") == "检出受信任的状态发布脚本"
        )
        self.assertEqual(checkout["uses"], "actions/checkout@v5")
        self.assertEqual(checkout["with"]["ref"], "${{ github.workflow_sha }}")
        self.assertEqual(
            checkout["with"]["sparse-checkout"],
            ".github/scripts/publish_npu_ci_status.js",
        )
        self.assertFalse(checkout["with"]["persist-credentials"])
        self.assertIn("publish_npu_ci_status.js", self.finalize_loader)
        self.assertNotIn("${{", self.finalize_loader)
        self.assertLess(len(self.finalize_loader), 21000)
        self.assertIn("module.exports = async function", self.finalize_script)

    def test_pr_comment_only_expands_failed_diagnostics(self):
        script = self.finalize_script
        self.assertIn("npu-ci-diagnostics-v1", script)
        self.assertIn("npu-ci-stage-report-v1", script)
        self.assertIn("!platform.stageReportOk", script)
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
        self._assert_statuses(records, ["success"] * 7)
        self.assertIn("NPU CI 通过", comment)
        self.assertIn("4/4", comment)
        self.assertIn("分项结果", comment)
        self.assertNotIn("case1_current_default", comment)
        self.assertLess(len(comment), 48000)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_scoped_diagnostic_does_not_overwrite_required_statuses(self):
        completed, comment, records = self._run_finalize_script(
            ops="chunk_fwd_o"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        statuses = [item["payload"] for item in records if item["kind"] == "status"]
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["context"], SCOPED_STATUS_CONTEXT)
        self.assertEqual(statuses[0]["state"], "success")
        self.assertTrue(
            all(status["context"] not in STATUS_CONTEXTS for status in statuses)
        )
        self.assertIn("chunk_fwd_o", comment)
        self.assertIn("不执行后续无关分项", comment)
        self.assertNotIn("GDR Example/ST（4 条精度用例）", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_scoped_missing_stage_report_reproduces_only_opp_package(self):
        completed, comment, records = self._run_finalize_script(
            ops="chunk_fwd_o",
            missing_stage_report=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        statuses = [item["payload"] for item in records if item["kind"] == "status"]
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["context"], SCOPED_STATUS_CONTEXT)
        self.assertEqual(statuses[0]["state"], "failure")
        self.assertIn("CI_STAGE='opp-package'", comment)
        self.assertIn("CI_OPS='chunk_fwd_o'", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_marks_failed_stage_and_all_downstream_checks_failed(self):
        completed, comment, records = self._run_finalize_script(
            stage_failure="opp-package"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "failure", "failure", "failure", "failure", "failure", "failure"],
        )
        self.assertIn("opp-package: error: stage failed", comment)
        self.assertIn("CI_STAGE=opp-package", comment)
        self.assertIn("未执行", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_success_after_failed_stage(self):
        completed, comment, records = self._run_finalize_script(
            invalid_stage_sequence=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["failure"] * 7)
        self.assertIn("失败后", comment)
        self.assertIn("必须跳过", comment)
        self.assertIn("CI_STAGE='environment-contracts'", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_localizes_unexpected_required_skip(self):
        completed, comment, records = self._run_finalize_script(
            skipped_required_stage=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "failure", "success", "success", "success", "failure"],
        )
        self.assertIn("CI_STAGE='all'", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_fails_all_checks_when_stage_report_is_missing(self):
        completed, comment, records = self._run_finalize_script(
            missing_stage_report=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["failure"] * 7)
        self.assertIn("分项结果未生成", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_missing_required_accuracy_case(self):
        completed, comment, records = self._run_finalize_script(
            missing_required_case=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("缺少必跑精度用例：case1_current_default", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_altered_required_accuracy_contract(self):
        completed, comment, records = self._run_finalize_script(
            altered_required_contract=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn(
            "必跑精度用例契约不匹配：case1_current_default (tokens)",
            comment,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_non_increasing_required_boundaries(self):
        completed, comment, records = self._run_finalize_script(
            invalid_required_boundaries=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn(
            "必跑精度用例契约不匹配：gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16 (cu_seqlens)",
            comment,
        )
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_missing_required_metric(self):
        completed, comment, records = self._run_finalize_script(
            missing_required_metric=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("metric 集合不匹配", comment)
        self.assertIn("CI_STAGE='gdr-example-st'", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_duplicate_required_metric(self):
        completed, comment, records = self._run_finalize_script(
            duplicate_required_metric=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("包含重复 metric", comment)
        self.assertIn("CI_STAGE='gdr-example-st'", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_altered_metric_threshold(self):
        completed, comment, records = self._run_finalize_script(
            altered_metric_threshold=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("metric 阈值不匹配", comment)
        self.assertIn("CI_STAGE='gdr-example-st'", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_localizes_accuracy_failure_and_reproduction(self):
        completed, comment, records = self._run_finalize_script(
            accuracy_failure=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "failure", "failure"],
        )
        self.assertIn("精度异常", comment)
        self.assertIn("allclose=false", comment)
        self.assertIn("CI_STAGE=gdr-example-st", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_fails_closed_for_malformed_items(self):
        completed, comment, records = self._run_finalize_script(malformed=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertTrue(any(item["kind"] == "failed" for item in records))
        self.assertIn("NPU CI 失败", comment)
        self.assertIn("不是对象", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_neutralizes_untrusted_markdown_and_mentions(self):
        completed, comment, records = self._run_finalize_script(malicious=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
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
        self._assert_statuses(records, ["success"] * 6 + ["failure"])
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_report_diagnostic_mismatch(self):
        completed, comment, records = self._run_finalize_script(inconsistent="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("accuracy 与精度报告不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_explains_matrix_artifact_mismatch(self):
        completed, comment, records = self._run_finalize_script(matrix_result="failure")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * 6 + ["failure"])
        self.assertIn("但双平台 artifact 均记录为通过", comment)
        self.assertIn("汇总异常复现", comment)
        self.assertEqual(comment.count("CI_STAGE='all'"), 2)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_incomplete_accuracy_report(self):
        completed, comment, records = self._run_finalize_script(incomplete=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("精度报告未完整结束", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_compile_payload_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="compile")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * 6 + ["failure"])
        self.assertIn("broken compiler input", comment)
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_accuracy_failure_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("精度成功但包含失败用例", comment)
        self.assertIn("hidden-failure", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_case_return_code_mismatch(self):
        completed, comment, records = self._run_finalize_script(invalid_return_code=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "success", "failure"],
        )
        self.assertIn("执行状态与 return_code 不一致", comment)


if __name__ == "__main__":
    unittest.main()
