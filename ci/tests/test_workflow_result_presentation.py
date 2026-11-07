#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEFAULT_STATUS_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "npu-ci-default-status.yml"
)
PUBLISH_SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "publish_npu_ci_status.js"
BRANCH_PROTECTION_PATH = REPO_ROOT / "scripts" / "github" / "apply_branch_protection.sh"
RUN_CHECKS_PATH = REPO_ROOT / "ci" / "run_checks.sh"
RUN_CONTAINER_PATH = REPO_ROOT / "ci" / "run_ci_container.sh"
CONTRACT_TEST_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-contract-tests.yml"
CHECKS_CONFIG_PATH = REPO_ROOT / ".github" / "scripts" / "npu_ci_checks.json"
CHECKS_SCRIPT_PATH = CHECKS_CONFIG_PATH.with_suffix(".js")
CHECKS_CONFIG = json.loads(CHECKS_CONFIG_PATH.read_text(encoding="utf-8"))
STAGE_KEYS = tuple(check["key"] for check in CHECKS_CONFIG["required"][:-1])
STATUS_CONTEXTS = tuple(check["context"] for check in CHECKS_CONFIG["required"])
SCOPED_STATUS_CONTEXT = CHECKS_CONFIG["diagnostic"]["context"]
CONTRACT_TEST_CONTEXT = CHECKS_CONFIG["contract_test"]


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
        publish_script_path=None,
        current_pr=None,
        pr_read_error=False,
        expected_base_ref="main",
        expected_base_sha="c" * 40,
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
  pulls: { get: async () => {
    if (process.env.CURRENT_PR_READ_ERROR === '1') throw new Error('PR API unavailable');
    return { data: JSON.parse(process.env.CURRENT_PR) };
  } },
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
                    "NPU_PUBLISH_SCRIPT_PATH": str(publish_script_path or PUBLISH_SCRIPT_PATH),
                    "NPU_CI_MODE": "quick",
                    "NPU_OPS": ops,
                    "NPU_PR_NUMBER": "527",
                    "NPU_COMMENT_ID": "123",
                    "NPU_EXPECTED_HEAD_SHA": sha,
                    "NPU_EXPECTED_BASE_REF": expected_base_ref,
                    "NPU_EXPECTED_BASE_SHA": expected_base_sha,
                    "CURRENT_PR": json.dumps({
                        "state": "open", "draft": False,
                        "head": {"sha": sha},
                        "base": {"ref": "main", "sha": "c" * 40},
                        **(current_pr or {}),
                    }),
                    "CURRENT_PR_READ_ERROR": "1" if pr_read_error else "0",
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
        workflow_run=None,
        workflow_run_error=False,
        run_attempt="1",
        files=None,
        pr_author="ci-admin-author",
        author_permission="admin",
        expected_snapshot=None,
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
const filesEndpoint = async () => {};
const commentEndpoint = async () => {};
const files = JSON.parse(process.env.PREPARE_FILES);
globalThis.context = {
  actor: 'ci-admin',
  eventName: 'workflow_dispatch',
  payload: { inputs: {
    pr_number: '527',
    ci_mode: process.env.PREPARE_MODE,
    ops: process.env.PREPARE_OPS,
    ...JSON.parse(process.env.PREPARE_EXPECTED_SNAPSHOT),
  } },
  repo: { owner: 'example', repo: 'repo' },
  runId: 99,
  runNumber: 12,
  serverUrl: 'https://github.com',
};
globalThis.github = {
  paginate: async (endpoint) => endpoint === statusEndpoint
    ? JSON.parse(process.env.PREPARE_STATUSES)
    : endpoint === filesEndpoint ? files : [],
  rest: {
    actions: {
      getWorkflowRun: async (payload) => {
        record('workflow_run', payload);
        if (process.env.PREPARE_WORKFLOW_RUN_ERROR === '1') throw new Error('run API unavailable');
        return { data: JSON.parse(process.env.PREPARE_WORKFLOW_RUN) };
      },
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
      listFiles: filesEndpoint,
      get: async () => ({ data: {
        number: 527, changed_files: files.length, state: 'open', draft: false,
        user: { login: process.env.PREPARE_PR_AUTHOR },
        base: { sha: 'c'.repeat(40), ref: 'main' },
        head: { sha: process.env.PREPARE_HEAD_SHA, repo: { full_name: 'example/repo' } },
      } }),
    },
    repos: {
      createCommitStatus: async (payload) => record('status', payload),
      getBranch: async () => ({ data: { name: 'main', protected: true, commit: { sha: 'c'.repeat(40) } } }),
      get: async () => ({ data: { default_branch: 'main' } }),
      getContent: async () => ({data:{type:'file',content:Buffer.from(JSON.stringify({users:['ci-admin']})).toString('base64')}}),
      getCollaboratorPermissionLevel: async ({ username }) => {
        record('permission', username);
        return { data: { permission: username === process.env.PREPARE_PR_AUTHOR
          ? process.env.PREPARE_AUTHOR_PERMISSION : 'read' } };
      },
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
                    "NPU_CI_WORKFLOW_SHA": "c" * 40,
                    "NPU_CI_WORKFLOW_REF": "main",
                    "PREPARE_HEAD_SHA": sha,
                    "PREPARE_MODE": mode,
                    "PREPARE_OPS": ops,
                    "PREPARE_PR_AUTHOR": pr_author,
                    "PREPARE_AUTHOR_PERMISSION": author_permission,
                    "PREPARE_EXPECTED_SNAPSHOT": json.dumps(expected_snapshot or {}),
                    "PREPARE_FILES": json.dumps(files or []),
                    "NPU_CI_RUN_ATTEMPT": run_attempt,
                    "NPU_CI_AUTH_SCRIPT": str(REPO_ROOT / ".github/scripts/npu_ci_authorization.js"),
                    "NPU_CI_CHECKS_SCRIPT": str(CHECKS_SCRIPT_PATH),
                    "PREPARE_WORKFLOW_RUN": json.dumps(workflow_run if workflow_run is not None else {
                        "status": workflow_run_status,
                        "event": "workflow_dispatch",
                        "path": ".github/workflows/ci.yml",
                        "head_sha": "c" * 40,
                        "head_branch": "main",
                        "repository": {"full_name": "example/repo"},
                    }),
                    "PREPARE_WORKFLOW_RUN_ERROR": "1" if workflow_run_error else "0",
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

    def test_platforms_share_container_tmpdir_and_isolate_host_storage(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        job = workflow["jobs"]["ascend-npu"]
        env = job["env"]
        self.assertEqual(
            {item["platform"] for item in job["strategy"]["matrix"]["include"]},
            {"a2", "a5"},
        )
        self.assertEqual(env["CI_JOBS"], "16")
        self.assertEqual(env["CI_CPACK_JOBS"], "16")
        self.assertEqual(env["CI_TMPDIR"], "/tmp/fla-npu-ci")
        identity = "${{ matrix.platform }}-${{ github.run_id }}-${{ github.run_attempt }}"
        self.assertEqual(
            env["CI_HOST_TMPDIR"], f"/home/huangjunzhe/ci/tmp/container-{identity}"
        )
        self.assertEqual(env["TMPDIR"], "/home/huangjunzhe/ci/tmp")
        self.assertEqual(env["CI_CONTAINER_NAME"], f"fla-npu-ci-{identity}")
        self.assertEqual(env["CI_CACHE_ROOT"], "/home/huangjunzhe/ci/cache_root")
        self.assertEqual(env["CI_NPU_LOCK_DIR"], "/home/huangjunzhe/ci/tmp/npu-locks")
        container_script = RUN_CONTAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('mount_args+=(-v "$host_tmpdir:$container_tmpdir")', container_script)
        self.assertIn('CI_TMPDIR must be an absolute container path', container_script)

    def test_npu_matrix_enables_fail_fast(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        job = workflow["jobs"]["ascend-npu"]
        self.assertTrue(job["strategy"]["fail-fast"])

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell test")
    def test_proxy_values_reach_later_steps_and_reject_newlines(self):
        step = next(step for step in self.matrix_steps if step.get("name") == "启动代理并获取端口")
        merge_script = next(
            step["run"] for step in self.matrix_steps if step.get("name") == "现场合并 PR 代码"
        )
        fetch_function = merge_script[
            merge_script.index("fetch_with_retry() {"):merge_script.index("fetch_more_history() {")
        ]
        scripts = {
            "checkout fallback": step["run"],
            "fetch fallback": (
                "set -euo pipefail\ncalls=0\n"
                "timeout() { calls=$((calls + 1)); [[ $calls -gt 1 ]]; }\n"
                "sleep() { :; }\n"
                + fetch_function + "\nfetch_with_retry origin test-ref\n"
            ),
        }
        values = {
            "http_proxy": "http://127.0.0.1:7899",
            "https_proxy": "http://user:password@127.0.0.1:7899",
            "all_proxy": "socks5h://127.0.0.1:7891",
            "no_proxy": "localhost,127.0.0.1,::1",
            "GIT_HTTP_PROXY": "http://127.0.0.1:7899",
            "GIT_HTTPS_PROXY": "http://127.0.0.1:7899",
            "PROXY_HTTP_PORT": "7899", "PROXY_SOCKS_PORT": "7891", "PROXY_MIXED_PORT": "7898",
        }
        mixed = {**values, **{name.upper(): "http://stale.invalid:1" for name in values if name.islower()}}
        mixed["http_proxy"] = ""
        mixed["HTTP_PROXY"] = values["http_proxy"]
        cases = [
            ("lowercase", values, values, False),
            ("uppercase", {name.upper(): value for name, value in values.items()}, values, False),
            ("mixed lowercase wins and empty lowercase falls back", mixed, values, False),
            ("missing optional variables", {}, {"no_proxy": ""}, False),
            ("line feed", {"http_proxy": "http://127.0.0.1:7899\nINJECTED=value"}, {}, True),
            ("carriage return in uppercase", {"HTTPS_PROXY": "http://127.0.0.1:7899\rINJECTED=value"}, {}, True),
        ]
        proxy_names = {*values, "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
        for script_name, script in scripts.items():
            for case_name, proxy_values, expected, invalid in cases:
                with self.subTest(script=script_name, case=case_name), tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    proxy_script = root / "proxy script.sh"
                    # A missing setting must not resurrect an inherited proxy or
                    # the wildcard bypass used during the direct fetch attempt.
                    inherited = {
                        name: "*" if name.lower() == "no_proxy" else "http://inherited.invalid:1"
                        for name in proxy_names if not name.startswith("PROXY_")
                    }
                    proxy_script.write_text(
                        "proxyon() {\n  :\n" + "".join(
                            f"  export {name}={shlex.quote(value)}\n"
                            for name, value in proxy_values.items()
                        ) + "}\n",
                        encoding="utf-8",
                    )
                    github_env = root / "github-env"
                    env = {name: value for name, value in os.environ.items() if name not in proxy_names}
                    env.update(inherited, CI_PROXY_SCRIPT=str(proxy_script), GITHUB_ENV=str(github_env))
                    shell_checks = "\n" + "\n".join(
                        f'[[ "${{{name}-}}" == {shlex.quote(expected.get(name, ""))} ]] || '
                        f'{{ echo "Canonical {name} unavailable to subprocess" >&2; exit 1; }}'
                        for name in values
                    )
                    shell_checks += (
                        "\nfor alias in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY; do\n"
                        '  [[ ! -v "$alias" ]] || { echo "Uppercase proxy alias retained" >&2; exit 1; }\n'
                        "done\n"
                    )
                    completed = subprocess.run(
                        ["bash", "-c", script + "\nbash -c " + shlex.quote(shell_checks)], env=env,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    output = github_env.read_text(encoding="utf-8") if github_env.exists() else ""
                    if invalid:
                        self.assertNotEqual(completed.returncode, 0)
                        self.assertIn("contains a newline", completed.stderr)
                        self.assertNotIn("INJECTED", output)
                        self.assertNotIn("NPU_CI_PROXY_ENABLED", output)
                    else:
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        pairs = [line.split("=", 1) for line in output.splitlines()]
                        keys = [name.casefold() for name, _ in pairs]
                        self.assertEqual(len(keys), len(set(keys)), "Duplicate case-insensitive environment key")
                        self.assertEqual(dict(pairs), {**expected, "NPU_CI_PROXY_ENABLED": "true"})

    def _run_matrix_storage_step(self, name, root, overrides=None):
        step = next(step for step in self.matrix_steps if step["name"].startswith(name))
        job = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]["ascend-npu"]
        substitutions = {
            "${{ matrix.platform }}": "a5",
            "${{ github.run_id }}": "42",
            "${{ github.run_attempt }}": "1",
        }
        env = os.environ.copy()
        env.update(
            {
                "CI_IMAGE": "ci-test:latest",
                "CI_ACCURACY_PLATFORM": "a5",
                "GITHUB_RUN_ID": "42",
                "GITHUB_RUN_ATTEMPT": "1",
                "CI_RAW_LOG_FILE": ".ci-tmp/full.log",
                "CI_TEST_COMMAND_LOG": str(root / "commands.jsonl"),
                "CI_TEST_PYTHON": sys.executable,
                "CI_TEST_DOCKER_FAILURE": "",
            }
        )
        for key in (
            "CI_TMPDIR", "CI_HOST_TMPDIR", "TMPDIR", "CI_CACHE_ROOT",
            "CI_NPU_LOCK_DIR", "CI_CONTAINER_NAME",
        ):
            value = job["env"][key]
            for expression, replacement in substitutions.items():
                value = value.replace(expression, replacement)
            env[key] = value
        env.update(overrides or {})
        # Rebase the deployment prefix into a sandbox; execute the workflow's
        # shell and its guards unchanged, with Docker and removals only recorded.
        deployed_root = "/home/huangjunzhe/ci/"
        sandbox_root = str(root / "runner storage") + "/"
        script = step["run"].replace(deployed_root, sandbox_root)
        for key, value in env.items():
            env[key] = value.replace(deployed_root, sandbox_root)
        wrapper = r"""
record_command() {
    "$CI_TEST_PYTHON" - "$@" <<'PY'
import json
import os
import sys
with open(os.environ["CI_TEST_COMMAND_LOG"], "a", encoding="utf-8") as output:
    output.write(json.dumps(sys.argv[1:]) + "\n")
PY
}
docker() {
    record_command docker "$@"
    if [[ "${CI_TEST_DOCKER_FAILURE:-}" == "${1:-}" ]]; then
        return 39
    fi
}
rm() { record_command rm "$@"; }
"""
        completed = subprocess.run(
            ["bash", "-c", wrapper + "\n" + script],
            cwd=root,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        log = root / "commands.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return completed, records, env

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("make"), "Linux make test")
    def test_prepare_build_honors_ci_parallelism_and_preserves_default(self):
        wrapper = r"""
cat() {
    if [[ "$1" == /proc/cpuinfo ]]; then
        printf 'processor : 0\nprocessor : 1\nprocessor : 2\nprocessor : 3\n'
    else
        command cat "$@"
    fi
}
cmake() {
    "$CI_TEST_PYTHON" - <<'PY'
from pathlib import Path
Path("Makefile").write_text("prepare_build:\n\t@printf '%s\\n' '$(MAKEFLAGS)' > \"$$CI_TEST_MAKE_LOG\"\n")
PY
}
export -f cat cmake
exec bash "$CI_TEST_PREPARE_SCRIPT" -s "$CI_TEST_SOURCE" -b "$CI_TEST_BUILD"
"""
        for parallelism, expected in ((None, 8), ("", 8), ("16", 16)):
            with self.subTest(parallelism=parallelism), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                make_log = root / "make-flags"
                env = os.environ.copy()
                env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
                env.update(
                    MAKEFLAGS="-j64",
                    CI_TEST_PYTHON=sys.executable,
                    CI_TEST_PREPARE_SCRIPT=str(REPO_ROOT / "cmake/scripts/prepare.sh"),
                    CI_TEST_SOURCE=str(REPO_ROOT),
                    CI_TEST_BUILD=str(root / "build"),
                    CI_TEST_MAKE_LOG=str(make_log),
                )
                if parallelism is not None:
                    env["CMAKE_BUILD_PARALLEL_LEVEL"] = parallelism
                completed = subprocess.run(
                    ["bash", "-c", wrapper], env=env,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                flags = shlex.split(make_log.read_text(encoding="utf-8"))
                self.assertIn(f"-j{expected}", flags)
                self.assertNotIn("-j64", flags)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_default_parallelism_reaches_opp_build_and_adapter_ninja_environment(self):
        fixture = (REPO_ROOT / "ci/tests/test_run_checks_stage_pipeline.sh").read_text(encoding="utf-8")
        fixture = fixture.split("\nrun_case ", 1)[0]
        fixture = fixture.replace("CI_JOBS=1", "CI_JOBS=")
        fixture = fixture.replace("CI_CPACK_JOBS=1", "CI_CPACK_JOBS=")
        fixture = fixture.replace(
            "bash() {\n",
            """bash() {
    if [[ "${1:-}" == "build.sh" ]]; then
        "$CI_TEST_REAL_PYTHON3" -c 'import json, os, sys; print(json.dumps({"args": sys.argv[1:], "env": {key: os.environ.get(key) for key in ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "MAKEFLAGS", "FLA_NPU_BUILD_ARGS")}}))' "$@" >>"$CI_TEST_BUILD_ENV_LOG"
    fi
""",
        )
        fixture += "\nrun_case parallelism all false 0 success success,success,success,success,success true\n"
        for soc in ("ascend910b", "ascend950"):
            with self.subTest(soc=soc), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "ci/tests").mkdir(parents=True)
                for source in (
                    RUN_CHECKS_PATH,
                    REPO_ROOT / "ci/classify_changes.py",
                    REPO_ROOT / "ci/manage_npu_ci_stage_report.py",
                ):
                    shutil.copyfile(source, root / "ci" / source.name)
                (root / "ci/tests/parallelism.sh").write_text(
                    fixture.replace("ascend910b", soc), encoding="utf-8"
                )
                build_log = root / "build-environment.jsonl"
                env = os.environ.copy()
                for key in ("CI_JOBS", "CI_CPACK_JOBS", "MAX_JOBS", "MAKEFLAGS", "CMAKE_BUILD_PARALLEL_LEVEL", "FLA_NPU_BUILD_ARGS"):
                    env.pop(key, None)
                inherited_build_args = "-O3 -j64" if soc == "ascend950" else ""
                if inherited_build_args:
                    env["FLA_NPU_BUILD_ARGS"] = inherited_build_args
                env.update({"CI_TEST_REAL_PYTHON3": sys.executable, "CI_TEST_BUILD_ENV_LOG": str(build_log)})
                completed = subprocess.run(
                    ["bash", "-c", "source() { :; }; export -f source; exec bash ci/tests/parallelism.sh"],
                    cwd=root, env=env, text=True, encoding="utf-8", capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                builds = [json.loads(line) for line in build_log.read_text().splitlines()]
                self.assertTrue(any(item["args"] == ["scripts/build_wheel.py"] for item in builds))
                for build in builds:
                    self.assertEqual(build["env"], {
                        "MAX_JOBS": "16", "CMAKE_BUILD_PARALLEL_LEVEL": "16", "MAKEFLAGS": "-j16",
                        "FLA_NPU_BUILD_ARGS": (inherited_build_args + " -j16").strip(),
                    })
                # pip wheel rebuilds the OPP through the root setup.py. Run
                # that actual command builder with the exported child-process
                # environment, without invoking a compiler or setuptools.
                setup_path = REPO_ROOT / "setup.py"
                tree = ast.parse(setup_path.read_text(encoding="utf-8"))
                builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_run_package")
                wheel_commands = []
                artifacts = runpy.run_path(str(REPO_ROOT / "scripts/fla_npu_artifacts.py"))
                namespace = {
                    "os": os, "shutil": shutil, "shlex": shlex,
                    "REPO_ROOT": root, "get_soc": artifacts["get_soc"], "DEFAULT_VENDOR_NAME": "fla_npu",
                    "_check_ops_filter": lambda value: None,
                    "_run": lambda command, cwd: wheel_commands.append(command),
                    "_find_single_run_package": lambda: root / "fake.run",
                }
                exec(compile(ast.Module(body=[builder], type_ignores=[]), str(setup_path), "exec"), namespace)
                with mock.patch.dict(os.environ, {**builds[0]["env"], "FLA_NPU_SOC": soc}, clear=True):
                    namespace["_build_run_package"]()
                command = wheel_commands[0]
                self.assertEqual(command[:2], ["bash", "build.sh"])
                self.assertIn(f"--soc={soc}", command)
                self.assertIn("--pkg", command)
                self.assertEqual(command[-1], "-j16")
                if inherited_build_args:
                    self.assertEqual(command[-3:], ["-O3", "-j64", "-j16"])

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_temp_cleanup_preserves_persistent_cache_and_npu_locks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "repo" / ".ci-tmp").mkdir(parents=True)
            completed, records, env = self._run_matrix_storage_step("清理 ", root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            recursive_removals = [item[-1] for item in records if item[:2] == ["rm", "-rf"]]
            self.assertEqual(recursive_removals, [env["CI_HOST_TMPDIR"]])
            self.assertNotIn(env["TMPDIR"], recursive_removals)
            self.assertNotIn(env["CI_CACHE_ROOT"], recursive_removals)
            self.assertNotIn(env["CI_NPU_LOCK_DIR"], recursive_removals)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_a2_cleanup_does_not_remove_static_tmpdir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "repo" / ".ci-tmp").mkdir(parents=True)
            completed, records, env = self._run_matrix_storage_step(
                "清理 ", root, {"CI_ACCURACY_PLATFORM": "a2", "CI_HOST_TMPDIR": "/home/huangjunzhe/ci/tmp/container-a2-42-1"}
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            removals = [item[-1] for item in records if item[:2] == ["rm", "-rf"]]
            self.assertEqual(removals, [env["CI_HOST_TMPDIR"]])
            self.assertNotIn(env["TMPDIR"], removals)

    def test_device_jobs_validate_fetched_pr_locally_after_merge(self):
        device_source = "\n".join(step.get("run", "") for step in self.matrix_steps)
        self.assertNotIn("trusted-ci", device_source)
        self.assertNotIn("rm -rf repo/ci", device_source)
        self.assertNotIn("cp -a", device_source)
        self.assertFalse(any(step.get("uses", "").startswith("actions/github-script@")
                             for step in self.matrix_steps))
        self.assertNotIn("api.github.com", device_source)
        self.assertNotIn("extract_archive", device_source)
        merge_index = next(index for index, step in enumerate(self.matrix_steps)
                           if "git merge --no-ff" in step.get("run", ""))
        guard_index = next(index for index, step in enumerate(self.matrix_steps)
                           if step.get("id") == "validate_pr")
        container_index = next(index for index, step in enumerate(self.matrix_steps)
                               if step.get("id") == "npu_ci")
        self.assertLess(merge_index, guard_index)
        self.assertLess(guard_index, container_index)
        self.assertEqual(self.matrix_steps[guard_index]["working-directory"], "repo")
        # Real local Git tests in test_workflow_source_checkout exercise fixed
        # refs, ancestry, reruns, and denied network access for this guard.

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_admin_author_ci_changes_publish_without_confirmation_parameters(self):
        prepared, initial_records = self._run_prepare_script(
            [], files=[{"filename": ".github/workflows/ci.yml", "status": "modified"}])
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertIn({"kind": "output", "payload": {"name": "should_run", "value": "true"}}, initial_records)
        self.assertEqual([item["payload"] for item in initial_records if item["kind"] == "permission"],
                         ["ci-admin-author"])

        completed, _, records = self._run_finalize_script()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * len(STATUS_CONTEXTS))
        for item in records:
            if item["kind"] == "status":
                self.assertNotIn("ci-ok=", item["payload"]["description"])
                self.assertLessEqual(len(item["payload"]["description"]), 140)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_scoped_publisher_truncates_long_status_descriptions(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp)
            publisher = fixture / PUBLISH_SCRIPT_PATH.name
            publisher.write_bytes(PUBLISH_SCRIPT_PATH.read_bytes())
            (fixture / CHECKS_SCRIPT_PATH.name).write_bytes(CHECKS_SCRIPT_PATH.read_bytes())
            profile = json.loads(CHECKS_CONFIG_PATH.read_text(encoding="utf-8"))
            profile["diagnostic"]["label"] = "long diagnostic label " * 15
            (fixture / CHECKS_CONFIG_PATH.name).write_text(json.dumps(profile), encoding="utf-8")
            completed, _, records = self._run_finalize_script(
                ops="chunk_fwd_o", publish_script_path=publisher)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        statuses = [item["payload"] for item in records if item["kind"] == "status"]
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["context"], SCOPED_STATUS_CONTEXT)
        self.assertEqual(len(statuses[0]["description"]), 140)
        self.assertTrue(statuses[0]["description"].endswith("..."))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_temp_cleanup_rejects_paths_outside_current_platform_run(self):
        for suffix in (
            "", "npu-locks", "container-a2-42-1", "container-a5-41-1",
            "container-a5-42-2", "container-a5-42-1/../npu-locks", "../cache_root",
        ):
            with self.subTest(path=suffix), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "repo" / ".ci-tmp").mkdir(parents=True)
                completed, records, _ = self._run_matrix_storage_step(
                    "清理 ", root,
                    {"CI_HOST_TMPDIR": "/home/huangjunzhe/ci/tmp/" + suffix},
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Refusing to clean an unexpected temporary directory", completed.stderr)
                self.assertFalse(any(item[:2] == ["rm", "-rf"] for item in records))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_cleanup_rejects_symlinks_to_unrelated_data(self):
        for target in ("host-temp", "log"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                log_dir = root / "repo" / ".ci-tmp"
                log_dir.mkdir(parents=True)
                unrelated = root / "unrelated"
                unrelated.mkdir()
                if target == "host-temp":
                    link = root / "runner storage" / "tmp" / "container-a5-42-1"
                    link.parent.mkdir(parents=True)
                    link.symlink_to(unrelated, target_is_directory=True)
                else:
                    (log_dir / "full.log").symlink_to(unrelated / "keep.log")
                completed, records, _ = self._run_matrix_storage_step("清理 ", root)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Refusing to clean", completed.stderr)
                self.assertFalse(any(item[:2] == ["rm", "-rf"] for item in records))

    def test_status_contexts_are_consistent_across_workflows_and_protection(self):
        sources = (
            WORKFLOW_PATH.read_text(encoding="utf-8"),
            DEFAULT_STATUS_WORKFLOW_PATH.read_text(encoding="utf-8"),
            self.finalize_script,
        )
        self.assertEqual(len(STATUS_CONTEXTS), 6)
        self.assertEqual(len(set(STATUS_CONTEXTS)), len(STATUS_CONTEXTS))
        self.assertEqual(CHECKS_CONFIG["required"][-1]["key"], "report-integrity")
        for status_context in STATUS_CONTEXTS:
            for source in sources:
                self.assertNotIn(status_context, source)
        self.assertIn("require('./npu_ci_checks')", self.finalize_script)
        self.assertIn("checks.prepareChecks", self.prepare_script)
        self.assertIn("npu_ci_checks.js", sources[1])
        for retired_context in (
            "NPU CI / A2+A5 " + "手动验证",
            "NPU CI / A2+A5 " + "精度检查",
        ):
            for source in sources:
                self.assertNotIn(retired_context, source)
        self.assertIn(CONTRACT_TEST_CONTEXT, CONTRACT_TEST_WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.assertIn(CONTRACT_TEST_CONTEXT, (REPO_ROOT / "docs" / "repository-rules.md").read_text(encoding="utf-8"))

    def _run_branch_protection(self, *, fallback_token=False, missing_token=False, malformed_checks=False, empty_required=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture_repo = root / "checkout with spaces"
            script = fixture_repo / "scripts/github/apply_branch_protection.sh"
            script.parent.mkdir(parents=True)
            script.write_bytes(BRANCH_PROTECTION_PATH.read_bytes())
            profile = json.loads((REPO_ROOT / ".github/scripts/npu_ci_checks.json").read_text(encoding="utf-8"))
            # Use different values to prove the runtime reads the profile next
            # to its script instead of embedding names or reading from cwd.
            profile["required"][0]["context"] = "fixture / first required check"
            profile["contract_test"] = "fixture / contract check"
            if empty_required:
                profile["required"] = []
            checks = fixture_repo / ".github/scripts/npu_ci_checks.json"
            checks.parent.mkdir(parents=True)
            checks.write_text("not JSON" if malformed_checks else json.dumps(profile), encoding="utf-8")
            mock_bin = root / "mock-bin"
            mock_bin.mkdir()
            curl = mock_bin / "curl"
            curl.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CI_TEST_CURL_RECORD']).write_text(json.dumps({"
                "'args':sys.argv[1:], 'payload':json.load(sys.stdin)}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            curl.chmod(0o755)
            cwd = root / "unrelated working directory"
            cwd.mkdir()
            record_file = root / "curl-request.json"
            env = os.environ.copy()
            env.update(PATH=str(mock_bin) + os.pathsep + env.get("PATH", ""),
                       CI_TEST_CURL_RECORD=str(record_file),
                       GITHUB_REPOSITORY="fixture/project", GITHUB_API_URL="https://github.invalid/api")
            env.pop("GITHUB_TOKEN", None)
            env.pop("GH_TOKEN", None)
            if not missing_token:
                env["GH_TOKEN" if fallback_token else "GITHUB_TOKEN"] = "fixture-token"
            completed = subprocess.run(["bash", str(script), "release-v2"], cwd=cwd, env=env,
                                       text=True, encoding="utf-8", capture_output=True, check=False)
            record = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else None
            return completed, record, profile

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_branch_protection_payload_uses_shared_profile_from_unrelated_cwd(self):
        completed, record, profile = self._run_branch_protection()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIsNotNone(record)
        payload = record["payload"]
        self.assertEqual(payload["required_status_checks"], {
            "strict": True,
            "contexts": [item["context"] for item in profile["required"]] + [profile["contract_test"]],
        })
        self.assertNotIn(profile["diagnostic"]["context"], payload["required_status_checks"]["contexts"])
        self.assertEqual(payload["required_pull_request_reviews"], {
            "dismissal_restrictions": {}, "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True, "require_last_push_approval": True,
            "required_approving_review_count": 2,
            "bypass_pull_request_allowances": {"users":["weinachuan"], "teams":[], "apps":[]},
        })
        self.assertEqual({key: value for key, value in payload.items()
                          if key not in {"required_status_checks", "required_pull_request_reviews"}}, {
            "enforce_admins": True, "restrictions": None, "required_conversation_resolution": True,
            "allow_force_pushes": False, "allow_deletions": False, "block_creations": False,
        })
        self.assertIn("https://github.invalid/api/repos/fixture/project/branches/release-v2/protection", record["args"])
        self.assertIn("Authorization: Bearer fixture-token", record["args"])
        self.assertIn("PUT", record["args"])
        self.assertIn("Applied branch protection to fixture/project:release-v2.", completed.stdout)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_branch_protection_preserves_gh_token_fallback(self):
        completed, record, _ = self._run_branch_protection(fallback_token=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Authorization: Bearer fixture-token", record["args"])

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_branch_protection_missing_token_or_invalid_profile_never_calls_curl(self):
        for options in ({"missing_token": True}, {"malformed_checks": True}, {"empty_required": True}):
            with self.subTest(options=options):
                completed, record, _ = self._run_branch_protection(**options)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(record)

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
        self.assertIn("CI_TMPDIR=%q", script)
        self.assertIn("manage_npu_ci_stage_report.py", script)
        self.assertLess(
            script.index('manage_npu_ci_stage_report.py --output "$stage_report_file" init'),
            script.index('ci_tmpdir="$(select_ci_tmpdir)"'),
        )
        self.assertIn("finalize_bootstrap_failure", script)
        self.assertIn("quick|full)", script)
        self.assertNotIn("bash gdn-verify.sh", script)

        tests_only_body = script.split("run_tests_only()", 1)[1].split(
            "run_gdr_example_st()", 1
        )[0]
        self.assertIn("install_latest_arch_wheel", tests_only_body)
        self.assertNotIn("test.sh", tests_only_body)
        self.assertIn("existing GDR/Example-ST stage", tests_only_body)

    def test_ci_contract_tests_run_outside_the_npu_environment_stage(self):
        run_checks = RUN_CHECKS_PATH.read_text(encoding="utf-8")
        workflow = CONTRACT_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")
        discovery = "python3 -m unittest discover -s ci/tests -p 'test_*.py' -b"
        self.assertNotIn("-s ci/tests", run_checks)
        self.assertNotIn("test_run_checks_stage_pipeline.sh", run_checks)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertIn("actions/setup-node@v6", workflow)
        self.assertIn("node-version: '24'", workflow)
        self.assertIn("python -m unittest discover -s ci/tests -p 'test_*.py' -b", workflow)
        self.assertIn("bash ci/tests/test_run_checks_stage_pipeline.sh", workflow)
        self.assertLess(
            workflow.index("actions/setup-node@v6"),
            workflow.index("python -m unittest discover -s ci/tests"),
        )

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
                self.assertEqual(len(pending), 6)
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
    def test_prepare_reuse_requires_current_base_and_workflow_run_identity(self):
        run = {
            "status": "completed", "event": "workflow_dispatch",
            "path": ".github/workflows/ci.yml", "head_sha": "c" * 40,
            "head_branch": "main", "repository": {"full_name": "example/repo"},
        }
        variants = {
            "old_base": {"workflow_run": {**run, "head_sha": "b" * 40}},
            "old_base_branch": {"workflow_run": {**run, "head_branch": "maintenance"}},
            "other_event": {"workflow_run": {**run, "event": "push"}},
            "other_workflow": {"workflow_run": {**run, "path": ".github/workflows/package.yml"}},
            "other_repository": {"workflow_run": {**run, "repository": {"full_name": "example/other"}}},
            "missing_base": {"workflow_run": {key: value for key, value in run.items() if key != "head_sha"}},
            "missing_branch": {"workflow_run": {key: value for key, value in run.items() if key != "head_branch"}},
            "missing_identity": {"workflow_run": {"status": "completed"}},
            "run_api_error": {"workflow_run_error": True},
        }
        for ops in ("", "chunk_fwd_o"):
            key = hashlib.sha256(ops.encode()).hexdigest()[:10] if ops else "all"
            contexts = [SCOPED_STATUS_CONTEXT] if ops else STATUS_CONTEXTS
            for state in ("success", "pending"):
                statuses = [{
                    "id": index + 1, "context": status_context, "state": state,
                    "description": f"result (quick,{key},A2+A5)",
                    "target_url": "https://github.com/example/repo/actions/runs/88",
                    "created_at": f"2026-09-14T00:00:{index:02d}Z",
                } for index, status_context in enumerate(contexts)]
                for name, options in variants.items():
                    with self.subTest(scope=ops or "all", state=state, variant=name):
                        run_options = dict(options)
                        if "workflow_run" in run_options:
                            run_options["workflow_run"] = {
                                **run_options["workflow_run"],
                                "status": "in_progress" if state == "pending" else "completed",
                            }
                        completed, records = self._run_prepare_script(
                            statuses, ops=ops, workflow_run_status="in_progress", **run_options)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        self.assertIn({"kind": "output", "payload": {"name": "should_run", "value": "true"}}, records)
                        pending = [item["payload"] for item in records if item["kind"] == "status"]
                        self.assertEqual(len(pending), len(contexts))
                        self.assertTrue(all(item["state"] == "pending" for item in pending))
                        self.assertEqual(sum(item["kind"] == "workflow_run" for item in records), 1)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_does_not_follow_other_repository_status_urls(self):
        for target_url in (
            "https://github.com/example/other/actions/runs/88",
            "https://github.com.evil.example/example/repo/actions/runs/88",
        ):
            statuses = [{
                "context": status_context, "state": "success",
                "description": "result (quick,all,A2+A5)",
                "target_url": target_url, "created_at": "2026-09-14T00:00:00Z",
            } for status_context in STATUS_CONTEXTS]
            with self.subTest(target_url=target_url):
                completed, records = self._run_prepare_script(statuses)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn({"kind": "output", "payload": {"name": "should_run", "value": "true"}}, records)
                self.assertFalse(any(item["kind"] == "workflow_run" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_reuses_same_base_pending_run_with_one_api_read(self):
        for ops in ("", "chunk_fwd_o"):
            key = hashlib.sha256(ops.encode()).hexdigest()[:10] if ops else "all"
            statuses = [{
                "context": SCOPED_STATUS_CONTEXT if ops else STATUS_CONTEXTS[-1],
                "state": "pending", "description": f"result (quick,{key},A2+A5)",
                "target_url": "https://github.com/example/repo/actions/runs/88",
                "created_at": "2026-09-14T00:00:00Z",
            }]
            with self.subTest(scope=ops or "all"):
                completed, records = self._run_prepare_script(
                    statuses, ops=ops, workflow_run_status="in_progress")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn({"kind": "output", "payload": {"name": "should_run", "value": "false"}}, records)
                self.assertFalse(any(item["kind"] == "status" for item in records))
                self.assertEqual(sum(item["kind"] == "workflow_run" for item in records), 1)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_prepare_reuses_published_scoped_success_for_admin_author_ci_changes(self):
        ops = "chunk_fwd_o"
        files = [{"filename": "ci/run_checks.sh", "status": "modified"}]
        prepared, initial_records = self._run_prepare_script(
            [], ops=ops, files=files)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        outputs = {item["payload"]["name"]: item["payload"]["value"]
                   for item in initial_records if item["kind"] == "output"}
        self.assertEqual(outputs["should_run"], "true")
        published, _, final_records = self._run_finalize_script(ops=ops)
        self.assertEqual(published.returncode, 0, published.stderr)
        statuses = [item["payload"] for item in final_records if item["kind"] == "status"]
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["context"], SCOPED_STATUS_CONTEXT)
        self.assertEqual(statuses[0]["state"], "success")
        self.assertNotIn("ci-ok=", statuses[0]["description"])

        completed, records = self._run_prepare_script(
            statuses, ops=ops, files=files)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn({"kind": "output", "payload": {"name": "should_run", "value": "false"}}, records)
        self.assertFalse(any(item["kind"] == "status" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_authorized_requester_cannot_run_ci_changes_from_non_admin_author(self):
        completed, records = self._run_prepare_script(
            [], files=[{"filename": "ci/run_checks.sh", "status": "modified"}],
            author_permission="write")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn({"kind": "output", "payload": {"name": "should_run", "value": "true"}}, records)
        self._assert_statuses(records, ["failure"] * len(STATUS_CONTEXTS))
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertTrue(any("ci-admin-author" in message and "Admin" in message for message in failures))
        self.assertEqual([item["payload"] for item in records if item["kind"] == "permission"],
                         ["ci-admin-author"])

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

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
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

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
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
            expected_snapshot={
                "expected_head_sha": "a" * 40,
                "expected_base_ref": "main",
                "expected_base_sha": "c" * 40,
                "expected_workflow_sha": "c" * 40,
            },
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
        self.assertEqual(fallback["env"]["NPU_CI_WORKFLOW_SHA"], "${{ github.workflow_sha }}")
        self.assertIn("github.rest.repos.getContent", fallback["with"]["script"])
        self.assertIn(".github/scripts/npu_ci_checks.json", fallback["with"]["script"])
        self.assertIn("ref: workflowSha", fallback["with"]["script"])
        self.assertNotIn("require(", fallback["with"]["script"])
        self.assertIn("state: 'failure'", fallback["with"]["script"])

    def _run_terminal_fallback(self, *, checks_config=CHECKS_CONFIG, ops=""):
        fallback = next(step for step in self.finalize_steps
                        if step.get("name") == "状态发布异常时收敛全部分项")
        wrapper = """
const records = [];
globalThis.context = {
  serverUrl: 'https://github.com', repo: { owner: 'example', repo: 'repo' }, runId: 42,
};
globalThis.github = { rest: { repos: {
  getContent: async (payload) => {
    records.push({ kind: 'checks-read', payload });
    return { data: { type: 'file', encoding: 'base64',
      content: Buffer.from(process.env.CHECKS_CONFIG_FIXTURE).toString('base64') } };
  },
  createCommitStatus: async (payload) => records.push({ kind: 'status', payload }),
} } };
globalThis.core = { setFailed: (message) => records.push({ kind: 'failed', payload: message }) };
(async () => {
""" + fallback["with"]["script"] + """
  process.stdout.write(JSON.stringify(records));
})().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
"""
        with tempfile.TemporaryDirectory() as temp:
            # Checkout failed: neither ci-control nor any helper/config file is
            # present locally. The workflow-pinned Content API remains usable.
            self.assertFalse((Path(temp) / "ci-control").exists())
            env = os.environ.copy()
            env.pop("NPU_CI_CHECKS_SCRIPT", None)
            env.update(GITHUB_WORKSPACE=temp, NPU_CI_WORKFLOW_SHA="b" * 40,
                       CHECKS_CONFIG_FIXTURE=json.dumps(checks_config),
                       NPU_EXPECTED_HEAD_SHA="a" * 40, NPU_OPS=ops)
            completed = subprocess.run(["node"], input=wrapper, cwd=temp, env=env,
                                       text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        records = json.loads(completed.stdout)
        self.assertEqual([item["payload"] for item in records if item["kind"] == "checks-read"], [{
            "owner": "example", "repo": "repo", "path": ".github/scripts/npu_ci_checks.json",
            "ref": "b" * 40,
        }])
        return records

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_publisher_and_terminal_fallback_share_changed_context_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp)
            publisher = fixture / PUBLISH_SCRIPT_PATH.name
            publisher.write_bytes(PUBLISH_SCRIPT_PATH.read_bytes())
            helper = fixture / CHECKS_SCRIPT_PATH.name
            helper.write_bytes(CHECKS_SCRIPT_PATH.read_bytes())
            profile = json.loads(CHECKS_CONFIG_PATH.read_text(encoding="utf-8"))
            for index, check in enumerate(profile["required"]):
                check["context"] = f"NPU CI / fixture / required {index}"
                check["label"] = f"fixture stage {index}"
            profile["diagnostic"]["context"] = "NPU CI / fixture / diagnostic"
            (fixture / CHECKS_CONFIG_PATH.name).write_text(json.dumps(profile), encoding="utf-8")
            for ops in ("", "chunk_fwd_o"):
                with self.subTest(ops=ops):
                    completed, comment, records = self._run_finalize_script(
                        ops=ops, publish_script_path=publisher)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    published = [item["payload"]["context"] for item in records if item["kind"] == "status"]
                    fallback = self._run_terminal_fallback(checks_config=profile, ops=ops)
                    terminal = [item["payload"]["context"] for item in fallback if item["kind"] == "status"]
                    expected = ([profile["diagnostic"]["context"]] if ops else
                                [check["context"] for check in profile["required"]])
                    self.assertEqual(published, expected)
                    self.assertEqual(terminal, expected)
                    self.assertTrue(all(item["payload"]["state"] == "failure" for item in fallback
                                        if item["kind"] == "status"))
                    self.assertIn(profile["required"][0]["label"], comment)

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

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
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
        self.assertIn("always()", self.matrix_steps[cleanup]["if"])
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
            set(checkout["with"]["sparse-checkout"].splitlines()),
            {".github/scripts/publish_npu_ci_status.js", ".github/scripts/npu_ci_checks.js",
             ".github/scripts/npu_ci_checks.json"},
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
        self._assert_statuses(records, ["success"] * 6)
        self.assertIn("NPU CI 通过", comment)
        self.assertIn("4/4", comment)
        self.assertIn("分项结果", comment)
        self.assertNotIn("case1_current_default", comment)
        self.assertLess(len(comment), 48000)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_publisher_rejects_changed_head_or_pr_lifecycle(self):
        changes = {
            "head_changed": {"head": {"sha": "d" * 40}},
            "retargeted": {"base": {"ref": "maintenance", "sha": "c" * 40}},
            "closed": {"state": "closed"},
            "merged": {"merged": True},
            "merged_at": {"merged_at": "2026-11-04T00:00:00Z"},
            "draft": {"draft": True},
        }
        for ops in ("", "chunk_fwd_o"):
            for name, current_pr in changes.items():
                with self.subTest(scope=ops or "all", change=name):
                    completed, comment, records = self._run_finalize_script(
                        ops=ops, current_pr=current_pr)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    statuses = [item["payload"] for item in records if item["kind"] == "status"]
                    self.assertEqual(len(statuses), 1 if ops else len(STATUS_CONTEXTS))
                    self.assertTrue(all(item["state"] == "failure" for item in statuses))
                    self.assertTrue(any(item["kind"] == "failed" for item in records))
                    self.assertIn("失效", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_publisher_accepts_a_base_commit_advance(self):
        completed, _, records = self._run_finalize_script(
            current_pr={"base": {"ref": "main", "sha": "b" * 40}}
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * len(STATUS_CONTEXTS))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_publisher_fails_closed_without_current_pr_or_target_identity(self):
        for options in (
            {"pr_read_error": True},
            {"expected_base_ref": ""},
        ):
            with self.subTest(options=options):
                completed, _, records = self._run_finalize_script(**options)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self._assert_statuses(records, ["failure"] * len(STATUS_CONTEXTS))

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
            ["success", "failure", "failure", "failure", "failure", "failure"],
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
        self._assert_statuses(records, ["failure"] * 6)
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
            ["success", "success", "failure", "success", "success", "failure"],
        )
        self.assertIn("CI_STAGE='all'", comment)
        self.assertTrue(any(item["kind"] == "failed" for item in records))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_fails_all_checks_when_stage_report_is_missing(self):
        completed, comment, records = self._run_finalize_script(
            missing_stage_report=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["failure"] * 6)
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
            ["success", "success", "success", "success", "failure", "failure"],
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
        self._assert_statuses(records, ["success"] * 5 + ["failure"])
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_report_diagnostic_mismatch(self):
        completed, comment, records = self._run_finalize_script(inconsistent="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "failure"],
        )
        self.assertIn("accuracy 与精度报告不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_explains_matrix_artifact_mismatch(self):
        completed, comment, records = self._run_finalize_script(matrix_result="failure")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * 5 + ["failure"])
        self.assertIn("但双平台 artifact 均记录为通过", comment)
        self.assertIn("汇总异常复现", comment)
        self.assertEqual(comment.count("CI_STAGE='all'"), 2)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_incomplete_accuracy_report(self):
        completed, comment, records = self._run_finalize_script(incomplete=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "failure"],
        )
        self.assertIn("精度报告未完整结束", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_compile_payload_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="compile")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(records, ["success"] * 5 + ["failure"])
        self.assertIn("broken compiler input", comment)
        self.assertIn("顶层状态与 execution/accuracy 不一致", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_hidden_accuracy_failure_in_success(self):
        completed, comment, records = self._run_finalize_script(hidden_payload="accuracy")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "failure"],
        )
        self.assertIn("精度成功但包含失败用例", comment)
        self.assertIn("hidden-failure", comment)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_finalize_script_rejects_case_return_code_mismatch(self):
        completed, comment, records = self._run_finalize_script(invalid_return_code=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_statuses(
            records,
            ["success", "success", "success", "success", "failure", "failure"],
        )
        self.assertIn("执行状态与 return_code 不一致", comment)


if __name__ == "__main__":
    unittest.main()
