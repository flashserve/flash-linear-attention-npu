#!/usr/bin/env python3
"""Pin the branch selection contract for comment and manual NPU CI runs."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
COMMENT_DISPATCH_PATH = (
    REPO_ROOT / ".github" / "workflows" / "npu-ci-comment-dispatch.yml"
)


def _workflow_on(workflow):
    # PyYAML 1.1 treats the YAML 1.2 key ``on`` as boolean ``True``.
    return workflow.get("on", workflow.get(True))


class WorkflowBranchDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.dispatch_workflow = yaml.safe_load(
            COMMENT_DISPATCH_PATH.read_text(encoding="utf-8")
        )
        cls.dispatch_script = next(
            step["with"]["script"]
            for step in cls.dispatch_workflow["jobs"]["dispatch"]["steps"]
            if step.get("uses") == "actions/github-script@v8"
        )

    def test_actual_ci_is_manual_dispatch_only_and_has_internal_requester_input(self):
        workflow = yaml.safe_load(self.ci_text)
        triggers = _workflow_on(workflow)

        self.assertEqual(list(triggers), ["workflow_dispatch"])
        self.assertNotIn("issue_comment:", self.ci_text)
        self.assertIn("requested_by", triggers["workflow_dispatch"]["inputs"])
        self.assertIn("${{ github.ref_name }}", self.ci_text)
        self.assertIn("${{ github.workflow_sha }}", self.ci_text)

    def test_comment_dispatcher_has_write_permission_and_checks_target_workflow(self):
        triggers = _workflow_on(self.dispatch_workflow)
        permissions = self.dispatch_workflow["permissions"]

        self.assertEqual(triggers["issue_comment"]["types"], ["created"])
        self.assertEqual(permissions["actions"], "write")
        self.assertIn("workflow_id: 'ci.yml'", self.dispatch_script)
        self.assertIn("ref: targetRef", self.dispatch_script)
        self.assertIn(".github/workflows/ci.yml", self.dispatch_script)
        self.assertIn("workflow_dispatch", self.dispatch_script)

    def _run_dispatch(self, comment, available_workflows=None):
        encoded_workflow = base64.b64encode(
            b"name: NPU CI\non:\n  workflow_dispatch:\n"
        ).decode("ascii")
        default_workflows = ".github/workflows/ci.yml"
        available = available_workflows or default_workflows
        wrapper = f"""
const fs = require('fs');
const records = [];
const record = (kind, payload) => records.push({{ kind, payload }});
globalThis.context = {{
  actor: 'ci-admin',
  payload: {{
    issue: {{ number: 527 }},
    comment: {{ body: process.env.PR_COMMENT }},
  }},
  repo: {{ owner: 'example', repo: 'repo' }},
}};
globalThis.github = {{
  rest: {{
    repos: {{
      getCollaboratorPermissionLevel: async () => ({{ data: {{ permission: 'admin' }} }}),
      getBranch: async (payload) => ({{ data: {{ name: payload.branch }} }}),
      getContent: async (payload) => {{
        const available = (process.env.AVAILABLE_WORKFLOWS || '{default_workflows}')
          .split(',')
          .map((item) => item.trim())
          .filter(Boolean);
        if (!available.includes(payload.path)) {{
          const error = new Error('Not Found');
          error.status = 404;
          throw error;
        }}
        return {{
          data: {{
            type: 'file',
            encoding: 'base64',
            content: '{encoded_workflow}',
          }},
        }};
      }},
    }},
    pulls: {{
      get: async () => ({{
        data: {{
          draft: false,
          base: {{ ref: 'main' }},
          head: {{ sha: 'a'.repeat(40) }},
        }},
      }}),
    }},
    actions: {{
      createWorkflowDispatch: async (payload) => record('dispatch', payload),
    }},
  }},
}};
globalThis.core = {{
  notice: (message) => record('notice', message),
  setFailed: (message) => record('failed', message),
}};
(async () => {{
{self.dispatch_script}
  fs.writeFileSync(process.env.RECORD_FILE, JSON.stringify(records), 'utf8');
}})().catch((error) => {{
  console.error(error && error.stack || error);
  process.exitCode = 1;
}});
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            record_file = Path(temp_dir) / "records.json"
            env = os.environ.copy()
            env["PR_COMMENT"] = comment
            env["RECORD_FILE"] = str(record_file)
            env["AVAILABLE_WORKFLOWS"] = available
            completed = subprocess.run(
                ["node"],
                input=wrapper,
                text=True,
                encoding="utf-8",
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                check=False,
            )
            records = json.loads(record_file.read_text(encoding="utf-8"))
        return completed, records

    def test_comment_defaults_to_pull_request_base_branch(self):
        completed, records = self._run_dispatch("/run-npu-ci quick")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(dispatches[0]["ref"], "main")
        self.assertEqual(dispatches[0]["inputs"]["pr_number"], "527")
        self.assertEqual(dispatches[0]["inputs"]["ci_mode"], "quick")
        self.assertEqual(dispatches[0]["inputs"]["requested_by"], "ci-admin")

    def test_comment_can_select_another_ci_definition_branch(self):
        completed, records = self._run_dispatch(
            "/run-npu-ci full ci_ref=release-v2 ops=chunk_fwd_o"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(dispatches[0]["ref"], "release-v2")
        self.assertEqual(dispatches[0]["inputs"]["ci_mode"], "full")
        self.assertEqual(dispatches[0]["inputs"]["ops"], "chunk_fwd_o")


    def test_comment_dispatches_split_platform_workflows_when_available(self):
        completed, records = self._run_dispatch(
            "/run-npu-ci quick",
            available_workflows=(
                ".github/workflows/ci.yml,"
                ".github/workflows/npu-ci-a2.yml,"
                ".github/workflows/npu-ci-a5.yml"
            ),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual(
            [item["workflow_id"] for item in dispatches],
            ["npu-ci-a2.yml", "npu-ci-a5.yml"],
        )
        for item in dispatches:
            self.assertEqual(item["ref"], "main")
            self.assertEqual(item["inputs"]["pr_number"], "527")
            self.assertEqual(item["inputs"]["ci_mode"], "quick")

    def test_comment_falls_back_to_legacy_combined_workflow(self):
        completed, records = self._run_dispatch(
            "/run-npu-ci quick",
            available_workflows=".github/workflows/ci.yml",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual([item["workflow_id"] for item in dispatches], ["ci.yml"])


if __name__ == "__main__":
    unittest.main()
