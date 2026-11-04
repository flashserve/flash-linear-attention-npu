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
        self.assertEqual(set(triggers["workflow_dispatch"]["inputs"]), {
            "pr_number", "ci_mode", "ops", "requested_by", "expected_head_sha",
            "expected_base_ref", "expected_base_sha", "expected_workflow_sha",
        })
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
        self.assertIn("auth.selectDispatchInputs(workflowText, {", self.dispatch_script)

    def _run_dispatch(self, comment):
        encoded_workflow = base64.b64encode(
            CI_WORKFLOW_PATH.read_bytes()
        ).decode("ascii")
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
  paginate: async () => [],
  rest: {{
    repos: {{
      getCollaboratorPermissionLevel: async () => ({{ data: {{ permission: 'admin' }} }}),
      getBranch: async (payload) => ({{ data: {{ name: payload.branch, protected: true, commit: {{sha: 'c'.repeat(40)}} }} }}),
      get: async () => ({{ data: {{ default_branch: 'main' }} }}),
      getContent: async (payload) => payload.path === '.github/npu-ci-users.json' ? ({{ data: {{ type: 'file', content: Buffer.from(JSON.stringify({{users:['ci-admin']}})).toString('base64') }} }}) : ({{
        data: {{
          type: 'file',
          encoding: 'base64',
          content: '{encoded_workflow}',
        }},
      }}),
    }},
    pulls: {{
      get: async () => ({{
        data: {{
          number: 527, changed_files: 0, draft: false, state: 'open',
          base: {{ sha: 'c'.repeat(40), ref: 'main' }},
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
  await (async () => {{
{self.dispatch_script}
  }})();
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
            env["NPU_CI_RUN_ATTEMPT"] = '1'
            env["NPU_CI_AUTH_SCRIPT"] = str(REPO_ROOT / '.github/scripts/npu_ci_authorization.js')
            env["NPU_CI_CHECKS_SCRIPT"] = str(REPO_ROOT / '.github/scripts/npu_ci_checks.js')
            env["RECORD_FILE"] = str(record_file)
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

    def test_comment_cannot_select_another_ci_definition_branch(self):
        completed, records = self._run_dispatch(
            "/run-npu-ci full ci_ref=release-v2 ops=chunk_fwd_o"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual(dispatches, [])
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertIn("不支持", failures[0])


if __name__ == "__main__":
    unittest.main()
