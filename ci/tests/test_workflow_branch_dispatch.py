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
WORKFLOW_SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "npu_ci_workflow.js"


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
        cls.workflow_script = WORKFLOW_SCRIPT_PATH.read_text(encoding="utf-8")

    def test_actual_ci_is_manual_dispatch_only_and_has_internal_requester_input(self):
        workflow = yaml.safe_load(self.ci_text)
        triggers = _workflow_on(workflow)

        self.assertEqual(list(triggers), ["workflow_dispatch"])
        self.assertNotIn("issue_comment:", self.ci_text)
        self.assertEqual(set(triggers["workflow_dispatch"]["inputs"]), {
            "pr_number", "ci_mode", "ops", "requested_by", "expected_head_sha",
            "expected_base_ref", "expected_base_sha", "expected_workflow_sha",
            "workflow_source",
        })
        self.assertIn("${{ github.ref_name }}", self.ci_text)
        self.assertIn("${{ github.workflow_sha }}", self.ci_text)

    def test_comment_dispatcher_has_write_permission_and_checks_target_workflow(self):
        triggers = _workflow_on(self.dispatch_workflow)
        permissions = self.dispatch_workflow["permissions"]

        self.assertEqual(triggers["issue_comment"]["types"], ["created"])
        self.assertEqual(permissions["actions"], "write")
        self.assertIn("workflow.dispatchNpuWorkflow({", self.dispatch_script)
        self.assertIn("workflow_id: 'ci.yml'", self.workflow_script)
        self.assertIn(".github/workflows/ci.yml", self.workflow_script)
        self.assertEqual(self.dispatch_workflow["jobs"]["dispatch"]["environment"], "npu-ci-dispatch")
        checkout = self.dispatch_workflow["jobs"]["dispatch"]["steps"][0]
        self.assertEqual(checkout["with"]["ref"], "${{ github.workflow_sha }}")
        self.assertIn(".github/scripts/npu_ci_workflow.js", checkout["with"]["sparse-checkout"])

    def _run_dispatch(self, comment, *, files=None, workflow_text=None, base_ref="main"):
        encoded_workflow = base64.b64encode(
            (workflow_text.encode("utf-8") if workflow_text is not None else CI_WORKFLOW_PATH.read_bytes())
        ).decode("ascii")
        wrapper = f"""
const fs = require('fs');
const records = [];
const record = (kind, payload) => records.push({{ kind, payload }});
const files = JSON.parse(process.env.DISPATCH_FILES);
globalThis.context = {{
  actor: 'ci-admin',
  payload: {{
    issue: {{ number: 527 }},
    comment: {{ body: process.env.PR_COMMENT }},
  }},
  repo: {{ owner: 'example', repo: 'repo' }},
}};
globalThis.github = {{
  paginate: async () => files,
  rest: {{
    repos: {{
      getCollaboratorPermissionLevel: async () => ({{ data: {{ permission: 'admin' }} }}),
      getBranch: async (payload) => ({{ data: {{ name: payload.branch, protected: true, commit: {{sha: 'c'.repeat(40)}} }} }}),
      getCommit: async (payload) => {{
        record('commit', payload);
        return {{data: {{sha: 'd'.repeat(40), parents: [{{sha: 'c'.repeat(40)}}, {{sha: 'a'.repeat(40)}}]}}}};
      }},
      get: async () => ({{ data: {{ default_branch: 'main' }} }}),
      getContent: async (payload) => {{
        record('content', payload);
        return payload.path === '.github/npu-ci-users.json' ? ({{ data: {{ type: 'file', content: Buffer.from(JSON.stringify({{users:['ci-admin']}})).toString('base64') }} }}) : ({{
        data: {{
          type: 'file',
          encoding: 'base64',
          content: '{encoded_workflow}',
        }},
        }});
      }},
    }},
    pulls: {{
      listFiles: async () => {{}},
      get: async () => ({{
        data: {{
          number: 527, changed_files: files.length, draft: false, state: 'open',
          user: {{login: 'ci-author'}}, mergeable: true, merge_commit_sha: 'd'.repeat(40),
          base: {{ sha: 'c'.repeat(40), ref: process.env.DISPATCH_BASE_REF }},
          head: {{ sha: 'a'.repeat(40), repo: {{full_name: 'example/fork'}} }},
        }},
      }}),
    }},
    actions: {{
      createWorkflowDispatch: async (payload) => record('dispatch', payload),
    }},
    git: {{createRef: async (payload) => record('tag', payload)}},
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
            env["NPU_CI_WORKFLOW_SCRIPT"] = str(WORKFLOW_SCRIPT_PATH)
            env.pop("NPU_CI_DISPATCH_TOKEN", None)
            env["DISPATCH_FILES"] = json.dumps(files or [{"filename": "fla/example.cpp", "status": "modified"}])
            env["DISPATCH_BASE_REF"] = base_ref
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
        self.assertEqual(dispatches[0]["inputs"]["workflow_source"], "base")
        self.assertFalse(any(item["kind"] == "tag" for item in records))

    def test_admin_ci_changes_dispatch_merge_snapshot_workflow(self):
        for filename in (".github/workflows/ci.yml", ".github/scripts/helper.js", "ci/README.md"):
            with self.subTest(filename=filename):
                completed, records = self._run_dispatch(
                    "/run-npu-ci quick", files=[{"filename": filename, "status": "modified"}])
                self.assertEqual(completed.returncode, 0, completed.stderr)
                dispatch = next(item["payload"] for item in records if item["kind"] == "dispatch")
                tag = f"npu-ci/pr-527/{'d' * 40}"
                self.assertEqual(dispatch["ref"], tag)
                self.assertEqual(dispatch["inputs"]["workflow_source"], "pr-merge")
                self.assertEqual(dispatch["inputs"]["expected_workflow_sha"], "d" * 40)
                self.assertEqual(dispatch["inputs"]["expected_base_sha"], "c" * 40)
                self.assertEqual([item["payload"] for item in records if item["kind"] == "tag"], [{
                    "owner": "example", "repo": "repo", "ref": f"refs/tags/{tag}", "sha": "d" * 40}])
                workflow_reads = [item["payload"] for item in records if item["kind"] == "content"
                                  and item["payload"]["path"] == ".github/workflows/ci.yml"]
                self.assertEqual([request["ref"] for request in workflow_reads], ["d" * 40])

    def test_legacy_maintenance_workflow_keeps_declared_input_interface(self):
        legacy = "on:\n  workflow_dispatch:\n    inputs:\n" + "".join(
            f"      {name}:\n        required: false\n"
            for name in ("pr_number", "ci_mode", "ops", "requested_by"))
        completed, records = self._run_dispatch(
            "/run-npu-ci full", files=[{"filename": ".github/workflows/ci.yml", "status": "modified"}],
            workflow_text=legacy, base_ref="v26.10.0")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatch = next(item["payload"] for item in records if item["kind"] == "dispatch")
        self.assertEqual(dispatch["ref"], f"npu-ci/pr-527/{'d' * 40}")
        self.assertEqual(dispatch["inputs"], {
            "pr_number": "527", "ci_mode": "full", "ops": "", "requested_by": "ci-admin"})

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
