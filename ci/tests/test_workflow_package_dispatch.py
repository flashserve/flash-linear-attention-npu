#!/usr/bin/env python3
"""Pin the branch selection contract for PR comment /package builds."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMENT_DISPATCH_PATH = (
    REPO_ROOT / ".github" / "workflows" / "npu-ci-comment-dispatch.yml"
)
PACKAGE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "package.yml"


def _workflow_on(workflow):
    # PyYAML 1.1 treats the YAML 1.2 key ``on`` as boolean ``True``.
    return workflow.get("on", workflow.get(True))


class WorkflowPackageDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dispatch_workflow = yaml.safe_load(
            COMMENT_DISPATCH_PATH.read_text(encoding="utf-8")
        )
        cls.package_workflow = yaml.safe_load(
            PACKAGE_WORKFLOW_PATH.read_text(encoding="utf-8")
        )
        cls.dispatch_script = next(
            step["with"]["script"]
            for step in cls.dispatch_workflow["jobs"]["dispatch-package"]["steps"]
            if step.get("uses") == "actions/github-script@v8"
        )

    def test_package_workflow_declares_build_inputs(self):
        inputs = _workflow_on(self.package_workflow)["workflow_dispatch"]["inputs"]

        for name in ("build_ref", "build_sha", "build_repo"):
            self.assertIn(name, inputs)
        for legacy in ("head_sha", "head_ref", "head_repo_full"):
            self.assertNotIn(legacy, inputs)
        self.assertTrue(inputs["build_sha"]["required"])
        self.assertTrue(inputs["build_repo"]["required"])

    def _run_dispatch(self, comment, permission="admin", draft=False,
                      merge_conflict=False):
        wrapper = f"""
const fs = require('fs');
const records = [];
const record = (kind, payload) => records.push({{ kind, payload }});
globalThis.context = {{
  actor: 'ci-admin',
  ref: 'refs/heads/main',
  payload: {{
    issue: {{ number: 527 }},
    comment: {{ body: process.env.PR_COMMENT }},
  }},
  repo: {{ owner: 'example', repo: 'repo' }},
}};
globalThis.github = {{
  rest: {{
    repos: {{
      getCollaboratorPermissionLevel: async () => ({{
        data: {{ permission: process.env.PR_PERMISSION || 'admin' }},
      }}),
      getBranch: async (payload) => {{
        if (payload.branch === 'no-such-branch') {{
          throw new Error('Not Found');
        }}
        return {{ data: {{ name: payload.branch, commit: {{ sha: 'b'.repeat(40) }} }} }};
      }},
    }},
    pulls: {{
      get: async () => ({{
        data: {{
          draft: process.env.PR_DRAFT === '1',
          base: {{ ref: 'main' }},
          head: {{
            sha: 'a'.repeat(40),
            ref: 'feat/pkg',
            repo: {{ full_name: 'someone/fork-repo' }},
          }},
          // PR 合并预览 commit (base 分支 + PR 改动); 有冲突时为 null
          merge_commit_sha:
            process.env.PR_MERGE_CONFLICT === '1' ? null : 'c'.repeat(40),
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
  warning: (message) => record('warning', message),
  setFailed: (message) => record('failed', message),
}};
(async () => {{
  try {{
{self.dispatch_script}
  }} finally {{
    fs.writeFileSync(process.env.RECORD_FILE, JSON.stringify(records), 'utf8');
  }}
}})().catch((error) => {{
  console.error(error && error.stack || error);
  process.exitCode = 1;
}});
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            record_file = Path(temp_dir) / "records.json"
            env = os.environ.copy()
            env["PR_COMMENT"] = comment
            env["PR_PERMISSION"] = permission
            env["PR_DRAFT"] = "1" if draft else "0"
            env["PR_MERGE_CONFLICT"] = "1" if merge_conflict else "0"
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

    def _assert_single_dispatch(self, records):
        dispatches = [item["payload"] for item in records if item["kind"] == "dispatch"]
        self.assertEqual(len(dispatches), 1)
        self.assertRegex(dispatches[0]["inputs"]["datetime"], r"^\d{8}-\d{6}$")
        return dispatches[0]

    def test_comment_defaults_to_pull_request_merge_commit(self):
        # 默认编译 PR 合并结果 (base 分支 + PR 改动)，不是纯 base 分支
        completed, records = self._run_dispatch("/package")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatch = self._assert_single_dispatch(records)
        self.assertEqual(dispatch["workflow_id"], "package.yml")
        self.assertEqual(dispatch["ref"], "refs/heads/main")
        self.assertEqual(dispatch["inputs"]["pr_number"], "527")
        self.assertEqual(dispatch["inputs"]["build_ref"], "main")
        self.assertEqual(dispatch["inputs"]["build_sha"], "c" * 40)
        self.assertEqual(dispatch["inputs"]["build_repo"], "example/repo")
        self.assertEqual(dispatch["inputs"]["socs"],
                         "ascend910b,ascend910_93,ascend950")
        self.assertEqual(dispatch["inputs"]["arches"], "arm,x86")
        self.assertEqual(dispatch["inputs"]["requested_by"], "ci-admin")

    def test_build_and_report_require_guard(self):
        # guard 是 write 权限绕过评论分发器直接 dispatch 时的唯一门禁，
        # build / report 必须显式依赖它
        jobs = self.package_workflow["jobs"]
        self.assertIn("guard", jobs)
        self.assertIn("guard", jobs["build"]["needs"])
        self.assertIn("guard", jobs["report"]["needs"])

    def _guard_script(self):
        return next(
            step["with"]["script"]
            for step in self.package_workflow["jobs"]["guard"]["steps"]
            if step.get("uses") == "actions/github-script@v8"
        )

    def _run_guard(self, actor="github-actions[bot]", requested_by="ci-admin",
                   permission="admin", build_repo="example/repo",
                   build_sha=None, head_repo="someone/fork-repo",
                   commit_missing=False):
        if build_sha is None:
            build_sha = "c" * 40
        guard_script = self._guard_script()
        wrapper = f"""
const fs = require('fs');
const records = [];
const record = (kind, payload) => records.push({{ kind, payload }});
globalThis.context = {{
  actor: process.env.GUARD_ACTOR,
  payload: {{
    inputs: {{
      pr_number: '527',
      socs: '',
      arches: '',
      requested_by: process.env.GUARD_REQUESTED_BY,
      build_sha: process.env.GUARD_BUILD_SHA,
      build_ref: 'main',
      build_repo: process.env.GUARD_BUILD_REPO,
      datetime: '20260929-120000',
    }},
  }},
  repo: {{ owner: 'example', repo: 'repo' }},
}};
globalThis.github = {{
  rest: {{
    repos: {{
      getCollaboratorPermissionLevel: async () => ({{
        data: {{ permission: process.env.GUARD_PERMISSION }},
      }}),
      getCommit: async () => {{
        if (process.env.GUARD_COMMIT_MISSING === '1') {{
          throw new Error('Not Found');
        }}
        return {{ data: {{}} }};
      }},
    }},
    pulls: {{
      get: async () => ({{
        data: {{ head: {{ repo: {{ full_name: process.env.GUARD_HEAD_REPO }} }} }},
      }}),
    }},
  }},
}};
globalThis.core = {{
  info: (message) => record('info', message),
  warning: (message) => record('warning', message),
  setFailed: (message) => record('failed', message),
}};
(async () => {{
  try {{
{guard_script}
  }} finally {{
    fs.writeFileSync(process.env.RECORD_FILE, JSON.stringify(records), 'utf8');
  }}
}})().catch((error) => {{
  console.error(error && error.stack || error);
  process.exitCode = 1;
}});
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            record_file = Path(temp_dir) / "records.json"
            env = os.environ.copy()
            env["GUARD_ACTOR"] = actor
            env["GUARD_REQUESTED_BY"] = requested_by
            env["GUARD_PERMISSION"] = permission
            env["GUARD_BUILD_REPO"] = build_repo
            env["GUARD_BUILD_SHA"] = build_sha
            env["GUARD_HEAD_REPO"] = head_repo
            env["GUARD_COMMIT_MISSING"] = "1" if commit_missing else "0"
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
            records = json.loads(record_file.read_text(encoding="utf8"))
        return completed, records

    def test_guard_accepts_dispatcher_and_direct_admin_paths(self):
        # 评论分发路径 (bot 触发 + requested_by=admin) 与手工 admin 直接触
        # 发均放行
        for kwargs in (
            dict(actor="github-actions[bot]", requested_by="ci-admin"),
            dict(actor="boss-admin", requested_by=""),
        ):
            with self.subTest(**kwargs):
                completed, records = self._run_guard(**kwargs)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertFalse(
                    [item for item in records if item["kind"] == "failed"],
                    records)
                self.assertTrue(
                    any("门禁通过" in item["payload"]
                        for item in records if item["kind"] == "info"))

    def test_guard_rejects_unauthorized_requesters(self):
        # bot 触发但缺 requested_by / requested_by 非 admin / write 权限
        # 账号伪造 requested_by 直接 dispatch，均必须拦截
        cases = [
            (dict(actor="github-actions[bot]", requested_by="",
                  permission="admin"), "无法确定请求者"),
            (dict(actor="github-actions[bot]", requested_by="mallory",
                  permission="write"), "无权触发 Package 编译"),
            # 真人直接触发时 requested_by 不可信: 即便伪造 admin 用户名，
            # 也必须校验触发者本人 (失败信息点名 mallory 而非 ci-admin)
            (dict(actor="mallory", requested_by="ci-admin",
                  permission="write"), "mallory 无权触发"),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                completed, records = self._run_guard(**kwargs)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                failures = [
                    item["payload"] for item in records
                    if item["kind"] == "failed"
                ]
                self.assertEqual(len(failures), 1)
                self.assertIn(expected, failures[0])

    def test_guard_restricts_build_repo_and_sha(self):
        # build_repo 仅限本仓库 / PR head 仓库；build_sha 必须是
        # 40 位十六进制且存在于对应仓库
        cases = [
            (dict(build_repo="evil/evil"), "不在允许范围"),
            (dict(build_sha="shortsha"), "40 位十六进制"),
            (dict(commit_missing=True), "不存在于"),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                completed, records = self._run_guard(**kwargs)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                failures = [
                    item["payload"] for item in records
                    if item["kind"] == "failed"
                ]
                self.assertEqual(len(failures), 1)
                self.assertIn(expected, failures[0])

        # PR head 仓库 (fork PR 的 branch=head 编译) 属于白名单例外
        completed, records = self._run_guard(
            build_repo="someone/fork-repo", build_sha="a" * 40)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(
            [item for item in records if item["kind"] == "failed"], records)

    def test_unmergeable_pull_request_is_rejected(self):
        # PR 有冲突时 GitHub 不提供 merge commit，默认出包必须明确失败
        completed, records = self._run_dispatch(
            "/package", merge_conflict=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("无法自动合并", failures[0])
        self.assertFalse([item for item in records if item["kind"] == "dispatch"])

    def test_comment_can_select_another_build_branch(self):
        completed, records = self._run_dispatch(
            "/package branch=release-v2 soc=ascend950 arch=x86")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatch = self._assert_single_dispatch(records)
        self.assertEqual(dispatch["inputs"]["build_ref"], "release-v2")
        self.assertEqual(dispatch["inputs"]["build_sha"], "b" * 40)
        self.assertEqual(dispatch["inputs"]["build_repo"], "example/repo")
        self.assertEqual(dispatch["inputs"]["socs"], "ascend950")
        self.assertEqual(dispatch["inputs"]["arches"], "x86")

    def test_comment_can_build_pull_request_head(self):
        completed, records = self._run_dispatch("/package branch=head")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        dispatch = self._assert_single_dispatch(records)
        self.assertEqual(dispatch["inputs"]["build_ref"], "feat/pkg")
        self.assertEqual(dispatch["inputs"]["build_sha"], "a" * 40)
        self.assertEqual(dispatch["inputs"]["build_repo"], "someone/fork-repo")

    def test_missing_branch_is_rejected(self):
        completed, records = self._run_dispatch("/package branch=no-such-branch")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("找不到出包分支", failures[0])
        self.assertFalse([item for item in records if item["kind"] == "dispatch"])

    def test_branch_param_validation(self):
        for comment, expected in [
            ("/package branch=a branch=b", "branch 参数只能指定一次"),
            ("/package branch=", "branch 参数不能为空"),
        ]:
            with self.subTest(comment=comment):
                completed, records = self._run_dispatch(comment)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                failures = [
                    item["payload"] for item in records
                    if item["kind"] == "failed"
                ]
                self.assertEqual(len(failures), 1)
                self.assertIn(expected, failures[0])
                self.assertFalse(
                    [item for item in records if item["kind"] == "dispatch"])

    def test_unknown_soc_param_is_rejected(self):
        completed, records = self._run_dispatch("/package soc=ascend9999")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("soc 参数不合法", failures[0])

    def test_non_admin_requester_is_rejected(self):
        completed, records = self._run_dispatch("/package", permission="write")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("无权触发 Package 编译", failures[0])
        self.assertFalse([item for item in records if item["kind"] == "dispatch"])

    def test_draft_pull_request_is_rejected(self):
        completed, records = self._run_dispatch("/package", draft=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        failures = [item["payload"] for item in records if item["kind"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("仍是草稿", failures[0])
        self.assertFalse([item for item in records if item["kind"] == "dispatch"])


if __name__ == "__main__":
    unittest.main()
