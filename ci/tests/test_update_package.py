"""Exercise manual cache refresh authorization and its container-only boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class PackageCommandTest(unittest.TestCase):
    def dispatch(self, command="/update-package", permission="admin", rerun_permission="admin", attempt="1"):
        harness = r'''
const calls = [];
const github = {rest: {
  repos: {
    getCollaboratorPermissionLevel: async ({username}) => ({data: {permission:
      username === 'rerunner' ? process.env.RERUN_PERMISSION : process.env.PERMISSION}}),
    getBranch: async x => {calls.push(['branch', x]); return {data: {}};},
  },
  pulls: {get: async () => ({data: {number: 7, state: 'open', base: {ref: 'v0.6.0'}}})},
  actions: {createWorkflowDispatch: async x => calls.push(['dispatch', x])},
  issues: {createComment: async x => calls.push(['comment', x])},
}};
const context = {repo: {owner: 'org', repo: 'repo'}, actor: 'requester',
  serverUrl: 'https://github.com', payload: {issue: {number: 7}, comment: {body: process.env.COMMAND}}};
const core = {notice: x => calls.push(['notice', x])};
(async () => {
  try {await require(process.env.SCRIPT)({github,context,core});}
  catch(e) {calls.push(['error',e.message]);}
  console.log(JSON.stringify(calls));
})();
'''
        env = dict(os.environ, SCRIPT=str(ROOT / ".github/scripts/dispatch_update_package.js"),
                   COMMAND=command, PERMISSION=permission, RERUN_PERMISSION=rerun_permission,
                   NPU_CI_RUN_ATTEMPT=attempt, NPU_CI_TRIGGERING_ACTOR="rerunner")
        result = subprocess.run(["node", "-e", harness], env=env, capture_output=True,
                                text=True, encoding="utf-8", check=True)
        return json.loads(result.stdout)

    def test_default_updates_pr_target_with_no_ci_or_compilation_dispatch(self):
        calls = self.dispatch()
        dispatch = next(value for kind, value in calls if kind == "dispatch")
        self.assertEqual(dispatch["workflow_id"], "update-package.yml")
        self.assertEqual(dispatch["ref"], "v0.6.0")
        self.assertEqual(dispatch["inputs"], {"base_ref": "v0.6.0", "requested_by": "requester"})
        self.assertEqual(sum(kind == "comment" for kind, _ in calls), 1)

    def test_ci_definition_ref_does_not_change_daily_branch(self):
        calls = self.dispatch("/update-package ci_ref=cache-dev")
        dispatch = next(value for kind, value in calls if kind == "dispatch")
        self.assertEqual(dispatch["ref"], "cache-dev")
        self.assertEqual(dispatch["inputs"]["base_ref"], "v0.6.0")

    def test_unauthorized_actor_and_rerunner_cannot_refresh(self):
        for settings in ({"permission": "write"}, {"attempt": "2", "rerun_permission": "write"}):
            with self.subTest(settings=settings):
                calls = self.dispatch(**settings)
                self.assertTrue(any(kind == "error" for kind, _ in calls))
                self.assertFalse(any(kind == "dispatch" for kind, _ in calls))

    def test_embedded_or_unsupported_commands_do_not_dispatch(self):
        for command in ("please /update-package", "/update-package full", "/update-package ci_ref=", "/update-package ci_ref=a ci_ref=b"):
            with self.subTest(command=command):
                self.assertFalse(any(kind == "dispatch" for kind, _ in self.dispatch(command)))

    def test_update_workflow_authorizes_manual_and_worker_reruns(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/update-package.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["update"]
        self.assertEqual(job["needs"], "authorize")
        rerun = next(step for step in job["steps"] if step.get("name") == "校验刷新任务重跑权限")
        self.assertEqual(rerun["if"], "github.run_attempt > 1")
        self.assertIn("requireAuthorizedCiUser", rerun["with"]["script"])
        self.assertEqual(job["env"]["CI_PACKAGE_CACHE"], "/home/huangjunzhe/ci/pkg-cache")


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell test")
class PackageUpdateContainerTest(unittest.TestCase):
    def test_refresh_needs_no_npu_and_preserves_shared_cache_on_success_or_failure(self):
        for status in (0, 19):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                commands = root / "bin"
                commands.mkdir()
                docker = commands / "docker"
                docker.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
a=sys.argv[1:]
with open(os.environ['RECORD'],'a') as f:f.write(json.dumps(a)+'\\n')
if '--cidfile' in a:
    pathlib.Path(a[a.index('--cidfile')+1]).write_text('update-container')
    work=next(a[i+1].split(':')[0] for i,x in enumerate(a[:-1]) if x=='-v' and a[i+1].endswith(':/ci-update'))
    pathlib.Path(work,'package.json').write_text('{"kind":"update"}')
    sys.exit(int(os.environ['STATUS']))
''')
                docker.chmod(0o755)
                cache = root / "pkg-cache"
                cache.mkdir()
                (cache / "retained").write_text("shared cache")
                scratch = root / "tmp"
                env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                           CI_PACKAGE_CACHE=str(cache), TMPDIR=str(scratch), RECORD=str(root / "calls"), STATUS=str(status))
                result = subprocess.run(["bash", str(ROOT / "ci/update_package_cache.sh")], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, status, result.stderr)
                calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
                run = next(call for call in calls if "--cidfile" in call)
                self.assertEqual(run[-3:], ["python3", "ci/prepare_ci_package.py", "--update-package"])
                self.assertIn(str(cache) + ":/ci-pkg-cache", run)
                self.assertNotIn("--privileged", run)
                self.assertNotIn("--device", run)
                self.assertIn(["rm", "-f", "update-container"], calls)
                self.assertEqual((cache / "retained").read_text(), "shared cache")
                self.assertFalse(list(scratch.iterdir()))
