"""Run issue version commands and index generation against an in-memory GitHub API."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
NODE = (
    os.environ.get("ISSUE_WORKFLOW_NODE")
    or shutil.which("node")
    or shutil.which("nodejs")
    or shutil.which("node.exe")
)
OLD_BRANCH_RULE = (
    "- 版本标签：与版本分支同名，只表示“该 bug 的目标修复版本”；一个 bug 可以带多个版本标签，"
    "表示在旧版本未修复并顺延到后续版本。"
)
OLD_REQUIREMENT_RULE = (
    "- 需求排期：需求类 issue 不打版本标签、不按版本排期，统一在本 issue 的"
    "“未完成需求 / 已完成需求”里按算子跟踪进展。"
)


def issue(number, labels, **extra):
    return {
        "number": number, "title": f"Issue {number}", "body": "", "state": "open",
        "labels": labels, "assignees": [], "updated_at": "2026-11-23T01:00:00Z",
        "html_url": f"https://example.invalid/issues/{number}", **extra,
    }


@unittest.skipUnless(NODE, "Node.js is required to execute workflow scripts")
class IssueVersionWorkflowTests(unittest.TestCase):
    def run_workflow(self, filename, job, issues, **options):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text(encoding="utf-8"))
        scripts = [step["with"]["script"] for step in workflow["jobs"][job]["steps"]
                   if "script" in step.get("with", {})]
        self.assertEqual(len(scripts), 1)
        data = {
            "script": scripts[0], "env": workflow.get("env", {}), "issues": issues,
            "branches": ["main", "v26.9.1", "v26.10.0", "v26.11.0"],
            **options,
        }
        wrapper = "const input = " + json.dumps(data) + ";\n" + r"""
const calls = [];
const issues = new Map(input.issues.map(issue => [issue.number, issue]));
const names = issue => issue.labels.map(label => typeof label === 'string' ? label : label.name);
const record = (method, args) => calls.push({method, ...args});
const restIssues = {
  listForRepo: 'listForRepo', listEventsForTimeline: 'listEventsForTimeline',
  get: async args => ({data: issues.get(args.issue_number)}),
  getLabel: async args => ({data: {name: args.name}}),
  createLabel: async args => record('createLabel', args),
  addLabels: async args => {
    record('addLabels', args);
    if (input.failAdd) throw new Error('simulated add failure');
    if (input.dropAdd) return;
    const current = issues.get(args.issue_number);
    current.labels = [...new Set([...names(current), ...args.labels])];
  },
  removeLabel: async args => {
    record('removeLabel', args);
    const current = issues.get(args.issue_number);
    current.labels = names(current).filter(name => name !== args.name);
  },
  createComment: async args => record('createComment', args),
  update: async args => {
    record('update', args);
    Object.assign(issues.get(args.issue_number), args);
  },
  create: async args => {
    record('create', args);
    const created = {number: 999, state: 'open', node_id: 'index', ...args};
    issues.set(999, created);
    return {data: created};
  },
};
const github = {
  rest: {
    issues: restIssues,
    repos: {
      getBranch: async args => {
        if (!input.branches.includes(args.branch)) throw new Error('missing branch');
        return {data: {name: args.branch}};
      },
      get: async () => ({data: {default_branch: 'main'}}),
    },
    reactions: {createForIssueComment: async args => record('reaction', args)},
    actions: {createWorkflowDispatch: async args => record('dispatch', args)},
  },
  paginate: async (method, args) => {
    record('paginate', args);
    if (method === 'listEventsForTimeline') return [];
    if (method !== 'listForRepo') throw new Error('unexpected method');
    const required = (args.labels || '').split(',').filter(Boolean);
    // Model GitHub's AND semantics so a mistaken comma-separated OR query fails these tests.
    return [...issues.values()].filter(issue =>
      (args.state === 'all' || !args.state || issue.state === args.state) &&
      required.every(label => names(issue).includes(label)));
  },
  graphql: async () => ({repository: {pinnedIssues: {nodes: [{issue: {number: 999}}]}}}),
};
const summary = {addHeading() {return this;}, addRaw() {return this;}, async write() {}};
const core = {
  notice() {}, warning() {}, summary,
  setFailed: message => {throw new Error(message);},
};
const context = {
  repo: {owner: 'example', repo: 'operators'},
  eventName: input.eventName || 'issue_comment',
  payload: {
    issue: {number: 1}, ref: input.branch || 'v26.11.0', inputs: {branch: input.branch || 'v26.11.0'},
    comment: {body: input.command || '', id: 1, author_association: input.association || 'MEMBER'},
  },
};
Object.assign(process.env, input.env, {
  OP_NAMES_FILE: '__no_operator_names_in_unit_test__', INCLUDE_LINKED_PRS: 'false',
});
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  let error;
  try {
    await new AsyncFunction('github', 'context', 'core', 'require', input.script)(github, context, core, require);
  } catch (caught) {error = caught.message;}
  process.stdout.write(JSON.stringify({issues: [...issues.values()], calls, error}));
})();
"""
        completed = subprocess.run(
            [NODE], input=wrapper, text=True, encoding="utf-8", capture_output=True,
            cwd=ROOT, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        if not options.get("failAdd") and not options.get("dropAdd"):
            self.assertNotIn("error", result)
        return result

    def command(self, labels, command, **options):
        return self.run_workflow(
            "version-tags.yml", "issue-version-command", [issue(1, labels)],
            command=command, **options,
        )

    def index(self, entries, body=""):
        return self.run_workflow(
            "requirements-index.yml", "refresh-index",
            [*entries, issue(999, ["requirements-index"], body=body)],
        )

    def test_bug_and_enhancement_can_select_main(self):
        for kind in ("bug", "enhancement"):
            with self.subTest(kind=kind):
                result = self.command([kind, "v26.9.1"], "/version main")
                self.assertEqual(set(result["issues"][0]["labels"]), {kind, "v26.9.1", "main"})

    def test_only_adds_new_branch_before_removing_old_branch(self):
        result = self.command(["enhancement", "v26.9.1"], "/version only main")
        self.assertEqual(result["issues"][0]["labels"], ["enhancement", "main"])
        operations = [call["method"] for call in result["calls"]]
        self.assertLess(operations.index("addLabels"), operations.index("removeLabel"))
        dispatches = {call["workflow_id"]: call for call in result["calls"] if call["method"] == "dispatch"}
        self.assertEqual(dispatches["issue-labels.yml"]["inputs"], {"issue_number": "1"})
        self.assertEqual(dispatches["issue-labels.yml"]["ref"], "main")
        self.assertIn("requirements-index.yml", dispatches)

    def test_failed_add_keeps_last_branch(self):
        result = self.command(["bug", "main"], "/version only v26.9.1", failAdd=True)
        self.assertIn("simulated add failure", result["error"])
        self.assertEqual(result["issues"][0]["labels"], ["bug", "main"])
        self.assertFalse(any(call["method"] == "removeLabel" for call in result["calls"]))

    def test_silent_label_failure_is_not_reported_as_success(self):
        result = self.command(["enhancement", "main"], "/version v26.11.0", dropAdd=True)
        self.assertIn("标签读回与请求不一致", result["error"])
        self.assertFalse(any(call["method"] == "createComment" for call in result["calls"]))
        self.assertFalse(any(call["method"] == "dispatch" for call in result["calls"]))

    def test_conflicting_primary_categories_reject_version_command(self):
        for labels in (["bug", "question", "main"], ["bug", "enhancement", "main"]):
            result = self.command(labels, "/version v26.11.0")
            self.assertFalse(any(call["method"] == "addLabels" for call in result["calls"]))

    def test_form_and_command_share_the_same_issue_queue(self):
        forms = yaml.safe_load((ROOT / ".github/workflows/issue-labels.yml").read_text(encoding="utf-8"))
        commands = yaml.safe_load((ROOT / ".github/workflows/version-tags.yml").read_text(encoding="utf-8"))
        prefix = "issue-labels-${{ github.event.issue.number ||"
        for workflow in (forms, commands):
            self.assertTrue(workflow["concurrency"]["group"].startswith(prefix))
            self.assertEqual(workflow["concurrency"]["queue"], "max")

    def test_commands_cannot_clear_all_branches(self):
        for command in ("/version none", "/version clear", "/version -"):
            with self.subTest(command=command):
                result = self.command(["enhancement", "main"], command)
                self.assertEqual(result["issues"][0]["labels"], ["enhancement", "main"])
                comments = [call["body"] for call in result["calls"] if call["method"] == "createComment"]
                self.assertIn("必须保留至少一个目标分支", comments[0])

    def test_nonexistent_branch_and_special_category_are_rejected(self):
        for labels, command in ((["bug", "main"], "/version v26.12.0"),
                                (["question"], "/version main")):
            with self.subTest(labels=labels, command=command):
                result = self.command(labels, command)
                self.assertEqual(result["issues"][0]["labels"], labels)
                self.assertFalse(any(call["method"] == "addLabels" for call in result["calls"]))

    def test_new_version_still_carries_only_bugs(self):
        result = self.run_workflow(
            "version-tags.yml", "sync-version-tag",
            [issue(1, ["bug", "main"]), issue(2, ["enhancement", "main"])],
            eventName="create",
        )
        self.assertIn("v26.11.0", result["issues"][0]["labels"])
        self.assertEqual(result["issues"][1]["labels"], ["enhancement", "main"])

    def test_main_creation_does_not_trigger_bulk_rollover(self):
        result = self.run_workflow(
            "version-tags.yml", "sync-version-tag", [issue(1, ["bug", "v26.9.1"])],
            eventName="create", branch="main",
        )
        self.assertFalse(any(call["method"] == "addLabels" for call in result["calls"]))

    def test_summary_queries_categories_with_or_and_deduplicates(self):
        result = self.index([
            issue(1, ["bug", "v26.9.1"]), issue(2, ["enhancement", "main"]),
            issue(3, ["bug", "enhancement", "main"]),
        ])
        body = next(item["body"] for item in result["issues"] if item["number"] == 999)
        self.assertIn("bug/enhancement 总数：3", body)
        self.assertIn("需求总数：2", body)
        self.assertIn("[#1]", body)
        self.assertIn("[#2]", body)

    def test_main_sorts_first_and_multibranch_issue_appears_in_each_group(self):
        result = self.index([issue(1, ["enhancement", "v26.9.1", "main", "v26.10.0"])])
        body = next(item["body"] for item in result["issues"] if item["number"] == 999)
        section = body.split("### 各目标分支的缺陷与需求进展", 1)[1]
        self.assertLess(section.index("#### main"), section.index("#### v26.10.0"))
        self.assertLess(section.index("#### v26.10.0"), section.index("#### v26.9.1"))
        self.assertEqual(section.count("[#1]"), 3)
        self.assertIn("其他目标分支", section)
        self.assertNotIn("needs-version", result["issues"][0]["labels"])

    def test_missing_targets_are_reported_without_bulk_labeling_old_requirements(self):
        result = self.index([issue(1, ["bug"]), issue(2, ["enhancement"])])
        self.assertIn("needs-version", result["issues"][0]["labels"])
        self.assertEqual(result["issues"][1]["labels"], ["enhancement"])
        body = next(item["body"] for item in result["issues"] if item["number"] == 999)
        self.assertIn("待补目标分支 bug/enhancement：2 个", body)

    def test_migrates_only_the_two_exact_old_manual_lines(self):
        custom = "维护者自己的计划必须保留。\n引用：" + OLD_REQUIREMENT_RULE
        manual = f"{custom}\n{OLD_BRANCH_RULE}\n{OLD_REQUIREMENT_RULE}\n"
        previous = manual + "<!-- requirements-index:auto -->\n旧自动内容\n<!-- requirements-index:auto:end -->\n保留尾注"
        result = self.index([], previous)
        body = next(item["body"] for item in result["issues"] if item["number"] == 999)
        self.assertIn(custom, body)
        self.assertNotIn("\n" + OLD_BRANCH_RULE + "\n", body)
        self.assertNotIn("\n" + OLD_REQUIREMENT_RULE + "\n", body)
        self.assertIn("- 目标分支标签：bug 和 enhancement", body)
        self.assertIn("- 需求排期：需求按目标分支跟踪", body)
        self.assertNotIn("旧自动内容", body)
        self.assertTrue(body.endswith("\n保留尾注"))


if __name__ == "__main__":
    unittest.main()
