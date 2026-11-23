"""Validate issue classification and synchronize selections from issue forms."""

import json
import os
import re
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TYPES = {"bug", "enhancement"}
PRIMARY_TYPES = TYPES | {"question", "committer-promotion"}
BRANCH = re.compile(r"^(?:main|v[0-9]{2,4}\.[0-9]{1,2}\.[0-9]{1,2})$")
PENDING = "needs-issue-info"


def field(body, heading):
    """Read an exact form heading outside fenced code; reject duplicates."""
    sections = []
    active = None
    fence = None
    for line in (body or "").splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            if active is not None:
                active.append(line)
            continue
        if fence is not None:
            if active is not None:
                active.append(line)
            continue
        if re.match(r"^#{1,6}\s", line):
            active = None
            if line.strip() == f"### {heading}":
                active = []
                sections.append(active)
        elif active is not None:
            active.append(line)
    if not sections:
        return None
    if len(sections) != 1:
        return "<duplicate>"
    return "\n".join(sections[0]).strip()


def plan_issue(issue, branches, event):
    labels = {item if isinstance(item, str) else item["name"] for item in issue.get("labels", [])}
    body = issue.get("body") or ""
    selected_type = field(body, "问题类型")
    result = {"skip": False, "add": [], "remove": [], "errors": []}
    if issue.get("pull_request") is not None or "requirements-index" in labels:
        result["skip"] = True
        return result
    available = {name for name in branches if BRANCH.fullmatch(name)}
    desired = set(labels)
    errors = result["errors"]

    def selected_changed(heading):
        snapshot = event.get("issue")
        if snapshot is not None:
            # A queued event must not undo a newer form edit or label decision.
            if field(snapshot.get("body"), heading) != field(body, heading):
                return False
            previous_labels = {
                item if isinstance(item, str) else item["name"]
                for item in snapshot.get("labels", [])
            }
            managed = PRIMARY_TYPES if heading == "问题类型" else {
                name for name in labels | previous_labels if BRANCH.fullmatch(name)
            }
            if labels & managed != previous_labels & managed:
                return False
        if event.get("action") == "opened":
            return True
        change = event.get("changes", {}).get("body", {})
        return event.get("action") == "edited" and "from" in change and (
            field(change["from"], heading) != field(body, heading)
        )

    if selected_type is not None:
        if selected_type not in TYPES:
            errors.append("问题类型必须选择 bug 或 enhancement。")
        elif selected_changed("问题类型") or not labels & PRIMARY_TYPES:
            desired -= PRIMARY_TYPES
            desired.add(selected_type)
    if len(desired & PRIMARY_TYPES) != 1:
        errors.append("必须且只能有一个主分类标签：bug、enhancement、question 或 committer-promotion。")

    if desired & TYPES:
        selected_branches = field(body, "目标分支")
        current_branches = {name for name in labels if BRANCH.fullmatch(name)}
        if selected_branches is not None:
            requested = {name.strip() for name in selected_branches.split(",")}
            if not requested or not requested <= available:
                errors.append("目标分支必须选择已存在的 main 或 v<年份>.<月份>.<序号> 分支。")
            elif selected_changed("目标分支") or not current_branches & available:
                desired -= current_branches
                desired |= requested
        if not desired & available:
            errors.append("至少需要一个真实存在的目标分支标签。")
    if errors:
        desired.add(PENDING)
    else:
        desired.discard(PENDING)
    result["add"] = sorted(desired - labels)
    result["remove"] = sorted(labels - desired)
    return result


class GitHub:
    def __init__(self):
        self.base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.repo = os.environ["GITHUB_REPOSITORY"]
        self.token = os.environ["GITHUB_TOKEN"]

    def request(self, method, path, data=None):
        request = Request(
            f"{self.base}/repos/{self.repo}" + (f"/{path}" if path else ""),
            data=json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json"},
            method=method,
        )
        with urlopen(request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else None

    def list_all(self, path):
        items = []
        page = 1
        while True:
            batch = self.request("GET", f"{path}?per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def main():
    from urllib.parse import quote

    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as stream:
        event = json.load(stream)
    number = event.get("issue", {}).get("number") or event.get("inputs", {}).get("issue_number")
    if not str(number).isdigit() or int(number) <= 0:
        raise ValueError("必须提供有效的 Issue 编号。")
    github = GitHub()
    issue = github.request("GET", f"issues/{number}")
    branches = [entry["name"] for entry in github.list_all("branches")]
    plan = plan_issue(issue, branches, event)
    if plan["skip"]:
        print("跳过 PR 或系统需求汇总 Issue。")
        return 0
    existing = {entry["name"] for entry in github.list_all("labels")}
    for name in plan["add"]:
        if name not in existing:
            description = "Issue 分类或目标分支信息待补充" if name == PENDING else f"目标分支：{name}"
            try:
                github.request("POST", "labels", {"name": name, "color": "d4c5f9", "description": description})
            except HTTPError as error:
                if error.code != 422:
                    raise
                # A concurrent issue may have created the same branch label.
                github.request("GET", f"labels/{quote(name, safe='')}")
    # Add first: replacing a target must never temporarily remove all targets.
    if plan["add"]:
        github.request("POST", f"issues/{number}/labels", {"labels": plan["add"]})
    for name in plan["remove"]:
        try:
            github.request("DELETE", f"issues/{number}/labels/{quote(name, safe='')}")
        except HTTPError as error:
            if error.code != 404:
                raise
    if plan["add"] or plan["remove"]:
        updated = github.request("GET", f"issues/{number}")
        actual = {entry if isinstance(entry, str) else entry["name"] for entry in updated.get("labels", [])}
        missing = set(plan["add"]) - actual
        remaining = set(plan["remove"]) & actual
        if missing or remaining:
            raise RuntimeError(
                f"Issue 标签写入后读回不一致：缺少 {sorted(missing)}；未移除 {sorted(remaining)}。"
            )
        # GITHUB_TOKEN label changes do not trigger the other issues workflow.
        repository = github.request("GET", "")
        github.request("POST", "actions/workflows/requirements-index.yml/dispatches",
                       {"ref": repository["default_branch"]})
    for error in plan["errors"]:
        print(error)
    print("Issue 标签检查未通过。" if plan["errors"] else "Issue 分类和目标分支标签检查通过。")
    return int(bool(plan["errors"]))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HTTPError as error:
        print(f"GitHub API 请求失败（HTTP {error.code}）。", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
