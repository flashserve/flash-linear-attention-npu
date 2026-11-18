#!/usr/bin/env python3
"""Create a deterministic daily build plan shared by ARM and x86.

Both machines resolve branch commits at the same Asia/Shanghai cutoff.  The
cutoff, rather than the wall-clock time of each runner, makes the resulting
branch/SHA pairs deterministic even when the two cron jobs start a few minutes
apart.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from daily_branches import is_formal_release


SCHEMA = "fla-npu-daily-plan-v1"
# China Standard Time has no daylight-saving transitions.  A fixed UTC+08:00
# timezone keeps the planner portable even on minimal Python installations
# without the optional tzdata package.
TIMEZONE = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
DEFAULT_REPO = "flashserve/flash-linear-attention-npu"
DEFAULT_BRANCHES = ("main", "v26.10.0", "v26.11.0")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_iso(value: str) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def request_json(url: str, token: str, attempts: int = 6,
                 allow_not_found: bool = False):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fla-npu-daily-plan",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    transient = (urllib.error.URLError, TimeoutError, OSError)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and allow_not_found:
                return None
            if exc.code < 500 or attempt + 1 == attempts:
                raise RuntimeError(f"GitHub API 请求失败: HTTP {exc.code}: {url}") from exc
        except transient as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(
                    f"GitHub API 请求失败: {type(exc).__name__}: {url}") from exc
        wait = min(60, 5 * (2 ** attempt))
        print(f"[daily-plan] GitHub 网络异常，{wait}s 后重试 "
              f"({attempt + 1}/{attempts - 1})", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise RuntimeError(f"GitHub API 请求失败: {url}")


def release_effective_at(release: dict, branch: str,
                         cutoff_utc: dt.datetime) -> bool:
    if not is_formal_release(release, branch):
        return False
    published = parse_iso(str(release.get("published_at") or
                              release.get("created_at") or ""))
    # A published GitHub Release normally has published_at.  If a compatible
    # API omits it, fail closed and treat the formal release as already active.
    return published is None or published <= cutoff_utc


def has_formal_release(branch: str, repo: str, api: str, token: str,
                       cutoff_utc: dt.datetime) -> tuple[bool, str]:
    if branch == "main":
        return False, ""
    for tag in (f"{branch}-release", f"release-{branch}-release", branch):
        url = (f"{api.rstrip('/')}/repos/{repo}/releases/tags/"
               f"{urllib.parse.quote(tag, safe='')}")
        release = request_json(url, token, allow_not_found=True)
        if release and release_effective_at(release, branch, cutoff_utc):
            return True, str(release.get("html_url") or tag)
    return False, ""


def resolve_commit(branch: str, repo: str, api: str, token: str,
                   cutoff_utc: dt.datetime) -> str:
    query = urllib.parse.urlencode({
        "sha": branch,
        "until": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "per_page": 1,
    })
    url = f"{api.rstrip('/')}/repos/{repo}/commits?{query}"
    commits = request_json(url, token)
    if not isinstance(commits, list) or not commits:
        raise RuntimeError(
            f"分支 {branch} 在截点 {cutoff_utc.isoformat()} 前没有可构建提交")
    sha = str(commits[0].get("sha", "")).lower()
    if not SHA_RE.fullmatch(sha):
        raise RuntimeError(f"分支 {branch} 返回了非法 commit SHA: {sha!r}")
    return sha


def build_plan(day: dt.date, cutoff_hour: int, branches: list[str],
               repo: str, api: str, token: str, latest: bool = False) -> dict:
    cutoff_local = dt.datetime.combine(
        day, dt.time(hour=cutoff_hour), tzinfo=TIMEZONE)
    if latest:
        cutoff_local = dt.datetime.now(TIMEZONE)
        day = cutoff_local.date()
    cutoff_utc = cutoff_local.astimezone(dt.timezone.utc)
    selected: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for branch in dict.fromkeys(branches):
        if branch != "main" and not re.fullmatch(
                r"v[0-9]+(?:[.][0-9A-Za-z_-]+)+", branch):
            raise RuntimeError(f"不支持的 daily 分支格式: {branch}")
        released, release_url = has_formal_release(
            branch, repo, api, token, cutoff_utc)
        if released:
            skipped.append({"branch": branch, "reason": "formal-release",
                            "release": release_url})
            continue
        if latest:
            url = f"{api.rstrip('/')}/repos/{repo}/branches/{urllib.parse.quote(branch, safe='')}"
            head = request_json(url, token)
            sha = str(head.get("commit", {}).get("sha", "")).lower()
            if not SHA_RE.fullmatch(sha):
                raise RuntimeError(f"最新分支 HEAD 不合法: {branch}")
        else:
            sha = resolve_commit(branch, repo, api, token, cutoff_utc)
        selected.append({"branch": branch, "sha": sha})

    return {
        "schema": SCHEMA,
        "version_scheme": "daily-date-sha7-no-g-v2",
        "resolution": "branch-head" if latest else "scheduled-cutoff",
        "date": day.strftime("%Y%m%d"),
        "timezone": "Asia/Shanghai",
        "cutoff": cutoff_local.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "repo": repo,
        "branches": selected,
        "skipped": skipped,
    }


def validate_plan(plan: dict, expected_date: str = "") -> list[dict[str, str]]:
    from daily_version import validate_date, version_for
    validate_date(str(plan.get("date", "")))
    if plan.get("schema") != SCHEMA:
        raise RuntimeError(f"daily plan schema 不支持: {plan.get('schema')!r}")
    if expected_date and plan.get("date") != expected_date:
        raise RuntimeError(
            f"daily plan 日期不匹配: {plan.get('date')} != {expected_date}")
    rows = plan.get("branches")
    if not isinstance(rows, list):
        raise RuntimeError("daily plan branches 不是列表")
    seen: set[str] = set()
    for row in rows:
        branch = str(row.get("branch", ""))
        sha = str(row.get("sha", "")).lower()
        if not branch or branch in seen or not SHA_RE.fullmatch(sha):
            raise RuntimeError(f"daily plan 分支记录非法: {row!r}")
        row["sha"] = sha
        version_for(branch, plan["date"], sha)
        seen.add(branch)
    return rows


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def local_day(value: str) -> dt.date:
    if value:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    return dt.datetime.now(TIMEZONE).date()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ARM/x86 共用的 daily commit 计划")
    parser.add_argument("--date", default="", help="北京时间日期 YYYYMMDD")
    parser.add_argument("--latest", action="store_true", help="手动触发时锁定当前分支 HEAD；生成一次后共用于双机")
    parser.add_argument("--cutoff-hour", type=int, default=1,
                        help="北京时间提交截点小时，默认 1")
    parser.add_argument("--branch", action="append", dest="branches",
                        help="候选分支，可重复；默认 main、v26.10.0、v26.11.0")
    parser.add_argument("--repo", default=os.getenv(
        "FLA_PKG_TARGET_GH_REPO", DEFAULT_REPO))
    parser.add_argument("--api", default="https://api.github.com")
    parser.add_argument("--output", type=Path, help="写入 plan JSON")
    parser.add_argument("--read", type=Path,
                        help="校验已有 plan 并输出 branch<TAB>sha")
    parser.add_argument("--expect-date", default="")
    args = parser.parse_args()

    try:
        if args.read:
            plan = json.loads(args.read.read_text(encoding="utf-8"))
            for row in validate_plan(plan, args.expect_date):
                print(f"{row['branch']}\t{row['sha']}")
            return 0
        if not 0 <= args.cutoff_hour <= 23:
            raise RuntimeError("--cutoff-hour 必须在 0..23")
        if not args.output:
            raise RuntimeError("生成计划必须指定 --output")
        day = local_day(args.date)
        token = os.getenv("GITHUB_TOKEN", "").strip()
        plan = build_plan(day, args.cutoff_hour,
                          args.branches or list(DEFAULT_BRANCHES),
                          args.repo, args.api, token, latest=args.latest)
        validate_plan(plan, day.strftime("%Y%m%d"))
        atomic_write(args.output, plan)
        print(f"[daily-plan] date={plan['date']} cutoff={plan['cutoff']} "
              f"branches={len(plan['branches'])}", file=sys.stderr)
        for row in plan["branches"]:
            print(f"[daily-plan] pin {row['branch']}={row['sha']}", file=sys.stderr)
        for row in plan["skipped"]:
            print(f"[daily-plan] skip {row['branch']}: {row['reason']} "
                  f"{row.get('release', '')}", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must print one actionable error
        print(f"[daily-plan][ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
