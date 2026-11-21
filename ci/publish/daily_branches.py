#!/usr/bin/env python3
"""Print branches that have no published formal release; diagnostics go to stderr.

Exact formal tags are checked, so historical daily tags such as
v26.10.0-20261028-020000 never disable daily builds. Public lookups work without
a token. Network/auth/rate-limit errors stop the run instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def is_formal_release(release: dict, branch: str) -> bool:
    if release.get("draft") or release.get("prerelease"):
        return False
    tag = release.get("tag_name", "")
    if tag in (f"{branch}-release", f"release-{branch}-release"):
        return True
    # A bare branch tag was sometimes used for daily builds. Accept it only
    # when it has a wheel with the exact formal version, without a local suffix.
    if tag == branch:
        version = branch.removeprefix("v")
        return any(
            str(asset.get("name", "")).startswith("flash_linear_attention_npu_")
            and f"-{version}-" in str(asset.get("name", ""))
            and str(asset.get("name", "")).endswith(".whl")
            for asset in release.get("assets") or []
        )
    return False


def get_release(repo: str, tag: str, api: str) -> dict | None:
    url = (f"{api.rstrip('/')}/repos/{repo}/releases/tags/"
           f"{urllib.parse.quote(tag, safe='')}")
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "fla-npu-daily-branches",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code < 500 or attempt == 2:
                raise RuntimeError(f"正式发布状态查询失败: {tag}，HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 2:
                raise RuntimeError(f"正式发布状态查询失败: {tag}，{type(exc).__name__}") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"正式发布状态查询失败: {tag}")


def select_daily_branches(branches: list[str], repo: str, api: str) -> list[str]:
    selected = []
    for branch in dict.fromkeys(branches):
        if branch == "main":
            selected.append(branch)
            continue
        published = None
        for tag in (f"{branch}-release", f"release-{branch}-release", branch):
            release = get_release(repo, tag, api)
            if release and is_formal_release(release, branch):
                published = release
                break
        if published:
            print(f"[daily] 跳过已正式发布分支 {branch}: "
                  f"{published.get('html_url') or published['tag_name']}", file=sys.stderr)
        else:
            selected.append(branch)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="过滤已正式发布的 daily 分支")
    parser.add_argument("branches", nargs="+")
    parser.add_argument("--repo", default=os.getenv(
        "FLA_PKG_TARGET_GH_REPO", "flashserve/flash-linear-attention-npu"))
    parser.add_argument("--api", default="https://api.github.com")
    args = parser.parse_args()
    try:
        branches = select_daily_branches(args.branches, args.repo, args.api)
    except Exception as exc:
        print(f"[daily][ERROR] {exc}", file=sys.stderr)
        return 1
    for branch in branches:
        print(branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
