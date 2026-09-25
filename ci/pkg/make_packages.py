#!/usr/bin/env python3
"""flash-linear-attention-npu 多 SOC / 多架构 wheel 自动编译脚本.

用法
----
    python ci/pkg/make_packages.py [--soc <列表>] [--arch <列表>] [选项]

    --soc   逗号分隔的 SOC 列表，可选值: ascend910b, ascend910_93, ascend950
            缺省 = 依次编译全部三个 SOC
    --arch  逗号分隔的架构列表，可选值: arm, x86
            缺省 = 编译全部两个架构
    --branch <分支名>
            切换到指定分支并同步远端最新代码后编译
            (git fetch + 强制重置，本地未提交修改会丢失)
    --pr <PR 编号>  (须与 --branch 同用)
            带指定 PR 出包: 校验 PR 的 base 分支必须为 --branch 指定的
            分支 (不一致直接拒绝)，校验通过后把 PR head 合入本地代码
            (git merge) 再执行出包流程。PR 状态非 open 时拒绝。
            --gh-release 模式下上传到独立 tag <分支名>-pr<PR 编号>，
            不会覆盖每日构建的 Release。多次执行无需手工清理: 每次
            都先把分支强重置到远端最新再重新合入；冲突时自动中止
            合并并恢复干净工作区。
    --gh-release
            cron / 独立调用模式: 在固定目录 (FLA_PKG_CRON_REPO_DIR) clone
            目标仓 flashserve/flash-linear-attention-npu 最新代码后编译，
            wheel 上传到目标仓的 GitHub Release (无需 PR)。
            x86 编译目录固定为远程 FLA_PKG_REMOTE_CRON_BASE。
    指定多个 SOC 时按顺序逐个执行编译。

独立调用示例
------------
    # clone 目标仓最新代码 (默认分支)，编译并上传目标仓 Release
    python ci/pkg/make_packages.py --gh-release \\
        --soc ascend910b,ascend950 --gh-token <token>

    # 同上，但编译目标仓指定分支
    python ci/pkg/make_packages.py --gh-release --branch v26.9.0 \\
        --soc all --gh-token <token>

    # 带指定 PR 出包 (校验 base 分支后合入编译，上传独立 Release)
    python ci/pkg/make_packages.py --gh-release --branch main --pr 123 \\
        --soc all --gh-token <token>

    # 仅本地编译当前仓指定分支最新代码 (不上传)
    python ci/pkg/make_packages.py --branch v26.9.0 --soc all --arch arm

    # CI 内部调用 (PR 评论触发，当前仓即 flashserve，产物走 Actions
    # Artifacts)
    python ci/pkg/make_packages.py --soc <soc> --arch <arch> --head-sha <sha>

编译行为
--------
* 并行: arm 与 x86 两个队列同时执行 (本地 Docker 与远程服务器各干
  各的)，同一架构内多个 SOC 按顺序串行 (资源独占)。
* 每个组合编译前清理: 仓库下的 build/ 与 third_party/ 一律清掉，
  不同 SOC / 分支的中间产物互相污染会导致编译偶发失败；third_party
  由编译脚本自动重新拉取，每个组合都从干净状态开始编译。
* 失败自动重试: 单个组合编译失败后自动重试 (默认 3 次，
  FLA_PKG_BUILD_RETRIES 可调)；每次重试与首次编译流程一致，先清
  缓存 (wheel 输出目录、build/、third_party/) 再重编。用户中断
  (Ctrl-C) 不重试。
* arm : 在本机 Docker 容器内执行
        "FLA_NPU_SOC=<SOC> python3 scripts/build_wheel.py"。
        容器内 <repo> 挂载到 /workspace/repo (third_party 在仓库目录
        内由编译脚本自动下载)，wheel 输出到独立目录。
* x86 : 通过 SSH 连接远程服务器 (默认 root@192.168.13.246，IdentityFile
        /workspace/huangjunzhe/ci/id_rsa)，把本地当前仓库 (已包含 PR 代码)
        直接同步到远程后按 SOC 顺序执行同样的编译命令，再把 wheel 同步回
        本机，无需在远程重新拉取代码。
        启动时若远程不可达，x86 工作线程在 FLA_PKG_CONNECT_WAIT (默认
        1800s) 窗口内每 30s 重试连接且每次尝试都打日志 (在独立线程执行，
        不阻塞 arm 队列)，窗口耗尽才判定远程编译不可用。
        编译进程通过 nohup 在远程后台运行、与 SSH 连接完全解耦: 网络闪断
        只会中断日志轮询，不会杀死编译；恢复后自动追上进度。
        轮询时若 status 未写出，会通过 pid + 服务器启动时间 (/proc/stat
        btime) 主动确认远程编译进程仍存活；服务器重启 / 进程被杀导致编译
        丢失时立即判定失败，不会傻等到超时。
* 版本号一致性: 向 arm / x86 编译环境注入实际分支名与 commit
  (FLA_NPU_BRANCH_NAME / FLA_NPU_COMMIT_ID，wheel 版本号优先读这两个
  变量)，远程 x86 工作区无 .git 也能生成与 arm 一致的版本号 (main
  分支带 +main.<commit>；detached HEAD 解析不到分支名时不注入)。

产物
----
每个 (SOC, ARCH) 组合产出一个目录:
    <out-dir>/<Release-<ref>-<DATETIME>-PR<pr>-<SOC>-<ARCH>-<commit>>/*.whl
同时写出 <out-dir>/artifact-name.txt (本次组合的产物名) 与
<out-dir>/manifest.json (所有组合的编译状态汇总，含 Release 上传信息)。
默认 <out-dir> 为仓内 ci/pkg/out；--gh-release 模式为仓外
<FLA_PKG_CRON_REPO_DIR>-out (避免 git clean 清掉产物)。

产物目录防残留: 不同分支的 wheel 版本号不同 (同名不互相覆盖)，固定
输出目录会把旧分支的包误当成本次产物上传。因此每个组合编译前清空其
wheel 输出目录 (本地与远程皆然)，Release 上传也只认编译返回的本次
产物名单。--gh-release 模式上传完成后清掉本地 wheels/artifacts，
日志保留在 <out-dir>/logs 供排查。

GitHub Release (--gh-release)
-----------------------------
所有组合执行完 (成功 + 失败一起收尾) 后一次性上传: 成功组合的 wheel
统一上传到 Release，失败 / 未执行 (中断) 的组合在 Release 描述中
注明原因。部分组合失败不影响成功包的上传。

环境变量
--------
元数据 (CI 注入；本地执行时自动推导):
    FLA_PKG_PR_ID        PR 编号
    FLA_PKG_REF_NAME     分支名 (用于产物命名)
    FLA_PKG_COMMIT_ID    commit id (默认取完整 sha 的前 7 位)
    FLA_PKG_DATETIME     时间戳 YYYYMMDD-HHMMSS (默认当前东八区时间)
本地 arm 编译:
    FLA_PKG_DOCKER_IMAGE        默认 fla-npu-ci:9.1.0-910b
远程 x86 编译:
    FLA_PKG_REMOTE_HOST      默认 192.168.13.246
    FLA_PKG_REMOTE_USER      默认 root
    FLA_PKG_SSH_KEY          默认 /workspace/huangjunzhe/ci/id_rsa
    FLA_PKG_REMOTE_WORKROOT  默认 /root/fla-npu-pkg
    FLA_PKG_REMOTE_PYTHON    默认 /root/miniconda3/envs/atk_fla/bin/python
    FLA_PKG_REMOTE_CANN_ENV  默认 source /usr/local/Ascend/ascend-toolkit/set_env.sh
远程 x86 编译运行控制:
    FLA_PKG_CONNECT_WAIT     启动时远程连接重试窗口秒数，默认 1800 (半小时内
                             每 30s 重试一次，每次尝试都打日志)
    FLA_PKG_BUILD_TIMEOUT    单个组合编译超时秒数，默认 14400 (4 小时)
    FLA_PKG_POLL_INTERVAL    编译状态轮询间隔秒数，默认 30
    FLA_PKG_POLL_FAIL_LIMIT  轮询连续失败容忍秒数，默认 3600 (超过才判定失败)
编译失败自动重试:
    FLA_PKG_BUILD_RETRIES    单个组合编译失败后的自动重试次数，默认 3
构建参数透传 (可选):
    FLA_NPU_OPS / FLA_NPU_BUILD_ARGS  原样传给编译命令 (调试用)
GitHub Release 上传 (--gh-release，代码来源目标仓):
    FLA_PKG_GH_TOKEN / GITHUB_TOKEN   API token (--gh-token 优先)
    FLA_PKG_TARGET_REPO_URL     目标仓地址 (默认 flashserve/flash-linear-attention-npu)
    FLA_PKG_TARGET_GH_REPO      Release 上传的 owner/name (默认 flashserve/flash-linear-attention-npu)
    FLA_PKG_CRON_REPO_DIR       本地 clone 目录 (默认 /workspace/huangjunzhe/ci/cron-pkg-repo，
                                与 CI 其他过程的目录隔离)
    FLA_PKG_REMOTE_CRON_BASE    远程 x86 编译目录 (默认 /data/huangjunzhe/ci/cron-pkg-repo)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SUPPORTED_SOCS = ("ascend910b", "ascend910_93", "ascend950")
SUPPORTED_ARCHES = ("arm", "x86")

DEFAULT_DOCKER_IMAGE = "fla-npu-ci:9.1.0-910b"
DEFAULT_REMOTE_HOST = "192.168.13.246"
DEFAULT_REMOTE_USER = "root"
DEFAULT_SSH_KEY = "/workspace/huangjunzhe/ci/id_rsa"
DEFAULT_REMOTE_WORKROOT = "/root/fla-npu-pkg"
DEFAULT_REMOTE_PYTHON = "/root/miniconda3/envs/atk_fla/bin/python"
DEFAULT_REMOTE_CANN_ENV = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"

# --gh-release (cron / 独立调用) 模式: 代码始终来自目标仓 flashserve，
# 在固定目录 clone 最新代码后编译，产物上传到目标仓 Release。
# 仓库存储目录专用 (cron-pkg-repo)，与 CI 其他过程的目录完全隔离，
# 避免其他过程并行操作同一份代码 / third_party 导致编译互相干扰。
TARGET_REPO_URL = "https://github.com/flashserve/flash-linear-attention-npu"
TARGET_GH_REPO = "flashserve/flash-linear-attention-npu"
DEFAULT_CRON_REPO_DIR = "/workspace/huangjunzhe/ci/cron-pkg-repo"
DEFAULT_REMOTE_CRON_BASE = "/data/huangjunzhe/ci/cron-pkg-repo"

# 同步本地仓库到远程 x86 服务器时排除的路径 (体积大或会在远程重新生成)。
# 以 "/" 开头的模式只匹配仓库根目录，其余匹配任意层级。
REPO_SYNC_EXCLUDES = [
    "/build", "/build_out", "/output", "/dist",
    "/third_party", "/ci/pkg/out", "/.ci-tmp", "/.ci-cache",
    "/.ci-pkg-cache", "__pycache__", "*.pyc", ".pytest_cache",
]

# wheel 收集模式: 兼容旧统一包名 flash_linear_attention_npu-* 与
# per-tier 包名 flash_linear_attention_npu_a2/_a3/_a5-* (a4a7958 起，
# 包名按 SOC 档位区分)。远程 out_dir 编译前 rm -f *.whl、本地 wheel_dir
# 每次尝试前清空，不会混入旧产物。
WHEEL_GLOB = "flash_linear_attention_npu*-*.whl"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[pkg][{stamp}] {message}", flush=True)


def env_or(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def clean_build_state(repo_root: Path) -> None:
    """清掉仓库下的 build/ 与 third_party/。

    不同 SOC / 分支的中间产物互相污染会导致编译偶发失败，每个组合
    编译前都调用；third_party 由编译脚本自动重新拉取。
    """
    for sub in ("build", "third_party"):
        target = repo_root / sub
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)


# ---------------------------------------------------------------------------
# 元数据 (产物命名)
# ---------------------------------------------------------------------------

def sanitize_ref_name(ref: str) -> str:
    ref = re.sub(r"[^A-Za-z0-9._-]+", "-", ref.strip())
    return ref.strip("-") or "unknown"


def git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, encoding="utf-8",
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def run_git(repo_root: Path, *args: str) -> None:
    """带日志的 git 执行，失败抛异常。"""
    argv = ["git", *args]
    print(f"$ {' '.join(shlex.quote(a) for a in argv)}", flush=True)
    result = subprocess.run(argv, cwd=str(repo_root),
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} 失败 (exit={result.returncode}): "
            f"{(result.stderr or '').strip()[-800:]}")
    output = (result.stdout or "").strip()
    if output:
        print("    " + output[-500:], flush=True)


def update_repo(repo_root: Path, branch: str) -> None:
    """切换到指定分支并同步远端最新代码 (等效 git pull 最新)。

    会把本地分支强重置到 origin/<branch> 并清理未跟踪文件 (含
    build/ 与 third_party/ 编译残留)，本地未提交修改会丢失。
    """
    log(f"同步分支 {branch} 到远端最新 (origin/{branch})")
    # 必须用显式分支 refspec: 仓里存在与分支同名的 tag 时 (远端同时有
    # refs/tags/v26.9.1 和 refs/heads/v26.9.1)，"git fetch origin <branch>"
    # 会按 tag 解析 (tag 优先级高于 branch)，既拿不到分支最新提交、也
    # 不更新 origin/<branch> 跟踪引用，导致长期编旧代码且无感知
    run_git(repo_root, "fetch", "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    run_git(repo_root, "checkout", "-f", "-B", branch, f"origin/{branch}")
    # 排除默认产物目录，避免上次编译的 wheel 被 clean 清掉
    run_git(repo_root, "clean", "-fdx", "-e", "ci/pkg/out")
    log(f"当前代码: {git_output(repo_root, 'rev-parse', 'HEAD')[:12]}")


def clone_or_update_repo(repo_dir: Path, branch: str) -> str:
    """在固定目录 clone 目标仓最新代码 (已存在则增量更新)。

    branch 为空时使用目标仓默认分支 (origin/HEAD)。返回实际使用的分支名。
    """
    url = env_or("FLA_PKG_TARGET_REPO_URL", TARGET_REPO_URL)
    if (repo_dir / ".git").is_dir():
        origin = git_output(repo_dir, "remote", "get-url", "origin")
        if origin.rstrip("/") != url.rstrip("/") and \
                origin.rstrip("/.git") != url.rstrip("/.git"):
            raise RuntimeError(
                f"{repo_dir} 已存在但 origin 是 {origin}，不是目标仓 {url}；"
                f"请清理该目录后重试")
        log(f"目标仓已存在，增量更新: {repo_dir}")
    else:
        log(f"clone 目标仓: {url} -> {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", url, str(repo_dir)],
            capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone {url} 失败:\n{(result.stderr or '').strip()[-800:]}")
    if not branch:
        symref = git_output(repo_dir, "symbolic-ref", "--short",
                            "refs/remotes/origin/HEAD")
        branch = symref[len("origin/"):] if symref.startswith("origin/") else ""
    if not branch:
        raise SystemExit("无法确定目标仓默认分支，请通过 --branch 指定")
    update_repo(repo_dir, branch)
    return branch


def fetch_pull_request_info(pr_number: int, api_base: str, repo: str,
                            token: str) -> dict:
    """查询 PR 元信息 (base 分支 / head commit / 状态)。

    有 token 时带认证 (GitHub 限额更高)，公开仓无 token 也能查询。
    """
    url = f"{api_base.rstrip('/')}/repos/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "fla-npu-make-packages"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def merge_pull_request(repo_root: Path, pr_number: int, branch: str,
                       api_base: str, repo: str, token: str) -> str:
    """校验 PR base 分支一致后，把 PR head 合入当前分支，返回合并后 HEAD。

    多次执行无需手工清理: 调用前分支已被 update_repo 强重置到
    origin/<branch>，上一次的合并提交不会残留，每次都基于远端最新
    代码重新合入。合并冲突时中止合并并恢复干净工作区，可重复执行。
    """
    try:
        info = fetch_pull_request_info(pr_number, api_base, repo, token)
    except Exception as exc:  # noqa: BLE001 — 转成可读错误
        raise RuntimeError(
            f"查询 PR #{pr_number} 失败 ({repo}): {exc}") from exc
    base = (info.get("base") or {}).get("ref", "")
    head_sha = (info.get("head") or {}).get("sha", "")
    state = info.get("state", "")
    if base != branch:
        raise SystemExit(
            f"PR #{pr_number} 的 base 分支是 '{base}'，与 --branch 指定的 "
            f"'{branch}' 不一致，拒绝出包")
    if state != "open":
        raise SystemExit(
            f"PR #{pr_number} 当前状态为 {state} (非 open)，拒绝出包")
    if not head_sha:
        raise SystemExit(f"PR #{pr_number} 缺少 head commit 信息")
    log(f"PR #{pr_number}: base={base}，head={head_sha[:12]}，state=open")
    log(f"拉取 PR #{pr_number} 的 head 代码 (pull/{pr_number}/head)")
    run_git(repo_root, "fetch", "origin", f"pull/{pr_number}/head")
    try:
        run_git(repo_root,
                "-c", "user.name=fla-npu-pkg",
                "-c", "user.email=fla-npu-pkg@localhost",
                "merge", "--no-edit", "-m", f"Merge PR #{pr_number}",
                "FETCH_HEAD")
    except Exception as exc:  # noqa: BLE001 — 冲突需中止并还原工作区
        subprocess.run(["git", "merge", "--abort"], cwd=str(repo_root),
                       capture_output=True)
        raise RuntimeError(
            f"PR #{pr_number} 合入 {branch} 失败 (多半与分支最新代码冲突，"
            f"请先在远端把 PR rebase / merge 到最新后重试): {exc}") from exc
    merged_sha = git_output(repo_root, "rev-parse", "HEAD")
    log(f"PR #{pr_number} 合入完成，合并后 HEAD: {merged_sha[:12]}")
    return merged_sha


def resolve_metadata(repo_root: Path) -> dict:
    commit = os.getenv("FLA_PKG_COMMIT_ID", "").strip()
    if not commit:
        commit = os.getenv("GITHUB_SHA", "").strip() or git_output(
            repo_root, "rev-parse", "--short=7", "HEAD")
    commit = re.sub(r"[^A-Za-z0-9]+", "", commit)[:7]

    ref = os.getenv("FLA_PKG_REF_NAME", "").strip()
    if not ref:
        branch = git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        ref = branch if branch and branch != "HEAD" else "unknown"

    datetime_str = os.getenv("FLA_PKG_DATETIME", "").strip()
    if not datetime_str:
        now = datetime.now(timezone(timedelta(hours=8)))
        datetime_str = now.strftime("%Y%m%d-%H%M%S")

    pr_id = os.getenv("FLA_PKG_PR_ID", "").strip() or "0"
    return {"ref": sanitize_ref_name(ref), "datetime": datetime_str,
            "pr_id": pr_id, "commit": commit}


def artifact_name(meta: dict, soc: str, arch: str) -> str:
    return (f"Release-{meta['ref']}-{meta['datetime']}-PR{meta['pr_id']}"
            f"-{soc}-{arch}-{meta['commit']}")


# ---------------------------------------------------------------------------
# 子进程 / SSH / rsync
# ---------------------------------------------------------------------------

def run_logged(argv: list, log_file: Path, prefix: str = "",
               env: dict | None = None, cwd: Path | None = None) -> int:
    """执行命令，输出同时打到本脚本 stdout 与日志文件。"""
    print(f"$ {' '.join(shlex.quote(a) for a in argv)}", flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(shlex.quote(a) for a in argv)}\n")
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(cwd) if cwd else None, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(f"{prefix}{line}", end="", flush=True)
        returncode = process.wait()
        handle.write(f"[exit={returncode}]\n")
    return returncode


class Remote:
    """远程 x86 服务器的 SSH / rsync 封装。"""

    def __init__(self):
        self.host = env_or("FLA_PKG_REMOTE_HOST", DEFAULT_REMOTE_HOST)
        self.user = env_or("FLA_PKG_REMOTE_USER", DEFAULT_REMOTE_USER)
        self.key = env_or("FLA_PKG_SSH_KEY", DEFAULT_SSH_KEY)
        self.workroot = env_or("FLA_PKG_REMOTE_WORKROOT", DEFAULT_REMOTE_WORKROOT)
        self.python = env_or("FLA_PKG_REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON)
        self.cann_env = env_or("FLA_PKG_REMOTE_CANN_ENV", DEFAULT_REMOTE_CANN_ENV)
        if not Path(self.key).is_file():
            raise RuntimeError(f"SSH 私钥不存在或不可读: {self.key}")
        self.ssh_options = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=20",
            "-o", "BatchMode=yes",
        ]
        self.rsync_shell = "ssh " + " ".join(self.ssh_options[0:6])

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def exec(self, script: str, *, timeout: int | None = None) -> subprocess.CompletedProcess:
        argv = (["ssh", "-i", self.key, *self.ssh_options, self.target, "bash", "-s"])
        return subprocess.run(
            argv, input=script, capture_output=True, text=True,
            errors="replace", timeout=timeout)

    def check(self, script: str, *, timeout: int | None = None) -> str:
        result = self.exec(script, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"远程命令执行失败 (exit={result.returncode}):\n{script}\n---\n{detail}")
        return result.stdout

    def _rsync(self, argv: list, desc: str, *, attempts: int,
               wait_seconds: int = 15) -> None:
        """执行 rsync，网络闪断自动重试 (rsync 本身支持增量续传)。"""
        last_error = ""
        for attempt in range(1, attempts + 1):
            result = subprocess.run(argv, capture_output=True, text=True,
                                     errors="replace")
            if result.returncode == 0:
                if attempt > 1:
                    log(f"[提示] {desc} 在第 {attempt} 次重试后成功")
                return
            last_error = (result.stderr or "").strip()[-2000:]
            if attempt < attempts:
                log(f"[警告] {desc} 失败 (第 {attempt}/{attempts} 次，"
                    f"exit={result.returncode})，{wait_seconds}s 后重试")
                time.sleep(wait_seconds)
        raise RuntimeError(f"{desc} 重试 {attempts} 次后仍失败:\n{last_error}")

    def rsync(self, src: str, dst: str, *, timeout: int = 3600,
              excludes: list | None = None, delete: bool = False,
              attempts: int = 3) -> None:
        # rsync 不会创建远程父目录，先确保其存在
        parent = dst.rsplit("/", 1)[0] if "/" in dst.rstrip("/") else ""
        if parent:
            self.check(f"mkdir -p {shlex.quote(parent)}")
        argv = ["timeout", str(timeout), "rsync", "-a"]
        if delete:
            argv.append("--delete")
        for pattern in (excludes or []):
            argv += ["--exclude", pattern]
        argv += ["-e", self.rsync_shell, src, f"{self.target}:{dst}"]
        self._rsync(argv, f"rsync 同步失败 ({src} -> {dst})",
                    attempts=attempts)

    def pull(self, remote_path: str, local_dir: Path, *, timeout: int = 600,
             attempts: int = 5, includes: list | None = None) -> None:
        """把远程目录拉回本机 (wheel 回传，编译耗时久，多重试避免白编)。

        includes 非空时只拉取匹配的文件 (如只回传 *.whl)。
        """
        argv = ["timeout", str(timeout), "rsync", "-a"]
        if includes:
            argv += ["--include=*/"]
            for pattern in includes:
                argv += ["--include", pattern]
            argv += ["--exclude=*"]
        argv += ["-e", self.rsync_shell,
                 f"{self.target}:{remote_path}/", f"{local_dir}/"]
        self._rsync(argv, f"rsync 回传失败 ({remote_path} -> {local_dir})",
                    attempts=attempts)


# ---------------------------------------------------------------------------
# arm: 本地 Docker 容器编译
# ---------------------------------------------------------------------------

def build_arm(repo_root: Path, soc: str, wheel_dir: Path, log_file: Path) -> list[Path]:
    image = env_or("FLA_PKG_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)

    # 每个组合编译前清掉仓库下的 build/ 与 third_party/: 不同 SOC /
    # 分支的中间产物互相污染会导致编译偶发失败；third_party 由编译
    # 脚本自动重新拉取，一律从干净状态开始
    clean_build_state(repo_root)

    inspect = subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, text=True)
    if inspect.returncode != 0:
        raise RuntimeError(
            f"Docker 镜像 {image} 不存在。请先构建 CI 镜像，例如:\n"
            f"  docker build -t {image} -f ci/Dockerfile .")

    wheel_dir.mkdir(parents=True, exist_ok=True)
    command = (
        "source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null "
        "|| source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true\n"
        "cd /workspace/repo\n"
        f"FLA_NPU_SOC={soc} python3 scripts/build_wheel.py "
        f"--wheel-dir /workspace/pkg-out"
    )
    argv = [
        "docker", "run", "--rm",
        "--name", f"fla-pkg-{soc}-arm-{int(time.time())}",
        "--network", "host",
        "-v", f"{repo_root}:/workspace/repo",
        "-v", f"{wheel_dir}:/workspace/pkg-out",
        "-e", f"FLA_NPU_SOC={soc}",
        "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "-e", "HOME=/root",
    ]
    for name in ("FLA_NPU_OPS", "FLA_NPU_BUILD_ARGS",
                 "FLA_NPU_BRANCH_NAME", "FLA_NPU_COMMIT_ID"):
        value = os.getenv(name, "").strip()
        if value:
            argv += ["-e", f"{name}={value}"]
    argv += [image, "bash", "-lc", command]

    returncode = run_logged(argv, log_file, prefix="    ")
    if returncode != 0:
        raise RuntimeError(f"arm 容器编译失败 (soc={soc}, exit={returncode})，详见日志 {log_file}")

    wheels = sorted(wheel_dir.glob(WHEEL_GLOB))
    if not wheels:
        raise RuntimeError(f"arm 编译完成但未找到 wheel (soc={soc})，详见日志 {log_file}")
    return wheels


# ---------------------------------------------------------------------------
# x86: 远程服务器编译
# ---------------------------------------------------------------------------

class RemoteWorkspace:
    """远程服务器上的代码工作区。

    CI 触发: 工作区在 <workroot>/<workkey> 下，每次触发独立目录，
    直接把本地当前仓库 (已检出 PR 代码) 同步过去，无需重新拉取。
    --gh-release: 固定目录 <FLA_PKG_REMOTE_CRON_BASE>，同样同步本地
    (已 clone 目标仓最新代码) 的仓库过去。
    """

    def __init__(self, remote: Remote, repo_root: Path, workkey: str,
                 base: str | None = None):
        self.remote = remote
        self.repo_root = repo_root
        self.base = base or f"{remote.workroot}/{workkey}"
        self.repo_dir = f"{self.base}/repo"

    def prepare(self) -> None:
        remote = self.remote
        remote.check(
            f"mkdir -p {shlex.quote(self.base)}\n"
            # 清理 7 天前的旧 CI 工作区，避免磁盘占满
            f"find {shlex.quote(remote.workroot)} -maxdepth 1 -type d -name 'pr-*' "
            f"-mtime +7 -exec rm -rf {{}} + 2>/dev/null || true\n"
            # 固定目录 (cron) 场景: 清理过期的编译输出
            f"find {shlex.quote(self.base)}/out -maxdepth 1 -mindepth 1 -type d "
            f"-mtime +7 -exec rm -rf {{}} + 2>/dev/null || true\n"
            "echo REMOTE_READY")
        log(f"远程工作区就绪: {self.base}")

        # 把本地当前仓库 (已包含 PR 代码) 整体同步到远程
        # (build/ 与 third_party/ 已在排除列表，远程编译时自动重新拉取)
        log(f"同步本地代码到远程: {self.repo_root} -> {self.repo_dir}")
        remote.rsync(f"{self.repo_root}/", self.repo_dir,
                     excludes=REPO_SYNC_EXCLUDES, delete=True)
        log("远程代码同步完成")

    @staticmethod
    def _poll_script(out_dir: str, offset: int, boot_ref: str = "") -> str:
        """生成状态轮询脚本: 返回状态标记、日志总字节数、增量日志内容。

        offset 为已读取的日志字节数；返回 LOGBYTES 后本地据此推进 offset，
        下次用 tail -c +<offset+1> 只取新增部分。

        status 未写出时不能默认当作"编译中": 服务器重启 / 进程被 kill 都
        会让 status 永远不出现。此时通过 pid 文件 + 编译进程命令行 + 服务器
        启动时间 (/proc/stat btime) 三重校验，编译进程确认已丢失则返回
        LOST:<reason>，本地立即判定失败而不是傻等超时。
        """
        status_file = shlex.quote(f"{out_dir}/status")
        log_file = shlex.quote(f"{out_dir}/build.log")
        pid_file = shlex.quote(f"{out_dir}/pid")
        boot_ref_q = shlex.quote(boot_ref or "unknown")
        return (
            f"if [ -f {status_file} ]; then "
            f'echo "__FLA_PKG_STATUS__:$(cat {status_file})"; '
            "else\n"
            "  lost=''\n"
            "  boot_now=$(awk '/^btime/{print $2}' /proc/stat 2>/dev/null "
            "|| echo unknown)\n"
            f"  if [ {boot_ref_q} != 'unknown' ] "
            f"&& [ \"$boot_now\" != {boot_ref_q} ]; then\n"
            "    lost='server_rebooted'\n"
            "  else\n"
            f"    pid_val=$(cat {pid_file} 2>/dev/null)\n"
            "    if [ -z \"$pid_val\" ]; then\n"
            "      lost='pid_file_missing'\n"
            "    elif ! ps -p \"$pid_val\" > /dev/null 2>&1; then\n"
            "      lost='process_gone'\n"
            "    else\n"
            "      pargs=$(ps -p \"$pid_val\" -o args= 2>/dev/null)\n"
            "      case \"$pargs\" in "
            "*timeout*build.sh*) ;; *) lost='process_replaced';; esac\n"
            "    fi\n"
            "  fi\n"
            "  if [ -n \"$lost\" ]; then "
            'echo "__FLA_PKG_STATUS__:LOST:$lost"; '
            "else echo __FLA_PKG_STATUS__:RUNNING; fi\n"
            "fi\n"
            f"size=$(wc -c < {log_file} 2>/dev/null || echo 0)\n"
            'echo "__FLA_PKG_LOGBYTES__:$size"\n'
            f"start={offset + 1}\n"
            f'if [ "$size" -lt {offset} ]; then start=1; fi\n'  # 日志被截断则从头读
            'take=$((size - start + 1))\n'
            '[ "$take" -lt 0 ] && take=0\n'
            f"tail -c +$start {log_file} 2>/dev/null | head -c $take\n"
        )

    @staticmethod
    def _parse_poll(output: str) -> tuple[str, int, str]:
        """解析轮询输出: (状态, 日志总字节, 增量日志内容)。"""
        lines = output.split("\n", 2)
        if len(lines) < 3 or not lines[0].startswith("__FLA_PKG_STATUS__:"):
            raise ValueError(f"轮询输出缺少状态标记: {output[:120]!r}")
        if not lines[1].startswith("__FLA_PKG_LOGBYTES__:"):
            raise ValueError(f"轮询输出缺少字节标记: {lines[1][:120]!r}")
        status = lines[0].split(":", 1)[1].strip()
        size = int(lines[1].split(":", 1)[1].strip() or "0")
        return status, size, lines[2]

    def build(self, soc: str, wheel_dir: Path, log_file: Path,
              stop: "threading.Event | None" = None) -> list[Path]:
        """远程 x86 编译 (detached 模式)。

        编译脚本经 base64 写到远程、由 nohup 后台启动，与 SSH 连接完全解耦
        (网络闪断不会杀死编译)；结束后退出码写入 status 文件。本地用独立
        短连接每 FLA_PKG_POLL_INTERVAL 秒轮询一次，顺带拉取增量日志；每次
        轮询失败都记录警告 (含已连续失败时长) 并继续，超过
        FLA_PKG_POLL_FAIL_LIMIT 秒连续失败才判定失败。status 未写出时主动
        探活远程编译进程 (pid + uptime)，进程丢失 (如服务器重启) 立即
        失败，不再傻等到超时。

        stop 非空时，置位后轮询尽快退出 (远程编译进程不受影响)。
        """
        remote = self.remote
        out_dir = f"{self.base}/out/{soc}-x86"
        build_env = ""
        for name in ("FLA_NPU_OPS", "FLA_NPU_BUILD_ARGS",
                     "FLA_NPU_BRANCH_NAME", "FLA_NPU_COMMIT_ID"):
            value = os.getenv(name, "").strip()
            if value:
                build_env += f"export {name}={shlex.quote(value)}\n"

        build_timeout = int(env_or("FLA_PKG_BUILD_TIMEOUT", "14400"))
        poll_interval = int(env_or("FLA_PKG_POLL_INTERVAL", "30"))
        poll_fail_limit = int(env_or("FLA_PKG_POLL_FAIL_LIMIT", "3600"))

        # 1) 编译脚本 (base64 编码写入，规避 stdin 管道缓冲的边界问题)
        #    编译前清掉远程旧 wheel: wheel 输出目录固定，增量构建遇到同名
        #    旧文件会跳过重新生成，旧分支的包会被误当成本次产物回传
        #    同时清掉仓库下的 build/ 与 third_party/: 不同 SOC / 分支的
        #    中间产物互相污染会导致编译偶发失败，third_party 由编译脚本
        #    自动重新拉取，一律从干净状态开始
        build_script = (
            "#!/usr/bin/env bash\n"
            f"{remote.cann_env}\n"
            f"cd {shlex.quote(self.repo_dir)}\n"
            f"rm -rf {shlex.quote(self.repo_dir)}/build "
            f"{shlex.quote(self.repo_dir)}/third_party\n"
            f"mkdir -p {shlex.quote(out_dir)}\n"
            f"rm -f {shlex.quote(out_dir)}/*.whl\n"
            f"{build_env}"
            f"FLA_NPU_SOC={soc} {shlex.quote(remote.python)} "
            f"scripts/build_wheel.py --wheel-dir {shlex.quote(out_dir)}\n"
        )
        encoded = base64.b64encode(build_script.encode("utf-8")).decode("ascii")
        remote.check(
            f"mkdir -p {shlex.quote(out_dir)}\n"
            f"rm -f {shlex.quote(out_dir)}/status "
            f"{shlex.quote(out_dir)}/build.log\n"
            f"echo {encoded} | base64 -d > {shlex.quote(out_dir)}/build.sh\n",
            timeout=120)

        # 2) nohup 后台启动: 输出重定向到 build.log，stdin 关闭，
        #    编译不依赖 SSH 连接存活；同时记录后台进程 pid 与服务器
        #    启动时间 (/proc/stat 的 btime 秒级时间戳，比 uptime -s 兼容
        #    性好: BusyBox uptime 不支持 -s，输出随时间变化会误判重启)，
        #    供轮询阶段探活校验
        startup = remote.check(
            f"cd {shlex.quote(out_dir)}\n"
            "nohup bash -c "
            f"'timeout {build_timeout} bash build.sh; echo $? > status' "
            "> build.log 2>&1 < /dev/null &\n"
            "echo $! > pid\n"
            "echo BOOT:$(awk '/^btime/{print $2}' /proc/stat 2>/dev/null "
            "|| echo unknown)\n",
            timeout=120)
        boot_ref = ""
        for line in startup.splitlines():
            if line.startswith("BOOT:"):
                boot_ref = line.split(":", 1)[1].strip()
        log(f"远程后台编译已启动 (超时上限 {build_timeout}s)，"
            f"每 {poll_interval}s 轮询一次状态")

        # 3) 轮询直到拿到退出码 (deadline 含缓冲，容忍 status 落盘延迟)
        deadline = time.monotonic() + build_timeout + 600
        fail_deadline: float | None = None
        offset = 0
        returncode: int | None = None
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as handle:
            while returncode is None:
                if stop is not None and stop.is_set():
                    raise RuntimeError(
                        "收到本地中断请求，停止等待远程编译 "
                        "(远程进程可能仍在运行，重跑该组合前请先确认)")
                if stop is not None:
                    stop.wait(poll_interval)
                else:
                    time.sleep(poll_interval)
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"x86 远程编译超时 (> {build_timeout}s，soc={soc})，"
                        f"远程进程可能仍在运行，详见日志 {log_file}")

                poll_error: str | None = None
                status = size = None
                chunk = ""
                try:
                    probe = remote.exec(
                        self._poll_script(out_dir, offset, boot_ref),
                        timeout=60)
                except subprocess.TimeoutExpired:
                    poll_error = "SSH 轮询连接超时"
                else:
                    if probe.returncode != 0:
                        poll_error = (probe.stderr or probe.stdout or ""
                                      ).strip()[-300:] or f"exit={probe.returncode}"
                    else:
                        try:
                            status, size, chunk = self._parse_poll(probe.stdout)
                        except ValueError as exc:
                            poll_error = str(exc)

                if poll_error is not None:
                    now = time.monotonic()
                    if fail_deadline is None:
                        fail_deadline = now + poll_fail_limit
                    if now >= fail_deadline:
                        raise RuntimeError(
                            f"x86 远程状态轮询连续失败超过 {poll_fail_limit}s "
                            f"(soc={soc})，编译进程可能仍在远程运行，"
                            f"详见日志 {log_file}")
                    # 每次失败都打日志 (含已连续失败时长)，避免静默等待
                    # 看起来像卡死；多行错误压缩成单行
                    waited = poll_fail_limit - (fail_deadline - now)
                    log(f"[警告] 状态轮询失败 (已连续 {waited:.0f}s / "
                        f"容忍 {poll_fail_limit}s，远程编译不受影响): "
                        f"{' '.join(poll_error.split())[:200]}")
                    continue
                if fail_deadline is not None:
                    log("[提示] 状态轮询已恢复，继续跟踪远程编译")
                fail_deadline = None

                if chunk:
                    handle.write(chunk)
                    handle.flush()
                    for line in chunk.splitlines(keepends=True):
                        print(f"    {line}", end="", flush=True)
                if size is not None and size >= offset:
                    offset = size
                if status.startswith("LOST"):
                    reason = status.split(":", 1)[1] if ":" in status else ""
                    detail = {
                        "server_rebooted":
                            "远程服务器已重启，后台编译进程随重启被终止",
                        "process_gone":
                            "远程编译进程已不存在 (可能被手动 kill 或服务器重启)",
                        "pid_file_missing":
                            "远程 pid 记录文件丢失，无法确认编译进程状态",
                        "process_replaced":
                            "原编译进程已退出 (pid 已被其他进程复用)",
                    }.get(reason, f"远程编译进程丢失 ({reason or '未知原因'})")
                    handle.write(f"[remote lost: {reason}]\n")
                    raise RuntimeError(
                        f"x86 远程编译已中断 (soc={soc}): {detail}，"
                        f"status 未写出，需重新编译该组合。详见日志 {log_file}")
                if status != "RUNNING":
                    try:
                        returncode = int(status)
                    except ValueError:
                        returncode = -1
                        log(f"[警告] 远程 status 异常 ({status!r})，按失败处理")
                    handle.write(f"[remote exit={returncode}]\n")

        if returncode != 0:
            raise RuntimeError(
                f"x86 远程编译失败 (soc={soc}, exit={returncode})，详见日志 {log_file}")

        # 4) wheel 回传 (只拉 wheel 文件，自动重试)
        wheel_dir.mkdir(parents=True, exist_ok=True)
        remote.pull(out_dir, wheel_dir, includes=[WHEEL_GLOB])
        wheels = sorted(wheel_dir.glob(WHEEL_GLOB))
        if not wheels:
            raise RuntimeError(f"x86 编译完成但未找到 wheel (soc={soc})，详见日志 {log_file}")
        return wheels


# ---------------------------------------------------------------------------
# GitHub Release 上传 (--gh-release)
# ---------------------------------------------------------------------------

class GitHubApiError(RuntimeError):
    """GitHub API 返回非成功状态码。"""

    def __init__(self, method: str, url: str, code: int, detail: str):
        super().__init__(
            f"GitHub API {method} {url} 失败 (HTTP {code}): {detail}")
        self.code = code


class GitHubRelease:
    """GitHub Release 上传客户端 (标准 GitHub REST API，兼容服务可用 --gh-api 覆盖)。"""

    def __init__(self, repo: str, token: str, api_base: str, prerelease: bool):
        self.repo = repo
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.prerelease = prerelease

    def _request(self, method: str, url: str, *,
                 json_body: dict | None = None, raw: bytes | None = None,
                 ok: tuple = (200, 201)) -> dict:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fla-npu-make-packages",
        }
        data: bytes | None = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            data = raw
            headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise GitHubApiError(method, url, exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"GitHub API 连接失败 ({method} {url}): {exc.reason}") from exc
        if status not in ok:
            raise RuntimeError(
                f"GitHub API {method} {url} 返回意外状态 {status}")
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def find_release(self, tag: str) -> dict | None:
        url = (f"{self.api_base}/repos/{self.repo}/releases/tags/"
               f"{urllib.parse.quote(tag)}")
        try:
            return self._request("GET", url)
        except GitHubApiError as exc:
            if exc.code == 404:
                return None
            raise

    def ensure_release(self, tag: str, title: str, target_sha: str,
                       body: str) -> dict:
        existing = self.find_release(tag)
        if existing is not None:
            log(f"复用已存在的 Release: {tag}")
            return existing
        payload = {
            "tag_name": tag,
            "name": title or tag,
            "body": body or "",
            "target_commitish": target_sha,
            "prerelease": self.prerelease,
            "draft": False,
        }
        try:
            return self._request(
                "POST", f"{self.api_base}/repos/{self.repo}/releases",
                json_body=payload)
        except GitHubApiError as exc:
            if exc.code == 422:  # tag 已存在或 Release 已被并发创建
                existing = self.find_release(tag)
                if existing is not None:
                    return existing
            raise

    def update_body(self, release: dict, body: str) -> None:
        """更新 Release 描述 (全部组合结束后写入最终状态汇总)。"""
        self._request(
            "PATCH",
            f"{self.api_base}/repos/{self.repo}/releases/{release['id']}",
            json_body={"body": body})

    def upload_asset(self, release: dict, name: str, path: Path) -> None:
        # 同名旧产物先删除 (重复触发 / 断点续传场景)
        for asset in release.get("assets") or []:
            if asset.get("name") == name:
                self._request(
                    "DELETE",
                    f"{self.api_base}/repos/{self.repo}/releases/assets/"
                    f"{asset['id']}", ok=(200, 204))
                break
        upload_url = release.get("upload_url") or ""
        base = (upload_url.split("{", 1)[0] if upload_url else
                f"{self.api_base}/repos/{self.repo}/releases/"
                f"{release.get('id')}/assets")
        url = f"{base}?name={urllib.parse.quote(name)}"
        self._request("POST", url, raw=path.read_bytes())


class ReleaseUploader:
    """Release 批量上传器: 所有组合执行完 (含失败 / 中断收尾) 后统一上传。

    finalize() 在收尾阶段一次性上传全部成功组合的 wheel，并把 Release
    描述更新为最终状态表: 失败组合注明原因，中断时未执行的组合注明
    "未执行"。部分组合失败不影响成功包的上传。
    """

    def __init__(self, gh_config: dict, args: argparse.Namespace, meta: dict,
                 head_sha: str, out_dir: Path):
        self.gh = GitHubRelease(gh_config["repo"], gh_config["token"],
                                args.gh_api, args.gh_prerelease)
        # tag 用分支名 (每日构建: 每个分支固定一个 Release，重复触发时
        # 复用)；--pr 出包时用独立 tag <分支名>-pr<PR 编号>，避免覆盖
        # 每日构建的 Release。title 为 <分支名>[-pr<PR 编号>]-<日期时间>
        # 不带 Release- 前缀与 commit
        self.pr = args.pr
        base_name = (f"{meta['ref']}-pr{meta['pr_id']}" if args.pr
                     else meta["ref"])
        self.tag = args.gh_tag or base_name
        self.title = args.gh_title or f"{base_name}-{meta['datetime']}"
        self.meta = meta
        self.head_sha = head_sha
        self.out_dir = out_dir

    def finalize(self, planned: list, results: list,
                 interrupted_reason: str = "") -> dict | None:
        """收尾: 一次性上传所有成功组合的 wheel，描述注明失败 / 未执行原因。

        planned 为全部计划组合 (soc, arch) 列表；results 为已执行的组合
        结果。没有任何组合成功时不创建 Release，返回 None。
        """
        snapshot = list(results)
        succ = [r for r in snapshot if r["status"] == "success"]
        if not succ:
            log("没有任何组合编译成功，跳过 Release 上传")
            return None

        done = {(r["soc"], r["arch"]): r for r in snapshot}
        rows = []
        for soc, arch in planned:
            record = done.get((soc, arch))
            if record is None:
                state, detail = "未执行", interrupted_reason or "未执行"
            elif record["status"] == "success":
                state, detail = "成功", f"{record['seconds']:.0f}s"
            else:
                state, detail = "失败", (record["error"] or "未知错误")[:200]
            rows.append(f"| {soc} | {arch} | {state} | {detail} |")

        errors: list = []   # (组合, 错误信息)
        notes: list = []
        if interrupted_reason:
            notes.append(f"任务中断: {interrupted_reason}，未执行组合未编译")

        def compose_body() -> str:
            all_notes = notes + (["部分产物上传失败: " + "; ".join(
                f"{where} {err}" for where, err in errors)] if errors else [])
            return "\n".join([
                "flash-linear-attention-npu 每日构建", "",
                *([f"- 含 PR: `#{self.pr}` (已合入 `{self.meta['ref']}`)"]
                  if self.pr else []),
                f"- 分支: `{self.meta['ref']}`  commit: `{self.head_sha[:12]}`",
                f"- 时间: {self.meta['datetime']} (UTC+8)",
                f"- SOC: {', '.join(dict.fromkeys(s for s, _ in planned))}"
                f"  架构: {', '.join(dict.fromkeys(a for _, a in planned))}",
                "",
                *[item for note in all_notes for item in (f"> {note}", "")],
                "| SOC | 架构 | 状态 | 说明 |",
                "| --- | --- | --- | --- |",
                *rows,
            ])

        try:
            release = self.gh.ensure_release(self.tag, self.title,
                                             self.head_sha, compose_body())
        except Exception as exc:  # noqa: BLE001 — Release 创建失败不丢编译结果
            log(f"[警告] Release 创建失败: {exc}")
            return {"tag": self.tag, "url": "", "assets": [],
                    "error": str(exc)}

        # 一次性上传全部成功组合的 wheel (单个失败不中断其余)。
        # 按 record["wheels"] 名单上传而不是 glob 目录，防止目录中
        # 意外残留的旧文件被一并传上去
        assets: list = []
        used_names: set = set()
        for record in succ:
            wheel_dir = (self.out_dir / "wheels"
                         / f"{record['soc']}-{record['arch']}")
            for fname in record["wheels"]:
                wheel = wheel_dir / fname
                if not wheel.is_file():
                    errors.append((f"{record['soc']}/{record['arch']}",
                                   f"wheel 文件缺失: {fname}"))
                    log(f"[警告] {record['soc']}/{record['arch']} "
                        f"wheel 文件缺失，跳过上传: {fname}")
                    continue
                name = fname
                if name in used_names:  # 同名 wheel 加前缀区分
                    name = f"{record['soc']}-{record['arch']}__{fname}"
                try:
                    self.gh.upload_asset(release, name, wheel)
                    used_names.add(name)
                    assets.append(name)
                    log(f"Release 产物已上传: {name}")
                except Exception as exc:  # noqa: BLE001
                    errors.append((f"{record['soc']}/{record['arch']}",
                                   str(exc)))
                    log(f"[警告] {record['soc']}/{record['arch']} "
                        f"Release 上传失败: {exc}")

        # 最终描述统一 PATCH 一次 (补上传失败注记，并覆盖复用场景旧描述)
        error_text = "; ".join(f"{where} {err}" for where, err in errors)
        try:
            self.gh.update_body(release, compose_body())
        except Exception as exc:  # noqa: BLE001 — 描述更新失败不影响产物
            log(f"[警告] Release 描述更新失败: {exc}")
            error_text = (error_text + "; " if error_text else "") + str(exc)
        return {"tag": self.tag,
                "url": release.get("html_url", ""),
                "assets": assets,
                "error": error_text}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_list(value: str, allowed: tuple, what: str) -> list:
    items: list = []
    for token in re.split(r"[,\s]+", value.strip()):
        if not token:
            continue
        token = token.lower()
        if token == "all":
            items.extend(allowed)
            continue
        if token not in allowed:
            raise SystemExit(
                f"不支持的 {what} 值: {token} (可选: {', '.join(allowed)})")
        items.append(token)
    seen, ordered = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def main() -> int:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="flash-linear-attention-npu 多 SOC / 多架构 wheel 编译",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(repo_default),
                        help="CI 模式的代码根目录 (默认: 脚本所在仓库；"
                             "--gh-release 模式忽略此项，使用目标仓 clone)")
    parser.add_argument("--soc", default=",".join(SUPPORTED_SOCS),
                        help=f"逗号分隔的 SOC 列表 (默认全部: {','.join(SUPPORTED_SOCS)})")
    parser.add_argument("--arch", default=",".join(SUPPORTED_ARCHES),
                        help=f"逗号分隔的架构列表 (默认全部: {','.join(SUPPORTED_ARCHES)})")
    parser.add_argument("--head-sha", default="",
                        help="PR 代码的完整 commit sha (CI 注入用，默认取 HEAD)")
    parser.add_argument("--out-dir", default="",
                        help="产物输出目录 (默认 <repo>/ci/pkg/out)")
    parser.add_argument("--list", action="store_true",
                        help="列出支持的 SOC / 架构组合并退出")
    parser.add_argument("--branch", default="",
                        help="指定分支: 同步该分支远端最新代码后编译 "
                             "(会重置本地分支并清理未跟踪文件，保留 third_party)")
    parser.add_argument("--pr", type=int, default=None,
                        help="带指定 PR 出包 (须与 --branch 同用): 校验 PR "
                             "的 base 分支必须为 --branch 指定的分支，"
                             "通过后把 PR 合入本地代码再执行出包流程; "
                             "--gh-release 时上传到独立 Release tag "
                             "<分支名>-pr<PR 编号>，不覆盖每日构建")
    parser.add_argument("--gh-release", action="store_true",
                        help="cron/独立模式: 在固定目录 clone 目标仓 "
                             "(flashserve) 最新代码编译，并上传到目标仓 "
                             "GitHub Release (无需 PR)")
    parser.add_argument("--gh-repo", default="",
                        help="Release 上传的 GitHub 仓库 owner/name "
                             "(默认 flashserve/flash-linear-attention-npu)")
    parser.add_argument("--gh-token", default="",
                        help="GitHub API token (默认环境变量 "
                             "FLA_PKG_GH_TOKEN / GITHUB_TOKEN)")
    parser.add_argument("--gh-tag", default="",
                        help="Release tag (默认分支名)")
    parser.add_argument("--gh-title", default="",
                        help="Release 标题 (默认 <分支名>-<日期时间>)")
    parser.add_argument("--gh-api", default="https://api.github.com",
                        help="GitHub API 地址 (兼容服务可覆盖)")
    parser.add_argument("--gh-prerelease", action="store_true",
                        help="将 Release 标记为预发布")
    args = parser.parse_args()

    if args.list:
        for soc in SUPPORTED_SOCS:
            for arch in SUPPORTED_ARCHES:
                print(f"{soc} / {arch}")
        return 0

    socs = parse_list(args.soc, SUPPORTED_SOCS, "SOC")
    arches = parse_list(args.arch, SUPPORTED_ARCHES, "架构")
    if not socs or not arches:
        raise SystemExit("--soc/--arch 解析结果为空")
    if args.pr is not None and not args.branch:
        raise SystemExit("--pr 需要同时通过 --branch 指定分支 "
                         "(用于校验 PR 的 base 分支)")

    # 代码来源:
    #   --gh-release (cron / 独立调用): 固定目录 clone 目标仓
    #       flashserve/flash-linear-attention-npu 最新代码
    #   CI 触发: 当前仓 (--repo-root，PR 已检出，即 flashserve 仓)
    gh_config: dict | None = None
    gh_repo = args.gh_repo or env_or("FLA_PKG_TARGET_GH_REPO", TARGET_GH_REPO)
    if args.gh_release:
        token = (args.gh_token or os.getenv("FLA_PKG_GH_TOKEN", "").strip()
                 or os.getenv("GITHUB_TOKEN", "").strip())
        if not token:
            raise SystemExit(
                "--gh-release 需要 token: --gh-token 或环境变量 "
                "FLA_PKG_GH_TOKEN / GITHUB_TOKEN")
        cron_repo = Path(env_or("FLA_PKG_CRON_REPO_DIR",
                                DEFAULT_CRON_REPO_DIR))
        branch = clone_or_update_repo(cron_repo, args.branch)
        gh_config = {"repo": gh_repo, "token": token}
        if args.pr is not None:
            # --pr: 校验 base 分支后把 PR 合入刚同步好的分支最新代码
            merge_pull_request(cron_repo, args.pr, branch, args.gh_api,
                               gh_repo, token)
        repo_root = cron_repo
        head_sha = git_output(repo_root, "rev-parse", "HEAD")
        log(f"GitHub Release 上传已使能: {gh_config['repo']} ({args.gh_api})")
        log(f"代码来源: {env_or('FLA_PKG_TARGET_REPO_URL', TARGET_REPO_URL)}"
            f" @ {branch} (clone 于 {repo_root})")
    else:
        repo_root = Path(args.repo_root).resolve()
        if not repo_root.is_dir():
            raise SystemExit(f"代码目录不存在: {repo_root}")
        branch = args.branch
        if branch:
            update_repo(repo_root, branch)
            if args.head_sha:
                log("[提示] 已按分支同步最新代码，忽略 --head-sha，使用实际 HEAD")
            head_sha = git_output(repo_root, "rev-parse", "HEAD")
        else:
            head_sha = args.head_sha or \
                os.getenv("FLA_PKG_COMMIT_ID", "").strip() or \
                git_output(repo_root, "rev-parse", "HEAD")
        if args.pr is not None:
            # --pr 本地模式: 合入后重新取 HEAD (合并提交)，后续版本号 /
            # 产物名 / 元数据都以合并后的实际代码为准
            merge_pull_request(repo_root, args.pr, branch, args.gh_api,
                               gh_repo,
                               args.gh_token
                               or os.getenv("FLA_PKG_GH_TOKEN", "").strip()
                               or os.getenv("GITHUB_TOKEN", "").strip())
            head_sha = git_output(repo_root, "rev-parse", "HEAD")
    if not head_sha:
        raise SystemExit("无法确定 HEAD commit，请通过 --head-sha 指定")

    # 向编译环境注入版本号环境变量 (所有分支): wheel 版本号优先读这两个
    # 变量 (fla_npu_artifacts.get_branch_name / get_commit_id)，注入后远程
    # x86 (工作区无 .git) 的版本号与 arm 一致 (main 分支带 +main.<commit>)
    build_branch = branch or git_output(
        repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if build_branch and build_branch != "HEAD":
        os.environ["FLA_NPU_BRANCH_NAME"] = build_branch
        os.environ["FLA_NPU_COMMIT_ID"] = head_sha
        log(f"分支构建: 注入 FLA_NPU_BRANCH_NAME={build_branch} / "
            f"FLA_NPU_COMMIT_ID={head_sha[:7]}，保证 arm/x86 wheel 版本号一致")

    if args.pr is not None:
        # --pr 出包: 产物名 / 远程工作区 / manifest 都标记 PR 编号，
        # 不影响无 --pr 的原有流程 (CI 模式仍读自身的 FLA_PKG_PR_ID)
        os.environ["FLA_PKG_PR_ID"] = str(args.pr)
    meta = resolve_metadata(repo_root)
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif args.gh_release:
        # cron 模式产物目录放仓外，避免 clone 仓更新时 git clean 清掉产物
        out_dir = cron_repo.parent / f"{cron_repo.name}-out"
    else:
        out_dir = repo_root / "ci" / "pkg" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    log(f"仓库: {repo_root} @ {head_sha[:12]}")
    log(f"SOC: {socs}  架构: {arches}")
    log(f"产物命名: Release-{meta['ref']}-{meta['datetime']}-"
        f"PR{meta['pr_id']}-<SOC>-<ARCH>-{meta['commit']}")
    log(f"产物目录: {out_dir}")

    # 远程 x86 编译准备 (仅需一次)。放在 x86 工作线程里执行:
    # 服务器不可达时持续重试连接而不阻塞 arm 队列
    workspace: RemoteWorkspace | None = None

    def init_x86_workspace() -> None:
        """x86 工作线程专用: 初始化远程编译环境。

        连接失败时在 FLA_PKG_CONNECT_WAIT (默认 1800s) 窗口内每 30s
        重试一次且每次尝试都打日志；窗口耗尽或被中断则放弃，x86 组合
        标记为远程编译不可用。私钥缺失等配置错误不重试、直接失败。
        """
        nonlocal workspace
        try:
            remote = Remote()
        except Exception as exc:  # noqa: BLE001 — 配置错误重试无意义
            log(f"[错误] 远程 x86 编译环境准备失败: {exc}")
            return
        connect_wait = int(env_or("FLA_PKG_CONNECT_WAIT", "1800"))
        deadline = time.monotonic() + connect_wait
        attempt = 0
        while True:
            if stop.is_set():
                log("[x86] 连接重试被中断，远程编译不可用")
                return
            attempt += 1
            try:
                probe = remote.exec("echo ok", timeout=30)
                if probe.returncode != 0:
                    raise RuntimeError(
                        (probe.stderr or probe.stdout or "").strip()[-300:])
                break  # 连接成功
            except Exception as exc:  # noqa: BLE001 — 连接失败需持续重试
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log(f"[错误] 无法连接远程服务器 {remote.target} "
                        f"(已持续重试 {connect_wait}s): "
                        f"{' '.join(str(exc).split())[:300]}")
                    log("[提示] 请确认服务器状态 / 私钥路径 "
                        f"({remote.key}) / 网络连通性")
                    return
                waited = connect_wait - remaining
                log(f"[警告] 远程服务器连接失败 (第 {attempt} 次，已重试 "
                    f"{waited:.0f}s / {connect_wait}s): "
                    f"{' '.join(str(exc).split())[:200]}")
                time.sleep(30)
        if attempt > 1:
            log(f"[提示] 远程服务器已恢复连接 (第 {attempt} 次尝试成功)")
        try:
            workkey = f"pr{meta['pr_id']}-{head_sha[:8]}"
            # --gh-release: x86 固定编译目录；CI: 每次触发独立工作区
            remote_base = (env_or("FLA_PKG_REMOTE_CRON_BASE",
                                  DEFAULT_REMOTE_CRON_BASE)
                           if args.gh_release else None)
            workspace = RemoteWorkspace(remote, repo_root, workkey,
                                        base=remote_base)
            workspace.prepare()
        except Exception as exc:  # noqa: BLE001 — 远程初始化失败不应中断整个脚本
            log(f"[错误] 远程 x86 编译环境准备失败: {exc}")

    # 组合执行: arm 与 x86 并行 (本地 Docker 与远程服务器同时干活)，
    # 同一架构内多个 SOC 串行 (单机资源独占，避免互相拖慢)
    planned = [(soc, arch) for soc in socs for arch in arches]
    results: list = []
    stop = threading.Event()
    uploader = ReleaseUploader(gh_config, args, meta, head_sha, out_dir) \
        if gh_config is not None else None

    def run_combo(soc: str, arch: str) -> None:
        prefix = f"[{soc}/{arch}]"
        name = artifact_name(meta, soc, arch)
        log(f"{prefix} === 开始编译 -> {name} ===")
        started = time.time()
        record = {"soc": soc, "arch": arch, "artifact": name,
                  "status": "failed", "wheels": [], "seconds": 0.0, "error": ""}
        try:
            wheel_dir = out_dir / "wheels" / f"{soc}-{arch}"
            log_file = out_dir / "logs" / f"{soc}-{arch}.log"
            # 失败自动重试: 每次尝试 (含重试) 与首次编译流程一致，
            # 先清缓存再重编 (wheel 输出目录在此清，build/ 与
            # third_party/ 由 build_arm / 远程编译脚本各自清)
            retries_done = 0
            max_retries = int(env_or("FLA_PKG_BUILD_RETRIES", "3"))
            if arch != "arm" and workspace is None:
                raise RuntimeError(
                    "远程 x86 编译不可用 (服务器连接或准备失败)，"
                    f"跳过 {soc}/{arch}")
            wheels: list = []
            for attempt in range(max_retries + 1):
                # 编译前清空上次产物: 不同分支的 wheel 版本号不同 (同名不
                # 覆盖)，不清理会把旧分支的包误当成本次产物上传
                if wheel_dir.exists():
                    shutil.rmtree(wheel_dir)
                wheel_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if arch == "arm":
                        wheels = build_arm(repo_root, soc, wheel_dir, log_file)
                    else:
                        wheels = workspace.build(soc, wheel_dir, log_file,
                                                stop=stop)
                    break
                except Exception as exc:  # noqa: BLE001 — 失败自动重试
                    if stop.is_set() or attempt >= max_retries:
                        raise
                    retries_done += 1
                    log(f"{prefix} === 编译失败: {exc}")
                    log(f"{prefix} 清除缓存后自动重试 "
                        f"(第 {retries_done}/{max_retries} 次)")
            artifact_dir = out_dir / "artifacts" / name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for wheel in wheels:
                target = artifact_dir / wheel.name
                if target.exists():
                    target.unlink()
                target.hardlink_to(wheel)
                record["wheels"].append(wheel.name)
            record["status"] = "success"
            log(f"{prefix} === 编译成功 "
                f"({time.time() - started:.0f}s, {len(wheels)} 个 wheel) ===")
        except Exception as exc:  # noqa: BLE001 — 单个组合失败不中断其余组合
            record["error"] = str(exc)
            if retries_done:
                record["error"] += f" (已自动重试 {retries_done} 次仍失败)"
            record["status"] = "failed"
            log(f"{prefix} === 编译失败: {record['error']}")
        finally:
            record["seconds"] = round(time.time() - started, 1)
            results.append(record)
            if record["status"] == "success":
                (out_dir / "artifact-name.txt").write_text(
                    f"{name}\n", encoding="utf-8")

    def arch_worker(arch: str) -> None:
        if arch != "arm":
            init_x86_workspace()  # 连不上时在此线程内重试，不拖 arm 队列
        for soc in socs:
            if stop.is_set():
                log(f"[{arch}] 已请求中断，跳过 {soc}/{arch} 及后续组合")
                return
            run_combo(soc, arch)

    interrupted = ""
    if len(arches) > 1:
        log("arm 与 x86 队列并行执行，同架构内 SOC 按顺序串行")
    threads = [threading.Thread(target=arch_worker, args=(arch,),
                                name=f"build-{arch}", daemon=True)
               for arch in arches]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop.set()
        interrupted = "用户中断 (Ctrl-C)"
        log("[中断] 已请求停止后续组合，正在编译的组合完成后收尾；"
            "已完成的产物将在收尾时统一上传，未执行组合将注明'未执行'")
        # 给工作线程短暂机会完成手头的上传动作
        for thread in threads:
            thread.join(timeout=10)

    # 收尾: 一次性上传所有成功组合的 wheel，描述注明失败 / 未执行原因
    release_info: dict | None = None
    if uploader is not None:
        try:
            release_info = uploader.finalize(
                planned, results, interrupted_reason=interrupted)
        except Exception as exc:  # noqa: BLE001 — 收尾失败不应丢弃编译结果
            release_info = {"tag": "", "url": "", "assets": [], "error": str(exc)}
            log(f"[错误] GitHub Release 收尾失败: {exc}")

    manifest = {
        "head_sha": head_sha,
        "ref": meta["ref"], "datetime": meta["datetime"],
        "pr_id": meta["pr_id"], "commit": meta["commit"],
        "socs": socs, "arches": arches,
        "interrupted": interrupted,
        "results": results,
    }
    if release_info is not None:
        manifest["release"] = release_info
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # --gh-release 收尾清理: 产物已上传 Release，本地 wheels/artifacts
    # 不再需要，清掉防止跨分支运行时新旧 wheel 混杂 (上传有错误时保留，
    # 便于人工补传)
    if uploader is not None and release_info is not None \
            and not release_info.get("error"):
        for sub in ("wheels", "artifacts"):
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        log("已清理本地产物缓存 (wheels/artifacts)，日志保留在 "
            f"{out_dir / 'logs'}")

    print("\n========== 编译汇总 ==========")
    failed = 0
    for soc, arch in planned:
        record = next((r for r in results
                       if (r["soc"], r["arch"]) == (soc, arch)), None)
        if record is None:
            print(f"[未执行] {soc:<14} {arch:<4}  "
                  f"({interrupted or '未执行'})")
            failed += 1
            continue
        mark = "成功" if record["status"] == "success" else "失败"
        print(f"[{mark}] {record['soc']:<14} {record['arch']:<4} "
              f"{record['seconds']:>7.1f}s  {record['artifact']}")
        if record["status"] == "success":
            for wheel in record["wheels"]:
                print(f"        - {wheel}")
        else:
            failed += 1
            print(f"        错误: {record['error'][:500]}")
    if release_info is not None:
        if release_info.get("error"):
            print(f"Release 上传失败: {release_info['error'][:500]}")
        else:
            print(f"Release: {release_info['url'] or release_info['tag']}")
            for asset in release_info["assets"]:
                print(f"        - {asset}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    if interrupted:
        log(f"任务被中断: {interrupted} (已执行 {len(results)}/{len(planned)} 个组合)")
        return 130
    if failed:
        log(f"共 {len(results)} 个组合，其中 {failed} 个失败")
        return 1
    if release_info is not None and release_info.get("error"):
        log("GitHub Release 上传失败")
        return 1
    log(f"共 {len(results)} 个组合全部编译成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
