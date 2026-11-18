#!/usr/bin/env python3
"""Build wheel matrices for the standalone publishing project.

run_compile.sh selects branches, pins source commits and controls publication.
Local ARM and x86 builders use the preloaded CI image, pin its image ID, prepare
the complete dependency set once and build with 64 jobs. Daily wheels can be
uploaded to OBS immediately after each successful platform build. The normal
daily flow never uses the legacy GitHub Release uploader.

The optional remote x86 mode requires FLA_PKG_REMOTE_HOST, FLA_PKG_REMOTE_USER
and FLA_PKG_SSH_KEY. The recommended deployment runs each CPU architecture on
its own host with the same plan JSON; no cross-host SSH is needed for builds.
See README.md for configuration, scheduling, recovery and report locations.
"""

from __future__ import annotations

import argparse
import errno
import base64
import json
import os
import platform
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

from npu_platforms import ARCH_TO_WHEEL, SOC_TO_PACKAGE
from docker_config import ARM_IMAGE, X86_IMAGE

SUPPORTED_SOCS = tuple(SOC_TO_PACKAGE)
SUPPORTED_ARCHES = tuple(ARCH_TO_WHEEL)

DEFAULT_DOCKER_IMAGE = ARM_IMAGE
DEFAULT_X86_DOCKER_IMAGE = X86_IMAGE
DEFAULT_REMOTE_HOST = ""
DEFAULT_REMOTE_USER = ""
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SSH_KEY = str(Path.home() / ".ssh" / "id_rsa")
DEFAULT_REMOTE_WORKROOT = "/data/fla-npu-pkg"
BUILD_JOB_LIMIT = 64
DOCKER_CPU_LIMIT = "64"
DEFAULT_OBS_BUCKET = "ascend-artifcat-packages"
DEFAULT_OBS_ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
OBS_PREFIX_ROOT = "pypi/packages/ascend/repos/pypi/nightly"

# --gh-release (cron / 独立调用) 模式: 代码始终来自目标仓 flashserve，
# 在固定目录 clone 最新代码后编译，产物上传到目标仓 Release。
# 仓库存储目录专用 (cron-pkg-repo)，与 CI 其他过程的目录完全隔离，
# 避免其他过程并行操作同一份代码 / third_party 导致编译互相干扰。
TARGET_REPO_URL = "https://github.com/flashserve/flash-linear-attention-npu"
TARGET_GH_REPO = "flashserve/flash-linear-attention-npu"
DEFAULT_CRON_REPO_DIR = str(PACKAGE_DIR / "repos" / "default")
DEFAULT_REMOTE_CRON_BASE = "/data/fla-npu-publish/cron-pkg-repo"

# 推送门禁失败日志目录: 存在编译失败时不推送 Release，失败汇总
# fail_log_<时间>.log 落盘到这里供排查
DEFAULT_FAIL_LOG_DIR = str(PACKAGE_DIR / "logs")

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


def enforce_build_job_limit() -> None:
    """Force build.sh to use -j64, replacing any caller-supplied -j value."""
    build_args = os.getenv("FLA_NPU_BUILD_ARGS", "").strip()
    build_args = re.sub(r"(?<!\S)-j\s*\d+(?=\s|$)", " ", build_args)
    build_args = " ".join(build_args.split())
    os.environ["FLA_NPU_BUILD_ARGS"] = (
        f"{build_args} -j{BUILD_JOB_LIMIT}".strip())


def publish_incremental_obs(soc: str, arch: str, wheel_dir: Path,
                            out_dir: Path) -> dict:
    """Upload one completed daily wheel immediately, without pruning.

    This hook is opt-in from run_compile.sh.  OBS credentials stay on the host
    and are never forwarded into the Docker build container.  The final batch
    publisher still runs after all SOCs finish and safely retries/skips these
    objects, so an incremental failure does not discard a successful build.
    """
    if env_or("FLA_PKG_INCREMENTAL_OBS", "0") != "1":
        return {"status": "disabled"}
    if env_or("FLA_PKG_BUILD_TYPE", "daily") != "daily":
        return {"status": "disabled"}

    package = SOC_TO_PACKAGE[soc]
    project = re.sub(r"[-_.]+", "-", package).lower()
    prefix = f"{OBS_PREFIX_ROOT}/{project}"
    publisher = PACKAGE_DIR / "obs_publish.py"
    if not publisher.is_file():
        raise RuntimeError(f"OBS 发布脚本不存在: {publisher}")

    report = out_dir / f"obs-daily-{package}-{arch}-incremental.json"
    command = [
        sys.executable, str(publisher),
        "--src", str(wheel_dir),
        "--bucket", env_or("FLA_PKG_OBS_BUCKET", DEFAULT_OBS_BUCKET),
        "--endpoint", env_or("FLA_PKG_OBS_ENDPOINT", DEFAULT_OBS_ENDPOINT),
        "--prefix", prefix,
        "--patterns", "*.whl",
        "--sync",
        "--report", str(report),
    ]
    if env_or("FLA_PKG_OBS_FORCE", "0") == "1":
        command.append("--force")
    for target in (prefix, prefix + "/latest"):
        current = command.copy()
        current[current.index("--prefix") + 1] = target
        target_report = report if target == prefix else report.with_name(report.stem + "-latest.json")
        current[current.index("--report") + 1] = str(target_report)
        log(f"[{soc}/{arch}] === 立即发布 OBS -> {target} ===")
        completed = subprocess.run(current, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{soc}/{arch} OBS 增量发布失败，exit={completed.returncode}")
    log(f"[{soc}/{arch}] === OBS 增量发布成功 ===")
    return {"status": "success", "prefix": prefix, "report": str(report)}


def read_public_version(repo_root: Path) -> str:
    """Read the release version declared by the checked-out source tree."""
    version_file = repo_root / "fla" / "__init__.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"无法读取版本文件 {version_file}: {exc}") from exc
    match = re.search(r'^__version__\s*=\s*[\'\"]([^\'\"]+)[\'\"]',
                      text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"无法从 {version_file} 解析 __version__")
    return match.group(1).strip()


def prepare_release_version(repo_root: Path, expected_version: str) -> str:
    """Promote only the matching .dev0 marker in the build checkout.

    The Git commit stays unchanged. The original marker is returned for the
    manifest, and the next checkout restores the source version for daily.
    """
    actual_version = read_public_version(repo_root)
    if not re.fullmatch(r"\d+\.\d+\.\d+", expected_version):
        raise RuntimeError(f"正式版本号必须是 X.Y.Z: {expected_version}")
    if actual_version == expected_version:
        return actual_version
    if actual_version != expected_version + ".dev0":
        raise RuntimeError(
            f"正式包版本不匹配: --release-version={expected_version}，"
            f"源码 fla/__init__.py={actual_version}")
    version_file = repo_root / "fla" / "__init__.py"
    text = version_file.read_text(encoding="utf-8")
    pattern = r'^(?P<prefix>__version__\s*=\s*)(?P<quote>[\'\"])(?P<version>[^\'\"]+)(?P=quote)'
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1 or matches[0].group("version") != actual_version:
        raise RuntimeError("正式包版本声明不唯一，拒绝自动适配")
    match = matches[0]
    text = text[:match.start("version")] + expected_version + text[match.end("version"):]
    version_file.write_text(text, encoding="utf-8")
    log(f"正式版本适配（仅构建工作区）: {actual_version} -> {expected_version}; "
        "仅修改 fla/__init__.py 的版本声明，源码 commit 不变")
    return actual_version


def release_branch_daily_local_version(branch: str, public_version: str,
                                       head_sha: str) -> str:
    """Return the non-repeating local version for a release-branch daily.

    A release branch may still carry ``<version>.dev0`` in the source tree.
    The repository helper removes that suffix from the public part, but older
    revisions compare the branch label with the untrimmed value and therefore
    produce ``26.10.0+26.10.0.dev<sha>``.  Supplying the local version from the
    CI layer keeps ``main`` unchanged while making a matching release branch
    read ``26.10.0+dev<sha>``.
    """
    normalized_branch = branch.strip()
    for prefix in ("refs/heads/", "origin/"):
        if normalized_branch.startswith(prefix):
            normalized_branch = normalized_branch[len(prefix):]
    match = re.fullmatch(r"v(?P<version>\d+(?:\.\d+)+)", normalized_branch)
    if not match:
        return ""
    daily_base = re.sub(r"\.dev\d+$", "", public_version)
    if match.group("version") != daily_base:
        return ""
    commit = re.sub(r"[^0-9A-Za-z]+", "", head_sha).lower()[:7]
    return f"dev{commit}" if commit else "dev"


BUILD_ENV_NAMES = (
    "FLA_NPU_OPS",
    "FLA_NPU_BUILD_ARGS",
    "FLA_NPU_BRANCH_NAME",
    "FLA_NPU_COMMIT_ID",
    "FLA_NPU_LOCAL_VERSION",
    "FLA_NPU_PACKAGE_VERSION",
    "FLA_NPU_DISABLE_LOCAL_VERSION",
    "FLA_NPU_BUILD_OFFLINE_BUNDLE",
    "FLA_PKG_BUILD_TYPE",
    "CI_SEED_THIRD_PARTY",
    "PIP_NO_CACHE_DIR",
    "CCACHE_DISABLE",
)


def docker_image(arch: str, x86_image: str = "") -> str:
    """编译镜像: 按架构一套固定基座。

    镜像里钉着 CANN / torch / torch_npu / 工具链，发布门禁的水位断言
    (GLIBCXX <= 3.4.29、manylinux_2_34) 就是针对这套基座算的。编译一旦
    回落到宿主环境，宿主升级 gcc (13 -> GLIBCXX_3.4.32) 或 CANN 就会静默
    改变产物水位，用户侧才报 "version 'GLIBCXX_3.4.32' not found"。

    arm 侧全部 SOC 共用 910b 镜像 (FLA_PKG_DOCKER_IMAGE 可覆盖)；同名
    tag 在各编译机上按架构各自构建: arm 机上是 arm64 镜像 (arm 侧目前
    没有 950 镜像，沿用 910b 镜像)。x86 侧全部 SOC 统一用 --x86-docker
    指定的镜像，默认 950 镜像 (ci/Dockerfile.ascend950，仅 x86_64，与
    发布路径 pypi.yml 的 a5 行同一镜像)。
    """
    if arch == "x86":
        return x86_image or DEFAULT_X86_DOCKER_IMAGE
    return env_or("FLA_PKG_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)


def wheel_check_version_args() -> str:
    if env_or("FLA_PKG_BUILD_TYPE", "daily") == "release":
        return ""
    expected = os.getenv("FLA_NPU_PACKAGE_VERSION", "").strip()
    return " --expect-version " + shlex.quote(expected) if expected else " --allow-local-version"


DEPENDENCY_SCRIPTS = ("prepare_publish_dependencies.py", "ci_git_dependencies.py")


def dependency_preparation_command() -> str:
    fresh = (env_or("FLA_PKG_NO_CACHE", "0") == "1"
             or os.getenv("CI_SEED_THIRD_PARTY", "true") != "true")
    return ("python3 /workspace/prepare_publish_dependencies.py"
            + (" --fresh" if fresh else "") + "\n")


def x86_build_script(repo_dir: str, out_dir: str, soc: str, image: str,
                     container: str, container_env: str) -> str:
    """远程 x86 编译脚本 (在远程机上以 root 执行)。

    宿主只做三件事: 清掉上一次的 build/ 与 third_party/、准备 wheel 输出目录、
    启动容器。编译本体跑在 CI 镜像里，CANN / torch / torch_npu / 工具链全部
    取自镜像，宿主 pip 一次或升级 gcc / CANN 都不会再改变产物水位。

    容器内顺序: 重建 third_party 种子 (两侧架构拉到同一份 catlass/opbase，
    kernel 才同源) -> 编译 -> 跑发布门禁的水位检查 (GLIBCXX <= 3.4.29、
    manylinux_2_34、tier/arch 命名)，x86 的水位越线在每日包里就红。
    """
    allow_local = wheel_check_version_args()
    inner = (
        "set -euo pipefail\n"
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        # launcher 在构建机上链接 torch，镜像里的 torch 必须可导入
        "python3 -c 'import torch; print(\"[pkg] torch\", torch.__version__)'\n"
        "echo \"[pkg] 构建基座: ${FLA_NPU_CI_BASELINE:-unknown} $(uname -m)\"\n"
        + dependency_preparation_command()
        + f"FLA_NPU_SOC={soc} python3 scripts/build_wheel.py "
        "--wheel-dir /workspace/pkg-out\n"
        "tier=$(python3 -c \"import sys; sys.path.insert(0, 'scripts'); "
        "from fla_npu_artifacts import get_tier; print(get_tier())\")\n"
        "python3 scripts/check_pypi_wheel.py /workspace/pkg-out/*.whl "
        "--expect-tier \"$tier\" --expect-arch \"$(uname -m)\""
        + allow_local + "\n"
    )
    return (
        "#!/usr/bin/env bash\n"
        "cd " + shlex.quote(repo_dir) + "\n"
        "rm -rf " + shlex.quote(repo_dir) + "/build "
        + shlex.quote(repo_dir) + "/third_party "
        + shlex.quote(repo_dir) + "/build_out "
        + shlex.quote(repo_dir) + "/output\n"
        "mkdir -p " + shlex.quote(out_dir) + "\n"
        "rm -f " + shlex.quote(out_dir) + "/*.whl\n"
        "echo \"[pkg] 编译镜像: $(docker image inspect --format '{{.Id}}' "
        + shlex.quote(image) + ")\"\n"
        "docker run --rm --cpus " + shlex.quote(DOCKER_CPU_LIMIT)
        + " --name " + shlex.quote(container) + " --network host \\\n"
        "  -v " + shlex.quote(repo_dir) + ":/workspace/repo \\\n"
        "  -v " + shlex.quote(out_dir) + ":/workspace/pkg-out \\\n"
        "  -e FLA_NPU_SOC=" + soc + " -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "
        "-e HOME=/root \\\n"
        + container_env
        + "".join("  -v " + shlex.quote(str(Path(repo_dir).parent / "publish-tools" / script)
                     + ":/workspace/" + script + ":ro") + " " for script in DEPENDENCY_SCRIPTS)
        + "  " + shlex.quote(image) + " bash -lc " + shlex.quote(inner) + "\n"
    )


def clean_build_state(repo_root: Path) -> None:
    """清掉仓库下的 build/ 与 third_party/。

    不同 SOC / 分支的中间产物互相污染会导致编译偶发失败，每个组合
    编译前都调用；third_party 由编译脚本自动重新拉取。
    """
    for sub in ("build", "build_out", "output", "third_party"):
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
    """Run Git; retry transient clone/fetch failures without rebuilding wheels."""
    network_op = bool(args and args[0] in ("clone", "fetch"))
    argv = ["git"]
    if network_op:
        # GitHub occasionally resets long HTTP/2 fetches on the ARM host
        # (curl 92 / early EOF).  HTTP/1.1 plus an explicit low-speed window is
        # more stable through the local mihomo proxy.
        argv += [
            "-c", "http.version=HTTP/1.1",
            "-c", "http.lowSpeedLimit=1024",
            "-c", "http.lowSpeedTime=120",
        ]
    argv += list(args)
    try:
        configured_attempts = int(os.getenv("FLA_PKG_GIT_RETRIES", "6"))
    except ValueError:
        configured_attempts = 6
    try:
        git_timeout = max(30, int(os.getenv("FLA_PKG_GIT_TIMEOUT", "300")))
    except ValueError:
        git_timeout = 300
    attempts = max(1, configured_attempts) if network_op else 1
    transient = ("gnutls", "tls", "ssl", "timed out", "couldn't connect",
                 "connection reset", "early eof", "remote end hung up",
                  "could not resolve", "rpc failed", "failed to connect",
                  "http/2", "curl 56", "curl 92", "stream 0 was not closed",
                  "network is unreachable", "connection refused")
    child_env = os.environ.copy()
    child_env["GIT_TERMINAL_PROMPT"] = "0"
    for attempt in range(attempts):
        print(f"$ {' '.join(shlex.quote(a) for a in argv)}", flush=True)
        try:
            result = subprocess.run(
                argv, cwd=str(repo_root), capture_output=True, text=True,
                env=child_env, timeout=git_timeout)
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                argv, 124, stdout=exc.stdout or "",
                stderr=f"git network command timed out after {git_timeout}s")
        if result.returncode == 0:
            break
        detail = (result.stderr or "").strip()
        if attempt + 1 < attempts and any(word in detail.lower() for word in transient):
            wait = min(60, 5 * (2 ** attempt))
            log(f"Git network failure; wait {wait}s then retry "
                f"{attempt + 1}/{attempts - 1}")
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit={result.returncode}): "
            f"{detail[-800:]}")
    output = (result.stdout or "").strip()
    if output:
        print("    " + output[-500:], flush=True)


def update_repo(repo_root: Path, branch: str, commit_sha: str = "") -> None:
    """切换到指定分支并同步远端最新代码 (等效 git pull 最新)。

    会把本地分支强重置到 origin/<branch> 并清理未跟踪文件 (含
    build/ 与 third_party/ 编译残留)，本地未提交修改会丢失。
    """
    if commit_sha and not re.fullmatch(r"[0-9a-fA-F]{40}", commit_sha):
        raise RuntimeError(f"非法 commit SHA: {commit_sha!r}")
    commit_sha = commit_sha.lower()
    if commit_sha:
        log(f"同步分支 {branch} 并锁定 daily plan commit {commit_sha}")
    else:
        log(f"同步分支 {branch} 到远端最新 (origin/{branch})")
    # 必须用显式分支 refspec: 仓里存在与分支同名的 tag 时 (远端同时有
    # refs/tags/v26.9.1 和 refs/heads/v26.9.1)，"git fetch origin <branch>"
    # 会按 tag 解析 (tag 优先级高于 branch)，既拿不到分支最新提交、也
    # 不更新 origin/<branch> 跟踪引用，导致长期编旧代码且无感知
    run_git(repo_root, "fetch", "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    target = f"origin/{branch}"
    if commit_sha:
        if git_output(repo_root, "cat-file", "-t", commit_sha) != "commit":
            # The branch may have advanced after the 01:00 cutoff and a shallow
            # fetch may not contain the planned commit.  Fetch that exact object
            # without changing the branch pin.
            try:
                run_git(repo_root, "fetch", "--depth", "1", "origin", commit_sha)
            except RuntimeError:
                # Some servers reject unadvertised SHA wants.  Deepen only the
                # selected branch and check the planned object again.
                run_git(repo_root, "fetch", "--deepen", "200", "origin",
                        f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
        actual = git_output(repo_root, "rev-parse", f"{commit_sha}^{{commit}}")
        if actual.lower() != commit_sha:
            raise RuntimeError(f"无法取得计划 commit: {branch}={commit_sha}")
        target = commit_sha
    run_git(repo_root, "checkout", "-f", "-B", branch, target)
    # 排除默认产物目录，避免上次编译的 wheel 被 clean 清掉
    run_git(repo_root, "clean", "-fdx", "-e", "ci/pkg/out")
    actual_head = git_output(repo_root, "rev-parse", "HEAD").lower()
    if commit_sha and actual_head != commit_sha:
        raise RuntimeError(
            f"计划 commit 检出不一致: {branch} expected={commit_sha} actual={actual_head}")
    log(f"当前代码: {actual_head[:12]}")


def clone_or_update_repo(repo_dir: Path, branch: str,
                         commit_sha: str = "") -> str:
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
        # 每个发布分支使用独立固定目录，不需要完整历史；浅克隆能显著降低
        # 首次 daily/release 准备时间，后续 update_repo 仍显式抓取该分支。
        clone_args = ["clone", "--depth", "1", "--no-tags"]
        if branch:
            clone_args += ["--single-branch", "--branch", branch]
        clone_args += [url, str(repo_dir)]
        run_git(repo_dir.parent, *clone_args)
    if not branch:
        symref = git_output(repo_dir, "symbolic-ref", "--short",
                            "refs/remotes/origin/HEAD")
        branch = symref[len("origin/"):] if symref.startswith("origin/") else ""
    if not branch:
        raise SystemExit("无法确定目标仓默认分支，请通过 --branch 指定")
    update_repo(repo_dir, branch, commit_sha)
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
        if not self.host or not self.user:
            raise RuntimeError("Remote builds require FLA_PKG_REMOTE_HOST and FLA_PKG_REMOTE_USER")
        if not Path(self.key).is_file():
            raise RuntimeError(f"SSH 私钥不存在或不可读: {self.key}")
        self.ssh_options = [
            "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=20",
            "-o", "BatchMode=yes",
        ]
        # rsync 的 ssh 必须显式带上 -i <key> 与 BatchMode=yes: 远端通常只授权
        # FLA_PKG_SSH_KEY 这一把 key (默认 key 恰好也被授权时才会"看起来正常"，
        # 已授权的默认 key 可能掩盖配置问题)，缺 -i 会导致 Permission denied，缺
        # BatchMode 则可能卡在交互式提示上。
        self.rsync_shell = ("ssh -i " + shlex.quote(self.key) + " "
                            + " ".join(self.ssh_options))

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

def image_digest(image: str) -> str:
    """镜像的不可变 ID: 记进日志，事后能回溯产物是哪套基座编的。"""
    try:
        out = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            encoding="utf-8", stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
    return out.split(":")[-1][:12] or "unknown"


def build_local(repo_root: Path, soc: str, wheel_dir: Path, log_file: Path,
                arch: str = "arm", x86_image: str = "") -> list[Path]:
    if platform.machine() != ARCH_TO_WHEEL[arch]:
        raise RuntimeError(f"宿主架构 {platform.machine()} 无法本地构建 {arch}")
    image = docker_image(arch, x86_image)

    # 每个组合编译前清掉仓库下的 build/ 与 third_party/: 不同 SOC /
    # 分支的中间产物互相污染会导致编译偶发失败；third_party 由编译
    # 脚本自动重新拉取，一律从干净状态开始
    inspect = subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, text=True)
    if inspect.returncode != 0:
        raise RuntimeError(
            f"无法访问专用 Docker 镜像 {image}: {inspect.stderr.strip()[-500:]}\n"
            f"请检查 Docker 权限并执行 {PACKAGE_DIR / 'build_docker.sh'}")

    clean_build_state(repo_root)

    # Resolve once so a concurrent CI image upgrade cannot change this build
    # or its ownership-cleanup environment halfway through the operation.
    image_id = json.loads(inspect.stdout)[0]["Id"]
    log(f"{arch} 本地编译镜像: {image} ({image_id})")
    wheel_dir.mkdir(parents=True, exist_ok=True)
    allow_local = wheel_check_version_args()
    no_cache = env_or("FLA_PKG_NO_CACHE", "0") == "1"
    prepare_dependencies = dependency_preparation_command()
    command = (
        "set -e\n"
        "source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null "
        "|| source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        "cd /workspace/repo\n"
        + prepare_dependencies +
        f"FLA_NPU_SOC={soc} python3 scripts/build_wheel.py "
        f"--wheel-dir /workspace/pkg-out\n"
        "tier=$(python3 -c \"import sys; sys.path.insert(0, 'scripts'); "
        "from fla_npu_artifacts import get_tier; print(get_tier())\")\n"
        "python3 scripts/check_pypi_wheel.py /workspace/pkg-out/*.whl "
        "--expect-tier \"$tier\" --expect-arch \"$(uname -m)\""
        + allow_local
    )
    argv = [
        "docker", "run", "--rm",
        "--cpus", DOCKER_CPU_LIMIT,
        "--name", f"fla-publish-{soc}-{arch}-{int(time.time())}",
        "--network", "host",
        # Match repository CI; restore bind-mount ownership in finally below.
        "--user", "0:0",
        "-v", f"{repo_root}:/workspace/repo",
        "-v", f"{wheel_dir}:/workspace/pkg-out",
        "-e", f"FLA_NPU_SOC={soc}",
        "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "-e", "HOME=/tmp",
    ]
    for script in DEPENDENCY_SCRIPTS:
        argv += ["-v", f"{PACKAGE_DIR / script}:/workspace/{script}:ro"]
    if no_cache:
        argv += ["-e", "PIP_NO_CACHE_DIR=1", "-e", "CCACHE_DISABLE=1"]
    for name in BUILD_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            argv += ["-e", f"{name}={value}"]
    argv += [image_id, "bash", "-lc", command]

    try:
        returncode = run_logged(argv, log_file, prefix="    ")
    finally:
        # Shared CI images may need root for their toolkit scripts. Only the
        # two task-owned bind mounts are chowned; symlinks are not followed.
        subprocess.run([
            "docker", "run", "--rm", "--network", "none", "--user", "0:0",
            "--entrypoint", "chown",
            "-v", f"{repo_root}:/workspace/repo",
            "-v", f"{wheel_dir}:/workspace/pkg-out",
            image_id, "-Rh", f"{os.getuid()}:{os.getgid()}",
            "/workspace/repo", "/workspace/pkg-out",
        ], check=True)
    if returncode != 0:
        raise RuntimeError(f"{arch} 容器编译失败 (soc={soc}, exit={returncode})，详见日志 {log_file}")

    wheels = sorted(wheel_dir.glob(WHEEL_GLOB))
    if not wheels:
        raise RuntimeError(f"{arch} 编译完成但未找到 wheel (soc={soc})，详见日志 {log_file}")
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
                 base: str | None = None, x86_image: str = ""):
        self.remote = remote
        self.repo_root = repo_root
        self.base = base or f"{remote.workroot}/{workkey}"
        self.repo_dir = f"{self.base}/repo"
        # x86 侧全部 SOC 共用的编译镜像 (--x86-docker，默认 950 镜像)
        self.x86_image = x86_image

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
        tools_dir = f"{self.base}/publish-tools"
        remote.check("mkdir -p " + shlex.quote(tools_dir))
        for script in DEPENDENCY_SCRIPTS:
            remote.rsync(str(PACKAGE_DIR / script), tools_dir + "/")
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
        image = docker_image("x86", self.x86_image)
        container = f"fla-publish-{soc}-x86-{int(time.time())}"
        # 编译环境由容器提供，宿主只负责 rsync 与启动 docker: CANN / torch /
        # torch_npu / 工具链都取自镜像，宿主 pip 一次或升级 gcc / CANN 不会再
        # 改变产物 (旧路径直接跑宿主的 atk_fla conda env，是水位越线的来源)。
        container_env = ""
        for name in BUILD_ENV_NAMES:
            value = os.getenv(name, "").strip()
            if value:
                container_env += f"  -e {name}={shlex.quote(value)} \\\n"

        build_timeout = int(env_or("FLA_PKG_BUILD_TIMEOUT", "14400"))
        poll_interval = int(env_or("FLA_PKG_POLL_INTERVAL", "30"))
        poll_fail_limit = int(env_or("FLA_PKG_POLL_FAIL_LIMIT", "3600"))

        # 1) 编译脚本 (base64 编码写入，规避 stdin 管道缓冲的边界问题)
        #    编译前清掉远程旧 wheel: wheel 输出目录固定，增量构建遇到同名
        #    旧文件会跳过重新生成，旧分支的包会被误当成本次产物回传
        #    同时清掉仓库下的 build/ 与 third_party/: 不同 SOC / 分支的
        #    中间产物互相污染会导致编译偶发失败，third_party 由镜像提供的
        #    种子重建，一律从干净状态开始
        # 宿主不再需要 CANN / conda: 容器自带 toolkit 与 python，只启动 docker
        build_script = x86_build_script(self.repo_dir, out_dir, soc, image,
                                        container, container_env)
        encoded = base64.b64encode(build_script.encode("utf-8")).decode("ascii")
        remote.check(
            # 镜像缺失要在启动编译前就失败，并说清补法: 950 镜像按
            # ci/Dockerfile.ascend950 在 x86 编译机上构建
            f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1 || {{ "
            f"echo '[pkg] 编译镜像缺失: {image}；在该编译机上执行 "
            f"docker build -t {image} -f ci/Dockerfile.ascend950 . "
            "(仅 x86_64；或用 --x86-docker 指定其他已有镜像)' >&2; exit 3; }\n"
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
        """创建全新的 Release (每次构建一个；tag 冲突时才回退复用)。"""
        payload = {
            "tag_name": tag,
            "name": title or tag,
            "body": body or "",
            "target_commitish": target_sha,
            "prerelease": self.prerelease,
            "draft": False,
            "make_latest": "false",  # 不带 Latest 标记
        }
        try:
            return self._request(
                "POST", f"{self.api_base}/repos/{self.repo}/releases",
                json_body=payload)
        except GitHubApiError as exc:
            # tag 已有 Release (并发触发 / --gh-tag 显式指定已有 tag) 时
            # 回退复用该 Release
            if exc.code == 422:
                existing = self.find_release(tag)
                if existing is not None:
                    log(f"tag 已有 Release，回退复用: {tag}")
                    return existing
            raise

    def find_latest_release(self) -> dict | None:
        try:
            return self._request("GET", f"{self.api_base}/repos/{self.repo}/releases/latest")
        except GitHubApiError as exc:
            if exc.code == 404:
                return None
            raise

    def mark_latest(self, release: dict) -> None:
        self._request("PATCH", f"{self.api_base}/repos/{self.repo}/releases/{release['id']}",
                      json_body={"make_latest": "true"})

    def update_body(self, release: dict, body: str, title: str) -> None:
        """更新 Release 标题与描述 (全部组合结束后写入最终状态汇总)。

        复用已存在的 Release 时，标题/描述都要刷新为本次构建的值——
        否则标题停留在首次创建时的日期与 commit，看起来像"没有新构建"。
        保留当前 Latest 归属；修改说明不能取消已有标记或把旧版本升为 Latest。
        """
        latest = self.find_latest_release()
        self._request(
            "PATCH",
            f"{self.api_base}/repos/{self.repo}/releases/{release['id']}",
            json_body={"name": title, "body": body,
                       "make_latest": "true" if latest and latest["id"] == release["id"] else "false"})

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


def write_fail_log(failed_records: list, meta: dict, head_sha: str,
                   out_dir: Path) -> Path | None:
    """存在编译失败时把失败汇总写入固定目录 fail_log_<时间>.log。

    内容为每个失败组合的错误信息、构建日志路径及其末尾输出，不翻
    <out-dir>/logs 即可定位失败原因。写入失败只打警告，不影响主流程，
    返回写出的日志路径 (失败返回 None)。
    """
    fail_dir = Path(DEFAULT_FAIL_LOG_DIR)
    log_path = fail_dir / f"fail_log_{meta['datetime']}.log"
    lines = [
        "flash-linear-attention-npu 编译失败汇总 (存在失败，未推送 Release)",
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"分支: {meta['ref']}  commit: {head_sha[:12]}",
        f"产物目录: {out_dir}",
        f"失败组合: {len(failed_records)} 个",
        "",
    ]
    for record in failed_records:
        build_log = out_dir / "logs" / f"{record['soc']}-{record['arch']}.log"
        lines.append(f"==== {record['soc']}/{record['arch']} 失败 "
                     f"(耗时 {record['seconds']:.0f}s) ====")
        lines.append(f"错误: {record['error']}")
        lines.append(f"构建日志: {build_log}")
        if build_log.is_file():
            try:
                tail = build_log.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-80:]
            except OSError as exc:
                tail = [f"<构建日志读取失败: {exc}>"]
            lines.append("---- 构建日志末尾 ----")
            lines.extend(tail)
        lines.append("")
    try:
        fail_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        log(f"[警告] 失败汇总日志写入失败 ({log_path}): {exc}")
        return None
    return log_path


class ReleaseUploader:
    """Release 批量上传器: 所有组合执行完 (含失败 / 中断收尾) 后统一上传。

    finalize() 在收尾阶段先判定推送门禁: 存在编译失败时不推送 Release，
    改为写失败汇总日志 (write_fail_log) 供排查；无编译失败时一次性上传
    全部成功组合的 wheel，并把 Release 描述更新为最终状态表，中断时
    未执行的组合注明 "未执行"。
    """

    def __init__(self, gh_config: dict, args: argparse.Namespace, meta: dict,
                 head_sha: str, out_dir: Path):
        self.gh = GitHubRelease(gh_config["repo"], gh_config["token"],
                                args.gh_api, args.gh_prerelease)
        # 每次构建创建一个全新的 Release (不复用旧的)。tag 每次要唯一:
        # <分支名>[-pr<PR 编号>]-<日期时间>，如 v26.9.1-20260928-020005；
        # --gh-tag 显式指定时用指定值 (tag 已有 Release 时回退复用)。
        # title 为 【 分支名 】<YYYYMMDD> commit: <sha12>
        # 分支与 tag 重名时，git rev-parse --abbrev-ref HEAD 为消歧会带
        # heads/ 前缀 (如 heads/v26.9.1，sanitize 后成 heads-v26.9.1)；
        # Release 的 tag/title/正文只按分支名显示，去掉该前缀
        self.pr = args.pr
        branch_name = meta["ref"]
        if branch_name.startswith("heads-"):
            branch_name = branch_name[len("heads-"):]
        self.branch_name = branch_name
        base_name = (f"{branch_name}-pr{meta['pr_id']}" if args.pr
                     else branch_name)
        self.tag = args.gh_tag or f"{base_name}-{meta['datetime']}"
        # title 文案: 【 分支名 】<YYYYMMDD> commit: <sha 前 12 位>；
        # main 分支显示为"主线"，--pr 出包时分支名后带 -pr<PR 编号>
        display = "主线" if branch_name == "main" else branch_name
        if args.pr:
            display = f"{display}-pr{meta['pr_id']}"
        self.title = (args.gh_title or
                      f"【 {display} 】{meta['datetime'][:8]} "
                      f"commit: {head_sha[:12]}")
        self.meta = meta
        self.head_sha = head_sha
        self.out_dir = out_dir

    def finalize(self, planned: list, results: list,
                 interrupted_reason: str = "") -> dict | None:
        """收尾: 判定推送门禁后一次性上传所有成功组合的 wheel。

        planned 为全部计划组合 (soc, arch) 列表；results 为已执行的组合
        结果。存在编译失败时不推送 Release (改为写失败汇总日志)；没有
        任何组合成功时不创建 Release。两种情况都不上传，返回带 error
        说明的结果或 None。
        """
        snapshot = list(results)
        # 推送门禁: 只要存在编译失败，本次构建整体不算通过，不推送
        # Release，改为在固定目录落盘失败汇总日志供排查
        failed_records = [r for r in snapshot if r["status"] == "failed"]
        if failed_records:
            fail_path = write_fail_log(failed_records, self.meta,
                                       self.head_sha, self.out_dir)
            error = (f"存在 {len(failed_records)} 个组合编译失败，"
                     "跳过 Release 推送")
            if fail_path is not None:
                error += f" (失败日志: {fail_path})"
            log(error)
            return {"tag": self.tag, "url": "", "assets": [], "error": error}
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
                *([f"- 含 PR: `#{self.pr}` (已合入 `{self.branch_name}`)"]
                  if self.pr else []),
                f"- 分支: `{self.branch_name}`  commit: `{self.head_sha[:12]}`",
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

        # 最终描述统一 PATCH 一次 (补上传失败注记，并覆盖复用场景旧描述;
        # 标题也一并刷新为本次构建的日期与 commit)
        error_text = "; ".join(f"{where} {err}" for where, err in errors)
        try:
            self.gh.update_body(release, compose_body(), self.title)
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
    parser.add_argument("--x86-docker", default=DEFAULT_X86_DOCKER_IMAGE,
                        metavar="IMAGE",
                        help="x86 侧全部 SOC 共用的编译容器镜像 (远程宿主只"
                             "同步代码并启动容器；默认 "
                             f"{DEFAULT_X86_DOCKER_IMAGE}，仅 x86_64)")
    parser.add_argument("--local-x86", action="store_true",
                        help="x86 在当前 x86_64 宿主的 Docker 内编译，无需 SSH")
    parser.add_argument("--head-sha", default="",
                        help="PR 代码的完整 commit sha (CI 注入用，默认取 HEAD)")
    parser.add_argument("--commit-sha", default="",
                        help="把 --branch 锁定到此完整 commit SHA；daily 双机计划使用")
    parser.add_argument("--out-dir", default="",
                        help="产物输出目录 (默认 <repo>/ci/pkg/out)")
    parser.add_argument("--list", action="store_true",
                        help="列出支持的 SOC / 架构组合并退出")
    parser.add_argument("--branch", default="",
                        help="指定分支: 同步该分支远端最新代码后编译 "
                             "(会重置本地分支并清理未跟踪文件，保留 third_party)")
    parser.add_argument("--standalone", action="store_true",
                        help="独立编包模式: 在固定目录 clone/更新目标仓源码，"
                             "但不发布 GitHub Release")
    parser.add_argument("--build-type", choices=("daily", "release"),
                        default="daily",
                        help="包类型；默认 daily。release 会关闭本地版本号并"
                             "开启离线 third-party bundle")
    parser.add_argument("--release-version", default="",
                        help="正式包版本，例如 26.9.1；build-type=release 时必填，"
                             "并与源码 fla/__init__.py 严格核对")
    parser.add_argument("--pr", type=int, default=None,
                        help="带指定 PR 出包 (须与 --branch 同用): 校验 PR "
                             "的 base 分支必须为 --branch 指定的分支，"
                             "通过后把 PR 合入本地代码再执行出包流程; "
                             "--gh-release 时每次构建创建新 Release "
                             "(tag 含 -pr<PR 编号>)，不覆盖其他构建")
    parser.add_argument("--gh-release", action="store_true",
                        help="仅 build-type=release: 在固定目录 clone 目标仓 "
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

    enforce_build_job_limit()
    log(f"编译并行度固定为 -j{BUILD_JOB_LIMIT}，Docker CPU 上限 "
        f"{DOCKER_CPU_LIMIT} 核")

    socs = parse_list(args.soc, SUPPORTED_SOCS, "SOC")
    arches = parse_list(args.arch, SUPPORTED_ARCHES, "架构")
    if not socs or not arches:
        raise SystemExit("--soc/--arch 解析结果为空")
    if args.local_x86 and platform.machine() != "x86_64":
        raise SystemExit("--local-x86 必须在 x86_64 编译机上执行")
    if args.pr is not None and not args.branch:
        raise SystemExit("--pr 需要同时通过 --branch 指定分支 "
                         "(用于校验 PR 的 base 分支)")
    if args.commit_sha:
        args.commit_sha = args.commit_sha.lower()
        if not args.branch:
            raise SystemExit("--commit-sha 必须与 --branch 一起使用")
        if not re.fullmatch(r"[0-9a-f]{40}", args.commit_sha):
            raise SystemExit("--commit-sha 必须是 40 位十六进制 SHA")
    if args.build_type == "release":
        if not args.branch:
            raise SystemExit("--build-type release 必须通过 --branch 指定发布分支")
        if not args.release_version:
            raise SystemExit("--build-type release 必须指定 --release-version")
    elif args.release_version:
        raise SystemExit("--release-version 只能与 --build-type release 一起使用")
    if args.gh_release and args.build_type != "release":
        raise SystemExit("daily 禁止使用 --gh-release；每日包只能发布到 OBS")

    # 代码来源:
    #   --gh-release (cron / 独立调用): 固定目录 clone 目标仓
    #       flashserve/flash-linear-attention-npu 最新代码
    #   CI 触发: 当前仓 (--repo-root，PR 已检出，即 flashserve 仓)
    gh_config: dict | None = None
    gh_repo = args.gh_repo or env_or("FLA_PKG_TARGET_GH_REPO", TARGET_GH_REPO)
    standalone = args.standalone or args.gh_release
    if standalone:
        cron_repo = Path(env_or("FLA_PKG_CRON_REPO_DIR",
                                DEFAULT_CRON_REPO_DIR))
        branch = clone_or_update_repo(cron_repo, args.branch, args.commit_sha)
        token = ""
        if args.gh_release:
            token = (args.gh_token or os.getenv("FLA_PKG_GH_TOKEN", "").strip()
                     or os.getenv("GITHUB_TOKEN", "").strip())
            if not token:
                raise SystemExit(
                    "--gh-release 需要 token: --gh-token 或环境变量 "
                    "FLA_PKG_GH_TOKEN / GITHUB_TOKEN")
            gh_config = {"repo": gh_repo, "token": token}
        if args.pr is not None:
            # --pr: 校验 base 分支后把 PR 合入刚同步好的分支最新代码
            merge_pull_request(cron_repo, args.pr, branch, args.gh_api,
                               gh_repo, token)
        repo_root = cron_repo
        head_sha = git_output(repo_root, "rev-parse", "HEAD")
        if gh_config is not None:
            log(f"GitHub Release 上传已使能: {gh_config['repo']} ({args.gh_api})")
        else:
            log("独立编包模式: 仅同步源码和编包，不发布 GitHub Release")
        log(f"代码来源: {env_or('FLA_PKG_TARGET_REPO_URL', TARGET_REPO_URL)}"
            f" @ {branch} (clone 于 {repo_root})")
    else:
        repo_root = Path(args.repo_root).resolve()
        if not repo_root.is_dir():
            raise SystemExit(f"代码目录不存在: {repo_root}")
        branch = args.branch
        if branch:
            update_repo(repo_root, branch, args.commit_sha)
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

    os.environ["FLA_PKG_BUILD_TYPE"] = args.build_type
    os.environ.pop("FLA_NPU_PACKAGE_VERSION", None)
    release_source_version = ""
    if args.build_type == "release":
        expected_version = args.release_version.removeprefix("v")
        release_source_version = prepare_release_version(repo_root, expected_version)
        actual_version = read_public_version(repo_root)
        if actual_version != expected_version:
            raise SystemExit(
                f"正式包版本不匹配: --release-version={expected_version}，"
                f"源码 fla/__init__.py={actual_version}")
        os.environ["FLA_NPU_DISABLE_LOCAL_VERSION"] = "TRUE"
        os.environ["FLA_NPU_BUILD_OFFLINE_BUNDLE"] = "TRUE"
        os.environ.pop("FLA_NPU_LOCAL_VERSION", None)
        log(f"正式包模式: 版本 {actual_version}，关闭本地版本号并启用离线 bundle")
    else:
        # 防止调用方环境残留 release 开关，把 daily 错编成无 local version 的正式包。
        os.environ.pop("FLA_NPU_DISABLE_LOCAL_VERSION", None)
        os.environ.pop("FLA_NPU_BUILD_OFFLINE_BUNDLE", None)
        log("每日包模式: 保留分支/commit 本地版本号，不启用正式包开关")

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
    elif os.getenv("FLA_PKG_REF_NAME", "").strip():
        # CI 模式 (detached HEAD 检出，默认为 PR 合并结果 commit) 解析
        # 不到分支名，用工作流注入的 FLA_PKG_REF_NAME 兜底 (PR base
        # 分支名或 branch= 指定分支)，保证从 main 等分支出的包与 cron
        # 发布版本号一致
        ci_ref = os.getenv("FLA_PKG_REF_NAME").strip()
        if ci_ref != "HEAD":
            os.environ["FLA_NPU_BRANCH_NAME"] = ci_ref
            os.environ["FLA_NPU_COMMIT_ID"] = head_sha
            log(f"CI 分支构建: 注入 FLA_NPU_BRANCH_NAME={ci_ref} / "
                f"FLA_NPU_COMMIT_ID={head_sha[:7]}，保证 arm/x86 wheel "
                "版本号一致")

    if args.build_type == "daily":
        from daily_version import version_for, today, prepare_version_hook
        daily_date = env_or("FLA_PKG_DAILY_DATE", today())
        os.environ["FLA_PKG_DAILY_DATE"] = daily_date
        package_version = version_for(build_branch, daily_date, head_sha)
        os.environ.pop("FLA_NPU_LOCAL_VERSION", None)
        os.environ["FLA_NPU_PACKAGE_VERSION"] = package_version
        prepare_version_hook(repo_root)
        log(f"每日包版本: {package_version}; date={daily_date}; commit={head_sha}")

    if args.pr is not None:
        # --pr 出包: 产物名 / 远程工作区 / manifest 都标记 PR 编号，
        # 不影响无 --pr 的原有流程 (CI 模式仍读自身的 FLA_PKG_PR_ID)
        os.environ["FLA_PKG_PR_ID"] = str(args.pr)
    meta = resolve_metadata(repo_root)
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif standalone:
        # 独立模式产物目录放仓外，避免 clone 仓更新时 git clean 清掉产物
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
            # standalone: x86 固定编译目录；CI: 每次触发独立工作区
            remote_base = (env_or("FLA_PKG_REMOTE_CRON_BASE",
                                  DEFAULT_REMOTE_CRON_BASE)
                           if standalone else None)
            workspace = RemoteWorkspace(remote, repo_root, workkey,
                                        base=remote_base,
                                        x86_image=args.x86_docker)
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
            if arch == "x86" and not args.local_x86 and workspace is None:
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
                    if arch == "arm" or args.local_x86:
                        wheels = build_local(repo_root, soc, wheel_dir, log_file,
                                             arch=arch, x86_image=args.x86_docker)
                    else:
                        wheels = workspace.build(soc, wheel_dir, log_file,
                                                stop=stop)
                    if args.build_type == "daily":
                        from daily_version import verify_wheel_version
                        for wheel in wheels:
                            verify_wheel_version(wheel, os.environ["FLA_NPU_PACKAGE_VERSION"])
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
                try:
                    target.hardlink_to(wheel)
                except OSError as exc:
                    if exc.errno not in (errno.EPERM, errno.EXDEV, errno.EACCES):
                        raise
                    shutil.copy2(wheel, target)
                record["wheels"].append(wheel.name)
            record["status"] = "success"
            log(f"{prefix} === 编译成功 "
                f"({time.time() - started:.0f}s, {len(wheels)} 个 wheel) ===")
            try:
                record["incremental_obs"] = publish_incremental_obs(
                    soc, arch, wheel_dir, out_dir)
            except Exception as exc:  # noqa: BLE001 - final publisher retries
                record["incremental_obs"] = {
                    "status": "failed", "error": str(exc)}
                log(f"{prefix} [警告] {exc}；继续编译，整批发布时重试")
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
        if arch == "x86" and not args.local_x86:
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
        "build_type": args.build_type,
        "branch": build_branch,
        "release_version": args.release_version.removeprefix("v"),
        "release_source_version": release_source_version,
        "release_version_adapted": bool(release_source_version and
                                        release_source_version != args.release_version.removeprefix("v")),
        "head_sha": head_sha,
        "daily_date": os.getenv("FLA_PKG_DAILY_DATE", "") if args.build_type == "daily" else "",
        "package_version": os.getenv("FLA_NPU_PACKAGE_VERSION", ""),
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
