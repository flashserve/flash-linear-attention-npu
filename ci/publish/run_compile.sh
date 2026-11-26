#!/usr/bin/env bash
# Build and publish flash-linear-attention-npu daily/release wheels.
set -Eeuo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1
# Keep the deployment's explicit proxy settings; do not assume a local proxy.
if [[ "${FLA_PKG_PROXY_DISABLED:-0}" == 1 ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    unset all_proxy ALL_PROXY GIT_HTTP_PROXY GIT_HTTPS_PROXY
fi

BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DRIVER="$BASE/make_packages.py"
PUBLISHER="$BASE/publish_from_manifest.py"
PYTHON="${FLA_PKG_PYTHON:-$BASE/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=/usr/bin/python3
if [[ -d "$BASE/.deps" ]]; then
    export PYTHONPATH="$BASE/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
BUILD_JOBS=64
export FLA_NPU_BUILD_ARGS="${FLA_NPU_BUILD_ARGS:+$FLA_NPU_BUILD_ARGS }-j$BUILD_JOBS"

log() { printf '[fla-publish][%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }
need_env() { [[ -n "${!1:-}" ]] || die "缺少环境变量 $1"; }

usage() {
    cat <<'EOF'
用法：
  ./run_compile.sh [--build-type daily] [--branch main] [选项]
  ./run_compile.sh --build-type release --branch v26.9.1 \
      --release-version 26.9.1 [选项]

默认行为：daily；候选分支 main、v26.10.0、v26.11.0；跳过已有正式 Release 的分支。
默认在当前机器编译全 NPU 平台（清单见 npu_platforms.py），只发布 OBS nightly。

选项：
  --build-type daily|release  包类型，默认 daily
  --branch BRANCH            构建分支；daily 可重复指定，release 只能一个
  --branch-sha BRANCH=SHA    把分支锁定到 40 位 SHA；可重复，daily plan 使用
  --release-version VERSION  release 必填，例如 26.9.1
  --soc LIST                 默认平台清单中的全部 SOC
  --arch LIST                默认当前机器架构 arm 或 x86
  --x86-docker IMAGE         x86 编译镜像
  --build-only               只编包，不发布（调试用）
  --plan                     只校验参数/环境并打印计划
  -h, --help                 显示帮助

发布凭证只从进程环境变量读取：
  daily/release: OBS_ACCESS_KEY, OBS_SECRET_KEY
  release:       GITHUB_TOKEN

x86_64 机器自动本地编包；仅从 ARM 机远程编 x86 时需要：
  FLA_PKG_REMOTE_HOST, FLA_PKG_REMOTE_USER, FLA_PKG_SSH_KEY,
  FLA_PKG_REMOTE_WORKROOT, FLA_PKG_REMOTE_CRON_BASE
EOF
}

build_type=daily
release_version=
soc_list=$("$PYTHON" "$BASE/npu_platforms.py" --soc-list)
all_socs=$soc_list
host_arch=$(uname -m)
case "$host_arch" in
    aarch64) arch_list=arm ;;
    x86_64) arch_list=x86 ;;
    *) die "不支持的宿主架构: $host_arch" ;;
esac
x86_docker="${FLA_PKG_X86_DOCKER_IMAGE:-$("$PYTHON" "$BASE/docker_config.py" x86)}"
export FLA_PKG_DOCKER_IMAGE="$("$PYTHON" "$BASE/docker_config.py" arm)"
build_only=0
plan_only=0
branches=()
branch_sha_specs=()
declare -A branch_sha_by_name=()

while (( $# )); do
    case "$1" in
        --build-type) [[ $# -ge 2 ]] || die "$1 缺少参数"; build_type=$2; shift 2 ;;
        --branch) [[ $# -ge 2 ]] || die "$1 缺少参数"; branches+=("$2"); shift 2 ;;
        --branch-sha) [[ $# -ge 2 ]] || die "$1 缺少参数"; branch_sha_specs+=("$2"); shift 2 ;;
        --release-version) [[ $# -ge 2 ]] || die "$1 缺少参数"; release_version=$2; shift 2 ;;
        --soc) [[ $# -ge 2 ]] || die "$1 缺少参数"; soc_list=$2; shift 2 ;;
        --arch) [[ $# -ge 2 ]] || die "$1 缺少参数"; arch_list=$2; shift 2 ;;
        --x86-docker) [[ $# -ge 2 ]] || die "$1 缺少参数"; x86_docker=$2; shift 2 ;;
        --build-only) build_only=1; shift ;;
        --plan) plan_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知参数: $1" ;;
    esac
done
export FLA_PKG_X86_DOCKER_IMAGE="$x86_docker"

case "$build_type" in daily|release) ;; *) die "--build-type 只能是 daily 或 release" ;; esac
[[ -f "$DRIVER" ]] || die "编包脚本不存在: $DRIVER"
[[ -f "$PUBLISHER" ]] || die "发布脚本不存在: $PUBLISHER"

if ((${#branches[@]} == 0)); then
    if [[ "$build_type" == daily ]]; then
        read -r -a branches <<< "${FLA_PKG_DAILY_BRANCHES:-main v26.10.0 v26.11.0}"
    else
        die "release 必须指定 --branch"
    fi
fi
if [[ "$build_type" == release ]]; then
    ((${#branches[@]} == 1)) || die "release 一次只能指定一个分支"
    [[ -n "$release_version" ]] || die "release 必须指定 --release-version"
else
    [[ -z "$release_version" ]] || die "daily 不能指定 --release-version"
fi
for branch in "${branches[@]}"; do
    [[ "$branch" == main || "$branch" =~ ^v[0-9]+([.][0-9A-Za-z_-]+)+$ ]] || \
        die "不支持的分支格式: $branch"
done

for spec in "${branch_sha_specs[@]}"; do
    [[ "$spec" == *=* ]] || die "--branch-sha 格式必须是 BRANCH=SHA: $spec"
    pin_branch=${spec%%=*}
    pin_sha=${spec#*=}
    pin_sha=${pin_sha,,}
    [[ "$pin_branch" == main || "$pin_branch" =~ ^v[0-9]+([.][0-9A-Za-z_-]+)+$ ]] || \
        die "--branch-sha 分支格式非法: $pin_branch"
    [[ "$pin_sha" =~ ^[0-9a-f]{40}$ ]] || die "--branch-sha 必须使用 40 位 SHA: $spec"
    [[ -z "${branch_sha_by_name[$pin_branch]:-}" ]] || \
        die "--branch-sha 重复指定分支: $pin_branch"
    branch_sha_by_name[$pin_branch]=$pin_sha
done
if ((${#branch_sha_specs[@]})); then
    for branch in "${branches[@]}"; do
        [[ -n "${branch_sha_by_name[$branch]:-}" ]] || \
            die "锁定模式下每个分支都必须有 SHA: 缺少 $branch"
    done
    for pin_branch in "${!branch_sha_by_name[@]}"; do
        found=0
        for branch in "${branches[@]}"; do
            [[ "$branch" == "$pin_branch" ]] && found=1
        done
        (( found )) || die "--branch-sha 指定了未选中的分支: $pin_branch"
    done
fi

# Query only published formal releases, before credential checks or any builds.
# --plan performs this read-only lookup too, showing the actual filtered plan.
if [[ "$build_type" == daily ]] && ((${#branch_sha_specs[@]} == 0)); then
    daily_selection=$("$PYTHON" "$BASE/daily_branches.py" "${branches[@]}") || \
        die "无法确认分支正式发布状态，本次未编包"
    branches=()
    if [[ -n "$daily_selection" ]]; then
        mapfile -t branches <<< "$daily_selection"
    fi
    if ((${#branches[@]} == 0)); then
        log "所有候选分支已正式发布，本次 daily 无需编包"
        exit 0
    fi

fi
if [[ "$build_type" == daily ]]; then
    if ((${#branch_sha_specs[@]})); then
        log "使用 daily plan 锁定的分支/SHA，不再按当前 HEAD 漂移"
    fi
    # GITHUB_TOKEN 仅用于计划/正式 Release 状态查询。daily 后续编译和
    # OBS 发布不需要 GitHub 写权限，也绝不保留该凭证。
    unset GITHUB_TOKEN
    unset FLA_PKG_GH_TOKEN
fi

case "$arch_list" in arm|x86|arm,x86|x86,arm) ;; *) die "--arch 只能选 arm、x86 或 arm,x86" ;; esac
if [[ ",$arch_list," == *,arm,* && "$host_arch" != aarch64 ]]; then
    die "ARM 包需在 aarch64 编译机执行"
fi

publish_enabled=1
(( build_only )) && publish_enabled=0
if (( publish_enabled )); then
    need_env OBS_ACCESS_KEY
    need_env OBS_SECRET_KEY
    if [[ "$build_type" == release ]]; then need_env GITHUB_TOKEN; fi
    [[ "$soc_list" == "$all_socs" || "$soc_list" == all ]] || \
        die "发布要求平台清单中的全部 SOC；自定义子集只能与 --build-only 一起使用"
fi

# Daily 每个平台编译成功后立即增量上传 OBS；正式包仍保持整批发布，
# --build-only 也绝不启用上传钩子。
if (( publish_enabled )) && [[ "$build_type" == daily ]]; then
    export FLA_PKG_INCREMENTAL_OBS=1
else
    unset FLA_PKG_INCREMENTAL_OBS
fi

if [[ ",$arch_list," == *,x86,* && "$host_arch" != x86_64 ]]; then
    need_env FLA_PKG_REMOTE_HOST
    need_env FLA_PKG_REMOTE_USER
    need_env FLA_PKG_SSH_KEY
    need_env FLA_PKG_REMOTE_WORKROOT
    need_env FLA_PKG_REMOTE_CRON_BASE
    [[ -r "$FLA_PKG_SSH_KEY" ]] || die "SSH 私钥不可读: $FLA_PKG_SSH_KEY"
fi

if [[ "$build_type" == daily ]]; then
    publish_target=obs-nightly
    export FLA_PKG_DAILY_DATE="${FLA_PKG_DAILY_DATE:-$(date '+%Y%m%d')}"
    log "PLAN daily_date=$FLA_PKG_DAILY_DATE destinations=nightly,nightly/project/latest"
    github_release=disabled
else
    publish_target=obs-release-and-github-release
    github_release=enabled
fi
log "PLAN type=$build_type branches=${branches[*]} soc=$soc_list arch=$arch_list jobs=$BUILD_JOBS"
for branch in "${branches[@]}"; do
    if [[ -n "${branch_sha_by_name[$branch]:-}" ]]; then
        log "PLAN pin $branch=${branch_sha_by_name[$branch]}"
    fi
done
log "PLAN publish=$publish_enabled target=$publish_target github_release=$github_release x86_image=$x86_docker"
if (( plan_only )); then
    log "参数和环境检查通过；--plan 不执行编包或上传"
    exit 0
fi

# Fail before cloning source if the dedicated local Docker environment is unavailable.
if [[ ",$arch_list," == *,arm,* || "$host_arch" == x86_64 ]]; then
    bash "$BASE/build_docker.sh" --check
fi

exec 9>"$BASE/.run_compile.lock"
if ! flock -n 9; then
    log "已有编包任务持有锁，退出" >&2
    exit 75
fi

mkdir -p "$BASE/repos" "$BASE/out" "$BASE/logs"
overall=0
run_id=$(date '+%Y%m%d-%H%M%S')

for branch in "${branches[@]}"; do
    safe_branch=${branch//[^A-Za-z0-9._-]/-}
    export FLA_PKG_CRON_REPO_DIR="$BASE/repos/$safe_branch"
    out_dir="$BASE/out/$build_type/$safe_branch"
    branch_log="$BASE/logs/$run_id-$build_type-$safe_branch.log"

    args=(--standalone --build-type "$build_type" --branch "$branch"
          --soc "$soc_list" --arch "$arch_list" --x86-docker "$x86_docker"
          --out-dir "$out_dir")
    if [[ -n "${branch_sha_by_name[$branch]:-}" ]]; then
        args+=(--commit-sha "${branch_sha_by_name[$branch]}")
    fi
    if [[ "$build_type" == release ]]; then
        args+=(--release-version "$release_version")
    fi
    if [[ "$host_arch" == x86_64 ]]; then
        args+=(--local-x86)
    fi

    log "BEGIN build branch=$branch log=$branch_log"
    set +e
    "$PYTHON" "$DRIVER" "${args[@]}" 2>&1 | tee "$branch_log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        log "FAILED build branch=$branch exit=$rc" >&2
        overall=1
        continue
    fi
    log "SUCCESS build branch=$branch"

    if (( ! publish_enabled )); then
        log "SKIP publish branch=$branch (--build-only)"
        continue
    fi

    publish_args=(--manifest "$out_dir/manifest.json"
                  --build-type "$build_type" --branch "$branch")
    if [[ "$build_type" == release ]]; then
        publish_args+=(--release-version "$release_version"
                       --gh-tag "$branch-release"
                       --gh-title "flash-linear-attention-npu $release_version")
    else
        log "POLICY daily publish: OBS nightly only; GitHub Release disabled"
    fi
    log "BEGIN publish branch=$branch type=$build_type"
    if "$PYTHON" "$PUBLISHER" "${publish_args[@]}"; then
        log "SUCCESS publish branch=$branch type=$build_type"
    else
        rc=$?
        log "FAILED publish branch=$branch exit=$rc" >&2
        overall=1
    fi
done

log "END type=$build_type exit=$overall"
exit "$overall"
