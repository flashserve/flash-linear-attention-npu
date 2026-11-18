#!/usr/bin/env bash
# The publishing project consumes CI images and never overwrites shared tags.
set -Eeuo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
case "${1:-}" in
  ''|--check) ;;
  -h|--help)
    echo '用法: ./build_docker.sh [--check]；检查 CI 共用镜像，不创建独立发布镜像。'
    exit 0 ;;
  *) echo "未知参数: $1" >&2; exit 2 ;;
esac
case "$(uname -m)" in
  aarch64) arch=arm; platform=linux/arm64 ;;
  x86_64) arch=x86; platform=linux/amd64 ;;
  *) echo '不支持的宿主架构' >&2; exit 2 ;;
esac
image=$(python3 "$BASE/docker_config.py" "$arch")
docker info >/dev/null
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "缺少 CI 共用镜像 $image；请按源码仓库 CI 流程构建或加载。发布入口不会自行改写 CI 镜像。" >&2
  exit 2
fi
actual=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")
[[ "$actual" == "$platform" ]] || { echo "镜像架构不匹配: $actual != $platform" >&2; exit 2; }
docker image inspect --format 'CI 共用镜像检查通过: {{.Id}} ({{.Os}}/{{.Architecture}})' "$image"
