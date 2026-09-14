#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:-flashserve/flash-linear-attention-npu}"
branch="${1:-main}"
api_url="${GITHUB_API_URL:-https://api.github.com}"
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [[ -z "$token" ]]; then
    echo "GITHUB_TOKEN or GH_TOKEN with repository administration permission is required." >&2
    exit 2
fi

curl -fsSL \
    -X PUT \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${api_url}/repos/${repo}/branches/${branch}/protection" \
    -d @- <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "NPU CI / A2+A5 / 01 环境、wheel 与运行时契约",
      "NPU CI / A2+A5 / 02 全量 OPP 构建",
      "NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局",
      "NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配",
      "NPU CI / A2+A5 / 05 GDR Example/ST",
      "NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装",
      "NPU CI / A2+A5 / 07 报告与 commit 校验",
      "CI 契约测试"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismissal_restrictions": {},
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "require_last_push_approval": true,
    "required_approving_review_count": 2,
    "bypass_pull_request_allowances": {
      "users": [
        "weinachuan"
      ],
      "teams": [],
      "apps": []
    }
  },
  "restrictions": null,
  "required_conversation_resolution": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false
}
JSON

echo "Applied branch protection to ${repo}:${branch}."
