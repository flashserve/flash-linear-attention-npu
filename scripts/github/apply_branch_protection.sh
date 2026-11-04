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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checks_file="${script_dir}/../../.github/scripts/npu_ci_checks.json"
payload="$(python3 - "$checks_file" <<'PY'
import json
import sys
from pathlib import Path

checks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(checks["required"], list) or not checks["required"]:
    raise SystemExit("The CI profile must define required checks.")
contexts = [item["context"] for item in checks["required"]] + [checks["contract_test"]]
if any(not isinstance(name, str) or not name.strip() for name in contexts) or len(set(contexts)) != len(contexts):
    raise SystemExit("CI required check names must be nonempty, unique strings.")

print(json.dumps({
    "required_status_checks": {"strict": True, "contexts": contexts},
    "enforce_admins": True,
    "required_pull_request_reviews": {
        "dismissal_restrictions": {},
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": True,
        "require_last_push_approval": True,
        "required_approving_review_count": 2,
        "bypass_pull_request_allowances": {
            "users": ["weinachuan"], "teams": [], "apps": [],
        },
    },
    "restrictions": None,
    "required_conversation_resolution": True,
    "allow_force_pushes": False,
    "allow_deletions": False,
    "block_creations": False,
}, ensure_ascii=False))
PY
)"

curl -fsSL \
    -X PUT \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${api_url}/repos/${repo}/branches/${branch}/protection" \
    -d @- <<<"$payload"

echo "Applied branch protection to ${repo}:${branch}."
