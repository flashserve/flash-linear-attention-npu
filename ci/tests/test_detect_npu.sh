#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$tmp_dir/bin"
cat >"$tmp_dir/bin/npu-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "info" ]]; then
    echo "unexpected npu-smi arguments: $*" >&2
    exit 2
fi
cat "$NPU_SMI_FIXTURE"
EOF
chmod +x "$tmp_dir/bin/npu-smi"

cat >"$tmp_dir/a2.txt" <<'EOF'
+------------------------------------------------------------------------------------------------+
| NPU   Name        | Health |
| 2     910B3       | OK     |
+------------------------------------------------------------------------------------------------+
No running processes found in NPU 2
EOF

cat >"$tmp_dir/a5.txt" <<'EOF'
+------------------------------------------------------------------------------------------------+
| NPU | Name        | Health |
| 0   | Ascend950PR | OK     | 0 |
+------------------------------------------------------------------------------------------------+
No running processes found in NPU 0
EOF

run_detect() {
    local fixture="$1"
    shift
    PATH="$tmp_dir/bin:$PATH" NPU_SMI_FIXTURE="$fixture" bash "$repo_dir/ci/detect_npu.sh" "$@"
}

assert_equal() {
    local expected="$1"
    local actual="$2"
    local description="$3"
    if [[ "$actual" != "$expected" ]]; then
        echo "[FAIL] $description" >&2
        echo "expected: $expected" >&2
        echo "actual:   $actual" >&2
        exit 1
    fi
}

assert_contains_line() {
    local expected="$1"
    local actual="$2"
    local description="$3"
    if ! grep -Fqx -- "$expected" <<<"$actual"; then
        echo "[FAIL] $description" >&2
        echo "missing line: $expected" >&2
        echo "actual output:" >&2
        echo "$actual" >&2
        exit 1
    fi
}

a2_env="$(run_detect "$tmp_dir/a2.txt" --env)"
assert_contains_line "NPU_SELECTED_DEVICE=2" "$a2_env" "A2 device ID"
assert_contains_line "NPU_SELECTED_NAME=910B3" "$a2_env" "A2 device name"
assert_contains_line "NPU_SELECTED_HEALTH=OK" "$a2_env" "A2 health"
assert_contains_line "NPU_SELECTED_FREE=1" "$a2_env" "A2 idle state"
assert_contains_line "NPU_SOC=ascend910b" "$a2_env" "A2 SOC"
assert_equal "2" "$(run_detect "$tmp_dir/a2.txt" --candidates)" "A2 candidates"

a5_env="$(run_detect "$tmp_dir/a5.txt" --env)"
assert_contains_line "NPU_SELECTED_DEVICE=0" "$a5_env" "A5 device ID"
assert_contains_line "NPU_SELECTED_NAME=Ascend950PR" "$a5_env" "A5 device name"
assert_contains_line "NPU_SELECTED_HEALTH=OK" "$a5_env" "A5 health"
assert_contains_line "NPU_SELECTED_FREE=1" "$a5_env" "A5 idle state"
assert_contains_line "NPU_SOC=ascend950" "$a5_env" "A5 SOC"
assert_equal "ascend950" "$(run_detect "$tmp_dir/a5.txt" --soc)" "A5 SOC mode"
assert_equal "0" "$(run_detect "$tmp_dir/a5.txt" --candidates)" "A5 candidates"

echo "detect_npu A2/A5 parser tests passed"
