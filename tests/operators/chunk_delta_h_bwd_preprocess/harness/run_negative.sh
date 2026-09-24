#!/usr/bin/env bash
# 反向用例执行：逐条生成非法输入 → 调用 aclnn → 校验返回码是否符合 op_cases 的期望。
#   bash run_negative.sh [work_dir]
# 需要先按 README 编好同目录下的 run_case（aclnn 直调取数程序），并 source custom 包的 set_env。
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="$HERE/../../../op_cases/chunk_delta_h_bwd_preprocess.json"
WORK="${1:-$HERE/negative_cases}"
mkdir -p "$WORK"

# ACLNN 返回码：ACLNN_ERR_PARAM_INVALID = 161001
code_of() {
  case "$1" in
    ACLNN_ERR_PARAM_INVALID) echo 161001 ;;
    ACLNN_ERR_PARAM_NULLPTR) echo 161000 ;;
    *) echo "$1" ;;
  esac
}

echo "=================== NEGATIVE CASES ==================="
printf "%-34s %-12s %-12s %-10s\n" "case" "expected" "got" "verdict"
bad=0
for cid in $(python3 -c "import json;print(' '.join(c['id'] for c in json.load(open('$SPEC'))['negative_cases']))"); do
  d="$WORK/$cid"
  rm -rf "$d"; mkdir -p "$d"
  python3 "$HERE/make_negative_case.py" --dir "$d" "$cid" > /dev/null
  exp_name=$(python3 -c "import json;print([c['expected_return_code'] for c in json.load(open('$SPEC'))['negative_cases'] if c['id']=='$cid'][0])")
  exp=$(code_of "$exp_name")
  out=$(timeout 300 "$HERE/run_case" $(cat "$d/run_case_args.txt") 2>&1)
  rc=$?
  echo "$out" > "$d/run.log"
  got=$(echo "$out" | grep -oE "(GetWorkspaceSize failed|aclnn execute failed) [0-9]+" | tail -1 | awk '{print $NF}')
  if [ -z "$got" ]; then
    verdict="NOT_REJECTED(rc=$rc)"
    bad=1
  elif [ "$got" = "$exp" ]; then
    verdict="PASS"
  else
    verdict="CODE_MISMATCH"
    bad=1
  fi
  printf "%-34s %-12s %-12s %-10s\n" "$cid" "$exp" "${got:--}" "$verdict"
done
echo "=================== SUMMARY: $([ $bad -eq 0 ] && echo ALL_PASS || echo HAS_FAILURE) ==================="
exit $bad
