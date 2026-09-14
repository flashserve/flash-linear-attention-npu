#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

test_parent="$repo_dir/.ci-tmp"
test_parent_preexisting=false
if [[ -d "$test_parent" ]]; then
    test_parent_preexisting=true
fi
mkdir -p "$test_parent"
test_root="$(mktemp -d "$test_parent/stage-pipeline-test.XXXXXX")"

cleanup() {
    local resolved_parent
    local resolved_root
    resolved_parent="$(realpath "$test_parent")"
    resolved_root="$(realpath "$test_root")"
    case "$resolved_root" in
        "$resolved_parent"/stage-pipeline-test.*)
            rm -rf -- "$resolved_root"
            if [[ "$test_parent_preexisting" != "true" ]]; then
                rmdir "$resolved_parent" 2>/dev/null || true
            fi
            ;;
        *)
            echo "Refusing to remove unexpected stage pipeline test path: $resolved_root" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT

CI_TEST_REAL_PYTHON3="${CI_TEST_REAL_PYTHON3:-$(command -v python3)}"
export CI_TEST_REAL_PYTHON3

python3() {
    local entrypoint="${1:-}"
    if [[ "$entrypoint" == "ci/manage_npu_ci_stage_report.py" ]]; then
        if [[ "${CI_TEST_FAIL_MANAGER_INIT:-false}" == "true" && \
              "${*: -1}" == "init" ]]; then
            echo "injected stage report initialization failure" >&2
            return 25
        fi
        command "$CI_TEST_REAL_PYTHON3" "$@"
        return
    fi
    printf 'python3' >>"$CI_TEST_COMMAND_LOG"
    printf ' %q' "$@" >>"$CI_TEST_COMMAND_LOG"
    printf '\n' >>"$CI_TEST_COMMAND_LOG"
    if [[ "${CI_TEST_FAIL_STAGE_ONE:-false}" == "true" && \
          "$entrypoint" == "tests/test_wheel_environment.py" ]]; then
        echo "injected environment contract failure" >&2
        return 23
    fi
    if [[ "$entrypoint" == "-" && -n "${CI_TEST_FAKE_PACKAGE_DIR:-}" ]]; then
        printf '%s\n' "$CI_TEST_FAKE_PACKAGE_DIR"
    elif [[ "$entrypoint" == "ci/run_example_st_cases.py" ]]; then
        mkdir -p "$(dirname "$CI_ACCURACY_REPORT_FILE")"
        printf '{}\n' >"$CI_ACCURACY_REPORT_FILE"
    fi
    return 0
}

bash() {
    printf 'bash' >>"$CI_TEST_COMMAND_LOG"
    printf ' %q' "$@" >>"$CI_TEST_COMMAND_LOG"
    printf '\n' >>"$CI_TEST_COMMAND_LOG"
    if [[ "${1:-}" == "build.sh" && " $* " == *" --pkg "* ]]; then
        if [[ "${CI_TEST_FAIL_STAGE:-}" == "opp-package" ]]; then
            echo "injected OPP package failure" >&2
            return 24
        fi
        mkdir -p build_out
        printf '#!/usr/bin/env bash\nexit 0\n' >build_out/fla_npu_linux-stage-test.run
        chmod +x build_out/fla_npu_linux-stage-test.run
    fi
    return 0
}

export -f python3 bash

assert_report() {
    local report_file="$1"
    local expected_top_status="$2"
    local expected_statuses="$3"
    command "$CI_TEST_REAL_PYTHON3" - "$report_file" "$expected_top_status" "$expected_statuses" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[3].split(",")
stages = (
    "environment-contracts",
    "opp-package",
    "standalone-layout",
    "torch-adapter",
    "gdr-example-st",
    "scoped-overlay",
)
actual = [report["stages"][stage]["status"] for stage in stages]
assert report["complete"] is True, report
assert report["status"] == sys.argv[2], report
assert actual == expected, (actual, expected, report)
PY
}

run_case() {
    local name="$1"
    local requested_stage="$2"
    local fail_stage_one="$3"
    local expected_exit_code="$4"
    local expected_top_status="$5"
    local expected_statuses="$6"
    local run_example_st="${7:-false}"
    local fail_stage="${8:-}"
    local fail_manager_init="${9:-false}"
    local case_dir="$test_root/$name"
    local case_repo="$case_dir/repo"
    local report_file="${case_repo#"$repo_dir"/}/.ci-tmp/stage-report.json"
    local command_log="$case_dir/commands.log"
    local output_log="$case_dir/output.log"
    local fake_package_dir="$case_repo/fake-python/fla_npu"
    local fake_opp_dir="$case_repo/fake-opp"
    local exit_code

    mkdir -p \
        "$case_dir/tmp" \
        "$case_repo/ci" \
        "$case_repo/tests" \
        "$case_repo/torch_custom/fla_npu/test" \
        "$fake_package_dir/opp/vendors/fla_npu_transformer/op_api/lib" \
        "$fake_opp_dir/vendors/fla_npu_transformer/op_api/lib"
    cp ci/run_checks.sh ci/manage_npu_ci_stage_report.py "$case_repo/ci/"
    touch "$fake_package_dir/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    : >"$command_log"
    set +e
    (
        cd "$case_repo"
        CI_STAGE="$requested_stage" \
        CI_STAGE_REPORT_FILE=.ci-tmp/stage-report.json \
        CI_ACCURACY_REPORT_FILE=.ci-tmp/accuracy.json \
        CI_TEST_COMMAND_LOG="$command_log" \
        CI_TEST_FAIL_STAGE_ONE="$fail_stage_one" \
        CI_TEST_FAIL_STAGE="$fail_stage" \
        CI_TEST_FAIL_MANAGER_INIT="$fail_manager_init" \
        CI_TEST_FAKE_PACKAGE_DIR="$fake_package_dir" \
        CI_MODE=quick \
        CI_SOC=ascend910b \
        CI_RUN_EXAMPLE_ST="$run_example_st" \
        CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK=false \
        CI_RUN_WHEEL_API_CHECK=false \
        CI_RUN_SCOPED_WHEEL_INSTALL_CHECK=false \
        CI_TMPDIR="$case_dir/tmp" \
        CI_TMPDIR_MIN_KB=1 \
        CI_JOBS=1 \
        CI_CPACK_JOBS=1 \
        PYTORCH_VERSION=2.7.1 \
        ASCEND_OPP_PATH="$fake_opp_dir" \
        /bin/bash ci/run_checks.sh
    ) >"$output_log" 2>&1
    exit_code=$?
    set -e

    if (( exit_code != expected_exit_code )); then
        echo "Unexpected exit code for $name: $exit_code (expected $expected_exit_code)" >&2
        sed -n '1,240p' "$output_log" >&2
        return 1
    fi
    if [[ "$expected_statuses" != "none" ]]; then
        assert_report "$report_file" "$expected_top_status" "$expected_statuses"
    fi
    if grep -Eq -- '-s ci/tests|test_run_checks_stage_pipeline\.sh' "$command_log"; then
        echo "CI control-plane contract tests ran inside the NPU stage." >&2
        return 1
    fi

    if [[ "$name" == "scoped" ]]; then
        grep -Fq 'bash ci/prepare_ci_cache.sh' "$command_log"
        grep -Fq 'bash build.sh --pkg --soc=ascend910b' "$command_log"
    elif [[ "$name" == "prerequisite-failure" ]]; then
        if grep -Fq 'bash ci/prepare_ci_cache.sh' "$command_log"; then
            echo "The OPP stage ran after the environment stage failed." >&2
            return 1
        fi
        grep -Fq '[CI][REPRO] CI_STAGE=environment-contracts' "$output_log"
        grep -Fq 'bash ci/run_ci_container.sh' "$output_log"
    elif [[ "$name" == "opp-failure" ]]; then
        if grep -Fxq 'bash build.sh' "$command_log"; then
            echo "The PyTorch stage ran after the OPP stage failed." >&2
            return 1
        fi
        grep -Fq '[CI][REPRO] CI_STAGE=opp-package' "$output_log"
    elif [[ "$name" == "bootstrap-failure" ]]; then
        grep -Fq 'injected stage report initialization failure' "$output_log"
        grep -Fq '[CI][REPRO] CI_STAGE=environment-contracts' "$output_log"
        grep -Fq 'bash ci/run_ci_container.sh' "$output_log"
    else
        grep -Fq 'bash build.sh --pkg --soc=ascend910b' "$command_log"
        grep -Fq 'bash build.sh' "$command_log"
        grep -Fq 'python3 ci/run_example_st_cases.py' "$command_log"
    fi
}

run_case \
    scoped \
    opp-package \
    false \
    0 \
    success \
    success,success,skipped,skipped,skipped,skipped

run_case \
    disabled-stage-continues \
    all \
    false \
    0 \
    success \
    success,success,skipped,success,success,skipped \
    true

run_case \
    prerequisite-failure \
    all \
    true \
    1 \
    failure \
    failure,skipped,skipped,skipped,skipped,skipped

run_case \
    opp-failure \
    all \
    false \
    1 \
    failure \
    success,failure,skipped,skipped,skipped,skipped \
    true \
    opp-package

run_case \
    bootstrap-failure \
    all \
    false \
    25 \
    failure \
    none \
    false \
    '' \
    true

echo "run_checks stage pipeline integration tests passed"
