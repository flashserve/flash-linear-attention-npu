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
    if [[ "${CI_TEST_FAIL_STABLE_COVERAGE:-false}" == "true" && \
          "$entrypoint" == "torch_custom/fla_npu/tools/stable_coverage.py" ]]; then
        echo "injected Stable coverage failure" >&2
        return 26
    fi
    if [[ "$entrypoint" == "-" && -n "${CI_TEST_FAKE_PACKAGE_DIR:-}" ]]; then
        [[ "${FLA_NPU_ENV:-}" == "$CI_TEST_FAKE_PACKAGE_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash" ]]
        [[ -n "${FLA_NPU_OP_API_LIBS:-}" ]]
        if [[ "$CI_BUILD_KIND" == incremental ]]; then
            [[ "$ASCEND_CUSTOM_OPP_PATH" == "$CI_TEST_FAKE_PACKAGE_DIR/opp/vendors/fla_npu_overlay_transformer:"* ]]
        fi
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
    if [[ "${1:-}" == ci/prepare_ci_package.sh ]]; then
        printf 'package kind=%s ops=%s\n' "$CI_BUILD_KIND" "$CI_BUILD_OPS" >>"$CI_TEST_COMMAND_LOG"
        if [[ "${CI_TEST_FAIL_STAGE:-}" == opp-package ]]; then
            echo "injected package preparation failure" >&2
            return 24
        fi
        if [[ -n "${CI_TEST_BUILD_ENV_LOG:-}" ]]; then
            command "$CI_TEST_REAL_PYTHON3" -c 'import json, os; print(json.dumps({key: os.environ.get(key) for key in ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "MAKEFLAGS", "FLA_NPU_BUILD_ARGS", "CI_BUILD_KIND", "CI_BUILD_OPS")}))' >>"$CI_TEST_BUILD_ENV_LOG"
        fi
        mkdir -p .ci-tmp
        printf 'export FLA_NPU_ENV=%q\nexport FLA_NPU_OPP_PATH=%q\nexport FLA_NPU_OVERLAY_VENDOR=%q\n' \
            "$CI_TEST_FAKE_PACKAGE_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash" \
            "$CI_TEST_FAKE_PACKAGE_DIR/opp" \
            "$([[ "$CI_BUILD_KIND" == incremental ]] && echo fla_npu_overlay_transformer || true)" \
            >.ci-tmp/npu-ci-package-env.sh
    elif [[ "${1:-}" == ci/run_atk_checks.sh ]]; then
        [[ "$NPU_DEVICE_ID" == "$CI_TEST_EXPECT_DEVICE" ]]
        [[ "$SOC" == ascend910b ]]
        [[ "$ATK_OUTPUT_ROOT" == "$PWD/.ci-tmp/atk-results" ]]
        [[ -n "${FLA_NPU_OP_API_LIBS:-}" ]]
        printf 'atk ops=%s device=%s\n' "$CI_ATK_OPS" "$NPU_DEVICE_ID" >>"$CI_TEST_COMMAND_LOG"
        if [[ "${CI_TEST_FAIL_STAGE:-}" == atk-accuracy ]]; then
            echo "injected ATK accuracy failure: operator missing >=200 accuracy cases" >&2
            return 27
        fi
    fi
    return 0
}

source() {
    case "$1" in /usr/local/Ascend/*) return 0 ;; esac
    builtin source "$@"
}
export -f python3 bash source

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
    "atk-accuracy",
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
    local fail_stable_coverage="${10:-false}"
    local build_kind="${11:-base}" mode="${12:-quick}" atk_ops="${13:-}" build_ops="${14:-}"
    local device="${15:-0}" run_torch_tests="${16:-false}"
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
        "$fake_package_dir/opp/vendors/fla_npu_transformer/bin" \
        "$fake_opp_dir/vendors/fla_npu_transformer/op_api/lib"
    cp ci/run_checks.sh ci/manage_npu_ci_stage_report.py "$case_repo/ci/"
    touch "$fake_package_dir/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    cat >"$fake_package_dir/opp/vendors/fla_npu_transformer/bin/set_env.bash" <<'ENV'
printf 'package-env-loaded\n' >>"$CI_TEST_COMMAND_LOG"
_base="$CI_TEST_FAKE_PACKAGE_DIR/opp/vendors/fla_npu_transformer"
_overlay="$CI_TEST_FAKE_PACKAGE_DIR/opp/vendors/fla_npu_overlay_transformer"
if [[ -n "$FLA_NPU_OVERLAY_VENDOR" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$_overlay:$_base"
    export FLA_NPU_OP_API_LIBS="$_overlay/op_api/lib/libcust_opapi.so:$_base/op_api/lib/libcust_opapi.so"
else
    export ASCEND_CUSTOM_OPP_PATH="$_base"
    export FLA_NPU_OP_API_LIBS="$_base/op_api/lib/libcust_opapi.so"
fi
ENV
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
        CI_TEST_FAIL_STABLE_COVERAGE="$fail_stable_coverage" \
        CI_TEST_FAKE_PACKAGE_DIR="$fake_package_dir" \
        CI_MODE="$mode" \
        CI_BUILD_KIND="$build_kind" CI_BUILD_OPS="$build_ops" CI_ATK_OPS="$atk_ops" CI_BASE_REF=main \
        CI_CONTAINER_DEVICE="$device" CI_DEVICE_ID="$device" CI_TEST_EXPECT_DEVICE="$device" \
        CI_RUN_TORCH_TESTS="$run_torch_tests" \
        CI_SOC=ascend910b \
        CI_RUN_EXAMPLE_ST="$run_example_st" \
        CI_RUN_WHEEL_API_CHECK=false \
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
    if grep -Eq -- 'bash build.sh|setup.py|pip wheel|-s ci/tests|test_run_checks_stage_pipeline\.sh' "$command_log"; then
        echo "NPU dispatcher performed a hidden build or nested control-plane tests." >&2
        return 1
    fi
    if [[ "$expected_statuses" == success,* ]]; then
        grep -Fqx \
            'python3 torch_custom/fla_npu/tools/stable_coverage.py --strict' \
            "$command_log"
        grep -Fqx \
            'python3 torch_custom/fla_npu/tools/op_abi_parity.py' \
            "$command_log"
    fi

    if [[ "$expected_top_status" == failure ]]; then
        grep -Fq '[CI][REPRO]' "$output_log"
        for variable in CI_BUILD_KIND CI_BUILD_OPS CI_ATK_OPS CI_BASE_REF CI_DEVICE_ID; do
            grep -Fq "$variable=" "$output_log"
        done
    fi
    if [[ "$expected_statuses" == failure,* || "$name" == bootstrap-failure ]]; then
        ! grep -Fq 'bash ci/prepare_ci_package.sh' "$command_log"
    elif [[ "$expected_statuses" == success,success,* ]]; then
        [[ "$(grep -Fxc 'bash ci/prepare_ci_package.sh' "$command_log")" == 1 ]]
        local expected_kind="$build_kind"
        [[ "$mode" == full ]] && expected_kind=full
        grep -Fq "package kind=$expected_kind" "$command_log"
        if [[ "$requested_stage" == all ]]; then
            if [[ "$mode" == full || "$run_torch_tests" == true ]]; then
                grep -Fq 'bash test.sh --device' "$command_log"
            else
                ! grep -Fq 'bash test.sh' "$command_log"
            fi
            if [[ -n "$atk_ops" && "$atk_ops" != null ]]; then
                grep -Fq "atk ops=$atk_ops device=$device" "$command_log"
            elif [[ "$expected_top_status" == success ]]; then
                ! grep -Fq 'bash ci/run_atk_checks.sh' "$command_log"
                command "$CI_TEST_REAL_PYTHON3" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["stages"]["atk-accuracy"]["reason"] == "no ATK operators selected", p' "$report_file"
            fi
        fi
    fi
}

run_case scoped opp-package false 0 success success,success,skipped,skipped,skipped,skipped false '' false false incremental quick '' chunk_fwd_o
run_case base all false 0 success success,success,success,success,success,skipped true
run_case incremental-atk all false 0 success success,success,success,success,success,success true '' false false incremental quick chunk_fwd_o chunk_fwd_o 3
run_case incremental-atk-disabled all false 0 success success,success,success,success,success,skipped true '' false false incremental quick null chunk_fwd_o
run_case full all false 0 success success,success,success,success,success,skipped true '' false false base full
run_case torch-only all false 0 success success,success,success,success,success,skipped true '' false false torch
run_case quick-extra-torch all false 0 success success,success,success,success,success,skipped true '' false false base quick '' '' 0 true
run_case disabled-example all false 0 success success,success,success,success,skipped,skipped false
run_case prerequisite-failure all true 1 failure failure,skipped,skipped,skipped,skipped,skipped
run_case stable-coverage-failure all false 1 failure failure,skipped,skipped,skipped,skipped,skipped false '' false true
run_case opp-failure all false 1 failure success,failure,skipped,skipped,skipped,skipped true opp-package
run_case atk-failure all false 1 failure success,success,success,success,success,failure true atk-accuracy false false incremental quick chunk_fwd_o chunk_fwd_o
run_case bootstrap-failure all false 25 failure none false '' true

echo "run_checks stage pipeline integration tests passed"
