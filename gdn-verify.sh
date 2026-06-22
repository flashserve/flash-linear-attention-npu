#!/bin/bash
# gdn-verify.sh — GDN 仓库一键验证脚本
#
# 用法:
#   1) conda activate wnc && source <cann_path>/set_env.sh   (用户在脚本外完成)
#   2) bash gdn-verify.sh              # 全量验证（编译+装包+测试+examples）
#   3) bash gdn-verify.sh --skip-compile   # 跳过编译，只装包+测试
#   4) bash gdn-verify.sh --mode single --op chunk_bwd_dv_local  # 单算子验证
#
# 选项:
#   --mode full|single         验证模式（默认 full）
#   --op NAME                  单算子模式下的算子名
#   --device N                 指定 NPU device（默认自动选择第一个空闲的）
#   --skip-compile             跳过阶段1-3（编译+装包）
#   --skip-test                跳过阶段5（单算子测试）
#   --skip-example             跳过阶段6（整网示例）

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ================================================================
# 参数解析
# ================================================================
MODE="full"
SINGLE_OP=""
DEVICE_ID=""
SKIP_COMPILE=false
SKIP_TEST=false
SKIP_EXAMPLE=false

usage() {
    cat <<EOF
用法: bash gdn-verify.sh [OPTIONS]

前置条件（用户在脚本外完成）:
    conda activate <env> && source <cann_path>/set_env.sh

选项:
  --mode full|single         验证模式（默认 full）
  --op NAME                  单算子模式下的算子名
  --device N                 指定 NPU device（默认自动选择第一个空闲的）
  --skip-compile             跳过编译+装包阶段
  --skip-test                跳过单算子测试阶段
  --skip-example             跳过整网示例阶段
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2";       shift 2 ;;
        --op)            SINGLE_OP="$2";  shift 2 ;;
        --device)        DEVICE_ID="$2";  shift 2 ;;
        --skip-compile)  SKIP_COMPILE=true; shift ;;
        --skip-test)     SKIP_TEST=true;  shift ;;
        --skip-example)  SKIP_EXAMPLE=true; shift ;;
        --help|-h)       usage ;;
        *)               usage ;;
    esac
done

# ================================================================
# 常量定义
# ================================================================
ALL_OPS="causal_conv1d,chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd"
TEST_DIR="$SCRIPT_DIR/torch_custom/fla_npu/test"
TEST_SCRIPT="$TEST_DIR/test.sh"
EXAMPLE_SCRIPT="$SCRIPT_DIR/examples/flash_gated_delta_rule.py"

# 共享状态（各阶段填充）
declare -A COMPILE_RESULTS=()
WHL_OK=false
RUN_OK=false
declare -A TEST_RESULTS=()
EXAMPLE_OK=false
FAIL_DETAILS=""
SOC_FOR_INSTALL=""
CANNDIR_FOR_INSTALL=""
CANNDIR=""
SKIPPED_COMPILE=false

# ================================================================
# 工具函数
# ================================================================
die() { echo "[FATAL] $*" >&2; exit 1; }

detect_cann_version() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        CANNDIR="$ASCEND_HOME_PATH"
    elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        CANNDIR="$ASCEND_TOOLKIT_HOME"
    else
        local bisheng_path
        bisheng_path=$(which bisheng 2>/dev/null || true)
        if [[ -n "$bisheng_path" ]]; then
            CANNDIR=$(dirname "$(dirname "$bisheng_path")")
        fi
    fi

    if [[ -z "$CANNDIR" ]]; then
        die "请先 source CANN set_env.sh（无法检测 CANN 路径）"
    fi

    CANN_VERSION_STR=""
    local version_candidates=(
        "$CANNDIR/version.cfg"
        "$CANNDIR/version.info"
        "$CANNDIR/opp/version.info"
    )
    for version_file in "${version_candidates[@]}"; do
        [[ -f "$version_file" ]] || continue
        if [[ "$version_file" == *.cfg ]]; then
            CANN_VERSION_STR=$(grep -oP 'version=\K[0-9.]+' "$version_file" 2>/dev/null | head -1)
        else
            CANN_VERSION_STR=$(
                grep -oP '(?:Version|version)=\K[0-9]+\.[0-9]+(?:\.[0-9]+)?' "$version_file" 2>/dev/null | head -1
            )
            if [[ -z "$CANN_VERSION_STR" ]]; then
                CANN_VERSION_STR=$(sed -n 's/^[Vv]ersion=\([0-9][0-9.]*\).*/\1/p' "$version_file" 2>/dev/null | head -1)
            fi
            if [[ -z "$CANN_VERSION_STR" ]]; then
                CANN_VERSION_STR=$(head -1 "$version_file" | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            fi
        fi
        [[ -n "$CANN_VERSION_STR" ]] && break
    done
    if [[ -n "$CANN_VERSION_STR" ]]; then
        CANN_MAJOR_MINOR=$(echo "$CANN_VERSION_STR" | cut -d. -f1-2)
    else
        CANN_MAJOR_MINOR=""
    fi
    if [[ -z "$CANN_VERSION_STR" ]]; then
        echo "[WARN] 无法检测 CANN 版本，假设为 9.0"
        CANN_MAJOR_MINOR="9.0"
    fi
    echo "[INFO] CANN 版本: $CANN_VERSION_STR ($CANN_MAJOR_MINOR)"
}

get_soc_list() {
    local socs
    if [[ "$CANN_MAJOR_MINOR" == "8.5" ]]; then
        socs="ascend910b ascend910_93"
    else
        socs="ascend910b ascend910_93 ascend950"
    fi
    echo "$socs"
}

detect_soc_from_npu() {
    local name
    name=$(npu-smi info 2>/dev/null | awk -F'|' '/^\|[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]*\|/ {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
        split($2, fields, /[[:space:]]+/);
        print fields[2];
        exit;
    }')
    if [[ -z "$name" ]]; then
        echo "[WARN] 无法从 npu-smi 解析 NPU 型号，假设为 ascend910b"
        name="910B"
    fi
    case "$name" in
        *910B*|*910b*) echo "ascend910b" ;;
        *910_93*|*910*93*) echo "ascend910_93" ;;
        *950*|*Ascend950*) echo "ascend950" ;;
        *) echo "[WARN] 未知 NPU 型号: $name，假设为 ascend910b"; echo "ascend910b" ;;
    esac
}

find_run_file() {
    ls "$SCRIPT_DIR/build_out"/fla-npu-*.run 2>/dev/null | head -1
}

source_vendor_env() {
    local vendor_env="${CANNDIR}/vendors/fla_npu_transformer/bin/set_env.bash"
    if [[ -f "$vendor_env" ]]; then
        # vendor set_env.bash 会 append ${ASCEND_CUSTOM_OPP_PATH}；脚本启用了 set -u，需先给默认值
        export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
        # shellcheck disable=SC1090
        source "$vendor_env"
        echo "[INFO] 已加载自定义算子环境: $vendor_env"
    else
        echo "[WARN] 未找到 $vendor_env；若测试报 libopapi 找不到，请先安装 .run 并 source 该文件"
    fi
}

auto_select_device() {
    local dev
    if [[ -x "$SCRIPT_DIR/ci/detect_npu.sh" ]]; then
        dev=$("$SCRIPT_DIR/ci/detect_npu.sh" --selected 2>/dev/null || true)
        if [[ -n "$dev" ]]; then
            echo "$dev"
            return
        fi
    fi
    dev=$(npu-smi info 2>/dev/null | awk '/^\| [0-9]+ /{id=$2; next} /No running processes/ && id!=""{print id; exit}' ORS='')
    if [[ -z "$dev" ]]; then
        dev=$(npu-smi info 2>/dev/null | awk '/^\| [0-9]+ *\|/{print $2; exit}')
    fi
    if [[ -z "$dev" ]]; then
        dev="0"
    fi
    echo "$dev"
}

# ================================================================
# 阶段1：编译
# ================================================================
compile_one() {
    local soc=$1 type=$2 ops=$3
    local key="B_${soc}_${type}"
    local ops_arg=""
    [[ -n "$ops" ]] && ops_arg=" --ops=$ops"

    echo ""
    echo "--- 编译: SOC=$soc 类型=$type ---"
    rm -rf build

    local log_file="/tmp/gdn_compile_${soc}_${type}.log"
    local start_ts=$(date +%s)

    if bash build.sh --pkg --soc="$soc" --vendor_name=fla_npu $ops_arg > "$log_file" 2>&1; then
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "  [OK]  ${elapsed}s  log: $log_file"
        COMPILE_RESULTS["$key"]="OK"
    else
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        local tail_msg
        tail_msg=$(tail -5 "$log_file" | head -3)
        echo "  [FAIL] ${elapsed}s  log: $log_file"
        echo "    尾部输出: $tail_msg"
        COMPILE_RESULTS["$key"]="FAIL"
        FAIL_DETAILS+=$'\n'"编译失败: SOC=$soc $type → $log_file"
    fi
}

run_compile_stage() {
    echo ""
    echo "=========================================="
    echo "阶段1: 编译 (CANN $CANN_MAJOR_MINOR)"
    echo "=========================================="

    local socs
    socs=$(get_soc_list)

    local ops_for_single
    if [[ "$MODE" == "single" ]]; then
        ops_for_single="$SINGLE_OP"
    else
        ops_for_single="$ALL_OPS"
    fi

    for soc in $socs; do
        compile_one "$soc" "整包" ""
    done

    for soc in $socs; do
        compile_one "$soc" "单算子" "$ops_for_single"
    done
}

# ================================================================
# 阶段2：whl 编译安装
# ================================================================
run_whl_stage() {
    echo ""
    echo "=========================================="
    echo "阶段2: whl 包编译安装"
    echo "=========================================="

    local log_file="/tmp/gdn_whl.log"
    cd torch_custom/fla_npu

    if bash build.sh > "$log_file" 2>&1; then
        if grep -q 'Successfully installed fla-npu' "$log_file"; then
            echo "[OK] fla-npu-1.0.0 安装成功"
            WHL_OK=true
        else
            echo "[WARN] 未检测到安装成功标志，检查日志: $log_file"
            WHL_OK=true
        fi
    else
        echo "[FAIL] whl 编译安装失败 → $log_file"
        tail -10 "$log_file"
        WHL_OK=false
        FAIL_DETAILS+=$'\n'"whl 安装失败 → $log_file"
    fi
    cd "$SCRIPT_DIR"
}

# ================================================================
# 阶段3：.run 包安装
# ================================================================
run_install_stage() {
    echo ""
    echo "=========================================="
    echo "阶段3: .run 包安装"
    echo "=========================================="

    SOC_FOR_INSTALL=$(detect_soc_from_npu | tail -1)
    CANNDIR_FOR_INSTALL="$CANNDIR"
    echo "[INFO] NPU SOC: $SOC_FOR_INSTALL"
    echo "[INFO] CANN 路径: $CANNDIR_FOR_INSTALL"

    echo ""
    echo "--- 重新编译 $SOC_FOR_INSTALL 整包（匹配当前 NPU） ---"
    rm -rf build
    local compile_log="/tmp/gdn_install_compile.log"
    if ! bash build.sh --pkg --soc="$SOC_FOR_INSTALL" --vendor_name=fla_npu > "$compile_log" 2>&1; then
        echo "[FAIL] 重编失败 → $compile_log"
        FAIL_DETAILS+=$'\n'".run 安装前重编失败 ($SOC_FOR_INSTALL)"
        RUN_OK=false
        return
    fi
    echo "[OK] 重编成功"

    local run_file
    run_file=$(find_run_file)
    if [[ -z "$run_file" ]]; then
        echo "[FAIL] 找不到 .run 文件"
        RUN_OK=false
        FAIL_DETAILS+=$'\n'".run 文件未找到"
        return
    fi

    echo "--- 安装 $run_file ---"
    local install_log="/tmp/gdn_install_run.log"
    if bash "$run_file" --install-path="$CANNDIR_FOR_INSTALL" --install-for-all --quiet > "$install_log" 2>&1; then
        if grep -q 'SUCCESS' "$install_log"; then
            echo "[OK] .run 包安装成功"
            RUN_OK=true
        else
            echo "[OK] .run 包安装完成（未检测到 SUCCESS 字样，继续）"
            RUN_OK=true
        fi
    else
        echo "[FAIL] .run 包安装失败 → $install_log"
        tail -20 "$install_log"
        RUN_OK=false
        FAIL_DETAILS+=$'\n'".run 安装失败 → $install_log"
        return
    fi
    source_vendor_env
}

# ================================================================
# 阶段4：测试准备（自动选择 device）
# ================================================================
prepare_test_env() {
    echo ""
    echo "=========================================="
    echo "阶段4: 测试环境准备"
    echo "=========================================="

    if [[ -z "$DEVICE_ID" ]]; then
        DEVICE_ID=$(auto_select_device)
        echo "[INFO] 自动选择 NPU device: $DEVICE_ID"
    else
        echo "[INFO] 使用指定 NPU device: $DEVICE_ID"
    fi

    pip install ml_dtypes -q 2>/dev/null || true
    source_vendor_env
    echo "[INFO] 依赖检查完成"
}

# ================================================================
# 阶段5：测试
# ================================================================
run_test_stage() {
    echo ""
    echo "=========================================="
    echo "阶段5: 单算子测试"
    echo "=========================================="

    if [[ ! -f "$TEST_SCRIPT" ]]; then
        echo "[WARN] test.sh 不存在: $TEST_SCRIPT"
        return
    fi

    local args="--device $DEVICE_ID"
    if [[ "$MODE" == "single" ]]; then
        args="$args --op $SINGLE_OP"
    fi

    cd "$TEST_DIR"
    local test_output_file="/tmp/gdn_test_output.txt"
    bash "$TEST_SCRIPT" $args > "$test_output_file" 2>&1
    local test_exit_code=$?
    cat "$test_output_file"
    cd "$SCRIPT_DIR"

    local section="" name
     while IFS= read -r line; do
         if [[ "$line" =~ ^[[:space:]]*\[PASS\][[:space:]]+([^[:space:]]+) ]]; then
             TEST_RESULTS["${BASH_REMATCH[1]}"]="PASS"
         elif [[ "$line" =~ ^[[:space:]]*\[FAIL\][[:space:]]+([^[:space:]→]+) ]]; then
             TEST_RESULTS["${BASH_REMATCH[1]}"]="FAIL"
         elif [[ "$line" =~ ^[[:space:]]*\[TIMEOUT\][[:space:]]+([^[:space:]→]+) ]]; then
             TEST_RESULTS["${BASH_REMATCH[1]}"]="TIMEOUT"
         fi
     done < "$test_output_file"

    return $test_exit_code
}

# ================================================================
# 阶段6：Examples 整网验证
# ================================================================
run_example_stage() {
    echo ""
    echo "=========================================="
    echo "阶段6: Examples 整网验证"
    echo "=========================================="

    if [[ ! -f "$EXAMPLE_SCRIPT" ]]; then
        echo "[WARN] 示例脚本不存在: $EXAMPLE_SCRIPT"
        EXAMPLE_OK=false
        FAIL_DETAILS+=$'\n'"examples/flash_gated_delta_rule.py 不存在"
        return
    fi

    local log_file="/tmp/gdn_example.log"
    local start_ts=$(date +%s)
    if timeout 600 python3 "$EXAMPLE_SCRIPT" > "$log_file" 2>&1; then
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[OK] ${elapsed}s"
        EXAMPLE_OK=true
    elif [[ $? -eq 124 ]]; then
        echo "[TIMEOUT]"
        EXAMPLE_OK=false
        FAIL_DETAILS+=$'\n'"examples 整网验证超时 (600s)"
    else
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[FAIL] ${elapsed}s → $log_file"
        tail -10 "$log_file" 2>/dev/null
        EXAMPLE_OK=false
        FAIL_DETAILS+=$'\n'"examples 整网验证失败 → $log_file"
    fi
}

# ================================================================
# 阶段7：生成测试报告
# ================================================================
print_report() {
    echo ""
    echo "================================================================="
    echo "                  GDN 验证测试报告"
    echo "================================================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "环境: CANN $CANN_MAJOR_MINOR"
    echo "NPU:  $(npu-smi info 2>/dev/null | grep -oP 'Ascend\S+' | head -1) (选用 device $DEVICE_ID)"
    echo "SOC:  $SOC_FOR_INSTALL"
    echo ""

    # 编译结果
    echo "-----------------------------------------------------------------"
    echo "一、编译结果 (CANN $CANN_MAJOR_MINOR)"
    echo "-----------------------------------------------------------------"
    local socs comp_total=0 comp_ok=0
    socs=$(get_soc_list)
    for soc in $socs; do
        for type in "整包" "单算子"; do
            local key="B_${soc}_${type}"
            local result="${COMPILE_RESULTS[$key]:--}"
            comp_total=$((comp_total + 1))
            [[ "$result" == "OK" ]] && comp_ok=$((comp_ok + 1))
            printf "| %-20s | %-13s | %-4s | %s\n" "$soc" "$type" "" "$result"
        done
    done
    echo ""
    echo "编译通过: $comp_ok / $comp_total"

    # 安装结果
    echo ""
    echo "-----------------------------------------------------------------"
    echo "二、安装"
    echo "-----------------------------------------------------------------"
    if $SKIPPED_COMPILE; then
        echo "whl 编译安装: (已跳过)"
        echo ".run 包安装:  (已跳过)"
    else
        echo "whl 编译安装: $($WHL_OK && echo 'OK' || echo 'FAIL')"
        echo ".run 包安装:  $($RUN_OK && echo 'OK' || echo 'FAIL')"
    fi

    # 测试结果
    echo ""
    echo "-----------------------------------------------------------------"
    echo "三、单算子测试结果 (device=$DEVICE_ID)"
    echo "-----------------------------------------------------------------"
    if $SKIP_TEST; then
        echo "(已跳过)"
    else
        local op_names=(
            "prepare_wy_repr_bwd_full"
            "chunk_gated_delta_rule_bwd_dhu"
            "chunk_bwd_dv_local"
            "causal_conv1d"
            "prepare_wy_repr_bwd_da"
            "chunk_bwd_dqkwg"
            "gdn_fwd_o"
            "gdn_fwd_h"
            "recompute_wu_fwd"
        )
        local test_pass=0 test_fail=0 test_timeout=0
        for op in "${op_names[@]}"; do
            local result="${TEST_RESULTS[$op]:--}"
            [[ "$result" == "PASS" ]] && test_pass=$((test_pass + 1))
            [[ "$result" == "FAIL" ]] && test_fail=$((test_fail + 1))
            [[ "$result" == "TIMEOUT" ]] && test_timeout=$((test_timeout + 1))
            printf "| %-36s | %-7s |\n" "$op" "$result"
        done
        echo ""
        echo "PASS: $test_pass  FAIL: $test_fail  TIMEOUT: $test_timeout"
        echo "详细日志: torch_custom/fla_npu/test/test_output/"
    fi

    # Examples
    echo ""
    echo "-----------------------------------------------------------------"
    echo "四、Examples 整网"
    echo "-----------------------------------------------------------------"
    echo "flash_gated_delta_rule.py: $($EXAMPLE_OK && echo 'OK' || echo 'FAIL')"

    # 失败详情
    if [[ -n "$FAIL_DETAILS" ]]; then
        echo ""
        echo "-----------------------------------------------------------------"
        echo "五、失败/异常详情"
        echo "-----------------------------------------------------------------"
        echo "$FAIL_DETAILS"
    fi

    echo ""
    echo "================================================================="
}

# ================================================================
# 主流程
# ================================================================

trap 'echo "[中断] 脚本被用户终止"; exit 130' INT

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       GDN 仓库一键验证                    ║"
echo "╠══════════════════════════════════════════╣"
echo "║  模式: $(printf '%-32s' "$MODE")║"
if [[ "$MODE" == "single" ]]; then
    echo "║  算子: $(printf '%-32s' "$SINGLE_OP")║"
fi
echo "║  CANN: 用户预加载                          ║"
echo "╚══════════════════════════════════════════╝"

detect_cann_version

# --- 编译阶段 ---
if $SKIP_COMPILE; then
    echo ""
    echo "[INFO] 跳过编译+装包阶段"
    SKIPPED_COMPILE=true
    SOC_FOR_INSTALL=$(detect_soc_from_npu | tail -1)
    echo "[INFO] NPU SOC: $SOC_FOR_INSTALL"
else
    run_compile_stage
    run_whl_stage
    run_install_stage
fi

# --- 测试阶段 ---
prepare_test_env

if $SKIP_TEST; then
    echo ""
    echo "[INFO] 跳过单算子测试阶段"
else
    run_test_stage || true
fi

if $SKIP_EXAMPLE; then
    echo ""
    echo "[INFO] 跳过整网示例阶段"
else
    run_example_stage
fi

# --- 报告 ---
print_report

exit 0
