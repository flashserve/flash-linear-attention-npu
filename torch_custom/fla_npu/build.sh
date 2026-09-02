#!/bin/bash
set -e
cd "$(dirname "$0")"

# Same interpreter resolution as gen.sh: the caller may pass FLA_NPU_PYTHON or
# PYTHON (the root setup.py sets PYTHON=sys.executable), otherwise fall back to
# whatever 'python3' resolves to via PATH.
PY="${FLA_NPU_PYTHON:-${PYTHON:-python3}}"

FLA_NPU_PYTHON="$PY" bash gen.sh npu_custom.yaml
"$PY" setup.py bdist_wheel
"$PY" -m pip install ./dist/fla_npu-1.0.0-*.whl --force-reinstall --no-deps

# ASCEND_CUSTOM_OPP_PATH：CANN 在初始化时注册自定义 OPP；若 Python/ATK/Celery 等进程
# 会先初始化 CANN，需要在进程启动前设置该变量，否则可能出现 SelectBin 找不到 kernel
# （如 aclnnStatus=561103）。
_fla_npu_site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
_fla_npu_vendor="${_fla_npu_site}/fla_npu/opp/vendors/fla_npu_transformer"
echo ""
echo "[fla-npu] 若使用 fla_npu 的进程会先初始化 CANN（Python/ATK/Celery 等），请在启动前执行："
echo "[fla-npu]   export ASCEND_CUSTOM_OPP_PATH=\"${_fla_npu_vendor}:${_fla_npu_vendor}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-}\""

# The fla_npu runtime loads libcust_opapi.so only from the OPP tree embedded in
# the installed package (fla_npu/opp/vendors/fla_npu_transformer). The standalone
# wheel built here ships only the OPP skeleton, so importing fla_npu fails with
# FileNotFoundError once the external-vendor runtime fallback was removed (PR #322).
# Overlay the compiled custom OPP from the just-built fla-npu-*.run package into
# the installed package, then refresh the installed wheel RECORD so pip uninstall
# also removes the embedded OPP. Unlike main, the v26.6.0 run installer has no
# --install wheel-merge, so we install the OPP directly into the package-local
# opp/ tree and finalize the RECORD ourselves.
run_pkg=""
shopt -s nullglob
for cand in ../../build_out/fla-npu-*.run ../../build/fla-npu-*.run; do
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        run_pkg="$cand"
        break
    fi
done
shopt -u nullglob
if [ -z "$run_pkg" ]; then
    echo "[ERROR] No fla-npu-*.run package found to overlay the embedded OPP into the installed wheel." >&2
    exit 1
fi
chmod +x "$run_pkg"
pkg_dir="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/fla_npu"
"$run_pkg" --quiet --install-path="$pkg_dir/opp"
echo ""
echo "[fla-npu] run 包 OPP 已安装到：${pkg_dir}/opp/vendors/fla_npu_transformer"
echo "[fla-npu]   若进程会先初始化 CANN（Python/ATK/Celery 等），请在启动前执行："
echo "[fla-npu]   export ASCEND_CUSTOM_OPP_PATH=\"${pkg_dir}/opp/vendors/fla_npu_transformer:${pkg_dir}/opp/vendors/fla_npu_transformer/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-}\""
"$PY" "$(dirname "$0")/finalize_wheel_opp.py" --package-dir "$pkg_dir"
