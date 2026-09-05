#!/bin/bash
set -e
cd "$(dirname "$0")"
bash gen.sh npu_custom.yaml
rm -rf build dist flash_linear_attention_npu.egg-info fla_npu.egg-info
python3 setup.py bdist_wheel
shopt -s nullglob
wheels=(dist/flash_linear_attention_npu-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one flash-linear-attention-npu wheel, found ${#wheels[@]}." >&2
    exit 1
fi
python3 -m pip install "${wheels[0]}" --force-reinstall --no-deps --no-cache-dir

# ASCEND_CUSTOM_OPP_PATH：CANN 在初始化时注册自定义 OPP；若 Python/ATK/Celery 等进程
# 会先初始化 CANN，需要在进程启动前设置该变量，否则可能出现 SelectBin 找不到 kernel
# （如 aclnnStatus=561103）。
_fla_npu_site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
_fla_npu_vendor="${_fla_npu_site}/fla_npu/opp/vendors/fla_npu_transformer"
echo ""
echo "[fla-npu] 若使用 fla_npu 的进程会先初始化 CANN（Python/ATK/Celery 等），请在启动前执行："
echo "[fla-npu]   export ASCEND_CUSTOM_OPP_PATH=\"${_fla_npu_vendor}:${_fla_npu_vendor}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-}\""

# The fla_npu runtime loads libcust_opapi.so from the OPP tree embedded in the
# installed package (fla_npu/opp/vendors/fla_npu_transformer). The standalone
# wheel only ships the OPP skeleton, so overlay the compiled custom OPP from the
# just-built run package into the installed package and refresh the wheel RECORD
# before any consumer imports fla_npu.
run_pkg=""
shopt -s nullglob
for cand in ../../build_out/fla_npu_linux-*.run ../../build/fla_npu_linux-*.run; do
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        run_pkg="$cand"
        break
    fi
done
shopt -u nullglob
if [ -z "$run_pkg" ]; then
    echo "[ERROR] No fla_npu_linux-*.run package found to overlay the embedded OPP into the installed wheel." >&2
    exit 1
fi
chmod +x "$run_pkg"
"$run_pkg" --install --quiet
echo ""
echo "[fla-npu] run 包 OPP 已安装到：${_fla_npu_vendor}"
echo "[fla-npu]   若进程会先初始化 CANN（Python/ATK/Celery 等），请在启动前执行："
echo "[fla-npu]   export ASCEND_CUSTOM_OPP_PATH=\"${_fla_npu_vendor}:${_fla_npu_vendor}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-}\""
