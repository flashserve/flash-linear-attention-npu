#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
# Adapted for flash-linear-attention-npu by Tianjin University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

vendor_name=customize
targetdir=/usr/local/Ascend/opp
target_custom=0

sourcedir=$PWD/packages
vendordir=vendors/$vendor_name

QUIET="y"
INSTALL_FOR_ALL="n"
WHEEL_INSTALL="y"
MODE_SET="n"


while true
do
    case $1 in
    --quiet)
        QUIET="y"
        shift
    ;;
    --full|--install)
        WHEEL_INSTALL="y"
        MODE_SET="y"
        shift
    ;;
    --cann)
        WHEEL_INSTALL="n"
        MODE_SET="y"
        shift
    ;;
    --install-path=*)
        INSTALL_PATH=$(echo $1 | cut -d"=" -f2-)
        INSTALL_PATH=${INSTALL_PATH%*/}
        # An explicit install path targets the CANN OPP layout, unless a
        # wheel/cann install mode was already requested explicitly.
        if [ "${MODE_SET}" = "n" ]; then
            WHEEL_INSTALL="n"
            MODE_SET="y"
        fi
        shift
    ;;
    --install-for-all)
        INSTALL_FOR_ALL="y"
        shift
    ;;
    --*)
        shift
    ;;
    *)
        break
    ;;
    esac
done

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    printf '[ops_custom] [%s] %s\n' "${cur_date}" "$1"
}

get_python_bin() {
    local candidate
    for candidate in python python3; do
        if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info[0] >= 3 else 1)' >/dev/null 2>&1; then
            command -v "${candidate}"
            return
        fi
    done
    echo ""
}

get_wheel_opp_root() {
    local python_bin
    python_bin=$(get_python_bin)
    if [ -z "${python_bin}" ]; then
        log "[WARNING] python is required to locate the installed fla_npu wheel OPP."
        return 1
    fi

    "${python_bin}" - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("fla_npu")
if spec is None or spec.origin is None:
    print("fla_npu is not importable. Install flash-linear-attention-npu wheel first.", file=sys.stderr)
    raise SystemExit(1)

package_dir = Path(spec.origin).resolve().parent
print(package_dir / "opp")
PY
}

install_wheel_opp_package() {
    local src_vendor="${sourcedir}/${vendordir}"
    local wheel_opp_root

    if [ ! -d "${src_vendor}" ]; then
        log "[ERROR] The run package does not contain ${src_vendor}."
        exit 1
    fi

    wheel_opp_root=$(get_wheel_opp_root)
    if [ -z "${wheel_opp_root}" ]; then
        log "[WARNING] fla_npu wheel is not importable, falling back to CANN OPP install."
        return 1
    fi
    # A scoped vendor has shared opapi/tiling libraries. Compose it with the
    # unchanged base instead of overwriting libraries used by other operators.
    local python_bin
    local status
    python_bin=$(get_python_bin)
    log "[INFO] Install this run package incrementally; preserve the base wheel and its other operators."
    "${python_bin}" -m fla_npu_opp_manage --overlay --source "${src_vendor}"
    status=$?
    if [ "${status}" -ne 0 ]; then
        log "[ERROR] Incremental installation failed. A wheel with fla_npu_opp_manage is required; the base was not overwritten."
        exit "${status}"
    fi
    log "[INFO] Installation completed; see the reported activation state. Restart Python processes before use."
    exit 0
}

if [ "${WHEEL_INSTALL}" = "y" ]; then
    install_wheel_opp_package
fi
if [ -n "${INSTALL_PATH}" ]; then
    if [[ ! "${INSTALL_PATH}" = /* ]]; then
        log "[ERROR] use absolute path for --install-path argument"
        exit 1
    fi
    if [ ! -d ${INSTALL_PATH} ]; then
        mkdir ${INSTALL_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${INSTALL_PATH}  failed"
            exit 1
        fi
    fi
    targetdir=${INSTALL_PATH}
elif [ -n "${ASCEND_CUSTOM_OPP_PATH}" ]; then
    if [[ "${ASCEND_CUSTOM_OPP_PATH}" == *:* ]]; then
        log "[ERROR] environment variable ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH} is set and \
        has multiple path in it (colon inside), which will cause the custom op installed incorrectly. \
        Please use the --install-path option to specify an installation path instead."
        exit 1
    fi
    if [ ! -d ${ASCEND_CUSTOM_OPP_PATH} ]; then
        mkdir -p ${ASCEND_CUSTOM_OPP_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${ASCEND_CUSTOM_OPP_PATH}  failed"
        fi
    fi
    targetdir=${ASCEND_CUSTOM_OPP_PATH}
else
    if [ "x${ASCEND_OPP_PATH}" == "x" ]; then
        log "[ERROR] env ASCEND_OPP_PATH no exist"
        exit 1
    fi
    targetdir="${ASCEND_OPP_PATH}"
fi

if [ ! -d $targetdir ];then
    log "[ERROR] $targetdir no exist"
    exit 1
fi

if [ ! -x $targetdir ] || [ ! -w $targetdir ] || [ ! -r $targetdir ];then
    log "[WARNING] The directory $targetdir does not have sufficient permissions. \
    Please check and modify the folder permissions (e.g., using chmod), \
    or use the --install-path option to specify an installation path and \
    change the environment variable ASCEND_CUSTOM_OPP_PATH to the specified path."
fi

upgrade()
{
    if [ ! -d ${sourcedir}/$vendordir/$1 ]; then
        log "[INFO] no need to upgrade ops $1 files"
        return 0
    fi

    if [ ! -d ${targetdir}/$vendordir/$1 ];then
        log "[INFO] create ${targetdir}/$vendordir/$1."
        mkdir -p ${targetdir}/$vendordir/$1
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/$1 failed"
            return 1
        fi
    else
        has_same_file=-1
        for file_a in ${sourcedir}/$vendordir/$1/*; do
            file_b=${file_a##*/};
            if [ "ls ${targetdir}/$vendordir/$1" = "" ]; then
                log "[INFO] ${targetdir}/$vendordir/$1 is empty !!"
		        return 1
	          fi
            grep -q $file_b <<<`ls ${targetdir}/$vendordir/$1`;
            if [[ $? -eq 0 ]]; then
                echo "  - ${file_b}"
                has_same_file=0
            fi
        done
        if [ 0 -eq $has_same_file ]; then
            echo
            if test $QUIET = "n"; then
                echo "[INFO]: has old version in ${targetdir}/$vendordir/$1, you want to Overlay Installation, please enter:[o]; or replace directory installation, please enter:[r]; or not install, please enter:[n]."

                while true
                do
                    read orn
                    if [ "$orn" = n ]; then
                        return 0
                    elif [ "$orn" = o ]; then
                        break;
                    elif [ "$orn" = r ]; then
                        [ -n "${targetdir}/$vendordir/$1/" ] && rm -rf "${targetdir}/$vendordir/$1"/*
                        break;
                    else
                        log "[ERROR] input error, please input again!"
                    fi
                done
            else
                [ -n "${targetdir}/$vendordir/$1/" ] && rm -rf "${targetdir}/$vendordir/$1"/*
            fi
        fi
        log "[INFO] replace or merge old ops $1 files ......"
    fi

    log "[INFO] copy new ops $1 files ......"
    if [ -d ${targetdir}/$vendordir/$1/ ]; then
        chmod -R +w "$targetdir/$vendordir/$1/" >/dev/null 2>&1
    fi
    cp -rf ${sourcedir}/$vendordir/$1/* $targetdir/$vendordir/$1/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 files failed"
        return 1
    fi

    return 0
}
upgrade_proto()
{
    if [ ! -f ${sourcedir}/$vendordir/custom.proto ]; then
        log "[INFO] no need to upgrade custom.proto files"
        return 0
    fi
    if [ ! -d ${targetdir}/$vendordir/framework/caffe ];then
        log "[INFO] create ${targetdir}/$vendordir/framework/caffe."
        mkdir -p ${targetdir}/$vendordir/framework/caffe
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/framework/caffe failed"
            return 1
        fi
    else
        if [ -f ${targetdir}/$vendordir/framework/caffe/custom.proto ]; then
            # 有老版本,判断是否要覆盖式安装
            if test $QUIET = "n"; then
                  echo "[INFO] ${targetdir}/$vendordir/framework/caffe has old version"\
                "custom.proto file. Do you want to replace? [y/n] "

                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        return 0
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        log "[ERROR] input error, please input again!"
                    fi
                done
            fi
        fi
        log "[INFO] replace old caffe.proto files ......"
    fi
    chmod -R +w "$targetdir/$vendordir/framework/caffe/" >/dev/null 2>&1
    cp -rf ${sourcedir}/$vendordir/custom.proto ${targetdir}/$vendordir/framework/caffe/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new custom.proto failed"
        return 1
    fi
	log "[INFO] copy custom.proto success"

    return 0
}

upgrade_file()
{
    if [ ! -e ${sourcedir}/$vendordir/$1 ]; then
        log "[INFO] no need to upgrade ops $1 file"
        return 0
    fi

    log "[INFO] copy new $1 files ......"
    cp -f ${sourcedir}/$vendordir/$1 $targetdir/$vendordir/$1
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 file failed"
        return 1
    fi

    return 0
}

delete_optiling_file()
{
  if [ ! -d ${targetdir}/vendors ];then
    log "[INFO] $1 not exist, no need to uninstall"
    return 0
  fi
  sys_info=$(uname -m)
  if [ ! -d ${sourcedir}/$vendordir/$1/ai_core/tbe/op_tiling/lib/linux/${sys_info} ];then
    rm -rf ${sourcedir}/$vendordir/$1/ai_core/tbe/op_tiling/liboptiling.so
  fi
  return 0
}

log "[INFO] copy uninstall sh success"

previous_opp="none"
if [ -d "${targetdir}/${vendordir}" ]; then
    previous_opp="${targetdir}/${vendordir}"
fi

if [ ! -d ${targetdir}/vendors ];then
        log "[INFO] create ${targetdir}/vendors."
        mkdir -p ${targetdir}/vendors
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/vendors failed"
            exit 1
        fi
fi
chmod u+w ${targetdir}/vendors

log "[INFO] upgrade framework"
upgrade framework
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op proto"
upgrade op_proto
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op impl"
delete_optiling_file op_impl
upgrade op_impl
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op api"
upgrade op_api
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade version.info"
upgrade_file version.info
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade ABI manifest"
upgrade_file abi_manifest.json
if [ $? -ne 0 ];then
    exit 1
fi

upgrade_proto
if [ $? -ne 0 ];then
    exit 1
fi

# set the set_env.bash
if [ -n "${INSTALL_PATH}" ] && [ -d ${INSTALL_PATH} ]; then
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}
    bin_path="${_ASCEND_CUSTOM_OPP_PATH}/bin"
    set_env_variable="#!/bin/bash\nexport ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH}"
    if [ ! -d ${bin_path} ]; then
        mkdir -p ${bin_path} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${bin_path} failed"
            exit 1
        fi
    fi
    echo -e ${set_env_variable} > ${bin_path}/set_env.bash
    if [ $? -ne 0 ]; then
        log "[ERROR] write ASCEND_CUSTOM_OPP_PATH to set_env.bash failed"
        exit 1
    else
        log "[INFO] fla_npu custom OPP installed to: ${_ASCEND_CUSTOM_OPP_PATH}"
        log "[INFO] using requirements: when custom module install finished or before you run the custom module, execute the command [ export ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-} ] to set the environment path"
        log "[WARNING] --install-path=${INSTALL_PATH} was specified; CANN will not auto-discover this custom OPP, please export ASCEND_CUSTOM_OPP_PATH as above and restart the process."
    fi
else
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}
    config_file=${targetdir}/vendors/config.ini
    if [ ! -f ${config_file} ]; then
        touch ${config_file}
        chmod 640 ${config_file}
        echo "load_priority=$vendor_name" > ${config_file}
        if [ $? -ne 0 ];then
            log "[ERROR] echo load_priority failed"
            exit 1
        fi
    else
        found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
        found_vendor=$(echo $found_vendors | sed "s/\<$vendor_name\>//g" | tr ',' ' ')
        vendor=$(echo $found_vendor | tr -s ' ' ',')
        if [ "$vendor" != "" ]; then
            sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
        fi
    fi
    if test $INSTALL_FOR_ALL = "y"; then
        chmod 755 ${config_file}
    fi
    log "[INFO] fla_npu custom OPP installed to: ${_ASCEND_CUSTOM_OPP_PATH}"
    log "[INFO] using requirements: when custom module install finished or before you run the custom module, execute the command [ export ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-} ] to set the environment path"
fi

if [ -d ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/ ]; then
    chmod -R 440 ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/* >/dev/null 2>&1
fi

echo "SUCCESS"
printf '[fla-npu] Installation successful: mode=CANN (external OPP; not wheel base/overlay), artifact=run\n'
printf '[fla-npu] previous OPP: %s\n' "${previous_opp}"
printf '[fla-npu] installed OPP: %s\n' "${targetdir}/${vendordir}"
exit 0
