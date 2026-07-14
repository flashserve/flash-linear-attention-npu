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

QUIET="n"
INSTALL_FOR_ALL="n"
WHEEL_INSTALL="n"


while true
do
    case $1 in
    --quiet)
        QUIET="y"
        shift
    ;;
    --full|--install)
        WHEEL_INSTALL="y"
        shift
    ;;
    --install-path=*)
        INSTALL_PATH=$(echo $1 | cut -d"=" -f2-)
        INSTALL_PATH=${INSTALL_PATH%*/}
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
        log "[ERROR] python is required to locate the installed fla_npu wheel OPP."
        exit 1
    fi

    "${python_bin}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("fla_npu")
if spec is None or spec.origin is None:
    raise SystemExit("fla_npu is not importable. Install flash-linear-attention-npu wheel first.")

package_dir = Path(spec.origin).resolve().parent
print(package_dir / "opp")
PY
}

copy_wheel_file() {
    local src_file="$1"
    local dst_file="$2"
    mkdir -p "$(dirname "${dst_file}")"
    cp -a "${src_file}" "${dst_file}"
}

to_snake_name() {
    echo "$1" | sed -E 's/([A-Z]+)([A-Z][a-z])/\1_\2/g; s/([a-z0-9])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]'
}

support_status_tag() {
    local status="$1"

    case "${status}" in
        warning)
            printf 'WARNING'
            ;;
        notice)
            printf 'NOTICE'
            ;;
        ok)
            printf 'OK'
            ;;
        *)
            printf 'INFO'
            ;;
    esac
}

collect_vendor_op_names() {
    local vendor_dir="$1"
    local abi_dir="${vendor_dir}/op_api/include/aclnnop"
    local kernel_root="${vendor_dir}/op_impl/ai_core/tbe/kernel"

    (
        if [ -d "${abi_dir}" ]; then
            while IFS= read -r header_file; do
                local file_name
                file_name=$(basename "${header_file}")
                case "${file_name}" in
                    aclnn_ops_transformer*.h)
                        continue
                        ;;
                    aclnn_*.h)
                        echo "${file_name}" | sed -E 's/^aclnn_(.*)\.h$/\1/'
                        ;;
                esac
            done < <(find "${abi_dir}" -type f -name 'aclnn_*.h' | sort)
        fi

        if [ -d "${kernel_root}" ]; then
            while IFS= read -r src_op_dir; do
                local rel_dir="${src_op_dir#${kernel_root}/}"
                case "${rel_dir}" in
                    config/*)
                        continue
                        ;;
                esac
                to_snake_name "$(basename "${src_op_dir}")"
            done < <(find "${kernel_root}" -mindepth 2 -maxdepth 2 -type d | sort)
        fi
    ) | sort -u
}

collect_op_aclnn_abi_changes() {
    local src_vendor="$1"
    local dst_vendor="$2"
    local op_name="$3"
    local src_abi_dir="${src_vendor}/op_api/include/aclnnop"
    local dst_abi_dir="${dst_vendor}/op_api/include/aclnnop"
    local rel_file

    for rel_file in "aclnn_${op_name}.h" "level2/aclnn_${op_name}.h"; do
        if [ -f "${src_abi_dir}/${rel_file}" ] && [ ! -f "${dst_abi_dir}/${rel_file}" ]; then
            echo "ADDED ${rel_file}"
        elif [ -f "${src_abi_dir}/${rel_file}" ] && [ -f "${dst_abi_dir}/${rel_file}" ] && ! cmp -s "${src_abi_dir}/${rel_file}" "${dst_abi_dir}/${rel_file}"; then
            echo "MODIFIED ${rel_file}"
        elif [ ! -f "${src_abi_dir}/${rel_file}" ] && [ -f "${dst_abi_dir}/${rel_file}" ]; then
            echo "DELETED ${rel_file}"
        fi
    done
}

op_has_source_aclnn_header() {
    local src_vendor="$1"
    local op_name="$2"
    local src_abi_dir="${src_vendor}/op_api/include/aclnnop"

    [ -f "${src_abi_dir}/aclnn_${op_name}.h" ] || [ -f "${src_abi_dir}/level2/aclnn_${op_name}.h" ]
}

format_change_summary() {
    local change_file="$1"
    local summary=""
    local line

    while IFS= read -r line; do
        if [ -z "${summary}" ]; then
            summary="${line}"
        else
            summary="${summary}; ${line}"
        fi
    done <"${change_file}"
    echo "${summary}"
}

log_op_status() {
    local level="$1"
    local status="$2"
    local op_name="$3"
    local reason="$4"
    local tag

    tag=$(support_status_tag "${status}")
    log "[${level}]   $(printf '%-7s %-36s - %s' "${tag}" "${op_name}" "${reason}")"
}

log_op_scope_file() {
    local title="$1"
    local op_file="$2"
    local level="$3"
    local op_name

    log "[${level}] ${title}"
    if [ ! -s "${op_file}" ]; then
        log "[${level}]   (none detected)"
        return
    fi

    while IFS= read -r op_name; do
        log "[${level}]   ${op_name}"
    done <"${op_file}"
}

log_included_op_status() {
    local src_vendor="$1"
    local dst_vendor="$2"
    local op_name="$3"
    local changes_file
    local summary

    changes_file=$(mktemp)
    collect_op_aclnn_abi_changes "${src_vendor}" "${dst_vendor}" "${op_name}" >"${changes_file}"
    summary=$(format_change_summary "${changes_file}")

    if grep -qE '^(MODIFIED|DELETED) ' "${changes_file}"; then
        log_op_status "WARNING" "warning" "${op_name}" "unsupported after install because aclnn ABI changed: ${summary}"
    elif grep -q '^ADDED ' "${changes_file}"; then
        log_op_status "WARNING" "notice" "${op_name}" "included in run package, but aclnn ABI is new in the installed wheel: ${summary}"
    elif ! op_has_source_aclnn_header "${src_vendor}" "${op_name}"; then
        log_op_status "WARNING" "notice" "${op_name}" "included in run package, but no aclnn ABI header was found to compare"
    else
        log_op_status "INFO" "ok" "${op_name}" "included in run package and aclnn ABI is unchanged"
    fi

    rm -f "${changes_file}"
}

confirm_partial_shared_lib_impact() {
    local src_vendor="$1"
    local dst_vendor="$2"
    local src_ops_file
    local dst_ops_file
    local unavailable_ops_file
    local op_name

    src_ops_file=$(mktemp)
    dst_ops_file=$(mktemp)
    unavailable_ops_file=$(mktemp)

    collect_vendor_op_names "${src_vendor}" >"${src_ops_file}"
    collect_vendor_op_names "${dst_vendor}" >"${dst_ops_file}"
    comm -23 "${dst_ops_file}" "${src_ops_file}" >"${unavailable_ops_file}"

    log_op_scope_file "This run package contains operators:" "${src_ops_file}" "INFO"
    log "[INFO] Operator support status after installing this run package:"
    log "[INFO]   $(printf '%-7s %s' "$(support_status_tag "warning")" "unsupported after install")"
    log "[INFO]   $(printf '%-7s %s' "$(support_status_tag "notice")" "requires manual attention")"
    log "[INFO]   $(printf '%-7s %s' "$(support_status_tag "ok")" "supported after install")"

    while IFS= read -r op_name; do
        log_included_op_status "${src_vendor}" "${dst_vendor}" "${op_name}"
    done <"${src_ops_file}"

    if [ -s "${unavailable_ops_file}" ]; then
        log "[WARNING] Installing this run package replaces shared opapi/tiling/proto libraries in the wheel with the scoped build from this package."
        log "[WARNING] The following installed operators are not included in this run package and will not be usable after replacement:"
        while IFS= read -r op_name; do
            log_op_status "WARNING" "warning" "${op_name}" "not included in this run package; shared opapi/tiling/proto libraries will be replaced"
        done <"${unavailable_ops_file}"
    fi

    rm -f "${src_ops_file}" "${dst_ops_file}" "${unavailable_ops_file}"

    if [ "${QUIET}" = "y" ]; then
        log "[WARNING] --quiet is set, treating scoped run package replacement impact as confirmed."
        return
    fi

    log "[INFO] Continue to replace the installed wheel OPP with this scoped run package? [y/n]"
    while true; do
        read yn
        if [ "${yn}" = "y" ]; then
            return
        elif [ "${yn}" = "n" ]; then
            log "[INFO] Exit without installing run package."
            exit 0
        else
            log "[WARNING] Input error, please input y or n to choose!"
        fi
    done
}

remove_deleted_aclnn_headers() {
    local src_vendor="$1"
    local dst_vendor="$2"
    local src_abi_dir="${src_vendor}/op_api/include/aclnnop"
    local dst_abi_dir="${dst_vendor}/op_api/include/aclnnop"
    local op_name
    local rel_file

    if [ ! -d "${src_abi_dir}" ] || [ ! -d "${dst_abi_dir}" ]; then
        return
    fi

    while IFS= read -r op_name; do
        for rel_file in "aclnn_${op_name}.h" "level2/aclnn_${op_name}.h"; do
            if [ -f "${dst_abi_dir}/${rel_file}" ] && [ ! -f "${src_abi_dir}/${rel_file}" ]; then
                rm -f "${dst_abi_dir}/${rel_file}"
            fi
        done
    done < <(collect_vendor_op_names "${src_vendor}")
}

merge_vendor_to_wheel_opp() {
    local src_vendor="$1"
    local dst_vendor="$2"
    local src_kernel_root="${src_vendor}/op_impl/ai_core/tbe/kernel"
    local dst_kernel_root="${dst_vendor}/op_impl/ai_core/tbe/kernel"
    local src_op_dir
    local rel_dir
    local dst_op_dir

    mkdir -p "${dst_vendor}"

    if [ -d "${src_kernel_root}" ]; then
        while IFS= read -r src_op_dir; do
            rel_dir="${src_op_dir#${src_kernel_root}/}"
            case "${rel_dir}" in
                config/*)
                    continue
                    ;;
            esac
            dst_op_dir="${dst_kernel_root}/${rel_dir}"
            if [ -d "${dst_op_dir}" ]; then
                rm -rf "${dst_op_dir}"
            fi
        done < <(find "${src_kernel_root}" -mindepth 2 -maxdepth 2 -type d)
    fi

    remove_deleted_aclnn_headers "${src_vendor}" "${dst_vendor}"

    if command -v rsync >/dev/null 2>&1; then
        rsync -a --checksum "${src_vendor}/" "${dst_vendor}/"
    else
        cp -a "${src_vendor}/." "${dst_vendor}/"
    fi

    if [ -f "${dst_vendor}/op_api/lib/libcust_opapi.so" ]; then
        copy_wheel_file "${dst_vendor}/op_api/lib/libcust_opapi.so" "${dst_vendor}/op_api/lib/libopapi.so"
    fi
}

update_wheel_vendors_config() {
    local vendors_root="$1"
    local config_file="${vendors_root}/config.ini"
    local existing=""
    local merged="${vendor_name}"
    local item

    mkdir -p "${vendors_root}"
    if [ -f "${config_file}" ]; then
        existing=$(grep -w "load_priority" "${config_file}" | tail -n 1 | cut --only-delimited -d"=" -f2-)
    fi

    IFS=',' read -ra vendor_items <<< "${existing}"
    for item in "${vendor_items[@]}"; do
        item=$(echo "${item}" | xargs)
        if [ -n "${item}" ] && [ "${item}" != "${vendor_name}" ]; then
            merged="${merged},${item}"
        fi
    done
    echo "load_priority=${merged}" >"${config_file}"
}

install_wheel_opp_package() {
    local src_vendor="${sourcedir}/${vendordir}"
    local wheel_opp_root
    local dst_vendor

    if [ ! -d "${src_vendor}" ]; then
        log "[ERROR] The run package does not contain ${src_vendor}."
        exit 1
    fi

    wheel_opp_root=$(get_wheel_opp_root)
    wheel_opp_root="${wheel_opp_root%/}"
    dst_vendor="${wheel_opp_root}/${vendordir}"

    if [ ! -d "${dst_vendor}" ]; then
        log "[ERROR] Target wheel OPP vendor not found: ${dst_vendor}. Install the full flash-linear-attention-npu wheel first."
        exit 1
    fi

    log "[INFO] Merge FLA NPU run package into installed wheel OPP root: ${wheel_opp_root}"
    log "[INFO] Source vendor ${src_vendor}"
    log "[INFO] Target vendor ${dst_vendor}"

    confirm_partial_shared_lib_impact "${src_vendor}" "${dst_vendor}"
    merge_vendor_to_wheel_opp "${src_vendor}" "${dst_vendor}"
    update_wheel_vendors_config "${wheel_opp_root}/vendors"

    log "[INFO] FLA NPU wheel OPP update completed. Restart Python processes to load the new libcust_opapi.so and kernels."
    echo "SUCCESS"
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
                echo -n "${file_b} "
                has_same_file=0
            fi
        done
        if [ 0 -eq $has_same_file ]; then
            echo
            if test $QUIET = "n"; then
                echo "[INFO]: has old version in ${targetdir}/$vendordir/$1, \
                you want to Overlay Installation , please enter:[o]; \
                or replace directory installation , please enter: [r]; \
                or not install , please enter:[n]."

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

upgrade_proto
if [ $? -ne 0 ];then
    exit 1
fi

# set the set_env.bash
if [ -n "${INSTALL_PATH}" ] && [ -d ${INSTALL_PATH} ]; then
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}
    bin_path="${_ASCEND_CUSTOM_OPP_PATH}/bin"
    set_env_variable="#!/bin/bash\nexport ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:\${ASCEND_CUSTOM_OPP_PATH}\nexport LD_LIBRARY_PATH=${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:\${LD_LIBRARY_PATH}"
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
        log "[INFO] using requirements: when custom module install finished or before you run the custom module, \
        execute the command [ source ${bin_path}/set_env.bash ] to set the environment path"
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
    log "[INFO] using requirements: when custom module install finished or before you run the custom module, \
        execute the command [ export LD_LIBRARY_PATH=${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:\${LD_LIBRARY_PATH} ] to set the environment path"
fi

if [ -d ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/ ]; then
    chmod -R 440 ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/* >/dev/null 2>&1
fi

echo "SUCCESS"
exit 0
