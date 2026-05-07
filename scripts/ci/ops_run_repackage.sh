#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

TOP_DIR=""
PKG_PATH=""
RUN_PKG_SAVE_PATH=""
OS_ARCH=$(uname -m)
PKG_NAME=""
SOC=""

# -----------------------------
# 函数定义
# -----------------------------

parse_args() {
    local arg

    # 循环遍历所有命令行参数
    for arg in "$@"; do

        # 使用 case 语句匹配参数格式
        case "$arg" in
            # 匹配 --soc=... 格式
            --soc=*)
                # 截取等号后面的值，赋值给全局变量 SOC
                SOC="${arg#*=}"
                ;;
            # 匹配 --top-dir=... 格式
            --top_dir=*)
                TOP_DIR="${arg#*=}"
                ;;
            # 匹配 --pkg-path=... 格式
            --pkg_path=*)
                PKG_PATH="${arg#*=}"
                ;;
            --run_pkg_save_path=*)
                RUN_PKG_SAVE_PATH="${arg#*=}"
                ;;
            --pkg_name=*)
                PKG_NAME="${arg#*=}"
                ;;
            # 处理未知参数
            *)
                echo "ERROR: Unknown param $arg" >&2
                ;;
        esac
    done
}


# 打印日志
log() {
    echo "$*"
}

# 错误退出
die() {
    log "ERROR: $*"
    exit 1
}

# 确保目录存在
ensure_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log "Creating directory: $dir"
        mkdir -p "$dir" || die "Failed to create directory $dir"
    fi
}

# 确保文件存在
ensure_file() {
    local file="$1"
    [[ -f "$file" ]] || die "Required file not found: $file"
}

# -----------------------------
# 主流程开始
# -----------------------------

log "Starting ops packaging process..."
parse_args "$@"

WORKDIR=${TOP_DIR}/${PKG_PATH}         # 当前工作目录
TEMP_RUN_DIR="./run_files"              # 临时存放拷贝的 .run 文件
HOST_RUN_NAME="host.run"
HOST_EXTRACT_DIR="host"
MAKESELF_TARGET_DIR="build/makeself"
RUNFILE_TARGET_DIR="build/_CPack_Packages/makeself_staging"
PACKAGE_SCRIPT="${WORKDIR}/scripts/package/package.py"
MERGE_SCRIPT="${WORKDIR}/scripts/package/common/py/merge_binary_info_config.py"
PKG_OUTPUT_DIR="build/_CPack_Packages/makeself_staging"
RUN_PACKAGE_SAVE_AB_PATH=${TOP_DIR}/${RUN_PKG_SAVE_PATH}
ARCHIVE_RUN_DIR="${TOP_DIR}/vendor/hisi/build/delivery/${SOC}/"
MERGE_OPS_SCRIPT="${WORKDIR}/scripts/package/common/py/json_merger.py"


cd ${WORKDIR}/ || exit

# 1. 创建临时目录存放 .run 文件
ensure_dir "$TEMP_RUN_DIR"
cd "$TEMP_RUN_DIR"
log "Working in temporary directory: $(pwd)"

# 2. 拷贝 xx/kernel 下所有 .run 文件（重命名防重名）
counter=1
find "$RUN_PACKAGE_SAVE_AB_PATH" -name "cann-*custom_operator_group*.run" -type f | while read -r runfile; do
     # 获取父级目录名（basename of dirname）
    parent_dir=$(basename "$(dirname "$runfile")")

    #basename_orig=$(basename "$runfile" .run)
    new_name="kernel_${parent_dir}_${counter}.run"
    cp -v "$runfile" "./$new_name" || die "Failed to copy $runfile"
    log "Copied $runfile -> $new_name"
    ((counter++))
done

# 拷贝makeself目录至build目录下，不存在build目录则创建
ensure_dir "../$MAKESELF_TARGET_DIR"
cp -rf "$TOP_DIR"/open_source/makeself/* "../$MAKESELF_TARGET_DIR" || die "Failed to copy $MAKESELF_TARGET_DIR"

# 检查是否有 kernel run 文件
[[ -n "$(find . -name 'kernel_*.run' -type f 2>/dev/null)" ]] || \
    die "No .run files found in $RUN_PACKAGE_SAVE_AB_PATH"

# 3. 拷贝 host/cann.run 为 host.run, 通常生成的原始host包名中没有custom
cd $RUN_PACKAGE_SAVE_AB_PATH
host_file_name=$(find . -type f -name "*.run" | grep -v "custom")
host_file_name="${host_file_name#./}"
cd "${WORKDIR}"
cd "${TEMP_RUN_DIR}"
host_run_src="$RUN_PACKAGE_SAVE_AB_PATH/$host_file_name"
if [ -e $host_run_src ]; then

    cp -v $host_run_src "./$HOST_RUN_NAME" || die "Failed to copy host run file"
    log "Copied host run file: $host_run_src -> $HOST_RUN_NAME"
else
    log "Host run file not found: $host_run_src"
fi

# 4. 解压 host.run
if [[ ! -x "./$HOST_RUN_NAME" ]]; then
    chmod +x "./$HOST_RUN_NAME" || die "Cannot make $HOST_RUN_NAME executable"
fi

log "Extracting $HOST_RUN_NAME to ./$HOST_EXTRACT_DIR"
"./$HOST_RUN_NAME" --extract="$HOST_EXTRACT_DIR" --noexec || \
    die "Failed to extract $HOST_RUN_NAME"

# 移除 host.run
rm -f "./$HOST_RUN_NAME"
log "Removed $HOST_RUN_NAME after extraction"

ensure_dir "./$HOST_EXTRACT_DIR"

# 5. 遍历其余 .run 文件（kernel_* 开头）
first_run=true
shopt -s nullglob  # 避免 glob 无匹配时报错
for runfile in kernel_*.run; do
    [[ -f "$runfile" ]] || continue

    if [[ ! -x "$runfile" ]]; then
        chmod +x "$runfile" || die "Cannot make $runfile executable"
    fi

    # 提取基础文件名（不含 .run）
    base_name="${runfile%.run}"
    extract_dir="./$base_name"

    log "Processing $runfile -> extracting to $extract_dir"
    "./$runfile" --extract="$extract_dir" --noexec || \
        die "Failed to extract $runfile"

    PARENT_DIR="${extract_dir}/packages/vendors"
    full_path=$(find "$PARENT_DIR" -maxdepth 1 -type d -name "custom_*_transformer" | head -n 1)
    if [ -n "$full_path" ]; then
    # 4. 提取目录名 (这就是你要的 custom_operator_group_3_transformer)
        kernel_dir_name=$(basename "$full_path")

        echo "kernel_dir_name is  $target_dir_name"
    else
        echo "Not find kernel_dir"
        continue
    fi
    # 检查解压后目录结构
    kernel_src_dir=$extract_dir/packages/vendors/$kernel_dir_name/op_impl/ai_core/tbe/kernel/${SOC}
    config_src_dir=$extract_dir/packages/vendors/$kernel_dir_name/op_impl/ai_core/tbe/kernel/config/${SOC}

    [[ -d $kernel_src_dir ]] || log "Kernel source dir not found: "$kernel_src_dir
    [[ -d $config_src_dir ]] || log "Config source dir not found: "$config_src_dir

    if [[ ! -d $kernel_src_dir || ! -d "$config_src_dir" ]]; then
        continue  # 跳过本次循环的后续步骤，进入下一次循环
    fi

    # 获取 $ascend 子目录名（假设只有一个）
    #ascend_dir=$(find $kernel_src_dir -mindepth 1 -maxdepth 1 -type d | head -n1 | xargs basename)
    ascend_dir=$SOC

    if $first_run; then
        # 第一个文件：拷贝整个 kernel 目录
        target_kernel_dir="$HOST_EXTRACT_DIR/${PKG_NAME}/built-in/op_impl/ai_core/tbe/kernel/$SOC/$PKG_NAME/"
        ensure_dir $target_kernel_dir
        cp -rf "$kernel_src_dir"/* "./$target_kernel_dir" || \
            die "Failed to copy first kernel files"
        log "First run: copied full kernel to $target_kernel_dir"
        dest_conf_ascend_first="$HOST_EXTRACT_DIR/${PKG_NAME}/built-in/op_impl/ai_core/tbe/kernel/config/$SOC/ops_transformer"
        ensure_dir $dest_conf_ascend_first
        # config文件拷贝
        cp -v $config_src_dir/* $dest_conf_ascend_first/
        first_run=false
    else
        # 非第一个文件：增量合并

        # a. 拷贝 kernel/$ascend/* 到 host/.../kernel/$ascend/
        dest_kern_ascend="$HOST_EXTRACT_DIR/${PKG_NAME}/built-in/op_impl/ai_core/tbe/kernel/$SOC/$PKG_NAME"
        ensure_dir "$dest_kern_ascend"
        cp -rf "$kernel_src_dir"/* "$dest_kern_ascend"/ || \
            die "Failed to copy kernel ascend files"

        # b. 处理 config/$ascend/ 下的 JSON 文件
        src_conf_ascend=$config_src_dir
        dest_conf_ascend="$HOST_EXTRACT_DIR/${PKG_NAME}/built-in/op_impl/ai_core/tbe/kernel/config/$SOC/ops_transformer"

        # 遍历所有 JSON 文件
        for json_file in "$src_conf_ascend"/*.json; do
            [[ -f "$json_file" ]] || continue
            json_basename=$(basename "$json_file")

            if [[ "$json_basename" == "binary_info_config.json" ]]; then
                target_json="$dest_conf_ascend/binary_info_config.json"
                #jq -s 'add' $json_file $target_json > temp.json && mv -f temp.json $target_json
                log "Executing merge_binary_info_config.py to merge final package..."
                python3 "$MERGE_SCRIPT" \
                    --base-file=$json_file \
                    --update-file=$target_json \
                    --output-file=binary_info_config.json
                # 覆盖HOST中的config文件
                mv -f binary_info_config.json $target_json
            elif [[ "$json_basename" == "relocatable_kernel_info_config.json" && -f "$dest_conf_ascend/relocatable_kernel_info_config.json" ]]; then
                target_json="$dest_conf_ascend/relocatable_kernel_info_config.json"
                #jq -s 'add' $json_file $target_json > temp.json && mv -f temp.json $target_json
                log "Executing to merge relocatable_kernel_info_config.json"
                python3 "$MERGE_SCRIPT" \
                    --base-file=$json_file \
                    --update-file=$target_json \
                    --output-file=relocatable_kernel_info_config.json
                mv -f relocatable_kernel_info_config.json $target_json
            else
                target_json="$dest_conf_ascend/$json_basename"
                if [[ -f "$target_json" ]]; then
                    python3 "$MERGE_OPS_SCRIPT" \
                        --base-file=$target_json \
                        --update-file=$json_file \
                        --output-file=$target_json
                else
                    cp -v "$json_file" "$dest_conf_ascend"/ || \
                        die "Warning: failed to copy $json_file"
                fi
            fi
        done
    fi

    # 可选：清理解压目录（节省空间）
    # rm -rf "$extract_dir"
done

filelist_src_path="$HOST_EXTRACT_DIR/share/info/ops_transformer/script/filelist.csv"
rm $filelist_src_path

# 6. 拷贝 host/ 到 makeself 目录
ensure_dir "../$RUNFILE_TARGET_DIR"
cp -rf "$HOST_EXTRACT_DIR"/* "../$RUNFILE_TARGET_DIR"/ || \
    die "Failed to copy host content to makeself directory"

log "Host content copied to $RUNFILE_TARGET_DIR"

# 7. 执行打包脚本
cd "../" || echo "Failed to go back to workdir"

# 执行 package.py
log "Executing package.py to generate final package..."
python3 "$PACKAGE_SCRIPT" \
    --pkg_name "$PKG_NAME" \
    --makeself_dir "${WORKDIR}/$MAKESELF_TARGET_DIR" \
    --pkg-output-dir "${WORKDIR}/$PKG_OUTPUT_DIR" \
    --independent_pkg \
    --chip_name "$SOC" \
    --os_arch linux-"$OS_ARCH"

log "Packaging completed successfully!"


# 8. 归档全量构建算子编译包至hdfs目录
ensure_dir "$ARCHIVE_RUN_DIR"
cp -r "$PKG_OUTPUT_DIR"/*.run $ARCHIVE_RUN_DIR