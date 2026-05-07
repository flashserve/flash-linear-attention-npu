#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 需要用户配置的 #######################################################################################################
# 代码仓目录（绝对路径）
CODE_PATH=$1
if [ ! -d $CODE_PATH ];then
    echo "[ERROR] \"$CODE_PATH\" 目录不存在!"
    exit 1
fi

# LOG目录（绝对路径）
LOG_PATH=$CODE_PATH/log_ut

# 打印的Ascend LOG等级
export ASCEND_GLOBAL_LOG_LEVEL=3

# 用例超时时间
TIMEOUT=30m

# 要跑的测试类型
TEST_LIST=""
TEST_LIST+="host "
TEST_LIST+="api "
TEST_LIST+="kernel "

# 要跑的算子仓
OP_REPO_LIST=""
OP_REPO_LIST+="moe "
OP_REPO_LIST+="attention "
OP_REPO_LIST+="gmm "
OP_REPO_LIST+="ffn "
OP_REPO_LIST+="posembedding "
OP_REPO_LIST+="mc2 "
OP_REPO_LIST+="mhc "

# 读取算子option配置 ################################################################################################
op_config_yaml="$CODE_PATH/tests/test_config.yaml"
op_temp_dir="$CODE_PATH/log_ut/op_content/"
function get_op_option()
{
    op_name=$1
    op_content_file=$op_temp_dir$op_name".txt"
    sed -n "/^  $op_name:/,/^$/p" $op_config_yaml > $op_content_file

    #cat $op_name.txt #打印算子配置内容
    op_option="option"
    option_list=`sed -n "/$op_option/,/^$/p" $op_content_file | grep -v '#' | grep ' - ' | awk -F'-' '{print $2}'`
    op_option_content=""
    for op in ${option_list[@]};
    do
        if [ -z "$op_option_content" ]
        then
            op_option_content+=$op
        else
            op_option_content+=","
            op_option_content+=$op
        fi
        #echo $op #打印option配置内容
    done

    if [ -z "$op_option_content" ]
    then
        echo $op_name
    else
        echo $op_option_content
    fi
}

# 批跑测试 #############################################################################################################
# 创建log目录，构建csv表头
mkdir -p $LOG_PATH
mkdir -p $op_temp_dir
echo -n "repo,op,fwk," &> $LOG_PATH/results.csv
mkdir -p $LOG_PATH/op_test
echo -n "op_test," &>> $LOG_PATH/results.csv

echo "" &>> $LOG_PATH/results.csv

export ASCEND_SLOG_PRINT_TO_STDOUT=1
for OP_REPO in $OP_REPO_LIST; do
    printf "%-50s" "[$OP_REPO]" | tr ' ' '-'
    printf "%-50s" "[op_test]" | tr ' ' '-'
    echo ""

    cd $CODE_PATH/$OP_REPO
    OP_LIST=$(ls -d */ | sed 's/\///g' | grep -v "3rd" | grep -v "common")" "
    for OP in $OP_LIST; do
        echo -n "$OP_REPO,$OP," &>> $LOG_PATH/results.csv
        printf "%-50s" $OP

        op_option_list=$(get_op_option $OP)
        if [ -z "$op_option_list" ]
        then
            echo "$OP_REPO $OP could not find option!" &>> $LOG_PATH/ut_error.log
            echo -ne "\033[33mNA\033[0m"
            echo ""
            continue
        else
            echo "$OP_REPO $OP option is:"$op_option_list &>> $LOG_PATH/ut.log
        fi

        cd $CODE_PATH
        timeout $TIMEOUT bash build.sh -u --ops=$op_option_list --cov --soc=ascend310p,ascend910b,ascend910_93,ascend950 &> $LOG_PATH/op_test/$OP.log
        if [ $? -ne 0 ]; then
            echo -ne "\033[31mFAIL\033[0m            "
            echo -n "FAIL," &>> $LOG_PATH/results.csv
        else
            echo -ne "\033[32mPASS\033[0m            "
            echo -n "PASS," &>> $LOG_PATH/results.csv
        fi

        if [ -f build/cov_result/coverage.info ]; then
            echo "$OP ut coverage file info is:" >> $LOG_PATH/ut.log
            ls -sh build/cov_result/coverage.info &>> $LOG_PATH/ut.log
            cat build/cov_result/coverage.info >> ./coverage.info
        else
            echo "$OP ut do not generate!" >> $LOG_PATH/ut.log
        fi
        
        echo ""
        echo "" &>> $LOG_PATH/results.csv
    done
done

# 删除当前编译文件，以免影响调用逻辑
rm -r $CODE_PATH/build/cov_result
