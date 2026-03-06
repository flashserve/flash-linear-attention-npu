# source /root/data_nvme0n1/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
ascend_path="/data/zs/run/Ascend/cann-9.0.0/"
# ascend_path="/data/huangjunzhe/Ascend/cann-9.0.0"

ascend_path_orig=${ascend_path}/../
custom_path="/data/huangjunzhe/GDN/custom"
code_path="/data/huangjunzhe/GDN/ops-transformer_GDN"
source ${ascend_path}/set_env.sh

compi=$1
compi_y="compile"

test_script_path=/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests

##path=/root/data_nvme0n1/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/result/case_01
data_path=$2
if [ -n "$data_path" ]; then
    code_path=${test_script_path}/../../../
    example_path=${test_script_path}/../examples/
    custom_path=${test_script_path}/../../../../custom
fi
echo "[run.sh] code_path: ${code_path}"
echo "[run.sh] example_path: ${example_path}"
echo "[run.sh] custom_path: ${custom_path}"
alias log='export ASCEND_SLOG_PRINT_TO_STDOUT=1; export ASCEND_GLOBAL_LOG_LEVEL=0'
alias unlog='unset ASCEND_SLOG_PRINT_TO_STDOUT; unset ASCEND_GLOBAL_LOG_LEVEL'


if [ "$compi" = "$compi_y" ]; then
    unset ASCEND_SLOG_PRINT_TO_STDOUT; unset ASCEND_GLOBAL_LOG_LEVEL
    # rm /root/data_nvme0n1/huangjunzhe/GDN/target/test_gdn
    # echo "/root/data_nvme0n1/huangjunzhe/GDN/target/test_gdn deleted!"
    # export TMPDIR=/root/data_nvme0n1/huangjunzhe/tmp
    cd ${code_path}
    bash build.sh --pkg --ops=chunk_bwd_dqkwg
    #bash build.sh --pkg --ops=prepare_wy_repr_bwd_full
    if [ $? -ne 0 ]; then
        exit 1
    fi
    bash ${code_path}/build/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
    if [ $? -ne 0 ]; then
        exit 1
    fi
    # clear
fi

export LD_LIBRARY_PATH=${ascend_path_orig}/cann/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

source ${custom_path}/vendors/custom_transformer/bin/set_env.bash
cd ${example_path}
g++ -std=c++17 -g test_chunk_bwd_dqkwg.cpp -L${ascend_path_orig}/ascend-toolkit/latest/lib64 -lascendcl -lcust_opapi -lnnopbase -L${custom_path}/vendors/custom_transformer/op_api/lib/  -I${custom_path}/vendors/custom_transformer/op_api/include -I${ascend_path}/aarch64-linux/include/ -I${ascend_path}/x86_64-linux/include/ -I${ascend_path}/x86_64-linux/include/aclnnop/ -o test_gdn
#g++ -std=c++17 -g /root/data_nvme0n1/huangjunzhe/GDN/code/old/ops-transformer_GDN_2/chunk_gated_delta_rule/prepare_wy_repr_bwd_full/examples/test_aclnn_chunk_bwd_dv_local.cpp -L/root/data_nvme0n1/huangjunzhe/Ascend/ascend-toolkit/latest/lib64 -lascendcl -lcust_opapi -lnnopbase -L/root/data_nvme0n1/huangjunzhe/GDN/code/custom/vendors/custom_transformer/op_api/lib/  -I/root/data_nvme0n1/huangjunzhe/GDN/code/custom/vendors/custom_transformer/op_api/include -I/root/data_nvme0n1/huangjunzhe/Ascend/cann-9.0.0/aarch64-linux/include/ -I/root/data_nvme0n1/huangjunzhe/Ascend/cann-8.5.0/x86_64-linux/include/ -I/root/data_nvme0n1/huangjunzhe/Ascend/cann-8.5.0/x86_64-linux/include/aclnnop/ -o test_gdn

#g++ -std=c++17 -g test_gmms.cpp -L/data/huangjunzhe/Ascend/ascend-toolkit/latest/lib64 -lascendcl -lopapi -lnnopbase -L/data/huangjunzhe/Ascend/ascend-toolkit/8.1.RC1/opp/vendors/customize/op_api/lib/  -I/data/huangjunzhe/Ascend/ascend-toolkit/latest/include/ -o test_gmms
chmod +x test_gdn
LD_LIBRARY_PATH=${custom_path}/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
./test_gdn $2


# conda activate gdn_py39
# export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# python3 /root/data_nvme0n1/huangjunzhe/GDN/target/result/to_pt.py /root/data_nvme0n1/huangjunzhe/GDN/target/result/cpu_model