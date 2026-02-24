path=/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/result/cpu_for_test
source /data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
# conda activate clx
compi=$1
compi_y="all"

if [ "$compi" = "$compi_y" ]; then
    python3 test.py regen #标杆生成pt
    python3 /data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/pre_handle.py ${path} # pt -> bin
    # cd /root/data_nvme0n1/huangjunzhe/GDN/code/old/
    bash run.sh compile  ##重新编译并运行/root/data_nvme0n1/huangjunzhe/GDN/target/test_aclnn_gdn.cpp
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/to_py.py ${path} # bin -> pt
echo "ct single ${path}/gen/dg_npu.pt ${path}/dg_cpu.pt --calc_count 1000000 --dtype float16"
ct single ${path}/gen/dw_npu.pt ${path}/gen/dw_cpu_ht.pt --calc_count 1000000 --dtype float16
ct single ${path}/gen/dg_npu.pt ${path}/gen/dg_cpu_ht.pt --calc_count 1000000 --dtype float32
ct single ${path}/gen/dq_npu.pt ${path}/gen/dq_cpu_ht.pt --calc_count 1000000 --dtype float16
ct single ${path}/gen/dk_npu.pt ${path}/gen/dk_cpu_ht.pt --calc_count 1000000 --dtype float16

ct viz ${path}/gen/dw_npu.pt ${path}/gen/dw_cpu_ht.pt
ct viz ${path}/gen/dg_npu.pt ${path}/gen/dg_cpu_ht.pt
ct viz ${path}/gen/dq_npu.pt ${path}/gen/dq_cpu_ht.pt
ct viz ${path}/gen/dk_npu.pt ${path}/gen/dk_cpu_ht.pt