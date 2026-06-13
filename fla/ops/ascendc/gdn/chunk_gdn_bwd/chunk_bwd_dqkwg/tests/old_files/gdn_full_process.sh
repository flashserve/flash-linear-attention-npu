casefolder=$2
dtype=$3
gtype=$4
if [ -z "$casefolder" ]; then
    casefolder="case_02"
fi
if [ -z "$dtype" ]; then
    dtype="bf16"
fi
if [ -z "$gtype" ]; then
    gtype=${dtype}
fi
echo "[full] casefolder ${casefolder}, dtype ${dtype}, gtype ${gtype}"

path=/data/huangjunzhe/GDN/result/result/${casefolder}
test_script_path=/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests
# source /root/data_nvme0n1/huangjunzhe/Ascend.open/cann-8.5.0-beta.1/set_env.sh
# conda activate clx
compi=$1
compi_y="all"
compi_y2="compile"
compi_y3="run"

if [ "$compi" = "$compi_y" ]; then
    mkdir ${path}
    mkdir ${path}/gen
    echo "[full] GENDIR maked: ${path}/gen"
    python3 test.py regen ${path} ${casefolder} #标杆生成pt
    # python3 test.py noregen ${path} # 从GPU输入读取内容,生成cpu结果
    python3 ${test_script_path}/pre_handle.py ${path} ${dtype} ${gtype} # pt -> bin
    bash run.sh nocompile ${path} ##重新编译并运行/root/data_nvme0n1/huangjunzhe/GDN/target/test_aclnn_gdn.cpp
fi

if [ "$compi" = "$compi_y2" ]; then
    bash run.sh compile ${path}
fi

if [ "$compi" = "$compi_y3" ]; then
    bash run.sh nocompile ${path}
fi

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 ${test_script_path}/to_py.py ${path} # bin -> pt
echo "ct single ${path}/gen/dg_npu.pt ${path}/dg_cpu.pt --calc_count 1000000 --dtype xxx"
ct single ${path}/gen/dw_npu.pt ${path}/gen/dw_cpu_ht.pt --calc_count 1000000 --dtype ${dtype}
ct single ${path}/gen/dg_npu.pt ${path}/gen/dg_cpu_ht.pt --calc_count 1000000 --dtype ${gtype}
ct single ${path}/gen/dq_npu.pt ${path}/gen/dq_cpu_ht.pt --calc_count 1000000 --dtype ${dtype}
ct single ${path}/gen/dk_npu.pt ${path}/gen/dk_cpu_ht.pt --calc_count 1000000 --dtype ${dtype}
ct viz ${path}/gen/dw_npu.pt ${path}/gen/dw_cpu_ht.pt --out_dir ${path} --name dw
ct viz ${path}/gen/dg_npu.pt ${path}/gen/dg_cpu_ht.pt --out_dir ${path} --name dg
ct viz ${path}/gen/dq_npu.pt ${path}/gen/dq_cpu_ht.pt --out_dir ${path} --name dq
ct viz ${path}/gen/dk_npu.pt ${path}/gen/dk_cpu_ht.pt --out_dir ${path} --name dk

# ct dual ${path}/gen/dw_npu.pt ${path}/gen/dw_gpu_ht.pt ${path}/gen/dw_cpu_ht.pt --dtype ${dtype}
# ct dual ${path}/gen/dg_npu.pt ${path}/gen/dg_gpu_ht.pt ${path}/gen/dg_cpu_ht.pt --dtype ${gtype}
# ct dual ${path}/gen/dq_npu.pt ${path}/gen/dq_gpu_ht.pt ${path}/gen/dq_cpu_ht.pt --dtype ${dtype}
# ct dual ${path}/gen/dk_npu.pt ${path}/gen/dk_gpu_ht.pt ${path}/gen/dk_cpu_ht.pt --dtype ${dtype}

# md5sum ${path}/gen/dw_npu.pt
# md5sum ${path}/gen/dg_npu.pt
# md5sum ${path}/gen/dq_npu.pt
# md5sum ${path}/gen/dk_npu.pt

