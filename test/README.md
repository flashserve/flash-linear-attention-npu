# ATK 算子测试

本目录保存仓内 Ascend C 算子的 ATK 看护资产。ATK 是唯一测试执行入口；CT 只用于
`ct viz` 和 `ct dual analyze`。进入执行矩阵的用例最终只有成功或失败，环境、工具、
executor、结果文件或解析异常都按失败处理。

每个算子目录只维护四类文件：

```text
test/<op_name>/
|-- atk_<op_name>.json
|-- <op_name>.yaml
|-- gen_<op_name>.py
`-- executor_<op_name>.py
```

ATK 生成的 `result/`、`atk_output/`、日志、XLSX、profiling、sanitizer 输出和 Python
缓存不得提交。

## 工具版本

| 工具 | 锁定或已验证版本 |
| --- | --- |
| ATK | [AscendTest/ATK v26.7.8](https://gitcode.com/AscendTest/ATK/releases/v26.7.8)，tag `v26.7.8`，commit `34da785689a6ea8687479a0c4a0a5c48843bbcb6`，`atk --version` 为 `26.7.8`，Python 3.11 |
| ATK aarch64 wheel | `atk-26.7.8-cp311-cp311-linux_aarch64.whl`，SHA256 `2a06bd8132245af1717e9f3de18474150827058d6a5762e621db9fb2cc166f33` |
| ATK x86_64 wheel | `atk-26.7.8-cp311-cp311-linux_x86_64.whl`，SHA256 `fe4e2b23f2e95ecc41ef7b4a9368941a1c709a64c1633be3c31be2cc882990b9` |
| CT | 已验证 `ct-tool 0.9.1`；尚无可公开核验的正式获取地址和安装包校验值，发布验收前必须由测试团队补齐 |
| CANN | 已验证 `9.1.0.beta.1` |
| 驱动/固件 | 已验证驱动 `25.5.0`、固件 `7.8.0.5.216` |
| SOC | A2 `ascend910b` 已验证；A3 `ascend910_93` 和 A5 `ascend950` 以各算子索引和实际报告为准 |

ATK 升级必须单独评审并重跑代表性固定 case。ATK 和 CT 都只用于测试，不进入项目
运行时 wheel。

## 算子索引

| 算子 | 公开 API | route | SOC | 看护 case | 标杆 |
| --- | --- | --- | --- | --- | --- |
| [`chunk_kda_fwd`](./chunk_kda_fwd/) | `fla_npu.ops.ascendc.chunk_kda_fwd` | `ascendc`、`aclnn`、`direct_launch` | A2、A5 | 500：每个平台 200 正向、50 负向 | GPU Torch FP64 真值 + GPU Triton 同精度对照 |

`chunk_kda_fwd` 已验证的 A5 本地 DUT + GPU Docker 远端双标杆部署和逐步命令见
[`chunk_kda_fwd/README.md`](./chunk_kda_fwd/README.md)。

`chunk_kda_fwd` 的正向模型 profile 固定为 `B=1`、`H=HV=96`、`K=V=128`、
`T=8192/16384`、`chunk_size=64`。连续的偶数/奇数 case 分别覆盖 recompute 和
export，结构参数与随机输入完全相同。

| 平台/route | case ID |
| --- | --- |
| A2 `ascendc` 正向 | `0-167` |
| A2 `aclnn` 正向 | `168-183` |
| A2 `direct_launch` 正向 | `184-199` |
| A2 `aclnn` 负向 | `200-249` |
| A5 `ascendc` 正向 | `250-417` |
| A5 `aclnn` 正向 | `418-433` |
| A5 `direct_launch` 正向 | `434-449` |
| A5 `aclnn` 负向 | `450-499` |

专项 case：A2 性能 `0,16`、确定性 `4,18`、sanitizer `8,16`；A5 对应为
`250,266`、`254,268`、`258,266`。本轮 GPU 双标杆只覆盖 `chunk_size=64`；128 暂不进入
ATK 精度矩阵。当前冻结矩阵
未登记 `dirty_data` case，不能据此
声称已完成 NaN 脏数据验证。

受限 `direct_launch` ABI 接收已经计算好的 FP32 `gk` 和 FP32 `beta`。executor 在该
route 内计算 `gk`，并将已按模型输入量化的 BF16 `beta` 无损提升为 FP32；数值与其他
route 的模型输入保持一致。

## 环境准备

从官方 release 下载与主机架构、Python ABI 匹配的 wheel，先校验 SHA256，再安装到
独立虚拟环境：

```bash
sha256sum atk-26.7.8-cp311-cp311-linux_<arch>.whl
python3.11 -m venv <atk_venv>
source <atk_venv>/bin/activate
python -m pip install --upgrade pip
python -m pip install ./atk-26.7.8-cp311-cp311-linux_<arch>.whl
atk --version
```

加载 CANN、当前构建的 OPP 包和仓内 Python API：

```bash
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=<physical_device_id>
export PYTHONPATH=<repo_root>/torch_custom/fla_npu:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=<writable_cache_dir>

which atk
atk --version
npu-smi info
```

GPU 节点使用独立的 ATK 26.7.8 Python 3.11 环境，并确认 CUDA Torch、Triton 和待用的
KDA Triton 实现来自同一环境：

```bash
source <gpu_atk_venv>/bin/activate
export CUDA_VISIBLE_DEVICES=<physical_gpu_id>
export PYTHONPATH=<triton_kda_source_root>:${PYTHONPATH:-}
# 只有替换默认 fla-org 实现时才设置：
# export KDA_ATK_TRITON_CALLABLE=<python_module>:<chunk_kda_fwd_callable>

python -c 'import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())'
python -c 'from fla.ops.kda.chunk_fwd import chunk_kda_fwd; print(chunk_kda_fwd)'
```

`KDA_ATK_TRITON_CALLABLE` 的函数需接受 fla-org `chunk_kda_fwd` 的关键字参数，并按
`(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)` 返回 12 项。
executor 会把其 sequence-major 中间量归一成 NPU 公共接口的 head-major 输出，并在
recompute case 中按公共策略隐藏不应导出的张量。对照组输入保持原 BF16/FP16/FP32 dtype，
不允许在 callable 内改用 FP64。

未设置该变量时默认调用
`fla.ops.kda.chunk_fwd:chunk_kda_fwd`。项目对齐提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77` 的 fla-org CUDA Triton 实现只支持
`chunk_size=32/64`，满足本轮冻结矩阵。executor 对误传的 128 会直接报错，不会把 128
静默改写为 64。若 GPU 环境安装的就是该提交，可不设置 `KDA_ATK_TRITON_CALLABLE`。

## 远端节点服务

ATK 26.7.8 的[多机在线执行指南](https://gitcode.com/AscendTest/ATK/blob/main/ATK使用指南/任务执行.md#多机在线执行)
要求先在远端设备启动 server，并在发起端的 `node` 中填写该 server 可达的 IP 和端口。
GPU 在 Docker 内、任务从 A5 本机发起的已验证拓扑以
[`chunk_kda_fwd/README.md`](./chunk_kda_fwd/README.md) 为准。远端 GPU 不共享发起端文件系统时，
任务必须加 `--syc_dataset`；分布式执行不要加 `-sp`，使用 `-mt 1` 控制并发。
GPU 节点单独开一个终端：

```bash
source <gpu_atk_venv>/bin/activate
export CUDA_VISIBLE_DEVICES=<physical_gpu_id>
export PYTHONPATH=<triton_kda_source_root>:${PYTHONPATH:-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

atk server \
  --host 0.0.0.0 \
  --port <GPU_PORT> \
  --devices 0 \
  --name gpu_reference \
  --output_path ./atk_output/gpu_server \
  --plugin_path <repo_root>/test/chunk_kda_fwd/executor_chunk_kda_fwd.py \
  --timeout 8000
```

若测试命令不在 NPU 节点本机发起，NPU 节点也启动 server：

```bash
source <npu_atk_venv>/bin/activate
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=<physical_npu_id>
export PYTHONPATH=<repo_root>/torch_custom/fla_npu:${PYTHONPATH:-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

atk server \
  --host 0.0.0.0 \
  --port <NPU_PORT> \
  --devices 0 \
  --name npu_dut \
  --plugin_path <repo_root>/test/chunk_kda_fwd/executor_chunk_kda_fwd.py \
  --timeout 8000
```

容器内默认端口是 9090；若映射为 `9090 -> <HOST_PORT>`，后续 `node --port` 必须填写
`<HOST_PORT>`。官方常用拓扑是在 NPU 节点本机发起命令、仅 GPU 远端，此时 NPU node 去掉
`--host <NPU_IP> --port <NPU_PORT>` 即可。下面命令保留两端占位，适用于独立发起端连接两个
server 的部署。

构建与目标 SOC 匹配的 `chunk_kda_fwd` OPP 包：

```bash
bash build.sh --pkg --soc=<ascend910b_or_ascend950> \
  --vendor_name=fla_npu --ops=chunk_kda_fwd
```

执行 `direct_launch` case 前还要构建并安装仓内受限直调扩展：

```bash
cd examples/fast_kernel_launch_example
python3 -m pip install -r requirements.txt
NPU_ARCH=<ascend910b_or_ascend950> \
  FAST_KERNEL_OP_NAME=chunk_kda_fwd \
  python3 -m build --wheel -n
python3 -m pip install --force-reinstall --no-deps dist/ascend_ops-*.whl
```

## 生成与校验

```bash
cd test/chunk_kda_fwd
atk case \
  -f ./chunk_kda_fwd.yaml \
  -p ./gen_chunk_kda_fwd.py \
  -dt 1 \
  -en 0 \
  -s 20260810

python3 ./gen_chunk_kda_fwd.py \
  --output ./atk_chunk_kda_fwd.generated.json \
  --summary
```

相同 YAML、gen 和 seed 必须生成稳定 case ID 与稳定结构。生成后检查 ATK schema、
case 数量、SOC、route、layout、chunk 边界、可选输入、负向返回码和覆盖摘要，不要静默
覆盖已评审的 `atk_chunk_kda_fwd.json`。

## 精度与调用路径

ATK 26.7.8 会在 `--bm_device gpu` 指定的 GPU 节点上创建两种 executor 任务：benchmark
任务 `is_benchmark_task=true` 生成 Torch FP64 真值，普通 remote 任务
`is_benchmark_task=false` 运行 `KDA_ATK_TRITON_CALLABLE`。NPU 节点只运行被测 route。
executor 在三路都先用 CPU generator 和相同 seed 生成数值，再量化到用例声明的原始 dtype；
只有真值路会把量化后的所有浮点输入提升到 FP64，因此三路输入值严格同源。
GPU node 的 `--is_compare true` 是有意保留的：它让普通 GPU Triton 输出参与对比；
`--bm_device gpu` 会额外创建 FP64 真值任务。不要再加 `--is_bm true`，该参数不是这里的
双标杆任务角色开关。

下面保留 NPU/GPU 的远端 IP 和端口位置。`<NPU_PORT>`、`<GPU_PORT>` 是对应 ATK server
在节点注册时使用的端口，不是 SSH 端口。先用一条 `chunk_size=64` case 验证两端连通和
三种角色都实际执行：

```bash
export KDA_ATK_TRACE_SEED=1
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_role_probe \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
    -c ./atk_chunk_kda_fwd.json \
    --task accuracy \
    --bm_device gpu \
    -p ./executor_chunk_kda_fwd.py \
    -s 0 -e 1 \
    --save_data output \
    --syc_dataset \
    -mt 1 \
    -to 2000
unset KDA_ATK_TRACE_SEED
```

日志中同一 case/seed 应分别出现：GPU `benchmark=True high_precision=True`、GPU
`benchmark=False triton_control=True`、NPU `benchmark=False`。缺少任一角色都不能开始全量。

A2 全量正向：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_a2_accuracy \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task accuracy \
  --bm_device gpu \
  -p ./executor_chunk_kda_fwd.py \
  -s 0 -e 200 \
  --syc_dataset \
  -mt 1 \
  -to 2000
```

A5 将范围改为 `-s 250 -e 450`。三条 route 也可以使用上表中的 ID 组成白名单：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_route \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task accuracy \
  --bm_device gpu \
  -p ./executor_chunk_kda_fwd.py \
  -wl '[<route_case_ids>]' \
  --syc_dataset \
  -mt 1 \
  -to 2000
```

YAML 中的 `cv_fused_double_benchmark` 是唯一精度标准。benchmark GPU 任务用 PyTorch
小算子拼接出 FP64 真值，普通 remote GPU 任务运行同输入 dtype 的 Triton KDA；NPU、真值和
对照组使用同一组量化后输入。所有公开输出分别判定。shell 返回码为 0 但报告中存在 failed
case 时，整体仍为失败。

Triton remote 对照输出必须来自配置的同精度实现本身，不允许用 `nextafter`、混入 FP64
计算或其他后处理人为增大对照组误差。
长序列或近零真值导致相对误差比值不稳定时，必须保持原输入分布和精度标准，使用
`accuracy_lt` 与 `ct dual analyze` 复检；复检未通过时继续定位实现或标杆语义。

executor 在向 ATK 返回公开浮点输出前统一提升为 FP32，使 BF16 DUT、同精度 Triton
基线和 FP64 高精度真值满足比较器的 dtype 一致要求；提升只发生在算法计算完成后，不改变
真值和对照组的计算 dtype。

单 case 定位必须保存三端输出：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_single \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task accuracy \
  --bm_device gpu \
  -p ./executor_chunk_kda_fwd.py \
  -s <case_id> -e <case_id_plus_one> \
  --save_data output \
  --syc_dataset \
  -mt 1 \
  -to 2000
```

## 负向用例

A2：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task run \
  -p ./executor_chunk_kda_fwd.py \
  -s 200 -e 250 \
  -sp \
  -to 2000
```

A5 将范围改为 `-s 450 -e 500`。ATK 26.7.8 的 `CaseConfig` 没有独立的
`expected_return_code` 字段，因此返回码保存在内嵌 `case_spec` 中；executor 先从异常中解析
实际 `aclnnStatus`，与该值及关键错误文本逐条精确比较，再抛出与 `expected_error_msg` 匹配的
预期异常。必须检查总任务数、执行成功/失败数和每条 case 结果，不能只看 shell 返回码。

## 确定性

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task accuracy_dc \
  --bm_device gpu \
  -p ./executor_chunk_kda_fwd.py \
  -wl '[4,18]' \
  --syc_dataset \
  -mt 1 \
  -to 2000
```

A5 使用 `-wl '[254,268]'`。检查全部公开输出，并保留首个不一致轮次、输出与位置。

## 精度失败与复检

任何精度失败先执行单 case `--save_data output`，再用 CT 查看 DUT、同精度 Triton 基线和
GPU Torch FP64 真值：

```bash
ct viz <npu_output> <torch_fp64_gpu_output>
ct viz <triton_same_precision_gpu_output> <torch_fp64_gpu_output>
```

若差异呈结构性分布，回到 kernel 的读取、索引、搬运、计算、同步或写回定位根因，修复后
重跑平台全量。只有确认属于非结构性数值误差时才运行 50 轮复检：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_recheck \
  node --name gpu_reference --backend gpu \
    --host <GPU_IP> --port <GPU_PORT> --devices 0 \
    --is_compare true \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task accuracy_lt \
  --bm_device gpu \
  -p ./executor_chunk_kda_fwd.py \
  -wl '[<failed_case_ids>]' \
  --loop_nums 50 \
  --disable_id_seed \
  --syc_dataset \
  -mt 1 \
  -to 8000

ct dual analyze <atk_recheck_result.xlsx>
```

`accuracy_lt` 不加 `-sp`。H=96 FP64 真值使用 `-mt 1`，避免多个大张量任务同时占用 GPU
显存；`-to 8000` 为 ATK 50 轮输出汇集和 post-process 保留足够等待窗口。复检只随机化输入
数值，shape、dtype、layout、SOC、route、
属性、indices 和可选输入状态保持不变；不得修改输入 range、case 结构或精度标准。
CT 0.9.1 在聚合统计失败时仍可能返回退出码 0，因此还必须读回摘要，确认总用例、有效样本、
失败数和通过数符合预期，不能只看 shell 返回码。

只有完整 accuracy 已检查全部公开输出，并且保存数据和可视化已确认唯一待复检输出时，
才可设置 `KDA_ATK_VISIBLE_OUTPUTS=<output_name>` 做定向 50 轮复检，以避免保存无关的大张量。
多个输出用逗号分隔。该变量不得用于首次精度测试、全量通过统计或隐藏其他失败输出；不设置
时 executor 返回当前 case 的全部公开输出。

## 性能

性能只使用 ATK `performance_device` 集成的 device profiler，不使用 Python wall time：

```bash
atk node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 \
    -o ./atk_output_performance \
  task \
  -c ./atk_chunk_kda_fwd.json \
  --task performance_device \
  -p ./executor_chunk_kda_fwd.py \
  -wl '[0,16]' \
  --performance_data 20,100,80 \
  --save_data profile \
  -sp \
  -to 2000
```

A5 使用 `-wl '[250,266]'`。报告必须实际采集到目标 kernel，包含非空 device 数据、
case ID、SOC、route、采集次数和 device 耗时。基线和回退阈值需要另行评审，失败时不得
自动覆盖基线。

## mssanitizer

先使用 sanitizer 选项构建 debug 包，并确认 opc 命令包含
`--op_debug_level=1 --op_debug_config=dump_cce,sanitizer`。执行前抽查目标对象：

```bash
nm <chunk_kda_fwd_object> | grep sanitizer
```

以 memcheck 为例：

```bash
ATK_BIN=$(command -v atk)
mssanitizer --tool=memcheck --log-file ./mssanitizer_memcheck.log -- \
  "$ATK_BIN" node --name npu_dut --backend npu \
    --host <NPU_IP> --port <NPU_PORT> --devices 0 task \
  -c ./atk_chunk_kda_fwd.json \
  --task run \
  -p ./executor_chunk_kda_fwd.py \
  -wl '[8,16]' \
  -sp \
  -to 2000
```

其余三类只替换工具与日志名：

```bash
mssanitizer --tool=racecheck --log-file ./mssanitizer_racecheck.log -- <atk_command>
mssanitizer --tool=initcheck --log-file ./mssanitizer_initcheck.log -- <atk_command>
mssanitizer --tool=synccheck --log-file ./mssanitizer_synccheck.log -- <atk_command>
```

A5 使用 `-wl '[258,266]'`。这里由外层 `mssanitizer --tool=<tool>` 选择检查类型，不要再给
同一条 ATK 命令叠加 `-ms` 形成两层 sanitizer。日志必须出现
`Start <tool> sanitizer on kernel ...` 或锁定
版本的等价命中信息；只有 `No active sanitizer tool on kernel ...` 时本次测试失败。还要
检查 ATK 总任务数和每条 case 结果，并区分真实问题与工具对手工流水的保守报告。

## 结果判读

每次执行至少记录 ATK 版本、代码 commit、case JSON 摘要、SOC、route、逻辑设备号、
总用例数、成功数、失败数和专项结论。优先处理 `ERROR`，再分析精度 `FAIL`。公开 PR、
issue、评论和测试总结只写测试项与结果，不包含账号、机器、地址、绝对路径、token、日志
路径或其他环境细节。
