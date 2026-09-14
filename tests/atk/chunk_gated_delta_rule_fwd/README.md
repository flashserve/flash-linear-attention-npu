# chunk_gated_delta_rule_fwd ATK 验证

本目录验证公开接口 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。融合算子在 A2/A3
使用私有 `arch22` 实现，在 A5 使用私有 `arch35` 实现；A3 当前只具备注册和编译验收环境。

## 支持范围

- `q/k`：`[B,Hk,T,128]`，BF16/FP16。
- `v`：`[B,Hv,T,V]`，BF16/FP16，`V=128/256`，`Hv % Hk == 0`。
- `g/beta`：`[B,T,Hv]`；`g` 为 FP32，`beta` 与 q/k/v 同 dtype。
- `chunk_size`：64 或 128。
- 支持定长、变长、GVA、可选初始状态和可选最终状态。
- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。

## 精度标杆

精度使用 ATK 原生 `cv_fused_double_benchmark`：

1. NPU DUT：`chunk_gated_delta_rule_fwd`；
2. NPU benchmark：公开算子链 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`solve_tri`、
   `recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`；
3. CPU golden：相同冻结输入上的 FP64 recurrence。

公开 `solve_tri` 来自 main 已合入的 PR 398；融合 kernel 不引用该公开实现，只有双标杆链路调用它。
比较阈值为最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5。

## 用例

### A5 新路径双标杆

`atk_chunk_gated_delta_rule_fwd_new_path.json` 固定 336 条 BF16 用例，`K=V=128`、
`chunk_size=64`，覆盖 `Hv/Hk=1/2/3/4`（96/116/64/60 条）、定长、变长尾块、
初态和可选最终状态。输入范围为 Q/K/Beta `[-5,5]`、V `[-1024,1024]`、g `[-100,0]`。
专用执行器保留输入数值，只在调用 BSND 新路径和 GPU Triton 标杆时转换 Q/K/V
布局；比较 O 与可选 final_state。当前固定 `use_exp2=false`、
`use_qk_l2norm_in_kernel=false`、`disable_recompute=true`。

GPU 环境需加载与测试匹配的上游 FLA Triton 源码，从本用例目录启动服务：
先设置 `GPU_PORT`；NPU 侧设置 `GDN_GPU_HOST` 为可达的服务地址。

```bash
atk server --host 127.0.0.1 --port "$GPU_PORT" --devices 0 \
  --name gpu_reference --plugin_path ./executor_chunk_gated_delta_rule_fwd_new_path.py \
  --bind_cpu_type 1 --timeout 8000
```

在 NPU 环境加载当前源码的 Torch wrapper/OPP 与 CANN 后运行；GPU 地址可通过
SSH 转发提供。默认三路按固定 case seed 独立生成，需核对有效输入；
如需 ATK 跨节点同步，可设置 `GDN_ATK_SYNC_DATASET=1`：

```bash
GDN_GPU_PORT="$GPU_PORT" \
  bash scripts/run_new_path_double_benchmark.sh 0
```

ATK `summary` 和 `statistic` 应同时核对执行成功、精度通过和各输出详情。
`SpecialValueOnly` 表示特殊值检查通过但没有常规误差比例，不应计作非零数值覆盖。
新路径内存/确定性/性能可在同一 JSON 中选取代表用例：
运行内存检查前先设置 `MSS_LOG_PATH` 为可写的日志文件路径。

```bash
atk node --name npu_dut --backend npu --devices 0 task \
  -c ./atk_chunk_gated_delta_rule_fwd_new_path.json \
  -p ./executor_chunk_gated_delta_rule_fwd_new_path.py \
  --task accuracy_dc --dc_loop_nums 50 -wl '[0,21,24,27,51,63,67,75,78,81]'

touch "$MSS_LOG_PATH"
mssanitizer --tool=memcheck -- atk node --name npu_dut --backend npu --devices 0 task \
  -c ./atk_chunk_gated_delta_rule_fwd_new_path.json \
  -p ./executor_chunk_gated_delta_rule_fwd_new_path.py \
  --task run --mssanitizer -msl "$MSS_LOG_PATH" \
  -wl '[0,21,24,27,51,63,67,75,78,81]'

atk node --name npu_dut --backend npu --devices 0 task \
  -c ./atk_chunk_gated_delta_rule_fwd_new_path.json \
  -p ./executor_chunk_gated_delta_rule_fwd_new_path.py \
  --task performance_device -wl '[0,21,24,67,75]'
```

正式 device task 耗时结论还需用 `msprof` 的 `op_summary` 核对；不能把
ATK 精度通过视为内存、确定性或性能通过。上述范围只属于 A5 新路径，
不替代下述旧路径 500 条矩阵。

- `atk_chunk_gated_delta_rule_fwd.json`：既有泛化 500 条冻结矩阵，五种场景各 100 条，
  覆盖 BF16/FP16、MHA/GVA、V128/V256、chunk 64/128、定长/变长及状态组合。
- `scripts/cases/legacy500_adapted.json`：既有 BF16/MHA 历史 500 条回归矩阵，不作为默认入口。
- `atk_chunk_gated_delta_rule_fwd_perf.json`：A5 两条模型 case：
  - 推理：`B=1,Hk=16,Hv=32,T=11274,K=V=128,chunk=64`，变长并输出最终状态；
  - 训练：`B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64`，定长无状态输出。
- `atk_chunk_gated_delta_rule_fwd_mss.json`：从冻结矩阵抽取 6 条精简用例，覆盖 V128/V256、
  chunk 64/128、定长/变长、FP16/BF16 和状态输出。

A5 模型 shape 分别来源于 `推理model.csv` 和 `训练model.csv`，原文件 SHA256 为
`a8f21a5ddc23b824b2b5ccc625d33db95dd9441b33f2b0c0e3e313e72aeaa363` 与
`87e9bb1027c44eaf8cc2f5fc4d24256b22e0ff00c16162b5fb11aaaf24aca8ca`。

## TilingKey 覆盖

| TilingKey | 选择条件 | 普通/边界用例 | SoC | 实际选择证据 |
| --- | --- | --- | --- | --- |
| 1 | `V=128`，未进入 A5 推理模型特化 | MSS 0、2、4 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 2 | `V=256` | MSS 1、3、5 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 301 | A5 性能 case 0 的推理模式，初态为 BF16/FP32 | MSS 尚未覆盖 | A5 | 原模型 shape 已实测命中；MSS 待补 |

## 执行

先执行不依赖 NPU/ATK 的 ACLNN ABI 合同，确认公开参数顺序、ctypes 类型和默认路径映射：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

公开 `aclnnChunkGatedDeltaRuleFwd` 保留完整扩展 ABI。当前 Phase6 默认路径使用
`layout=BNSD`、`useExp2=false`、`allowNegEigval=false`、`stateVFirst=false`，且
`aLog/dtBias` 与扩展中间输出为空；`finalStateOutOptional` 是否为空决定是否输出 final state。
尚未实现的扩展组合会显式返回参数错误，不会静默忽略。

正式 500 条双标杆精度使用可恢复分片入口；默认每 25 条启动一个 fresh ATK 进程，避免
六算子 benchmark 长进程状态累积：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

冒烟或单分片可直接使用三节点入口（默认 `-mt 5`）：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

复跑历史矩阵：

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/scripts/cases/legacy500_adapted.json" \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

性能、确定性和内存检测仍使用仓内统一入口：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=mssanitizer
```

正式结论必须记录代码 commit、ATK/CANN 版本、SoC、实际加载的 OPP、case JSON 哈希和原始报告。
