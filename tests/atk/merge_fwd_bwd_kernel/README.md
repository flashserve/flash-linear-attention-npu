# MergeFwdBwdKernel ATK 工程

Phase A：显式输入 YAML、ATK 关系约束生成器、CPU/NPU executor 和逻辑候选 JSON。
入口为 `fla_npu.ops.ascendc.npu_merge_fwd_bwd_kernel`；CPU 标杆在
`scripts/merge_fwd_bwd_kernel_cpu.py`，以 FP64 按源码文档递推，最终写回原 dtype 的 `h`。
本次未执行 NPU 精度、性能、确定性或内存验收。

## 接口和约束来源

源码根目录：`fla/ops/ascendc/cp/merge_fwd_bwd_kernel/`。

| 约束 | C++ 来源（相对源码根目录） | YAML / generator 映射 |
| --- | --- | --- |
| 输入顺序 `h, ag_hm, pre_or_post_num_ranks, rank, forward, state_v_first` | `op_host/op_api/aclnn_merge_fwd_bwd_kernel.cpp` 的 `GetWorkspaceSize` | 显式同名输入；两个 bool 均覆盖 |
| `ag_hm=[S,HV,128,256]`, `1<=S<=1024`, `1<=HV<=256` | `op_host/op_tiling/merge_fwd_bwd_kernel_tiling.cpp` | YAML 离散小规模候选，generator 定点修正范围与末两维 |
| `h=[HV,128,128]`，与 ag_hm 同 FP32/BF16 | 同上；`op_host/merge_fwd_bwd_kernel_def.cpp` | 从 ag_hm 对齐 h shape/dtype |
| `1<=N<=S`，`0<=rank<S`；forward:rank>=N；backward:rank+N<S | tiling 中 `unitWorld` 后的分支校验 | S=1 固定 N=1/rank=0；其余修正到对应方向合法区间 |
| h 连续、公有格式、原地写入 | `op_host/op_api/aclnn_merge_fwd_bwd_kernel.cpp` | ATK 默认连续 ND 张量；CPU 保持 alias |

返回单个 `h`，shape、dtype 和存储均与输入 h 一致；`state_v_first` 只交换末两轴物理布局。
CPU 使用 `docs/design.md`：forward `src(i)=rank-N+i`，backward `src(i)=rank+N-i`，
`h=He_0; h=M_i@h+He_i`。常规随机值域和分布来自本次读取的 ATK 精度标准：
均匀 [-5,5] 与正态（mean [-5,5]、std [0.1,2]）各 50%。
当前候选覆盖小 world/head、单步/多步、两个方向和两个 state layout；不是全部上限组合。
维度必须正且固定 rank，因此不生成非法 empty/scalar；Inf/NaN 可通过 `-en` 额外生成。
索引/workspace 按源码使用 int64/uint64；当前小规模矩阵不声称覆盖 INT32 溢出。

## 生成与执行

在已安装 ATK 的环境运行：

```bash
cd tests/atk/merge_fwd_bwd_kernel
timeout 60 atk case -f merge_fwd_bwd_kernel.yaml -p gen_merge_fwd_bwd_kernel.py -dt 1 -en 0 -s 20261030
timeout 60 atk case -f merge_fwd_bwd_kernel.yaml -p gen_merge_fwd_bwd_kernel.py -s 20261030
atk node -b cpu task -c atk_merge_fwd_bwd_kernel.json -p executor_merge_fwd_bwd_kernel.py -s 0 -e 1 -tk accuracy
```

生成输出为 `result/merge_fwd_bwd_kernel/json/all_merge_fwd_bwd_kernel.json`；
筛选并检查后保存为 `atk_merge_fwd_bwd_kernel.json`。
准备 NPU 环境后可用统一入口 `bash tests/atk/run_test_cpu.sh -op=merge_fwd_bwd_kernel -scope=accuracy`。
未提供用户模型性能 case，因此不制造 `_perf.json`；未实际验证 TilingKey，不制造 `_mss.json` 或正式验收结论。

## Phase A 验证记录（2026-10-30）

使用隔离 CPU 容器中的 ATK 26.9.8 / PyTorch 2.7.1+cpu。
`atk case -dt 1 -en 0 -s 20261030` dry-run 通过；正式生成 128 条；按用户要求归档时仅保留首条候选，共 1 条，`default_seed=20261030`。
候选为 ATK 实际生成，归档只做首条截取、重编号和固定执行种子，不合成替代 JSON。
正式生成报告无超大 shape，但去重前重复率 10.94% 导致 Case Metrics.result=false；
去重后的候选满足全部 dtype/shape/方向/rank/count 约束。
`scripts/check_shapes.py` 的 24 组手算/布局/inplace 检查通过。
验证命令：`python scripts/check_shapes.py`。CPU shape 检查不代表 NPU 精度验收。

ATK CPU 单条实际执行：`success 1, failed 0`（仅 CPU 执行，不含 NPU 对比）。
正式生成后运行 `python scripts/archive_cases.py` 可重现种子固定、静态去重和编号归档。
