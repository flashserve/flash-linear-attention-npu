# ChunkKdaBwd ATK 工程

**Phase A 形状占位工程：CPU 全零输出仅用于验证返回结构、shape 和 dtype，不能作为精度标杆。**
NPU 路径真实调用 `fla_npu.ops.ascendc.npu_chunk_kda_bwd`，显式固定 `implementation="legacy"`，
避免 A5 V2 与原版的 dtype 契约差异。未执行 NPU 精度或其它正式验收。

## 输入、输出与来源

源码根为 `fla/ops/ascendc/kda/chunk_kda_bwd/`；另核对
`torch_custom/fla_npu/csrc/src/stable_chunk_kda_bwd.cpp` 和公开 `_stable.py` wrapper。

| 约束 | C++ 来源（相对源码根目录） | YAML / generator / executor |
| --- | --- | --- |
| dense `[B,H,T,D]`，packed `[H,T,D]`，B/H/T>0，K=V=128 | `op_host/op_api/aclnn_chunk_kda_bwd.cpp`；`chunk_kda_bwd_c_tiling_processor.h` | YAML 离散正值，generator 对齐所有关联输入，packed 去掉 B |
| q/k/v/d_o、Aqk/Akk、w/qg/kg/v_new/h 是 BF16，gk 是 FP32 | ACLNN A2/A3 `MatchesTensor`；C tiling dtype 校验 | BF16 为跨平台公共子集；beta 覆盖 BF16/FP32，raw_g 覆盖 BF16/FP32 |
| Aqk/Akk 尾64，beta 为 Token 去末维；h 为 NT-first `[B,Nc,H,128,128]` / `[Nc,H,128,128]` | ACLNN `matrix/scalar/state` 与 A/C tiling processor | generator 从 q、C、规范元数据推导，禁止将 h 写成 head-first |
| cu_seqlens/chunk_indices 成对，起0止T且 sequence-major | ACLNN `nc` 和 packed 校验循环 | packed 构造 `[0,T//2,T]` 和每序列的规范 chunk 对，允许空首序列 |
| C=64，safe/use_gate/disable_recompute/use_exp2=true，state_v_first=false | ACLNN `CHECK_COND` 与 wrapper | YAML 固定当前已支持配置，保留所有 saved cache |
| raw_g 同 Token shape，A_log=[H]，dt_bias=[H,128] FP32；lower_bound∈[-5,0) | ACLNN `MatchesTensor`；C processor；公开 wrapper | YAML 声明值域；generator 对齐 shape；with_bias 控制可选 bias |
| 不支持 initial_state/dht，q/k rstd 成对且 legacy A5 不支持 | ACLNN / stable adapter | 本矩阵固定这些可选参数为 None（公开 wrapper 默认值），不生成伪支持场景 |

固定 legacy 返回顺序：

| 输出 | shape | dtype |
| --- | --- | --- |
| dq / dk | q / k 的 head-major shape | FP32 |
| dv | v.shape | BF16 |
| db | beta.shape | FP32 |
| dg | q.shape | FP32 |
| dh0 | None | — |
| dA | `[H]` | FP32 |
| dbias | 有 bias 时 `[H,128]`，否则 None | FP32 |

输出 dtype 直接依据 stable adapter 原版路径的 `allocate_sizes(..., kFloat, ...)`；
它与独立 Finalize 的 token-major db/dg 不同，也与 V2 的 BF16 dq/dk 不同。
CPU 入口为 `scripts/chunk_kda_bwd_cpu.py`。执行器保留 ATK 浮点数据，只构造元数据并移除辅助 attrs。

## 用例范围与运行

YAML 离散覆盖短序列、32/64/128 chunk 附近、dense/packed、bias 有无、beta/raw_g dtype 和 scale。
均匀 [-5,5] / 正态（mean [-5,5], std [0.1,2]）各50%，来自本次读取的精度标准；
这一未校准值域与零占位均不能用于声称精度达标。正 rank/正维是当前契约，未构造非法空 tensor/scalar。
当前小规模参数空间不声称覆盖全上限或 INT32 溢出；workspace 由源码的 uint64/int64 及 overflow 检查保护。

```bash
cd tests/atk/chunk_kda_bwd
timeout 60 atk case -f chunk_kda_bwd.yaml -p gen_chunk_kda_bwd.py -dt 1 -en 0 -s 20261030
timeout 60 atk case -f chunk_kda_bwd.yaml -p gen_chunk_kda_bwd.py -s 20261030
atk node -b cpu task -c atk_chunk_kda_bwd.json -p executor_chunk_kda_bwd.py -s 0 -e 1 -tk accuracy
```

生成位置 `result/chunk_kda_bwd/json/all_chunk_kda_bwd.json`，检查后固化为 `atk_chunk_kda_bwd.json`。
替换真实 CPU 标杆并校准值域后再运行统一 `-scope=accuracy`。
本次不制造用户未提供的模型性能 `_perf.json` 或未经实测的 TilingKey `_mss.json`。

## Phase A 验证记录（2026-10-30）

使用隔离 CPU 容器中的 ATK 26.9.8 / PyTorch 2.7.1+cpu。
`atk case -dt 1 -en 0 -s 20261030` dry-run 通过；正式生成 192 条；按用户要求归档时仅保留首条候选，共 1 条，`default_seed=20261030`。
候选为 ATK 实际生成，归档只做首条截取、重编号和固定执行种子，不合成替代 JSON。
正式生成报告无超大 shape，Case Metrics.result=true；归档按要求仅保留首条候选。
全部192条原始候选均通过现有公开 wrapper 的参数准备校验，
`scripts/check_shapes.py` 的80组 dense/packed/bias/dtype/返回结构检查通过。
验证命令：`python scripts/check_shapes.py`。CPU shape 检查不代表 NPU 精度验收。

ATK CPU 单条实际执行：`success 1, failed 0`（仅 CPU 执行，不含 NPU 对比）。
正式生成后运行 `python scripts/archive_cases.py` 可重现种子固定、静态去重和编号归档。
