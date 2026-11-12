# R1 变更验证记录（recurrent_kda / A5）

状态：本 MR 交付范围已完成（实现、case A 实测、精度与语义边界验证、原始产物归档）。
1.20x 逐 case 达标与全 case 正式验收属后续优化轮次，不在本 MR 验收范围。

## 1. 变更

仅 `arch35`（Ascend950PR）`recurrent_kda`：用 `NeedStore` 将逐 token 状态 GM 写回合并为
「末 token 必写、无 `ssm_state_indices` 只写末次、有 indices 仅在槽切换处写」；arch22 不变，
算术与归约顺序不变。设计说明见 [design.md](design.md) §14。

## 2. 环境与方法

| 项 | 值 |
| --- | --- |
| 设备 / 软件 | Ascend950PR（arch35），独占；CANN 9.1.0、torch 2.7.1 + torch_npu 2.7.1 |
| 对比 | 旧版 `9f3bcfcf` ↔ R1 新版，同一构建命令与私有 OPP 隔离目录 |
| 采集 | `torch_npu.profiler`（ACL），warmup=5 / active=10；计时区外构造输入，调用前后同步 |
| 口径 | `op_summary` 的 `Task Duration(us)` 均值，10 次采样（std <2%） |
| 用例 | `a5_decode_a`：B=1,T=8,H=8,HV=96,K=V=128,BSND；对照组 `chunk_kda_fwd`：B=1,T=2048,H=HV=32,chunk=64 |

## 3. 结果

| 指标 | 旧版 | 新版 | 加速比 |
| --- | ---: | ---: | ---: |
| `RecurrentKda` kernel duration | 27.71 us | 25.13 us | **1.10x** |
| `RecurrentKda` wall-clock | 57.97 us | 56.42 us | 1.03x |
| 对照 `chunk_kda_fwd` 三核合计 | 447.24 us | 446.70 us | 1.00x |

精度：`recurrent_kda` Pass(1/1)、`chunk_kda_fwd` Pass(48/48)。kernel 均值差 2.58 us，
合并标准误 0.199 us（t≈12.9），大于 2σ 阈值 0.94 us，收益可排除测量噪声；对照组在噪声内。

**达标结论**：1.20x 目标对应 kernel ≤23.092 us / wall ≤48.308 us，当前 1.10x **未达标**，
由后续优化轮次承接。「写回字节降 87.5%」为流量口径，不作为耗时或 B/C/D 可达性结论。

## 4. 语义与边界验证（5/5 PASS）

| Case | 验证维度 | 结果 |
| --- | --- | --- |
| A | out / final_state NPU vs CPU fp64 逐元素比对 | PASS |
| B | `inplace_final_state` True vs False | PASS，max_abs=0 |
| C | `ssm_state_indices` 槽切换 | PASS，out max_abs=0 |
| D | varlen 空段 | PASS |
| E | `output_final_state=False` | PASS，final_state=None |

容差 A：`atol=2.0, rtol=0.1`（bf16 累积误差）；B–E：`1e-2`。原始结果归档于 A5 测试环境
`archive_r1_iteration/`（24 文件 / 434K，含 CSV、复现脚本与对比报告）。

## 5. 交付范围

本 MR 已完成：R1 实现、case A 实测与精度、语义边界验证 5/5、Level1 管线采集与原始产物归档。
后续轮次：`storeStateEveryToken=1` 调用链接线与同二进制对照、旧版 Level1 收益归因、
B/C/D 全 case 基线与 1.20x 逐 case 达标；G2/G3/G6 属全 case 正式验收阶段，不阻塞本 MR。
