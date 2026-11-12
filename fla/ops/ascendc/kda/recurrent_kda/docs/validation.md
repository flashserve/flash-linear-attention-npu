# R1 状态写回合并实验记录（recurrent_kda / A5 · 历史版本）

> 归档记录，非待合入的代码变更。本实验基于 fork 上 `9f3bcfcf` 的旧 arch35 实现；
> 主仓 PR #1096 已在新的 class-based engine 中等价实现无 indices 场景的状态写回合并。
> 下文数据**不代表当前 main 的性能**，也不作为当前实现的验证结论。

## 1. 实验变更

旧实现中 decode 常态 `shouldStoreState_` 恒真，每个 token 都把整块 state 从 UB 写回 GM；
而 state 在 UB 内原地递推、不回读 GM，外部只能观察到每个 state 槽的最后一次写入。
R1 用 `NeedStore` 门控写回：末 token 必写；无 `ssm_state_indices` 时只写末次；
有 indices 时仅在槽切换处写。不改算术与归约顺序。

## 2. 环境与方法

| 项 | 值 |
| --- | --- |
| 设备 / 软件 | Ascend950PR（arch35），独占；CANN 9.1.0、torch 2.7.1 + torch_npu 2.7.1 |
| 对比 | 旧版 `9f3bcfcf` ↔ R1 新版，同一构建命令与私有 OPP 隔离目录 |
| 采集 | `torch_npu.profiler`（ACL），warmup=5 / active=10；计时区外构造输入，调用前后同步 |
| 口径 | `op_summary` 的 `Task Duration(us)` 均值，10 次采样（std <2%） |
| 用例 | `a5_decode_a`：B=1,T=8,H=8,HV=96,K=V=128,BSND；对照 `chunk_kda_fwd`：B=1,T=2048,H=HV=32,chunk=64 |

## 3. 结果

| 指标 | 旧版 | R1 | 加速比 |
| --- | ---: | ---: | ---: |
| `RecurrentKda` kernel duration | 27.71 us | 25.13 us | **1.10x** |
| `RecurrentKda` wall-clock | 57.97 us | 56.42 us | 1.03x |
| 对照 `chunk_kda_fwd` 三核合计 | 447.24 us | 446.70 us | 1.00x |

精度 Pass(1/1)（对照组 `chunk_kda_fwd` 48/48）；语义边界验证 5/5 PASS：out / final_state
逐元素比对、`inplace_final_state` 位级一致、`ssm_state_indices` 槽切换、varlen 空段、
`output_final_state=False`。kernel 均值差 2.58 us、合并标准误 0.199 us（t≈12.9），
大于 2σ 阈值 0.94 us；对照组在噪声内。

1.20x 目标对应 kernel ≤23.092 us / wall ≤48.308 us，本实验 1.10x **未达标**；
「写回字节降 87.5%」为流量口径，不作为耗时或全 case 可达性结论。

## 4. 与主仓现状的关系

- PR #1096 的 `RecurrentKdaEngine::Process()` 已实现：仅当 `inplace_final_state || output_final_state`
  时写回 GM，且无 `ssm_state_indices` 时只在请求末 token 写回 —— 与本实验的主要收益点等价。
- 有 indices 时逐 token 写回是 API 契约（每个有效 token 对应槽位都要写，见
  [design.md](design.md)「目标与结构」）；上游另有 `OrderStateSnapshotWrites` 与 `PipeBarrier<PIPE_MTE3>()`
  保证同槽覆盖顺序。本实验的「仅槽切换处写」不满足该契约，故不向当前实现移植。
- 原始 profiling CSV、复现脚本与结果保存在实验环境，未纳入 PR（与上游验证矩阵口径一致）。

结论：R1 不再作为待合入的代码变更，仅保留实测结论与设计对照，避免后续重复评估。
