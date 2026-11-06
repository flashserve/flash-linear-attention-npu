# 2026-11-06：仅保留 RWKV8 的 experimental 验证

本轮移除 recompute 样例、其专用公共算法头、Python 包装和 ATK 工程。当前真实实验算子
只有 PR #446 的 `fused_recurrent_rwkv8`。多算子发现/依赖/注册继续用临时夹具验证，
未注册算子拒绝用不存在的测试名称验证。experimental 不支持 overlay。

RWKV8 的 host、kernel、adapter、CPU 标杆、输入生成器和原 ATK JSON 均未修改。
实验 Python 去除样例专用的 CPU 整数数组参数分支，保留同设备检查及现有调用方式。
`--ops=fused_recurrent_rwkv8` 与 `--ops=all` 当前选择同一算子集合，不宣称真实多算子执行覆盖。

## 环境与源码

- A2：`z00879345@179.178.1.2`，Ascend 910B2 / aarch64；装载隔离和正式对照使用卡 2，
  RWKV8 原 ATK 数值回归使用卡 1（卡 2 异常见下文）。
- 工作根：`/home/z00879345/experimental-rwkv-only-20261106-gTlGAi`。
- CANN 9.0.0、conda `zzy-cann900`、Python 3.11、torch/torch_npu 2.12.0、GCC 11。
- ATK 26.9.8，在独立 venv 固定 Kombu 5.6.2；测试环境和结果均新建。
- 构建输入为 `8df36bf7` 上的清理工作区，`source_dirty=true`；归档 Git tree 为
  `7bcafcd6d451ee7f860cc9d0420a74741fc21869`。
- 输入文件 SHA256 清单摘要：`ed40f8371eccf8324d76f5c892be3a3ed6e654cfa6a266007a2c277efecfe2a2`。
- 构建后仅补充文档；按统一换行符比较，最终代码和测试与归档输入一致。

## 本轮结果

| 检查 | 结果与范围 |
| --- | --- |
| CI 单元测试 | 298/298，无跳过 |
| 实验 Python 契约 | 21/21，无跳过 |
| 单算子构建 | `--ops=fused_recurrent_rwkv8` 的 `.run`、空目录安装、审计、wheel 构建通过 |
| 全量实验构建 | `--ops=all` 同样通过；两个清单均只有 RWKV8，common required 为空，wheel 无 recompute 文件 |
| 装载隔离 | L01–L28：28/28；使用本轮单算子 wheel，正式目录内容哈希不变 |
| Python 公开入口 | 仅导出 `fused_recurrent_rwkv8`；未注册名称按预期拒绝 |
| RWKV8 原 ATK，卡 1 | 本轮单算子 wheel，case `[0,3)`；默认队列 3/3、inline 3/3，原 executor/JSON/有限值检查与 `mixed_tolerance_bm` 均未修改 |
| 正式算子生命周期，卡 2 | `chunk_fwd_o` 安装实验包前、共存、卸载实验包后，各 2/2；正式包内容哈希不变 |

运行时留存 dispatcher、实际加载的库路径和 host 库 SHA256。`--ops=all` 的制品完成构建审计；
本轮正式计数的数值和装载回归使用单算子 wheel，不把两个选项等价解释为两个 wheel 均已完整跑过。

## 卡 2 异常与交叉定位

首次在卡 2 运行原 ATK 时，3 条用例均被 NPU 有限值检查报为 NaN/Inf；旧的单算子、
双算子 wheel，以及旧虚拟环境也复现，不能将这些尝试记为通过。
从失败用例保存的六个真实输入及 RWKV8 输出在 CPU 检查中均为有限值。
不导入任何 fla 包的独立检查进一步复现：卡 2 上 `torch.isfinite` 对全 1 张量
`[1,4,4096,64]` 也返回部分 false；同样检查在空闲卡 1 全部为 true。
该对照将异常缩小到当前卡 2 的基础运行路径，尚未确定底层原因，未重置设备或修改系统环境。

卡 2 另用私有 ATK 副本，将 `_finite_tuple` 中的 `output.detach()` 改为
`output.detach().cpu()`，保持原有限值要求、输入、CPU 标杆和精度阈值，默认队列/inline
各 3/3。该结果仅作定位补充；最终定向数值结论采用卡 1 上未经修改的原 ATK 结果。
仓内公共测试工具和 RWKV8 executor 未作此替换。

失败日志、捕获张量、基础检查、私有副本和成功重跑均保留在本轮工作根；主要证据为
`evidence/result-summary.json`、`load-isolation-001/summary.json`、`rwkv-card1-{queue,inline}.log`、
`stable-lifecycle/summary.json`、`clean-environment-finite*.json` 与 `capture-analysis.json`。

工作根中的 `a2-evidence.tar.gz` 已同步回本地，SHA256 为
`d502862e6addaa2d39617fac34be90235b0e412ceba6fa4c26dc6e2a55ed1612`。
本轮两个 wheel 的构建指纹均为 `799b4432c42a`；单算子 wheel SHA256 为
`20cc3598a0f7ed42ab05342af58654dc000cb2e2c9ae7ed47bf8183114986738`，
全量实验 wheel SHA256 为 `8d6bc54525d9e6496480a351c8fd56d23b8cd58ac61ef2b8614b544cce5572ba`。

## 范围说明

正式对照 wheel 复用 2026-11-04 在主干基线 `a13871e8` 上构建的 `chunk_fwd_o` 测试包，
在本轮新环境安装；没有修改或重建正式算子。正式 ATK 对照使用既有 A2 配置
`use_exp2=False`、`BNSD`，只调整任务副本，不修改仓内正式 executor。

本轮是移除样例后的定向集成回归，未重跑 RWKV8 全部 209 条精度、完整安装负例矩阵、
reverse、非默认 stream、确定性、MSS、性能或正式全算子验收，也未执行 A5。
旧的双算子构建与 recompute 数值结果保留在历史记录中，不计入本轮结果。
