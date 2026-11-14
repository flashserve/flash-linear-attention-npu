# ChunkKdaBwdRecompute

## 功能

融合 KDA safe-gate、chunk cumsum 与 w/u/qg/kg 重计算。
公开调用、完整 shape/dtype 和可选项见 [docs/api.md](docs/api.md)，
A6 数据流与资源设计见 [docs/design.md](docs/design.md)。

## 输入

q/k 使用 source head，v/g/beta/A 使用 value head；`HV%HK=0`。
dense BNSD，`K=V=128`、`chunk_size=64`，q/k/v/A 为 BF16，g/beta 为 BF16 或 FP32。

## 输出

返回 `(gk,w,u,qg,kg)`，gk 为 FP32，其余 BF16；关闭 gate 时 gk 不对外返回。

## 属性

保留 `use_gate_in_kernel`、`use_exp2`、`lower_bound` 和 `input_layout` 的既有调用方式。
`input_layout="NTD"` 不改变当前 rank-4 存储契约。

## 支持范围

注册 A2/A3/A5 和新增的 A6 `ascend960dt`。A6 使用 1:1 MIX、CG=1、双输入/双结果区与双 L1 slot。
变长输入约束见 API，A6 的数据流和资源布局见 [设计文档](docs/design.md)。

## 实现约束

A6 使用 `__NPU_ARCH__==9201`、1 AIC : 1 AIV、CG=1、双输入/双结果区和双 L1 slot。
测试入口与用例组织见 [ATK README](../../../../../tests/atk/chunk_kda_bwd_recompute/README.md)。
