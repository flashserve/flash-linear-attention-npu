# ChunkKdaFwdFinalize

[设计文档](docs/design.md) | [API 文档](docs/api.md)

## 功能

`ChunkKdaFwdFinalize` 消费 Prepare 的 `qg_scaled/Aqk` 和 FwdH 的
`v_new/h`，计算 KDA 前向的最终 attention 输出。它不重新计算 Prepare 或
chunk 间状态，也不产生 `h/final_state`。四个输入均已按 value head 展开，
Finalize 本身不执行 query/value head 映射。

稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_finalize

attn_out = chunk_kda_fwd_finalize(
    qg_scaled,
    aqk,
    v_new,
    h,
    output_layout="BSND",
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
)
```

每个长度不超过 64 的 chunk 计算：

```text
attn_out = cast_bf16(fp32(qg_scaled @ h) + fp32(Aqk @ v_new))
```

两项矩阵乘以 BF16 输入、FP32 累加。`Aqk` 已按 Prepare 的 `scale` 缩放；
`qg_scaled` 也已缩放，因此 Finalize 不再乘 `scale`。尾 chunk 只消费有效
token 行和有效 key 列。

## 输入输出

输入 tensor 必须使用非私有 ND descriptor，非连续视图由 L2 连续化；
输出必须是连续 ND。所有 tensor 固定为 BF16；`K=V=128`，
`chunk_size=64`。
`HV` 是 value head 数，`C` 是每条序列的 chunk 数；变长时 `C` 为所有序列
chunk 数之和。`Aqk` 最后一维始终是 64，包括尾 chunk。

| 数据 | dense shape | packed shape |
| --- | --- | --- |
| `qg_scaled` | `[B,HV,T,128]` | `[HV,T,128]` |
| `Aqk` | `[B,HV,T,64]` | `[HV,T,64]` |
| `v_new` | `[B,HV,T,128]` | FwdH 主路径 `[1,HV,T,128]`；独立调用也接受 `[HV,T,128]` |
| `h` | `[B,C,HV,128,128]` | `[1,C,HV,128,128]` |
| `attn_out` | `BSND [B,T,HV,128]` 或 `BNSD [B,HV,T,128]` | `TND [T,HV,128]` 或 `NTD [HV,T,128]` |

三个 token 输入始终为 head-major，与输出 layout 无关。packed 模式下
`v_new` 的 FwdH 主路径保留 rank-4 输出格式，不压掉首维。
`state_v_first=false`
时，`h` 的末两维物理顺序是 `[K,V]`；为 true 时是 `[V,K]`。

`cu_seqlens` 可选，按 host int array 传入：首元素为 0，末元素为 T，
元素严格递增。rank-4 变长输入要求 `B=1`。`chunk_indices` 可选，
仅能与 `cu_seqlens` 同时传入，并按 sequence-major 顺序依次列出
`(sequence_id, local_chunk_id)`。不传时按 `cu_seqlens` 生成相同顺序。

本独立算子的 `output_layout` 可选择四种格式；完整
`ChunkKdaFwd` 的模型级公开 `attn_out` 仍固定 BSND/TND，
不因独立 Finalize 的布局选项改变。[KDA 模型符号表](../README.md#model-shape-symbols)
描述完整模型默认布局，独立阶段以上表为准。

## 调用边界

独立算子支持 A2 (`ascend910b`)、A3 (`ascend910_93`) 和 A5
(`ascend950`)。算子定义和 aclnn 接口属于本目录；公共 Python 入口
`fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 由 Stable-ABI 适配层提供，
适配代码在 `torch_custom/fla_npu/csrc/src/stable_chunk_kda_fwd_finalize.cpp`，
Python wrapper 在 `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`，
注册统一在 `stable_ops.cpp`。Python 入口的形状、dtype、layout 和变长元数据
约束与 aclnn 接口一致。ATK 用例另使用算子私有的直调适配。

## 验证

ATK 用例位于 [`tests/atk/chunk_kda_fwd_finalize`](../../../../../tests/atk/chunk_kda_fwd_finalize/README.md)，
以直接输入格式覆盖四种输出 layout、dense/packed、变长、尾 chunk 和
`state_v_first`。独立 Finalize 没有 HK 输入，因此这些用例不能单独证明
`HK<HV` 的 GVA 映射；需在 Prepare/FwdH/Finalize 组合链路验证。
