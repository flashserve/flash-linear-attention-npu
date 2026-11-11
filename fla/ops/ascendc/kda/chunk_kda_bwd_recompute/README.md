# ChunkKdaBwdRecompute

[设计文档](docs/design.md) | [API 文档](docs/api.md)

## 功能

`ChunkKdaBwdRecompute` 在 KDA 反向的 `disable_recompute=false` 路径上，
融合重算前向检查点：safe-gate + chunk 内 cumsum（`gk`）以及
`recompute_w_u_fwd` 语义下的 `w/u/qg/kg`。它不计算 chunk 间状态递推，
也不产生 `h/v_new`；后续由 `ChunkFwdH` 等算子接续。

稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

gk, w, u, qg, kg = chunk_kda_bwd_recompute(
    q,
    k,
    v,
    g,
    beta,
    A,
    chunk_size=64,
    A_log=A_log,
    dt_bias=dt_bias,
    cu_seqlens=None,
    chunk_indices=None,
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
    input_layout="BNSD",
)
```

该入口经 Stable-ABI / ctypes 适配层调用 `aclnnChunkKdaBwdRecompute`。
Python 返回顺序以调用方常用的 `gk` 开头；aclnn 侧输出登记顺序为
`w/u/qg/kg/gk`，适配层在返回前重排。

Shape 符号与布局约定见 [KDA 模型符号表](../README.md#model-shape-symbols)。

## Gate 与重计算

令 `x = g + dt_bias`。在 `use_gate_in_kernel=true`（独立入口默认）下使用
safe-gate：

```text
gate = lower_bound * sigmoid(exp(A_log) * x)
gk_i = cumsum(gate)_i * (1 / ln(2))    # use_exp2=true
```

随后按 GQA 将 `q/k` 展开到 `HV`，并在每个 chunk 内：

```text
e2   = exp2(gk)
qg   = q_hv * e2
kbg  = k_hv * beta * e2
vb   = v * beta
kg   = k_hv * exp2(gk_last - gk)       # gk_last 为 chunk 末行
w    = A @ kbg                         # BF16 输入、按 chunk 有效行
u    = A @ vb
```

`A` 最后一维始终是 `chunk_size`（含尾 chunk 有效列约束由 kernel 处理）。
`w/u` 的矩阵乘对齐 NPU Cube：先将参与乘的操作数落到 BF16 再累加。

## 输入输出

`q/k/v/A` 固定为 BF16，`K=V=128`，`chunk_size=64`。`g/beta` 支持 BF16
或 FP32；`A_log` / `dt_bias` 为 FP32。张量须为非私有 ND 排布；L2 会对
非连续输入做连续化。

| 输入 | dense shape | 说明 |
| --- | --- | --- |
| `q/k` | `[B, HK, T, 128]` | BF16；GQA 时 `HV % HK == 0` |
| `v/g` | `[B, HV, T, 128]` | `v` 为 BF16；`g` 为 BF16/FP32 |
| `beta` | `[B, HV, T]` | BF16/FP32 |
| `A` | `[B, HV, T, 64]` | BF16，对应前向的 `Akk`（或等价三角系统解） |
| `A_log` | `[HV]` | `use_gate_in_kernel=true` 时必传 |
| `dt_bias` | `[HV, 128]` 或等价扁平 | 可选 |

`cu_seqlens` / `chunk_indices` 为可选 host int array：变长时两者须同时
提供；当前独立入口与 ATK 主路径以定长 dense BNSD 为主。

| 输出 | shape | dtype |
| --- | --- | --- |
| `gk` | `[B, HV, T, 128]` | FP32；可选（aclnn 指针为空时用内部缓冲，不公开写出） |
| `w/qg/kg` | `[B, HV, T, 128]` | BF16 |
| `u` | `[B, HV, T, 128]` | BF16 |

Python 固定返回五元组：

```text
(gk, w, u, qg, kg)
```

独立 aclnn 是否写出 `gk` 由 `gkOut` 指针是否为空决定；`use_gate_in_kernel`
在独立 L2 上由「是否提供 `gkOut`」推导，L0 仍接收显式布尔属性。

## 属性

| 名称 | 默认值 | 说明 |
| --- | --- | --- |
| `chunk_size` | `64` | 仅支持 `64` |
| `use_gate_in_kernel` | `true` | true 时必须提供 `A_log` |
| `use_exp2` | `true` | true 走 log2 域累计 gate |
| `lower_bound` | `-5.0` | safe-gate 下须位于 `[-5, 0)` |
| `input_layout` | `"BNSD"` | 仅接受 `BNSD` / `NTD`；本算子不按该串做布局转换，张量须已是 head-major rank-4（packed 由总层先做 batch=1 视图） |

## 支持范围

- SoC 注册：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。
  当前 ATK 与性能验收以 A5 为主。
- `K = V = 128`，`chunk_size = 64`；`HV % HK == 0`。
- 独立入口输入布局固定为 dense BNSD（或总层给出的等价 rank-4 视图）；
  拒绝 `BSND` / `TND` 作为 `input_layout`。
- 在整体 `chunk_kda_bwd` 的 A2/A3 编排中，本算子位于
  `Recompute → FwdH → Prepare → Dhu → Finalize` 链的起点；详见
  [`../chunk_kda_bwd/docs/A2A3_design.md`](../chunk_kda_bwd/docs/A2A3_design.md)。

## 实现

Mix **1 AIC : 2 AIV**：Vector 完成 gate cumsum、`qg/kg` 与 `kbg/vb` 准备，
Cube 完成 `A @ kbg` / `A @ vb`。A5 走 `op_kernel/arch35/`；A2/A3 走同目录
非 arch35 实现。Stage、L1/workspace 与同步见 [设计文档](docs/design.md)。

## 验证

完整 aclnn / 返回码见 [API 文档](docs/api.md)。ATK 用例位于
[`tests/atk/chunk_kda_bwd_recompute`](../../../../../tests/atk/chunk_kda_bwd_recompute/README.md)，
含精度矩阵、CPU 标杆与 8K/16K Event 性能脚本。离线冒烟还可参考
`torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_recompute.py`。

精度门限（max abs diff，相对融合 CPU 标杆）：`gk/qg/kg < 0.05`，
`w/u < 0.15`（`u` 因 BF16 GEMM 约 0.12 属预期）。
