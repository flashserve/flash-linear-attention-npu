# ChunkKdaFwd

## 功能

`ChunkKdaFwd` 对齐不涉及 CP 切分的 FLA `chunk_kda_fwd` 顶层语义。公共接口接收 raw gate 或已激活的
自然对数 gate。Gate、Prepare、PostWu、FwdH 和 Finalize 复用同一个私有 L0；A5 的非对齐多 chunk
场景按四个阶段依次提交，其他场景保持单次提交。该内部调度不改变 ACLNN 或 Python 公共接口。

Shape 符号与布局约定见 [KDA 模型符号表](../README.md#model-shape-symbols)。

## Gate 公式

令 `x = g + dt_bias`。逐 token、逐 K 维的自然对数衰减为：

```text
use_gate_in_kernel = false:
    gate = g

use_gate_in_kernel = true, safe_gate = false:
    gate = -exp(A_log) * softplus(x)

use_gate_in_kernel = true, safe_gate = true:
    gate = lower_bound * sigmoid(exp(A_log) * x)
```

随后在每个 chunk 内计算：

```text
gk_i = cumsum(gate)_i / ln(2)
```

因此后续 `exp2(gk)` 与自然指数 gate 严格绑定，不暴露额外 gate scale。

## 输入

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q/k` | 必选 | 输入 layout 对应 Shape；FP16/BF16 | Query/Key |
| `v` | 必选 | 输入 layout 对应 Shape；与 q 同 dtype | Value |
| `g` | 必选 | 输入 layout 对应 K 维 Shape；FP32/BF16 | raw gate 或已激活自然对数 gate |
| `beta` | 必选 | 去掉 g 的 K 维；FP32/BF16 | Delta 系数 |
| `A_log` | 条件必选 | `[H_v]`，FP32 | `use_gate_in_kernel=true` 时必选 |
| `dt_bias` | 可选 | `[H_v*K]`，FP32 | gate bias |
| `initial_state` | 可选 | `[N,H_v,K,V]` 或 `[N,H_v,V,K]`，FP32 | 由 `state_v_first` 解释 |
| `cu_seqlens` | 可选 | `[N+1]`，INT64 | 变长序列 |
| `chunk_indices` | 可选 | `[2*N_c]`，INT64 | canonical chunk 顺序 |

`layout` 只描述上述输入。BSND/TND 由 L2 使用 `l0op::Transpose` 转为内部 BNSD/NTD。

## 输出

Python 返回顺序为：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

- `attn_out` 固定为 BSND/TND。
- `final_state` 固定按序列排列，末两维服从 `state_v_first`。
- `Aqk/Akk` 始终返回，固定为 head-major。
- `gk/w/u/qg/kg/v_new` 是供反向使用的 head-major 中间量。
- 公开 `h` 固定为 sequence-major；内部 `hCompute` 保持 head-major 供 Finalize 使用。
- 第 12 个返回值是 Python 层对 `initial_state` 的原对象透传，不是 aclnn 输出。

输出保留策略对齐 fla-org
[`chunk_kda_fwd`](https://github.com/fla-org/flash-linear-attention/blob/0f0f0c97af39343855b43bbbaddcedfda5cb9d77/fla/ops/kda/chunk_fwd.py)
提交 `0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

| 条件 | 返回 |
| --- | --- |
| `output_final_state=true` | 返回 `final_state`，否则为 `None` |
| `use_gate_in_kernel=false` 或 `disable_recompute=true` | 返回 `gk` |
| 始终 | 返回 `Aqk/Akk` |
| `disable_recompute=true` | 返回 `w/u/qg/kg/v_new` |
| `disable_recompute=true` 或 `return_intermediate_states=true` | 返回 `h` |

这是 `fla_npu.ops.ascendc.chunk_kda_fwd` 的低层 12 返回值语义；不涉及 CP。aclnn L2 不接收
`output_final_state/disable_recompute/return_intermediate_states`，每个可选输出是否写出仅由对应
输出指针是否为空决定。`w/u/qg/kg/v_new/h` 的 L0 阶段固定写内部 compute 张量，L2 仅在
对应指针非空时通过 `ViewCopy` 导出；`gkOut` 非空时直接复用为 `gkCompute`，避免目标场景
额外复制整张 FP32 gate。内部 `hCompute` 是 FwdH 到 Finalize 的必需 head-major 阶段结果；
公开 `hOut` 非空时，L2 转为 sequence-major 后导出。`hOut` 为空时仍创建 `hCompute`，但不
作为第 11 个 Python 返回值公开。

## 属性

| 名称 | 默认值 | 支持范围 |
| --- | --- | --- |
| `layout` | `BSND` | `BSND/BNSD/TND/NTD` |
| `scale` | 必传 | 通常为 `K**-0.5` |
| `chunk_size` | `64` | `64/128` |
| `output_final_state` | `false` | bool |
| `safe_gate` | `false` | bool |
| `lower_bound` | `-5.0` | safe raw gate 时 `[-5,0)` |
| `use_gate_in_kernel` | `false` | bool |
| `disable_recompute` | `false` | bool |
| `return_intermediate_states` | `false` | bool |
| `state_v_first` | `false` | bool |
| `epsilon` | `1e-6` | `use_qk_l2norm_in_kernel=true` 时的 rsqrt 下限 |
| `use_qk_l2norm_in_kernel` | `false` | true 时由算子内部归一化 q/k |
| `use_beta_sigmoid_in_kernel` | `false` | true 时由算子内部对 beta 取 sigmoid |
| `allow_neg_eigval` | `false` | true 时必须同时 `use_beta_sigmoid_in_kernel=true` |
| `use_exp2` | `true` | true 走 `exp2` 门控，false 走自然指数 |

## 支持范围

- A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。
- `K/V` 只支持两档且必须同档：`K=V=64` 或 `K=V=128`。混合档（如 `K=64,V=128`）
  与其它取值（含 `V=256`）都不支持，会在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`。
- `chunk_size` 为 64/128。
- TND/NTD 均支持多 head。
- 变长调用最多 1024 条逻辑序列，rank-4 变长输入要求 B=1。

## 场景分发

Python 入口 `fla_npu.ops.ascendc.npu_chunk_kda_fwd` 负责场景选择：

| 场景 | 走的 aclnn 入口 | L0 实现 |
| --- | --- | --- |
| `q/k/v` 为 BF16、`K=V=128`、`chunk_size=64`、`cu_seqlens` 严格递增、输出连续 | `aclnnChunkKdaFwdV2` | `ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize` 三个独立算子组合 |
| 其余场景（FP16、`K=V=64`、`chunk_size=128`、含空序列、输出非连续） | `aclnnChunkKdaFwd`（签名与 ABI 未变） | 本算子的私有 L0 融合实现 |

两个入口共用同一套参数校验、输出语义和返回码契约：`state_v_first` 均由算子原生解释，
公开的 `gk/Aqk/Akk/w/u/qg/kg/v_new` 始终是 head-major，`h` 始终是 sequence-major。

组合分支与融合分支共用同一组归一化/gate 开关，默认值即历史语义（q/k 由调用方预先归一化、
beta 由调用方预先 sigmoid、门控走 `exp2`）：

| 参数 | 默认值 |
| --- | --- |
| `epsilon` | `1e-6` |
| `useQkL2normInKernel` | `false` |
| `useBetaSigmoidInKernel` | `false` |
| `allowNegEigval` | `false` |
| `useExp2` | `true` |

私有 L0 融合实现只覆盖这组默认值，开关取非默认值时调用必须落在三算子组合的场景范围内，
否则返回 `ACLNN_ERR_PARAM_INVALID`。`ChunkKdaFwdPrepare` 的编译期输出档位由公开输出指针
组合推导（`none`/`forward`/`save`）：`Aqk/Akk` 是公开必选输出，`forward` 档只额外搬出 `Akk`，
不再像 `recompute` 档那样多搬 `qHat/kHat/qRstd/kRstd/betaEff`。

分发规则见 [API 文档](docs/api.md#l0-实现按场景分发)。

### 编译依赖

组合分支在 `chunk_kda_fwd_three_stage.cpp` 里直接 include 并调用另外三个独立算子的 op_api，
因此按算子裁剪编译时必须把这四个算子一起编，否则产物缺少子算子的 tiling / kernel 注册，
`aclnnChunkKdaFwdV2GetWorkspaceSize` 会在 tiling 阶段失败（只编 `chunk_kda_fwd` 时
`libcust_opapi.so` 甚至会缺 `l0op::ChunkKdaFwdPrepare/ChunkFwdH/ChunkKdaFwdFinalize` 符号）：

```sh
FLA_NPU_SOC=ascend910b \
FLA_NPU_OPS=chunk_kda_fwd,chunk_kda_fwd_prepare,chunk_kda_fwd_finalize,chunk_fwd_h \
python scripts/build_wheel.py
```

融合分支（`aclnnChunkKdaFwd`）只依赖 `chunk_kda_fwd` 自身及其 kernel 源码依赖，
只编 `chunk_kda_fwd` 时可用；但它不会带上组合分支需要的三个独立算子。

## 验证

唯一用例规格是 `tests/op_cases/chunk_kda_fwd.json`。数值测试位于
`tests/operators/chunk_kda_fwd/accuracy/`，性能使用 `tests/operators/chunk_kda_fwd/performance/profile.py`
和 `msopprof`。

完整 API 见 [API 文档](docs/api.md)，阶段和内存设计见 [设计文档](docs/design.md)。
