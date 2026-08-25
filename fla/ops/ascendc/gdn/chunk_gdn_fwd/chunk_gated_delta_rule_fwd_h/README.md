# ChunkGatedDeltaRuleFwdH

## 功能

共享的 chunk 间状态递推算子，供 GDN 和 KDA 使用。公开短名为 `chunk_fwd_h`，
物理算子继续复用 `ChunkGatedDeltaRuleFwdH`。输入固定为 head-major BNSD：

```text
k:   [B,H_k,T,K]
w:   [B,H_v,T,K]
u:   [B,H_v,T,V]
g:   [B,H_v,T]       optional
gk:  [B,H_v,T,K]     optional
```

`g` 和 `gk` 必须且只能提供一个，是否为空是唯一的模型模式判据：

- GDN v1：传原始 `k` 和 scalar gate `g`，`gk=None`；
- KDA/GDN2：传 Prepare 生成的 `kg` 作为 `k`，传 key-wise gate `gk`，`g=None`。

两者同时为空或同时非空均返回 `ACLNN_ERR_PARAM_INVALID`。`use_exp2` 和 `state_v_first`
都不参与模型模式判断。

Host 通过 template tiling key 直接选择两套互斥的编译期模式，不在同一个 kernel 实例内运行时判断 gate 模式：

| 模式 | kernel 模板 |
|---|---|
| GDN v1 / `g-only` | `GDNFwdHKernel<..., gateMode=G, expMode>` |
| KDA/GDN2 / `gk-only` | `GDNFwdHKernel<..., gateMode=GK, expMode>` |

template tiling key 保留 `V_TILE x GATE_MODE(G/GK) x EXP_MODE(exp/exp2)` 三维；当前只登记
`V_TILE=128`，因此共 4 个语义组合。input、gate 和 state dtype 不进入 tiling key，分别直接使用算子编译生成的
`DTYPE_K`、`DTYPE_G/DTYPE_GK` 和 `DTYPE_FINAL_STATE`。公开分派不实例化双空或双 gate 模式；
`useExp2` 是两套 gate 模式下的正交编译期参数。

## 计算

对每个 chunk，四个 stage 的公共骨架与模式分支如下：

```text
stage0:
  P = w @ h_prev

stage1:
  v_new = u - P
  GDN v1:    v_stage2 = gate(g_last - g) * v_new   # v_new_decay
  KDA/GDN2:  v_stage2 = v_new                      # 不做额外 decay

stage2:
  GDN v1:    delta_h = raw_k^T @ v_new_decay
  KDA/GDN2:  delta_h = kg^T @ v_new

stage3:
  GDN v1:    h_next = gate(g_last) * h_prev + delta_h
  KDA/GDN2:  h_next = diag(gate(gk_last)) * h_prev + delta_h
```

两种 gate 模式共用同一指数合同：`gate(x)` 在 `use_exp2=false` 时为 `exp(x)`，在
`use_exp2=true` 时为 `exp2(x)`。因此独立传入 `gk` 时，其数值域必须与 `use_exp2` 匹配；
KDA/GDN2 的 Prepare 生成 log2 域 `gk`，调用本算子时固定使用 `use_exp2=true`。KDA/GDN2
的 chunk 内相对衰减已经吸收到 `kg`，Stage1 不得再次对 `v_new` 衰减。`use_exp2` 只控制
指数语义，不得替代 `g/gk` 选择模型模式。
`save_new_value` 是兼容属性，首版仅支持 `True`。

## Python API

```python
from fla_npu.ops.ascendc import chunk_fwd_h

h, v_new, final_state = chunk_fwd_h(
    k,
    w,
    u,
    g=None,
    *,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    state_v_first=False,
)
```

`state_v_first=True` 只改变公开 h/state 的最后 K/V 两轴。Python adapter 在进入物理 aclnn 前
把 `initial_state` 规范化为 `[N,H_v,K,V]`，并始终向物理接口传
`stateVFirst=false`；返回时再把 h/final_state 转回 `[V,K]`。

## 输出

| 输出 | 必选性 | Shape |
| --- | --- | --- |
| `h` | 必选 | `[B,H_v,N_c,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |
| `v_new` | 必选 | `[B,H_v,T,V]` |
| `final_state` | 可选 | `[N,H_v,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |

`h` 是每个 chunk 的起始状态，反向会继续使用，因此保持 head-major。`final_state` 仅供调用者输出，
为保持既有 Python 接口兼容，`output_final_state=false` 时三元组第三项返回 `None`。物理 aclnn
内部使用的 `[0]` Tensor 只是固定 launch 参数的占位，不作为 Python 输出暴露；其 dtype 在存在
`initial_state` 时与 initial state 一致，否则默认为 FP32。

## aclnn

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

## 支持范围

- A2/A3/A5。
- FP16/BF16 数据路径；gate 使用 FP32 或与数据路径相同的 dtype；初始/最终状态支持 BF16/FP32，
  二者必须同 dtype，无 initial state 时默认 FP32。这里的 state dtype 仅定义 initial/final state 的 GM
  边界存储格式，不表示 chunk 间 rolling state 已按同一 dtype 实现递推。
- V=128；`chunk_size=64`。
- 分核按每个紧凑序列的 value head 数计算：`max_load=ceil(H_v/available_cores)`，启用核数为
  `ceil(H_v/max_load)`，同一核负责负载均衡的连续 head 区间。每核 head 数可以超过 4，但每个
  chunk 按最多 4 个 head 拆成多轮，完成同一 chunk 的全部 head 轮次后再进入下一 chunk。
  8 个 workspace slot 沿用 DHU 的两 bank ready/free 协议，固定使用
  `slot=round_bank*4+round_head_offset`。例如 `H_v=96` 时，32 核为每核 3 个 head；28 个
  可用核时启用 24 核、每核 4 个 head；A2 的 20 核分为 16 核各 5 个 head 和 4 核各 4 个
  head，5-head 核按 `4+1` 两轮处理。
- dense 时不传 `cu_seqlens/chunk_indices`；varlen 要求 B=1、`cu_seqlens` 从 0 到 T 严格递增，
  `chunk_indices` 使用 `[NT,2]` 的 canonical sequence-major 顺序。
- `state_v_first` true/false。
