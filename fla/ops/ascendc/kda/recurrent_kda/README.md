# RecurrentKda

`RecurrentKda` 是 KDA 的 fused recurrent 前向算子。算子在一个 AICore kernel 内完成 recurrent state decay、delta 更新和输出计算；`raw gate -> log gate` 和 `beta sigmoid` 可以在 kernel 内完成，不依赖 `KdaGateCumsum` 或 GDN recurrent 的接口。

完整接口和调用示例见 [API 文档](docs/api.md)，实现方案见 [设计文档](docs/design.md)。Shape 符号统一引用
[KDA 模型符号表](../README.md#model-shape-symbols)。

## Python 接口

```python
from fla_npu.ops.ascendc import recurrent_kda

out, final_state = recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=True,
)
```

## 语义

- `layout="BSND"`：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `layout="TND"`：`q/k=[T,H,K]`，`v=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state=None` 时，Python wrapper 会创建全零初始状态；显式传入时 shape 为 `[seq_num,HV,V,K]`。当前仅支持 `state_v_first=True`。
- `scale=None` 时，Python wrapper 使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时，kernel 内对每个 token 的 `q/k` 做 L2 normalize，然后对 `q` 乘 `scale`。
- `use_gate_in_kernel=False` 时，`g` 被视为已经预计算好的 step log gate，kernel 使用 `exp(g)` 做 state decay。
- `use_gate_in_kernel=True` 时，`g` 是 raw gate，必须传 `A_log`；可选 `dt_bias`。
  - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`。
  - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时，kernel 使用 `sigmoid(beta)`；若 `allow_neg_eigval=True`，再乘 2。

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ (q_t * scale)
```

## 当前限制

- `q/k/v/out` 仅支持 `BF16`。
- `g/beta` Python 入口支持 `FP32/BF16/FP16`，aclnn 预处理后以 `FP32` 输入 kernel。
- `A_log/dt_bias` 支持 `FP32`。
- `cu_seqlens` 为 Python `int[]`，不是 Tensor；每段长度必须不超过 8。
- Dense 输入未传 `cu_seqlens` 时，`T <= 8`。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- 仅支持 `state_v_first=True`，state layout 为 `[seq_num, HV, V, K]`；底层 aclnn 接口要求显式传入 `initialState`。
- `HV` 必须能被 `H` 整除；`H/HV <= 256`，`K/V <= 512`。
