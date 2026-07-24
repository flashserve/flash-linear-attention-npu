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
    *,
    cu_seqlens,
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
- `initial_state=None` 时，Python wrapper 会创建全零初始状态；显式传入时是原位更新的 state pool，shape 为 `[state_capacity,HV,V,K]`。当前仅支持 `state_v_first=True`。
- `scale=None` 时，Python wrapper 使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时，kernel 内对每个 token 的 `q/k` 做 L2 normalize，然后对 `q` 乘 `scale`。
- `use_gate_in_kernel=False` 时，`g` 被视为已经预计算好的 step log gate，kernel 使用 `exp(g)` 做 state decay。
- `use_gate_in_kernel=True` 时，`g` 是 raw gate，必须传 `A_log`；可选 `dt_bias`。
  - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`。
  - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时，kernel 使用 `sigmoid(beta)`；若 `allow_neg_eigval=True`，再乘 2。
- Python/aclnn/legacy 入口支持非连续 `initial_state`。Python 主入口返回 `final_state` 时与输入保持相同
  storage 和 stride；legacy Torch 入口只返回 `out`，最终状态通过 `initial_state` 原位更新。
- `cu_seqlens` 是必传的同设备 INT32/INT64 tensor，shape 为 `[seq_num+1]`，使用与 fla-org 一致的
  累积 offset 语义。首项必须为 0，末项等于有效 packed token 数且可小于图捕获的 token capacity，
  相邻差值是各序列长度。host 不读取其值，兼容 ACLGraph capture/replay。
- `ssm_state_indices` 支持 packed `[T]` 和 speculative `[seq_num,max_step]`。显式索引模式允许 `state_capacity > seq_num`，并仅更新命中的槽。
- 空序列不读取索引或 state，适用于 packed batch 中的 padding sequence。

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
- `cu_seqlens` 为必传、与 q 同设备的 INT32/INT64 Tensor；offset 必须单调不减，末项不得超过输入
  token capacity，各相邻差值必须不超过 8。末项小于 capacity 时，仅有效 token 对应的输出和 state
  更新有定义，padding tail 输出不作保证。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- 仅支持 `state_v_first=True`，state layout 为 `[state_capacity, HV, V, K]`；底层 aclnn 接口要求显式传入可变 state。
- `HV` 必须能被 `H` 整除；`H/HV <= 256`；`K/V` 仅支持 `K=128,V=128` 或 `K=128,V=256`。
