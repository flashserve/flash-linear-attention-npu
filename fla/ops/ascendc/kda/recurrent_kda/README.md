# RecurrentKda

`RecurrentKda` 是 KDA 的 fused recurrent 前向算子。算子在一个 AICore kernel 内完成 recurrent state decay、delta 更新和输出计算；`raw gate -> log gate` 和 `beta sigmoid` 可以在 kernel 内完成，不依赖 `KdaGateCumsum` 或 GDN recurrent 的接口。

完整接口和调用示例、非连续 State/QKV 的支持边界见 [API 文档](docs/api.md)，实现方案见
[设计文档](docs/design.md)。Shape 符号统一引用
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
    initial_state,
    cu_seqlens=cu_seqlens,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
)
```

## 语义

- `layout="BSND"`：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `layout="TND"`：`q/k=[T,H,K]`，`v=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 为必须显式传入的状态池。原位和非原位模式均要求非 `None`；
  使用零初态时，由调用方在与 `q` 相同的设备上创建对应布局的零状态张量。
  `state_v_first=True` 对应 `[state_capacity,HV,V,K]`，`state_v_first=False` 对应
  `[state_capacity,HV,K,V]`。支持 slot/head 维带间隔的非连续 view，但内部二维 state
  矩阵必须稠密。
- Q/K/V 支持融合 storage 切片形成的规则非连续 view；算子保留各自 storage offset，并按 token/head
  stride 直接读取。不满足直接访问约束的 view 在 ACLNN 侧自动回退连续化。
- `scale=None` 时，Python wrapper 使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时，kernel 内对每个 token 的 `q/k` 做 L2 normalize，然后对 `q` 乘 `scale`。
- `use_gate_in_kernel=False` 时，`g` 被视为已经预计算好的 step log gate，kernel 使用 `exp(g)` 做 state decay。
- `use_gate_in_kernel=True` 时，`g` 是 raw gate，必须传 `A_log`；可选 `dt_bias`。
  - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`。
  - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时，kernel 使用 `sigmoid(beta)`；若 `allow_neg_eigval=True`，再乘 2。
- Python/aclnn 入口支持非连续 `initial_state`。Python 主入口在原位模式返回输入 State，非原位模式
  直接读取初态且不修改输入，仅在 `output_final_state=True` 时分配并返回同 shape/dtype 的独立状态池。
  非原位输出池中未写入的槽位（包括空序列对应槽位）内容未定义，不保证保留输入值。
- `cu_seqlens` 是可选的同设备 INT32/INT64 tensor，shape 为 `[seq_num+1]`，使用与 fla-org 一致的
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

## 支持范围与已知限制

- `q/k/v/out` 仅支持 `BF16`。
- Python/aclnn 通路的 Q/K/V 非连续快路径要求 feature stride 为 1、head 间不重叠、token 间不重叠且 stride 为正；
  BSND 在 `B>1` 时还要求 `batchStride == T * tokenStride`。已验证的 feature-gap、zero-stride
  与 batch-gap view 自动回退连续化；A5 上 TND 和 BSND `B=1` 也支持不重叠的 head-major view，
  A5 上 BSND `B>1` 的 head-major view 回退连续化。A2/A3 保留原 token-major 约束。
  正 stride 的 token/head 地址区间重叠时明确拒绝。
- A5 L0 直调在 BSND `B>1` 且提供 `cu_seqlens` 时，对 Q/K/V 分别要求 `batchStride == T * tokenStride`，
  不满足时由 tiling 拒绝。公开 Python/aclnn 仍先按上述规则连续化，保留非连续 QKV 输入支持。
- `g/beta` Python 入口支持 `FP32/BF16/FP16`，ACLNN 按需连续化；A5 保留输入 dtype，在 VF 中转换为 `FP32` 计算。
- `A_log/dt_bias` 支持 `FP32`。
- `cu_seqlens` 可选；为空时 BSND 按 batch 行划分、TND 为一条序列；传入时为与 q 同设备的 INT32/INT64 Tensor；offset 必须单调不减，末项不得超过输入
  token capacity。A2/A3 支持超过 8 token 的 dense 单序列与 varlen 分段，内部按 8-token 窗口递推；
  A5 的 V=128/V=256 均支持长序列。末项小于 capacity 时，仅有效 token 对应的输出和 state
  更新有定义，padding tail 输出不作保证。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- 支持 V-first `[state_capacity,HV,V,K]` 和 K-first `[state_capacity,HV,K,V]`，state dtype 为
  FP32/BF16。非连续 state 仅允许 slot/head 外层维存在间隔；最后两维必须为行主序稠密矩阵且各
  slot/head 的地址区间不得重叠。底层 aclnn 接口要求显式传入 state。
- `HV` 必须能被 `H` 整除；`H/HV <= 256`；`K/V` 仅支持 `K=128,V=128` 或 `K=128,V=256`。

## A5 实现

V=128/V=256 使用 PR #1083 风格的 `RecurrentKdaEngine` 类，统一管理初始化、任务循环、输入搬运、逐 token VF 和输出搬运。三个 State bank 轮换递推，输入和 out 使用两个槽，每个 tile 最多包含四个 token；tile 计算完成后批量搬出 out。Mutex 保护 State、输入、out 和 bias 的复用，下一任务初态搬入可与当前任务最后一个 token 的计算重叠。

V-first 以两行为粒度读取两遍 State，先收集 prediction 到512 B scratch，再计算 residual 覆盖同一 scratch，逐行广播 residual 并重读 State 完成更新和 out。K-first 保持紧凑 [K,V] 布局，参考 x8-pr 沿连续 V 维读取两遍 State，prediction 和 out 各用两个 FP32 寄存器累加，512 B scratch 保存 alpha*k 系数。BF16 State 在寄存器中转换，跨 token 仅保留 BF16 结果；out 使用舍入前的 FP32 更新值。Q/K 的 L2 normalize 使用 `x/sqrt(sum(x*x)+1e-6)`。具体内存与同步见 [设计文档](docs/design.md)。

公开接口、`cu_seqlens`、State 索引和非连续 view 的语义保持不变；元数据在设备每次执行时读取。V=256 沿 V 拆成两个128列分块并复用相同 VF 和缓冲；A2/A3 保留兼容实现。
