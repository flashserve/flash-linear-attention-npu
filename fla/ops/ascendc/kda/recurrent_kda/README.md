# RecurrentKda

`RecurrentKda` 是 KDA 的 fused recurrent 前向算子。算子在一个 AICore kernel 内完成 recurrent state decay、delta 更新和输出计算；`raw gate -> log gate` 和 `beta sigmoid` 可以在 kernel 内完成，不依赖 `KdaGateCumsum` 或 GDN recurrent 的接口。

完整接口和调用示例见 [API 文档](docs/api.md)，实现方案及非连续输入的 stride 约束见
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
    initial_state=None,
    *,
    cu_seqlens=None,
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

- 不传 `cu_seqlens`：固定长度，`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- 传入 `cu_seqlens`：变长 packed，`q/k=[T,H,K]`，`v=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。`layout` 为兼容参数，忽略其值。
- `initial_state=None` 且 `inplace_final_state=False` 时，Python wrapper 创建全零 FP32 状态；原位模式须显式传入 state pool。
  `state_v_first=True` 对应 `[state_capacity,HV,V,K]`，`state_v_first=False` 对应
  `[state_capacity,HV,K,V]`。支持 slot/head 维带间隔的非连续 view，但内部二维 state
  矩阵必须稠密。
- Q/K/V 支持融合 storage 切片形成的规则非连续 view；算子保留各自 storage offset，并按 token/head
  stride 直接读取。不满足直接访问约束的 view 在 ACLNN 侧自动回退连续化。
- `scale=None` 时，Python wrapper 使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时，kernel 内对每个 token 的 `q/k` 按 `x/sqrt(sum(x*x)+1e-6)` 做 L2 normalize，然后对 `q` 乘 `scale`。
- 不传 `A_log` 时，`g` 是预计算的 step log gate，kernel 使用 `exp(g)` 做 state decay；忽略 `dt_bias`、`safe_gate` 和 `lower_bound`，不校验、搬运或读取未启用的 bias。
- 传入 `A_log` 时，`g` 是 raw gate，`dt_bias` 可选。`use_gate_in_kernel` 为兼容参数，忽略其值。
  - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`。
  - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时，kernel 使用 `sigmoid(beta)`；若 `allow_neg_eigval=True`，再乘 2。
- Python/aclnn 入口支持非连续 `initial_state`；原位模式返回同 storage/stride 的 state，非原位模式返回独立 state 并保持输入不变。legacy Torch 入口不属于当前支持范围。
- `cu_seqlens` 可选；为空时固定长度输入按 batch 行划分。显式传入时须为同设备 INT32/INT64 tensor，shape 为 `[seq_num+1]`，使用与 fla-org 一致的
  累积 offset 语义。首项必须为 0，末项等于有效 packed token 数且可小于图捕获的 token capacity，
  相邻差值是各序列长度。A5 host 读取其值，以最大有效序列长度从 `{1,2,4,8}` 选择共用 X，上限8；全空取1。
  当前 ACLNN 第一阶段在调用方保证元数据就绪后，同步复制该元数据到 host，executor 保存本次值；packed 模式不支持在 ACLGraph 捕获中调用，修改边界后须重新创建 executor。固定长度路径不增加这次同步和复制。
- `ssm_state_indices` 支持 packed `[T]` 和 speculative `[seq_num,max_step]`。显式索引模式允许 `state_capacity > seq_num`，并仅更新命中的槽。
- 空序列不读取索引或 state，适用于 packed batch 中的 padding sequence。活跃 indices/accepted 由 host 校验并保存本次 executor 的快照；此类调用需要读取设备元数据，不支持 ACLGraph 捕获，修改后须重新创建 executor。无 indices 的调用不增加该检查开销。

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ (q_t * scale)
```

A5 的寄存器计算为 FP32；当 State dtype 为 BF16 时，当前 token 的 out 使用舍入前的更新结果，
跨 token State 按 RINT 转 BF16 后保存在 UB。K-first 全程保持 K-first，在 UB 增加行间空隙，
沿连续 V 方向用两遍 Load/Store 完成 prediction、State 更新及 out 累计。

## 当前限制

- `q/k/v/out` 仅支持 `BF16`。
- Q/K/V 直接非连续访问要求 feature stride 为 1、head 间不重叠、token 间不重叠且 stride 为正；
  BSND 在 `B>1` 时要求 batch 间地址不重叠；Q/K/V 可有独立的 batch stride。feature-gap、zero-stride
  等不满足直接读取条件的 view 自动回退连续化；重叠 descriptor 明确拒绝。
- `g/beta` 支持 `FP32/BF16/FP16`，A5 kernel 按原 dtype 读入后在寄存器转 FP32。
- `A_log/dt_bias` 支持 `FP32`。
- `cu_seqlens` 传入时为与 q 同设备的 INT32/INT64 Tensor；offset 必须单调不减，末项不得超过输入
  token capacity，A5 通过容量不超过8的窗口递推支持更长序列；A2/A3 按各自硬件实现限制执行。末项小于 capacity 时，仅有效 token 对应的输出和 state
  更新有定义，padding tail 输出不作保证。
- A2/A3/A5 共用 cu_seqlens、A_log 判定及未启用参数的忽略语义；真机验证范围为 A5/A2。
- 支持 V-first `[state_capacity,HV,V,K]` 和 K-first `[state_capacity,HV,K,V]`，state dtype 为
  FP32/BF16。非连续 state 仅允许 slot/head 外层维存在间隔；最后两维必须为行主序稠密矩阵且各
  slot/head 的地址区间不得重叠。底层 aclnn 接口要求显式传入 state。
- `HV` 必须能被正整数 `H` 整除；无256的 head 数上限，受字段、地址和设备容量限制。A5 仅支持 `K=V=128`；A2/A3 保留各自维度和资源约束。

A5 的 V-first 在所有 X 下均使用两遍 State 读取，并按 State dtype 选择双行交错调度；
K-first FP32/BF16 使用连续 V 方向两遍计算，prediction 系数复用既有512B尾部区域从 UB 广播。
out 在寄存器收集完整 token，整 tile 完成后批量搬出；必要同步和下一任务 State 预取保留。
精度验收、分路径性能及验证范围见[测试记录](../../../../../tests/atk/recurrent_kda/README.md)。
