# ChunkKdaFwdPrepare

[A5设计文档](docs/design.md) | [A2设计文档](docs/design_a2.md) | [API 文档](docs/api.md)

## 功能

`ChunkKdaFwdPrepare` 完成 KDA 前向中的 Q/K 归一化、gate 前缀和、chunk 内三角系统准备和
Post-WU 计算。它只负责 Prepare 边界，不计算 chunk 间状态递推和最终 attention 输出。

稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

outputs = chunk_kda_fwd_prepare(
    q,
    k,
    v,
    g,
    beta,
    scale=1.0,
    layout="BNSD",
    chunk_size=64,
    epsilon=1e-6,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=-5.0,
    use_exp2=True,
    a_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    backward_mode="save",
)
```

该入口经 Stable-ABI 适配层调用 `aclnnChunkKdaFwdPrepare`，不注册 legacy `torch.ops.npu` 接口。

`backward_mode` 决定 13 个输出槽中哪些真正分配与写出（`none`/`forward`/`recompute`/`save`，
与上文“输出保留策略”一一对应；默认 `save` 即“13 项全部返回”的历史行为）。未选中的槽在
L2 层传 `nullptr`，不参与公开 GM 写回，因此调用方按需取 `q_hat/k_hat/q_rstd/k_rstd/beta_eff`
即可，不需要关心档位之外的槽位。

格式要求：本算子的 L2 只拒绝私有（分形）格式，`ND`/`NCHW`/`NCL`/`NHWC` 等非私有拼写均可传入；
张量需连续（`CheckContiguous`）。

## 输入输出

`q/k/v` 固定为 BF16，K/V 维固定为 128，`chunk_size` 固定为 64。`g/beta` 支持
BF16 或 FP32，`a_log/dt_bias` 固定为 FP32。

| 输入 | dense shape | 无 batch shape | 说明 |
| --- | --- | --- | --- |
| `q/k` | BNSD `[B,HK,T,128]`；BSND `[B,T,HK,128]` | NTD `[HK,T,128]`；TND `[T,HK,128]` | BF16 |
| `v` | BNSD `[B,HV,T,128]`；BSND `[B,T,HV,128]` | NTD `[HV,T,128]`；TND `[T,HV,128]` | BF16 |
| `g` | BNSD `[B,HV,T,128]`；BSND `[B,T,HV,128]` | NTD `[HV,T,128]`；TND `[T,HV,128]` | BF16/FP32，跟随 layout |
| `beta` | BNSD `[B,HV,T]`；BSND `[B,T,HV]` | NTD `[HV,T]`；TND `[T,HV]` | BF16/FP32，跟随 layout |
| `a_log` | `[HV]` | `[HV]` | 仅 kernel 内计算 gate 时必传 |
| `dt_bias` | `[HV*128]` | `[HV*128]` | 可选，逻辑 shape 为 `[HV,128]` |

`cu_seqlens` 和 `chunk_indices` 是可选 host int array，不是 tensor。前者 shape 为
`[sequence_count+1]`；后者若提供，扁平长度必须为 `2*total_chunks`。

A2 支持四种 layout、g/beta 的 BF16/FP32 组合、Exp/Exp2 和四档输出策略；
实现位于 `op_kernel/arch22/`。本次设备验证范围为 A2，A3 尚需独立验证；A5 沿用 arch35 实现。

`B/T/HK/HV` 必须为正数且可由 `uint32_t` 表示；GVA 要求 `0 < HK <= HV` 且
`HV % HK == 0`。所有输出固定为 head-major：

`layout` 同时描述 q/k/v/g/beta。仅 BSND/TND 的跨 head DMA 要求
`(HK-1)*128*2`、`(HV-1)*128*2`、`(HV-1)*128*sizeof(g)` 和
`(HV-1)*sizeof(beta)` 不超过 `UINT32_MAX`；BNSD/NTD 按 head 连续读取。
不根据 shape 猜测或自动切换布局语义，也不增加整张输入的布局转置。

| 输出 | dense shape | 无 batch shape | dtype |
| --- | --- | --- | --- |
| `gk` | `[B,HV,T,128]` | `[HV,T,128]` | FP32 |
| `Aqk/Akk` | `[B,HV,T,64]` | `[HV,T,64]` | BF16 |
| `w/u/qg/kg/qg_scaled` | `[B,HV,T,128]` | `[HV,T,128]` | BF16 |
| `q_hat/k_hat` | `[B,HK,T,128]` | `[HK,T,128]` | BF16 |
| `q_rstd/k_rstd` | `[B,HK,T]` | `[HK,T]` | FP32 |
| `beta_eff` | `[B,HV,T]` | `[HV,T]` | FP32 |

返回顺序固定为 13 个槽位；未按当前策略保留的槽位返回 `None`：

```text
gk, Aqk, Akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

算子 IR 和 kernel ABI 中这 13 个输出全部声明为 `REQUIRED`，因此输出编号以及 workspace、
tiling 参数位置不会随策略变化。L2 接口允许未保留的七项传 `nullptr`；L0 在交给 launcher
前使用不会被当前编译实例写入的合法 descriptor 占位，避免部分 CANN 版本压缩空输出。

按实际消费者分类时，同一个公开输出可以同时属于正向与反向保存量：

| 类别 | 数据 | 消费方 |
| --- | --- | --- |
| 后续正向使用 | `gk/w/u/kg`；`Aqk/qg_scaled` | FwdH；Finalize |
| 反向使用或保存 | `q_hat/k_hat/q_rstd/k_rstd/beta_eff/Aqk/Akk/gk/w/qg/kg` | KDA backward 或重计算策略 |
| 后续算子的可选状态结果 | `h/final_state` | 由 FwdH 产生，不属于 Prepare 的 13 个输出 |

其中 `qg_scaled` 只服务正向 Finalize；`Akk/qg` 在 Prepare 完成 Post-WU 后不再被
正向消费。`u` 服务 FwdH。`gk/Aqk/w/u/kg/qg_scaled` 是跨算子正向流水必需数据，
不受反向策略影响。

### 输出保留策略

`backward_mode` 只控制反向检查点的公开分配和 GM 写回，支持以下四档：

| `backward_mode` | 场景 | 13 个返回槽位中非空的数据 |
| --- | --- | --- |
| `"none"` | 完全不需要反向 | `gk/Aqk/w/u/kg/qg_scaled` |
| `"forward"` | 只要公开必选的 `Akk`，不需要反向重计算中间量 | 上述六项，加 `Akk` |
| `"recompute"` | 有反向，允许重计算 chunk-local 中间量 | 上述六项，加 `Akk/q_hat/k_hat/q_rstd/k_rstd/beta_eff` |
| `"save"` | 有反向，不重计算 | 全部 13 项；相对 `recompute` 额外保存 `qg` |

默认值为 `"save"`，保持原来 13 项全部返回的兼容行为。各档都执行相同的前向数学计算、
静态 UB/L1 布局和跨核同步，仅关闭未请求结果的公开 GM 写回；`Akk/qg` 等数据在本算子
内部仍按 C4/C5/C7 或 BF16 舍入语义使用。A2通过调用内不可变的tiling `outputMode`
选择四档；A5保持编译期输出策略。两者均不借用关闭的公开输出作为scratch。

完整 forward 的 `output_final_state` 与 `return_intermediate_states` 是和
`backward_mode` 正交的用户输出开关，只控制 `final_state/h`。这两个数据由 FwdH/完整
forward 产生，不属于 Prepare 的 13 个输出，也不会进入 Prepare 的 tiling 或 kernel 分支。

## 模式

- `use_qk_l2norm_in_kernel=false`：`q_hat=q`、`k_hat=k`，有效 token 的 rstd 为 1。
- `use_gate_in_kernel=false`：`g` 是自然对数域的单 token gate step；算子仍执行 cumsum。
- `use_gate_in_kernel=true`：由 `g + dt_bias` 和 `a_log` 计算 Softplus 或 SafeSigmoid gate。
- `use_beta_sigmoid_in_kernel=false`：`beta_eff=fp32(beta)`。
- `use_beta_sigmoid_in_kernel=true`：`beta_eff=sigmoid(beta)`；启用
  `allow_neg_eigval` 时为 `2*sigmoid(beta)`。
- `use_exp2=false/true`：分别在自然对数域或 log2 域保存累计 gate，数学语义等价。

`safe_gate=true` 只允许和 `use_gate_in_kernel=true` 一起使用，此时 `lower_bound` 必须位于
`[-5,0)`。`allow_neg_eigval=true` 必须同时启用 `use_beta_sigmoid_in_kernel`。

## 变长序列

`cu_seqlens` 是 host int array，必须从 0 开始、以总 token 数结束并单调不减。
`chunk_indices` 若提供，必须严格等于按 `chunk_size=64` 生成的
sequence-major `(sequence_id, local_chunk_id)` 序列。rank-4 变长容器要求 `B=1`。

## 实现

各平台共用公开接口，内部调度分别实现。

A2 使用 value-head pair：一个 AIC 配两个 AIV，每个 AIV 准备一个完整 value head。
两个 lane 映射到不同 Q/K head 时分别准备 Q/K；奇数 head 的 inactive lane 仍参与事件协议。
六个 Norm/Gate key 共用 BF16 resident relay、W/U inverse 复用、paired L0 和两代 GM workspace。
普通/GVA、四布局及变长尾块使用同一计算流水。具体资源和事件见 [A2 设计](docs/design_a2.md)。

A5 保留原有 MIX AIC/AIV 八阶段流水，见 [A5 设计](docs/design.md)。
