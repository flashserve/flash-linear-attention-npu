# Ascend 950 模型接入 MegaRecurrentKda（MXFP8 QKV）

目标是在模型的 KDA 推理层从 BF16 `hidden_states` 开始，用一次 `npu_mega_recurrent_kda` 调用得到 BF16 输出投影和本层更新后的状态。QKV 使用 MXFP8。decay、beta 和输出 gate 沿用分立的 BF16 权重。以下步骤适用于 BSH 和变长 TH。

## 1. 在模型仓定位替换范围

从模型仓根目录搜索调用点，再沿实际张量的生产者和消费者追踪到 KDA 层及其权重加载器。

```bash
rg -n -i 'dynamic_mx_quant|quant_matmul|causal_conv1d|kda_recoverssm|mega_kda_projected|megakdaprojected|layer_norm_gated' .
```

| 位置 | 接入动作 |
|---|---|
| 两路投影共同读取的 BF16 hidden states | 取这份输入。融合调用承担 QKV 的动态量化及矩阵乘，同时承担原 BF16 旁路的 decay、beta、输出 gate 投影。 |
| QKV 投影之后，输出投影之前 | 同一次调用完成四抽头因果卷积、KDA 递推、逐 head 门控 RMSNorm。替换掉这一分支原有的调用及只为它服务的拆列、拷贝和中间张量准备。已有 `MegaKdaProjected` 的模型也应从投影前接入。 |
| BF16 输出投影之后 | 将新返回的 `out` 接给原来消费输出投影结果的代码。保留模型原有的集合通信、残差和后续层。某个中间结果还有其它消费者时，单独保留该消费者需要的计算。 |

沿用模型已有的后端选择位置。Ascend 950、推理模式和下述契约匹配时启用融合路径，其余情况沿用模型原实现。在任何状态写入前完成路由选择。模型若要求额外返回完整更新后的 checkpoint，先核实其与 replay 的状态协议。

## 2. 在权重加载器和状态管理器完成映射

令 `H=7168`、`D=128`、`P=N*D`，N 是当前设备执行的 head 数。下表使用公开接口名作为下一节两个字典的键。权重按层准备一次。张量并行时使用当前 rank 对应的权重和状态分片。

| 模型数据 | 公开参数与准备方式 |
|---|---|
| MXFP8 QKV 权重及其 weight scale | `qkv_projection_weight` 为 E4M3FN，逻辑形状 `[1,H,3P]`，列顺序为 Q、K、V，存储格式为 FRACTAL_NZ。已有 NZ 权重直接复用。有逻辑 ND 权重时，在加载时对形状和方向已确认的 E4M3FN 张量调用 `torch_npu.npu_format_cast(w, 29)`。 |
| 每 32 个 H 元素的权重 scale | `qkv_projection_weight_scale` 为连续 ND `[1,H/64,3P,2]`，使用 E8M0 张量或保留相同位模式的 uint8。模型原动态量化的 `scale_alg` 传给 `qkv_scale_alg`（0 或 1）。激活的 FP8 编码和激活 scale 由算子内部生成。 |
| decay、beta、输出 gate 的原始权重 | `decay_projection_a_weight[D,H]`、`decay_projection_b_weight[P,D]`、`beta_projection_weight[N,H]`、`output_gate_projection_weight[P,H]`，均为 BF16。若模型加载时把它们合并成一次 MatMul，应在合并前绑定这四个权重。N=12 时合并矩阵的形状为 `[3084,7168]`；单独持有该矩阵时，须从 checkpoint 或加载器取得原始 decay 因子，切分 3084 行无法恢复这两个因子。 |
| 卷积、归一化、输出投影、衰减参数 | `conv1d_weight` 取原卷积权重（常用 `[4,3P]`），`output_norm_weight[D]` 为 FP32，`output_projection_weight[H,P]` 为 BF16，`a_log[N]` 和 `dt_bias[N,D]` 为 FP32。标量 `scale`、`lower_bound`、`rms_norm_eps` 取当前层实际值。 |
| 每层已有的可写状态 | `conv_state[slots,L,3P]` 为 BF16；`replay_u`、`replay_k`、`replay_decay` 的形状均为 `[slots,N,capacity,D]`，依次为 FP32、FP16、FP32。绑定原有存储，保留槽号和槽步长。 |
| 每层只读的 checkpoint | `checkpoint_state[slots,N,D,D]` 为 FP32，最后两维的语义为 V 行、K 列。 |
| 当前批次的设备元数据 | `state_indices[B]` 是请求对应的物理槽号，`conv_num_accepted_tokens[B]` 是每条序列已接受的 token 数，均为设备 INT32。TH 另外提供连续的设备 INT32/INT64 前缀 `cu_seqlens[B+1]`；BSH 使用 `None`。 |

当 `T=120,N=12` 时，QKV 权重逻辑形状为 `[1,7168,4608]`，物理形状为 `[1,144,448,16,32]`，权重 scale 为 `[1,112,4608,2]`。卷积状态池 `[492,6,4608]` 对应 492 个槽，按接口契约最多容纳长度 4 的序列。槽数、批大小与 head 数分别取自各自的数据结构。其他形状、合法范围及槽步长见[接口契约](api.md)。

## 3. 修改 forward

安装包含本算子 MXFP8 接口的 A5 wheel 及配套 OPP，步骤见[编译与安装指南](../../../../../../docs/离线编译与使用指南.md)。把下段放在模型 forward 的 KDA 层循环之前，输入为当前批次的 `hidden_states` 和 `cu_seqlens`。同一设备、同一批次的 token 顺序和前缀相同时，各层共享 `plan`。下一批或重排后重新生成；生成与消费使用同一 stream，跨 stream 时建立事件依赖。

```python
from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

plan = (
    npu_mega_recurrent_kda_metadata(cu_seqlens, tokens=hidden_states.shape[0])
    if cu_seqlens is not None else None
)
```

在每个 KDA 层的已选中分支执行下段。`weights` 和 `state` 是按上表绑定的本层对象；可直接把字典读取替换为模型的真实字段。各层使用各自的 hidden states、权重和状态池。

```python
out, conv_state, replay_u, replay_k, replay_decay = npu_mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=weights["qkv_projection_weight"],
    qkv_projection_weight_scale=weights["qkv_projection_weight_scale"],
    qkv_scale_alg=qkv_scale_alg,
    decay_projection_a_weight=weights["decay_projection_a_weight"],
    decay_projection_b_weight=weights["decay_projection_b_weight"],
    beta_projection_weight=weights["beta_projection_weight"],
    output_gate_projection_weight=weights["output_gate_projection_weight"],
    output_norm_weight=weights["output_norm_weight"],
    output_projection_weight=weights["output_projection_weight"],
    conv1d_weight=weights["conv1d_weight"],
    a_log=weights["a_log"],
    dt_bias=weights["dt_bias"],
    conv_state=state["conv_state"],
    checkpoint_state=state["checkpoint_state"],
    replay_u=state["replay_u"],
    replay_k=state["replay_k"],
    replay_decay=state["replay_decay"],
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    cu_seqlens=cu_seqlens,
    metadata=plan,
    scale=scale,
    lower_bound=lower_bound,
    rms_norm_eps=rms_norm_eps,
)
```

`out` 具有与 `hidden_states` 相同的 shape 和 BF16 dtype。后四个返回值与传入的对应状态池共享存储。将 `out` 接回模型，状态管理器继续持有原来的池。

## 4. 用模型输入验收

1. 从同一模型层取真实权重、输入、前缀和 accepted-token 数。为原路径和融合路径分别复制一份初始状态，保持各自的槽映射及步长。使用模型原来的量化算法和精度门限。
2. 对照两条路径的最终输出和更新后的四个状态池。核对只读 checkpoint、非活动槽和活动槽越过有效行的部分逐位相同。随后给两路相同的下一步输入，让各自使用上一步留下的状态，再比较一次。覆盖模型实际使用的等长及变长前缀，包括序列跨越第 64 行的情况。
3. 若出现差异，先从 `conv_state[slot,2:2+S]` 读回该序列的 BF16 QKV，与原模型的量化投影输出对照；投影匹配后，再查历史窗口、accepted-token、checkpoint 方向和门控公式。可复用[五项输出、原地写入和连续两步的测试](../../../../../../tests/atk/mega_recurrent_kda/test_mxfp8.py)。
4. 精度通过后重新采集 profiling。检查每层所选分支只执行一次融合 KDA；TH 的同一批次只需生成一次 metadata。性能按相同的输入、初始状态、层数、预热和设备计时口径比较。原模型取整段开始到最后一次输出投影结束的跨度，融合路径计入一次 metadata 和实际 KDA 层的调用。以真实模型结果记录收益。
