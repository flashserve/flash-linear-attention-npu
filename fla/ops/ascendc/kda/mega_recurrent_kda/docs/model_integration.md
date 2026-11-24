# MegaRecurrentKda 模型接入指南

在 KDA 推理层，将 BF16 `hidden_states` 直接传给 `npu_mega_recurrent_kda`，一次调用得到 BF16 输出投影和本层更新后的状态。Atlas A2 使用 INT8 QKV，Ascend 950（A5）使用 MXFP8 QKV。两者共用 BSH、变长 TH 和同一套状态管理方式。

## 1. 在模型仓定位替换范围

从模型仓根目录搜索调用点，再沿实际张量的生产者和消费者追踪到 KDA 层及其权重加载器。

```bash
rg -n -i 'dynamic_?(mx_?)?quant|quant_?(batch_?)?matmul|causal_conv1d|kda_recoverssm|mega_kda_projected|megakdaprojected|layer_norm_gated' .
```

| 位置 | 接入动作 |
|---|---|
| 两路投影共同读取的 BF16 hidden states | 取这份输入。融合调用承担 QKV 的动态量化及矩阵乘，同时承担原 BF16 旁路的 decay、beta、输出 gate 投影。 |
| QKV 投影之后，输出投影之前 | 同一次调用完成四抽头因果卷积、KDA 递推、逐 head 门控 RMSNorm。替换掉这一分支原有的调用及只为它服务的拆列、拷贝和中间张量准备。已有 `MegaKdaProjected` 的模型也应从投影前接入。 |
| BF16 输出投影之后 | 将新返回的 `out` 接给原来消费输出投影结果的代码。保留模型原有的集合通信、残差和后续层。某个中间结果还有其它消费者时，单独保留该消费者需要的计算。 |

沿用模型已有的后端选择位置。在推理模式下，设备与 QKV 权重匹配下表且其余输入满足接口契约时启用融合路径，其余情况沿用模型原实现。在任何状态写入前完成路由选择。模型若要求额外返回完整更新后的 checkpoint，先核实其与 replay 的状态协议。

## 2. 在权重加载器和状态管理器完成映射

令 `H=7168`、`D=128`、`P=N*D`，其中 N 是当前设备执行的 head 数。每层在加载时按当前 rank 绑定 Q、K、V 顺序的权重与配对的 scale，激活动态量化由融合算子完成。

| 加载到本层 `weights` 的字段 | Atlas A2 · INT8 | Ascend 950（A5）· MXFP8 |
|---|---|---|
| `qkv_projection_weight` | 直接复用 INT8 NZ 权重。支持逻辑形状 `[H,3P]` 的 FRACTAL_NZ 张量，或已打包的连续四维张量 `[3P/32,H/16,16,32]`。 | 复用逻辑形状 `[1,H,3P]` 的 E4M3FN FRACTAL_NZ 权重。若加载器拿到逻辑 ND 权重，确认方向后在加载时用 `torch_npu.npu_format_cast(w, 29)` 转为 NZ。 |
| `qkv_projection_weight_scale` | BF16 向量 `[3P]`，每个输出通道一个 scale。 | 连续 ND 张量 `[1,H/64,3P,2]`，沿 H 每 32 个元素一个 scale，使用 E8M0 或相同位模式的 uint8。 |
| `qkv_scale_alg` | 省略此键，沿用默认值 0。 | 保存模型动态量化使用的 `scale_alg`（0 或 1）。 |

例如 A2 模型批次 `B=8,N=6,S=4,D=128` 共 32 个 token，QKV 投影为 `[32,7168] → [32,2304]`。加载器直接绑定模型的 `[72,448,16,32]` INT8 权重和 `[2304]` BF16 scale。A5 的 `T=120,N=12` 示例对应逻辑权重 `[1,7168,4608]`、NZ 存储 `[1,144,448,16,32]` 和 scale `[1,112,4608,2]`。

以下输入由两种量化路径共用。张量并行时使用当前 rank 的权重和状态分片。

| 模型数据 | 公开参数与准备方式 |
|---|---|
| decay、beta、输出 gate 的原始权重 | `decay_projection_a_weight[D,H]`、`decay_projection_b_weight[P,D]`、`beta_projection_weight[N,H]`、`output_gate_projection_weight[P,H]`，均为 BF16。若模型加载时把它们合并成一次 MatMul，应在合并前绑定这四个权重。N=12 时合并矩阵的形状为 `[3084,7168]`；单独持有该矩阵时，须从 checkpoint 或加载器取得原始 decay 因子，切分 3084 行无法恢复这两个因子。 |
| 卷积、归一化、输出投影、衰减参数 | `conv1d_weight` 取原卷积权重（常用 `[4,3P]`），`output_norm_weight[D]` 为 FP32，`output_projection_weight[H,P]` 为 BF16，`a_log[N]` 和 `dt_bias[N,D]` 为 FP32。标量 `scale`、`lower_bound`、`rms_norm_eps` 取当前层实际值。 |
| 每层已有的可写状态 | `conv_state[slots,L,3P]` 为 BF16；`replay_u`、`replay_k`、`replay_decay` 的形状均为 `[slots,N,capacity,D]`，依次为 FP32、FP16、FP32。绑定原有存储，保留槽号和槽步长。 |
| 每层只读的 checkpoint | `checkpoint_state[slots,N,D,D]` 为 FP32，最后两维的语义为 V 行、K 列。 |
| 当前批次的设备元数据 | `state_indices[B]` 是请求对应的物理槽号，`conv_num_accepted_tokens[B]` 是每条序列已接受的 token 数，均为设备 INT32。TH 另外提供连续的设备 INT32/INT64 前缀 `cu_seqlens[B+1]`；BSH 使用 `None`。 |

例如，A5 的卷积状态池 `[492,6,4608]` 对应 492 个槽，按接口契约最多容纳长度 4 的序列。槽数、批大小与 head 数分别取自各自的数据结构。其他形状、合法范围及槽步长见[接口契约](api.md)。

## 3. 修改 forward

按目标设备安装包含相应量化接口的 wheel 和配套 OPP。见[编译与安装指南](../../../../../../docs/离线编译与使用指南.md)。把下段放在模型 forward 的 KDA 层循环之前，输入为当前批次的 `hidden_states` 和 `cu_seqlens`。同一设备、同一批次的 token 顺序和前缀相同时，各层共享 `plan`。下一批或重排后重新生成；生成与消费使用同一 stream，跨 stream 时建立事件依赖。

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
    qkv_scale_alg=weights.get("qkv_scale_alg", 0),
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

1. 从同一模型层取真实权重、输入、前缀和 accepted-token 数。为原路径和融合路径各复制一份相同的初始状态，保持槽映射及步长。
2. 先从融合调用的 `conv_state[slot, 2:2+S]` 取回当前序列卷积前的 BF16 QKV，其中 S 取该序列的实际长度。A2 对照模型的 `DynamicQuantV2 + QuantBatchMatmulV3`（NZ 权重入口为 `aclnnQuantMatmulWeightNz`）。A5 对照相同 `scale_alg` 的 `npu_dynamic_mx_quant + npu_quant_matmul`。[A2 INT8 测试](../../../../../../tests/atk/mega_recurrent_kda/test_int8.py) · [A5 MXFP8 测试](../../../../../../tests/atk/mega_recurrent_kda/test_mxfp8.py)
3. 按模型精度门限比较最终输出及四个原地状态，核对只读 checkpoint、非活动槽和活动槽有效行以外的值逐位相同。再给两路相同的下一步输入，各自沿用上一步更新的状态复验。覆盖模型实际使用的等长及变长前缀，包括序列跨越第 64 行的情况。若 QKV 已匹配而下游有差异，再查历史窗口、accepted-token、checkpoint 方向和门控公式。
4. 精度通过后重新采集 profiling。原模型从动态量化与辅助投影中最早启动的操作计至最后一次输出投影结束。在融合路径的时间线上确认每层只执行一次 KDA，计入全部层的完整调用。TH 另计同一批次共享的一次 metadata。两路使用相同输入、初始状态、层数、预热和设备计时口径。
