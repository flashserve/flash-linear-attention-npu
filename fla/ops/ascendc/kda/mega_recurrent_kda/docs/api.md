# MegaRecurrentKda 接口

Atlas A2/A3 和 Ascend 950 共用的推理接口。固定隐藏维 H=7168，head 维 D=128，head 数 N 由权重和状态形状确定。令 P=N×D，B 为序列数，T 为 token 总数。

```python
out, conv_state, replay_u, replay_k, replay_decay = npu_mega_recurrent_kda(
    hidden_states, qkv_projection_weight, decay_projection_a_weight,
    decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight,
    output_norm_weight, output_projection_weight, conv1d_weight, conv_state,
    checkpoint_state, replay_u, replay_k, replay_decay, state_indices,
    conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens=cu_seqlens,
    scale=scale, lower_bound=lower_bound, rms_norm_eps=rms_norm_eps,
)
```

| 输入 | 类型 | 形状与约定 |
|---|---|---|
| hidden_states | BF16 | BSH 为 `[B,S,H]`，TH 为 `[T,H]` |
| qkv_projection_weight（BF16 路径） | BF16 | `[3P,H]`，按 Q、K、V 排列 |
| decay_projection_a_weight / decay_projection_b_weight | BF16 | `[D,H]` / `[P,D]` |
| beta_projection_weight / output_gate_projection_weight | BF16 | `[N,H]` / `[P,H]` |
| output_norm_weight / output_projection_weight | FP32 / BF16 | `[D]` / `[H,P]` |
| conv1d_weight | BF16 | `[4,3P]`、`[3,N,4,D]` 或跨 head 共享的 `[3,4,D]` |
| conv_state | BF16 | `[slots,L,3P]`，L 至少为最长序列加 2 |
| checkpoint_state | FP32 | `[slots,N,D,D]`，以 V 行、K 列保存，只读 |
| replay_u / replay_k / replay_decay | FP32 / FP16 / FP32 | `[slots,N,capacity,D]`，capacity 覆盖最长序列 |
| state_indices / conv_num_accepted_tokens | INT32 | 各为 `[B]`。物理槽互异且落在 [0,slots)，已接受 token 数落在 [0,L−2] |
| a_log / dt_bias | FP32 | `[N]` / `[N,D]`，输入值有限 |
| cu_seqlens | INT32 或 INT64 | TH 使用长度 B+1 的严格递增前缀，首尾为 0 和 T。BSH 传 `None` |

BF16 路径除五个状态池的槽轴可留间隙外，各张量按行主序连续。一个槽的尾部不能与下一个槽重叠。每条序列至少含一个 token。scale、lower_bound、rms_norm_eps 须可用有限 FP32 表示，lower_bound 在 [−5,0] 内，rms_norm_eps 至少为最小正正规 FP32 值。

out 为新分配的 BF16 张量，形状与输入相同。其余四项返回原地更新的输入。每个活动槽仅写 conv_state 的前 S+2 行和各 replay 的前 S 行。其他槽、槽间填充和 checkpoint 保持原值。状态池用于推理。

C++ 先调用 `aclnnMegaRecurrentKdaGetWorkspaceSize` 再调用 `aclnnMegaRecurrentKda`。ACLNN 的三个整数元数据参数使用 host 端 `aclIntArray`。Python 接口接受 CPU 或当前设备上的元数据张量，设备元数据在调用前按当前 stream 顺序取回。元数据已在 CPU 的调用方可直接复用。

## Atlas A2/A3 与 Ascend 950 设备元数据接口

TH 布局可在一次 forward 的 KDA 层循环前生成序列调度表。同一设备上，各层的 token 排列和前缀值一致时可以共享该表。批次重排、前缀更新或 token 数变化后重新生成。生成和消费使用同一 stream，跨 stream 的调用方须建立事件依赖，并让调度表存活到消费结束。

```python
from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

plan = npu_mega_recurrent_kda_metadata(cu_seqlens, tokens=hidden_states.shape[0])
for layer_inputs in kda_layers:
    outputs = npu_mega_recurrent_kda(**layer_inputs, cu_seqlens=cu_seqlens, metadata=plan)
```

`cu_seqlens` 为设备上连续的 INT32 或 INT64 向量。输出为连续 INT32 向量，长度为 `2 + ceil(T/16)`。前两项记录是否跨越 64 行和最长序列长度，其余各项依次记录第 0、16、32… 行的序列编号。T=120 时有效数据为 40 B。调用方保证前缀、活动槽和卷积历史窗口符合上表约定。各层提供自己的状态池，设备直接读取 INT32 的 `state_indices` 与 `conv_num_accepted_tokens`。

BSH 用 `npu_mega_recurrent_kda(..., use_device_metadata=True)` 直接从 B、S 计算分块，同时从设备读取两个 INT32 向量。C++ 分别使用 `aclnnMegaRecurrentKdaMetadata` 和 `aclnnMegaRecurrentKdaV2` 的两段式接口。BSH 在 V2 中将前缀和调度表均传空。

## Ascend 950 MXFP8 QKV 投影

在 `npu_mega_recurrent_kda` 中传入 E4M3FN 权重和配对的 `qkv_projection_weight_scale` 启用 MXFP8。输入激活保持 BF16。QKV 沿 H 每 32 个元素动态量化为 E4M3FN，使用 E8M0 激活与权重 scale 作矩阵乘，向卷积交接 BF16。decay-A、decay-B、beta 和输出 gate 分别使用接口表中的 BF16 权重。模型集成时沿用这四个输入。

| 参数 | 模型输入 |
|---|---|
| `qkv_projection_weight` | E4M3FN，逻辑形状 `[1,H,3P]`，FRACTAL_NZ 存储形状 `[1,3P/32,H/16,16,32]`，通道按 Q、K、V 排列 |
| `qkv_projection_weight_scale` | 连续 ND E8M0，形状 `[1,H/64,3P,2]`。K 方向第 g 个分组、输出列 c 对应 `[0,g//2,c,g%2]`。Python 同时接受 E8M0 原始字节的 uint8 张量 |
| `qkv_scale_alg` | 0（默认）或 1，与模型 `npu_dynamic_mx_quant(..., scale_alg=...)` 取相同值 |

BSH 和 TH 均使用设备上的状态索引和已接受 token 数。TH 支持上节的跨层调度表。单独调用时也可直接提供设备上的前缀，由 Python 入口生成本次调度表。C++ 入口为 `aclnnMegaRecurrentKdaMxGetWorkspaceSize` 和 `aclnnMegaRecurrentKdaMx`。该路径额外申请 `T × 3P × 2` 字节用户 workspace，用于保存 BF16 QKV。T=120、N=12 时增加 1,105,920 字节。

权重加载、完整的 forward 调用和连续两步对照方法见[模型接入指南](model_integration.md)。

## Atlas A2/A3 INT8 QKV 投影

向同一 Python 入口传入 INT8 `qkv_projection_weight` 和 BF16 `qkv_projection_weight_scale[3P]`。权重的列顺序为 Q、K、V。支持逻辑形状 `[7168,3P]` 的 FRACTAL_NZ 张量，也支持模型已经打包好的连续四维张量 `[3P/32,448,16,32]`。通道 scale 与列一一对应。`qkv_scale_alg` 沿用默认值 0。

融合 kernel 从原始 BF16 hidden states 读取每个 token，动态量化为 INT8，使用 INT32 累加，再以 FP32 token scale 和通道 scale 反量化到 BF16，送入四抽头因果卷积。两级 decay、beta 和输出 gate 仍读取原始 BF16 输入。BSH、TH 和跨层 metadata 的用法与上节相同。C++ 使用 `aclnnMegaRecurrentKdaInt8GetWorkspaceSize` 和 `aclnnMegaRecurrentKdaInt8`。

例如模型批次 `B=8,N=6,S=4,D=128` 的投影是 `[32,7168] → [32,2304]`，对应 NZ 权重 `[72,448,16,32]`，scale 为 `[2304]`。可从 `conv_state[slot,2:2+S]` 读回实际送入卷积的 BF16 QKV，与模型的 `DynamicQuantV2 + QuantMatmulWeightNz` 逐项对照。
