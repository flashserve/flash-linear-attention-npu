# CP（Context Parallel）算子

本目录存放 Context Parallel（CP）相关的 Ascend C 算子。CP 的目标是把序列维切到多个 rank 上，并让跨 rank 的
状态与状态梯度在算子层可见、可合并。

当前包含：

| 算子 | 作用 | 状态 |
| --- | --- | --- |
| [`chunk_delta_h_bwd_preprocess`](chunk_delta_h_bwd_preprocess/README.md) | 反向状态预处理：把本 rank 边界序列的反向状态递推压缩成仿射摘要 `(E_r, P_r)` | 初版：host + kernel 六 Stage 已实现，aclnn 手写，A2/A5 编译与精度均已打通；Python 入口与列 tile 展平分核待后续 |

## CP 数据流中的位置

```text
每个 rank 的边界序列
   │
   ├─▶ ChunkDeltaHBwdPreprocess（本目录）        → dhm = [E_r | P_r]  (FP32)
   │                                                  │
   │                                        all-gather（框架侧，非算子）
   │                                                  ▼
   ├─▶ ChunkDeltaHBwdMerge（后续算子）           按逻辑时间逆序合并 → dht
   │                                                  │
   └─▶ 正式 chunk backward（已有算子，需消费 dht）  → dh0 / dq / dk / dv / dg
```

只实现 `chunk_delta_h_bwd_preprocess` 不足以完成 CP：正向还需要各 rank 的正向状态摘要与 prefix merge，
反向还需要本 rank 的正式 backward 消费非零 `dht`。详见
[`docs/agents/chunk-delta-h-cp-backward-preprocess.md`](../../../../docs/agents/chunk-delta-h-cp-backward-preprocess.md)
与 [`docs/agents/chunk-delta-h-cp-backward-preprocess-stages.md`](../../../../docs/agents/chunk-delta-h-cp-backward-preprocess-stages.md)。

## 与 Triton 参考实现的关系

上游 fla 的 `pre_process_bwd_kernel_merged`（`fla/ops/cp/chunk_delta_h.py`）一次 launch 同时算 `E_r` 与 `P_r`。
本目录的算子沿用同一数学语义与同一份输入约定，但按 Ascend C 的 Cube/Vector 分工把一次 launch 拆成
`V0 → C1 → V2 → C3 → V4 → C5` 六个 Stage（原八 Stage 版本把 `C1/C5`、`V4/V6` 分开）。

## 新增 CP 算子时的约定

- 目录命名沿用仓库现有风格：`fla/ops/ascendc/cp/<op_name>/`，`<op_name>` 与 `op_host` 目录名、OP_TYPE
  小写形式一致。
- 跨 rank 通信（all-gather / send-recv / scan）不放在算子内；算子只负责本地的摘要生成与本地合并。
- `dhm` 的 `[hv, K, V+K]` 逻辑布局与 `state_v_first` 无关，任何 CP 算子都不得因为状态布局而改写该布局。
- 公开文档、PR、评论中不得出现服务器、账号、绝对路径或调测环境信息。
