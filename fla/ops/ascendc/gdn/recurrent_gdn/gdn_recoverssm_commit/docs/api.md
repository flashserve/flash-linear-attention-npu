# GdnRecoverssmCommit 接口与支持范围

## 参考来源

- 参考提交：`0935884b3bc1e24ed36194539a9a420438587b39`。
- 参考实现：
  `vllm_ascend/ops/triton/gdn/gdn_recoverssm_commit.py`。
- 参考测试：
  `test_gdn_recoverssm_commit_npu.py` 和 `gdn_recoverssm_test_utils.py`。

本算子把 speculative GDN 窗口产生的 replay 记录逆序组合回 checkpoint。
`replay` 只读，更新后的 checkpoint 原地写回 `state`。

## 接口签名

GE/Ascend C 算子名为 `GdnRecoverssmCommit`。公开入口为：

```python
from fla_npu.ops.ascendc import gdn_recoverssm_commit

gdn_recoverssm_commit(
    state,
    replay,
    state_commit_cache_address_stride_table,
    state_indices,
    commit_lens,
    final_state_indices,
    boundary_state_indices,
    boundary_recovery_lens,
    *,
    spec_query_len,
    align_mode,
) -> None
```

CANN 根据 OpDef 自动生成以下 Stable-ABI 内部入口。

```text
aclnnInnerGdnRecoverssmCommitGetWorkspaceSize
aclnnInnerGdnRecoverssmCommit
```

| 位置 | 参数 | 方向 | Shape | dtype | 说明 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `state` | 输入、原地输出 | `[P,HV,128,128]` | BF16 或 FP32 | 分页 checkpoint；算子只写计划指定的页 |
| 1 | `replay` | 输入、只读 | `[P,HV,L,257]` | FP32 | `[correction 128, normalized key 128, decay 1]` |
| 2 | `state_commit_cache_address_stride_table` | 输入 | `[2,2,Layer]` | INT64 | checkpoint/replay 的基地址与 block stride（元素计） |
| 3 | `state_indices` | 输入 | `[B]` | INT32 或 INT64 | replay 来源页 |
| 4 | `commit_lens` | 输入 | `[B]` | INT32 或 INT64 | 接受 token 数，调用方保证位于 `[1,L]` |
| 5 | `final_state_indices` | 输入 | `[B]` | INT32 或 INT64 | final checkpoint 目标页，必须为正数 |
| 6 | `boundary_state_indices` | 输入 | `[B]` | INT32 或 INT64 | boundary 目标页；`0` 表示无目标 |
| 7 | `boundary_recovery_lens` | 输入 | `[B]` | INT32 或 INT64 | 从窗口起点恢复的 boundary 记录数 |
| 8 | `spec_query_len` | 属性 | 标量 | INT64 | `1 <= L <= 32` |
| 9 | `align_mode` | 属性 | 标量 | BOOL | 是否同时写 boundary checkpoint |

`state` 和 `replay` 的首维必须一致。地址表每列描述一个 layer：

```text
[0,0,l] = checkpoint data_ptr(l)
[0,1,l] = checkpoint stride(0)
[1,0,l] = replay data_ptr(l)
[1,1,l] = replay stride(0)
```

所有 layer 共用 `state` 的 head/value/key stride 和 `replay` 的
head/position/payload stride。

## 数学语义

对一个 `(layer, value head, request)`，令 `n = min(commit_lens[b], L)`。
从 `position = n-1` 倒序到 `0`：

```text
correction = replay[source, h, position, 0:128]
key        = replay[source, h, position, 128:256]
decay      = replay[source, h, position, 256]

weighted_key = key * accumulated_decay
accumulated_correction += outer(correction, weighted_key)
accumulated_decay *= decay
```

最后：

```text
initial = state[source, h]
result  = initial * accumulated_decay[None, :] + accumulated_correction
state[final, h] = result
```

实现与参考 Triton 的 `S_initial * final_decay + final_correction`
浮点求值顺序一致。checkpoint 初态先转换为 FP32，整个组合过程保持 FP32，
只在 BF16 checkpoint 最终写回时舍入一次。

## `align_mode`

- `False`：只写 `final_state_indices`。
- `True`：当 boundary 页有效时，额外从窗口起点组合
  `boundary_recovery_lens` 条记录，并写 boundary checkpoint。
- 如果 boundary 与 final 是同页且恢复长度等于有效 commit 长度，两次结果相同，
  只保留 final 写回。
- boundary 先于 final 写回；因此 boundary 与 final 目标重叠时 final 结果获胜，
  与参考实现一致。

## 支持范围

- SoC：`ascend950`。
- value head 数 `HV >= 1`；每个实现 program 独立处理一个 value head。
- `K=V=128`，`1 <= spec_query_len <= 32`，`Layer >= 1`。
- checkpoint：BF16 或 FP32；replay：FP32。
- `state` 的 key 维和 `replay` 的 payload 维 stride 为 1。
- `replay` 的 position stride 必须至少为 `257`，用于一次多 block `DataCopyPad`
  搬运整个 window。
- page、head、value/position 维允许正 stride；不允许内部重叠。
- 地址表必须连续，INT64，shape 为 `[2,2,Layer]`。
- `state_indices`、`commit_lens`、`final_state_indices`、
  `boundary_state_indices` 和 `boundary_recovery_lens` 必须为连续的一维张量。
- `state_indices`、`final_state_indices` 和有效 boundary 目标页必须在
  `[1,P)`；kernel 不执行设备侧页号回退。

## 边界与错误行为

- 调用方保证每个 request 的 `1 <= commit_lens <= spec_query_len`。
- 调用方保证 source/final page 合法且写区间不冲突。
- `boundary_state_indices == 0` 表示该 request 不写 boundary；非零值必须在
  `[1,P)`，且 `boundary_recovery_lens <= commit_lens`。
- host 在 descriptor 创建前检查 shape、dtype、stride、设备、地址表布局和 UB。
- 地址表中的设备指针值无法在 host 做内容校验，必须由上下文创建者保证合法。
- kernel 对上述合法契约不增加运行期防御分支。

## 性能口径

正确性首版以一个 AIV 处理一个 `(request, layer, value head, value tile)`，
每个 value tile 覆盖 64 行。性能结论使用 `msprof` 的
`Task Duration(us)`，不以 Python wall time 作为结论。PR 的主场景为
`B=12, HV=8` 且若干层；正式性能基线需在部署环境用同一 run 包和输入采集。
