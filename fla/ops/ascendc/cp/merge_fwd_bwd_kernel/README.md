# MergeFwdBwdKernel

[设计文档](docs/design.md) | [API 文档](docs/api.md)

## 功能

`MergeFwdBwdKernel` 把 CP 路径上各 rank 的仿射摘要 `(M, He)` 合并成一块
head 状态 `h`。它消费 `pre_process_fwd_kernel_merged`（或等价前处理）经
all-gather 得到的 `ag_hm`，不负责跨 rank 通信。

稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import merge_fwd_bwd_kernel

h = merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    forward=True,
    state_v_first=False,
)
```

该入口经 Stable-ABI / ctypes 适配层调用 `aclnnMergeFwdBwdKernel`。`h` 由
调用方提供，算子 **inplace 写回**；与 FLA `merge_fwd_bwd_kernel` 一样，
CP 路径不读 `h` 的原值。OpDef 仍把同一块 `h` 登记为输入和输出，图编译
才能看到这次写。

合并公式：

```text
h ← He_0
h ← M_i @ h + He_i    for i = 1 .. N-1
```

`ag_hm` 每个 head 的物理行为 `[K, V+K]`：前 `V` 列是 `He`，后 `K` 列是
`M`。GEMM 与加法在该顺序下进行；`state_v_first` 只改变写给用户的 `h`
物理末两维顺序，不改变 `ag_hm` 布局。

源 rank：

```text
forward:  src(i) = rank - N + i
backward: src(i) = rank + N - i
S = 1 且 N = 1 且 rank = 0: src = 0
```

`N = 1` 时没有矩阵乘，结果就是 `He_0`。

## 输入输出

| 数据 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `ag_hm` | FP32 / BF16 | `[S, HV, 128, 256]` | 每 rank、每 head 的 `[He \| M]` |
| `h` | 与 `ag_hm` 相同 | `[HV, 128, 128]` | inplace 输出缓冲，必须连续 |

`h` 由 kernel 直接写入，非连续在 aclnn 入口拒绝。`ag_hm` 可以不连续，
host 会先收成连续再读。两者的 storage/view format 必须是非私有格式
（ND、NCHW、NCL、NHWC）；NZ 等私有格式在入口直接拒绝。

`K = V = 128`，两种 `state_v_first` 下 shape 都是 `[HV, 128, 128]`，差别
只在 128×128 的元素顺序：`false` 为 `[K, V]`，`true` 为 `[V, K]`。

返回值就是写入后的 `h`。

## 属性

| 名称 | 默认值 | 说明 |
| --- | --- | --- |
| `forward` | `true` | true：从 `rank - N` 向 `rank` 合并；false：反向 |
| `rank` | `0` | 当前 rank，范围 `[0, S)` |
| `pre_or_post_num_ranks` | 必传 | 参与合并的 rank 数 `N`，合法调用要求 `N >= 1` |
| `state_v_first` | `false` | false：`h` 末两维 `[K, V]`；true：`[V, K]` |

## 支持范围

- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。
- `S ∈ [1, 1024]`，`HV ∈ [1, 256]`，`K = V = 128`。
- `forward=true` 要求 `rank >= N`；`forward=false` 要求 `rank + N < S`
  （`S = N = 1`、`rank = 0` 的单段特例除外）。

## 实现

一次 Mix launch（`KERNEL_TYPE_MIX_AIC_1_2`）：Vector 负责打包与加 `He`，
Cube 做 `M @ H`。按 head 奇偶把两个 AIV 拆开，每个 AIV 独占若干完整
head。用户 workspace 为 FP32 scratch `[3, HV, 128, 128]`（`N = 1` 不写
scratch）。A2/A3 与 A5 的流水、L1/UB 布局和跨核 flag 见
[设计文档](docs/design.md)。

## 验证

完整 shape、属性与 aclnn 签名见 [API 文档](docs/api.md)。本算子目前以
接口与设计文档为准；CP 链路中的位置见
[`../README.md`](../README.md)。
