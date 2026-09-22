# MergeFwdBwdKernel API

Context-parallel merge of per-rank `(M, He)` pairs into one head state.

```text
h ← He_0
h ← M_i @ h + He_i    for i = 1 .. N-1
```

`He` is the leading `V` columns of `ag_hm` and `M` is the trailing `K` columns.
Both stay `[K, V]` and `[K, K]`. The GEMM runs in that order. `state_v_first`
only changes the physical order of the caller-owned `h` buffer.

## 签名

```python
merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    *,
    forward=True,
    state_v_first=False,
)
```

```cpp
aclnnStatus aclnnMergeFwdBwdKernelGetWorkspaceSize(
    const aclTensor *h,
    const aclTensor *agHm,
    int64_t preOrPostNumRanks,
    int64_t rank,
    bool forward,
    bool stateVFirst,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

与 FLA `merge_fwd_bwd_kernel` 一样，只有一块 `h`。CP 路径不读取 `h` 的原值，结果写回这块内存。OpDef 仍把同一个 `h` 登记为输入和输出，图编译才能看到这次写。

## 输入

| 名称 | dtype | shape | 语义 |
| --- | --- | --- | --- |
| `ag_hm` | FP32 / BF16 | `[S, HV, 128, 256]` | 每个 rank 的 `He`（前 128 列）和 `M`（后 128 列） |
| `h` | 与 `ag_hm` 相同 | `[HV, 128, 128]` | inplace 输出缓冲 |

## 属性

| 名称 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `forward` | bool | true | true：从 `rank - N` 向 `rank` 合并；false：反向 |
| `rank` | int | 0 | 当前 rank，范围 `[0, S)` |
| `preOrPostNumRanks` | int | 0 | 参与合并的 rank 数 `N`，合法调用要求 `N >= 1` |
| `state_v_first` | bool | false | false：`h` 末两维为 `[K, V]`；true：末两维为 `[V, K]` |

`K = V = 128`，两种 layout 的 shape 都是 `[HV, 128, 128]`，差别只在 128×128 的元素顺序。

## 输出

返回值就是写入后的 `h`。调用方传入这块缓冲，和 FLA 一样。

## 支持范围

`S ∈ [1, 1024]`，`HV ∈ [1, 256]`，`K = 128`，`V = 128`。SoC：`ascend910b`、`ascend910_93`、`ascend950`。
