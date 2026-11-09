# PreProcessBwdKernelMerged API

## 1. 算子原型

```text
PreProcessBwdKernelMerged(
    q, k, w, d_o, dv, g?, gk?, cu_seqlens?,
    scale=1.0, chunk_size=64
) -> dhm
```

## 2. 参数

### 2.1 输入

| 参数 | dtype | Shape | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `q` | BF16 / FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `q`；KDA/GDN2 传已应用 key gate 的 `qg` |
| `k` | BF16 / FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `k`；KDA/GDN2 传 `kg`；必须与 `q` 同 shape |
| `w` | BF16 / FP16 | `[B, Hv, T, K]` | 是 | WY 辅助量 `w_wy`；不是公开输入 `w_gate` |
| `d_o` | BF16 / FP16 | `[B, Hv, T, V]` | 是 | 输出梯度 `dO` |
| `dv` | BF16 / FP16 | `[B, Hv, T, V]` | 是 | `dv_local`：不含 `K̄ @ dH` 项的本地 `dV` |
| `g` | BF16 / FP16 / FP32 | `[B, Hv, T]` | 否 | GDN 的 chunk-local 累计标量 log2 gate |
| `gk` | BF16 / FP16 | `[B, Hv, T, K]` | 否 | KDA/GDN2 的 chunk-local 累计逐 K log2 gate |
| `cu_seqlens` | INT64 | `[N+1]` | 否 | 本 segment 的 `[bos, eos)`；传入时要求 `B == 1` |

`g` 与 `gk` 必须二选一或同时缺省；同时非空时 host 直接拦截。

### 2.2 属性

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `scale` | float | 1.0 | 只乘 `Q̄ᵀ dO` 项；通常取 `K^-0.5`。缺省时按 `1.0` 处理并把 `isScale` 置 0 |
| `chunk_size` | int | 64 | 必须与 gate cumsum、WY 生成、正式 backward 一致；本版只支持 64 |

### 2.3 输出

| 参数 | dtype | Shape | 说明 |
| --- | --- | --- | --- |
| `dhm` | FP32 | `[Hv, K, V+K]` | `[..., 0:V] = E_r`；`[..., V:V+K] = P_r` |

`dhm` 的布局与 `state_v_first` 无关；本算子不做原地写入，输入张量均只读。

## 3. aclnn 接口

```c
aclnnStatus aclnnPreProcessBwdKernelMergedGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w, const aclTensor *dO, const aclTensor *dv,
    const aclTensor *gOptional, const aclTensor *gkOptional, const aclIntArray *cuSeqlensOptional,
    double scale, int64_t chunkSize, const aclTensor *dhmOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPreProcessBwdKernelMerged(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

调用示例：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
auto ret = aclnnPreProcessBwdKernelMergedGetWorkspaceSize(
    q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize, dhm,
    &workspaceSize, &executor);
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
ret = aclnnPreProcessBwdKernelMerged(workspace, workspaceSize, executor, stream);
```

Python 主入口（目标形态）：

```python
from fla_npu.ops.ascendc import pre_process_bwd_kernel_merged

dhm = pre_process_bwd_kernel_merged(
    q=qg, k=kg, w=w_wy, d_o=do, dv=dv_local, gk=gk,
    scale=K ** -0.5, chunk_size=64, cu_seqlens=cu_seqlens,
)
```

## 4. 模板参数与 tilingKey

tilingKey 由模板参数组合得到（host 侧 `GET_TPL_TILING_KEY(D_T_Q, D_T_G, GATE_MODE)`，声明见
[`op_kernel/pre_process_bwd_kernel_merged_tiling_key.h`](../op_kernel/pre_process_bwd_kernel_merged_tiling_key.h)）；
kernel 侧不使用 `TILING_KEY_IS` 分支，由 key 直接选中模板实例。

| 模板参数 | 取值 | 含义 |
| --- | --- | --- |
| `D_T_Q` | `10` / `20` | `q`/`k`/`w`/`d_o`/`dv` 的 dtype：BF16 / FP16 |
| `D_T_G` | `10` / `20` / `30` | 标量 gate `g` 的 dtype：BF16 / FP16 / FP32；无门控与 `gk` 路径取与 `D_T_Q` 相同 |
| `GATE_MODE` | `0` / `1` / `2` | 无门控 / 标量门控 `g` / 逐 K 门控 `gk` |

`ASCENDC_TPL_SEL` 共放开 8 种合法组合（BF16/FP16 × {无门控、标量 g(BF16)、标量 g(FP32)、gk}）。

## 5. 返回码与拦截

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` (`161001`) | `workspaceSize` 或 `executor` 输出指针为空；沿用旧接口行为，tiling/launcher 失败导致 L0 结果为空也映射为此状态 |
| `ACLNN_ERR_PARAM_INVALID` (`161002`) | 沿用旧接口行为，必选 `q`/`k`/`w`/`d_o`/`dv`/`dhm` 为空；新增边界检查包括 `dhm` shape/dtype/连续性非法及 host `cu_seqlens[0:2]` 越界 |
| `ACLNN_ERR_INNER` (`561000`) | 二阶段执行器运行失败；保持原接口映射，下层真实 rc 记录在 PLOG，原生首错消息不被外层覆盖 |

正式 EZ 消息表达根因，公开返回值独立保持现有兼容映射。不能仅凭 `161001` 推断根因是用户传空指针。

| 根因 | 正式消息 |
| --- | --- |
| 必选 Tensor 或输入 shape 缺失 | `EZ0004`，包含参数名和接口名 |
| `workspaceSize`/`executor` 输出指针为空 | `EZ0037`，包含参数名、接口名及可写输出指针要求 |
| 输入 shape 存在但 descriptor 缺失（包括已提供的 g/gk） | `EZ1008`，标明具体参数和 TilingContext 元数据构造原因 |
| shape、正维度、rank 错误 | `EZ0008` / `EZ0009` / `EZ0011` |
| dtype 不符 | `EZ0020` |
| 属性、列表长度、segment 边界及组合关系 | `EZ0024` / `EZ0025` / `EZ0026` / `EZ0027` |
| 非连续 `dhm` 输出 | `EZ0028`，实际 stride 与连续输出约束 |
| host 规划缺少框架资源、平台返回零核 | `EZ1008`，包含算子名和资源/规划原因 |

host tiling 继续统一检查输入 shape/dtype、`Hv % Hk == 0`、`K=V=128`、`chunk_size=64`、
门控互斥、门控 shape/dtype、metadata rank/长度及 `B=1`。读取 shape 维度前先验证 rank 和正维度。
`chunk_size` 按 `int64_t` 读取并检查，不能将 `(1<<32)+64` 截断为 64。

kernel 使用 uint32 偏移和 chunk ceiling arithmetic，因此 `T <= 4294967232`；超过此界限由 host 以
`EZ0026` 拒绝。ACLNN 的 host `cu_seqlens` 前两项要求 `0 <= bos <= eos <= T` 且可表示为 uint32。
`dhm` 必须是 FP32 连续输出；当前 kernel 不支持对任意 stride 的输出视图写回。

当前定向补丁不新增 C++ function-try-block 或修改框架分配异常承接方式；沿用基线。
测试中的原生执行器返回161001与公开执行接口返回561000是两个层次，不能混用。

报错文本会给出实际 shape、`Hk`/`Hv` 与触发的约束，便于定位；`g`/`gk` 同时非空的报错会明确指出二者互斥。

## 6. 典型 shape

shape 与取值范围的唯一维护处是 [README](../README.md) 的「支持的场景」「不支持（本版显式拦截）」「已知限制」三节；
下表只是典型调用档位示例。

| 场景 | `Hk`/`Hv` | `K` | `V` | `chunk_size` | `dhm` |
| --- | --- | --- | --- | --- | --- |
| GDN / KDA 主线 | 96 / 96 | 128 | 128 | 64 | `[96, 128, 256]` FP32（12 MiB） |
| KDA Kimi K3 档位 | 96 / 96 | 128 | 128 | 64 | 同上 |
| GVA | 32 / 96 | 128 | 128 | 64 | `[96, 128, 256]` |
