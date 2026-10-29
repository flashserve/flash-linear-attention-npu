# PreProcessFwdKernelMerged API

CP（context parallel）前处理算子：把一个打包窗口压成**仿射链** `hm = [h | m]`（`h` 为 `K×V`、
`m` 为 `K×K`），一次调用产出窗口内每条链的边界状态。跨 rank 的 `all_gather` 与合并由编排层完成，
不在本算子内。语义来源、逐 Stage 设计与对标差异见 [`design.md`](design.md)；
支持的场景、显式拦截与已知限制见算子 [`README.md`](../README.md)。

## 1. 算子原型

```text
PreProcessFwdKernelMerged(
    k, w, u,                  # k [1, HK, T, K] / w [1, HV, T, K] / u [1, HV, T, V]，均 BF16
    g? / gk?,                 # 二选一：g [1, HV, T] 或 gk [1, HV, T, K]，FP32 / BF16
    cu_seqlens,               # host list[int]，本窗口的段边界 [N+1]
    chunk_size = 64,
) -> hm[Nseq, HV, K, V+K]     # FP32；Nseq = len(cu_seqlens) - 1
```

- `B ≡ 1`（CP 契约：varlen 打包窗口）；`K = V = 128`、`chunk_size = 64` 为固定规格，其它值 host 拦截。
- `cu_seqlens` **必给**、严格递增，**允许子区间**（`cu[0] > 0` 或 `cu[-1] < T`）——与竞品跨卡调用
  `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` 的形态一致。
- 算法族 **GDN（`g`）+ KDA（`gk`）**；**DPLR 不支持**：`v` / `bg` 是 DPLR 专用参数，必须为空。

## 2. 参数

### 2.1 输入

布局 BNSD `[B, H, T, D]`，`B ≡ 1`。`HK` / `HV` 为 key / value 侧 head 数，**必须成倍数**
（`HV ≥ HK` 且 `HV % HK == 0`，即 GVA：`k` 在 `HK` 维，`w`/`u`/gate 在 `HV` 维，内部按
`hk = hv // (HV/HK)` 取 key head）。

| 参数 | dtype | Shape | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `k` | BF16 | `[1, HK, T, K]` | 是 | raw key；`gk` 路径同样按 `HK` 头（gate 侧才按 `HV`），GVA 下 `HK < HV` 合法 |
| `w` | BF16 | `[1, HV, T, K]` | 是 | erase/WY 输出；`h` 与 `m` 的左矩阵 |
| `u` | BF16 | `[1, HV, T, V]` | 是 | 取值来源（GDN/KDA 下 `v` 与 `u` 为同一张量，故只收 `u`） |
| `g` | FP32 / BF16 | `[1, HV, T]` | 二选一 | 标量 gate，**base-2 的 chunk 内累积对数衰减**；与 `gk` 互斥 |
| `gk` | FP32 / BF16 | `[1, HV, T, K]` | 二选一 | 逐 K gate（按 value head 给），同为 base-2 chunk 内累积量；与 `g` 互斥 |
| `v` | BF16 | `[1, HV, T, V]` | 否 | **DPLR 专用位，本版不支持**：必须为 `None`，传非空直接拒绝 |
| `bg` | BF16 | `[1, HK, T, K]` | 否 | **DPLR 专用位，本版不支持**：必须为 `None`，传非空直接拒绝 |
| `cu_seqlens` | host `list[int]` | `[N+1]` | 是 | 本窗口段边界：严格递增，`0 ≤ cu[0] < cu[-1] ≤ T`（允许子区间） |

`cu_seqlens` 是 **host 侧整型数组，不是张量**（`list[int]` → `at::OptionalIntArrayRef` → aclnn
`aclIntArray`），不占 GM 带宽；边界校验与每段 `bos` / `T_win` / `NT` 的展开都在 host tiling 里完成。

### 2.2 属性

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `64` | 本版固定 `64`，其它值 host 拦截 |
| `AFFINE_CHAIN_PRECISION` | - | `ieee` | 固定 FP32 累加（`tf32x3` 不支持），不作为参数暴露 |

### 2.3 输出

| 参数 | dtype | Shape | 说明 |
| --- | --- | --- | --- |
| `hm` | FP32 | `[Nseq, HV, K, V+K]` | 每条链 × 每个 head 的 `[K, V+K]`：左 `[0, V)` 为 `h`，右 `[V, V+K)` 为 `m` |

```text
hm[i, hv]                     : [K, V+K] = [128, 256]
  ├─ h = hm[i, hv][:, 0:V]     : [K, V]   = [128, 128]   行 = key 维、列 = value 维（带 stride 的视图，不是连续块）
  └─ m = hm[i, hv][:, V:V+K]   : [K, K]   = [128, 128]   传递矩阵，初值 I
```

**前导维是链条数 `Nseq`（不是 batch）**：`Nseq = len(cu_seqlens) - 1`。单段窗口 `Nseq = 1`，
去掉 size-1 维后与竞品跨卡调用的 `[HV, K, V+K]` 逐字节相同；多段时第 `i` 份 == 竞品针对第 `i` 段
单独调用一次的结果。`Nseq` 同时是并行度乘数（`Nwork = Nseq × HV`）。竞品 `hm` 的三种形态
（跨卡单 part / zigzag / 卡内切段）与逐处核对见 [`design.md` 1.4](design.md)。

## 3. aclnn 接口

```c
aclnnStatus aclnnPreProcessFwdKernelMergedGetWorkspaceSize(
    const aclTensor *k, const aclTensor *w, const aclTensor *u,
    const aclTensor *gOptional, const aclTensor *gkOptional,
    const aclTensor *bgOptional, const aclTensor *vOptional,
    const aclIntArray *cuSeqlensOptional, int64_t chunkSize, const aclTensor *hmOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPreProcessFwdKernelMerged(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`bgOptional` / `vOptional` 是 DPLR 专用位，**必须传 `nullptr`**；传非空会在 host 校验阶段直接返回
`ACLNN_ERR_PARAM_INVALID`。`gOptional` / `gkOptional` 二选一。

Python 入口（`fla_npu.ops.ascendc` 同时导出带 `npu_` 前缀与不带前缀两个名字）：

```python
from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged

hm = pre_process_fwd_kernel_merged(          # hm: [Nseq, HV, K, V+K] FP32
    k, w, u,
    gk=gk,                                   # 与 g 二选一
    cu_seqlens=[0, 512],                     # 允许子区间，如 [40, 512]
    chunk_size=64,                           # 固定 64
)                                            # v / bg 必须省略（DPLR 不支持）
```

接入落点（三条，均已落地）：

| # | 文件 | 改动 |
| --- | --- | --- |
| ① | `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` | 加 `_GET_WORKSPACE_ARGTYPES["aclnnPreProcessFwdKernelMerged"]`；加 `npu_pre_process_fwd_kernel_merged(...)`（`bg`/`v` 传非空抛 `NotImplementedError`；`hm` 由 Python 侧预分配后作为输出张量传入） |
| ② | `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` | `_ASCENDC_OPS` 注册 `"npu_pre_process_fwd_kernel_merged"`（自动导出两个名字） |
| ③ | `torch_custom/fla_npu/csrc/src/stable_pre_process_fwd_kernel_merged.cpp` + `stable_ops.cpp` | Stable-ABI 适配器与注册（`#include` / `m.def` / `m.impl`） |

## 4. 模板参数与 tilingKey

TilingKey 的唯一位是 `GATE_MODE`（host 侧 `SetTilingKey(gateMode + 1)`，kernel 侧由
`ASCENDC_TPL_SEL` 选模板实例，运行时不作 `TILING_KEY_IS` 分支）：

| 模板参数 | 取值 | 含义 |
| --- | --- | --- |
| `GATE_MODE` | `1` / `2` | `USE_G`（GDN 标量门控）/ `USE_GK`（KDA 逐 K 门控） |

**可达 2 个 TilingKey**（`3 = USE_BG`（DPLR）只保留模板槽位，host 永不产生）。
gate 的存储 dtype 不进 TilingKey：aclnn 层把 BF16 gate 统一 Cast 成 FP32 后下发，kernel 用
运行期 `tiling->gateDtype` 选择标量/向量路径。TilingKey 与 gate 模式的完整推导见
[`design.md` 3.2.1](design.md)。

## 5. 返回码与拦截

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | `k` / `w` / `u` / `hm` / `cu_seqlens` 为空（本算子 varlen-only，`cu_seqlens` 必给） |
| `ACLNN_ERR_PARAM_INVALID` | `bg` / `v` 非空（DPLR 不支持）；`g` 与 `gk` 同给或同缺；`K != 128`；`V != 128`；`chunk_size != 64`；`B != 1`；`HK > HV` 或 `HV % HK != 0`；`w` / `u` 的 shape 与 `k` 不匹配；`cu_seqlens` 元素数 < 2、非严格递增、含零长段或越界（`cu[i] < 0` 或 `cu[i] > T`）；空窗口（`T = 0`）；gate dtype 非法 |

`cu[0] > 0` 与 `cu[-1] < T`（**子区间窗口**）是合法用法，不拦截。`K` / `V` 的尾块由掩码处理，
`hm` 的越界区域不写、不参与比较。

## 6. 典型 shape

`K = V = 128`、`chunk_size = 64`，`hm` 展开为 `[Nseq, HV, 128, 256]` FP32。

| 场景 | `HK` / `HV` | `T` | gate | `cu_seqlens` | `hm` |
| --- | --- | --- | --- | --- | --- |
| 整窗单段（CP 单 part） | 32 / 32 | 512 | `g` | `[0, 512]` | `[1, 32, 128, 256]` |
| 变长打包（3 段） | 32 / 32 | 384 | `gk` | `[0, 64, 192, 384]` | `[3, 32, 128, 256]` |
| 子区间窗口（竞品用法） | 8 / 8 | 512 | `g` | `[40, 512]` | `[1, 8, 128, 256]` |
| GVA（1:2） | 2 / 4 | 256 | `gk` | `[0, 256]` | `[1, 4, 128, 256]` |
| 长窗口模型档 | 32 / 32 | 16384 | `g` | `[0, 16384]` | `[1, 32, 128, 256]` |

验收用例（111 条精度 + 4 条性能 + 5 条确定性/内存）见
[`tests/atk/pre_process_fwd_kernel_merged/`](../../../../../../tests/atk/pre_process_fwd_kernel_merged/)。
