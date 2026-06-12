# KDA Triton 算子合入说明书

**目标仓库**: `flash-linear-attention-npu-v2`  
**合入日期**: 2026-06-11  
**迁移更新**: 2026-06-12（KDA 目录从 `fla/ops/kda/` 迁移至 `fla/ops/triton/triton_core/kda/`）  
**验证状态**: ✅ NPU 验证通过，test_kda_chunk 34/34 用例全部 PASS

---

## 一、合入概述

### 1.1 合入目标

将源仓库 `Theta/flash-linear-attention` 中的 KDA (Kimi Delta Attention) Triton 算子迁移至目标仓库 `flash-linear-attention-npu-v2`，使目标仓库支持 `chunk_kda` 和 `fused_recurrent_kda` 算子调用。

### 1.2 核心原则

| 原则 | 说明 |
|------|------|
| **文件隔离** | KDA 依赖的共享模块全部放入 `fla/ops/triton/triton_core/kda/` 子目录，通过相对引用访问 |
| **零侵入** | 不修改目标仓库任何现有文件（`__init__.py`、`utils.py`、`cumsum.py` 等） |
| **1:1 直接拷贝** | KDA 特有文件直接从源仓库复制，仅修改 import 路径指向隔离内部副本 |
| **NPU 兼容** | 设备检测、GPU 特性标志、autocast 等均已适配 NPU |

### 1.3 合入结果

| 指标 | 数值 |
|------|------|
| 合入 KDA 核心文件 | 10 个 |
| 隔离共享依赖文件 | 8 个 |
| 隔离公共算子文件 | 2 个（+ 1 计划外发现） |
| GLA 依赖文件 | 1 个 |
| CP Stub 文件 | 2 个 |
| `__init__.py` 文件 | 4 个 |
| 测试文件 | 1 个（`test_kda_chunk.py`，34 用例全 PASS） |
| 总代码行数 | ≈8,043 行 |
| Import 重映射 | 50 处（初始）+ 55 处（迁移至 triton_core 后二次重映射） |
| 修改目标仓库已有文件 | **0 个** |

---

## 二、目录结构

合入后 `fla/ops/triton/triton_core/kda/` 的完整目录结构：

```
fla/ops/triton/triton_core/kda/
├── __init__.py                          # 导出 chunk_kda, fused_recurrent_kda
│
├── chunk.py                             # 主入口 ChunkKDAFunction + chunk_kda API
├── chunk_fwd.py                         # chunk 前向逻辑
├── chunk_bwd.py                         # chunk 反向逻辑（含 Triton kernel）
├── chunk_intra.py                       # intra-chunk 注意力矩阵（含 Triton kernel）
├── chunk_intra_token_parallel.py        # token 并行 intra-chunk kernel
├── fused_recurrent.py                   # 融合递推解码 kernel
├── gate.py                              # gate 计算（含 Triton kernel）
├── wy_fast.py                           # WY 表示重计算 kernel
├── naive.py                             # PyTorch 参考实现（仅依赖 einops）
│
├── _kda_utils/                          # KDA 隔离的共享工具（NPU 适配版）
│   ├── __init__.py                      # 统一导出（65行）
│   ├── utils.py                         # 来自 fla/utils.py（NPU 适配版，615行）
│   ├── index.py                         # 来自 fla/ops/utils/index.py
│   ├── op.py                            # 来自 fla/ops/utils/op.py
│   ├── constant.py                      # 来自 fla/ops/utils/constant.py
│   ├── softplus.py                      # 来自 fla/ops/utils/softplus.py
│   ├── cumsum.py                        # 来自 fla/ops/utils/cumsum.py（完整版含 vector/global）
│   └── l2norm.py                        # 来自 fla/modules/l2norm.py
│
├── _kda_common/                         # KDA 依赖但目标仓库缺失的公共算子
│   ├── __init__.py                      # 导出 chunk_gated_delta_rule_fwd_h, chunk_fwd_h, chunk_bwd_dh
│   ├── chunk_delta_h.py                 # 来自 fla/ops/common/chunk_delta_h.py（移除 @dispatch）
│   └── chunk_h.py                       # 来自 fla/ops/common/chunk_h.py（计划外发现依赖）
│
├── _kda_gla/                            # KDA 依赖的 GLA chunk 算子
│   ├── __init__.py                      # 导出 chunk_gla_fwd_o_gk
│   └── chunk.py                         # 来自 fla/ops/gla/chunk.py（完整拷贝，1384行）
│
└── _kda_cp/                             # Context Parallel stub（NPU 版不支持 CP）
    ├── __init__.py                      # FLACPContext 空类 + build_cp_context
    └── chunk_delta_h.py                 # 4 个 CP 函数 stub

tests/ops/
└── test_kda_chunk.py                    # Megatron 调用约定的 chunk_kda 测试（485行，34 用例）
```

---

## 三、Import 重映射规则

### 3.1 初始迁移（源仓库 → `fla.ops.kda.*`）

| 源 import 路径 | 初始目标 import 路径 | 说明 |
|---------------|---------------------|------|
| `fla.utils` | `fla.ops.kda._kda_utils.utils` | 工具函数、设备检测、autocast 等 |
| `fla.ops.utils` / `fla.ops.utils.index` | `fla.ops.kda._kda_utils.index` | `prepare_chunk_indices`, `prepare_chunk_offsets` |
| `fla.ops.utils` (chunk_local_cumsum) | `fla.ops.kda._kda_utils.cumsum` | cumsum 相关 |
| `fla.ops.utils.op` | `fla.ops.kda._kda_utils.op` | exp, exp2, log, log2, gather |
| `fla.ops.utils.constant` | `fla.ops.kda._kda_utils.constant` | RCP_LN2 |
| `fla.ops.utils.softplus` | `fla.ops.kda._kda_utils.softplus` | softplus, softplus2 |
| `fla.ops.utils.cumsum` | `fla.ops.kda._kda_utils.cumsum` | GLA chunk 中使用 |
| `fla.modules.l2norm` | `fla.ops.kda._kda_utils.l2norm` | l2norm_fwd, l2norm_bwd |
| `fla.ops.common.chunk_delta_h` | `fla.ops.kda._kda_common.chunk_delta_h` | chunk 前向 h 计算 |
| `fla.ops.common.chunk_h` | `fla.ops.kda._kda_common.chunk_h` | GLA chunk 依赖 |
| `fla.ops.gla.chunk` | `fla.ops.kda._kda_gla.chunk` | chunk_gla_fwd_o_gk |
| `fla.ops.cp` | `fla.ops.kda._kda_cp` | FLACPContext stub |
| `fla.ops.cp.chunk_delta_h` | `fla.ops.kda._kda_cp.chunk_delta_h` | CP 函数 stub |
| `fla.ops.backends` | **已移除** | `@dispatch` 装饰器一并移除 |

### 3.2 目录迁移（`fla.ops.kda.*` → `fla.ops.triton.triton_core.kda.*`）

KDA 目录从 `fla/ops/kda/` 迁移至 `fla/ops/triton/triton_core/kda/` 后，所有 import 前缀统一替换：

| 旧 import 前缀 | 新 import 前缀 |
|----------------|----------------|
| `fla.ops.kda.` | `fla.ops.triton.triton_core.kda.` |

涉及 16 个 `.py` 文件 + 1 个测试文件，共 55 处替换（含 docstring 中的示例代码），0 处残留。

---

## 四、NPU 适配详情

### 4.1 `_kda_utils/utils.py` 适配

源仓库 `fla/utils.py` 中大量 GPU 专用代码，已在隔离副本中做如下 NPU 适配：

| 适配项 | 修改内容 |
|--------|---------|
| 设备检测 | 添加 `_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()`，优先检测 NPU |
| `device` / `device_torch_lib` | NPU 时设为 `'npu'` / `torch.npu` |
| GPU 特性标志 | NPU 下 `IS_NVIDIA_HOPPER=False`, `IS_AMD=False`, `IS_INTEL_ALCHEMIST=False`, `USE_CUDA_GRAPH=False`, `IS_TF32_SUPPORTED=False`, `IS_TMA_SUPPORTED=False` |
| `check_shared_mem()` | NPU 下直接返回 `True` |
| `autocast` | PyTorch ≥2.4 时 `device='npu'`；<2.4 时 `torch.npu.device(index)` |
| `IS_NPU` 公开别名 | 添加 `IS_NPU = _IS_NPU` 供外部 import |
| GPU 调用安全 | 非 NPU 路径下 `torch.cuda.get_device_name/capability` 均包裹 `try/except` |
| `deprecate_kwarg` | `__version__` 引用添加 `try/except NameError` 保护（目标仓库无 `fla.__version__`） |
| `from fla import __version__` | `TYPE_CHECKING` 块添加 `try/except ImportError` |

### 4.2 `@dispatch` 装饰器移除

`_kda_common/chunk_delta_h.py` 中 `from fla.ops.backends import dispatch` 和 `@dispatch('common')` 已移除。目标仓库无 `fla.ops.backends` 模块，KDA 只使用默认实现路径。

### 4.3 Context Parallel Stub

`_kda_cp/` 提供 CP 接口的空实现，使 KDA 代码可导入但不执行 CP 路径（NPU 版不支持）：

```python
# _kda_cp/__init__.py
class FLACPContext:
    """Stub: Context Parallel is not supported in NPU version."""
    pass

# _kda_cp/chunk_delta_h.py
def chunk_gated_delta_rule_fwd_h_pre_process(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported")
# ... 其余 3 个函数同理
```

KDA 代码中 CP 相关路径均为 `if cp_context is not None:` 条件分支，传入 `None` 即跳过。

### 4.4 测试文件 NPU 适配

| 文件 | 修改 |
|------|------|
| `test_kda_chunk.py` | `from fla.utils import ...` → `from fla.ops.triton.triton_core.kda._kda_utils.utils import ...` |
| `test_kda_chunk.py` | `device` 从 `_kda_utils.utils` 导入（已含 NPU 检测） |
| `test_kda.py`（未合入） | `torch.device("cuda")` → `torch.device("npu" if ... else "cuda")` |

---

## 五、计划外发现与处理

### 5.1 `chunk_h.py` 额外依赖

**发现**：`_kda_gla/chunk.py` 依赖 `fla.ops.common.chunk_h`（`chunk_fwd_h`, `chunk_bwd_dh`），原计划未覆盖。

**处理**：从源仓库拷贝 `fla/ops/common/chunk_h.py` → `_kda_common/chunk_h.py`，修改 import 指向 `_kda_utils`，更新 `_kda_common/__init__.py` 增加导出。

### 5.2 `_kda_utils/__init__.py` 增加导出

相比原计划，额外导出了以下符号：

| 符号 | 原因 |
|------|------|
| `IS_NPU` | NPU 设备检测标志，多处需要 |
| `IS_NVIDIA` | 部分模块引用 |
| `IS_TMA_SUPPORTED` | `_kda_utils/utils.py` 中定义 |
| `USE_CUDA_GRAPH` | `_kda_common/chunk_delta_h.py` 引用 |
| `prepare_chunk_offsets` | `_kda_common/chunk_delta_h.py` 和 `_kda_common/chunk_h.py` 引用 |

### 5.3 `IS_NPU` 命名问题

**问题**：`utils.py` 中 NPU 检测变量为 `_IS_NPU`（私有前缀），`__init__.py` 导出 `IS_NPU` 会失败。

**修复**：添加 `IS_NPU = _IS_NPU` 公开别名赋值（line 460）。

### 5.4 目录迁移至 `triton_core/kda/`

**背景**：为与目标仓库目录结构对齐，KDA 目录从 `fla/ops/kda/` 迁移至 `fla/ops/triton/triton_core/kda/`。

**改动**：
- 物理移动整个 `kda/` 目录
- 批量替换所有 `.py` 文件中的 import 前缀：`fla.ops.kda.` → `fla.ops.triton.triton_core.kda.`
- 更新测试文件 `test_kda_chunk.py` 中的 import
- 更新文档中所有路径引用

---

## 六、执行阶段记录

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 创建 `fla/ops/kda/` 及子目录结构 | ✅ 已完成 |
| Phase 2 | 拷贝 10 个 KDA 核心文件 | ✅ 已完成 |
| Phase 3 | 拷贝 10 个共享依赖 + 4 个 stub/init + 1 个计划外依赖 | ✅ 已完成 |
| Phase 4 | 修改 50 处 import 路径 + 移除 @dispatch + 拷贝 chunk_h.py | ✅ 已完成 |
| Phase 5 | NPU 适配 + @dispatch 移除 + __init__.py 导出重写 | ✅ 已完成 |
| Phase 6 | 测试文件拷贝 + import/device 适配 | ✅ 已完成 |
| Phase 7 | 语法验证 + IS_NPU 修复 + NPU 验证 34 用例全 PASS | ✅ 已完成 |
| Phase 8 | 目录迁移至 `fla/ops/triton/triton_core/kda/` + 二次 import 重映射 | ✅ 已完成 |

---

## 七、验收结果

| # | 验收项 | 结果 |
|---|--------|------|
| 1 | `from fla.ops.triton.triton_core.kda import chunk_kda, fused_recurrent_kda` import 成功 | ✅ PASS |
| 2 | 目标仓库零影响：不修改任何已有文件 | ✅ PASS（0 个已有文件被修改） |
| 3 | `pytest tests/ops/test_kda_chunk.py -v` 全部 PASS | ✅ PASS（34/34） |
| 4 | `pytest tests/ops/test_kda.py -v` | ⏳ 未合入（按计划暂不合入） |
| 5 | py_compile 语法检查全量通过 | ✅ PASS（26 个 .py 文件） |
| 6 | 旧 import 残留 grep 检查 | ✅ PASS（0 处功能性残留） |
| 7 | 目录迁移后 import 一致性检查 | ✅ PASS（`fla.ops.kda.` 零残留，全部指向 `fla.ops.triton.triton_core.kda.`） |

---

## 八、调用方式

### 8.1 基本调用

```python
from fla.ops.triton.triton_core.kda import chunk_kda, fused_recurrent_kda

# chunk_kda: 训练/推理前向+反向
o, final_state = chunk_kda(q, k, v, g, beta, ...)

# fused_recurrent_kda: 推理解码
o, final_state = fused_recurrent_kda(q, k, v, g, beta, ...)
```

### 8.2 Megatron 调用约定

```python
# 推荐 Megatron 参数组合
o, _ = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,  # L2 norm 在外部完成
    use_gate_in_kernel=True,         # gate 在 kernel 内计算
    safe_gate=config.kda_safe_gate,
    lower_bound=config.kda_lower_bound,
)
```

### 8.3 依赖隔离说明

KDA 模块完全自包含，依赖链如下：

```
fla.ops.triton.triton_core.kda
├── chunk_kda / fused_recurrent_kda  (对外 API)
│
├── _kda_utils/    ← 工具函数隔离副本（NPU 适配版 utils.py）
├── _kda_common/   ← 公共算子隔离副本（chunk_delta_h, chunk_h）
├── _kda_gla/      ← GLA chunk 算子隔离副本
└── _kda_cp/       ← CP stub（NPU 不支持，所有函数 raise NotImplementedError）
```

**不会** import 目标仓库已有的 `fla/ops/triton/triton_core/` 中的其他文件（`utils.py`、`cumsum.py`、`l2norm.py` 等）或 `fla/ops/ascendc/` 中的任何内容。

---

## 九、风险项与缓解措施

| # | 风险 | 影响 | 缓解措施 | 状态 |
|---|------|------|---------|------|
| 1 | `_kda_utils/utils.py` 中 GPU 检测逻辑在 NPU 上报错 | NPU 环境无法初始化 | 添加 `_IS_NPU` 检测分支，NPU 下安全默认值；非 NPU 路径 `torch.cuda.*` 包裹 `try/except` | ✅ 已缓解 |
| 2 | `@dispatch` 装饰器依赖 `fla.ops.backends` | 导入失败 | 移除 `@dispatch` 装饰器和 import | ✅ 已缓解 |
| 3 | `_kda_gla/chunk.py` 链式依赖 | 依赖爆炸 | 发现 `chunk_h.py` 额外依赖并拷贝到 `_kda_common/` | ✅ 已缓解 |
| 4 | Triton autotune `num_warps`/`num_stages` NPU 不兼容 | Kernel launch 失败或性能差 | NPU Triton 可能忽略这些参数，或配置 `TRITON_ALL_BLOCKS_PARALLEL=1` | ⚠️ 待观察 |
| 5 | `tl.gather` NPU Triton 不支持 | chunk_intra 运行时错误 | `_kda_utils/op.py` 中 `gather` 已有 fallback 返回 `None` | ⚠️ 待观察 |
| 6 | `TRITON_ABOVE_3_4_0` 等版本检测与 NPU Triton 不兼容 | autotune_cache 不可用 | 添加版本兼容性检测 | ⚠️ 待观察 |
| 7 | `test_kda.py` 硬编码 `torch.device("cuda")` | 测试在 NPU 上失败 | 未合入此文件（按计划暂不合入），如需合入改为动态 device | ⏳ 按需 |
| 8 | `cu_seqlens_cpu` 参数签名差异 | varlen 测试失败 | 使用隔离的 `_kda_utils/index.py` 保持完整签名 | ✅ 已缓解 |
| 9 | `USE_CUDA_GRAPH` NPU 不适用 | autotune 配置问题 | NPU 下 `USE_CUDA_GRAPH = False` | ✅ 已缓解 |
| 10 | `IS_NPU` 公开别名缺失 | import 失败 | 添加 `IS_NPU = _IS_NPU` | ✅ 已修复 |

---

## 十、后续优化方向（本次不做）

1. **共享算子 merge**：将 `_kda_utils/` 中与目标仓库 `triton_core/` 功能重复的模块逐步统一
2. **AscendC 加速**：将 KDA 中性能瓶颈 kernel 用 AscendC 重写
3. **Context Parallel 支持**：实现完整的 `_kda_cp/` 替代 stub
4. **CP 相关测试合入**：`test_cp_kda.py`
5. **模型级测试合入**：`test_modeling_kda.py`
6. **NPU Triton autotune 优化**：针对 NPU 多 buffer/多核特性优化 autotune config
7. **test_kda.py 合入**：完整 KDA 算子测试（含 fused_recurrent、varlen 等）
8. **`_kda_utils/utils.py` 精简**：移除 KDA 不需要的函数（如 `deprecate_kwarg`、`require_version` 等），减小体积