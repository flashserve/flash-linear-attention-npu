# Triton L2Norm 重复编译问题记录

本文记录 `examples/flash_gated_delta_rule.py` 中 Q/K L2Norm Triton 实现的重复编译问题、根因、修复策略、验证方法和后续设计原则。

## 背景

`examples/flash_gated_delta_rule.py` 在 `use_qk_l2norm_in_kernel=True` 时会在 forward 中调用：

- `l2norm_fwd(q)`
- `l2norm_fwd(k)`

并在 backward 中调用：

- `l2norm_bwd(q, q_rstd, dq)`
- `l2norm_bwd(k, k_rstd, dk)`

这些实现位于 `fla/ops/triton/triton_core/l2norm.py`。在变长序列或多组不同 token 数的场景中，原实现会随 `T` 变化反复触发 Triton 编译/autotune，导致 CPU host 侧编译开销明显，整体表现为 host bound。

参考资料：

- [triton-lang/triton#5000](https://github.com/triton-lang/triton/issues/5000)
- [Ascend/MindSpeed-MM#2655](https://gitcode.com/Ascend/MindSpeed-MM/pull/2655/diffs)

## 现象

原 L2Norm 路径使用了类似下面的模式：

```python
@triton.autotune(..., key=['D', 'NB'])
@triton.jit
def l2norm_fwd_kernel(..., T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, ...):
    ...
```

其中：

- `T` 是 flatten 后的 token 数，即 `x.view(-1, x.shape[-1]).shape[0]`。
- `NB = triton.cdiv(T, 2048)`，同样会随 `T` 变化。
- `T: tl.constexpr` 会把序列长度写入 Triton 编译期特化参数。
- `NB` 进入 autotune key 后，不同长度会形成不同 autotune cache key。

因此在 `T=1024, 1536, 2048, ...` 等输入之间切换时，即使 `D` 没变，也可能产生新的 Triton 编译版本。训练或推理端会反复等待 host 编译，NPU 侧不能持续吃满。

## 根因

Triton 的 JIT 编译缓存并不只按 Python 函数名命中。缓存 key 会包含 kernel 源/IR、目标后端、dtype/stride/alignment 等调用信息，以及编译期常量和 meta 参数，例如 `tl.constexpr`、`BLOCK_SIZE`、`num_warps`、`num_stages`、autotune config 和 autotune key。

`tl.constexpr` 的语义是“这个值属于编译期”。它可以让 Triton 做静态 shape 推导、循环展开、常量折叠和更激进的后端优化，但代价是该值变化时需要一个新的特化版本。这个机制本身是正确的：tile 大小、feature dim、是否使用某条静态分支等参数通常就应该成为编译期常量。

本算子的问题在于，`T` 只是实际 token 数，主要用于边界检查和 block pointer 的 runtime shape，并不是决定核心计算 tile 形状的稳定参数。把它设为 `tl.constexpr` 后，变长序列中的高频变化值进入编译缓存维度；再叠加 `NB = f(T)` 进入 autotune key，就会把同一个 `D` 下的不同 `T` 扩散成多个编译版本。

因此更合理的特化维度应该是：

- `D`
- `BD = next_power_of_2(D)`
- `BT`
- dtype、layout、目标后端等真实影响 kernel 代码形态的因素

而不是动态 token 数 `T`，也不是从 `T` 推导出的 `NB`。

MindSpeed-MM PR #2655 的核心方向是删除 L2Norm 的 autotune，并显式传入固定 `BT`，避免 `NB` 和 autotune key 组合引入额外编译。这个方向是正确的。本仓需要保留对不同 `D` 的泛化能力，因此没有把 forward/backward 永久写死成单个 `BT`，而是用稳定的 `D/BD` 选择 `BT`。

## Triton 后端设计原则

### 1. 区分算法动态量和编译特化量

动态量是每次调用都会变的运行时数据，例如实际 token 数、batch 内有效长度、chunk 数、mask 边界等。这些值如果只影响边界判断和 pointer offset，应尽量作为 runtime scalar 传入。

编译特化量是决定 kernel 代码形态的参数，例如 block size、tile shape、静态分支、向量宽度、feature dim 上界、pipeline stage 等。这些值变化后通常确实需要新的编译版本。

### 2. `tl.constexpr` 只用于真正需要静态化的值

`tl.constexpr` 能换来更好的后端优化，但它会进入编译缓存维度。对于变长输入，不能为了写法方便就把 `T`、`N_CTX`、`num_blocks` 这类高频变化值标成 `tl.constexpr`。

如果某个 scalar 需要传给 kernel，但不希望它参与特化，可以使用：

```python
@triton.jit(do_not_specialize=['T'])
```

这样 `T` 仍可用于 runtime 边界判断，但不会因为 `T` 的不同值生成新的 Triton 编译版本。

### 3. Autotune key 必须稳定且必要

`@triton.autotune(key=[...])` 的 key 应只包含会改变最优 config 的稳定参数。把 `T` 或 `NB = f(T)` 放进 key，会导致 autotune 结果和编译缓存按动态长度拆分。

对于轻量 kernel 或者目标平台上 config 空间很小的 kernel，删除 autotune 并使用清晰的启发式通常更稳。这样既减少首次运行开销，也避免动态 shape 扩散缓存。

### 4. Tile 不能只按吞吐扩大

更大的 `BT` 通常能减少 program 数或提高数据复用，但也会增加 UB、寄存器、临时张量和后端调度压力。Ascend 后端对片上缓冲区有明确容量约束；当 tile 过大时，kernel 可能在后端编译阶段失败，而不是运行阶段才暴露。

因此 `BT` 的选择需要同时满足：

- 同一 `D` 下改变 `T` 不改变 `BT`。
- `D` 增大时能自动降低 `BT`。
- forward/backward 分别设置上限，因为 backward 通常有更多中间量和归约开销。
- 在目标后端上验证编译可行性，而不是只看 Python 侧公式。

### 5. 缓存验证应关注“是否产生新特化”

验证这种问题时，关键不是缓存目录里具体有多少文件，而是同一稳定特化维度下，改变动态量是否会生成新的编译产物。建议使用隔离的 Triton 编译缓存目录做 focused test：每个 `D` 的第一个 `T` 允许产生首次编译，第二个不同 `T` 不应继续产生新的特化版本。

公开记录里只保留验证结论和覆盖范围，避免把内部机器、镜像、路径、缓存文件数等环境细节写入 issue/PR。

## 修复策略

当前修复采用四条原则：

1. 删除 L2Norm kernel 上的 `@triton.autotune`。
2. 对 forward/backward kernel 使用 `@triton.jit(do_not_specialize=['T'])`，让 `T` 作为运行时参数传入。
3. 移除由 `T` 推导出的 `NB` 特化参数。
4. `BT` 只依赖 `BD` 和安全上限，不依赖 `T`。

当前选择逻辑：

```python
BT_LIST = (8, 16, 32, 64, 128)
FWD_TARGET_BLOCK_ELEMENTS = 8 * 1024
BWD_TARGET_BLOCK_ELEMENTS = 8 * 1024
FWD_MAX_BT = 32
BWD_MAX_BT = 64

def _select_l2norm_bt(bd: int, target_block_elements: int, max_bt: int) -> int:
    max_bt_by_size = max(1, target_block_elements // max(1, int(bd)))
    max_bt = min(int(max_bt), max_bt_by_size)
    candidates = [bt for bt in BT_LIST if bt <= max_bt]
    return candidates[-1] if candidates else BT_LIST[0]
```

映射示例：

| `BD` | forward `BT` | backward `BT` |
| ---: | ---: | ---: |
| 16 | 32 | 64 |
| 32 | 32 | 64 |
| 64 | 32 | 64 |
| 128 | 32 | 64 |
| 256 | 32 | 32 |
| 512 | 16 | 16 |

这不是按单个 shape 写死，而是按 feature dim 做稳定启发式选择。它满足：

- 同一个 `D` 下改变 `T` 不会改变 `BT`。
- `D` 增大时自动降低 `BT`，避免片上缓存使用过大。
- `D=128` 的主路径仍对应参考修复中的 `32/64` 配置。
- forward 固定上限为 `BT=32`，backward 固定上限为 `BT=64`，同时保留大 `D` 场景下自动收缩的泛化能力。

## 验证

### 静态检查

本地执行：

```bash
python -m py_compile fla/ops/triton/triton_core/l2norm.py
python -m py_compile examples/flash_gated_delta_rule.py fla/ops/triton/triton_core/l2norm.py
git diff --check
```

### Focused cache 验证

在 Ascend NPU 验证环境中使用隔离的 Triton 编译缓存，对同一个 `D` 连续运行两个不同 `T`。预期是每个 `D` 的第一个 `T` 允许产生首次编译，第二个 `T` 不应新增编译特化。

覆盖结果：

| `D` | 覆盖的 `T` 变化 | 结果 |
| ---: | --- | --- |
| 64 | 两个不同 token 长度 | 通过；第二个长度未产生新的编译特化 |
| 128 | 两个不同 token 长度 | 通过；第二个长度未产生新的编译特化 |
| 512 | 两个不同 token 长度 | 通过；第二个长度未产生新的编译特化 |
| 768 | 两个不同 token 长度 | 通过；第二个长度未产生新的编译特化 |

同时确认 forward/backward 输出均为 finite。

### Example/ST 验证

在 Ascend NPU 验证环境中执行 quick build，并运行小 shape 的 `examples/flash_gated_delta_rule.py` Example/ST。该流程完成 quick build、custom OPP 安装、`fla_npu` 构建和 Example/ST，退出码为 0。

## 后续维护建议

1. 新增 Triton kernel 时，先判断参数是算法动态量还是编译特化量。
2. 避免把动态 token 数、batch 内实际长度、chunk 数等高频变化值标成 `tl.constexpr`。
3. `@triton.autotune` 的 `key` 只放真正决定最优 config 的稳定参数，不要放由 `T` 推导出的 `NB`。
4. 需要运行时 scalar 且不希望参与特化时，优先考虑 `@triton.jit(do_not_specialize=[...])`。
5. 选择 `BT` 时同时考虑吞吐和后端片上缓存约束；forward/backward 分开设上限，并覆盖大 `D` 场景。
6. 对变长场景增加 focused cache 验证，确认同一稳定特化维度下改变动态量不会新增编译版本。
