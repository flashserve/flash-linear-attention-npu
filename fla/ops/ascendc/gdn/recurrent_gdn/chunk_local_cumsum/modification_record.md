# ChunkLocalCumsum 修改记录

记录日期: 2026-06-23
更新日期: 2026-06-24

## 目标

将 Triton 版本的 `chunk_local_cumsum` 算子迁移到
`/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum`，
保持输入输出和功能一致，要求可编译、可在 Ascend 910B 上运行，并用 Triton 输出作为标杆验证 AscendC 版本精度。

## 源算子

- Triton 源文件:
  `/home/m00913889/codex04/flash-linear-attention-npu/triton_ops/triton_core/cumsum.py`
- 源函数:
  `chunk_local_cumsum`

## 仓库迁移记录

更新时间: 2026-06-24 02:55:00 UTC

已将生成的 AscendC 算子、测试脚本、测试数据、性能数据和说明文档迁移到:

```text
/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum
```

后续维护和验证只使用 `flash-linear-attention-npu` 仓。迁移后完成以下调整:

- Triton golden 脚本改为从 FLA 仓根目录导入 `triton_ops.triton_core.cumsum.chunk_local_cumsum`。
- 精度脚本和性能脚本的 `REPO_ROOT` 均解析到 `/home/m00913889/codex04/flash-linear-attention-npu`。
- 精度脚本和性能脚本不再硬编码 `.run` 包文件名，改为从 FLA 仓 `build/_CPack_Packages/Linux/External/*/packages/vendors/custom_transformer` 动态发现当前构建产物。
- 报告和说明文档中的旧路径已更新为 FLA 仓路径。

迁移后验证命令:

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
SKIP_TRITON_GENERATE=1 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

验证结果:

- 构建结果: 通过
- 精度结果: 30/30 case 通过
- 对拍阈值: `atol=2e-5`, `rtol=2e-5`
- 最大 `max_abs`: `0.000001907`
- 所有 case `bad_count=0`

## 新增文件

| 文件 | 说明 |
| --- | --- |
| `CMakeLists.txt` | 算子目录构建入口 |
| `op_host/chunk_local_cumsum_def.cpp` | 算子原型定义、输入输出和属性注册 |
| `op_host/chunk_local_cumsum_infershape.cpp` | 输出 shape 推导 |
| `op_host/chunk_local_cumsum_tiling.cpp` | host tiling、参数检查、blockDim 计算 |
| `op_kernel/chunk_local_cumsum.cpp` | AscendC AIV kernel 实现 |
| `op_kernel/chunk_local_cumsum_tiling_data.h` | kernel tiling 数据结构 |
| `examples/generate_triton_cumsum_cases.py` | Triton golden 数据生成脚本 |
| `examples/test_aclnn_chunk_local_cumsum.cpp` | ACLNN 精度对拍 C++ 程序 |
| `examples/run_precision_test.sh` | 一键生成 golden、准备 OPP、编译并运行精度测试 |
| `operator_description.md` | 算子说明文档 |
| `precision_report.md` | 30 组 case 精度验证报告 |
| `test_report.md` | 测试摘要和结果明细 |
| `interface_alignment_report.md` | Triton 与 AscendC 入参对齐说明 |
| `modification_record.md` | 本修改记录 |

## 功能实现

- 算子名: `ChunkLocalCumsum`
- 输入:
  - `g`: `float32`, shape `[B, T, H]`
  - `cu_seqlens`: optional `int64`
  - `chunk_indices_out`: optional `int64`
- 输出:
  - `out`: `float32`, shape `[B, T, H]`
- 属性:
  - `chunk_size`
  - `reverse`
  - `scale`
  - `head_first`
  - `output_dtype`

实现功能与 Triton 版本一致:

- `reverse=false`: 在每个 chunk 内从 chunk 起点累加到当前位置。
- `reverse=true`: 在每个 chunk 内从当前位置累加到 chunk 终点。
- `scale` 默认为 `1.0`，等价于 Triton `scale=None`。
- 支持固定长度输入和 `cu_seqlens/chunk_indices_out` varlen 输入。

## 接口对齐修改

为贴近 Triton 函数签名，已完成以下调整:

- optional input 从 `chunk_indices` 改名为 `chunk_indices_out`。
- 删除 AscendC 旧有的 `has_scale` 属性。
- 新增 `head_first` 属性，默认 `false`。
- 新增 `output_dtype` 属性，默认 `"float32"`；当前仅支持 `float32`、`torch.float`、`torch.float32`。
- ACLNN 当前生成接口:

```cpp
aclnnChunkLocalCumsumGetWorkspaceSize(
    g,
    cuSeqlensOptional,
    chunkIndicesOutOptional,
    chunkSize,
    reverse,
    scale,
    headFirst,
    outputDtypeOptional,
    out,
    workspaceSize,
    executor)
```

说明: Triton 的 `chunk_indices_out` 是 Python dict；AscendC/ACLNN 层不能接收 dict，因此这里接收的是 `chunk_indices_out[str(block_t)]` 解析后的 `int64` tensor。

## Tiling 修改点

- `block_t` 与 Triton 原算子保持一致:

```text
block_t = next_power_of_2((1 << 17) / (H * chunk_size))
```

- 固定长度模式:

```text
num_blocks = ceil(T / block_t)
block_dim = min(AIV 核数, ceil(B * T * H / 16))
```

- varlen 模式:

```text
num_blocks = chunk_indices_out.element_count / 2
block_dim = 1
```

- tiling 内直接通过 `context->GetPlatformInfo()` 获取 AIV 核数，避免 ACLNN 动态路径中 `compileInfo` 为空导致 `GetWorkspaceSize` 失败。

## Kernel 修改点

- 使用 AIV kernel。
- fixed 路径按输出线性地址处理，每 16 个 `float` 作为一个 64B GM cache line 任务粒度。
- fixed 路径从输出 offset 反推 `(batch, local_t, h)`，再按当前 token 所属 chunk 计算局部 cumsum。
- varlen 路径按 `chunk_indices_out` 读取 `(seq_id, block_id)`，再通过 `cu_seqlens` 获取序列边界。
- scale 在写回前统一处理。

## 精度修复记录

### optional input 处理

问题:
固定长度模式下 ACLNN generated glue 对 optional 输入传 `nullptr` 会报错。

处理:
测试程序固定长度模式传空 `int64` tensor `{0}`，tiling 通过 optional 输入元素数判断是否进入 varlen。

### custom OPP 加载

问题:
`ASCEND_CUSTOM_OPP_PATH` 指向 OPP 根目录时，运行时找不到自定义算子 binary info。

处理:
测试脚本将 `ASCEND_CUSTOM_OPP_PATH` 指向 vendor 目录:

```text
.../vendors/custom_transformer
```

### compileInfo 为空

问题:
ACLNN 动态路径中 `compileInfo` 为空，`GetWorkspaceSize` 返回失败。

处理:
tiling 中直接从 `context->GetPlatformInfo()` 读取 AIV core 数。

### 910B 多核写相邻小尾块

问题:
30 组测试扩展后，`fixed_h8_c64` 出现精度失败。该 case 中 batch 边界处相邻输出落在同一条 64B GM cache line，但由不同 AIV core 写，触发写冲突。

处理:
fixed kernel 改为 16 个 `float` 一组分配任务，保证同一条 64B cache line 只由一个 AIV core 写。varlen 路径当前采用保守单核调度，避免不对齐序列边界出现同类问题。

## 测试用例修改

测试 case 从 5 个扩展到 30 个，覆盖:

- fixed 和 varlen
- `reverse=false/true`
- 带 `scale` 和不带 `scale`
- `chunk_size=16/32/64`
- 多 batch
- 单 chunk
- 多 block
- tail block
- 不同 H 维度，包括 `H=8/16/32/64/96/128`

## 精度对比方法

1. `examples/generate_triton_cumsum_cases.py` 调用 Triton 原算子生成 golden。
2. Triton 输出先与 PyTorch CPU reference 对比，阈值为 `atol=3e-5, rtol=3e-5`。
3. 通过后保存输入和 golden 到 `examples/testdata`。
4. `examples/test_aclnn_chunk_local_cumsum.cpp` 读取 case manifest，调用 ACLNN 自定义算子。
5. 将 AscendC 输出拷回 host，与 Triton golden 逐元素比较。
6. AscendC 对 Triton 的对拍阈值为 `atol=2e-5, rtol=2e-5`。

## 编译和测试命令

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

## 验证结果

- 编译结果: 通过
- 运行平台: Ascend 910B
- 精度结果: 30/30 case 通过
- 对拍阈值: `atol=2e-5`, `rtol=2e-5`
- 最大 `max_abs`: `0.000001907`
- 所有 case `bad_count=0`

详细结果见:

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/precision_report.md`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/testdata/summary.json`

## flash_gated_delta_rule 风格性能 case

更新时间: 2026-06-24

新增 `examples/generate_triton_cumsum_cases.py --extra-fgd-case`，按
`examples/flash_gated_delta_rule.py` 中 `chunk_local_cumsum(g, ...)` 的调用习惯构建
`g=[B,T,H]` case:

- case: `fgd_g_64_128_512_c16`
- shape: `[64, 128, 512]`
- `chunk_size`: 16
- `reverse`: false
- `scale`: 1.0
- `cu_seqlens`: none

说明:

- 对 `H=512`，如果沿用 `chunk_size=64`，源码公式得到 `BLOCK_T=4`，
  小于 `chunk_size`，不满足当前 Triton/AscendC 实现约束。
- 尝试 `chunk_size=4` 时 Triton 编译阶段出现 UB overflow。
- 因此本次采用当前源码可编译运行的最大有效 `chunk_size=16`，此时 `BLOCK_T=16`。

脚本更新:

- `examples/run_performance_profile.sh` 支持 `EXTRA_FGD_CASE=1`。
- 开启 `EXTRA_FGD_CASE=1` 且未指定 `CASE_DIR` 时，性能数据默认写入
  `examples/perfdata`，避免覆盖 30 个默认精度 case。

运行命令:

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
EXTRA_FGD_CASE=1 PERF_CASES="fgd_g_64_128_512_c16" \
  bash fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_performance_profile.sh
```

验证结果:

- Triton golden 对 CPU reference: `max_ref_diff=0`
- AscendC 对 Triton golden: `max_abs=0`, `max_rel=0`, `bad_count=0`
- `msprof op` Task Duration:
  - Triton: `32.820 us`
  - AscendC: `63045.078 us`

详细结果见:

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.md`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.json`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_compare`

## AscendC 计算逻辑优化

更新时间: 2026-06-24

优化目标:

- 不再使用 `GetValue` 对输入 `g` 做 GM 标量逐元素求和。
- `scale=1.0` 的常用路径不执行 scale 乘法。
- 保持 fixed/varlen、prefix/reverse 和非 1 scale 兼容。

修改:

- `op_kernel/chunk_local_cumsum.cpp`
  - 原实现按输出元素调用 `gGm_.GetValue` 重复读取同一 chunk 内历史行。
  - 新实现按 `(batch/block, chunk, H_tile)` 切任务。
  - 每行 H tile 使用 `DataCopy`/`DataCopyPad` 搬入 UB。
  - chunk 内使用 `Add`/`Adds` 在 UB 上向量累计。
  - `scale==1.0` 直接从 accumulator 写回 GM；`scale!=1.0` 保留 `Muls` 兼容路径。
- `op_host/chunk_local_cumsum_tiling.cpp`
  - blockDim 从按 cache line 数切分改为按 chunk/H tile 任务数切分。
- `examples/run_precision_test.sh`
  - 增加 `SKIP_TRITON_GENERATE=1`，可复用已有 Triton golden。
- `examples/run_performance_profile.sh`
  - 增加 `SKIP_TRITON_GENERATE=1`。
- `examples/test_aclnn_chunk_local_cumsum.cpp`
  - 增加 `CUMSUM_DUMP_MISMATCH=1` 调试输出。

验证:

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
SKIP_TRITON_GENERATE=1 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

结果:

- 编译: 通过
- 精度: 30/30 case 通过
- 最大 `max_abs`: `0.000001907`
- 所有 case `bad_count=0`

性能 case:

- case: `fgd_g_64_128_512_c16`
- shape: `[64, 128, 512]`
- `chunk_size=16`
- `scale=1.0`

`msprof op` 结果:

- 优化前 AscendC: `63045.078 us`
- 优化后 AscendC: `77.560 us`
- 加速比: `812.856x`
- 大 shape 精度: `max_abs=0`, `max_rel=0`, `bad_count=0`

详细结果见:

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/optimization_report.md`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.md`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.json`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/ascendc_optimized_final`
