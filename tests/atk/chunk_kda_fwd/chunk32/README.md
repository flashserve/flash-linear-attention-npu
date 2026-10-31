# chunk_kda_fwd / arch35 chunk32 ATK

本目录验证统一 `chunk_kda_fwd` 算子在 Ascend950 上的 chunk32 单 kernel 融合路径。
NPU 调用经过公开 Python 入口 `fla_npu.ops.ascendc.chunk_kda_fwd`，底层进入
`aclnnChunkKdaFwdV2`；正向精度用例不直接调用 fused 私有入口。CPU 侧使用同一 executor 中的
FP64 golden 和同精度 benchmark。

## 契约与 TilingKey

- `chunk_size=32` 仅用于 Ascend950 / A5；在 `ChunkKdaFwd` 内使用 TilingKey `32` 和独立 chunk32 TilingData。
- q/k/v/o 为 BF16，`K=V=128`；公开 V2 的 g/beta 各自可为 BF16 或 FP32，任一为 FP32 时统一上转。
- 支持 dense batch、varlen、GVA、非 128 对齐尾块、initial/final state 和 NTD 原生布局。
- `safe_gate=true`、`use_gate_in_kernel=true`，必须提供 `A_log`；`dt_bias=None` 等价于零偏置。
- `use_qk_l2norm_in_kernel=true`、`use_beta_sigmoid_in_kernel=true`、
  `allow_neg_eigval=false`、`use_exp2=true`，epsilon 固定为 `1e-6`。
- 不导出 recompute 中间态或反向保存值；`Aqk`/`Akk` 返回 `None`，`chunk_indices`
  仅接受 canonical sequence-major 顺序。

| 算子路径 | TilingKey | 选择与源码证据 | 本子套件用例 |
| --- | --- | --- | --- |
| 既有 `ChunkKdaFwd` 非 fused 路径 | `1/2` | `fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp` 按模板条件执行 `SetTilingKey(... ? 2 : 1)` | 不由 chunk32 子套件新增；仍归父套件 |
| `ChunkKdaFwd` chunk32 分支 | `32` | 公开 V2 host 在 `chunkSize == 32` 时选择 `RunChunkKdaFwd32Fused`；`op_host/chunk_kda_fwd_tiling.cpp` 执行 `SetTilingKey(32)` | accuracy 0–39、performance 0–1、mss 0–4 |

公开运行选择证据位于
`fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp` 的
`aclnnChunkKdaFwdV2GetWorkspaceSize`：chunk32 分支进入同一 `ChunkKdaFwd` 的 key 32，不经过
key 1/2 的旧 TilingData 或独立 fused 算子。

## ATK 元数据与冻结用例

`chunk_kda_fwd_chunk32.yaml` 沿用父套件的 legacy marker 输入形式；
`gen_chunk_kda_fwd_chunk32.py` 只读加载并校验下列正式 JSON，accuracy 生成入口逐项复用
冻结内容，performance 和 mss 仅校验、不重新生成。三份 JSON 不由生成器重写。

| 文件 | 数量 | 覆盖 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_chunk32.json` | 41 | id 0–33：layout、长序列、尾块、initial/final state；id 34–35：FP32 g/beta + GVA + final_state；id 36–39：NTD/TND varlen、空序列、canonical indices、BF16/FP32 g/beta；id 40：T=128、H=HV=96 单 group 同步回归 |
| `atk_chunk_kda_fwd_chunk32_perf.json` | 2 | B=1、H=HV=96、K=V=128、T=8192，BF16/FP32 g/beta |
| `atk_chunk_kda_fwd_chunk32_mss.json` | 5 | `ChunkKdaFwd` key 32 的 dense、state、尾块和两种 varlen layout 代表用例 |

静态校验命令：

```bash
python3 tests/atk/chunk_kda_fwd/chunk32/gen_chunk_kda_fwd_chunk32.py --summary
# accuracy=41 performance=2 mss=5
```

### legacy marker 后续迁移

当前 48 条冻结 case 均保持 `low_precision_marker`、`fp32_marker`、JSON 字符串
`case_spec` 及其镜像属性，不与真实签名输入混用。后续应另开迁移，将 YAML、三份 JSON
和生成器一起切换为 executor 已支持的 `q/k/v/g/beta`、可选 state/变长张量及真实属性；
迁移时逐 case 核对 id、seed、shape、dtype、可选输入和标杆结果，再重跑完整验收。完成该
整体迁移前，任何 case 都不得单独改用真实签名输入。

精度使用官方 `cv_fused_double_benchmark`：`max<=5`、`avg<=1.5`、`rmse<=1.5`。
报告必须包含真实 NPU 输出及误差记录；`cpu_cpu_reference: []` 表示没有形成比较，
不能计为通过或直接当作数值失败。

## 执行

```bash
# 可选：先指定 CANN 环境；当前 shell 还需准备好 ATK 和 fla_npu OPP。
export CANN_ENV=/path/to/cann/set_env.sh

bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh accuracy 0
bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh performance 0
bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh mssanitizer 0

# 公开 V2 的 H=HV=0 dense/NTD varlen 负向回归。
python3 tests/atk/chunk_kda_fwd/chunk32/negative_chunk32_zero_heads.py
```

设备号由执行环境选择。输出默认写入本目录的 `atk_output/`，可用 `ATK_OUTPUT_ROOT` 改到其他位置；
生成结果不得提交。性能结论以 device `Task Duration` 为准，不与 wall time 混用。
