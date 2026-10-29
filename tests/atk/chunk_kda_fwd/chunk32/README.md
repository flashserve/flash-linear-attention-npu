# chunk_kda_fwd / arch35 chunk32 ATK

本目录验证统一 `chunk_kda_fwd` 算子在 Ascend950 上的 chunk32 单 kernel 融合路径。
所有 NPU 调用都经过公开 `aclnnChunkKdaFwdV2`，CPU 侧使用同一 executor 中的 FP64 golden
和同精度 benchmark。公开调用统一经过 `aclnnChunkKdaFwdV2`，测试不依赖直接 fused 入口。

## 契约与 TilingKey

- TilingKey：`32`；仅用于 Ascend950 / A5。
- q/k/v/o 为 BF16，`K=V=128`；g/beta 必须同为 BF16 或同为 FP32。
- 支持 dense batch、varlen、GVA、非 128 对齐尾块、initial/final state 和 NTD 原生布局。
- `safe_gate=true`、`use_gate_in_kernel=true`，必须提供 `A_log`；`dt_bias=None` 等价于零偏置。
- `use_qk_l2norm_in_kernel=true`、`use_beta_sigmoid_in_kernel=true`、
  `allow_neg_eigval=false`、`use_exp2=true`。
- 不导出 recompute 中间态或反向保存值；`Aqk`/`Akk` 返回 `None`，`chunk_indices`
  仅接受 canonical sequence-major 顺序。

## 用例映射

| 文件 | 数量 | 覆盖 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_chunk32.json` | 40 | id 0–33：layout、长序列、尾块、initial/final state；id 34–35：FP32 g/beta + GVA + final_state；id 36–39：NTD/TND varlen、空序列、canonical indices、BF16/FP32 g/beta |
| `atk_chunk_kda_fwd_chunk32_perf.json` | 2 | B=1、H=HV=96、K=V=128、T=8192，BF16/FP32 g/beta |
| `atk_chunk_kda_fwd_chunk32_mss.json` | 5 | TilingKey 32 的 dense、state、尾块和两种 varlen layout 代表用例 |

精度使用官方 `cv_fused_double_benchmark`：`max<=5`、`avg<=1.5`、`rmse<=1.5`。
报告必须包含真实 NPU 输出及误差记录；`cpu_cpu_reference: []` 表示没有形成比较，
不能计为通过或直接当作数值失败。

## 执行

```bash
# 可选：先指定 CANN 环境；当前 shell 还需准备好 ATK 和 fla_npu OPP。
export CANN_ENV=/path/to/cann/set_env.sh

bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh accuracy 4
bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh performance 4
bash tests/atk/chunk_kda_fwd/chunk32/run_chunk32_atk.sh mssanitizer 4
```

只用卡 4 或 5。输出默认写入本目录的 `atk_output/`，可用 `ATK_OUTPUT_ROOT` 改到其他位置；
生成结果不得提交。性能结论以 device `Task Duration` 为准，不与 wall time 混用。
