# chunk32 ATK

`chunk_size=32` 走 `npu_chunk_gated_delta_rule_fwd` 的融合核，只在 Ascend950 上选 tiling key 32。精度标杆是 `mixed_tolerance_bm` 的 CPU 仿真。这个目录不挂第二路标杆。

## 用例

| 文件 | 用途 |
| --- | --- |
| `atk_chunk_gated_delta_rule_fwd_chunk32.json` | 混合容差精度 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_slim.json` | 精度冒烟 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_perf.json` | 性能 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json` | 内存检测，精度标准仍是 `mixed_tolerance_bm` |
| `atk_chunk_gated_delta_rule_fwd_chunk32_dc.json` | 确定性，一条 H=28、T=8192、导出全开 |

执行器是 `executor_chunk_gated_delta_rule_fwd_chunk32.py`。NPU 侧是 DUT，CPU 侧只认 `mixed_tolerance_bm`。

十元组里，`out_l2=false` 时 q_hat、k_hat、两个 rstd 和 `beta_eff` 为 None。`disable_recompute=false` 时 g_cumsum 和 A 为 None。`out_h=false` 时 h 为 None。`beta_eff` 跟着 `out_l2`，不单独打开。

## 环境变量

`run_npu` 在这次调用期间把 `TBE_PARALLEL_COMPILE_ENABLE` 和 `PARALLEL_COMPILE` 设为 `0`，返回前恢复进入函数时的值。没有别的路径开关。

## 跑法

在本目录下，设备号按现场改。精度：

```bash
atk node --name npu_dut --backend npu --devices 0 \
  task -c atk_chunk_gated_delta_rule_fwd_chunk32.json \
  -p executor_chunk_gated_delta_rule_fwd_chunk32.py \
  --task accuracy -sp -mt 1
```

确定性把 `--task` 换成 `accuracy_dc`，用例换成 `atk_chunk_gated_delta_rule_fwd_chunk32_dc.json`，并加 `--dc_loop_nums 50`。内存检测在同一执行器外面包 `mssanitizer`，用例用 `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json`。
