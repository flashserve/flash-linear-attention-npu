# chunk32 ATK

`chunk_size=32` 走 `npu_chunk_gated_delta_rule_fwd` 的融合核，只在 Ascend950 上选 tiling key 32。精度标杆是 `mixed_tolerance_bm` 的 CPU 仿真。这个目录不挂第二路标杆。

## 用例

`chunk_gated_delta_rule_fwd_chunk32.yaml` 声明 `q/k/v/g/beta` 和每个开关。`gen_chunk_gated_delta_rule_fwd_chunk32.py` 在 `atk case` 里按套件写上 shape 和属性，不把规格打成一条字符串。

| 文件 | 套件 | 条数 | 用途 |
| --- | --- | --- | --- |
| `atk_chunk_gated_delta_rule_fwd_chunk32.json` | `accuracy` | 1121 | 混合容差精度 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_slim.json` | `slim` | 35 | 精度冒烟 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_perf.json` | `perf` | 2 | 性能 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json` | `mss` | 60 | 内存检测，精度标准仍是 `mixed_tolerance_bm` |
| `atk_chunk_gated_delta_rule_fwd_chunk32_dc.json` | `dc` | 1 | 确定性，H=28、T=8192、导出全开。这条不写精度标准 |

在本目录生成精度套：

```bash
atk case -f chunk_gated_delta_rule_fwd_chunk32.yaml \
  -p gen_chunk_gated_delta_rule_fwd_chunk32.py -dt 1121 -en 0
```

其它套件加上 `CHUNK32_ATK_SUITE`，`-dt` 与上表条数一致，例如 `CHUNK32_ATK_SUITE=slim -dt 35`。产物在 `result/chunk_gated_delta_rule_fwd_chunk32/json/`。ATK 写出的顺序不一定是矩阵顺序，入库时按用例名里的序号排好。

`q/k/v` 的采样区间是 `[-0.05, 0.05]`，`g` 是 `[-0.02, 0.02]`，`beta` 是 `[-1, 1]`。正常门控保持在这个小区间里，避免 `exp2(cumsum(g))` 在长序列上溢出；大衰减仍由 `g_mode` 写入。`g_mode=normal` 使用采样到的 `g`；`strong` 把 `g` 写成 -4，`spike` 把前两个 token 写成 -100 和 1，`mix` 把第二个 chunk 写成 -4。初始状态不作为张量输入：`state_dtype` 为 `none` 时没有初始状态，`zero` 为全 0，`bf16` 和 `fp32` 用同一组固定小数。

执行器是 `executor_chunk_gated_delta_rule_fwd_chunk32.py`。NPU 侧是 DUT，CPU 侧只认 `mixed_tolerance_bm`。

十元组里，`disable_recompute=false` 时 g_cumsum、A、q_hat、k_hat、两个 rstd、`beta_eff` 和 h 都是 None。`true` 时导出 g_cumsum、A、q_hat、k_hat、两个 rstd 和 `beta_eff`；h 还要 `out_h=true`。`out_l2` 不再单独决定公开 hats。

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
