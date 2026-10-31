# chunk_gated_delta_rule_fwd chunk32 ATK 工程

## 输入约束

- `q/k/v` 按用例 layout 生成。BNSD/NTD 是 `[B,H,T,D]`，BSND/TND 是 `[B,T,H,D]`。
  公开 `g/beta` 始终是 `[B,T,Hv]`。公开 `o` 是 BSND `[B,T,Hv,V]`。
- `K` 和 `V` 固定为 128。`Hv` 是 `Hk` 的正整数倍。GVA 是 `Hv > Hk`。
- BNSD 不传 `cu_seqlens`，物理形状 `[B, H, T, D]`，`final_state` 为 `[B, Hv, 128, 128]`。
- NTD 是打包的 head-major 输入：`B=1`，张量 `[1, H, T, D]`，再加严格递增的
  `cu_seqlens`。`final_state` 为 `[N, Hv, 128, 128]`，`N = len(cu_seqlens) - 1`。
  token-major 的 TND（`[T, H, D]`）不在本算子接口里，所以没有对应用例。
- `initial_state` 为空、BF16 或 FP32。为空或显式 FP32 零时，`final_state` 是 FP32；
  输入是 BF16 时，`final_state` 也是 BF16。算子总会写出 `final_state`。
- `scale` 固定为 `1/sqrt(128)`。执行器同时生成 block 32 的 canonical
  `cu_seqlens/chunk_indices`。

## 标杆

同一份执行器按设备分三路，输入都从用例 seed 重建：

| 角色 | 设备 | 实现 |
| --- | --- | --- |
| DUT | NPU | `npu_chunk_gated_delta_rule_fwd(..., chunk_size=32)` |
| 混合容差标杆 / 双标杆 golden | CPU | `cpu_golden_fwd_emu.py` 的 `cpu_chunk_gated_delta_rule_fwd_emu` |
| 双标杆同精度标杆 | GPU | `gpu_triton_fwd.py` 的 `gpu_chunk_gated_delta_rule_fwd`，`chunk_size=32` |

CPU 仿真按 GPU Triton 前向逐步复现 BF16 舍入，chunk 为 32。L2 和 sigmoid 打开。
`allow_neg_eigval` 和 `state_v_first` 跟用例走。公开 `o` 比较前统一成 BSND。
`final_state` 在 BF16 初态时是 BF16，其余是 FP32。

双标杆标准与父目录 `chunk_gated_delta_rule_fwd` 相同：`cv_fused_double_benchmark`，
最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5。
NPU 节点名是 `npu_dut`，GPU 节点名是 `gold`，CPU golden 由 `-bd cpu` 自动拉起。

数值范围沿用算子精度测试：`q/k/v` 为 `randn * 0.05` 后收 BF16，`g` 为
`randn * 0.02` 的 FP32，`beta` 为 `randn` 后收 BF16。同一条用例的 NPU、CPU 和 GPU
用同一个 seed。

## Shape

`K=V=128`。`use_exp2=False`、核内 L2 和 beta sigmoid 是固定要求，不参与组合。T 不超过 8192。

BNSD 和 NTD 把对齐、尾块、MHA（H=4×4）、GVA（H=2×8）、初态 `none/bf16/fp32/zero`、
`allow_neg_eigval`、`state_v_first` 做完全积。BNSD 另有 `B=2,T=64`。NTD 的变长是对齐的
`64+64` 和不对齐的 `40+60`，再加一条不定长的 `T=100`。BSND、TND 用同样的开关，初态只取
`none/bf16`，各覆盖对齐和尾块。上限是 BNSD `T=8192` 和 NTD `3000+5000`。

可选导出不进上面的完全积，单独配 7 种开关组合（L2、g_cumsum/A、h，以及两两和三者全开）。
L2 打开时同时比较 `beta_eff`、`q_hat`、`k_hat`、`q_rstd`、`k_rstd`。每种组合落在
BNSD 对齐、BNSD 尾块 GVA、NTD 变长、BSND、TND 上。`g_cumsum` 是 chunk 内自然底数的和，
等于 Triton 里那份 cumsum 除以 `RCP_LN2`。`h` 和 `final_state` 在 `state_v_first`
打开时都是 `[V, K]`。

## 用例

| 文件 | 用途 |
| --- | --- |
| `atk_chunk_gated_delta_rule_fwd_chunk32.json` | 混合容差精度，300 条 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_double.json` | 双标杆精度，同一批 300 条 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_perf.json` | 性能 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json` | 确定性与内存检测 |

一共 300 条：BNSD 96、NTD 96、BSND 32、TND 32、8K 上限 2、可选导出 42。重生成四份 JSON：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/chunk32/gen_chunk_gated_delta_rule_fwd_chunk32.py
```

### 性能

| case | 场景 |
| --- | --- |
| 0 | BNSD，H=8×8，T=8192 |
| 1 | BSND GVA，H=4×8，T=4096，`allow_neg_eigval` 和 `state_v_first` 都打开 |

### TilingKey

这些用例都走 tiling key 32。

| TilingKey | 确定性 / 内存 case | 覆盖 |
| ---: | --- | --- |
| 32 | 0 | BNSD，无初态，T=64 |
| 32 | 1 | BSND GVA，BF16 初态，两个新开关都打开，T=96 |
| 32 | 2 | NTD，FP32 初态，序列 64/32 |
| 32 | 3 | TND GVA，FP32 零初态，两个新开关都打开，序列 48/32 |

## 运行

这份用例不挂在 `tests/atk` 的顶层 `-op=` 上，避免和 64/128 的测试集混在一次运行里。
第一个参数是物理 NPU 号，第二个参数是 scope：

```bash
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES
bash tests/atk/chunk_gated_delta_rule_fwd/chunk32/run_chunk32_atk.sh 6 accuracy
```

双标杆在 NPU 侧发起，远端 GPU server 承载 `gold` 节点。GPU 环境需要 CUDA、Triton 和
`fla`；没有安装包时设置 `FLA_ROOT`。`GPU_HOST` 必填。

```bash
GPU_HOST=<gpu-server-ip> GPU_HOST_PORT=9090 \
  bash tests/atk/chunk_gated_delta_rule_fwd/chunk32/run_chunk32_atk.sh 6 double
```

精度标准走源码安装的 ATK 26.9.8（`https://gitcode.com/Ascend/ATK.git`，
commit `99c8a84`）。这一版自带 `mixed_tolerance_bm`，不要再把它别名到
`single_bm`。`cv_fused_double_benchmark` 也由这一版 ATK 提供。
