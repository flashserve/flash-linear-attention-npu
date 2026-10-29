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

混合容差的 CPU 标杆是 `cpu_golden_fwd_emu.py` 里的
`cpu_chunk_gated_delta_rule_fwd_emu`。它按 GPU Triton 前向逐步复现 BF16 舍入和
`exp2`，chunk 为 32。L2 和 sigmoid 打开，`allow_neg_eigval` 关闭。
本目录的用例把 BNSD 和打包 NTD 都存成 `[B,H,T,D]`，所以标杆按 `bnsd` 读入，
比较前再把 `o` 转到 BSND。

数值范围沿用算子精度测试：`q/k/v` 为 `randn * 0.05` 后收 BF16，`g` 为
`randn * 0.02` 的 FP32，`beta` 为 `randn` 后收 BF16。同一条用例的 CPU 和 NPU
用同一个 seed。

## Shape

精度 604 条。四种布局、`allow_neg_eigval`、`state_v_first`、初态 `none/bf16/fp32/zero`、
MHA（H=4×4）和 GVA（H=2×8）做完全积，再乘上四种形状：`T=1024` 对齐、`T=1000` 尾块、
`B=2,T=512`、变长 `512+1024`。另有 `T=2048` 的 MHA 全组合，以及 BNSD/BSND 上 `T=4096` 的 GVA。
最长不超过 16K，每种布局一条。`K=V=128`。`use_exp2=False`、核内 L2 和 beta sigmoid 是固定要求，不参与组合。

## 用例

| 文件 | 用途 |
| --- | --- |
| `atk_chunk_gated_delta_rule_fwd_chunk32.json` | 精度，604 条 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_perf.json` | 性能 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json` | 确定性与内存检测 |

完全积是 512 条。`T=2048` 的 MHA 再加 64 条，`T=4096` 的 GVA 再加 24 条，16K 上限每种布局 1 条，一共 604 条。

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

精度标准走源码安装的 ATK 26.9.8（`https://gitcode.com/Ascend/ATK.git`，
commit `99c8a84`）。这一版自带 `mixed_tolerance_bm`，不要再把它别名到
`single_bm`。重生成三份 JSON：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/chunk32/gen_chunk_gated_delta_rule_fwd_chunk32.py
```
