# ChunkGatedDeltaRuleFwdPrepare ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_prepare` 的 ATK 单算子工程：`executor_chunk_gated_delta_rule_fwd_prepare.py`、`gen_chunk_gated_delta_rule_fwd_prepare.py`、`chunk_gated_delta_rule_fwd_prepare.yaml`，以及精度 / 性能 / MSS / 精简四份 JSON。

精度标准为 `mixed_tolerance_bm`（NPU DUT vs CPU 高精度 golden）。CPU 标杆为本目录 `scripts/cpu_golden.py` 的 `cpu_gdn_fwd_l2norm_to_recompute`。

## 输入约束

- 布局 BNSD：`q/k=[B,HK,T,K]`，`v=[B,HV,T,V]`，`g/beta=[B,HV,T]`。
- `K=128`，`V∈{128,256}`，`chunk_size=64`。
- `HV % HK == 0` 且 `HV/HK ∈ {1,2,3,4}`。任务按 HV 计数，K 头按 `hk = hv / (HV/HK)` 复用。
- `q/k/v` 当前仅 `BFLOAT16`；`g/beta` 为 `FLOAT`（golden / DUT 与单元测试一致）。
- `use_exp2`、`use_gate_in_kernel` 支持 True/False。`use_gate=True` 时 executor 生成 `a_log` / `dt_bias`（`[HV]` fp32），`g` 为 raw dt logits。
- `use_qk_l2norm_in_kernel`、`use_beta_sigmoid_in_kernel`、`allow_neg_eigval` 支持 True/False；`allow_neg_eigval=True` 要求 sigmoid。
- 精度 JSON 为 **50 个中型 shape × 24 组合法 flag = 1200**（bf16）。case id = `shape_idx * 24 + flag_idx`。合法 flag = `l2 × gate × {(sig,neg)=(T,T),(T,F),(F,F)} × exp2`。
- `use_qk_l2norm_in_kernel=False` 时 kernel 不做 L2norm、不写 hat/rstd；executor 在调用前对 **q 和 k** 做 L2norm。比较时 `_finite_tuple` 丢掉 `None` 的 rstd。
- 变长要求 `B=1` 且 `cu_seqlens` 与 `chunk_indices` 成对（Python 未传 `chunk_indices` 时自动生成）。JSON 用 `seqlens` 列表表示，executor 转成 `cu_seqlens`。
- 尾块 `T % 64 != 0`：只在该 chunk 填 0，按有效行写出。
- 精度 JSON 含 packed varlen；性能 JSON 只取定长 `T>=256` 且 `l2_sig1_neg1` 的前 6 条，不把变长和其它 flag 算进基线。

非法组合由 host 直接拒绝，不进精度 JSON：`allow_neg_eigval=True` 且 sigmoid=False；`dt_bias` 无 `a_log`。

## 中型精度矩阵

中型按 tiling 的 `totalChunks`，不是按 T 长短：

- 定长：`B * HV * ceil(T/64)`（对齐时即 `B*HV*(T/64)`）
- varlen：`HV * sum(ceil(s/64) for s in seqlens)`

全部 50 个 shape 的 chunk 数落在 **(256, 384]**（生成器校验 `>256`）。G≠3 时 pack=4：256 tiles = 64 packs = 32 AIC × 2 pack，**大于 256 保证每核至少 2 个 pack**。上沿 384 tiles = 96 packs ≈ 每核 3 pack。

`atk_chunk_gated_delta_rule_fwd_prepare.json` 为 **50 个中型 shape × 24 组合法 flag = 1200**（bf16）。case id = `shape_idx * 24 + flag_idx`，`seed = 20260817 + case_id`。每组 shape 按序覆盖：

| flag_idx | tag | l2 | gate | sigmoid | neg | exp2 |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `l2_sig1_neg1` | True | False | True | True | True |
| 1 | `l2_sig1_neg0` | True | False | True | False | True |
| 2 | `l2_sig0_neg0` | True | False | False | False | True |
| 3 | `nol2_sig1_neg1` | False | False | True | True | True |
| 4 | `nol2_sig1_neg0` | False | False | True | False | True |
| 5 | `nol2_sig0_neg0` | False | False | False | False | True |
| 6 | `l2_gate1_sig1_neg1` | True | True | True | True | True |
| 7 | `l2_gate1_sig1_neg0` | True | True | True | False | True |
| 8 | `l2_gate1_sig0_neg0` | True | True | False | False | True |
| 9 | `nol2_gate1_sig1_neg1` | False | True | True | True | True |
| 10 | `nol2_gate1_sig1_neg0` | False | True | True | False | True |
| 11 | `nol2_gate1_sig0_neg0` | False | True | False | False | True |
| 12 | `l2_sig1_neg1_exp0` | True | False | True | True | False |
| 13 | `l2_sig1_neg0_exp0` | True | False | True | False | False |
| 14 | `l2_sig0_neg0_exp0` | True | False | False | False | False |
| 15 | `nol2_sig1_neg1_exp0` | False | False | True | True | False |
| 16 | `nol2_sig1_neg0_exp0` | False | False | True | False | False |
| 17 | `nol2_sig0_neg0_exp0` | False | False | False | False | False |
| 18 | `l2_gate1_sig1_neg1_exp0` | True | True | True | True | False |
| 19 | `l2_gate1_sig1_neg0_exp0` | True | True | True | False | False |
| 20 | `l2_gate1_sig0_neg0_exp0` | True | True | False | False | False |
| 21 | `nol2_gate1_sig1_neg1_exp0` | False | True | True | True | False |
| 22 | `nol2_gate1_sig1_neg0_exp0` | False | True | True | False | False |
| 23 | `nol2_gate1_sig0_neg0_exp0` | False | True | False | False | False |

下表 `id` 是该 shape 的 `l2_sig1_neg1` case；同 shape 另外 23 组 flag 为 `id+1 … id+23`。`C` 为 chunk 数。

| 类别 | shape | id | B | HK | HV | T | V | C | 覆盖点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G×V 对齐 | `r1_T4160_V128` | 0 | 1 | 4 | 4 | 4160 | 128 | 260 | G=1 V128 |
| G×V 对齐 | `r2_T2112_V128` | 24 | 1 | 4 | 8 | 2112 | 128 | 264 | G=2 |
| G×V 对齐 | `r3_T1408_V128` | 48 | 1 | 4 | 12 | 1408 | 128 | 264 | G=3 pack=3 |
| G×V 对齐 | `r4_T1088_V128` | 72 | 1 | 4 | 16 | 1088 | 128 | 272 | G=4 |
| G×V 对齐 | `r1_T4160_V256` | 96 | 1 | 4 | 4 | 4160 | 256 | 260 | V=256 |
| G×V 对齐 | `r2_T2112_V256` | 120 | 1 | 4 | 8 | 2112 | 256 | 264 | G=2 V256 |
| G×V 对齐 | `r3_T1408_V256` | 144 | 1 | 4 | 12 | 1408 | 256 | 264 | |
| G×V 对齐 | `r4_T1088_V256` | 168 | 1 | 4 | 16 | 1088 | 256 | 272 | G=4 V256 |
| G×V 尾块 | `r1_T4192_V128` | 192 | 1 | 4 | 4 | 4192 | 128 | 264 | `T%64=32` |
| G×V 尾块 | `r2_T2144_V128` | 216 | 1 | 4 | 8 | 2144 | 128 | 272 | |
| G×V 尾块 | `r3_T1440_V128` | 240 | 1 | 4 | 12 | 1440 | 128 | 276 | |
| G×V 尾块 | `r4_T1120_V128` | 264 | 1 | 4 | 16 | 1120 | 128 | 288 | |
| G×V 尾块 | `r{1-4}_T*_V256` | 288–383 | 1 | 4 | 4–16 | 尾块 | 256 | 264–288 | V=256 尾块 |
| 其它 T/G | `g2_T1088_V128` | 384 | 1 | 8 | 16 | 1088 | 128 | 272 | |
| 其它 T/G | `g2_T1088_V256` | 408 | 1 | 8 | 16 | 1088 | 256 | 272 | |
| 其它 T/G | `g1_T2112_V128` | 432 | 1 | 8 | 8 | 2112 | 128 | 264 | |
| 其它 T/G | `g2_T3072_V128` | 456 | 1 | 4 | 8 | 3072 | 128 | 384 | 上沿 |
| 其它 T/G | `g4_T1536_V256` | 480 | 1 | 4 | 16 | 1536 | 256 | 384 | |
| 其它 T/G | `g3_T704_V128` | 504 | 1 | 8 | 24 | 704 | 128 | 264 | G=3 宽 HV |
| B>1 | `B2_g1_T1088` | 528 | 2 | 8 | 8 | 1088 | 128 | 272 | |
| B>1 | `B2_g2_T1088_V256` | 552 | 2 | 4 | 8 | 1088 | 256 | 272 | |
| B>1 | `B4_g1_T1088` | 576 | 4 | 4 | 4 | 1088 | 128 | 272 | |
| B>1 | `B2_g2_T992` | 600 | 2 | 6 | 12 | 992 | 128 | 384 | B>1 尾块 |
| 不满 pack / 奇数 HK | `partial_HV5_T3328` | 624 | 1 | 5 | 5 | 3328 | 128 | 260 | HV%4=1 |
| 不满 pack / 奇数 HK | `partial_HV6_T2816` | 648 | 1 | 6 | 6 | 2816 | 128 | 264 | HV%4=2 |
| 不满 pack / 奇数 HK | `partial_HV7_T2432` | 672 | 1 | 7 | 7 | 2432 | 128 | 266 | HV%4=3 |
| 不满 pack / 奇数 HK | `partial_HV9_T1920` | 696 | 1 | 9 | 9 | 1920 | 128 | 270 | |
| 不满 pack / 奇数 HK | `odd_HK5_G2_T1728_V256` | 720 | 1 | 5 | 10 | 1728 | 256 | 270 | |
| 不满 pack / 奇数 HK | `odd_HK7_G2_T1216_V256` | 744 | 1 | 7 | 14 | 1216 | 256 | 266 | |
| 不满 pack / 奇数 HK | `odd_HK11_T1536` | 768 | 1 | 11 | 11 | 1536 | 128 | 264 | |
| 不满 pack / 奇数 HK | `odd_HK5_G3_T1152` | 792 | 1 | 5 | 15 | 1152 | 128 | 270 | |
| 宽头 | `HK12_G1_T1408` | 816 | 1 | 12 | 12 | 1408 | 128 | 264 | |
| 宽头 | `HK12_G2_T704_V256` | 840 | 1 | 12 | 24 | 704 | 256 | 264 | |
| 宽头 | `HK16_G1_T1088` | 864 | 1 | 16 | 16 | 1088 | 128 | 272 | HK 上沿 |
| 宽头 | `HK16_G2_T576` | 888 | 1 | 16 | 32 | 576 | 128 | 288 | HV 上沿 |
| varlen | `varlen_g1_align` | 912 | 1 | 4 | 4 | 4352 | 128 | 272 | `[1088]*4` |
| varlen | `varlen_g1_tail` | 936 | 1 | 4 | 4 | 4800 | 128 | 300 | `[1600,1536,1664]` |
| varlen | `varlen_g2_v256` | 960 | 1 | 4 | 8 | 2176 | 256 | 272 | `[1088,1088]` |
| varlen | `varlen_g3_mix` | 984 | 1 | 4 | 12 | 1408 | 128 | 264 | `[448,512,448]` G=3 |
| varlen | `varlen_g4_v256` | 1008 | 1 | 2 | 8 | 2176 | 256 | 272 | `[1088,1088]` G=4 |
| varlen | `varlen_near_chunk` | 1032 | 1 | 8 | 8 | 2112 | 128 | 272 | `[1025,1087]` |
| varlen | `varlen_g2_tail` | 1056 | 1 | 8 | 16 | 1088 | 128 | 288 | `[544,544]` |
| varlen | `varlen_four_seq` | 1080 | 1 | 4 | 4 | 4160 | 128 | 260 | `[1024,1088,960,1088]` |
| 补齐 | `g3_T1504_tail` | 1104 | 1 | 4 | 12 | 1504 | 128 | 288 | G=3 尾块 |
| 补齐 | `B2_HK8_G1_T1536` | 1128 | 2 | 8 | 8 | 1536 | 128 | 384 | 上沿 |
| 补齐 | `T2144_g1_tail` | 1152 | 1 | 8 | 8 | 2144 | 128 | 272 | |
| 补齐 | `g4_T1152_V128` | 1176 | 1 | 4 | 16 | 1152 | 128 | 288 | G=4 V128 |

### 逻辑分支 → 精度 case id

| 逻辑分支 | 代表 case id | 说明 |
| --- | --- | --- |
| 核内 L2norm + 2×sigmoid | 0 | 默认路径，260 tiles / 65 packs |
| 核内 L2norm + sigmoid | 1 | |
| 核内 L2norm、无 sigmoid | 2 | 无 `beta_out` |
| 调用前 L2norm + 2×sigmoid | 3 | 不写 hat/rstd |
| 调用前 L2norm + sigmoid | 4 | |
| 调用前 L2norm、无 sigmoid | 5 | |
| fused gate + 2×sigmoid | 6 | `a_log` / `dt_bias`，`g` 为 raw dt |
| 调用前 L2norm + fused gate | 9–11 | `nol2`×gate 三组 sigmoid/neg |
| `use_exp2=False` | 12 | 自然 `exp` |
| fused gate + `use_exp2=False` | 18 | |
| `use_exp2=False` 其余 11 组 | 13–17, 19–23 | 含 nol2 / 无 sigmoid |
| G=1/2/3/4 整 chunk | 0 / 24 / 48 / 72 | |
| V=256 | 96, 120 | |
| 定长尾块 | 192 | T=4192 |
| 不满 pack | 624, 648, 672 | HV%4∈{1,2,3} |
| B>1 | 528, 576 | B=2 / B=4 |
| packed varlen 对齐 / 尾块 / 近 chunk | 912 / 936 / 1032 | |
| varlen × G=2/3/4 | 960 / 984 / 1008 | |
| HK/HV 中型上沿 | 864, 888 | HK=16，HV=32 |
| chunk 上沿 384 | 456, 480, 1128 | 96 packs |

## 性能用例

`atk_chunk_gated_delta_rule_fwd_prepare_perf.json` 为定长 `T>=256`、无 `seqlens`、`l2_sig1_neg1` 的前 6 条：

| perf 顺序 | 精度 case id | shape | B | HK | HV | T | V | C |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | `r1_T4160_V128` | 1 | 4 | 4 | 4160 | 128 | 260 |
| 1 | 24 | `r2_T2112_V128` | 1 | 4 | 8 | 2112 | 128 | 264 |
| 2 | 48 | `r3_T1408_V128` | 1 | 4 | 12 | 1408 | 128 | 264 |
| 3 | 72 | `r4_T1088_V128` | 1 | 4 | 16 | 1088 | 128 | 272 |
| 4 | 96 | `r1_T4160_V256` | 1 | 4 | 4 | 4160 | 256 | 260 |
| 5 | 120 | `r2_T2112_V256` | 1 | 4 | 8 | 2112 | 256 | 264 |

## TilingKey

host tiling 固定 `SetTilingKey(0)`。MSS 用同一 key 覆盖 V128/256、尾块、不满 pack、B>1、varlen、G=2/3/4，以及全部 24 组合法 flag。

| TilingKey | 选择条件 | 普通用例 | 边界用例 | `_mss.json`（精度 case id） | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ascend950 MIX 1:2，K=128，V=128/256，BT=64 | 0（`r1_T4160_V128` `l2_sig1_neg1`） | 192 尾块、648 不满 pack、936 varlen、3 `nol2` | 0, 192, 648, 120, 576, 936, 960, 984, 1032, 1, 2, 3, 172, 627, 965, 6, 198, 942, 9, 12, 18, 978, 10, 11, 13–17, 19–23 | A5 | host tiling 固定 `SetTilingKey(0)` |

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950`。内核当前只注册 `ascend950`。

## 重建 JSON

仓内冻结 JSON 由生成器写出（`gen_cases` 只生成精度候选用例，不写 `_perf` / `_mss` / `_slim`）：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd_prepare/gen_chunk_gated_delta_rule_fwd_prepare.py
```

修改 shape / flag 矩阵后需要重建并复核精度 / 性能 / MSS / 精简四份 JSON。

## 精简用例

`atk_chunk_gated_delta_rule_fwd_prepare_slim.json` 从 1200 条里各抽 1 条，共 **73**：

- 24 组合法 flag：全部落在 `r1_T4160_V128`（精度 id `0–23`）
- 其余 49 个 shape：只保留默认 `l2_sig1_neg1`（精度 id `24, 48, …, 1176`）

用于冒烟，不是正式验收入口。`run_test_cpu.sh -scope=accuracy` 默认仍读完整 1200 条。跑精简集：

```bash
ATK_GM_INIT_MODE=off \
ATK_ACCURACY_JSON=./atk_chunk_gated_delta_rule_fwd_prepare_slim.json \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare \
  -npu_device_id=0 -scope=accuracy -soc=ascend950
```

## 执行方式

本算子有 9 路输出（`l2norm=False` 时 rstd 为 `None`，比较时丢掉）。ATK 默认 GM 初始化会把 HBM 顶满，后续 case 会卡在 `rtStreamSynchronize`。精度请关 GM init；一次跑满 1200 若占卡，按 **12** 条分批（`CASE_END` 不含右端；50 条一批容易卡住）。

本容器没有 `/dev/davinciN`，ATK 只能看到 device 0；请用 `-npu_device_id=0`，`ATK_OUTPUT_ROOT` 建议用绝对路径。

```bash
ATK_GM_INIT_MODE=off \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950

ATK_GM_INIT_MODE=off CASE_START=0 CASE_END=12 \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950

bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -scope=gen_cases
```
