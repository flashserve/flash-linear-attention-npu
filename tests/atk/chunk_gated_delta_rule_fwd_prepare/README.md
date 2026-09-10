# ChunkGatedDeltaRuleFwdPrepare ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_prepare` 的 ATK 单算子工程：`executor_chunk_gated_delta_rule_fwd_prepare.py`、`gen_chunk_gated_delta_rule_fwd_prepare.py`、`chunk_gated_delta_rule_fwd_prepare.yaml`，以及精度 / 性能 / MSS 三份 JSON。

精度标准为 `mixed_tolerance_bm`（NPU DUT vs CPU 高精度 golden）。CPU 标杆为本目录 `scripts/cpu_golden.py` 的 `cpu_gdn_fwd_l2norm_to_recompute`。

## 输入约束

- 布局 BNSD：`q/k=[B,HK,T,K]`，`v=[B,HV,T,V]`，`g/beta=[B,HV,T]`。
- `K=128`，`V∈{128,256}`，`chunk_size=64`。
- `HV % HK == 0` 且 `HV/HK ∈ {1,2,3,4}`。任务按 HV 计数，K 头按 `hk = hv / (HV/HK)` 复用。
- `q/k/v` 当前仅 `BFLOAT16`；`g/beta` 为 `FLOAT`（golden / DUT 与单元测试一致）。
- `use_exp2`、`use_qk_l2norm_in_kernel`、`use_beta_sigmoid_in_kernel`、`allow_neg_eigval` 支持 True/False，`use_gate_in_kernel=False` 固定；`allow_neg_eigval=True` 要求 sigmoid。
- `use_qk_l2norm_in_kernel=False` 时 kernel 不做 L2norm、不写 hat/rstd；executor 在调用前对 **q 和 k** 做 L2norm。比较时 `_finite_tuple` 丢掉 `None` 的 rstd。
- 变长要求 `B=1` 且 `cu_seqlens` 与 `chunk_indices` 成对（Python 未传 `chunk_indices` 时自动生成）。JSON 用 `seqlens` 列表表示，executor 转成 `cu_seqlens`。
- 尾块 `T % 64 != 0`：只在该 chunk 填 0，按有效行写出。
- 精度 JSON 含 packed varlen；性能 JSON 只取定长 `T>=256` 且 `l2_sig1_neg1` 的前 6 条，不把变长和其它 flag 算进基线。

非法组合由 host 直接拒绝，不进精度 JSON：`use_gate=True`、`use_exp2=False`、`allow_neg_eigval=True` 且 sigmoid=False。

## 中型精度矩阵

中型按 tiling 的 `totalChunks`，不是按 T 长短：

- 定长：`B * HV * ceil(T/64)`（对齐时即 `B*HV*(T/64)`）
- varlen：`HV * sum(ceil(s/64) for s in seqlens)`

全部 50 个 shape 的 chunk 数落在 **(256, 384]**（生成器校验 `>256`）。G≠3 时 pack=4：256 tiles = 64 packs = 32 AIC × 2 pack，**大于 256 保证每核至少 2 个 pack**。上沿 384 tiles = 96 packs ≈ 每核 3 pack。

`atk_chunk_gated_delta_rule_fwd_prepare.json` 为 **50 个中型 shape × 6 组合法 flag = 300**（bf16）。case id = `shape_idx * 6 + flag_idx`，`seed = 20260817 + case_id`。每组 shape 按序覆盖：

| flag_idx | tag | l2 | sigmoid | neg | 含义 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `l2_sig1_neg1` | True | True | True | 核内 L2norm，`beta_eff = 2 * sigmoid(beta)` |
| 1 | `l2_sig1_neg0` | True | True | False | 核内 L2norm，`beta_eff = sigmoid(beta)` |
| 2 | `l2_sig0_neg0` | True | False | False | 核内 L2norm，不做 sigmoid；host dummy `beta_eff`，Python 回填 `beta.float()` |
| 3 | `nol2_sig1_neg1` | False | True | True | 调用前归一化 qk，`2 * sigmoid` |
| 4 | `nol2_sig1_neg0` | False | True | False | 调用前归一化 qk，`sigmoid` |
| 5 | `nol2_sig0_neg0` | False | False | False | 调用前归一化 qk，不做 sigmoid |

下表 `id` 是该 shape 的 `l2_sig1_neg1` case；同 shape 另外 5 组 flag 为 `id+1 … id+5`。`C` 为 chunk 数。

| 类别 | shape | id | B | HK | HV | T | V | C | 覆盖点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G×V 对齐 | `r1_T4160_V128` | 0 | 1 | 4 | 4 | 4160 | 128 | 260 | G=1 V128 |
| G×V 对齐 | `r2_T2112_V128` | 6 | 1 | 4 | 8 | 2112 | 128 | 264 | G=2 |
| G×V 对齐 | `r3_T1408_V128` | 12 | 1 | 4 | 12 | 1408 | 128 | 264 | G=3 pack=3 |
| G×V 对齐 | `r4_T1088_V128` | 18 | 1 | 4 | 16 | 1088 | 128 | 272 | G=4 |
| G×V 对齐 | `r1_T4160_V256` | 24 | 1 | 4 | 4 | 4160 | 256 | 260 | V=256 |
| G×V 对齐 | `r2_T2112_V256` | 30 | 1 | 4 | 8 | 2112 | 256 | 264 | G=2 V256 |
| G×V 对齐 | `r3_T1408_V256` | 36 | 1 | 4 | 12 | 1408 | 256 | 264 | |
| G×V 对齐 | `r4_T1088_V256` | 42 | 1 | 4 | 16 | 1088 | 256 | 272 | G=4 V256 |
| G×V 尾块 | `r1_T4192_V128` | 48 | 1 | 4 | 4 | 4192 | 128 | 264 | `T%64=32` |
| G×V 尾块 | `r2_T2144_V128` | 54 | 1 | 4 | 8 | 2144 | 128 | 272 | |
| G×V 尾块 | `r3_T1440_V128` | 60 | 1 | 4 | 12 | 1440 | 128 | 276 | |
| G×V 尾块 | `r4_T1120_V128` | 66 | 1 | 4 | 16 | 1120 | 128 | 288 | |
| G×V 尾块 | `r{1-4}_T*_V256` | 72–95 | 1 | 4 | 4–16 | 尾块 | 256 | 264–288 | V=256 尾块 |
| 其它 T/G | `g2_T1088_V128` | 96 | 1 | 8 | 16 | 1088 | 128 | 272 | |
| 其它 T/G | `g2_T1088_V256` | 102 | 1 | 8 | 16 | 1088 | 256 | 272 | |
| 其它 T/G | `g1_T2112_V128` | 108 | 1 | 8 | 8 | 2112 | 128 | 264 | |
| 其它 T/G | `g2_T3072_V128` | 114 | 1 | 4 | 8 | 3072 | 128 | 384 | 上沿 |
| 其它 T/G | `g4_T1536_V256` | 120 | 1 | 4 | 16 | 1536 | 256 | 384 | |
| 其它 T/G | `g3_T704_V128` | 126 | 1 | 8 | 24 | 704 | 128 | 264 | G=3 宽 HV |
| B>1 | `B2_g1_T1088` | 132 | 2 | 8 | 8 | 1088 | 128 | 272 | |
| B>1 | `B2_g2_T1088_V256` | 138 | 2 | 4 | 8 | 1088 | 256 | 272 | |
| B>1 | `B4_g1_T1088` | 144 | 4 | 4 | 4 | 1088 | 128 | 272 | |
| B>1 | `B2_g2_T992` | 150 | 2 | 6 | 12 | 992 | 128 | 384 | B>1 尾块 |
| 不满 pack / 奇数 HK | `partial_HV5_T3328` | 156 | 1 | 5 | 5 | 3328 | 128 | 260 | HV%4=1 |
| 不满 pack / 奇数 HK | `partial_HV6_T2816` | 162 | 1 | 6 | 6 | 2816 | 128 | 264 | HV%4=2 |
| 不满 pack / 奇数 HK | `partial_HV7_T2432` | 168 | 1 | 7 | 7 | 2432 | 128 | 266 | HV%4=3 |
| 不满 pack / 奇数 HK | `partial_HV9_T1920` | 174 | 1 | 9 | 9 | 1920 | 128 | 270 | |
| 不满 pack / 奇数 HK | `odd_HK5_G2_T1728_V256` | 180 | 1 | 5 | 10 | 1728 | 256 | 270 | |
| 不满 pack / 奇数 HK | `odd_HK7_G2_T1216_V256` | 186 | 1 | 7 | 14 | 1216 | 256 | 266 | |
| 不满 pack / 奇数 HK | `odd_HK11_T1536` | 192 | 1 | 11 | 11 | 1536 | 128 | 264 | |
| 不满 pack / 奇数 HK | `odd_HK5_G3_T1152` | 198 | 1 | 5 | 15 | 1152 | 128 | 270 | |
| 宽头 | `HK12_G1_T1408` | 204 | 1 | 12 | 12 | 1408 | 128 | 264 | |
| 宽头 | `HK12_G2_T704_V256` | 210 | 1 | 12 | 24 | 704 | 256 | 264 | |
| 宽头 | `HK16_G1_T1088` | 216 | 1 | 16 | 16 | 1088 | 128 | 272 | HK 上沿 |
| 宽头 | `HK16_G2_T576` | 222 | 1 | 16 | 32 | 576 | 128 | 288 | HV 上沿 |
| varlen | `varlen_g1_align` | 228 | 1 | 4 | 4 | 4352 | 128 | 272 | `[1088]*4` |
| varlen | `varlen_g1_tail` | 234 | 1 | 4 | 4 | 4800 | 128 | 300 | `[1600,1536,1664]` |
| varlen | `varlen_g2_v256` | 240 | 1 | 4 | 8 | 2176 | 256 | 272 | `[1088,1088]` |
| varlen | `varlen_g3_mix` | 246 | 1 | 4 | 12 | 1408 | 128 | 264 | `[448,512,448]` G=3 |
| varlen | `varlen_g4_v256` | 252 | 1 | 2 | 8 | 2176 | 256 | 272 | `[1088,1088]` G=4 |
| varlen | `varlen_near_chunk` | 258 | 1 | 8 | 8 | 2112 | 128 | 272 | `[1025,1087]` |
| varlen | `varlen_g2_tail` | 264 | 1 | 8 | 16 | 1088 | 128 | 288 | `[544,544]` |
| varlen | `varlen_four_seq` | 270 | 1 | 4 | 4 | 4160 | 128 | 260 | `[1024,1088,960,1088]` |
| 补齐 | `g3_T1504_tail` | 276 | 1 | 4 | 12 | 1504 | 128 | 288 | G=3 尾块 |
| 补齐 | `B2_HK8_G1_T1536` | 282 | 2 | 8 | 8 | 1536 | 128 | 384 | 上沿 |
| 补齐 | `T2144_g1_tail` | 288 | 1 | 8 | 8 | 2144 | 128 | 272 | |
| 补齐 | `g4_T1152_V128` | 294 | 1 | 4 | 16 | 1152 | 128 | 288 | G=4 V128 |

### 逻辑分支 → 精度 case id

| 逻辑分支 | 代表 case id | 说明 |
| --- | --- | --- |
| 核内 L2norm + 2×sigmoid | 0 | 默认路径，260 tiles / 65 packs |
| 核内 L2norm + sigmoid | 1 | |
| 核内 L2norm、无 sigmoid | 2 | 无 `beta_out` |
| 调用前 L2norm + 2×sigmoid | 3 | 不写 hat/rstd |
| 调用前 L2norm + sigmoid | 4 | |
| 调用前 L2norm、无 sigmoid | 5 | |
| G=1/2/3/4 整 chunk | 0 / 6 / 12 / 18 | |
| V=256 | 24, 30 | |
| 定长尾块 | 48 | T=4192 |
| 不满 pack | 156, 162, 168 | HV%4∈{1,2,3} |
| B>1 | 132, 144 | B=2 / B=4 |
| packed varlen 对齐 / 尾块 / 近 chunk | 228 / 234 / 258 | |
| varlen × G=2/3/4 | 240 / 246 / 252 | |
| HK/HV 中型上沿 | 216, 222 | HK=16，HV=32 |
| chunk 上沿 384 | 114, 120, 282 | 96 packs |

## 性能用例

`atk_chunk_gated_delta_rule_fwd_prepare_perf.json` 为定长 `T>=256`、无 `seqlens`、`l2_sig1_neg1` 的前 6 条：

| perf 顺序 | 精度 case id | shape | B | HK | HV | T | V | C |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | `r1_T4160_V128` | 1 | 4 | 4 | 4160 | 128 | 260 |
| 1 | 6 | `r2_T2112_V128` | 1 | 4 | 8 | 2112 | 128 | 264 |
| 2 | 12 | `r3_T1408_V128` | 1 | 4 | 12 | 1408 | 128 | 264 |
| 3 | 18 | `r4_T1088_V128` | 1 | 4 | 16 | 1088 | 128 | 272 |
| 4 | 24 | `r1_T4160_V256` | 1 | 4 | 4 | 4160 | 256 | 260 |
| 5 | 30 | `r2_T2112_V256` | 1 | 4 | 8 | 2112 | 256 | 264 |

## TilingKey

host tiling 固定 `SetTilingKey(0)`。MSS 用同一 key 覆盖 V128/256、尾块、不满 pack、B>1、varlen、G=2/3/4 和全部 6 组 flag。

| TilingKey | 选择条件 | 普通用例 | 边界用例 | `_mss.json`（精度 case id） | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ascend950 MIX 1:2，K=128，V=128/256，BT=64 | 0（`r1_T4160_V128` `l2_sig1_neg1`） | 48 尾块、162 不满 pack、234 varlen、3 `nol2` | 0, 48, 162, 30, 144, 234, 240, 246, 258, 1, 2, 3, 46, 159, 245 | A5 | host tiling 固定 `SetTilingKey(0)` |

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950`。内核当前只注册 `ascend950`。

## 默认用例

默认用例由生成器覆盖受支持的 flag 组合，具体数量以冻结 JSON 为准。

## 重建 JSON

仓内三份冻结 JSON 由生成器写出（`gen_cases` 只生成精度候选用例，不写 `_perf` / `_mss`）：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd_prepare/gen_chunk_gated_delta_rule_fwd_prepare.py
```

修改 shape / flag 矩阵后需要重建并复核三份 JSON。`atk_chunk_gated_delta_rule_fwd_prepare_g2.json` 是 G=2 过滤副本，非正式验收文件。

## 执行方式

本算子有 9 路输出（`l2norm=False` 时 rstd 为 `None`，比较时丢掉）。ATK 默认 GM 初始化会把 HBM 顶满，后续 case 会卡在 `rtStreamSynchronize`。精度请关 GM init；一次跑满 300 若占卡，按 50 条分批（`CASE_END` 不含右端）。中型 case 约 260–384 tiles，CPU golden 比短 T 更重。

```bash
ATK_GM_INIT_MODE=off \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=accuracy -soc=ascend950

ATK_GM_INIT_MODE=off CASE_START=0 CASE_END=50 \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=accuracy -soc=ascend950

bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -scope=gen_cases
```
