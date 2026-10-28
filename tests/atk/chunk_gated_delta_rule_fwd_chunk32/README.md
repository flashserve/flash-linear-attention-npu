# chunk_gated_delta_rule_fwd 迁移后 chunk32 ATK 工程

本目录独立验证迁移到主算子后的 chunk32 路径。DUT 是
`fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd(..., chunk_size=32)`，
不调用旧接口 `npu_chunk_gated_delta_rule_fwd_32`。精度标准使用 ATK 原生
`mixed_tolerance_bm`，与主算子的 64/128 测试集及旧 chunk32 测试集分开。
`o` 和 `final_state` 都按模型 BF16 精度判定：即使 final state 的存储为 FP32，
块间送入 Cube 的状态图像仍按设计舍入为 BF16。

## 输入约束

- `q/k/v` 按用例 layout 生成。现有用例是 BNSD/NTD，即 `[B,H,T,D]`。
  公开 `g/beta` 是 `[B,T,Hv]`，执行器在调用前从 head-major 转过来。
  公开 `o` 是 BSND `[B,T,Hv,V]`，标杆的 head-major `o` 在比较前转到同一布局。
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

标杆按三个前向算子自己的 CPU 标杆串起来，实现在 `scripts/reference.py`：

1. `chunk_gated_delta_rule_fwd_prepare` 的 `cpu_gdn_fwd_l2norm_to_recompute`
2. `chunk_fwd_h` 的 `executor_chunk_fwd_h._reference`
3. `chunk_fwd_o` 的 `_chunk_fwd_o_ref` 公式；变长下标按
   `test_npu_fwd_o_generalization._reference` 的 sequence-major chunk 槽

公共开关是这三个标杆的自然指数路径：`use_exp2=False`，L2 和 sigmoid 打开，
`allow_neg_eigval` 关闭。标杆 chunk 是 64，因为 `chunk_fwd_h` 的标杆按 64 写。
`k/w/u` 在进入 `chunk_fwd_h` 之前收到 BF16。32 算子里 D 和 u−P 的额外 BF16 往返
不写进标杆。

数值范围沿用算子精度测试：`q/k/v` 为 `randn * 0.05` 后收 BF16，`g` 为
`randn * 0.02` 的 FP32，`beta` 为 `randn` 后收 BF16。同一条用例的 CPU 和 NPU
用同一个 seed。

## Shape 来源

精度 shape 从 `chunk_gated_delta_rule_fwd` 的合同里取，只保留本算子能跑的组合：
`V=128`、`T<=384`、`B*Hv*T<=2048`。每个覆盖桶最多两条，每条三个 seed。
合同里 B=1 的对齐长度 64 和 128 没有和 `V=128` 配上，这两条 `TOKEN_EDGES`
按 `HEAD_PAIRS` 里的 `(1,1)`、`(2,4)`、`(1,2)` 补进矩阵。

`T=16384` 不在精度、性能、内存三份 JSON 里。那条性能仍走原来的 profiler。

## 用例

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| `atk_chunk_gated_delta_rule_fwd_chunk32.json` | 102 | 精度，34 个 shape × 3 个 seed |
| `atk_chunk_gated_delta_rule_fwd_chunk32_perf.json` | 2 | 性能 |
| `atk_chunk_gated_delta_rule_fwd_chunk32_mss.json` | 4 | 确定性与内存检测 |

精度 case 按 shape 连续占三个 id（`s0/s1/s2`）。

| case | 布局 | shape | 初态 | 来源 |
| --- | --- | --- | --- | --- |
| 0–2 | BNSD | B=2, H=1×2, T=256 | 无 | dense |
| 3–5 | BNSD | B=3, H=1×2, T=64 | 无 | dense |
| 6–8 | BNSD GVA | B=1, H=2×8, T=129 | 无 | dense |
| 9–11 | BNSD GVA | B=1, H=1×4, T=191 | 无 | dense |
| 12–14 | BNSD | B=1, H=4×4, T=191 | 无 | dense |
| 15–17 | BNSD | B=1, H=1×1, T=274 | 无 | dense |
| 18–20 | BNSD | B=1, H=1×1, T=1 | BF16 | state_initial_final |
| 21–23 | BNSD GVA | B=1, H=2×8, T=129 | BF16 | state_initial_final |
| 24–26 | NTD | B=1, H=1×1, T=1 | BF16 | state_initial_only |
| 27–29 | NTD GVA | B=1, H=2×8, T=129 | BF16 | state_initial_only |
| 30–32 | BNSD | B=1, H=1×1, T=1 | FP32 | state_initial_only |
| 33–35 | BNSD GVA | B=1, H=2×8, T=129 | FP32 | state_initial_only |
| 36–38 | NTD | B=1, H=1×1, T=1 | FP32 | state_initial_final |
| 39–41 | NTD GVA | B=1, H=2×8, T=129 | FP32 | state_initial_final |
| 42–44 | NTD GVA | T=64，一条序列 | 无 | varlen |
| 45–47 | NTD GVA | T=129，一条序列 | 无 | varlen |
| 48–50 | NTD GVA | T=244，序列 53/64/127 | 无 | varlen |
| 51–53 | NTD | T=129，H=4×4 | 无 | varlen |
| 54–56 | NTD | T=281，H=2×2 | 无 | varlen |
| 57–59 | BNSD | B=1, H=1×1, T=1 | 无 | dense |
| 60–62 | BNSD GVA | B=2, H=2×4, T=31 | 无 | dense |
| 63–65 | NTD | T=1 | 无 | varlen |
| 66–68 | NTD | T=14，H=2×2 | 无 | varlen |
| 69–71 | BNSD GVA | B=1, H=2×8, T=129 | FP32 零 | state_zero_final |
| 72–74 | BNSD GVA | B=2, H=1×2, T=256 | FP32 零 | state_zero_final |
| 75–77 | NTD GVA | T=129 | FP32 零 | state_zero_final |
| 78–80 | NTD GVA | T=244，序列 53/64/127 | FP32 零 | state_zero_final |
| 81–83 | BNSD | B=4, H=1×1, T=64 | 无 | dense |
| 84–86 | BNSD | B=1, H=1×1, T=64 | 无 | TOKEN_EDGES |
| 87–89 | BNSD | B=1, H=1×1, T=128 | 无 | TOKEN_EDGES |
| 90–92 | BNSD GVA | B=1, H=2×4, T=64 | 无 | TOKEN_EDGES |
| 93–95 | BNSD GVA | B=1, H=1×2, T=128 | FP32 | TOKEN_EDGES |
| 96–98 | BNSD | B=1, H=1×1, T=64 | BF16 | TOKEN_EDGES |
| 99–101 | NTD | T=128，序列 64/64，H=2×2 | 无 | TOKEN_EDGES |

### 性能

| case | 场景 |
| --- | --- |
| 0 | BNSD，B=1，H=4×4，T=512，无初态 |
| 1 | NTD GVA，H=2×4，T=256，序列 128/128，FP32 初态 |

### TilingKey

迁移后的专用路径使用 tiling key 32。精度 0–101、性能 0–1、
确定性/内存 0–3 都应走这一条。

| TilingKey | 确定性 / 内存 case | 覆盖 |
| ---: | --- | --- |
| 32 | 0 | BNSD，无初态，T=64 |
| 32 | 1 | BNSD GVA，BF16 初态，T=40 |
| 32 | 2 | BNSD，B=2，FP32 初态，T=32 |
| 32 | 3 | NTD GVA，序列 64/32 |

## 运行

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_chunk32 -soc=ascend950
```

也可使用本目录的独立包装脚本，第一个参数是物理 NPU 号，第二个参数是 scope：

```bash
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES
bash tests/atk/chunk_gated_delta_rule_fwd_chunk32/run_chunk32_atk.sh 6 accuracy
```

精度标准走源码安装的 ATK 26.9.8（`https://gitcode.com/Ascend/ATK.git`，
commit `99c8a84`）。这一版自带 `mixed_tolerance_bm`，不要再把它别名到
`single_bm`。重生成三份 JSON：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd_chunk32/gen_chunk_gated_delta_rule_fwd_chunk32.py
```
