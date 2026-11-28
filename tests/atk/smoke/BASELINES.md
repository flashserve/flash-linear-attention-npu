# 算子冒烟标杆与用例来源

标杆取自 [work_tool 的固定提交 72d3720](https://gitcode.com/chen-linxin4/work_tool/tree/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK)。以下目录映射便于与上游核对。

| 冒烟算子 | 上游 FLA_ATK 路径 |
|---|---|
| `chunk_fwd_h` | `chunk_gated_delta_rule_bwd/scripts/fwd_h_cpu.py`，数学实现位于 [fwd_h_cpu.py](cases/chunk_fwd_h/fwd_h_cpu.py)，由前向和 GDN 反向标杆共用 |
| `chunk_kda_fwd`、`chunk_kda_fwd_fused` | `chunk_kda_fwd/executor_chunk_kda_fwd_v2.py`、`chunk_kda_fwd/executor_chunk_kda_fwd.py` |
| `chunk_kda_fwd_prepare`、`chunk_kda_fwd_finalize` | `chunk_kda_fwd_prepare_a2/`、`chunk_fwd_finalize/` |
| `chunk_kda_bwd_prepare` | `chunk_kda_bwd_prepare_a3/` |
| `recurrent_gated_delta_rule` | `recurrent_gated_delta_rule/rgdn_A5/` |

其余标杆保留上游算子目录名和文件名。`chunk_gated_delta_rule_fwd_h` 的所有 A5 用例统一使用 `executor_gdn_fwd_h_state.py`。KDA Forward V2 和 GDN 反向整链的导入路径已适配仓内布局。

## 拆分前全量基线的入选依据

11 月 25 日和 26 日的 A2、A5 真卡基线使用 FLA wheel `0.dev20261124+main.fe8a05d`，平台各自按通过记录选取。11 月 26 日复测组保留执行器原始输出类型，其余组保留上次实测的输出类型处理。KDA chunk32 的 state 配置和 A5 的 ChunkFwdO 优化配置各自保留独立用例组。[标杆提交](https://gitcode.com/chen-linxin4/work_tool/tree/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK)

八个算子使用上述提交中的 `atk_<算子>.json` 精度用例。卷积另外保留可选 bias、状态以及反向的 BSH/BSND 和三种激活模式。每个平台的反向卷积保留了 54 条功能回归。以 `cv200` 结尾的两组分别由本仓生成器产生，每个算子 200 条，两个平台保留同一组输入供复测。

| 使用上游精度用例的算子 | 每个已登记平台的上游用例数 |
|---|---:|
| `causal_conv1d` / `causal_conv1d_bwd` | 2 / 200 |
| `chunk_gated_delta_rule_bwd` / `chunk_gdn_bwd_intra` | 128 / 128 |
| `chunk_gated_delta_rule_bwd_dhu` / `chunk_gated_delta_rule_bwd_finalize` | 64 / 64 |
| `merge_fwd_bwd_kernel` / `pre_process_bwd_kernel_merged` | 84 / 200 |

替换前的 [历史用例与配套执行器](https://github.com/zzy1205/flash-linear-attention-npu/tree/64f7548bc24d70f27157ad9a1fb3b7158b9b5447/tests/atk/smoke/cases) 可用于复现旧报告。在独立工作目录检出该提交，按上文的 `run --op ... --platform ...` 命令运行。GDN 反向四个算子的旧 generated、prior-200、gate-base 组，以及 merge 的 boundaries/prior 组、pre_process 的 prior 组，已切换为上游配套输入，新旧数量不代表覆盖场景一一对应。

用 wheel `0.dev20261126+main.fa6b536` 实测新生成的完整 `cv200` 组。数字为通过数 / 执行数。

| 算子 | A2 | A5 |
|---|---:|---:|
| `chunk_fwd_h` | 200 / 200 | 200 / 200 |
| `chunk_kda_bwd_recompute` | 200 / 200 | 0 / 200 |

A5 的 recompute 用例 ID 0–199 均在 `w`、`qg`、`kg` 超差。三路均执行了 200 次。已在该算子的 `suite.json` 设置 `"A5": false`，暂时排除其全部 6 组、711 条单算子用例。A2 照常调度。修复后可先用 `python3 tests/atk/smoke/run.py run --op chunk_kda_bwd_recompute --platform A5` 复测，再将 A5 设为 true 恢复 CI。

在仓库根目录重新生成这两组精度输入：

```bash
python3 tests/atk/smoke/cases/chunk_fwd_h/gen_chunk_fwd_h.py --output-dir /tmp/fwd-h-cases --summary
cd tests/atk/smoke/cases/chunk_kda_bwd_recompute
atk case -f chunk_kda_bwd_recompute.yaml -p gen_chunk_kda_bwd_recompute.py -dt 100 -en 0 -s 20260813
```

A5 的 `chunk_gated_delta_rule_fwd_h` 用例 `pr50-joint` ID 91 在本次入口抽查时超差，已从清单移出。以下五个守护算子暂缺可纳入的 CPU 双标杆通过用例，补齐后可按相同流程登记。

- `chunk_gated_delta_rule_fwd`
- `chunk_gated_delta_rule_fwd_prepare`
- `kda_gate_bwd_post`
- `kda_gate_bwd_post_varlen`
- `solve_tri`
