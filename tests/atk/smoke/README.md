# A2 和 A5 的 ATK 冒烟用例

每个 CI 任务运行一个算子在一个平台上已通过的用例。当前有 22 个算子、38 个算子与平台组合。A2 单独选入 3,718 条，A5 单独选入 9,058 条。本次冒烟中，名称以 `chunk_gdn` 或 `chunk_gated_delta` 开头的算子仅在 A5 登记用例。数字按来源组计数，同一输入可能出现在历史组和 PR 组。完整的输入、种子、输出比较方式和 CV 阈值随组保留。

## CI 如何发现任务

1. 检出本次要测的 FLA 提交，执行下面的命令。第一条检查每份用例 JSON 的数量、ID、执行器与精度标准。第二条把 `suites.json` 中有用例的算子和平台输出成 GitHub Actions 矩阵。
2. 在 discovery job 的 step 中把矩阵写入 `GITHUB_OUTPUT`，并将该 step 的 `matrix` 声明为 job output。
3. 下游 ATK job 设置 `needs: discover`、`strategy.matrix: ${{ fromJSON(needs.discover.outputs.matrix) }}` 和 `name: ATK / ${{ matrix.op }} / ${{ matrix.platform }}`。每项沿用对应 A2 或 A5 runner、设备分配和同一 FLA 提交构建的 wheel。

```bash
python3 tests/atk/smoke/run.py validate
echo "matrix=$(python3 tests/atk/smoke/run.py matrix)" >> "$GITHUB_OUTPUT"
```

在已有算子和平台下增加同组用例，更新 `cases/<算子>/<平台>/<组名>.json` 以及清单中该组的 `count`。新组要在 [suites.json](suites.json) 添加一项。首次登记新的算子与平台组合时，`matrix` 自动多输出一项。例如，`chunk_kda_fwd_fused` 目前只输出 A5，`chunk_gated_delta_rule_bwd_finalize` 目前也只有 A5。

## 在 CI 容器运行一项

容器需要已启用 CANN、已安装待测 FLA wheel、torch、torch_npu，以及支持 `cv_fused_double_benchmark` 的 ATK。基线使用 CANN 9.2.1，A2 为 ATK 26.7.8，A5 为 26.9.24。A2 的 XLSX 报告读取需要 openpyxl。

每个算子的 CPU 标杆放在 `cases/<算子>/executor_*.py`，A2、A5 共用。每个来源组在 `suites.json` 的 `reference` 字段指定实际文件。公共模块位于 `cases/common/`，算子专用的参考函数和输入生成器随算子存放。运行时直接使用仓内文件。

标杆取自 [work_tool 的固定提交 72d3720](https://gitcode.com/chen-linxin4/work_tool/tree/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK)。以下目录映射便于与上游核对。

| 冒烟算子 | 上游 FLA_ATK 路径 |
|---|---|
| `chunk_fwd_h` | `chunk_gated_delta_rule_bwd/scripts/fwd_h_cpu.py`，以 `executor_chunk_fwd_h.py` 存放 |
| `chunk_kda_fwd`、`chunk_kda_fwd_fused` | `chunk_kda_fwd/executor_chunk_kda_fwd_v2.py`、`chunk_kda_fwd/executor_chunk_kda_fwd.py` |
| `chunk_kda_fwd_prepare`、`chunk_kda_fwd_finalize` | `chunk_kda_fwd_prepare_a2/`、`chunk_fwd_finalize/` |
| `chunk_kda_bwd_prepare` | `chunk_kda_bwd_prepare_a3/` |
| `recurrent_gated_delta_rule` | `recurrent_gated_delta_rule/rgdn_A5/` |

其余标杆保留上游算子目录名和文件名。`chunk_gated_delta_rule_fwd_h` 的所有 A5 用例统一使用 `executor_gdn_fwd_h_state.py`。KDA Forward V2 和 GDN 反向整链的导入路径已适配仓内布局。

从 FLA 仓库根目录执行。CI 将 `--op` 和 `--platform` 替换为矩阵值，将 `--device` 替换为容器内分配的逻辑卡号。

```bash
python3 tests/atk/smoke/run.py run --op chunk_fwd_o --platform A2 \
  --device 0 --output atk-results
```

每个来源组单独启动 ATK，运行 NPU、CPU 同精度和 CPU 升精度三路。任一用例超差、缺少输出或缺少报告都会让命令返回非零。日志和原生报告写入独立的 `atk-results/<算子>-<平台>-*/<组名>/`。加 `--group <组名>` 可重跑指定组。

## 入选依据

11 月 25 日和 26 日的 A2、A5 真卡基线使用 FLA wheel `0.dev20261124+main.fe8a05d`，平台各自按通过记录选取。11 月 26 日复测组保留执行器原始输出类型，其余组保留上次实测的输出类型处理。KDA chunk32 的 state 配置和 A5 的 ChunkFwdO 优化配置各自保留独立用例组。[标杆提交](https://gitcode.com/chen-linxin4/work_tool/tree/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK)

A5 的 `chunk_gated_delta_rule_fwd_h` 用例 `pr50-joint` ID 91 在本次入口抽查时超差，已从清单移出。以下五个守护算子暂缺可纳入的 CPU 双标杆通过用例，补齐后可按相同流程登记。

- `chunk_gated_delta_rule_fwd`
- `chunk_gated_delta_rule_fwd_prepare`
- `kda_gate_bwd_post`
- `kda_gate_bwd_post_varlen`
- `solve_tri`
