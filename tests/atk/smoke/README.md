# A2 和 A5 的 ATK 冒烟用例

A2、A5 在各自的 NPU CI 任务中运行本次改动命中的算子用例。当前有 22 个算子、38 个算子与平台组合。A2 单独选入 3,718 条，A5 单独选入 9,055 条。本次冒烟中，名称以 `chunk_gdn` 或 `chunk_gated_delta` 开头的算子仅在 A5 登记用例。数字按来源组计数，同一输入可能出现在历史组和 PR 组。完整的输入、种子、输出比较方式和 CV 阈值随组保留。

## 哪些修改会触发冒烟

常规 NPU CI 对比本次实际合并的目标分支提交和合并结果。改动 [source_roots.json](source_roots.json) 中某个算子目录下的文件，就运行该算子在 [suites.json](suites.json) 中登记的全部用例。每个平台独立选择，多个目录取并集。删除文件也会命中，跨算子移动文件会检查新旧两个目录。

- 修改 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/`，A2 和 A5 都运行 `chunk_fwd_h`。
- 修改同级的 `chunk_gated_delta_rule_fwd_h/`，只有 A5 有登记用例。融合算子目录下的 `op_kernel/internal/operators/` 归其外层算子。
- 未命中已登记目录时跳过 ATK。显式指定 `ops=...` 的编译诊断沿用现有流程。

在本地预览选择结果时，将下面的 `BASE` 和 `MERGED_HEAD` 换成两次提交的 SHA。在 CI 的合并提交上也可以使用 `HEAD^1 HEAD`。

```bash
python3 tests/atk/smoke/run.py matrix --git-range BASE MERGED_HEAD
python3 tests/atk/smoke/run.py affected --platform A5 --git-range BASE MERGED_HEAD
```

`matrix` 省略 `--git-range` 会输出全量清单。新增守护算子时，在 `source_roots.json` 登记独立源码目录，在 `suites.json` 登记用例组和平台。修改清单后运行 `python3 tests/atk/smoke/run.py validate`。

## Runner 依赖和结果

在 runner 镜像内提供支持 `cv_fused_double_benchmark` 的 ATK，或在该 runner 上预配同架构和 Python ABI 的 Docker 卷 `fla-atk-smoke-cp312`。卷根目录应直接包含 `atk/` 和对应发行包元数据，镜像应提供 `atk` 命令。命中用例时 CI 将已有卷只读挂载到 `/opt/fla-atk-smoke`，冒烟进程优先从该目录导入。缺少双标杆时该平台任务失败。

冒烟复用 Example/ST 阶段已配置的算子包环境和容器逻辑卡号。Actions 日志中的 `[CI][ATK]` 列出选中的算子。报告保存在该次运行的 `atk-smoke-<平台>-<run_id>-<attempt>` 附件里，任一算子精度失败会使平台 CI 失败。

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
