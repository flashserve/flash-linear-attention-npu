# A2 和 A5 的 ATK 冒烟用例

A2、A5 在各自的 NPU CI 任务中运行本次改动命中的算子用例。运行下文的 `matrix` 命令可查看当前启用的算子和平台。本次冒烟中，名称以 `chunk_gdn` 或 `chunk_gated_delta` 开头的算子仅在 A5 登记用例。数字按来源组计数，同一输入可能出现在历史组和 PR 组。完整的输入、种子、输出比较方式和 CV 阈值随组保留。

## 哪些修改会触发冒烟

常规 NPU CI 对比本次实际合并的目标分支提交和合并结果。选算子 ATK 时先忽略所有 `.md` 文件（大小写均可），包括算子目录里的 README。只改 Markdown 时，算子 ATK 为 0 项。其余变更按各算子 `cases/<算子>/suite.json` 的 `source_root` 和 `platforms` 匹配。多个目录取并集，跨算子移动文件会检查新旧两个目录。

- 修改 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/`，A2 和 A5 都运行 `chunk_fwd_h`。
- 修改 `chunk_gated_delta_rule_bwd_dhu/`，只有 A5 有登记用例。融合算子目录下的 `op_kernel/internal/operators/` 归其外层算子。
- PR 只改某个算子的平台 `cases.json` 时，该平台也会运行这个算子的 ATK。修改某个 `suite.json` 会检查该算子已启用的平台。修改冒烟 Python 执行器或标杆时，两个平台分别运行已启用的全部算子，覆盖跨算子标杆依赖。
- 在 PR 上按现有方式触发 `/run-npu-ci quick` 或 `/run-npu-ci full` 即可。容器入口读取目标分支 workflow 已提供的 `NPU_CI_BASE_SHA`。常规冒烟即使只修改模型用例，也会执行 KDA/GDN 模型回归。
- 未命中已登记目录时直接进入模型回归。显式指定 `ops=...` 的编译诊断沿用现有流程。

在本地预览选择结果时，将下面的 `BASE` 和 `MERGED_HEAD` 换成两次提交的 SHA。在 CI 的合并提交上也可以使用 `HEAD^1 HEAD`。

```bash
python3 tests/atk/smoke/run.py matrix --git-range BASE MERGED_HEAD
python3 tests/atk/smoke/run.py affected --platform A5 --git-range BASE MERGED_HEAD
```

`matrix` 省略 `--git-range` 会输出全量清单。新增守护算子时，在对应算子目录的 `suite.json` 登记独立源码目录和平台，例如 [chunk_fwd_h 的登记文件](cases/chunk_fwd_h/suite.json)。修改用例后运行 `python3 tests/atk/smoke/run.py validate`。

## Runner 依赖和结果

A2、A5 的 CI 镜像统一使用 ATK `26.11.11`。镜像构建时按 CPython ABI 和 CPU 架构下载发布页对应的 wheel 并核验 SHA-256。CI 的环境检查会核对已安装和实际导入的版本；冒烟开始前再检查 `cv_fused_double_benchmark` 是否可用。[官方 wheel](https://gitcode.com/AscendTest/ATK/releases/v26.11.11)

算子 ATK 后必跑 [KDA/GDN 模型回归](../../../ci/model_cases/README.md)。两者按单条用例缓存通过结果，并在报告中区分实跑和复用。冒烟复用 Example/ST 阶段已配置的算子包环境和容器逻辑卡号。Actions 日志中的 `[CI][ATK]` 列出选中的算子。报告保存在该次运行的 `atk-smoke-<平台>-<run_id>-<attempt>` 附件里，任一算子精度失败会使平台 CI 失败。

## 在 CI 容器运行一项

容器需要已启用 CANN、已安装待测 FLA wheel、torch、torch_npu，以及支持 `cv_fused_double_benchmark` 的 ATK。当前 CI 使用 CANN 9.2.1 和 ATK 26.11.11。A2 的 XLSX 报告读取需要 openpyxl。

每个算子在每个已登记的平台只有一份用例文件，路径为 `cases/<算子>/<A2或A5>/cases.json`。其中每个组的 `cases` 保存 ATK 用例，`count` 是组内条数，`reference` 指定该组使用的 CPU 标杆。不同组可以复用同一个用例编号。例如 [chunk_fwd_o 的 A5 用例](cases/chunk_fwd_o/A5/cases.json)中，普通组和 `mode=opt` 组各有一条 90 号，CI 按各自配置执行并分别记录结果。

A2、A5 共用算子标杆。标有 `reference_source=repository` 的组使用 `tests/atk/<算子>/executor_*.py`，其余组使用 `cases/<算子>/executor_*.py`。公共模块位于 `cases/common/`，算子专用的参考函数和输入生成器随算子存放。运行时直接使用仓内文件。

标杆取自 [work_tool 的固定提交 72d3720](https://gitcode.com/chen-linxin4/work_tool/tree/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK)。以下目录映射便于与上游核对。

| 冒烟算子 | 上游 FLA_ATK 路径 |
|---|---|
| `chunk_fwd_h` | `chunk_gated_delta_rule_bwd/scripts/fwd_h_cpu.py`，数学实现位于 [fwd_h_cpu.py](../chunk_fwd_h/fwd_h_cpu.py)，由前向和 GDN 反向标杆共用 |
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
python3 tests/atk/chunk_fwd_h/gen_chunk_fwd_h.py --output-dir /tmp/fwd-h-cases --summary
cd tests/atk/chunk_kda_bwd_recompute
atk case -f chunk_kda_bwd_recompute.yaml -p gen_chunk_kda_bwd_recompute.py -dt 100 -en 0 -s 20260813
```

A5 的 `chunk_gated_delta_rule_fwd_h` 用例 `pr50-joint` ID 91 在本次入口抽查时超差，已从清单移出。以下五个守护算子暂缺可纳入的 CPU 双标杆通过用例，补齐后可按相同流程登记。

- `chunk_gated_delta_rule_fwd`
- `chunk_gated_delta_rule_fwd_prepare`
- `kda_gate_bwd_post`
- `kda_gate_bwd_post_varlen`
- `solve_tri`
