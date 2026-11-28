# A2 / A5 算子 ATK 冒烟

常规 NPU CI 在 Example/ST 后运行变更命中的算子，再运行已登记的模型用例。注册表为空时，两个平台均可执行现有的 Example/ST。显式指定 `ops=...` 的编译任务沿用原流程。

## 接入算子

在 `cases/<算子>/` 下放置 CPU 标杆、所需辅助文件和 `suite.json`。同一个算子可为 A2、A5 分别登记用例。下面是登记格式，其中 `source_root` 指向算子的独立源码目录。

```json
{"source_root":"fla/ops/ascendc/<分类>/<算子>","platforms":{"A2":true,"A5":false}}
```

每个已登记平台应提供 `<平台>/cases.json`。文件包含一个组列表。每组的 `group` 为组名，`count` 等于 `cases` 条数，`reference` 指定本算子目录下的 `executor_*.py`。组可通过 `require_fp64_golden` 要求高精度标杆输出 FP64。平台设为 `false` 时仍可本地复测。用以下命令校验完整清单、预览变更命中范围、或在已安装 FLA 和 ATK 的对应 NPU 容器中运行指定组。

```bash
python3 tests/atk/smoke/run.py validate
python3 tests/atk/smoke/run.py matrix --git-range BASE MERGED_HEAD
python3 tests/atk/smoke/run.py run --op OP --platform A5 --group GROUP --device 0
```

修改源码、算子登记或平台用例会触发对应的平台。修改冒烟 Python 标杆或执行框架会重新选择所有已启用的算子，以覆盖跨算子依赖。选择器忽略 Markdown。

## 判定、报告与缓存

A2 和 A5 镜像按架构安装并校验 ATK 26.11.11。每个组由 NPU DUT、CPU 同精度和 CPU 高精度标杆三路执行，核对用例标识、输出数量、形状和 `cv_fused_double_benchmark` 结果。实跑报告和复用证据随平台任务上传。

受保护目标分支的 workflow 在容器启动前恢复同平台、同架构及相同可执行源码的缓存，执行后有新增通过记录时保存快照。每条记录还校验镜像、安装产物、硬件和实际输入。缺少显式种子的 ATK 用例、未通过的结果、损坏的记录或环境变化的用例需要实跑。单纯删减同组的用例不会改变其他用例的缓存键。日志中的 `reused` 和 `to run` 分别显示复用和待执行数。
