# 仓上算子 ATK 测试工程方案说明书

> 文档状态：测试团队确认稿
>
> 目标交付日期：2026-08-15
>
> 适用对象：本仓 Ascend C 算子及其公开调用接口
>
> 文档性质：目标目录、测试资产、使用方法和验收范围说明，不包含具体实现

## 1. 背景与目标

本仓算子测试采用 ATK 原生工程和命令行使用方式。开发者直接使用 `atk case`、`atk node ... task`、`atk pytorch`、`atk aclnn` 等 ATK 命令生成和执行用例，不在仓内建设另一套测试框架或顶层命令。

目录组织参考 [ops-transformer grouped_matmul 测试目录](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul/tests/st) 的算子 ST 资产模式：测试目录按接口保存 `atk_*.json` 和 `executor_*.py`。本仓在该模式上补充泛化 YAML 和约束生成 gen，并做以下统一：

- 所有算子测试资产集中放在仓库根目录 `test/`。
- 算子实现目录下不保存 JSON、YAML、gen、executor 或专项测试脚本。
- 每个算子目录只提供四类资产：看护 JSON、泛化 YAML、约束生成 gen、executor。
- `test/README.md` 是全仓测试工程的统一使用入口。
- 精度、泛化、性能、内存检查、NaN 脏数据、确定性和精度复检均使用 ATK 已有能力。
- CT 只用于 `ct viz` 精度可视化和 `ct dual analyze` 双标杆聚合分析。

本方案需要看护：

- `fla_npu.ops.ascendc.<op_name>` 主调用路径。
- aclnn 两段式调用路径。
- Ascend C `<<<>>>` 直调路径。
- A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）适用矩阵。
- 精度、泛化、边界、负向、性能、内存异常、脏数据、确定性和回归场景。

## 2. 总体原则

### 2.1 ATK 是唯一测试执行入口

仓上说明、日常执行和 CI 均使用 ATK 命令。仓内不再引入其他测试框架入口，也不重复实现 ATK 已有的：

- 用例生成和约束求解。
- DUT、golden、benchmark 调度。
- 双标杆精度判定。
- `performance_device` 任务和其集成的 `msopprof` 性能采集。
- mssanitizer 调度和结果汇总。
- NaN 等特殊值生成。
- 确定性计算。
- 多轮精度复检。
- XLSX 和任务报告生成。

如果锁定 ATK 版本缺少上述通用能力，应升级 ATK 或向 ATK 仓贡献，不在本仓复制一套同类能力。

### 2.2 测试资产集中归档

所有 ATK 单算子测试资产只能出现在根目录 `test/`。例如：

```text
test/<op_name>/
```

不得在以下目录新增同类资产：

- `fla/ops/ascendc/<op_name>/`
- `torch_custom/fla_npu/fla_npu/ops/ascendc/`
- 算子 `op_host/`、`op_kernel/`、`op_api/`、`docs/` 或 `examples/`
- 其他算子实现目录

已有工程级 UT 不属于本文定义的 ATK 单算子测试资产；后续新增的 ATK ST、泛化和专项测试必须进入 `test/`。

### 2.3 每个算子只维护四类文件

每个算子只维护：

1. 用于持续看护的 `atk_<op_name>.json`。
2. 泛化用例和约束 `<op_name>.yaml`。
3. 泛化约束生成 `gen_<op_name>.py`。
4. 执行、标杆和调用适配 `executor_<op_name>.py`。

不得为精度、性能、Sanitizer、NaN、确定性或复检分别复制 executor 或 case 文件。场景差异通过同一 JSON 的 case 属性、tag、case ID、SOC 和 route 描述。

### 2.4 文档、约束和用例必须一致

YAML 和 gen 中的 dtype、shape、layout、属性、可选输入、平台差异和非法组合必须与算子 README、设计文档和 API 文档一致。

双向核对要求：

- 从文档每条公开约束都能找到 YAML/gen 中的合法域、关系约束或负向生成规则。
- 从 YAML/gen 每条限制都能找到对应的公开文档依据。
- 文档变更影响约束时，同一变更必须同步更新 YAML、gen 和看护 JSON。
- 代码新增或收紧拦截时，必须同步补充文档与负向 case。

gen 不得通过收窄输入范围、删除失败组合或改变随机分布规避真实问题。

### 2.5 结果只有成功或失败

进入执行矩阵的用例必须实际执行，最终只能成功或失败。

- 环境、设备、工具、executor、标杆、结果文件或解析失败，均按失败处理。
- 不产生跳过数、预期失败数或独立错误数。
- 明确不适用于某 SOC 或 route 的用例必须在执行前由冻结矩阵过滤，不进入本次用例总数。
- 不能通过缩小 case 范围、修改阈值、删除失败 case 或只看 shell 返回码制造通过结论。

### 2.6 CT 使用边界

仓上不建设 CT 执行框架，也不使用 CT 承担用例生成、算子执行、性能、Sanitizer 或普通精度判定。

只允许：

- `ct viz`：查看 ATK 保存的 DUT、golden、benchmark 输出差异。
- `ct dual analyze`：分析 ATK 双标杆或多轮复检 XLSX。

其他测试能力全部使用 ATK。

## 3. 目标目录结构

```text
test/
├── README.md
├── <op_name>/
│   ├── atk_<op_name>.json
│   ├── <op_name>.yaml
│   ├── gen_<op_name>.py
│   └── executor_<op_name>.py
├── <another_op>/
│   ├── atk_<another_op>.json
│   ├── <another_op>.yaml
│   ├── gen_<another_op>.py
│   └── executor_<another_op>.py
└── ...
```

目录规则：

- `<op_name>` 使用仓上稳定 Python API 的 snake_case 名称。
- 一个算子的三条调用路径共用同一个目录和四类文件。
- 一个实现对应多个公开 API 时，按公开 API 分目录；README 中维护实现与 API 的映射。
- ATK 生成的 `result/`、`atk_output/`、日志、XLSX、profiling 和 sanitizer 产物不得提交。
- 每个算子目录不新增独立 README；共性使用方法只维护在 `test/README.md`。

## 4. 四类算子测试资产

### 4.1 看护用例 JSON

文件：

```text
test/<op_name>/atk_<op_name>.json
```

用途：

- 作为日常回归、CI 和发布验收直接执行的 ATK case 文件。
- 保存已经评审并需要长期看护的典型、边界、负向、性能、Sanitizer、脏数据、确定性和历史问题 case。
- 保存每条 case 的稳定 ID、SOC、route、输入规格、期望返回码和场景标记。

最低信息要求：

| 信息 | 要求 |
| --- | --- |
| case ID | 当前算子内唯一且长期稳定 |
| route | `ascendc`、`aclnn`、`direct_launch` |
| SOC | 明确 A2、A3、A5 适用范围 |
| 输入 | shape、dtype、layout、range、特殊值和可选输入状态 |
| 属性 | 所有影响 tiling、模板或语义的属性 |
| 期望 | 成功输出或明确返回码、异常类型、关键错误信息 |
| tag | accuracy、boundary、negative、performance、sanitizer、dirty_data、determinism、recheck、regression 等 |
| seed | 需要随机数据时记录稳定 seed |

JSON 要求：

- 必须符合锁定 ATK 版本的可执行 schema。
- 所有公开输出必须进入精度或语义检查。
- 负向用例必须精确验证返回码或异常语义，不能只验证“出现了某个异常”。
- JSON 是看护资产，不在测试过程中被静默覆盖。
- 泛化发现的稳定问题 case 经评审后固化到该 JSON。

### 4.2 泛化 YAML

文件：

```text
test/<op_name>/<op_name>.yaml
```

用途：

- 声明 ATK CaseGen 所需的参数域、参数关系、生成策略和原生精度标准。
- 作为泛化用例生成的规格输入。
- 表达合法与非法组合，而不是保存执行代码。

至少覆盖：

- A2、A3、A5 支持差异。
- dtype、layout、format 和 range。
- batch、sequence、head、K、V、chunk 等维度范围。
- 最小值、最大值、对齐边界、非整除尾块和典型值。
- fixed/varlen、可选输入有无、空 tensor 和状态输入输出。
- 会切换 tiling、编译模板、核类型或算法分支的组合。
- 文档声明不支持的组合及期望返回码。
- ATK 原生双标杆标准，例如锁定版本支持的 `cv_fused_double_benchmark`。

YAML 不得包含：

- 算子调用代码。
- tensor 构造实现。
- golden 或 benchmark 实现。
- 自定义精度公式或与 ATK 原生标准冲突的阈值。
- 为绕过缺陷而新增的无公开依据排除条件。

### 4.3 泛化约束生成 gen

文件：

```text
test/<op_name>/gen_<op_name>.py
```

职责：

- 将 YAML 约束转换为 ATK CaseGen 可消费的生成规则。
- 补充 YAML 难以表达但文档明确规定的关系约束。
- 生成合法、非法、边界、特殊值和组合覆盖 case。
- 使用固定 seed 生成可重放结果。
- 输出覆盖摘要和被约束过滤的原因。
- 将需要提交的看护 case 物化为 ATK JSON。

gen 必须保持：

- 只做约束表达和 case 物化。
- 不调用 DUT、golden 或 benchmark。
- 不执行精度比较。
- 不捕获执行异常并伪造成功。
- 不在脚本中散落未登记的关键 shape 列表。
- 同一输入 YAML 和 seed 生成稳定 case ID 与稳定结构。

文档对齐检查至少包括：

| 文档内容 | gen/YAML 对应项 |
| --- | --- |
| dtype 支持表 | dtype 参数域和 SOC 条件 |
| Shape 范围 | min、max、values、multiple_of 和关系约束 |
| layout/format | 枚举域和组合限制 |
| 可选输入 | absent/present/empty 状态 |
| 属性约束 | 属性枚举、范围和跨字段关系 |
| 平台差异 | SOC 条件分支 |
| 不支持场景 | 非法组合及 expected return code |

### 4.4 executor

文件：

```text
test/<op_name>/executor_<op_name>.py
```

职责：

- 按 ATK plugin/executor 合同构造输入 tensor。
- 根据 case 中的 route 调用对应 DUT。
- 提供两个独立标杆所需的 golden 和 benchmark。
- 处理输出命名、有效区和 ATK 需要的结构转换。
- 将返回码、异常和执行状态按 ATK schema 返回。

三条路径：

| route | DUT |
| --- | --- |
| `ascendc` | `fla_npu.ops.ascendc.<op_name>` |
| `aclnn` | aclnn GetWorkspaceSize + execute 两段式接口 |
| `direct_launch` | Ascend C `<<<>>>` 直调 |

executor 不得包含：

- 未在 JSON/YAML 登记的用例规格。
- 自定义精度指标或阈值。
- 性能通过阈值。
- 通过捕获任意异常返回成功的逻辑。
- 为通过测试而清零有效输出、缩小输入 range 或修改 case。
- 按 route 改变算子公开语义。

`direct_launch` 当前缺少 ATK 通用 backend 时，应将通用 direct-launch backend、runner ABI、case/result schema 和测试贡献到 ATK 仓。贡献完成前的临时适配只能承担 launch，不得在本仓另建测试调度框架。

## 5. test/README.md 统一说明

`test/README.md` 是开发者唯一需要阅读的仓上测试入口，至少包含以下章节。

### 5.1 测试工程简介

- ATK 在本仓测试中的定位。
- 本文定义的四类算子资产。
- `test/` 目录树和命名规则。
- 支持的 SOC 与三条调用路径。
- 结果只有成功或失败的规则。

### 5.2 工具版本表

README 必须记录：

| 工具 | 需要记录的内容 |
| --- | --- |
| ATK | 官方仓库、tag/完整 commit、ATK version、Python 版本 |
| CT | 正式获取地址、版本、文件或安装包校验值 |
| CANN | 已验证版本范围 |
| 驱动/固件 | 已验证版本范围 |
| Python | 支持版本 |
| SOC | A2、A3、A5 验证情况 |

工具升级必须单独评审，并重新执行代表性固定 case。

### 5.3 算子索引

README 维护：

- 算子名。
- 目录链接。
- 公开 API。
- 支持 route。
- 支持 SOC。
- 看护 JSON case 数量。
- 泛化 profile。
- golden 和 benchmark 来源。

### 5.4 快速使用

README 应提供本文第 7 至 13 节的可直接修改参数后执行的命令，覆盖：

- ATK 获取和安装。
- 环境准备。
- 泛化生成。
- 全量和单 case 精度。
- 负向用例。
- 三条调用路径。
- NaN 脏数据。
- 确定性。
- 精度复检。
- ATK `performance_device`/`msopprof` 性能测试。
- 四类 mssanitizer。
- CT 获取、`ct viz` 和 `ct dual analyze`。
- 结果查看和常见失败定位。

## 6. ATK 获取与环境准备

### 6.1 获取 ATK

ATK 官方源码以 [AECG/atk](https://gitcode.com/AECG/atk) 为准。正式接入时必须锁定 tag 或完整 commit，不直接跟随 `main`。

目标安装示例：

```bash
git clone https://gitcode.com/AECG/atk.git <atk_source_dir>
git -C <atk_source_dir> checkout <locked_tag_or_full_commit>

python3 -m venv <atk_venv_dir>
source <atk_venv_dir>/bin/activate
python -m pip install --upgrade pip
python -m pip install -r <atk_source_dir>/atk/ATK-dev/requirements.txt
python -m pip install <atk_source_dir>/atk/ATK-dev

atk --version
```

接入要求：

- 安装前确认 ATK LICENSE、NOTICE 和第三方依赖许可。
- ATK 只作为测试工具，不进入本仓运行时 wheel。
- README 中记录验证过的 revision 和 `atk --version`。
- CI 与本地使用同一个锁定版本。
- 禁止依赖开发者机器上来源不明的 `atk`。

### 6.2 环境准备

```bash
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=<physical_device_id>
export PYTHONPATH=<repo_root>/torch_custom/fla_npu:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=<writable_cache_dir>

atk --version
which atk
npu-smi info
```

执行前必须确认：

- 已安装与目标 SOC 匹配的最新算子包。
- `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 命中当前安装包。
- `PYTHONPATH` 命中本仓需要验证的 Python API。
- ATK 看到的逻辑设备号与 `--devices` 一致。

## 7. 用例生成

进入算子测试目录：

```bash
cd test/<op_name>
```

按 YAML 和 gen 生成泛化用例：

```bash
atk case \
  -f ./<op_name>.yaml \
  -p ./gen_<op_name>.py \
  -dt 1 \
  -en 0 \
  -s 20260815
```

生成后必须检查：

- ATK case schema 校验成功。
- 合法与非法用例数量符合 profile。
- coverage report 覆盖必选 dtype、layout、边界、SOC 和功能分支。
- 生成 case 与算子文档一致。
- 相同 YAML、gen 和 seed 可重放。
- 没有因为已知失败缩小输入 range 或排除 case。

需要纳入持续看护的 case，物化并评审后写入：

```text
test/<op_name>/atk_<op_name>.json
```

## 8. 精度与调用路径测试

### 8.1 全量精度

使用 ATK 的 NPU + CPU 双节点 accuracy 任务：

```bash
atk node --backend npu --devices 0 -o ./atk_output \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy \
  -p ./executor_<op_name>.py \
  -sp \
  -to 2000
```

要求：

- YAML 中配置的 ATK 原生双标杆标准是唯一精度阈值来源。
- DUT、golden、benchmark 使用同一份输入。
- 所有公开输出分别判定。
- 总任务数、执行成功数、执行失败数和精度结论均需要检查。
- shell 返回码为 0 但 ATK 报告存在 failed case 时，整体仍为失败。

### 8.2 单 case 定位

```bash
atk node --backend npu --devices 0 -o ./atk_output_single \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy \
  -p ./executor_<op_name>.py \
  -s <case_id> \
  -e <case_id_plus_one> \
  --save_data output \
  -sp \
  -to 2000
```

单 case 定位不得代替全量回归。

### 8.3 三条调用路径

同一 JSON 中分别登记 `ascendc`、`aclnn` 和 `direct_launch` case，由同一 executor 按 route 调用。

执行方式保持为 ATK 命令，通过 case ID 范围或白名单选择 route 对应 case：

```bash
atk node --backend npu --devices 0 \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy \
  -p ./executor_<op_name>.py \
  -wl '[<route_case_ids>]' \
  -sp \
  -to 2000
```

如果锁定 ATK 版本提供 `atk pytorch` 或 `atk aclnn` 快捷入口，可以在 README 中补充等价命令，但仓上主命令必须保持可追溯到同一 JSON 和 executor。

`direct_launch` 通用能力贡献到 ATK 后，README 应使用 ATK 社区最终确认的 backend 或子命令，不提前固化未发布的命令名。

### 8.4 负向用例

```bash
atk node --backend npu --devices 0 task \
  -c ./atk_<op_name>.json \
  --task run \
  -p ./executor_<op_name>.py \
  -wl '[<negative_case_ids>]' \
  -sp \
  -to 2000
```

必须同时检查：

- ATK 总任务数、success/failed。
- 每条 case 的 expected return code。
- executor 捕获的异常类型、返回码或关键错误信息。
- 不允许 executor 捕获任意异常后统一返回成功。

## 9. 泛化、脏数据与确定性

### 9.1 泛化批跑

先执行第 7 节的 `atk case`，再对生成的 ATK JSON 执行：

```bash
atk node --backend npu --devices 0 -o ./atk_output_generalization \
  node --backend cpu task \
  -c <generated_atk_json> \
  --task accuracy \
  -p ./executor_<op_name>.py \
  -sp \
  -to 2000
```

泛化报告至少保留 seed、生成器版本、完整物化 case、coverage report 和失败分类。

### 9.2 NaN 脏数据

NaN 的注入区域和期望行为在 YAML/gen 中声明，由 ATK 生成对应 case，再使用 accuracy 任务执行：

```bash
atk node --backend npu --devices 0 -o ./atk_output_nan \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy \
  -p ./executor_<op_name>.py \
  -wl '[<nan_case_ids>]' \
  --save_data output \
  -sp \
  -to 2000
```

至少覆盖：

- padding。
- 非整除 tail。
- unused slot。
- 文档声明不参与计算的状态区。
- 有效区 NaN 的传播或拦截语义。

有效输出被 NaN 污染时必须失败，并保留注入 mask 和首个污染位置。

### 9.3 确定性

使用 ATK 确定性任务：

```bash
atk node --backend npu --devices 0 \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy_dc \
  -p ./executor_<op_name>.py \
  -wl '[<determinism_case_ids>]' \
  -sp \
  -to 2000
```

要求：

- 相同输入重复执行。
- 输出、workspace 和可变状态按算子语义重置。
- 比较全部公开输出。
- 保存首个不一致轮次、输出和位置。
- bitwise 或 numeric 契约必须与算子文档一致。

## 10. 精度复检

对精度失败或疑似随机误差 case 使用 ATK `accuracy_lt`：

```bash
atk node --backend npu --devices 0 -o ./atk_output_recheck \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task accuracy_lt \
  -p ./executor_<op_name>.py \
  -wl '[<failed_case_ids>]' \
  --loop_nums 50 \
  --disable_id_seed \
  -mt 64 \
  -to 2000
```

复检要求：

- 固定 shape、dtype、layout、SOC、route、属性、indices、mask 和可选输入状态。
- 只随机化允许变化的输入数值。
- 每轮记录独立 seed。
- 不修改原始输入 range、精度标准或 case 结构。
- 原始失败 case 始终保留在报告中。
- `accuracy_lt` 不使用 `-sp`，避免阻塞多轮调度。
- 白名单整体加引号，例如 `-wl '[61,96,97]'`。

ATK 生成复检 XLSX 后，可执行：

```bash
ct dual analyze <atk_recheck_result.xlsx>
```

`ct dual analyze` 只做聚合分析，不替代 ATK 复检执行。

## 11. 性能测试

性能统一使用 ATK 的 device performance 任务，由 ATK 调度其集成的 `msopprof` profiling 能力，不在仓上直接拼装 `msopprof` 命令，也不使用 Python wall time：

```bash
atk node --backend npu --devices 0 -o ./atk_output_performance \
  node --backend cpu task \
  -c ./atk_<op_name>.json \
  --task performance_device \
  -p ./executor_<op_name>.py \
  -wl '[<performance_case_ids>]' \
  --performance_data 20,100,80 \
  --save_data profile \
  -sp \
  -to 2000
```

`--performance_data` 的 warmup、采集次数和统计次数由测试团队按资源冻结，README 不得使用未经确认的默认值作为发布标准。

性能判定要求：

- ATK profiling 实际采集到目标 kernel。
- 报告能够确认 `msopprof` 或锁定 ATK 版本约定的等价 device profiler 实际生效。
- 报告不为空。
- case ID、SOC、route、kernel、采集次数和 device 耗时可追溯。
- 基线来源和回退阈值经过评审。
- 性能失败时不得自动覆盖基线。

## 12. mssanitizer

### 12.1 前置检查

执行前必须确认：

- 算子使用 sanitizer 选项编译。
- opc 参数包含 `--op_debug_level=1 --op_debug_config=dump_cce,sanitizer`。
- 目标算子对象存在 sanitizer 符号。
- 运行时命中当前安装包，不是同名 built-in 旧对象。

```bash
nm <operator_object_file> | grep sanitizer
```

### 12.2 执行方式

ATK 是被 mssanitizer 启动的测试命令，`-ms` 和 `-msl` 交给 ATK 关联日志：

```bash
ATK_BIN=$(command -v atk)

mssanitizer --tool=memcheck --log-file ./mssanitizer_memcheck.log -- \
  "$ATK_BIN" node --backend npu --devices 0 task \
  -c ./atk_<op_name>.json \
  --task run \
  -p ./executor_<op_name>.py \
  -wl '[<sanitizer_case_ids>]' \
  -ms \
  -msl ./mssanitizer_memcheck.log \
  -sp \
  -to 2000
```

其他工具只替换 `--tool` 和日志名：

```bash
mssanitizer --tool=racecheck --log-file ./mssanitizer_racecheck.log -- <atk_command>
mssanitizer --tool=initcheck --log-file ./mssanitizer_initcheck.log -- <atk_command>
mssanitizer --tool=synccheck --log-file ./mssanitizer_synccheck.log -- <atk_command>
```

结果要求：

- 日志必须出现 `Start <tool> sanitizer on kernel ...` 或锁定版本的等价有效命中信息。
- 只有 `No active sanitizer tool on kernel ...` 时，本次测试失败。
- 同时检查 ATK 任务总数和每条 case 结果，不能只看 mssanitizer 或 shell 返回码。
- 原始日志和聚合摘要都需要保留。
- 需要区分真实越界、竞争、未初始化、同步问题与工具保守报告。

## 13. CT 获取与精度定位

### 13.1 获取要求

截至本文编写时，未检索到可公开核验的 CT 正式发行地址。正式交付前，测试团队必须在 `test/README.md` 补齐：

- 正式获取地址。
- 安装方式。
- 固定版本。
- 安装包或文件校验值。
- 支持的 Python/CANN/系统版本。

不得从来源不明的共享目录、临时链接或未经校验的 Python 包安装 CT。

安装后至少验证：

```bash
ct viz --help
ct dual analyze --help
```

### 13.2 ct viz

精度失败 case 先用 ATK `--save_data output` 保存 DUT 和标杆输出，再执行：

```bash
ct viz <atk_saved_output_or_case_dir>
```

用于判断：

- 误差是否集中在 padding、tail 或无效区。
- 是否存在整片符号、幅值、维度映射或索引错误。
- 是否出现 NaN/Inf 污染。
- 是结构性错误还是随机数值误差。

`ct viz` 结果不能改变 ATK 原始精度状态。

### 13.3 ct dual analyze

```bash
ct dual analyze <atk_result.xlsx>
```

只用于：

- ATK 双标杆结果聚合。
- `accuracy_lt` 多轮复检结果分析。

不能用于替代 ATK accuracy、performance、Sanitizer、NaN、确定性或用例生成任务。

## 14. 可看护场景

| 场景 | ATK 入口 | 主要证据 |
| --- | --- | --- |
| 精度 | `--task accuracy` | 总数、成功数、失败数、每个输出精度 |
| 双标杆 | ATK accuracy + YAML 原生标准 | DUT、golden、benchmark、XLSX |
| 泛化 | `atk case` + `--task accuracy` | seed、coverage、物化 JSON |
| 负向 | `--task run` | expected/actual return code、错误信息 |
| 确定性 | `--task accuracy_dc` | 多轮输出和首次差异 |
| 精度复检 | `--task accuracy_lt` | 50 轮 seed、逐轮结果、聚合结果 |
| 性能 | `--task performance_device` | `msopprof` profiling、device 耗时、基线 |
| 内存检查 | mssanitizer + ATK `--task run` | 有效命中日志、告警摘要 |
| NaN 脏数据 | ATK 特殊值 case + `--task accuracy` | 注入区域、输出污染位置 |
| 调用路径 | 同一 JSON/executor 的 route case | 三路径输出、返回码和 launch 信息 |

## 15. 日常工作流

### 15.1 新算子接入

1. 在 `test/<op_name>/` 创建四类文件。
2. 从算子文档整理 dtype、shape、layout、属性、可选输入、平台差异和非法组合。
3. 将泛化规格写入 YAML，将关系约束写入 gen。
4. 在 executor 中接入 DUT、golden 和 benchmark。
5. 用 `atk case` 生成并检查 coverage。
6. 将典型、边界、负向、专项和回归 case 固化到看护 JSON。
7. 运行三条调用路径。
8. 运行全量精度、泛化、NaN 和确定性。
9. 对性能 case 执行 `performance_device`。
10. 对高风险 kernel 执行对应 mssanitizer。
11. 对精度失败 case 先保存输出并执行 `ct viz`；非结构性误差再执行 `accuracy_lt` 和 `ct dual analyze`。
12. 更新 `test/README.md` 算子索引。

### 15.2 问题修复

1. 使用原 case ID、SOC、route 和 seed 重放。
2. 不改变原 shape、layout、dtype、range 和调用路径。
3. 精度问题先保存输出并定位结构性错误。
4. 越界、竞争、未初始化和同步问题使用对应 sanitizer。
5. 修复 kernel、标杆或真实语义问题。
6. 重跑原 case、所属完整用例集和受影响平台。
7. 将问题 case 保留为 regression。

## 16. CI 与发布验收

### 16.1 快速看护

- 每个算子典型 accuracy case。
- 三条 route 各至少一个 case。
- 少量固定 seed 泛化。
- 主要边界和负向 case。
- NaN 主 case。
- 确定性代表 case。

### 16.2 全量看护

- 全部看护 JSON。
- full 泛化 profile。
- A2、A3、A5 适用矩阵。
- 全部公开输出精度。
- 性能基线矩阵。
- kernel 高风险变更对应 sanitizer。
- 精度失败的复检和分析。

### 16.3 通过规则

- ATK 报告总用例数必须等于冻结矩阵用例数。
- 所有进入矩阵的 case 均成功且专项结论通过。
- 无跳过数。
- 工具缺失、报告为空、目标 kernel 未采集或 sanitizer 未命中均为失败。
- push 新实现或更新 ATK/executor/YAML/gen 后，旧结果失效。

## 17. 结果与归档

ATK 默认结果目录按锁定版本生成，README 应说明实际位置。至少保留：

```text
atk_output/
├── json_or_case_manifest/
├── output/
├── report/
├── profile/
└── logs/
```

归档信息：

- ATK 版本和完整 commit。
- 算子版本或 commit。
- case JSON 摘要。
- YAML/gen/executor 摘要。
- SOC、route、device 逻辑编号。
- 总用例数、成功数、失败数。
- 精度、性能、确定性和内存检查结论。
- seed、复检轮次和重放信息。

公开 PR、issue、评论和测试总结只描述测试项和结果，不写服务器、账号、绝对路径、环境路径、日志路径、token 或内部信息。

## 18. 交付验收标准

- [ ] 仓上算子测试只使用 ATK 命令，不存在其他顶层测试框架入口。
- [ ] 所有 ATK 算子测试资产集中在根目录 `test/`，算子实现目录无重复资产。
- [ ] 每个算子目录只包含看护 JSON、泛化 YAML、gen 和 executor 四类文件。
- [ ] `test/README.md` 包含目录说明、工具获取、环境准备和全部测试命令。
- [ ] ATK 官方地址、锁定 revision、版本和许可信息已确认。
- [ ] CT 正式获取地址、版本和校验值已补齐。
- [ ] CT 只使用 `ct viz` 和 `ct dual analyze`。
- [ ] YAML/gen 与 README、设计文档和 API 文档完成双向约束核对。
- [ ] 看护 JSON 可被锁定 ATK 版本直接执行。
- [ ] executor 同时支持 `ascendc`、`aclnn` 和 `direct_launch` 目标路径。
- [ ] 每个算子配置两个独立标杆和 ATK 原生双标杆标准。
- [ ] 精度、泛化、负向、NaN、确定性、复检和性能均通过 ATK 任务执行。
- [ ] mssanitizer 四类工具均有明确 ATK 执行方式和有效命中校验。
- [ ] A2、A3、A5 适用矩阵已冻结并可分别汇总。
- [ ] 结果只有成功或失败，不产生跳过数。
- [ ] 生成物、日志、XLSX、profiling 和 sanitizer 产物未提交到 Git。

## 19. 需要测试团队确认的事项

1. 2026-08-15 交付算子清单及 A2、A3、A5 必测矩阵。
2. `test/<op_name>/` 命名采用算子名还是公开 API 名。
3. ATK 正式 tag/commit、Python 版本和安装方式。
4. ATK JSON、YAML 和 executor 的冻结 schema。
5. `atk case` 生成结果写入看护 JSON 的评审和更新流程。
6. 每个算子的 golden、benchmark 来源及独立性要求。
7. ATK 双标杆原生标准和配置位置。
8. 三条 route 的 case 划分及 `direct_launch` 上游贡献版本。
9. smoke/full 泛化数量、seed 和 coverage 要求。
10. NaN 标准注入区域和有效区声明方式。
11. 确定性任务循环次数及 bitwise/numeric 契约。
12. `accuracy_lt` 默认 50 轮和资源配置。
13. `performance_device` 的 `msopprof` 后端、warmup、采集次数、统计次数和性能阈值。
14. mssanitizer 四类工具的 CI 矩阵和 debug 包构建方式。
15. CT 正式获取地址、版本、校验值及两个允许命令的参数格式。
16. ATK 报告归档周期、大小限制和 CI 展示方式。
17. `test/README.md` 的维护责任人和更新准入规则。

以上事项确认后，应直接固化到 `test/README.md`、四类算子资产和 CI 配置中，不能只保留在线下约定中。
