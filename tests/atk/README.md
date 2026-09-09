# ATK 单算子验证工程

本目录保存 `flash-linear-attention-npu` 仓内 Ascend C 算子的 ATK 单算子验证工程。
所有精度、性能、确定性、内存检测和用例生成动作都通过 ATK 发起；公共脚本只负责拼装
ATK 命令，不在脚本内导出 `PYTHONPATH`。本文件维护 ATK 资产、执行和结果记录，不负责值域校准
或精度定位。

## 目录结构

```text
tests/atk/
|-- README.md
|-- run_test_cpu.sh
|-- common/
|   |-- _ascendc_common_executor.py
|   └-- check_atk_result.py
|-- <op_name>/
|   |-- README.md
|   |-- atk_<op_name>.json          # 逻辑分支覆盖用例（精度检测使用）
|   |-- atk_<op_name>_perf.json     # 性能精简用例（模型 case）
|   |-- atk_<op_name>_mss.json      # 内存检测精简用例（需覆盖所有 tilingKey）
|   |-- <op_name>.yaml
|   |-- gen_<op_name>.py
|   |-- scripts/
|   |   └-- <本算子专用脚本>
|   └-- executor_<op_name>.py
```

每个算子目录保留 `scripts/`，用于本算子专属杂项脚本、分析脚本或 CPU 标杆。当前除
`chunk_bwd_dqkwg/scripts/chunk_bwd_dqkwg_cpu.py` 外，其它算子的 `scripts/` 暂为空。

ATK 运行产生的 `atk_output/`、`result/`、profiling、sanitizer 日志、XLSX、Python 缓存
和临时输出不得提交。

## 文件职责

| 文件                                   | 职责                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `run_test_cpu.sh`                    | 统一入口，覆盖混合容差精度、性能、确定性、mssanitizer 和用例生成                         |
| `common/_ascendc_common_executor.py` | executor 共用的基础工具函数，例如 dtype 转换、case_spec 解析、确定性数据生成、有限值检查 |
| `<op>/executor_<op>.py`              | 本算子的输入构造、CPU 标杆、NPU DUT 调用和 ATK`FunctionApi`                            |
| `<op>/gen_<op>.py`                   | 本算子的 ATK 精度候选用例生成器                                                          |
| `<op>/scripts/`                      | 本算子专用的杂项脚本、分析脚本或辅助标杆，不放跨算子公共逻辑                             |
| `<op>/<op>.yaml`                     | ATK case 生成配置，shape 与 dtype 必须符合算子 README 和 tiling 限制                     |
| `<op>/atk_<op>.json`                 | 逻辑分支覆盖用例，精度检测使用                                                          |
| `<op>/atk_<op>_perf.json`            | 性能精简用例（模型 case）                                                                |
| `<op>/atk_<op>_mss.json`             | 内存检测与确定性精简用例（需覆盖所有 tilingKey）                                         |
| `<op>/README.md`                     | 本算子的输入限制、标杆来源、SoC 支持、TilingKey 清单、用例映射、实际选择记录和执行示例     |

`common/` 只放跨算子复用的基础函数。具体 CPU 标杆、`run_cpu`、`run_npu`、输入生成和
`FunctionApi` 必须留在各自算子目录中；若需要额外脚本，放入本算子的 `scripts/`。

## 用例规模与覆盖

ATK 只执行已固化的精度基线。执行前，接口、CPU 标杆、输入生成器和混合容差必须与校准记录
一致，JSON 使用其中的值域、有效区域和固定随机种子。本节只维护用例规模和覆盖维度。

`atk_<op_name>.json` 中的“全部用例”是指已设计的逻辑分支覆盖用例全部执行，不表示必须生成
大量 CASE。`atk_<op_name>_perf.json` 使用用户在算子开发开始时提供的模型 case，和功能精度
用例分开维护；性能用例只运行 NPU DUT，测量性能、资源占用和稳定性，不运行 CPU 标杆或做
CPU 精度对比。

每个算子的用例根据其接口和设计选择适用项：

- fixed length 和 varlen，包括短序列、多 chunk、尾 chunk 和相应元数据。
- 一一对应和 grouped/GVA head 关系，包括 head ratio 为 1 和大于 1。
- 关键 `chunkSize`、K/V 维度、主 dtype、辅助输入 dtype 和 layout。
- 可选输入存在或缺失、初始/最终状态和预留参数的当前处理方式。
- 普通块、边界块、partial chunk、padding、无效区和极端值域。
- 单 task 与同一 core 连续多 task，包括设计要求的 slot 复用和同步过程。
- 目标 SoC，以及公共逻辑涉及的其他支持 SoC。

正常用例使用已确认的 CPU 标杆生成预期结果；异常用例验证接口契约规定的 host 校验、错误
类型和返回码。具体算子的 README 列出适用的覆盖项及其对应 case。

## 正式验收用例包

每个进入单算子正式验收的算子必须具备以下三份可解析且非空的 JSON，三者来源不同，不能互相
替代：

| 文件 | 来源 | 完整条件 |
| --- | --- | --- |
| `atk_<op>.json` | `gen_cases` 生成的精度候选用例经筛选、补充后形成 | 每个可达逻辑分支具有独立最小用例，每个精度用例至少包含 3 个固定种子，并覆盖适用的正常、边界和异常场景 |
| `atk_<op>_perf.json` | 用户在算子开发开始时提供的模型 case | 每个模型 case 均保留原始 shape、dtype、属性和目标 SoC；存在性能目标时逐 case 记录基线、目标和统计方式 |
| `atk_<op>_mss.json` | 根据设计和实现中的全部可达 TilingKey 人工构造 | 每个可达 TilingKey 至少有一个最小代表用例，并覆盖该 key 下与内存、同步或复用有关的关键路径 |

`gen_cases` 只调用当前算子的 `gen_<op>.py` 生成精度候选用例，不生成 `_perf.json` 或
`_mss.json`。性能用例和 TilingKey 覆盖用例必须分别依据用户模型 case 和实际 TilingKey 人工
构造；脚本只检查文件完整性并执行测试。

算子 ATK README 必须建立三类映射：逻辑分支到精度 case id、用户模型 case 到性能 case id、
可达 TilingKey 到 `_mss.json` case id。文件存在但映射缺失或覆盖不全，仍视为用例包不完整。

## 运行前准备

调用脚本前需要在当前 shell 中准备好 ATK、CANN、OPP 和 Python 包路径：

```bash
source "$ATK_ENV/bin/activate"
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash
which atk
atk --version
npu-smi info
```

如果环境需要仓内 Python 包路径，请在调用脚本前自行设置。`run_test_cpu.sh` 不会修改
`PYTHONPATH`。

脚本启动时会校验 ATK 版本不低于 `26.8.8`（由 `REQUIRED_ATK_VERSION` 控制），低于该版本
直接退出；请确保 `atk --version` 输出的版本号满足要求。

可选环境变量：

| 变量                     | 说明                                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `ATK_ENV`              | ATK 虚拟环境目录；设置后脚本会 source`$ATK_ENV/bin/activate`                         |
| `CANN_ENV`             | CANN`set_env.sh` 路径；设置后脚本会 source                                           |
| `FLA_NPU_ENV`          | `fla_npu_transformer` 的 `set_env.bash` 路径；设置后脚本会 source                  |
| `ATK_OUTPUT_ROOT`      | ATK 输出根目录，默认是算子目录下的`./atk_output`                                     |
| `ATK_GM_INIT_MODE`     | GM 数据初始化模式，默认`on`；可设 `on/off`                                        |
| `REQUIRED_ATK_VERSION` | ATK 最低版本要求，默认`26.8.8`；一般无需修改                                         |
| `ATK_TIMEOUT`          | 精度阶段超时时间，默认`14400`                                                        |
| `DC_LOOP_NUMS`         | 确定性循环次数，默认`50`                                                             |
| `DC_TIMEOUT`           | 确定性阶段超时时间，默认`3600`                                                       |
| `PERFORMANCE_TIMEOUT`  | 性能阶段超时时间，默认`2000`                                                         |
| `MSS_TOOL`             | mssanitizer 工具，默认`memcheck`                                                     |
| `MSS_LOG_PATH`         | ATK`-msl` 日志路径；默认 `${ATK_OUTPUT_ROOT}/mssanitizer_<op>_<时间戳>.log`        |

## 统一脚本

基本用法：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name>
```

常用参数：

| 参数                    | 说明                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| `-op=<op_name>`       | `tests/atk` 下的算子目录名                                                                                   |
| `-npu_device_id=<id>` | 传给`atk node --devices` 的 NPU 设备号，默认`0`；`gen_cases` 不需要                                       |
| `-scope=<scope>`      | 执行动作，支持`all/accuracy/performance/determinism/mssanitizer/gen_cases`                                   |
| `-soc=<soc>`          | SOC 标识，支持`ascend910b/A2`、`ascend910_93/A3`、`ascend950/A5`；默认 `auto`，由 `npu-smi` 自动探测 |

`all` 包含 `accuracy`、`performance`、`determinism` 和 `mssanitizer`，并在运行前检查三份
用例 JSON 均可解析且非空。`gen_cases` 不在 `all` 中，必须显式指定。

示例：

```bash
bash tests/atk/run_test_cpu.sh -op=causal_conv1d
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=performance
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=determinism
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=gen_cases
```

## case 范围

不设置 case 范围时，脚本不会向 ATK 命令传入 `-s/-e`，ATK 会执行 JSON 中全部用例。

设置通用范围：

```bash
CASE_START=0 CASE_END=1 \
bash tests/atk/run_test_cpu.sh -op=causal_conv1d
```

也可以按动作单独设置范围：

| 变量                                  | 作用                 |
| ------------------------------------- | -------------------- |
| `ACCURACY_START/ACCURACY_END`       | 精度与 NaN 检测      |
| `PERFORMANCE_START/PERFORMANCE_END` | 性能测试             |
| `DETERMINISM_START/DETERMINISM_END` | 确定性验证           |
| `MSS_START/MSS_END`                 | mssanitizer 内存检测 |

如果只设置 start 或 end 中的一个，脚本会直接报错，避免范围表达不完整。

## 测试动作

### 全量精度执行

精度与 NaN 检测显式启动本机 NPU DUT 节点和 CPU 高精度 golden 节点；CPU 节点不再
提供同精度参考，精度标准统一为 `mixed_tolerance_bm`：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=accuracy
```

`chunk_gated_delta_rule_fwd` 使用 NPU DUT、六 ACLNN NPU benchmark 和 CPU FP64 golden
三路双标杆，精度入口为该算子 `README.md` 中的 `scripts/run_matrix.sh`；统一脚本仍用于其
性能、确定性和 mssanitizer。

### 性能执行

性能测试使用 ATK `performance_device`：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=performance
```

### 确定性执行

确定性验证使用 ATK `accuracy_dc`：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=determinism
```

### 内存检测执行

内存检测由 `mssanitizer` 包裹 ATK `run` 任务：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=mssanitizer
```

### 精度候选用例生成

精度候选用例生成通过 ATK `case` 执行：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=gen_cases
```

生成结果位于 `result/<op>/json/all_<op>.json`。开发者检查并筛选该文件后更新
`atk_<op>.json`；该命令不会修改三份正式验收 JSON。

生成相关变量：

| 变量                        | 默认值       | 说明                 |
| --------------------------- | ------------ | -------------------- |
| `GEN_CASES_DTYPE_NUMBERS` | `100`      | 传给`atk case -dt` |
| `GEN_CASES_EXTRA_NUMBERS` | `0`        | 传给`atk case -en` |
| `GEN_CASES_SEED`          | `20260813` | 传给`atk case -s`  |

## 执行阶段

开发迭代期间按以下顺序执行：

1. 确认 CPU 标杆版本，并完成 CPU 标杆自身验证；存在已确认的参考实现时，重新检查代表性用例。
2. 构建并安装当前代码，确认运行时加载的是本轮构建结果。
3. 需要新增或扩充精度用例时执行 `gen_cases`，检查生成结果后更新 `atk_<op>.json`。
4. 对每个受影响算子执行 `accuracy`；精度通过后，该版本才可以进入正式验收。

正式验收前，根据用户模型 case 准备 `_perf.json`，根据全部可达 TilingKey 准备 `_mss.json`，
并在算子 ATK README 中完成三类映射。正式验收时固定代码、CPU 标杆、三份测试文件和构建结果，
不设置 case 范围，对每个受影响算子执行 `all`；精度阶段必须执行全部 `(case, seed)` 组合，所有
组合均通过后才能判定精度验收通过。

## 算子索引

| 算子目录                           | 公开接口或调用入口                                     | 约束说明                                                                                    |
| ---------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `causal_conv1d`                  | `fla_npu.ops.ascendc.causal_conv1d`                  | 见[`causal_conv1d/README.md`](./causal_conv1d/README.md)                                   |
| `causal_conv1d_bwd`              | `fla_npu.ops.ascendc.causal_conv1d_bwd`              | 见[`causal_conv1d_bwd/README.md`](./causal_conv1d_bwd/README.md)                           |
| `chunk_bwd_dqkwg`                | `fla_npu.ops.ascendc.chunk_bwd_dqkwg`                | 见[`chunk_bwd_dqkwg/README.md`](./chunk_bwd_dqkwg/README.md)                               |
| `chunk_bwd_dv_local`             | `fla_npu.ops.ascendc.chunk_bwd_dv_local`             | 见[`chunk_bwd_dv_local/README.md`](./chunk_bwd_dv_local/README.md)                         |
| `chunk_fwd_o`                    | `fla_npu.ops.ascendc.chunk_fwd_o`                    | 见[`chunk_fwd_o/README.md`](./chunk_fwd_o/README.md)                                       |
| `chunk_gated_delta_rule_fwd`     | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`     | 见[`chunk_gated_delta_rule_fwd/README.md`](./chunk_gated_delta_rule_fwd/README.md)           |
| `chunk_gated_delta_rule_bwd_dhu` | `fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu` | 见[`chunk_gated_delta_rule_bwd_dhu/README.md`](./chunk_gated_delta_rule_bwd_dhu/README.md) |
| `chunk_gated_delta_rule_fwd_h`   | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h`   | 见[`chunk_gated_delta_rule_fwd_h/README.md`](./chunk_gated_delta_rule_fwd_h/README.md)     |
| `chunk_gated_delta_rule_fwd_prepare` | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_prepare` | 见[`chunk_gated_delta_rule_fwd_prepare/README.md`](./chunk_gated_delta_rule_fwd_prepare/README.md)；CPU 双标杆 + `mixed_tolerance_bm` |
| `chunk_kda_fwd`                  | `fla_npu.ops.ascendc.chunk_kda_fwd`                  | 见[`chunk_kda_fwd/README.md`](./chunk_kda_fwd/README.md)                                   |
| `chunk_local_cumsum`             | `fla_npu.ops.ascendc.chunk_local_cumsum`             | 见[`chunk_local_cumsum/README.md`](./chunk_local_cumsum/README.md)                         |
| `chunk_scaled_dot_kkt`           | `fla_npu.ops.ascendc.chunk_scaled_dot_kkt`           | 见[`chunk_scaled_dot_kkt/README.md`](./chunk_scaled_dot_kkt/README.md)                     |
| `kda_gate_cumsum`                | `fla_npu.ops.ascendc.kda_gate_cumsum`                | 见[`kda_gate_cumsum/README.md`](./kda_gate_cumsum/README.md)                               |
| `prepare_wy_repr_bwd`            | `fla_npu.ops.ascendc.prepare_wy_repr_bwd`            | 见[`prepare_wy_repr_bwd/README.md`](./prepare_wy_repr_bwd/README.md)                       |
| `prepare_wy_repr_bwd_da`         | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_da`         | 见[`prepare_wy_repr_bwd_da/README.md`](./prepare_wy_repr_bwd_da/README.md)                 |
| `prepare_wy_repr_bwd_full`       | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full`       | 见[`prepare_wy_repr_bwd_full/README.md`](./prepare_wy_repr_bwd_full/README.md)             |
| `recompute_w_u_fwd`              | `fla_npu.ops.ascendc.recompute_w_u_fwd`              | 见[`recompute_w_u_fwd/README.md`](./recompute_w_u_fwd/README.md)                           |
| `recurrent_gated_delta_rule`     | `fla_npu.ops.ascendc.recurrent_gated_delta_rule`     | 见[`recurrent_gated_delta_rule/README.md`](./recurrent_gated_delta_rule/README.md)         |
| `recurrent_kda`                  | `fla_npu.ops.ascendc.recurrent_kda`                  | 见[`recurrent_kda/README.md`](./recurrent_kda/README.md)                                   |
| `solve_tri`                      | `fla_npu.ops.ascendc.solve_tri`                      | 见[`solve_tri/README.md`](./solve_tri/README.md)                                           |

## 新增或维护算子

新增算子工程时按以下顺序处理：

1. 在 `tests/atk/<op_name>/` 下放置 `README.md`、三份正式验收 JSON、`<op_name>.yaml`、
   `gen_<op_name>.py`、`executor_<op_name>.py` 和 `scripts/`。
2. `atk_<op_name>.json` 从精度候选用例形成；`atk_<op_name>_perf.json` 按用户模型 case 手工
   建立；`atk_<op_name>_mss.json` 按全部可达 TilingKey 手工建立。
3. 在算子 README 中写清输入 shape、dtype、属性、可选输入、变长元数据和 tiling 限制。
4. `executor_<op_name>.py` 中保留本算子的 `build_inputs`、CPU 标杆、`run_cpu`、`run_npu` 和 `FunctionApi`。
5. 若需要公共基础函数，从 `tests/atk/common/_ascendc_common_executor.py` 引入；不要把算子专属逻辑放入 `common/`。
6. YAML 与 JSON 中的 shape 必须同时满足源码 README、tiling 检查和 executor 输入构造。
7. 修改后至少执行 Python 语法和导入检查；具备 NPU 环境时，按“执行阶段”和阶段 5 的验证路由运行对应测试。

### TilingKey 覆盖交付

新增或修改 TilingKey、模板选择条件或 tiling 分支时，算子 ATK README 必须维护完整的 TilingKey 覆盖表。清单来源包括 host tiling、模板注册和 kernel 分派代码，至少记录：

| TilingKey | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `<key>` | dtype、layout、shape、属性和平台条件 | case id | case id | case id | A2/A3/A5 | host tiling UT 或运行时记录 |

维护要求：

1. 每个可达 key 都要有普通和边界用例；同一 key 的不同运行时分支也要有对应覆盖。
2. 覆盖表中的每个 key 都必须注明对应的 case id，并能在 `gen_<op>.py` 或生成的 JSON 中找到这些 case。
3. `atk_<op>_mss.json` 至少放入每个 key 的精简用例；性能路径涉及某个 key 时，`atk_<op>_perf.json` 也要覆盖该 key。
4. 用例中的输入条件只表示预期 key，必须补充 host tiling UT 或运行时记录确认实际选中的 key。没有实际选择证据时，不得在 README 中标记为已覆盖。
5. 不同 SoC 的 tiling 条件不一致时，按 SoC 分别记录覆盖；不适用的 key 要注明原因。

executor 使用公共目录的推荐写法：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from _ascendc_common_executor import _case_spec
```

## 验收结果记录

正式验收结果写入当前算子的 ATK README，至少包含：

- CPU 标杆、测试文件和被测代码版本，以及参考实现对齐或用户确认情况。
- 目标 SoC、执行的测试动作和结果；精度用例总数、失败数和必要的错误分类。
- 逻辑分支、边界、异常、TilingKey、确定性和内存检查的覆盖结论。
- 接口或功能修改的新增、变化和原有场景回归结论。
- 性能相关改动的目标用例、优化前后数据、统计方式、瓶颈变化及其他支持场景的变化。

存在性能目标时，必须列出 `_perf.json` 中全部模型 case 的实测结果：

| case id | 模型 shape | SoC/dtype | 对比基线 | 性能目标 | 实测性能 | 与基线比值 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `<id>` | `<shape>` | `<soc>/<dtype>` | `<baseline>` | `<target>` | `<result>` | `<ratio>` | 达标/未达标 |

每个模型 case 单独判断，不能用平均值掩盖未达标 case。任一 case 未测试、测试条件与目标定义不一致
或未达到目标时，整体结论必须写为“未达标”；只有用户明确接受该差异后，才能将其作为例外归档。
最终记录同时给出达标数/总数、未达标 case 及差距，以及整体性能目标是否完成。

## 提交流程检查

提交前建议检查：

```bash
rg -n "_ascendc_common_executor|parents\\[1\\]" tests/atk
rg -n "atk_output|result/|\\.xlsx|__pycache__" tests/atk
```

预期结果：

- 公共工具只存在于 `tests/atk/common/_ascendc_common_executor.py`。
- 需要公共工具的 executor 都从 `parents[1] / "common"` 加载。
- 不提交 ATK 运行输出和 Python 缓存。
