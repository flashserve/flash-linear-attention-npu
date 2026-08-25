# ATK 单算子验证工程

本目录保存 `flash-linear-attention-npu` 仓内 Ascend C 算子的 ATK 单算子验证工程。
所有精度、性能、确定性、内存检测和用例生成动作都通过 ATK 发起；公共脚本只负责拼装
ATK 命令，不在脚本内导出 `PYTHONPATH`。

## 目录结构

```text
tests/atk/
|-- README.md
|-- run_test_cpu.sh
|-- common/
|   `-- _ascendc_common_executor.py
|-- <op_name>/
|   |-- README.md
|   |-- atk_<op_name>.json
|   |-- <op_name>.yaml
|   |-- gen_<op_name>.py
|   |-- scripts/
|   |   `-- <本算子专用脚本>
|   `-- executor_<op_name>.py
```

每个算子目录保留 `scripts/`，用于本算子专属杂项脚本、分析脚本或 CPU 标杆。当前除
`chunk_bwd_dqkwg/scripts/chunk_bwd_dqkwg_cpu.py` 外，其它算子的 `scripts/` 暂为空。

ATK 运行产生的 `atk_output/`、`result/`、profiling、sanitizer 日志、XLSX、Python 缓存
和临时输出不得提交。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `run_test_cpu.sh` | 统一入口，覆盖 CPU 双标杆精度、性能、确定性、mssanitizer 和用例生成 |
| `common/_ascendc_common_executor.py` | executor 共用的基础工具函数，例如 dtype 转换、case_spec 解析、确定性数据生成、有限值检查 |
| `<op>/executor_<op>.py` | 本算子的输入构造、CPU 标杆、NPU DUT 调用和 ATK `FunctionApi` |
| `<op>/gen_<op>.py` | 本算子的 ATK 泛化用例生成器 |
| `<op>/scripts/` | 本算子专用的杂项脚本、分析脚本或辅助标杆，不放跨算子公共逻辑 |
| `<op>/<op>.yaml` | ATK case 生成配置，shape 与 dtype 必须符合算子 README 和 tiling 限制 |
| `<op>/atk_<op>.json` | 已评审的 ATK 执行用例 |
| `<op>/README.md` | 本算子的输入限制、标杆来源、SOC 支持和执行示例 |

`common/` 只放跨算子复用的基础函数。具体 CPU 标杆、`run_cpu`、`run_npu`、输入生成和
`FunctionApi` 必须留在各自算子目录中；若需要额外脚本，放入本算子的 `scripts/`。

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

可选环境变量：

| 变量 | 说明 |
| --- | --- |
| `ATK_ENV` | ATK 虚拟环境目录；设置后脚本会 source `$ATK_ENV/bin/activate` |
| `CANN_ENV` | CANN `set_env.sh` 路径；设置后脚本会 source |
| `FLA_NPU_ENV` | `fla_npu_transformer` 的 `set_env.bash` 路径；设置后脚本会 source |
| `ATK_OUTPUT_ROOT` | ATK 输出根目录，默认是算子目录下的 `./atk_output` |
| `ATK_TIMEOUT` | 精度阶段超时时间，默认 `14400` |
| `PERFORMANCE_TIMEOUT` | 性能阶段超时时间，默认 `2000` |
| `MSS_TOOL` | mssanitizer 工具，默认 `memcheck` |
| `MSS_LOG_PATH` | ATK `-msl` 日志路径；不设置时使用脚本内置路径 |

## 统一脚本

基本用法：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -npu_device_id=<device_id>
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-op=<op_name>` | `tests/atk` 下的算子目录名 |
| `-npu_device_id=<id>` | 传给 `atk node --devices` 的 NPU 设备号；`gen_cases` 不需要 |
| `-scope=<scope>` | 执行动作，支持 `all/accuracy/performance/determinism/mssanitizer/gen_cases` |
| `-soc=<soc>` | 生成或运行时的 SOC 标识，支持 `ascend910b/A2`、`ascend910_93/A3`、`ascend950/A5` |

`all` 包含 `accuracy`、`performance`、`determinism` 和 `mssanitizer`。`gen_cases` 不在
`all` 中，必须显式指定。

示例：

```bash
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=gen_cases
```

## case 范围

不设置 case 范围时，脚本不会向 ATK 命令传入 `-s/-e`，ATK 会执行 JSON 中全部用例。

设置通用范围：

```bash
CASE_START=0 CASE_END=1 \
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6
```

也可以按动作单独设置范围：

| 变量 | 作用 |
| --- | --- |
| `ACCURACY_START/ACCURACY_END` | 精度与 NaN 检测 |
| `PERFORMANCE_START/PERFORMANCE_END` | 性能测试 |
| `DETERMINISM_START/DETERMINISM_END` | 确定性验证 |
| `MSS_START/MSS_END` | mssanitizer 内存检测 |

如果只设置 start 或 end 中的一个，脚本会直接报错，避免范围表达不完整。

## 测试动作

精度与 NaN 检测使用本机 NPU DUT 和本机 CPU reference 两个 ATK node：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -npu_device_id=<device_id> -scope=accuracy
```

性能测试使用 ATK `performance_device`：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -npu_device_id=<device_id> -scope=performance
```

确定性验证使用 ATK `accuracy_dc`：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -npu_device_id=<device_id> -scope=determinism
```

内存检测由 `mssanitizer` 包裹 ATK `run` 任务：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -npu_device_id=<device_id> -scope=mssanitizer
```

用例生成通过 ATK `case` 执行：

```bash
bash tests/atk/run_test_cpu.sh -op=<op_name> -scope=gen_cases
```

生成相关变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GEN_CASES_DTYPE_NUMBERS` | `100` | 传给 `atk case -dt` |
| `GEN_CASES_EXTRA_NUMBERS` | `0` | 传给 `atk case -en` |
| `GEN_CASES_SEED` | `20260813` | 传给 `atk case -s` |

## 算子索引

| 算子目录 | 公开接口或调用入口 | 约束说明 |
| --- | --- | --- |
| `causal_conv1d` | `fla_npu.ops.ascendc.causal_conv1d` | 见 [`causal_conv1d/README.md`](./causal_conv1d/README.md) |
| `causal_conv1d_bwd` | `fla_npu.ops.ascendc.causal_conv1d_bwd` | 见 [`causal_conv1d_bwd/README.md`](./causal_conv1d_bwd/README.md) |
| `chunk_bwd_dqkwg` | `fla_npu.ops.ascendc.chunk_bwd_dqkwg` | 见 [`chunk_bwd_dqkwg/README.md`](./chunk_bwd_dqkwg/README.md) |
| `chunk_bwd_dv_local` | `fla_npu.ops.ascendc.chunk_bwd_dv_local` | 见 [`chunk_bwd_dv_local/README.md`](./chunk_bwd_dv_local/README.md) |
| `chunk_fwd_o` | `fla_npu.ops.ascendc.chunk_fwd_o` | 见 [`chunk_fwd_o/README.md`](./chunk_fwd_o/README.md) |
| `chunk_gated_delta_rule_bwd_dhu` | `fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu` | 见 [`chunk_gated_delta_rule_bwd_dhu/README.md`](./chunk_gated_delta_rule_bwd_dhu/README.md) |
| `chunk_gated_delta_rule_fwd_h` | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h` | 见 [`chunk_gated_delta_rule_fwd_h/README.md`](./chunk_gated_delta_rule_fwd_h/README.md) |
| `chunk_kda_fwd` | `fla_npu.ops.ascendc.chunk_kda_fwd` | 见 [`chunk_kda_fwd/README.md`](./chunk_kda_fwd/README.md) |
| `chunk_local_cumsum` | `fla_npu.ops.ascendc.chunk_local_cumsum` | 见 [`chunk_local_cumsum/README.md`](./chunk_local_cumsum/README.md) |
| `chunk_scaled_dot_kkt` | `fla_npu.ops.ascendc.chunk_scaled_dot_kkt` | 见 [`chunk_scaled_dot_kkt/README.md`](./chunk_scaled_dot_kkt/README.md) |
| `kda_gate_cumsum` | `fla_npu.ops.ascendc.kda_gate_cumsum` | 见 [`kda_gate_cumsum/README.md`](./kda_gate_cumsum/README.md) |
| `prepare_wy_repr_bwd` | `fla_npu.ops.ascendc.prepare_wy_repr_bwd` | 见 [`prepare_wy_repr_bwd/README.md`](./prepare_wy_repr_bwd/README.md) |
| `prepare_wy_repr_bwd_da` | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_da` | 见 [`prepare_wy_repr_bwd_da/README.md`](./prepare_wy_repr_bwd_da/README.md) |
| `prepare_wy_repr_bwd_full` | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full` | 见 [`prepare_wy_repr_bwd_full/README.md`](./prepare_wy_repr_bwd_full/README.md) |
| `recompute_w_u_fwd` | `fla_npu.ops.ascendc.recompute_w_u_fwd` | 见 [`recompute_w_u_fwd/README.md`](./recompute_w_u_fwd/README.md) |
| `recurrent_gated_delta_rule` | `fla_npu.ops.ascendc.recurrent_gated_delta_rule` | 见 [`recurrent_gated_delta_rule/README.md`](./recurrent_gated_delta_rule/README.md) |
| `recurrent_kda` | `fla_npu.ops.ascendc.recurrent_kda` | 见 [`recurrent_kda/README.md`](./recurrent_kda/README.md) |
| `solve_tri` | `fla_npu.ops.ascendc.solve_tri` | 见 [`solve_tri/README.md`](./solve_tri/README.md) |

## 新增或维护算子

新增算子工程时按以下顺序处理：

1. 在 `tests/atk/<op_name>/` 下放置 `README.md`、`atk_<op_name>.json`、`<op_name>.yaml`、`gen_<op_name>.py`、`executor_<op_name>.py` 和 `scripts/`。
2. 在算子 README 中写清输入 shape、dtype、属性、可选输入、变长元数据和 tiling 限制。
3. `executor_<op_name>.py` 中保留本算子的 `build_inputs`、CPU 标杆、`run_cpu`、`run_npu` 和 `FunctionApi`。
4. 若需要公共基础函数，从 `tests/atk/common/_ascendc_common_executor.py` 引入；不要把算子专属逻辑放入 `common/`。
5. YAML 与 JSON 中的 shape 必须同时满足源码 README、tiling 检查和 executor 输入构造。
6. 修改后至少执行 `python` 语法导入检查；具备 NPU 环境时再跑 `accuracy`、`performance`、`determinism` 和 `mssanitizer`。

executor 使用公共目录的推荐写法：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from _ascendc_common_executor import _case_spec
```

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
