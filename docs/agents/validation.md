# 验证方法

## 验证分层

按改动风险选择验证范围，不要把“能编译”当作“功能正确”：

- 静态检查：`git diff --check`、schema/文档一致性检查、生成物检查。
- 环境检查：`python scripts/check_npu_env.py --build-only`。
- 构建验证：按目标 SOC 生成 wheel 或 OPP run 包。
- 打包验证：检查一体化 wheel、standalone wheel 和 run 包覆盖后的包名、import 面与 OPP 布局。
- 单算子验证：运行对应 `torch_custom/fla_npu/test/test_npu_<op>.py` 或 `test.sh --op <name>`。
- 端到端验证：运行 `examples/flash_gated_delta_rule.py` 或 `ci/run_example_st_cases.py`。
- 精度验证：对比参考实现，覆盖关键 shape、dtype、layout、dense/varlen 和边界 case。
- 性能验证：使用合适 profiling 工具，不用 Python wall time 直接作为性能结论。
- 内存/同步验证：疑似越界、未初始化、流水 hazard 或同步问题时，使用对应 sanitizer/profiling 方法验证。

## 测试矩阵

测试矩阵应覆盖语义路径、调度路径和值域路径，而不是只堆几个默认 shape：

- fixed length 和 varlen 都要覆盖；varlen 场景要覆盖 `cu_seqlens` 与 `chunk_indices` 成对出现、尾 chunk、短序列和多 chunk。
- head 关系要覆盖一一对应和 GVA/grouped 场景，例如 `H_out` 是 `H_qk` 的 1、2、4 倍，确认 head 映射和 workspace slot 没有串头。
- 目标维度要覆盖关键模板组合，例如 `chunkSize=64/128`、`V=128/256`、主 dtype 为 `fp16/bf16`，以及 gate/scale 等辅助输入与主输入 dtype 不同的 mixed 场景。
- 可选但当前不支持的参数要有反向用例，确认代码会明确拦截，而不是静默忽略或在 kernel 内崩溃。
- 输出支持非连续视图时，要验证最终 `ViewCopy` 或等价路径；不要只测 contiguous 输出。
- 多阶段 AIC/AIV 协同算子要覆盖长序列、多 chunk、多 head ratio，让同一个 core 连续处理多个 task，触发 workspace slot 复用和 ready/free flag 协议。

## 构建矩阵

常用 SOC 映射：

- A2：`ascend910b`
- A3：`ascend910_93`
- A5：`ascend950`

修改公共接口、公共 kernel 组件或跨平台逻辑时，应考虑多 SOC 编译和必要运行验证。若当前环境无法覆盖某个 SOC，应在结果中明确说明未覆盖原因。

## 打包和安装验证

一体化 wheel 和 `torch_custom/fla_npu` standalone wheel 都应使用 pip 项目名
`flash-linear-attention-npu`，安装后公开 import 名为 `fla_npu`。验证时至少确认：

- `python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl` 可安装。
- `python scripts/check_packaged_wheel_api.py` 通过。
- 安装后的 wheel 不依赖顶层 `fla` 包；Ascend C 入口是 `fla_npu.ops.ascendc`，Triton 入口是 `fla_npu.ops.triton`。
- standalone wheel + run 包 `--full` 或 `--install` 后，`site-packages/fla_npu/opp/vendors/fla_npu_transformer` 下能看到当前 run 包覆盖后的 op_api、tiling、kernel 和配置产物。
- wheel 和 run 包都要覆盖重复安装：同一个 wheel 连续强制安装两次、同一个 scoped run 包连续覆盖两次，最终 OPP 内容、`RECORD`、`set_env.bash` 和动态库选择保持一致。
- 源码或 Python 适配修改后重新构建 wheel，再覆盖旧 wheel 两次；新增公开适配时同时检查 API 可发现性和主要动态库加载通路。
- run 包覆盖后再强制安装新 wheel，确认 run 包增加的文件由更新后的 `RECORD` 清理，新 wheel 的 OPP 内容与归档完全一致。
- 使用 `python scripts/check_install_workflows.py --help` 查看统一看护入口。源码检查环境可用 `--skip-runtime-load`，但不能据此声明动态库加载通过；算子精度、泛化和性能仍使用各算子的专用测试。

## 精度问题处理

精度失败先分类，再决定处理方式：

- 如果误差呈结构性错位、整片符号/幅值异常、维度映射错误，优先回到索引、layout、任务分配、数据搬运和写回路径定位。
- 如果误差集中在无效区或 padding 区，先确认该区域的语义和测试后处理。
- 如果是随机数值误差，固定 shape/layout/属性后做多轮复检，再判断是否稳定劣于参考实现。

不要通过收窄输入 range、删除失败 case、降低覆盖强度或放宽阈值来制造通过结论。

## 单算子 ATK CPU 标杆一键验证

仓内单算子 NPU 看护使用 `tests/atk/run_test_cpu.sh` 调度 CPU 标杆流程。脚本不按算子名写
特殊分支，只要求 `tests/atk/<op>` 下存在 `atk_<op>.json` 和 `executor_<op>.py`。所有测试
动作都通过 ATK 发起；mssanitizer 阶段也只是在外层包裹同一条 ATK `task` 命令。该流程
不需要远端标杆服务：ATK 的 `--bm_device cpu` 负责 CPU 高精度真值，普通
`node --backend cpu` 负责 CPU 同精度对照。

### 前置准备

NPU 节点需要先准备 ATK、CANN、当前构建的 OPP 和仓内 Python 包：

```bash
cd <repo_root>
source <atk_venv>/bin/activate
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash

export TORCH_EXTENSIONS_DIR=<writable_cache_dir>

atk --version
npu-smi info -i <physical_npu_device>
```

`atk --version` 应与 `tests/atk/README.md` 中锁定版本一致。脚本会把
`-npu_device_id=<physical_npu_device>` 直接传给 ATK `node --devices`，例如
`-npu_device_id=6` 对应 `--devices 6`。不要在外部额外设置
`ASCEND_RT_VISIBLE_DEVICES` 造成设备号再次重映射。
`run_test_cpu.sh` 不会导出 `PYTHONPATH`；如当前环境无法 import 仓内 `fla_npu` 或
executor 依赖，请在调用脚本前自行设置。

mssanitizer 阶段需要使用带 sanitizer 信息的 debug OPP 包。构建时确认 `opc` 命令包含
`--op_debug_level=1 --op_debug_config=dump_cce,sanitizer`，执行前抽查目标对象中存在
sanitizer 符号：

```bash
nm <target_op_object> | grep sanitizer
```

`chunk_bwd_dqkwg` 的 ATK CPU 标杆使用
`tests/atk/chunk_bwd_dqkwg/chunk_bwd_dqkwg_cpu.py` 本地副本，executor 直接 import，
不在运行时按源码路径查找。
该算子本身是反向单算子，ATK case 顶层 `backward` 必须保持为 `false`，按“前向调用一个
反向算子”的方式做精度、性能、确定性和内存检测。

### 一键执行

在仓库根目录执行，`-op` 传 ATK 算子目录名：

```bash
bash tests/atk/run_test_cpu.sh \
  -op=<op> \
  -npu_device_id=<physical_npu_device>
```

示例：

```bash
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd \
  -npu_device_id=<physical_npu_device>
```

默认执行 `all`，顺序为 CPU 双标杆精度、性能、确定性和 mssanitizer。需要单独跑某一项时使用
`-scope=accuracy`、`-scope=performance`、`-scope=determinism` 或
`-scope=mssanitizer`。`gen_cases` 不属于 `all`，必须显式传入 `-scope=gen_cases`
才会触发。常用覆盖参数：

```bash
ATK_TIMEOUT=14400
CASE_START=0
CASE_END=1
ACCURACY_START=0
ACCURACY_END=1
PERFORMANCE_START=0
PERFORMANCE_END=1
DETERMINISM_START=0
DETERMINISM_END=1
MSS_START=0
MSS_END=1
PERFORMANCE_TIMEOUT=2000
MSS_TOOL=memcheck
MSS_KERNEL_NAME=<target_kernel_name>
MSS_LOG_PATH=<mssanitizer_log_path>
GEN_CASES_DTYPE_NUMBERS=100
GEN_CASES_EXTRA_NUMBERS=0
GEN_CASES_SEED=20260813
```

脚本默认自动识别 SOC。自动识别失败时按 A2 `ascend910b` 执行；A3 和 A5 可显式传入
`-soc=ascend910_93` 或 `-soc=ascend950`。A2、A3、A5 都使用同一套 ATK 命令和同一套
用例序号范围，不再按算子名或 SOC 写死 case ID。

ATK 文档 `ATK/docs/ATK使用指南/01 基础操作/任务执行.md` 说明 `--start 0 --end 2`
表示只执行下标 0 和 1；`ATK/docs/ATK使用指南/02 参考资料/任务执行参数说明.md` 也将
`-s/--start`、`-e/--end` 定义为执行起始和结束用例下标。因此 case id 混乱时只使用
`-s 0 -e 1` 这类序号切片表达“第几个 case”，不要用 `-wl` 依赖 JSON 内部 id。
默认不设置 case 范围，每个阶段执行全部用例。需要定位单条或子集时，由调用者按生成后的
JSON 顺序设置 `CASE_START/CASE_END` 或各阶段的 `*_START/*_END`。

### 脚本覆盖的 ATK 动作

泛化用例生成使用 ATK `case` 命令。ATK `case.py` 中 `-dt/--dtype_numbers`
表示每个 dtype 生成多少条普通用例，`-en/--extra_numbers` 表示边界用例数量；
`chunk_bwd_dqkwg` 的 q dtype 为 `bf16/fp16` 两类，因此默认 `-dt 100 -en 0`
会生成 200 条泛化用例。该动作不在 `all` 中：

```bash
bash tests/atk/run_test_cpu.sh -op=<op> -scope=gen_cases
```

等价的 ATK 命令为：

```bash
atk case -f ./<op>.yaml -p ./gen_<op>.py -dt 100 -en 0 -s 20260813
```

精度检查使用 CPU 高精度真值和 CPU 同精度对照。统一脚本会探测当前 ATK 的
`task --help`，仅在支持时加入 `--gm_init_flag` 和 `-sp`：

```bash
atk node --name npu_dut --backend npu --devices <npu_device_id> \
    --output_path ./atk_output/cpu_dual_reference \
  node --name cpu_reference --backend cpu \
    --output_path ./atk_output/cpu_dual_reference \
  task -c ./atk_<op>.json --task accuracy --bm_device cpu -p ./executor_<op>.py \
  -s <accuracy_start> -e <accuracy_end> [--gm_init_flag] [-sp] -mt 1 -to 14400
```

性能验证只使用 ATK `performance_device` 的 device profiler：

```bash
atk node --name npu_dut --backend npu --devices <npu_device_id> \
    --output_path ./atk_output/perf \
  task -c ./atk_<op>.json --task performance_device -p ./executor_<op>.py \
  -s <performance_start> -e <performance_end> --save_data profile [-sp] -to 2000
```

确定性验证使用 ATK `accuracy_dc`：

```bash
atk node --name npu_dut --backend npu --devices <npu_device_id> \
  task -c ./atk_<op>.json -p ./executor_<op>.py --task accuracy_dc \
  -s <determinism_start> -e <determinism_end>
```

内存检测由 `mssanitizer --tool=memcheck` 包裹 ATK `run` 任务。统一脚本仅在当前 ATK
支持时向 task 传入 `--mssanitizer -msl <log_path>`；无论是否支持 ATK 内存报告后处理，
都必须从外层 mssanitizer 原始日志确认目标 kernel 实际启动、结束且未检测到异常：

```bash
mssanitizer --tool=memcheck --kernel-name=<target_kernel_name> \
  --log-file=<mssanitizer_log_path> -- \
  atk node --name npu_dut --backend npu --devices <npu_device_id> \
  task -c ./atk_<op>.json -p ./executor_<op>.py --task run \
  [--mssanitizer -msl <mssanitizer_log_path>] \
  -s <mssanitizer_start> -e <mssanitizer_end>
```

每一项都必须同时检查 ATK 总任务数、失败数、精度或专项结论，以及 mssanitizer 日志是否真正命中目标 kernel。
没有命中 sanitizer 或报告中存在 failed case 时，本次验证不能记为通过。

## 结果记录

对外描述测试结果时，只写测试项和结果，不写本地机器、账号、绝对路径、临时目录或日志路径。若没有执行某项验证，写清楚原因，例如缺少 NPU、缺少 CANN 环境或依赖版本不满足。
