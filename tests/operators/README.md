# 统一算子测试入口

本目录保存算子测试执行代码，`tests/op_cases/<op_name>.json` 是能力范围和用例设计的唯一来源。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `<op>/accuracy/test_<op>.py` | 读取 JSON，调用 `fla_npu.ops.ascendc.<op>` 并比较 reference |
| `<op>/routes/test_aclnn_<op>.cpp` | 验证两段式 aclnn、workspace、executor、stream 和返回码 |
| `<op>/routes/test_direct_<op>.cpp` | 验证 tiling data、block dim、workspace 和 `<<<>>>` 发射 |
| `<op>/routes/test_legacy_<op>.py` | 仅在实现 legacy schema 时验证显式加载路径 |
| `<op>/routes/test_fast_kernel_<op>.py` | fast-kernel direct-launch Python 扩展回归；参数从唯一 JSON 读取 |
| `<op>/ut/op_host/` | InferShape、tiling 和参数拦截单元测试 |
| `<op>/ut/op_kernel/` | 可独立验证的 kernel 模板与 tiling data 测试 |
| `<op>/performance/` | 使用 JSON 中 `performance` 标签执行 profiling |
| `<op>/st/` | 真实上层组合和 Example 回归 |


## 历史资产迁移规则

- 算子源码目录不得保留 test/、tests/、ATK/；example/torch_custom 适配工程不得保留主线算子的独立用例目录，执行代码统一进入本目录对应算子的职责子目录。
- 历史 JSON、Python case 表和脚本 main 中的显式矩阵必须并入 tests/op_cases/<op_name>.json，并用 legacy 字段保留原格式、原 ID、执行资产和启用状态。
- ATK 的 all_*.json 不再提交；使用 tests/operators/_shared/legacy_cases.py 的 materialize 子命令临时生成，并在执行后删除。
- 原资产存在错误 schema 时不得静默修正或冒充覆盖；原样归档、标记 disabled 和原因，修正后再进入执行矩阵。
- 迁移完成后必须检查旧目录为空、manifest ID 唯一、原 JSON 可逐字段物化、执行资产存在以及 git diff 中没有生成物。

## 运行方式

统一精度入口会自动发现 `tests/op_cases/*.json`，不维护第二份算子列表：

```bash
bash tests/operators/run.sh --device 0
bash tests/operators/run.sh --device 0 --op chunk_bwd_dqkwg --soc ascend910b
```

```bash
python -m pytest tests/operators/<op_name>/accuracy/test_<op_name>.py
```

```bash
FLA_NPU_SOC=ascend950 \
FLA_NPU_CASE_TAGS=accuracy,generalization \
FLA_NPU_RUN_OPERATOR_TESTS=1 \
python -m pytest tests/operators/<op_name>/accuracy/test_<op_name>.py
```

```bash
python scripts/check_operator_compliance.py
```

默认聚合回归可直接执行：

```bash
FLA_NPU_RUN_OPERATOR_TESTS=1 python -m pytest tests/operators
```

该命令运行主精度、泛化、UT 和 ST；legacy 与 fast-kernel 属于可选扩展通路，默认只收集并明确跳过，
不得因当前 Python 环境未构建对应扩展而记为算子失败。两个可选通路必须在各自的独立构建环境中验证。

验证 `torch.ops.npu` legacy 通路时，先构建含 legacy extension 的 wheel，再显式打开通路开关：

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_BUILD_LEGACY_EXTENSION=1 \
python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl

FLA_NPU_RUN_OPERATOR_TESTS=1 FLA_NPU_RUN_LEGACY_TESTS=1 \
python -m pytest tests/operators/<op_name>/routes/test_legacy_<op_name>.py
```

验证 fast-kernel 通路时，使用专用脚本构建 `ascend_ops` 并执行对应 JSON 回归；脚本会设置
`FLA_NPU_RUN_FAST_KERNEL_TESTS=1`：

```bash
bash examples/fast_kernel_launch_example/build_and_test.sh <op_name>
```

手工运行已构建的 fast-kernel 测试时同样必须显式设置该开关：

```bash
FLA_NPU_RUN_FAST_KERNEL_TESTS=1 \
python -m pytest tests/operators/<op_name>/routes/test_fast_kernel_<op_name>.py
```

全部主线 Ascend C 算子的三平台构建和 JSON 泛化执行使用统一入口：

```bash
python3 ci/run_operator_build_matrix.py --soc ascend910b
python3 ci/run_operator_build_matrix.py --soc ascend910_93
python3 ci/run_operator_build_matrix.py --soc ascend950

python3 ci/run_operator_generalization.py --soc ascend910b --device 0
python3 ci/run_operator_generalization.py --soc ascend910_93 --device 0
python3 ci/run_operator_generalization.py --soc ascend950 --device 0

python3 ci/run_operator_accuracy.py --soc ascend910b --device 0
python3 ci/run_operator_accuracy.py --soc ascend910_93 --device 0
python3 ci/run_operator_accuracy.py --soc ascend950 --device 0
```

构建入口必须生成本次命令的新 run 包；泛化入口必须通过 `FLA_NPU_CASE_IDS` 逐 case 实际运行
`test_json_generalization_cases`；精度入口必须运行每个算子的 `test_main_ascendc_accuracy_backend`。设备 runner
超时后必须清理整个子进程组。三个入口的 `--dry-run` 结果均不能记为设备侧通过。

GitHub NPU CI 默认启用 `CI_RUN_OPERATOR_GENERALIZATION=true` 和 `CI_RUN_OPERATOR_ACCURACY=true`；指定 `CI_OPS` 时只跑
目标算子，未指定时覆盖全部已注册 Ascend C 算子。

测试报告只记录平台、用例 ID、结果、精度指标和性能结论，不记录本地路径或环境信息。
