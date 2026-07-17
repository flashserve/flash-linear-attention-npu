# 统一算子测试入口

本目录保存算子测试执行代码，`tests/op_cases/<op_name>.json` 是能力范围和用例设计的唯一来源。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `<op>/accuracy/test_<op>.py` | 读取 JSON，调用 `fla_npu.ops.ascendc.<op>` 并比较 reference |
| `<op>/routes/test_aclnn_<op>.cpp` | 验证两段式 aclnn、workspace、executor、stream 和返回码 |
| `<op>/routes/test_direct_<op>.cpp` | 验证 tiling data、block dim、workspace 和 `<<<>>>` 发射 |
| `<op>/routes/test_legacy_<op>.py` | 仅在实现 legacy schema 时验证显式加载路径 |
| `<op>/ut/op_host/` | InferShape、tiling 和参数拦截单元测试 |
| `<op>/ut/op_kernel/` | 可独立验证的 kernel 模板与 tiling data 测试 |
| `<op>/performance/` | 使用 JSON 中 `performance` 标签执行 profiling |
| `<op>/st/` | 真实上层组合和 Example 回归 |

## 运行方式

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

全部主线 Ascend C 算子的三平台构建和 JSON 泛化执行使用统一入口：

```bash
python3 ci/run_operator_build_matrix.py --soc ascend910b
python3 ci/run_operator_build_matrix.py --soc ascend910_93
python3 ci/run_operator_build_matrix.py --soc ascend950

python3 ci/run_operator_generalization.py --soc ascend910b --device 0
python3 ci/run_operator_generalization.py --soc ascend910_93 --device 0
python3 ci/run_operator_generalization.py --soc ascend950 --device 0
```

构建入口必须生成本次命令的新 run 包；泛化入口必须实际运行 `test_json_generalization_cases`。两个入口的 `--dry-run` 结果均不能记为设备侧通过。

测试报告只记录平台、用例 ID、结果、精度指标和性能结论，不记录本地路径或环境信息。
