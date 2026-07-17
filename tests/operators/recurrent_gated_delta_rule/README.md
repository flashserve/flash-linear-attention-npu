# RecurrentGatedDeltaRule 测试归档

## 1. 唯一用例规格

`tests/op_cases/recurrent_gated_delta_rule.json` 统一保存 shape、dtype、layout、属性、可选输入、SOC、运行通路、随机种子、
参考实现、容差和预期返回码。新 case 必须先进入 JSON，再由执行端按 case ID 消费。

## 2. 归档内容

| 路径 | 内容 |
| --- | --- |
| `common/case_matrix.py` | 本算子 JSON 加载、tag/route 筛选和 case ID 环境变量 |
| `accuracy/test_recurrent_gated_delta_rule.py` | `fla_npu.ops.ascendc` 主精度、泛化、边界和回归入口 |
| `routes/test_aclnn_recurrent_gated_delta_rule.cpp` | aclnn 两段式接口签名、workspace/executor/stream 契约 |
| `routes/test_direct_recurrent_gated_delta_rule.cpp` | host tiling 结果驱动的 `<<<>>>` 参数和 launch 契约 |
| `ut/op_host/test_contract.py` | manifest、SOC、返回码、host 负向用例静态契约 |
| `ut/op_kernel/test_contract.py` | kernel 入口、tiling key 说明和 direct launch 静态契约 |
| `performance/profile.py` | 读取 performance tag 并通过 msopprof 运行设备侧 profiling |
| `st/test_example.py` | example tag 与仓内数值执行后端的 ST 入口 |

- legacy 通路：未实现，不生成 `torch.ops.npu` 测试。

现有数值/reference 后端：`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta/test_accuracy.py`。该后端由 canonical 入口传入
`FLA_NPU_CASE_MANIFEST`、`FLA_NPU_CASE_IDS` 和 `FLA_NPU_OPERATOR`；关键 shape、dtype、属性组合不在
canonical 脚本中重复定义。

## 3. 执行命令

```bash
pytest -q tests/operators/recurrent_gated_delta_rule/accuracy/test_recurrent_gated_delta_rule.py
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/recurrent_gated_delta_rule/accuracy/test_recurrent_gated_delta_rule.py
FLA_NPU_CASE_TAGS=generalization FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/recurrent_gated_delta_rule/accuracy/test_recurrent_gated_delta_rule.py
pytest -q tests/operators/recurrent_gated_delta_rule/ut
python tests/operators/recurrent_gated_delta_rule/performance/profile.py --dry-run
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/recurrent_gated_delta_rule/st/test_example.py
```

A2/A3/A5 通过 `FLA_NPU_SOC` 选择。精度逐项比较全部公开输出并检查 NaN/Inf；性能只使用 msopprof
设备侧结果，按 JSON 的 `expect.requirement` 对比 Triton 或当前主线基线。报告记录平台、case 总数、通过数和
失败 case ID，不记录本地环境路径。
