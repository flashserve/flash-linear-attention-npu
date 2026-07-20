# RecurrentKda 测试归档

## 1. 唯一用例规格

`tests/op_cases/recurrent_kda.json` 统一保存 shape、dtype、layout、属性、可选输入、SOC、运行通路、随机种子、
参考实现、容差和预期返回码。新 case 必须先进入 JSON，再由执行端按 case ID 消费。

## 2. 归档内容

| 路径 | 内容 |
| --- | --- |
| `common/case_matrix.py` | 本算子 JSON 加载、tag/route 筛选和 case ID 环境变量 |
| `accuracy/test_recurrent_kda.py` | `fla_npu.ops.ascendc` 主精度、泛化和回归入口 |
| `routes/test_aclnn_recurrent_kda.cpp` | aclnn 两段式接口签名、workspace/executor/stream 契约 |
| `routes/test_direct_recurrent_kda.cpp` | host tiling 结果驱动的 `<<<>>>` 参数和 launch 契约 |
| `routes/test_legacy_recurrent_kda.py` | 显式加载 `torch.ops.npu` 后的 legacy 入口 |
| `ut/op_host/test_contract.py` | manifest、SOC、返回码、host 负向用例静态契约 |
| `ut/op_kernel/test_contract.py` | kernel 入口、tiling key 说明和 direct launch 静态契约 |

底层 PTA 调试入口位于 `fla/ops/ascendc/kda/recurrent_kda/tests/pta/`。标准测试入口只从
`tests/op_cases/recurrent_kda.json` 读取 case，不维护第二份 shape 表。

## 3. 执行命令

```bash
pytest -q tests/operators/recurrent_kda/accuracy/test_recurrent_kda.py
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/recurrent_kda/accuracy/test_recurrent_kda.py
FLA_NPU_RUN_OPERATOR_TESTS=1 FLA_NPU_RUN_LARGE_SHAPE_TESTS=1 FLA_NPU_CASE_IDS=recurrent_kda_kimi_tnd_h96_d128_runtime_cu_smoke pytest -q tests/operators/recurrent_kda/accuracy/test_recurrent_kda.py
pytest -q tests/operators/recurrent_kda/ut
FLA_NPU_RUN_LEGACY_TESTS=1 FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/recurrent_kda/routes/test_legacy_recurrent_kda.py
```

A2/A3/A5 通过 `FLA_NPU_SOC` 选择。Kimi H96/D128 smoke 仅检查 NPU launch、输出 shape 和抽样 finite，不跑 CPU golden。
完整 `T_total=12288` 长上下文规格保留为 stress case，不纳入默认通过矩阵。
测试结果只记录平台、case 总数、通过数和失败 case ID，不记录本地环境路径。
