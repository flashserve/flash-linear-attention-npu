# CausalConv1dBwd 测试归档

## 1. 唯一用例规格

`tests/op_cases/causal_conv1d_bwd.json` 统一保存 shape、dtype、layout、属性、可选输入、SOC、运行通路、随机种子、
参考实现、容差和预期返回码。新 case 必须先进入 JSON，再由执行端按 case ID 消费。

## 2. 归档内容

| 路径 | 内容 |
| --- | --- |
| `common/case_matrix.py` | 本算子 JSON 加载、tag/route 筛选和 case ID 环境变量 |
| `accuracy/test_causal_conv1d_bwd.py` | `fla_npu.ops.ascendc` 主精度、泛化、边界和回归入口 |
| `routes/test_aclnn_causal_conv1d_bwd.cpp` | aclnn 两段式接口签名、workspace/executor/stream 契约 |
| `routes/test_direct_causal_conv1d_bwd.cpp` | host tiling 结果驱动的 `<<<>>>` 参数和 launch 契约 |
| `ut/op_host/test_contract.py` | manifest、SOC、返回码、host 负向用例静态契约 |
| `ut/op_kernel/test_contract.py` | kernel 入口、tiling key 说明和 direct launch 静态契约 |
| `performance/profile.py` | 读取 performance tag 并通过 msopprof 运行设备侧 profiling |
| `st/test_example.py` | example tag 与仓内数值执行后端的 ST 入口 |

- legacy 通路：`torch.ops.npu.npu_causal_conv1d_bwd`，由主 route case 验证显式加载。

现有数值/reference 后端：`tests/operators/causal_conv1d_bwd/accuracy/backend.py`。该后端由 canonical 入口传入
`FLA_NPU_CASE_MANIFEST`、`FLA_NPU_CASE_IDS` 和 `FLA_NPU_OPERATOR`；关键 shape、dtype、属性组合不在
canonical 脚本中重复定义。

## 3. 历史资产迁移

算子源码目录以及 example/torch_custom 适配工程中原有的主线测试资产已迁入本目录，非标准目录已删除。case 数据只保留在唯一 manifest；脚本、reference、ATK executor、direct-launch harness、性能工具和 C++ UT 按职责归档，不再维护第二份 shape 表。

| 迁移集合 | 数量 | 唯一规格 |
| --- | ---: | --- |
| atk_regression | 150 | tests/op_cases/causal_conv1d_bwd.json |

使用 python3 -m tests.operators._shared.legacy_cases list --op causal_conv1d_bwd 可列出迁移 case。ATK 需要旧格式 JSON 时，用同一工具的 materialize 子命令生成临时文件。

## 4. 执行命令

```bash
pytest -q tests/operators/causal_conv1d_bwd/accuracy/test_causal_conv1d_bwd.py
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/causal_conv1d_bwd/accuracy/test_causal_conv1d_bwd.py
FLA_NPU_CASE_TAGS=generalization FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/causal_conv1d_bwd/accuracy/test_causal_conv1d_bwd.py
pytest -q tests/operators/causal_conv1d_bwd/ut
python tests/operators/causal_conv1d_bwd/performance/profile.py --dry-run
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/causal_conv1d_bwd/st/test_example.py
```

A2/A3/A5 通过 `FLA_NPU_SOC` 选择。精度逐项比较全部公开输出并检查 NaN/Inf；性能只使用 msopprof
设备侧结果，按 JSON 的 `expect.requirement` 对比 Triton 或当前主线基线。报告记录平台、case 总数、通过数和
失败 case ID，不记录本地环境路径。
