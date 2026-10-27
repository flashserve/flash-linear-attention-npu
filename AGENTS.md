# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。

## Ascend C 算子开发

- 本仓 `fla/ops/ascendc/**` 下的算子统一属于线性 Attention 算子域。接口、CPU 标杆、方案、
  kernel、host tiling 和性能优化直接使用
  [`CANNBot linear_attention 工作流`](docs/算子自动开发工作流.md)。
- 固定设置 `algorithm_family=linear_attention`、`workflow_id=catlass-linear-attention-v1`，直接作为
  family 分类结论。
- CANNBot 04 使用直调 host 和 kernel 完成逐 Stage、整 kernel 定向验证。
- CANNBot 05 继续使用直调工程完成全量精度、功能和性能验收；验收通过后 CANNBot workflow
  进入 `complete`。
- CANNBot workflow 完成后，按本仓规则接入 op_api/aclnn、Stable-ABI、Python 导出和 ATK 包装；
  适配层冒烟和单 case ATK 通过后，执行本仓 ATK full 验收。
- CANNBot 02 生成的 CPU 标杆直接作为 `tests/atk/<op>/reference.py` 交付。CANNBot 直调测试和
  本仓 ATK executor 导入同一文件，数学公式集中在该文件维护。
- CANNBot 可用且版本匹配是 Ascend C 算子研发的入口条件；缺少条件时记录并报告阻塞项。

## 任务路由

| 任务类型 | 必读内容 |
| --- | --- |
| Ascend C 接口、标杆、方案、kernel、host tiling 或性能 | [`docs/算子自动开发工作流.md`](docs/算子自动开发工作流.md) |
| op_api/aclnn、Stable-ABI、Python wrapper 或公共 runtime | [`docs/architecture/适配层接入指南.md`](docs/architecture/适配层接入指南.md)、[`docs/architecture/torch-npu-decoupled-architecture.md`](docs/architecture/torch-npu-decoupled-architecture.md) |
| ATK 资产、用例、executor 或验收 | [`tests/atk/README.md`](tests/atk/README.md) 和当前算子的 ATK README |
| wheel、OPP、构建或安装 | [`docs/开发者指南.md`](docs/开发者指南.md) 和相关构建脚本 |
| PR、分支、CODEOWNERS 或 CI | [`docs/repository-rules.md`](docs/repository-rules.md)、PR 模板和现有 workflow |
| Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README |

前一阶段的接口、标杆或设计发生变化时，从最早受影响的 CANNBot 阶段重新执行后续流程。
