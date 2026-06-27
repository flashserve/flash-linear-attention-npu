# Chunk KDA 算子与三方实现精度对比报告

## 1. 测试对象

本次验证覆盖以下新增接口：

- `torch.ops.npu.npu_chunk_kda_fwd`
- `torch.ops.npu.npu_chunk_kda_bwd`
- L0 `ChunkKdaFwd`
- L0 `ChunkKdaBwd`

对标对象为 fla-org KDA chunk 公式语义，以及仓内按该公式实现的 PyTorch reference：

- `tests/reference/chunk_kda_reference.py`
- `torch_custom/fla_npu/test/test_npu_chunk_kda.py` 中的 autograd golden

## 2. 覆盖范围

| Case | 覆盖点 | 结果 |
|---|---|---|
| AscendC custom 包构建 | `chunk_kda_fwd`、`chunk_kda_bwd` L0/L2 构建 | 通过 |
| torch_custom wheel 构建 | `npu_chunk_kda_fwd`、`npu_chunk_kda_bwd` ABI 注册 | 通过 |
| forward float32 | `chunk_size=64`、`initial_state`、`final_state`、全部中间量 | 通过 |
| forward varlen/GVA/V256 | `chunk_size=128`、`HV > H`、`cu_seqlens`、`V=256` | 通过 |
| forward bf16 | `chunk_size=32`、BF16 q/k/v/state、float32 gk/beta | 通过 |
| backward float32 | `dq/dk/dv/dbeta/dgk/dh0` vs autograd golden | 通过 |
| backward GVA/V256 | `chunk_size=128`、`HV > H`、`V=256` | 通过 |

## 3. 精度阈值

| 数据类型 | 阈值 |
|---|---|
| float32 | `rtol=3e-3, atol=3e-3` |
| bf16 | `rtol=2e-2, atol=2e-2` |

## 4. 测试结论

新增 KDA AscendC L0/L2 正反向算子已完成构建、安装和精度验证。测试覆盖了三方 KDA 关键语义：key-wise `gk` gate、GVA、varlen、`initial_state/output_final_state`、反向 `dht/dh0` 递推、`chunk_size=32/64/128` 入口，以及 `V=256` 典型模型场景。

所有已执行精度用例均通过，NPU 输出与 PyTorch reference/autograd golden 在设定阈值内一致。
