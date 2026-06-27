# Chunk KDA 精度验证报告

## 验证范围

- `npu_chunk_kda_fwd` 正向：BSND、TND、GVA、varlen、`chunk_size=32/64/128`、`V=256`。
- `npu_chunk_kda_bwd` 反向：BSND、TND、GVA、`chunk_size=64/128`、`V=256`。
- dtype：float32、float16、bfloat16；`gk/beta` 支持 float32、bfloat16。
- `npu_kda_gate_cumsum`：默认 gate 增量累计，以及 `use_gate_in_kernel=True && safe_gate=True` raw gate 路径。
- 对标：同仓 PyTorch reference/autograd，与 fla-org KDA 公式语义一致。

## 测试用例

| 用例 | 覆盖点 | 结果 |
| --- | --- | --- |
| `test_chunk_kda_fwd_matches_reference` | BSND fp32 正向、中间量、final_state | 通过 |
| `test_chunk_kda_fwd_chunk128_v256_gva_varlen` | BSND 正向、GVA、varlen、`chunk_size=128`、`V=256` | 通过 |
| `test_chunk_kda_fwd_bf16_chunk32_matches_reference` | BSND bf16 正向、`chunk_size=32` | 通过 |
| `test_chunk_kda_fwd_bf16_gate_matches_reference` | `gk/beta` 为 bfloat16，L2 fp32 gate cast，正向对齐 reference | 通过 |
| `test_chunk_kda_fwd_fp16_matches_reference` | BSND fp16 正向、中间量 | 通过 |
| `test_chunk_kda_fwd_tnd_matches_reference` | TND 正向、外部 TND/内部 NTD layout 转换 | 通过 |
| `test_kda_gate_cumsum_default_and_fwd_integration` | `npu_kda_gate_cumsum` 默认累计，并接入 KDA forward | 通过 |
| `test_kda_gate_cumsum_safe_gate_matches_reference` | `use_gate_in_kernel=True && safe_gate=True` raw gate 路径 | 通过 |
| `test_chunk_kda_bwd_matches_autograd` | BSND fp32 反向、`dq/dk/dv/dbeta/dgk/dh0` | 通过 |
| `test_chunk_kda_bwd_fp16_matches_autograd` | BSND fp16 反向 | 通过 |
| `test_chunk_kda_bwd_bf16_gate_matches_autograd` | `gk/beta` 为 bfloat16，反向对齐 autograd | 通过 |
| `test_chunk_kda_bwd_chunk128_v256_gva_matches_autograd` | BSND 反向、GVA、`chunk_size=128`、`V=256` | 通过 |
| `test_chunk_kda_bwd_tnd_matches_autograd` | TND 反向、外部 TND/内部 NTD layout 转换 | 通过 |
| 模型尺寸正向 smoke | `B=1,T=288,H=32,HV=64,K=128,V=256,chunk_size=128,fp16 + BF16 gate`，检查输出 shape、dtype 和 finite | 通过 |

## 结论

当前 KDA 正反向算子在上述功能和精度覆盖范围内与 PyTorch reference/autograd 对齐。BSND 对外 ABI 与 fla-org KDA 默认形状一致；TND 入口保持外部 token-first 布局，内部通过 L2 编排转换为 NTD/BNSD 连续布局以改善 kernel 访存连续性。BF16 gate 输入、safe raw gate 预处理和 backward fp32 workspace 预计算优化均已通过同一套回归。
