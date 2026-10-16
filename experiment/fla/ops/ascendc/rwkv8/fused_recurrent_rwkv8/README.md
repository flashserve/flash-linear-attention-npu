# FusedRecurrentRwkv8

## 功能

`FusedRecurrentRwkv8` 是 RWKV-v8（WKV7，z/b 参数化口径）的逐 token 递推前向算子，
面向推理路径（prefill/decode 共用一份 kernel）。语义对齐
BlinkDL/RWKV-LM 提交 `9521024`（`RWKV-v8/cuda/wkv7_cuda.cu` `forward_kernel` lines 10-52），
精度 peer 为 fla-org/flash-linear-attention 提交 `a4a2624b`
（`fla/ops/generalized_delta_rule/dplr` fused_recurrent 前向）。

递推公式（per head，state 数学朝向：行 = v/q 侧，列 = k/z 侧；接口 `initial_state` / s 快照以内核账本 `S^T` 朝向 `(B,H,K,V)` 传入/传出，零转置）：

```text
sa    = state @ z_t                          # 移除读出（z = -kk）
state = state * decay_t[None, :]             # 逐通道衰减，decay = exp(-exp(w))
      + sa[:, None] * b_t[None, :]           # 移除（rank-1，b = kk * a_inctx）
      + v_t[:, None] * k_t[None, :]          # delta-rule 写入
o_t   = state @ (q_t * scale)                # 读出（scale 作用在 q 上）
```

与官方 CUDA kernel 的差异：无 `T % 16` 约束（官方 kernel 要求 T 为 16 的倍数）。

> io 布局为 **BHTC = (B,H,T,C)**（H 在 T 前，2026-08-17 定档）：每个 (b,h) 的
> 数据在 GM 上连续，token 步长为 C。与 fla 的 (B,T,H,C) 口径不同，对接时需
> `transpose(1,2)`。

## 输入

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | `(B,H,T,N)`，FP32 | Query（读出向量） |
| `w` | 必选 | 同 q | log 域衰减参数，`decay = exp(-exp(w))` |
| `k` | 必选 | 同 q | Key（写入向量） |
| `v` | 必选 | 同 q | Value |
| `z` | 必选 | 同 q | 移除读出向量（`= -kk`，kk 为 L2 归一化） |
| `b` | 必选 | 同 q | 移除强度向量（`= kk * a_inctx`） |
| `initial_state` | 可选 | `(B,H,K,V)`，FP32 | 接口朝向（= 内核账本 `S^T`，与 s 快照一致），缺省为零态 |

## 输出

```text
(o, s, sa)
```

- `o`：`(B,H,T,N)`，FP32。
- `s`：`(B,H,T//chunk_len,N,N)`，FP32；chunk state 快照，官方 CUDA 转置布局，`outputChunkState=true` 时必填。
- `sa`：`(B,H,T,N)`，FP32；每 token 移除读出（更新前的 `state@z`），`outputSa=true` 时必填。

## 属性

| 名称 | 默认值 | 支持范围 |
| --- | --- | --- |
| `scale` | `1.0` | FP32 标量，作用于 q 读出 |
| `reverse` | `false` | T 维倒序递推（对齐 fla reverse） |
| `output_chunk_state` | `false` | 产出 chunk 快照 `s`（训练预埋） |
| `output_sa` | `false` | 产出每 token 的 `sa`（训练预埋） |
| `chunk_len` | `16` | `s` 快照间隔（INT64，>= 1）；默认 16 对齐官方 CUDA backward 的 chunk 重建粒度，非 16 值与官方 backward 不兼容 |

`s`/`sa` 为训练预埋输出（对齐官方 CUDA 三输出 y_/s_/sa_），由 OpDef 属性门控；
缺省关闭时给零尺寸 shape、kernel 跳过写出。

## 支持范围

- A2 (`ascend910b`)。
- FP32 only（输入与 state 均 FP32；bf16 路径待补）。
- `K <= 128`、`V <= 128`（均不切分，`K×V` 的 state 须整体放入单核 UB）；
  `B/T/H` 无对齐约束（含 `T=1` decode）。
- 仅前向；反向归 chunk 算子另行立项。

## 验证

用例规格是 `tests/cases.py`（正例 8 条 + 负例 3 条，由旧中央 manifest 迁移而来）；
CPU golden 为 `tests/pta/golden.py`（纯 PyTorch，不依赖 torch_npu/Triton，
golden 已与官方 GPU CUDA fixture 同机对拍：o/state rel-RMSE ~1e-7）。
PTA 精度脚本为 `tests/pta/test_accuracy.py`（NPU 对拍 CPU golden，11/11 PASS）；
ATK 单算子工程位于仓根 `tests/atk/fused_recurrent_rwkv8/`（accuracy/performance/
determinism/mssanitizer/gen_cases 全动作走 `tests/atk/run_test_cpu.sh`）。

交付形态为 OPP（aclnn）算子。e2e 验证见 `examples/test_aclnn_fused_recurrent_rwkv8.cpp`
（自包含：内置输入 + C++ CPU golden + rel-RMSE ≤ 0.002 对拍，4 组 case）：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=fused_recurrent_rwkv8 -j8
./build_out/fla-npu-*.run --install
bash build.sh --run_example fused_recurrent_rwkv8 eager cust --vendor_name=fla_npu --soc=ascend910b
```

aclnn 接口文档见 `docs/aclnnFusedRecurrentRwkv8.md`。

## 实现要点

- grid = `B*H`，每个核持一份完整的 `(K,V)` state 账本，逐 token 串行递推。
  K、V 均不切分（sa/o 对 K 求和要求每核拿全 K；V≤128 后 `K×V` 整体可入单核 UB）。
- state 以转置 `S^T` 驻留 UB（K=64、V=64 时 16KB），行 j 为 k/z 通道 j，
  递推全程化为 scalar×vector 的 Muls/Axpy，无跨 lane 归约。
- 初态 `(B,H,K,V)` 即 `S^T` 原样，按 K 行逐行载入（行距 V，逐行而非整块是为避开
  DataCopyPad blockLen 的 uint16 上限），零转置；s 快照写出同理逐行。
