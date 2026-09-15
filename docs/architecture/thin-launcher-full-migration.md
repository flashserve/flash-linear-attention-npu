# 全量算子薄层化：实现路线（Phase 0 起）

> 分支：`feat/fla-npu-thin-launcher`（#496）
> 决策（2026-09-09）：
> 1. ctypes 源码保留，默认不再使用（作为隐藏 fallback 与测试对照）；
> 2. 本阶段 ND-only，非 ND 输入回退 ctypes；
> 3. ABI 采用 spec + codegen（编译期展开，运行时性能≈0）；
> 4. 发布矩阵：主流 Python 版本 × linux x86_64/aarch64。

## 1. 目标

让 `fla_npu.ops.ascendc.*` 全部算子默认走 thin C++ 执行路径，同时保留：

- ctypes 源码与 `FLA_NPU_THIN_LAUNCHER=0` 回退（排障/对照）；
- 与 OPP aclnn 头"单一真相"对齐的开发方式（参照 ctypes 跟头改 wrapper 的体验）；
- 每算子 bit 级 parity 与 host 性能护栏。

## 2. 架构

```text
aclnn_*.h (安装 OPP, ABI 唯一真相)
   │  op_abi_validate.py（CI，离线）
   ▼
op_specs/<op>.json（参数顺序/类型/输出/mutated）
   │  codegen（开发期，生成类型化 C++，入库）
   ▼
csrc_thin 每算子类型化适配 + 通用两段式 launch
```

- Spec 与 ctypes wrapper 双通道一致：生成器可从 spec 同时产出 C++ 适配骨架；
- CI 校验：`op_abi_validate.py --header <aclnn_*.h> --spec <op>.json` 对比参数 kind 顺序；
- 校验只发生在构建/CI，不进入运行时，性能无影响。

## 3. Spec 字段（v0）

```json
{
  "aclnn_name": "aclnnRecurrentGatedDeltaRule",
  "python_name": "npu_recurrent_gated_delta_rule",
  "nd_only": true,
  "mutated": ["state"],
  "args": [
    {"name": "query", "kind": "tensor"},
    {"name": "key", "kind": "tensor"},
    {"name": "value", "kind": "tensor"},
    {"name": "beta", "kind": "tensor"},
    {"name": "state", "kind": "tensor"},
    {"name": "actual_seq_lengths", "kind": "tensor"},
    {"name": "ssm_state_indices", "kind": "tensor"},
    {"name": "g", "kind": "optional_tensor"},
    {"name": "gk", "kind": "optional_tensor"},
    {"name": "num_accepted_tokens", "kind": "optional_tensor"},
    {"name": "scale", "kind": "float"},
    {"name": "out", "kind": "tensor"}
  ]
}
```

kind 取值（v0）：`tensor`、`optional_tensor`、`int_array`、`cpu_int_array`、
`char_ptr`、`int64`、`float`、`double`、`bool`、`out_tensor`。
（`workspaceSize/executor` 由通用 executor 统一追加，不写入 spec。）

## 4. 分阶段

- Phase 0（当前）：spec 格式 + 校验工具 + recurrent 试点（spec/校验跑通，
  并产出 codegen 骨架约定）；
- Phase 1：按 profile 分批（recurrent 家族 → conv1d(#390 合入后) → chunk 系列 →
  kda 等中频 → 低频收尾），每批走 §5 checklist；
- Phase 2：全量白名单 + 完整测试 + vLLM mixed 基准；
- Phase 3：发布矩阵（多 python × x86_64/aarch64）与文档。

## 5. 每算子 checklist

1. 读 OPP `aclnn_*.h` + ctypes wrapper 的参数顺序/默认值/输出 shape；
2. 写 spec JSON；
3. `op_abi_validate.py` 校验通过；
4. codegen（或手写对齐）产出 C++ 适配；`_thin.py` 同签名 wrapper；
5. 白名单 + mutation 契约确认；
6. 测试：parity（连续/非连续/可选参数/边界）、mutation/autograd、
   非法输入报错记录、多 stream + 多线程（每线程独立 stream，vLLM worker
   形态）、确定性；
7. host benchmark：ctypes vs thin public vs direct（P50，200 次），记录入表；
8. ND-only 约束：非 ND 输入自动回退 ctypes。

## 6. 关键决策记录

- 非 ND 输入：thin 判定 `acl_format` 非 base 时回退 ctypes；
- ctypes：源码保留，默认不经由 dispatch（`FLA_NPU_THIN_LAUNCHER=0` 仍可强制）；
- wheel：薄层默认编译；发布按主流 python × linux x86_64/aarch64 出矩阵；
- conv1d：等 flash-linear-attention-npu #390 合入后统一其 ABI 再并入正式薄层。
