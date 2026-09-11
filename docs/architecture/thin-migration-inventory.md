# 全量算子薄层化：算子清单与批次表

> 基于 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 当前导出集合
> （#496 分支，未含 #390 的 causal_conv1d_update；后者合入后并入 conv1d 批次）。

## 迁移批次总览

| 批次 | 算子 | 说明 |
|---|---|---|
| A（已完成试点） | `npu_recurrent_gated_delta_rule` | thin 已启用、parity/多 stream 通过 |
| A | `npu_recurrent_kda` | recurrent 家族，in-place state |
| A | `npu_kda_gate_cumsum` | KDA 门控 cumsum（简单标量+数组） |
| B | `npu_causal_conv1d`（legacy）+ `causal_conv1d_update`(#390) | conv1d 家族；等 #390 合入统一 ABI |
| B | `npu_causal_conv1d_bwd` | 多输出 + char* layout，较复杂 |
| C | `npu_chunk_fwd_o`、`npu_chunk_gated_delta_rule_fwd_h`、`npu_chunk_fwd_h` | chunk GDN 主前向 |
| C | `npu_chunk_gated_delta_rule_fwd`、`npu_chunk_gated_delta_rule_fwd_prepare`、`_bwd_finalize`、`_bwd_dhu`、`npu_chunk_bwd_dv_local`、`npu_prepare_wy_repr_bwd*`、`npu_chunk_bwd_dqkwg` | chunk GDN 前后向细节，可选输出多 |
| C | `npu_chunk_local_cumsum`、`npu_chunk_scaled_dot_kkt`、`npu_recompute_w_u_fwd`、`npu_solve_tri` | chunk 工具算子；solve_tri 带 char* layout |
| D | `npu_chunk_kda_fwd`、`npu_chunk_kda_bwd`、`npu_chunk_kda_bwd_intra` | KDA 家族 |
| E | `npu_fast_gelu_custom`(+backward)、其余低频 | 收尾 |

## 分类口径

- **ND-only 可用**：输入都是普通 tensor/视图，无 5HD/NZ 风险 → 可进 thin；
- **参数形态复杂度**：
  - `简单`：全 tensor + 少量标量（int64/float）；
  - `中`：含 optional tensor / int-array / 输出 shape 推理；
  - `复杂`：多输出、char*、bwd、字符串 layout、多组 device+cpu metadata；
- **mutation**：`MUTATED_ARGUMENTS` 中列出的 in-place 参数需走 mutation 契约测试；
  `MUTATION_FLAGS` 声明"是否写回"的 flag（如
  `npu_recurrent_kda.inplace_final_state`），使热路径不必 `signature.bind`
  （910b：`npu_recurrent_kda` 公共调用 0.155 → 0.092 ms，见
  [cpp-thin-launcher-measurement-report.md §8](cpp-thin-launcher-measurement-report.md)）；
  设备侧契约用例：`tests/regression_mutation_contract.py`；
  纯 Python 用例：`torch_custom/fla_npu/test/test_ascendc_mutation_contract.py`。
- **离线门禁（不需要 NPU/torch，建议随 spec 改动一起跑）**：
  `torch_custom/fla_npu/tools/op_abi_validate.py`（spec 参数种类/顺序 vs
  aclnn 头）、`tools/op_spec_lint.py`（`alloc`/`when` 引用的符号必须在 spec
  参数或 helpers 里声明——拦住"'use_gate_in_kernel' was not declared"这类
  编译期才发现的问题）、`tools/op_policy_check.py`（spec 输出掩码/返回元组
  与权威 policy 模块在全部 flag 组合下等价）。
- **host 热度**：正式迁移顺序最终以服务 profile 决定；本表给出候选顺序。

## 详细清单

| 算子（python 名） | 批次 | 参数形态 | mutation | 备注 |
|---|---|---|---|---|
| npu_recurrent_gated_delta_rule | A | 简单-中（optional g/gk） | state | **已 thin** |
| npu_recurrent_kda | A | 中 | initial_state | recurrent 家族 |
| npu_kda_gate_cumsum | A | 简单 | - | KDA 门控 |
| npu_causal_conv1d（legacy） | B | 中（4 int-array） | conv_states | 等 #390 统一 |
| npu_causal_conv1d_update（#390 后） | B | 复杂（tensor+*_cpu 双通道/char*） | conv_state | 验证分支已实现原型 |
| npu_causal_conv1d_bwd | B | 复杂（char* layout、4 输出） | - | 需多输出 shape 规则 |
| npu_chunk_fwd_o | C | 中 | - | 主前向 |
| npu_chunk_gated_delta_rule_fwd_h | C | 中 | - | 主前向 |
| npu_chunk_fwd_h | C | 中（int-array cu_seqlens/chunk） | - | 纯 ctypes 入口 |
| npu_chunk_gated_delta_rule_fwd | C | 复杂（多可选输出） | - | Phase6 融合 |
| npu_chunk_gated_delta_rule_fwd_prepare | C | 复杂 | - | - |
| npu_chunk_gated_delta_rule_bwd_finalize | C | 复杂 | - | 多输出 |
| npu_chunk_gated_delta_rule_bwd_dhu | C | 复杂 | - | 多输出 |
| npu_chunk_bwd_dv_local | C | 复杂 | - | 多输出 |
| npu_prepare_wy_repr_bwd_full/bwd/da | C | 复杂 | - | 多输出 |
| npu_chunk_bwd_dqkwg | C | 复杂 | - | 多输出 |
| npu_chunk_local_cumsum | C | 简单 | - | - |
| npu_chunk_scaled_dot_kkt | C | 简单 | - | - |
| npu_recompute_w_u_fwd | C | 简单-中 | - | - |
| npu_solve_tri | C | 中（char* layout） | - | 需 char* 参数 kind |
| npu_chunk_kda_fwd | D | 中-复杂 | - | - |
| npu_chunk_kda_bwd | D | 复杂 | - | 多输出 |
| npu_chunk_kda_bwd_intra | D | 复杂 | - | 多输出 |
| npu_fast_gelu_custom / _backward | E | 简单 | - | 低频 |

> 精确迁移顺序最终以 host profile（调用次数 × 单次 host 时间）为准；
> 上表 A-E 为工程候选顺序，非最终发布顺序。

## 验证状态（持续更新）

| 算子 | JSON-only | 头校验 | parity | host（ctypes → thin public） |
|---|---|---|---|---|
| npu_recurrent_gated_delta_rule | ✅ | ✅ | 0.0 | ~0.5 → ~0.10 ms |
| npu_kda_gate_cumsum | ✅ | ✅ | 0.0 | 0.33 → 0.049 ms |
| npu_chunk_local_cumsum | ✅ | ✅ | 0.0（910b w16 + 950 env950c/run merge） | 0.37 → 0.089 ms |
| npu_chunk_scaled_dot_kkt | ✅ | ✅ | 0.0（k fp16/bf16 × g/beta fp32；OPP 仅编译该 dtype 域，fp16 g/beta 为 161002） | 0.44 → 0.067 ms |
| npu_recompute_w_u_fwd | ✅ | ✅ | 0.0 | 0.58 → 0.115 ms |
| npu_prepare_wy_repr_bwd_full | ✅ | ✅ | 0.0 | 0.77 → 0.135 ms |
| npu_prepare_wy_repr_bwd | ✅ | ✅ | 0.0（KH=4/VH=8、bf16+fp32） | 0.74 → 0.129 ms |
| npu_chunk_bwd_dv_local | ✅ | ✅ | 0.0 | 0.48 → 0.105 ms |
| npu_prepare_wy_repr_bwd_da | ✅ | ✅ | 0.0 | 0.69 → 0.135 ms |
| npu_chunk_bwd_dqkwg | ✅ | ✅ | 0.0 | 0.73 → 0.097 ms |
| npu_fast_gelu_custom | ✅（base aclnn） | - | 0.0 | 0.26 → 0.041 ms |
| npu_fast_gelu_custom_backward | ✅（base aclnn） | - | 0.0 | 0.30 → 0.042 ms |
| npu_chunk_gated_delta_rule_fwd_h | ✅（v3 when/alloc） | ✅ | 0.0（dense/final/varlen） | 0.57 → 0.131 ms |
| npu_chunk_fwd_h | ✅（v3） | ✅ | 0.0（dense/final/varlen） | 0.63 → 0.127 ms |
| npu_chunk_fwd_o | ✅（v3） | ✅ | 0.0（仅 BNSD 合法域） | 0.56 → 0.097 ms |
| npu_chunk_gated_delta_rule_bwd_dhu | ✅（v3） | ✅ | 0.0（canonical ≥2 序列；单序列 dense 两条路径同 NaN，内核边界） | 0.71 → 0.139 ms |
| npu_recurrent_kda | ✅ | ✅ | 0.0（BSND B2/T2/H2/HV4 dense、state_v_first、inplace + final_state；Ascend950PR） | 0.085 → 0.011 ms（950） |
| npu_chunk_gated_delta_rule_fwd | ✅（cpp_only 标量 + return_code + layout/varlen helpers；需上游 #495 的 op_api/ctypes 修正） | ✅ | **全域名**：dense + varlen（physical B=1、canonical chunk_indices）；layout BNSD/BSND/NTD/TND；A2 legacy 路径与 A5 新路径（`use_exp2`/`use_qk_l2norm`/`state_v_first`/`return_intermediate_states` 的 `h`）；GVA、chunk 64/128、`initial_state` fp32/bf16、`output_final_state` 均覆盖。实测：A2（910b）21 场景全绿含 varlen；A5（950）BSND+exp2+l2norm、+state_v_first、+return_h、TND varlen、legacy BNSD 全部 parity 0.0，且 cp312/torch2.9 与 cp310/torch2.7.1(fzy) 两套环境结果一致。A5 专属输出（q_hat/k_hat/rstd/beta_eff）与 ctypes 一样传 null；BSND 不带 exp2、V=256 在当前 build 双方同样报错（169104/161002） | 待补 |
| npu_chunk_gated_delta_rule_fwd_prepare | ✅ | ✅ | 0.0（9 输出；Ascend950PR）＋放宽域：`use_beta_sigmoid=False`（返回 `beta.to(fp32)`）、`output_a=False`、varlen 均已 0.0；仅 `a_log`/`dt_bias` 非空时 thin 报 161002（1-D 描述符待查）→ 保持回退 ctypes | 0.144 → 0.030 ms（950） |
| npu_chunk_gated_delta_rule_bwd_finalize | ✅ | ✅ | 0.0（5 输出；Ascend950PR，g/beta fp32、G=2 域）＋放宽域：`use_qk_l2_norm_in_kernel=False`/`use_beta_sigmoid_in_kernel=False`（不传 q_rstd/k_rstd/beta_raw）、`state_v_first=True` 均已 0.0 | 0.167 → 0.023 ms（950） |
| npu_causal_conv1d_bwd | ✅ | ✅（按文档签名） | 0.0（BNSD 域）；BSH/TND 两路径同 NaN（该构建 kernel 边界待查） | 0.58 → 0.084 ms |
| npu_chunk_kda_fwd | ✅（dense BSND 合法域；其它布局/flag 委托 ctypes） | ✅ | 0.0（10 输出 + None 语义） | 1.04 → 0.114 ms |
| npu_chunk_kda_bwd_intra | ✅（BNSD dense 单发射合法域；BSND 分段路径委托 ctypes） | ✅ | 0.0（4 输出） | 0.73 → 0.091 ms |
| npu_chunk_kda_bwd | ✅（dense BNSD 简单域：偶数头、T%64=0、gate off；tail/奇头/varlen 回退委托 ctypes） | ✅ | 0.0（dq/dk/dv/db/dg + 3×None） | 0.88 → 0.111 ms |
| npu_solve_tri | ✅（bsnd/bnsd dense + tnd，enabled） | ✅ | 0.0（fp16/bf16 × BT 16/32/64/128；**tnd varlen 已原生 thin 0.0**，用 `block_t = 1<<17/chunk_size` 生成 canonical chunk_indices）；**ntd 为上游 kernel 问题**（多次调用结果非确定：ctypes 返回全 0、thin 不等于 tnd 的转置）→ 保持回退 ctypes | 910b：0.372 → 0.098 ms；950：0.033 → 0.009 ms（dense bsnd fp16 BT64） |

## 下一步

已闭环 24/26 个库存算子（见验证状态表；count 不含 conv1d legacy 与
运行域未定的 composite fwd）。剩余算子/子域及其状态：

1. `npu_causal_conv1d`（legacy）：保持 ctypes 到上游 #390 合入后统一 ABI。
2. `causal_conv1d_update`（#390）：适配已在验证分支 PR #512 实现并跑通原型；
   #390 合入后并入本分支并做 910b/950 全量回归。#390 当前（2026-09-09）仍 open，
   最新 head 与本分支差异仅 examples/flash_gated_delta_rule.py，ABI 未变。
3. `npu_chunk_gated_delta_rule_fwd`（composite）：**已全量闭环**。根因是上游
   #495 修了该算子的 op_api/ctypes 参数透传；合并 main 后把 spec 从“dense BNSD”
   扩到全域名——引入 layout-aware helpers、varlen 的 seq/chunk 数推导、
   A5 路径的 `h` 输出与 `return_code`，并修了两处：`use_gate_in_kernel`/
   `use_beta_sigmoid_in_kernel` 必须是 `cpp_only`（否则 `when` 引用未声明符号），
   A5 专属输出必须像 ctypes 一样传 null（否则 A5 报 161002）。实测 A2 21 场景
   全绿，A5（950）A5 场景全绿，且在 torch 2.9/cp312 与 torch 2.7.1/cp310(fzy)
   两套环境结果一致。注意：T 非 chunk 整数倍时 `A` 的尾块 padding 行两侧都是
   未初始化内存（valid 区域仍 0.0）；只给 `cu_seqlens` 时 thin 会自动派生
   canonical `chunk_indices`（ctypes 要求成对提供，属 thin 的超集）。
4. `npu_solve_tri`：bsnd/bnsd dense 与 tnd（dense/varlen）已原生 thin 并 0.0；
   **ntd 是上游 kernel 问题**（结果非确定：ctypes 全 0、thin 不等于 tnd 转置），
   保持 ctypes 回退并建议上报。
5. 收尾：regression_thin_ops 21 场景已在 910b（本分支 wheel）全绿；
   regression_950_ops 4 场景（fwd_prepare / bwd_finalize / recurrent_kda /
   chunk_gated_delta_rule_fwd A5 域）已在 Ascend950PR（fzy py3.10 + torch 2.7.1）
   全绿，安装态 smoke 3/3 通过。

## 回退域收敛计划（目标：thin 覆盖 ctypes 的全部可用域）

当前仍会回退 ctypes 的点（已在代码里逐个核对）：

| 算子 | 回退条件 | 性质 | 结论/计划 |
| --- | --- | --- | --- |
| `npu_solve_tri` | `layout == "ntd"` | 上游 kernel 问题（非确定） | 保持回退；建议上游修 ntd |
| `npu_chunk_gated_delta_rule_fwd_prepare` | `a_log`/`dt_bias` 非空 | thin 1-D 描述符 161002 | 保持回退；待查 1-D/format 描述符 |
| `npu_chunk_kda_fwd` | 非 BSND、varlen、`output_final_state`、`return_intermediate_states` | spec 未展开（与 composite 同类，可做） | 下一步：layout helpers + 条件输出 + `return_code` |
| `npu_chunk_kda_bwd_intra` | BSND（分段）/varlen | 需把 ctypes wrapper 的**多发射分段**语义搬进 C++ | 需 codegen 支持子发射循环 |
| `npu_chunk_kda_bwd` | varlen/尾块/奇头等 | 同上（多发射 + 补齐） | 同上 |
| `npu_causal_conv1d`（legacy） | 全部（无 thin 入口） | 等上游 #390 统一 ABI | #390 合入后加 spec |
| `causal_conv1d_update` | 不在 #496 | 在 #512 验证分支 | #390 合入后并入 |

其余“回退”只是域判定（ctypes 本身也会拒绝该组合，例如 chunk 64 限制、
A2 上的 A5-only 组合），不属于覆盖缺口。
   历史记录：regression_thin_ops 20 场景（37 组）已在 910b（w16 wheel）与
   Ascend950PR（950d wheel）安装态全量执行并全绿；950-only 3 场景
   （fwd_prepare/bwd_finalize/recurrent_kda）与安装态 smoke 3/3 亦全绿。
   发布矩阵与实测记录见 thin-launcher-release-matrix.md。
