#!/usr/bin/env python3
"""Generate the standardized per-operator documentation and test entry tree.

The operator-specific facts below are deliberately explicit.  The generator
only keeps the repeated document structure synchronized; it does not infer
public constraints from kernel names or silently invent support ranges.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[1]

FAMILY_INFO = {
    "gdn": {
        "name": "Gated Delta Network (GDN)",
    },
    "kda": {
        "name": "Kimi Delta Attention (KDA)",
    },
}


def rows(*items):
    return list(items)


def mode_features(spec):
    return [item.strip() for item in re.split(r"[、，；]|\s+与\s+", spec["modes"]) if item.strip()]


OPS = {
    "chunk_bwd_dqkwg": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg",
        "title": "ChunkBwdDqkwg",
        "family": "gdn",
        "purpose": "Gated Delta Rule 分块反向链路的主梯度算子。它消费前向激活、chunk 状态和上游梯度，计算 `dQ`、`dK`、`dW` 与 `dG`，并支持 `H_v/H_k` 分组映射。",
        "math": dedent(r"""
            对 value head `h_v`，先映射 `h_k=floor(h_v/(H_v/H_k))`。算子按链式法则把
            `dO`、`dH` 和 `dV` 对 chunk 内 score、状态项及门控项的贡献合并：

            ```text
            dQ[h_k] += dS[h_v] @ K[h_k]
            dK[h_k] += dS[h_v]^T @ Q[h_k] + dW[h_v] * beta/gate terms
            dW[h_v]  = state/output branches reduced on V
            dG[h_v]  = reverse cumulative reduction of gate-dependent terms
            ```

            `dQ/dK` 在同一 key head 对应的多个 value head 上归约；尾块仅对有效 token 求值。
            `g` 是沿 `T` 的 chunk-local 累积 gate，要求调用者提供与前向完全一致的值。
        """),
        "inputs": rows(
            ("q", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Query"),
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key，与 q 同形"),
            ("v", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "chunk-local 累积 gate"),
            ("h", "必选", "[B,H_v,N_c,K,V]", "FP16/BF16", "ND", "前向保存的 chunk 状态"),
            ("dox", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "输出梯度"),
            ("dh", "必选", "[B,H_v,N_c,K,V]", "FP16/BF16", "ND", "chunk 状态梯度"),
            ("dv", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value 分支梯度"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平的 (seq_id,chunk_id)"),
            ("w", "预留", "-", "与 q 一致", "ND", "当前必须为 None"),
            ("g_gamma", "预留", "-", "与 g 一致", "ND", "当前必须为 None"),
        ),
        "outputs": rows(
            ("dq", "[B,H_k,T,K]", "与 q 一致", "Query 梯度；多 value head 贡献已归约"),
            ("dk", "[B,H_k,T,K]", "与 k 一致", "Key 梯度；多 value head 贡献已归约"),
            ("dw", "[B,H_v,T,K]", "与 q 一致", "WY 中间量 W 的梯度"),
            ("dg", "[B,H_v,T]", "与 g 一致", "累积 gate 的梯度"),
        ),
        "attrs": rows(
            ("scale", "float", "None", "Python 的 None 按 1.0 传入；注意力缩放需显式传 1/sqrt(K)"),
            ("chunk_size", "int", "64", "chunk 长度"),
            ("use_exp2", "bool", "false", "预留，当前仅 false"),
            ("transpose_state_layout", "bool", "false", "预留，当前仅 false"),
        ),
        "layouts": "BNSD；状态张量为 `[B,H_v,N_c,K,V]`",
        "dtype": "主张量 FP16/BF16；gate 可为 FP32；输出跟随对应输入",
        "modes": "定长与变长序列；变长序列的两个索引必须同时提供",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256。",
            "`chunk_size` 仅支持 64/128，尾块按有效长度处理。",
            "必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。",
            "`w`、`g_gamma` 当前为预留输入，必须传 `None`。",
            "`use_exp2` 与 `transpose_state_layout` 当前必须为 `false`。",
        ],
        "task": "定长按 `B*N_c` 分配 chunk，变长序列按 `chunk_indices` 分配；每个 chunk 内再按 value head 划分 AIC/AIV 工作。",
        "tiling_key": "仅保留 key=1 选择当前 CV 深融合实现，不编码 runtime shape、layout 或普通属性。该单值 key 是历史二进制入口约定，不带组合增长；后续可在 ABI 允许时移除。",
        "flow": "Part A 生成 `dW` 和 gate 末端项；Part B 计算 score 梯度；Part C/D 通过 Cube 矩阵乘生成 `dQ/dK`，AIV 完成逐元素门控、尾块 mask 与 `dG` 归约。",
        "memory": "workspace 按 core/head 建立 `BT*K` 与 `BT*BT` 环形槽，`wsDwOffset/wsMm5Offset/wsMm6Offset/wsMul1Offset` 分阶段复用；L1/L0 承载矩阵乘 tile，UB 承载 gate 与归约临时量。",
        "sync": "AIC 写 workspace 后用跨核阶段 flag 通知 AIV；AIV 消费后通过反向 free/ready 协议允许槽位复用。核内 MTE/Cube/Vector 事件成对使用，构建固定 `--cce-auto-sync=off`。",
        "python_sig": "chunk_bwd_dqkwg(q, k, v, g, h, dox, dh, dv, chunk_size, *, cu_seqlens=None, chunk_indices=None, w=None, g_gamma=None, scale=None, use_exp2=None, transpose_state_layout=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_bwd_dqkwg

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
            N_c = (T + chunk_size - 1) // chunk_size
            q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
            dox, dv = torch.randn_like(v), torch.randn_like(v)
            dh = torch.randn_like(h)
            dq, dk, dw, dg = chunk_bwd_dqkwg(
                q, k, v, g, h, dox, dh, dv, chunk_size,
                scale=K ** -0.5, w=None, g_gamma=None)
            torch.npu.synchronize()
            assert dq.shape == q.shape and dk.shape == k.shape
        """),
        "aclnn_call": "q, k, v, g, h, dox, dh, dv, nullptr, nullptr, nullptr, nullptr, scale, chunkSize, false, false, dq, dk, dw, dg, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_bwd_dqkwg/accuracy/backend.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("必选 tensor 为空，或 rank/shape/GVA 不符合约束", "ACLNN_ERR_PARAM_INVALID"),
            ("w/g_gamma 非空，或变长序列元数据只提供一个", "ACLNN_ERR_PARAM_INVALID"),
            ("use_exp2/transpose_state_layout 为 true", "Python: RuntimeError；aclnn: ACLNN_ERR_PARAM_INVALID"),
            ("执行器或 kernel launch 失败", "ACLNN_ERR_INNER"),
        ),
    },
    "chunk_bwd_dv_local": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local",
        "title": "ChunkBwdDvLocal",
        "family": "gdn",
        "purpose": "Gated Delta Rule 反向过程的 Value 本地梯度算子。它在每个 chunk 内根据 `Q/K` score、门控差和 `dO` 生成 `dV_local`，不承担跨 chunk 状态梯度。",
        "math": dedent(r"""
            对 `h_v` 映射 `h_k=floor(h_v/(H_v/H_k))`，在每个 chunk 内：

            ```text
            Ws       = K[h_k] @ Q[h_k]^T * scale
            Ws_gated = triu(Ws * exp(g_col - g_row), diagonal=0)
            dV_local = Ws_gated @ dO[h_v]
            ```

            上三角包含对角线；partial chunk 的无效行列在乘法前清零。
        """),
        "inputs": rows(
            ("q", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Query"),
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key，与 q 同形"),
            ("d_o", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "输出梯度"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "chunk-local 累积 gate"),
            ("g_gamma", "预留", "-", "FP32", "ND", "当前必须为 None"),
            ("A", "预留", "-", "FP16/BF16", "ND", "当前必须为 None"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平的 (seq_id,chunk_id)"),
        ),
        "outputs": rows(("d_v", "[B,H_v,T,V]", "与 d_o 一致", "Value 的 chunk-local 梯度")),
        "attrs": rows(
            ("scale", "double", "无", "通常为 1/sqrt(K)"),
            ("chunk_size", "int", "无", "chunk 长度"),
        ),
        "layouts": "BNSD；变长序列在 T 维拼接",
        "dtype": "q/k/d_o 为 FP16/BF16，g 可为 FP16/BF16/FP32",
        "modes": "定长与变长序列；变长序列的两个索引必须同时提供",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256。",
            "`chunk_size` 仅支持 64/128。",
            "必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。",
            "`g_gamma` 和 `A` 尚未实现，必须传 `None`。",
        ],
        "task": "定长以 `B*ceil(T/chunk_size)` 为 chunk 列表，变长序列直接消费规范化 `chunk_indices`；AIC 以 Q/K head 生成共享 score，AIV/AIC 以 value head 消费。",
        "tiling_key": "使用模板化 tiling：`strategy`(定长/变长序列)、`D_T_Q`、`D_T_G` 和 `V` 是有限编译期维度。key 只由模板实例生成，不承载 B/H/T 等普通 runtime shape；组合上限由 2 种策略、受支持 dtype 和 2 个 V 实例共同限定。",
        "flow": "Phase 1 AIC 计算 `K@Q^T`；Phase 1.5 AIV 扩展到 value head并应用 exp/mask；Phase 2 AIC 计算 gated score 与 dO 的矩阵乘并写 dV。",
        "memory": "user workspace 是 AIC/AIV 交接的 `chunk_size*chunk_size` score 环形槽；L1/L0A/L0B/L0C 放置两个矩阵乘 tile，UB 放置 gate、mask 和类型转换临时量。",
        "sync": "score 槽采用 AIC-ready/AIV-done/second-AIC-ready 的阶段协议，生产者复用前等待消费者释放；核内事件覆盖 MTE2、Vector、MTE3 与 Cube/Fixpipe，构建固定 `--cce-auto-sync=off`。",
        "python_sig": "chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_bwd_dv_local

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
            q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            d_v = chunk_bwd_dv_local(q, k, d_o, g, K ** -0.5, chunk_size)
            torch.npu.synchronize()
            assert d_v.shape == d_o.shape
        """),
        "aclnn_call": "q, k, dO, g, nullptr, nullptr, nullptr, nullptr, scale, chunkSize, out, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_bwd_dv_local/accuracy/backend.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("必选 tensor 为空", "ACLNN_ERR_PARAM_INVALID"),
            ("g_gamma/A 非空，或变长序列元数据只提供一个", "ACLNN_ERR_PARAM_INVALID"),
            ("shape/dtype/GVA/chunk_size 不受模板支持", "tiling 失败；aclnn 执行返回 ACLNN_ERR_INNER"),
            ("Python g_gamma/A 非空", "RuntimeError"),
        ),
    },
    "chunk_gated_delta_rule_bwd_dhu": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu",
        "title": "ChunkGatedDeltaRuleBwdDhu",
        "family": "gdn",
        "purpose": "沿 chunk 反向计算隐藏状态梯度，并把状态分支对 Value 的贡献累加到 `dV2`。当前交付 kernel 只消费标量 gate `g`；逐 K gate、初末状态和 `dh0` 输出保留在 ABI 中但尚未实现。",
        "math": dedent(r"""
            对当前 chunk，合并输出分支和状态分支的梯度贡献：

            ```text
            dV2_i = dV_i + state_value_contribution(dH_i, K_i, g_i)
            dH_i  = Q_i^T @ dO_i + W_i^T @ dV2_i
            ```

            GVA 下一个 key head 被 `H_v/H_k` 个 value head 复用；每个 value head 的状态和输出独立。
        """),
        "inputs": rows(
            ("q", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Query"),
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"),
            ("w", "必选", "[B,H_v,T,K]", "FP16/BF16", "BNSD", "WY 的 W"),
            ("d_o", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "输出梯度"),
            ("dv", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "已有 Value 梯度"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "标量 gate"),
            ("gK", "预留", "[B,H_v,T,K]", "FP16/BF16", "BNSD", "当前必须为 None/nullptr"),
            ("h0", "预留", "[B,H_v,K,V]", "FP16/BF16", "ND", "当前必须为 None/nullptr"),
            ("dht", "预留", "[B,H_v,K,V]", "FP16/BF16", "ND", "当前必须为 None/nullptr"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(
            ("dh", "[B,H_v,N_c,K,V]", "与 q 一致", "每个 chunk 起点的状态梯度"),
            ("dh0", "-", "-", "预留输出；Python 固定返回 None，aclnn/直调必须传 nullptr"),
            ("dv2", "[B,H_v,T,V]", "与 dv 一致", "合并状态贡献后的 Value 梯度"),
        ),
        "attrs": rows(
            ("scale", "double", "1.0", "Query 分支缩放"),
            ("chunk_size", "int", "64", "chunk 长度"),
            ("use_exp2", "bool", "false", "当前仅 false"),
            ("transpose_state_layout", "bool", "false", "当前仅 false"),
        ),
        "layouts": "BNSD；状态为 `[B,H_v,K,V]` 或 `[B,H_v,N_c,K,V]`",
        "dtype": "主张量 FP16/BF16；g 可额外为 FP32",
        "modes": "定长/变长序列、标量 g、FP16/BF16 主张量、FP16/BF16/FP32 gate",
        "limits": [
            "`K <= 128`、`V <= 256`；交付矩阵覆盖 K=128、V=128/256。",
            "`chunk_size` 仅支持 64/128；`H_v % H_k == 0`。",
            "`g` 必须提供；`gK`、`h0`、`dht` 和 `dh0` 当前为预留，必须为空。",
            "变长序列当前仅支持物理 `B=1`，两个索引必须同时提供。",
            "`use_exp2` 与 `transpose_state_layout` 当前必须为 false。",
        ],
        "task": "按 value head 和反向 chunk 序号分配状态递推任务；同一序列的 chunk 保持逆序依赖，不同 batch/head 可并行。",
        "tiling_key": "保留两个有限 key：1 表示 gate 与主张量同为 16 位类型，2 表示 `g` 为 FP32。区别必须在编译期选择 `GT` 模板，避免热循环 runtime dtype 分支；key 不包含 V、T 或普通属性，组合固定为 2。",
        "flow": "AIC 计算 `Q^T@dO` 和状态相关矩阵乘，AIV 处理 gate、反向递推与 dV 累加；从末 chunk 向首 chunk 更新 dH，并在 h0 存在时写 dH0。",
        "memory": "GM 状态梯度按 chunk 保存；workspace 为 Cube 与 Vector 的阶段结果，L1/L0 放矩阵乘 tile，UB 放 gate、尾块 mask 和 dV 累加片段。",
        "sync": "同一 head 的逆序 chunk 由任务协议保证先后；AIC/AIV 通过 workspace ready flag 交接。核内 MTE/Cube/Vector 事件保护 buffer 生命周期，构建固定 `--cce-auto-sync=off`。",
        "python_sig": "chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, scale, chunk_size, *, g=None, gK=None, h0=None, dht=None, cu_seqlens=None, chunk_indices=None, use_exp2=False, transpose_state_layout=False)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
            q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
            k = torch.randn_like(q)
            w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.float16)
            d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
            dv = torch.randn_like(d_o)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, K ** -0.5, chunk_size, g=g)
            torch.npu.synchronize()
            assert dv2.shape == dv.shape
        """),
        "aclnn_call": "q, k, w, dO, dv, g, nullptr, nullptr, nullptr, cuSeqlens, chunkIndices, scale, chunkSize, dh, nullptr, dv2, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_gated_delta_rule_bwd_dhu/accuracy/backend.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("必选 tensor 为空，或 g 未提供", "ACLNN_ERR_PARAM_INVALID"),
            ("gK/h0/dht/dh0 非空", "ACLNN_ERR_PARAM_INVALID"),
            ("chunk_size 不是 64/128，或变长序列索引只提供一个", "ACLNN_ERR_PARAM_INVALID"),
            ("Python use_exp2/transpose_state_layout 为 true", "RuntimeError"),
            ("执行器或 kernel launch 失败", "ACLNN_ERR_INNER"),
        ),
    },
    "prepare_wy_repr_bwd_da": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da",
        "title": "PrepareWyReprBwdDa",
        "family": "gdn",
        "purpose": "WY 表示反向的 dA 子算子。它根据 K/V、beta、前向 A、dW/dU 与 gate，计算 chunk 局部矩阵 A 的梯度，供完整 WY 反向继续生成 dK/dV/dBeta/dG。",
        "math": dedent(r"""
            在每个 value head/chunk 内，先将对应 key head 按 GVA 关系广播，再组合两条矩阵链：

            ```text
            dA = dU @ (V * beta)^T + dW @ (K * beta * exp(g))^T
            dA = causal_mask(dA) + A-dependent triangular correction
            ```

            `dA` 仅在 chunk 的有效因果区域有定义；尾块之外写零。
        """),
        "inputs": rows(
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"),
            ("v", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value"),
            ("beta", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "WY 权重"),
            ("A", "必选", "[B,H_v,T,chunk_size]", "FP16/BF16", "BNSD", "前向 chunk 局部矩阵"),
            ("dw", "必选", "[B,H_v,T,K]", "FP16/BF16", "BNSD", "W 梯度"),
            ("du", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "U 梯度"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "chunk-local gate"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(("dA", "[B,H_v,T,chunk_size]", "与 A 一致", "chunk 局部矩阵梯度")),
        "attrs": rows(("chunk_size", "int", "无", "必须等于 A/dA 最后一维")),
        "layouts": "BNSD；A/dA 最后一维为 chunk_size",
        "dtype": "K/V/A/dW/dU 为 FP16/BF16；beta/g 可为 FP32",
        "modes": "定长与变长序列，支持 GVA",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256。",
            "`chunk_size` 仅支持 64/128，并须等于 A/dA 的最后一维。",
            "必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。",
            "`cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。",
        ],
        "task": "按 `(batch/value_head/chunk)` 划分；GVA 通过 `h_k=floor(h_v/(H_v/H_k))` 选择 K，尾块把无效行列屏蔽。",
        "tiling_key": "当前仅使用 key=1 进入唯一模板化 kernel，不编码 runtime shape 或 dtype。该单值 key 是现有二进制入口约定，组合数为 1；后续 ABI 整理时应消除。",
        "flow": "AIC 分别计算 dU/V 与 dW/K 两条矩阵乘，AIV 应用 beta、gate、因果三角 mask 及 A 相关修正，最后写 dA。",
        "memory": "workspace 暂存两个 `chunk_size*chunk_size` 矩阵乘结果并按阶段复用；L1/L0 承载 K/V tile，UB 承载 beta/gate、mask 和 dA 合并片段。",
        "sync": "AIC 产出矩阵片段后通知 AIV 合并，AIV 写回或释放槽位后才允许下一轮覆盖；核内事件覆盖 MTE/Cube/Vector，构建固定 `--cce-auto-sync=off`。",
        "python_sig": "prepare_wy_repr_bwd_da(k, v, beta, A, dw, du, g, *, chunk_size, cu_seqlens=None, chunk_indices=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import prepare_wy_repr_bwd_da

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
            k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
            beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
            A = torch.randn(B, H_v, T, chunk_size, device="npu", dtype=torch.float16)
            dw = torch.randn(B, H_v, T, K, device="npu", dtype=torch.float16)
            du = torch.randn_like(v)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            dA = prepare_wy_repr_bwd_da(k, v, beta, A, dw, du, g, chunk_size=chunk_size)
            torch.npu.synchronize()
            assert dA.shape == A.shape
        """),
        "aclnn_call": "k, v, beta, A, dw, du, g, nullptr, nullptr, chunkSize, dA, &workspaceSize, &executor",
        "runner": "tests/operators/prepare_wy_repr_bwd_da/accuracy/backend.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("必选 tensor 为空，chunk_size 非 64/128，或变长序列元数据只提供一个", "ACLNN_ERR_PARAM_INVALID"),
            ("shape/dtype/GVA 不受模板支持", "tiling 失败；aclnn 执行返回 ACLNN_ERR_INNER"),
            ("执行器或 kernel launch 失败", "ACLNN_ERR_INNER"),
        ),
    },
    "prepare_wy_repr_bwd_full": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full",
        "title": "PrepareWyReprBwdFull",
        "family": "gdn",
        "purpose": "WY 表示完整反向算子。它消费 `dA/dW/dU`，在 chunk 内反传到 K、V、beta 与 gate，是 `prepare_wy_repr_bwd_da` 之后的主梯度阶段。",
        "math": dedent(r"""
            对每个 chunk，令 `Kb=K*beta*exp(g)`、`Vb=V*beta`，则链式法则包含：

            ```text
            dV     = A^T @ dU * beta
            dK     = A^T @ dW * beta * exp(g) + dA-related terms
            dBeta  = reduce_V(dVb * V) + reduce_K(dKb * K)
            dG     = reverse_chunk_reduce(dKb * K)
            ```

            `dA` 的因果区域和 GVA head 映射与前一阶段完全一致。
        """),
        "inputs": rows(
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"),
            ("v", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value"),
            ("beta", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "WY 权重"),
            ("A", "必选", "[B,H_v,T,chunk_size]", "FP16/BF16", "BNSD", "前向局部矩阵"),
            ("dA", "必选", "[B,H_v,T,chunk_size]", "FP16/BF16", "BNSD", "A 梯度"),
            ("dw", "必选", "[B,H_v,T,K]", "FP16/BF16", "BNSD", "W 梯度"),
            ("du", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "U 梯度"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "chunk-local gate"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(
            ("dk", "[B,H_k,T,K]", "与 k 一致", "Key 梯度；GVA value head 已归约"),
            ("dv", "[B,H_v,T,V]", "与 v 一致", "Value 梯度"),
            ("dbeta", "[B,H_v,T]", "与 beta 一致", "beta 梯度"),
            ("dg", "[B,H_v,T]", "与 g 一致", "gate 梯度"),
        ),
        "attrs": rows(("chunk_size", "int", "无", "必须等于 A/dA 最后一维")),
        "layouts": "BNSD；A/dA 最后一维为 chunk_size",
        "dtype": "主张量 FP16/BF16；beta/g 可为 FP32",
        "modes": "定长与变长序列，支持 GVA",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256。",
            "`chunk_size` 仅支持 64/128，并须等于 A/dA 最后一维。",
            "必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。",
            "`cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。",
        ],
        "task": "按 `(batch/value_head/chunk)` 划分主任务；dK 通过相同 key head 的 value-head group 归约，尾块按有效长度屏蔽。",
        "tiling_key": "使用两个固定模板实例覆盖当前 Cube tile/Value 维组合，key 不承载 B/H/T 等 runtime shape；组合上限固定为 2。保留 key 的原因是两个实例的 L1/L0 tile 类型在编译期不同，无法用普通 tiling data 安全切换。",
        "flow": "AIC 计算 A^T@dU、A^T@dW 和 dA 相关矩阵乘；AIV 应用 beta/gate、完成 dBeta/dG 归约，并把 GVA dK 贡献归并到 key head。",
        "memory": "workspace 分阶段保存 dK/dV 矩阵乘片段和 reduction 中间量；大 buffer 在消费者完成后复用，L1/L0 放 Cube tile，UB 放门控和归约片段。",
        "sync": "Cube 结果通过 workspace ready flag 交给 Vector，Vector 完成逐元素处理后释放槽位；同 key head 的归约写回有唯一 owner，避免跨核冲突。构建固定 `--cce-auto-sync=off`。",
        "python_sig": "prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size, *, cu_seqlens=None, chunk_indices=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import prepare_wy_repr_bwd_full

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 256, 64
            k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
            beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
            A = torch.randn(B, H_v, T, chunk_size, device="npu", dtype=torch.bfloat16)
            dA = torch.randn_like(A)
            dw = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
            du = torch.randn_like(v)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            dk, dv, dbeta, dg = prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size)
            torch.npu.synchronize()
            assert dk.shape == k.shape and dv.shape == v.shape
        """),
        "aclnn_call": "k, v, beta, A, dA, dw, du, g, nullptr, nullptr, chunkSize, dk, dv, dbeta, dg, &workspaceSize, &executor",
        "runner": "tests/operators/prepare_wy_repr_bwd_full/accuracy/backend.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("必选 tensor 为空，chunk_size 非 64/128，或变长序列元数据只提供一个", "ACLNN_ERR_PARAM_INVALID"),
            ("shape/dtype/GVA 不受模板支持", "tiling 失败；aclnn 执行返回 ACLNN_ERR_INNER"),
            ("执行器或 kernel launch 失败", "ACLNN_ERR_INNER"),
        ),
    },
}

OPS.update({
    "chunk_fwd_o": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o",
        "title": "ChunkFwdO",
        "family": "gdn",
        "purpose": "Gated Delta Rule 的 chunk 输出阶段。算子把当前 chunk 的 Q/K/V、chunk 起始状态 H 和累积 gate 合并，生成注意力输出 O；状态推进由 `chunk_gated_delta_rule_fwd_h` 单独完成。",
        "math": dedent(r"""
            对 value head `h_v` 映射 key head `h_k=floor(h_v/(H_v/H_k))`：

            ```text
            score = causal_mask((Q[h_k] @ K[h_k]^T) * scale * exp(g_row-g_col))
            O[h_v] = Q[h_k] @ H_start[h_v] * scale + score @ V[h_v]
            ```

            `H_start` 取当前 chunk 对应的状态切片；尾块只计算有效 token，输出保持 `[B,H_v,T,V]`。
        """),
        "inputs": rows(
            ("q", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Query"),
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"),
            ("v", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value"),
            ("h", "必选", "[B,H_v,N_c,K,V]", "FP16/BF16", "ND", "每个 chunk 的起始状态"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "累积标量 gate"),
            ("g_gamma", "预留", "-", "-", "-", "上层兼容参数，当前必须为 None"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(("o", "[B,H_v,T,V]", "与 v 一致", "chunk 注意力输出")),
        "attrs": rows(
            ("scale", "double", "无", "Query 缩放"),
            ("chunk_size", "int", "64", "chunk 长度"),
            ("transpose_state_layout", "bool", "false", "预留参数，当前必须 false"),
        ),
        "layouts": "BNSD；状态使用 `[B,H_v,N_c,K,V]`",
        "dtype": "Q/K/V/H 为 FP16/BF16；g 可为 FP32",
        "modes": "定长/变长序列、GVA、整块/尾块；g 为必选标量 gate",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。",
            "必须满足 `H_v % H_k == 0`，h 的 chunk 数必须与索引推导一致。",
            "变长序列当前仅支持物理 `B=1`，两个索引必须同时提供。",
            "`g` 是 kernel 必选输入；`g_gamma` 必须为 None，`transpose_state_layout` 必须为 false。",
        ],
        "task": "按 `(batch,value_head,chunk)` 分配，Q/K score 在同 key head 的 value-head group 间复用；h 通过全局 chunk 序号定位。",
        "tiling_key": "不使用 tiling key 分派 shape。主 dtype、gate dtype、定长/变长序列、K/V/chunk_size 均写入 tiling data，kernel 通过统一 Catlass 调度器处理，避免把 B/H/T 组合固化为 key。",
        "flow": "AIC 先计算 QK score 和 QH 状态项，AIV 应用 gate、因果/tail mask，AIC 再完成 score@V，最终与状态项相加写 O。",
        "memory": "workspace 保存 QK score 与 gated score 的阶段结果；L1/L0 承载 QK、QV 和 QH tile，UB 承载 gate、mask 与两个输出分支的逐元素合并。",
        "sync": "AIC/AIV 通过按 core/head 划分的 workspace 槽交接 score，ready/free flag 成对使用；核内 MTE/Cube/Vector 事件闭环，`--cce-auto-sync=off`。",
        "errors": rows(
            ("q/k/v/h/g/oOut、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR"),
            ("g_gamma 非 None、transpose_state_layout=true", "Python RuntimeError"),
            ("cu_seqlens/chunk_indices 未成对提供，或 chunk_size 非 64/128", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("rank、B/T、GVA head、H/K/V/output shape 或 dtype 不匹配", "ACLNN_ERR_PARAM_INVALID"),
        ),
        "python_sig": "chunk_fwd_o(q, k, v, h, scale, *, g=None, g_gamma=None, cu_seqlens=None, chunk_indices=None, chunk_size=None, transpose_state_layout=False)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_fwd_o

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
            N_c = (T + chunk_size - 1) // chunk_size
            q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
            h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            o = chunk_fwd_o(q, k, v, h, K ** -0.5, g=g, chunk_size=chunk_size)
            torch.npu.synchronize()
            assert o.shape == v.shape
        """),
        "aclnn_call": "q, k, v, h, g, nullptr, nullptr, scale, chunkSize, o, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_fwd_o/accuracy/backend.py",
        "reference": "torch_chunk_fwd_o_reference",
        "case": {
            "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
            "dtype": {"qkvh": "float16", "g": "float32"}, "layout": "BNSD",
            "attrs": {"scale": 0.0883883476, "chunk_size": 64}, "optional_inputs": {"g": "present", "cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 2, "H_k": 4, "H_v": 4, "T": 256, "K": 128, "V": 256, "chunk_size": 128, "N_c": 2},
            "variant_dtype": {"qkvh": "bfloat16", "g": "bfloat16"},
            "variant_attrs": {"scale": 0.0883883476, "chunk_size": 128},
            "tail_shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
            "tail_attrs": {"scale": 0.0883883476, "chunk_size": 64},
            "tail_optional_inputs": {"g": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
            "negative_shape": {"B": 1, "H_k": 3, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
            "negative_message": "HV divisible by HK",
            "extra_cases": [
                {
                    "id": "chunk_fwd_o_missing_gate", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"qkvh": "float16"}, "layout": "BNSD",
                    "attrs": {"scale": 0.0883883476, "chunk_size": 64},
                    "optional_inputs": {"g": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "g must not be nullptr"},
                },
                {
                    "id": "chunk_fwd_o_varlen_pair_required", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"qkvh": "float16", "g": "float32"}, "layout": "BNSD",
                    "attrs": {"scale": 0.0883883476, "chunk_size": 64},
                    "optional_inputs": {"g": "present", "cu_seqlens": [0, 128], "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "both be provided"},
                },
                {
                    "id": "chunk_fwd_o_invalid_feature_dim", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 64, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"qkvh": "float16", "g": "float32"}, "layout": "BNSD",
                    "attrs": {"scale": 0.125, "chunk_size": 64},
                    "optional_inputs": {"g": "present", "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "K must be 128"},
                },
                {
                    "id": "chunk_fwd_o_invalid_h_chunk_count", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 3},
                    "dtype": {"qkvh": "float16", "g": "float32"}, "layout": "BNSD",
                    "attrs": {"scale": 0.0883883476, "chunk_size": 64},
                    "optional_inputs": {"g": "present", "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "NC"},
                },
                {
                    "id": "chunk_fwd_o_invalid_chunk_order", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
                    "dtype": {"qkvh": "float16", "g": "float32"}, "layout": "BNSD",
                    "attrs": {"scale": 0.0883883476, "chunk_size": 64},
                    "optional_inputs": {"g": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [1, 0, 0, 0, 0, 1, 1, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "canonical"},
                },
            ],
        },
    },
    "chunk_gated_delta_rule_fwd_h": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h",
        "title": "ChunkGatedDeltaRuleFwdH",
        "family": "gdn",
        "purpose": "Gated Delta Rule 的跨 chunk 状态推进算子。它根据 K/W/U、标量 gate 或逐 K 维 gate，从 initial_state 递推各 chunk 起始状态 H、修正值 V_new，并按需返回 final_state。",
        "math": dedent(r"""
            令 `S_c` 为 chunk c 的起始状态，`G_c` 为该 chunk 末端门控：

            ```text
            V_new_c = U_c - W_c @ S_c
            S_(c+1) = decay(G_c) * S_c + K_gated_c^T @ V_new_c
            H[c]    = S_c
            ```

            `g` 路径使用每 head 标量衰减，`gk` 路径对 K 维逐元素衰减；二者至少提供一个。
        """),
        "inputs": rows(
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key 或已门控 KG"),
            ("w", "必选", "[B,H_v,T,K]", "FP16/BF16", "BNSD", "WY 的 W"),
            ("u", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "WY 的 U"),
            ("g", "条件可选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "标量 gate"),
            ("gk", "条件可选", "[B,H_v,T,K]", "FP16/BF16/FP32", "BNSD", "逐 K 维 gate"),
            ("initial_state", "可选", "[N,H_v,K,V]", "FP16/BF16/FP32", "ND", "每序列初始状态"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(
            ("h", "[B,H_v,N_c,K,V]", "与 k 一致", "每个 chunk 起始状态"),
            ("v_new", "[B,H_v,T,V]", "与 u 一致", "状态修正后的 Value"),
            ("final_state", "[N,H_v,K,V]", "跟随 initial_state 或 FP32", "按需返回末状态"),
        ),
        "attrs": rows(
            ("output_final_state", "bool", "false", "是否返回末状态"), ("chunk_size", "int", "64", "chunk 长度"),
            ("save_new_value", "bool", "true", "当前必须 true"), ("use_exp2", "bool", "false", "当前必须 false"),
            ("transpose_state_layout", "bool", "false", "当前必须 false"),
        ),
        "layouts": "BNSD；initial/final state 为 `[N,H_v,K,V]`",
        "dtype": "K/W/U 为 FP16/BF16；gate/state 可为 FP32",
        "modes": "定长/变长序列、g/gk、可选 initial/final state",
        "limits": [
            "`K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。",
            "`g` 与 `gk` 至少提供一个；`H_v % H_k == 0`。",
            "变长序列当前仅支持物理 `B=1`，索引必须完整且 sequence-major。",
            "`save_new_value=true`、`use_exp2=false`、`transpose_state_layout=false`。",
        ],
        "task": "每个 `(sequence,value_head)` 是有序状态链；不同序列/head 并行，同一链的 chunk 按先后顺序执行并写 H。",
        "tiling_key": "仅保留 2 个 key：V<=128 选择 key 1 的 128 列 Cube tile，128<V<=256 选择 key 2 的 256 列 tile。必须使用 key 是因为两种 Cube tile 和 KERNEL_TASK_TYPE 在编译期不同；dtype、g/gk、initial/final、定长/变长序列均由 tiling data 处理，不继续扩张 key。",
        "flow": "AIC 计算 W@S 与 K^T@V_new，AIV 计算门控衰减、V_new 和状态合并；逐 chunk 推进，最后按属性写 final_state。",
        "memory": "状态 S 在 GM/工作区按 sequence/head 独占，L1/L0 放两个矩阵乘 tile，UB 保存 gate、V_new 和状态逐元素更新片段。",
        "sync": "同一状态链由单一 owner 保证 chunk 顺序；AIC/AIV 通过 ready flag 交接 W@S 和 K^T@V。核内事件闭环，`--cce-auto-sync=off`。",
        "errors": rows(
            ("k/w/u/hOut/vNewOut、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR"),
            ("g 与 gk 同时为空，或 g/gk shape/dtype 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("save_new_value=false、use_exp2=true、transpose_state_layout=true", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("cu_seqlens/chunk_indices 未成对提供，或 chunk_size 非 64/128", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("K/W/U、initial/final state、H/V_new 或 GVA shape 不匹配", "ACLNN_ERR_PARAM_INVALID"),
        ),
        "python_sig": "chunk_gated_delta_rule_fwd_h(k, w, u, g=None, *, gk=None, initial_state=None, output_final_state=False, chunk_size=None, save_new_value=True, cu_seqlens=None, chunk_indices=None, use_exp2=False, transpose_state_layout=False)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
            k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
            w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
            u = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
                k, w, u, g, chunk_size=chunk_size, output_final_state=True)
            torch.npu.synchronize()
            assert v_new.shape == u.shape and final_state.shape == (B, H_v, K, V)
        """),
        "aclnn_call": "k, w, u, g, nullptr, nullptr, true, chunkSize, true, nullptr, nullptr, false, false, h, vNew, finalState, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_gated_delta_rule_fwd_h/accuracy/backend.py",
        "reference": "torch_chunk_gated_delta_rule_fwd_h_reference",
        "case": {
            "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
            "dtype": {"kwu": "float16", "g": "float32", "state": "float32"}, "layout": "BNSD",
            "attrs": {"chunk_size": 64, "output_final_state": True, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
            "optional_inputs": {"g": "present", "gk": None, "initial_state": None, "cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 1, "H_k": 4, "H_v": 8, "T": 256, "K": 128, "V": 256, "chunk_size": 128, "N_c": 2},
            "variant_dtype": {"kwu": "bfloat16", "gk": "float32", "state": "float32"},
            "variant_attrs": {"chunk_size": 128, "output_final_state": True, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
            "variant_optional_inputs": {"g": None, "gk": "present", "initial_state": "present", "cu_seqlens": None, "chunk_indices": None},
            "tail_shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
            "tail_dtype": {"kwu": "bfloat16", "g": "float32", "state": "float32"},
            "tail_attrs": {"chunk_size": 64, "output_final_state": True, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
            "tail_optional_inputs": {"g": "present", "gk": None, "initial_state": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
            "negative_optional_inputs": {"g": None, "gk": None, "initial_state": None, "cu_seqlens": None, "chunk_indices": None},
            "negative_message": "either g or gk",
            "extra_cases": [
                {
                    "id": "chunk_gated_delta_rule_fwd_h_reserved_option", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"kwu": "float16", "g": "float32", "state": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64, "output_final_state": False, "save_new_value": False, "use_exp2": False, "transpose_state_layout": False},
                    "optional_inputs": {"g": "present", "gk": None, "initial_state": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "save_new_value"},
                },
                {
                    "id": "chunk_gated_delta_rule_fwd_h_varlen_pair_required", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"kwu": "float16", "g": "float32", "state": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64, "output_final_state": False, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
                    "optional_inputs": {"g": "present", "gk": None, "initial_state": None, "cu_seqlens": [0, 128], "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "both be provided"},
                },
                {
                    "id": "chunk_gated_delta_rule_fwd_h_invalid_feature_dim", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 64, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"kwu": "float16", "g": "float32", "state": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64, "output_final_state": False, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
                    "optional_inputs": {"g": "present", "gk": None, "initial_state": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "K must be 128"},
                },
                {
                    "id": "chunk_gated_delta_rule_fwd_h_invalid_state_shape", "tags": ["negative", "boundary"],
                    "shape": {"B": 2, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"kwu": "float16", "g": "float32", "state": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64, "output_final_state": True, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
                    "optional_inputs": {"g": "present", "gk": None, "initial_state": {"status": "present", "shape": "[1,H_v,K,V]"}, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "initial_state"},
                },
                {
                    "id": "chunk_gated_delta_rule_fwd_h_invalid_chunk_order", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
                    "dtype": {"kwu": "bfloat16", "g": "float32", "state": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64, "output_final_state": True, "save_new_value": True, "use_exp2": False, "transpose_state_layout": False},
                    "optional_inputs": {"g": "present", "gk": None, "initial_state": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [1, 0, 0, 0, 0, 1, 1, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "canonical"},
                },
            ],
        },
    },
    "recompute_wu_fwd": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd",
        "title": "RecomputeWUFwd",
        "family": "gdn",
        "purpose": "按需重计算 WY 中间张量 W/U，减少前向保存显存。输入 K/V/beta/A 与标量 gate g，在每个 chunk 内输出 W 和 U。",
        "math": dedent(r"""
            对 value head `h_v` 与映射后的 key head `h_k`：

            ```text
            Vb = V[h_v] * beta[h_v]
            Kb = K[h_k] * beta[h_v] * exp(g[h_v])
            U  = A[h_v] @ Vb
            W  = A[h_v] @ Kb
            ```

            最后一个 chunk 使用实际有效行数，A 的其余列不参与结果。
        """),
        "inputs": rows(
            ("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"), ("v", "必选", "[B,H_v,T,V]", "FP16/BF16", "BNSD", "Value"),
            ("beta", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "WY 权重"), ("A", "必选", "[B,H_v,T,chunk_size]", "FP16/BF16", "BNSD", "局部矩阵"),
            ("g", "必选", "[B,H_v,T]", "FP16/BF16/FP32", "BNS", "标量 gate"),
            ("gk", "预留", "-", "-", "-", "当前 kernel 不消费，必须为 None"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"), ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引"),
        ),
        "outputs": rows(("w", "[B,H_v,T,K]", "与 k 一致", "WY 的 W"), ("u", "[B,H_v,T,V]", "与 v 一致", "WY 的 U")),
        "attrs": rows(("chunk_size", "int", "无", "chunk 长度并等于 A 最后一维")),
        "layouts": "BNSD",
        "dtype": "主张量 FP16/BF16；beta/g/gk 可为 FP32",
        "modes": "定长/变长序列、GVA、标量 gate g",
        "limits": [
            "当前实现仅支持 `K=128`、`V=128/256`、`chunk_size=64/128`。",
            "必须满足 `H_v % H_k == 0`，A 最后一维等于 chunk_size。",
            "g 必须提供；gk 当前未实现且必须为 None；变长序列物理 B=1。",
        ],
        "task": "按 value head/chunk 分配，两次矩阵乘共享 A tile；K 通过 GVA 映射读取。",
        "tiling_key": "保留 V=128 与 V=256 两个编译期 Cube tile 实例（key 1/2），普通 shape 和模式放 tiling data；组合固定为 2。",
        "flow": "AIV 生成 Vb/Kb，AIC 复用 A tile 分别计算 U/W，尾块由 valid row 控制写回。",
        "memory": "UB 保存 beta/gate 逐元素结果，L1/L0 复用 A tile，workspace 保存 AIV 到 AIC 的 Kb/Vb 中间块。",
        "sync": "AIV 产出 Kb/Vb 后通知 AIC，AIC 完成两个矩阵乘后释放对应槽位；`--cce-auto-sync=off`。",
        "errors": rows(
            ("k/v/beta/A/g/wOut/uOut、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR"),
            ("gk 非 None", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("cu_seqlens/chunk_indices 未成对提供，或 chunk_size 非 64/128", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("rank、B/T/H、GVA、A 最后一维、输出 shape 或 dtype 不匹配", "ACLNN_ERR_PARAM_INVALID"),
        ),
        "python_sig": "recompute_wu_fwd(k, v, beta, A, chunk_size, *, g=None, gk=None, cu_seqlens=None, chunk_indices=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import recompute_wu_fwd

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 256, 64
            k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
            beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
            A = torch.randn(B, H_v, T, chunk_size, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            w, u = recompute_wu_fwd(k, v, beta, A, chunk_size, g=g)
            torch.npu.synchronize()
            assert w.shape == (B, H_v, T, K) and u.shape == v.shape
        """),
        "aclnn_call": "k, v, beta, A, g, nullptr, nullptr, nullptr, chunkSize, w, u, &workspaceSize, &executor",
        "runner": "tests/operators/recompute_wu_fwd/accuracy/backend.py",
        "legacy_op": "npu_recompute_w_u_fwd",
        "reference": "torch_recompute_wu_reference",
        "case": {
            "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64},
            "dtype": {"kvA": "float16", "beta_g": "float32"}, "layout": "BNSD", "attrs": {"chunk_size": 64},
            "optional_inputs": {"g": "present", "gk": None, "cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 2, "H_k": 4, "H_v": 8, "T": 256, "K": 128, "V": 256, "chunk_size": 128},
            "variant_dtype": {"kvA": "bfloat16", "beta_g": "float32"},
            "variant_attrs": {"chunk_size": 128},
            "variant_optional_inputs": {"g": "present", "gk": None, "cu_seqlens": None, "chunk_indices": None},
            "tail_shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64},
            "tail_dtype": {"kvA": "bfloat16", "beta_g": "float32"},
            "tail_attrs": {"chunk_size": 64},
            "tail_optional_inputs": {"g": "present", "gk": None, "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
            "negative_shape": {"B": 1, "H_k": 3, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64}, "negative_message": "HV divisible by HK",
            "extra_cases": [
                {
                    "id": "recompute_wu_fwd_reserved_gk", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64},
                    "dtype": {"kvA": "float16", "beta_g": "float32", "gk": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64},
                    "optional_inputs": {"g": "present", "gk": "present", "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "gk is reserved"},
                },
                {
                    "id": "recompute_wu_fwd_missing_gate", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64},
                    "dtype": {"kvA": "float16", "beta": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64},
                    "optional_inputs": {"g": None, "gk": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "g must not be nullptr"},
                },
                {
                    "id": "recompute_wu_fwd_invalid_feature_dim", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 64, "V": 128, "chunk_size": 64},
                    "dtype": {"kvA": "float16", "beta_g": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64},
                    "optional_inputs": {"g": "present", "gk": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "K must be 128"},
                },
                {
                    "id": "recompute_wu_fwd_invalid_a_width", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 32},
                    "dtype": {"kvA": "float16", "beta_g": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64},
                    "optional_inputs": {"g": "present", "gk": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "A must be"},
                },
                {
                    "id": "recompute_wu_fwd_varlen_requires_b1", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64},
                    "dtype": {"kvA": "float16", "beta_g": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 64},
                    "optional_inputs": {"g": "present", "gk": None, "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "B must be 1"},
                },
            ],
        },
    },
    "solve_tri": {
        "root": "fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri",
        "title": "SolveTri",
        "family": "gdn",
        "replaces_triton": True,
        "triton_baseline": "fla_npu.ops.triton.solve_tril_npu",
        "purpose": "对每个 chunk 的严格下三角矩阵 A 计算 `(I+A)^-1`，用于 WY 表示求解。输入最后一维保存当前 token 行的 chunk 列，输出保持相同布局。",
        "math": dedent(r"""
            对每个 batch/head/chunk，取有效阶数 M 的严格下三角矩阵 A：

            ```text
            Y = inverse(I_M + tril(A, diagonal=-1))
            ```

            实现采用分块前代/三角逆；尾块只对有效 M 求逆，padding 列按接口约定写零。
        """),
        "inputs": rows(("x", "必选", "[B,H_v,T,chunk_size]、[B,T,H_v,chunk_size] 或 [T,H_v,chunk_size]", "FP16/BF16", "BHTD/BSND/TND", "严格下三角 A 的行存储"),
                       ("cu_seqlens", "TND 必选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
                       ("chunk_indices", "TND 必选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引")),
        "outputs": rows(("x_out", "与 x 相同", "与 x 相同", "(I+A) 的 chunk-wise 逆")),
        "attrs": rows(("layout", "str", "bsnd", "仅 bhtd/bsnd/tnd，小写")),
        "layouts": "BHTD `[B,H_v,T,chunk_size]`、BSND `[B,T,H_v,chunk_size]`、TND `[T,H_v,chunk_size]`；layout 字符串必须小写",
        "dtype": "FP16/BF16",
        "modes": "dense BHTD/BSND 与变长序列 TND，支持尾块",
        "limits": [
            "矩阵阶/最后一维 chunk_size 支持 16/32/64/128。",
            "layout 仅支持小写 `bhtd`、`bsnd`、`tnd`；TND 必须提供两个变长序列索引，定长布局 不接受变长序列索引。",
            "输入必须表示严格下三角 A；对角线由算子加单位阵。",
        ],
        "task": "按 `(batch/head/chunk)` 分配，chunk_size=128 时继续按内部 tile 求解，尾块使用局部 M。",
        "tiling_key": "固定使用 key 1 作为既有 kernel launch ABI，key 不承载 shape 或模式分派。chunk_size、dtype、layout、尾块和变长序列全部由 tiling data 处理，因此不会随 B/H/T 产生组合爆炸；保留 key 1 是因为 kernel 入口现有 TILING_KEY_IS(1) 编译契约。",
        "flow": "加载 A tile，清理上三角并注入单位对角，按对角块前代求逆，逐块更新剩余下三角并写回。",
        "memory": "UB/L1 保存当前三角 tile 和单位阵，L0/Cube 处理块乘更新；GM 输出与输入不别名。",
        "sync": "每个矩阵由单 core/协作组按对角块顺序推进，核内 MTE/Vector/Cube 事件保护 tile 复用；`--cce-auto-sync=off`。",
        "errors": rows(
            ("x/xOut/layout、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR"),
            ("layout 不是小写 bhtd/bsnd/tnd，或 layout 与 rank 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("TND 缺少任一变长序列索引，或定长布局 携带变长序列索引", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("chunk_size 非 16/32/64/128、xOut shape/dtype 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
        ),
        "python_sig": "solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout='bsnd')",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import solve_tri

            B, T, H, chunk_size = 1, 128, 4, 64
            x = torch.randn(B, T, H, chunk_size, device="npu", dtype=torch.float16)
            row = torch.arange(chunk_size, device="npu").view(1, 1, 1, chunk_size)
            pos = torch.arange(T, device="npu").view(1, T, 1, 1) % chunk_size
            x = torch.where(row < pos, x * 0.01, torch.zeros_like(x))
            y = solve_tri(x, layout="bsnd")
            torch.npu.synchronize()
            assert y.shape == x.shape
        """),
        "aclnn_call": "x, nullptr, nullptr, \"bsnd\", out, &workspaceSize, &executor",
        "runner": "tests/operators/solve_tri/accuracy/backend.py",
        "reference": "torch_linalg_triangular_reference",
        "case": {
            "shape": {"B": 1, "T": 128, "H_v": 4, "chunk_size": 64, "N_c": 2}, "dtype": {"x": "float16"}, "layout": "bsnd",
            "attrs": {"layout": "bsnd"}, "optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 2, "T": 256, "H_v": 8, "chunk_size": 128, "N_c": 2}, "variant_dtype": {"x": "bfloat16"},
            "variant_layout": "bhtd", "variant_attrs": {"layout": "bhtd"},
            "tail_shape": {"N": 2, "T": 193, "H_v": 4, "chunk_size": 64, "N_c": 4}, "tail_layout": "tnd", "tail_attrs": {"layout": "tnd"},
            "tail_optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
            "negative_layout": "tnd", "negative_attrs": {"layout": "tnd"}, "negative_optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
            "negative_message": "cu_seqlens",
            "extra_cases": [
                {
                    "id": "solve_tri_c16_boundary", "tags": ["accuracy", "generalization", "boundary"],
                    "shape": {"B": 1, "T": 33, "H_v": 2, "chunk_size": 16, "N_c": 3},
                    "dtype": {"x": "float16"}, "layout": "bsnd", "attrs": {"layout": "bsnd"},
                    "optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
                },
                {
                    "id": "solve_tri_invalid_chunk_size", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "T": 48, "H_v": 2, "chunk_size": 48, "N_c": 1},
                    "dtype": {"x": "float16"}, "layout": "bsnd", "attrs": {"layout": "bsnd"},
                    "optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "16, 32, 64 or 128"},
                },
                {
                    "id": "solve_tri_invalid_layout_case", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "T": 64, "H_v": 2, "chunk_size": 32, "N_c": 2},
                    "dtype": {"x": "float16"}, "layout": "BSND", "attrs": {"layout": "BSND"},
                    "optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "lowercase"},
                },
            ],
        },
    },
    "recurrent_gated_delta_rule": {
        "root": "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule",
        "title": "RecurrentGatedDeltaRule",
        "family": "gdn",
        "purpose": "面向 decode/投机推理的小步长 recurrent Gated Delta Rule。它按逻辑序列选择初始状态槽，逐 token 生成输出，并把每步状态写入 `ssm_state_indices` 指定的槽；支持标量 gate、逐 K gate 与接受 token 数。",
        "math": dedent(r"""
            对逻辑序列 b，`actual_seq_lengths[0]` 是前置跳过 token 数，后续元素是各序列长度。令序列起点为 `p_b`；无 `num_accepted_tokens` 时从 `ssm_state_indices[p_b]` 读取初始状态，有该输入时从 `ssm_state_indices[p_b+a_b-1]` 读取。对序列内 token t：

            ```text
            S_t = exp(g_t) * S_(t-1)                         # g 存在时
            S_t = S_t * exp(gk_t)[None, :]                   # gk 存在时
            delta_t = beta_t * (v_t - S_t @ k_t)
            S_t = S_t + outer(delta_t, k_t)
            o_t = S_t @ (q_t * scale)
            state_ref[ssm_state_indices[t]] = S_t
            ```

            `g` 与 `gk` 可独立使用、同时使用或同时为空；同时为空时不施加衰减。`state_ref` 是可变输入输出。
        """),
        "inputs": rows(
            ("query", "必选", "[T,H_k,K]", "BF16", "TND", "Query"), ("key", "必选", "[T,H_k,K]", "BF16", "TND", "Key"),
            ("value", "必选", "[T,H_v,V]", "BF16", "TND", "Value"), ("beta", "必选", "[T,H_v]", "BF16", "TN", "更新权重"),
            ("state_ref", "必选/可变", "[D_s,H_v,V,K]", "BF16/FP32", "ND", "状态槽，原地更新"),
            ("actual_seq_lengths", "必选", "[B+1]", "INT32", "ND", "首项为前置跳过长度，后续 B 项为各逻辑序列长度，总和为 T"),
            ("ssm_state_indices", "必选", "[T]", "INT32", "ND", "token 到状态槽映射"),
            ("g", "可选", "[T,H_v]", "FP32", "TN", "标量 gate"), ("gk", "可选", "[T,H_v,K]", "FP32", "TND", "逐 K 维 gate"),
            ("num_accepted_tokens", "可选", "[B]", "INT32", "ND", "每序列用于选择初始状态槽的位置，范围 [1,seq_len]"),
        ),
        "outputs": rows(("out", "[T,H_v,V]", "BF16", "recurrent 输出"), ("state_ref", "[D_s,H_v,V,K]", "与输入一致", "原地更新后的状态")),
        "attrs": rows(("scale_value", "float", "1.0", "推荐 1/sqrt(K)")),
        "layouts": "TND + 独立状态槽布局 `[D_s,H_v,V,K]`",
        "dtype": "Q/K/V/beta/out 为 BF16；state 为 BF16/FP32；gate FP32；索引 INT32",
        "modes": "变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state",
        "limits": [
            "每条序列本次有效 token 数不超过 8。",
            "g/gk 均为空表示单位衰减；二者存在时 dtype 必须为 FP32，shape 分别为 `[T,H_v]`、`[T,H_v,K]`。",
            "state 槽索引必须位于 `[0,D_s)`；actual_seq_lengths 有效项之和等于 T。",
            "`H_k/H_v <= 256`、`K/V <= 512` 且 `H_v % H_k == 0`。",
            "state_ref 为原地更新参数，不能 require_grad；非连续 state 依赖 CANN >= 9.1。",
        ],
        "task": "按序列/token/head 分配，映射到同一 state slot 的 token 保持程序顺序；不同槽可并行。",
        "tiling_key": "使用 tiling 模板注册但不使用数值 tiling key 分派；GetTilingKey 固定为 0。BF16 主路径、gate 组合、state dtype、token/head/dim 与 UB buffer profile 全部由 tiling data 描述。",
        "flow": "读取状态与 gate，AIC 计算 k@S/q@S 和 outer update，AIV 处理 beta、接受 mask 与 state 合并，写 out 并提交 state。",
        "memory": "state_ref 常驻 GM；UB 保存单步 gate/beta/mask，L1/L0 保存当前 state tile 和向量-矩阵计算片段。",
        "sync": "同一 slot 的读改写由任务归属和 token 顺序串行化，提交 state 前等待 Cube/Vector 完成；`--cce-auto-sync=off`。",
        "errors": rows(
            ("任一必选 tensor、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR"),
            ("Q/K/V/beta/out 非 BF16，state 非 BF16/FP32，gate 非 FP32，索引非 INT32", "ACLNN_ERR_PARAM_INVALID"),
            ("rank、T/H/K/V、GVA、state/out、g/gk 或 num_accepted_tokens shape 不匹配", "ACLNN_ERR_PARAM_INVALID"),
            ("每序列长度超过 8，accepted token 不在 [1,seq_len]，或索引值越界", "不支持输入；kernel 防护分支提前结束，调用方必须在提交前校验"),
        ),
        "python_sig": "recurrent_gated_delta_rule(query, key, value, beta, state_ref, actual_seq_lengths, ssm_state_indices, *, g=None, gk=None, num_accepted_tokens=None, scale_value=1.0)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import recurrent_gated_delta_rule

            B, L, H_k, H_v, K, V, D_s = 2, 2, 2, 4, 128, 128, 4
            T = B * L
            q = torch.randn(T, H_k, K, device="npu", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(T, H_v, V, device="npu", dtype=torch.bfloat16)
            beta = torch.rand(T, H_v, device="npu", dtype=torch.bfloat16)
            state = torch.zeros(D_s, H_v, V, K, device="npu", dtype=torch.float32)
            lengths = torch.tensor([0, L, L], device="npu", dtype=torch.int32)
            indices = torch.arange(T, device="npu", dtype=torch.int32)
            g = torch.zeros(T, H_v, device="npu", dtype=torch.float32)
            out, state = recurrent_gated_delta_rule(q, k, v, beta, state, lengths, indices, g=g, scale_value=K ** -0.5)
            torch.npu.synchronize()
            assert out.shape == v.shape
        """),
        "aclnn_call": "query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, nullptr, numAcceptedTokens, scaleValue, out, &workspaceSize, &executor",
        "runner": "tests/operators/recurrent_gated_delta_rule/accuracy/backend.py",
        "reference": "torch_recurrent_gated_delta_rule_reference",
        "case": {
            "shape": {"B": 2, "T": 4, "H_k": 2, "H_v": 4, "K": 128, "V": 128, "D_s": 4, "Q_a": 2},
            "dtype": {"qkv_beta_out": "bfloat16", "state_g": "float32", "indices": "int32"}, "layout": "TND",
            "attrs": {"scale_value": 0.0883883476}, "optional_inputs": {"g": "present", "gk": None, "num_accepted_tokens": None},
            "variant_shape": {"B": 4, "T": 8, "H_k": 4, "H_v": 8, "K": 128, "V": 128, "D_s": 8, "Q_a": 2},
            "variant_optional_inputs": {"g": None, "gk": "present", "num_accepted_tokens": [2, 1, 2, 1]},
            "tail_shape": {"B": 3, "T": 7, "H_k": 2, "H_v": 4, "K": 128, "V": 128, "D_s": 7, "Q_a": 3},
            "tail_optional_inputs": {"g": "present", "gk": None, "num_accepted_tokens": [3, 1, 2]},
            "negative_shape": {"B": 2, "T": 4, "H_k": 3, "H_v": 4, "K": 128, "V": 128, "D_s": 4, "Q_a": 2},
            "negative_optional_inputs": {"g": None, "gk": None, "num_accepted_tokens": None}, "negative_message": "HV divisible by HK",
            "extra_cases": [
                {
                    "id": "recurrent_gated_delta_rule_g_and_gk", "tags": ["accuracy", "generalization", "boundary"],
                    "shape": {"B": 2, "T": 6, "H_k": 2, "H_v": 4, "K": 128, "V": 128, "D_s": 6, "Q_a": 3},
                    "dtype": {"qkv_beta_out": "bfloat16", "state_g_gk": "float32", "indices": "int32"},
                    "layout": "TND", "attrs": {"scale_value": 0.0883883476},
                    "optional_inputs": {"g": "present", "gk": "present", "num_accepted_tokens": None},
                },
                {
                    "id": "recurrent_gated_delta_rule_invalid_gate_shape", "tags": ["negative", "boundary"],
                    "shape": {"B": 2, "T": 4, "H_k": 2, "H_v": 4, "K": 128, "V": 128, "D_s": 4, "Q_a": 2},
                    "dtype": {"qkv_beta_out": "bfloat16", "state_gk": "float32", "indices": "int32"},
                    "layout": "TND", "attrs": {"scale_value": 0.0883883476},
                    "optional_inputs": {"g": None, "gk": {"status": "present", "shape": "[T,2,K]"}, "num_accepted_tokens": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "gk must be [T, HV, K]"},
                },
                {
                    "id": "recurrent_gated_delta_rule_invalid_accepted_shape", "tags": ["negative", "boundary"],
                    "shape": {"B": 3, "T": 6, "H_k": 2, "H_v": 4, "K": 128, "V": 128, "D_s": 6, "Q_a": 2},
                    "dtype": {"qkv_beta_out": "bfloat16", "state": "float32", "indices": "int32"},
                    "layout": "TND", "attrs": {"scale_value": 0.0883883476},
                    "optional_inputs": {"g": None, "gk": None, "num_accepted_tokens": [2, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "B entries"},
                },
            ],
        },
    },
})

OPS.update({
    "causal_conv1d": {
        "root": "fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d",
        "title": "CausalConv1d",
        "family": "gdn",
        "purpose": "GDN 输入预处理的因果一维卷积。run_mode=0 执行定长/变长序列前向并维护卷积状态，run_mode=1 执行 decode/投机更新；可选 SiLU 激活和 head layout 转换。",
        "math": dedent(r"""
            对通道 d 和时间 t：

            ```text
            z[t,d] = bias[d] + sum(j=0..W-1, weight[j,d] * x[t-j,d])
            y[t,d] = z[t,d]                    activation_mode=0
            y[t,d] = z[t,d] * sigmoid(z[t,d]) activation_mode=1
            ```

            `t-j` 越过序列起点时读取该序列的 `conv_states`；update 模式只提交有效/已接受 token 并原地滚动状态。
        """),
        "inputs": rows(
            ("x", "必选", "[B,T,D] 或 [T,D]", "FP16/BF16", "BSH/TH", "输入序列"),
            ("weight", "必选", "[W,D]", "FP16/BF16", "ND", "depthwise 卷积权重"),
            ("bias", "可选", "[D]", "FP16/BF16", "ND", "偏置"),
            ("conv_states", "必选/可变", "[D_s,L_s,D]", "FP16/BF16", "ND", "历史输入状态，原地更新"),
            ("query_start_loc", "可选", "[B+1]", "INT64", "ND", "变长序列边界"),
            ("cache_indices", "可选", "[B]", "INT64", "ND", "序列到状态槽的映射"),
            ("initial_state_mode", "可选", "[B]", "INT64", "ND", "是否使用已有初始状态"),
            ("num_accepted_tokens", "可选", "[B]", "INT64", "ND", "投机解码接受数"),
        ),
        "outputs": rows(("y", "与 x 对应；head_num>0 时为 BNSD/NTD", "与 x 一致", "卷积输出"),
                           ("conv_states", "[D_s,L_s,D]", "与输入一致", "原地更新后的状态")),
        "attrs": rows(
            ("activation_mode", "int", "0", "0=无激活，1=SiLU"), ("pad_slot_id", "int", "-1", "跳过的缓存槽"),
            ("run_mode", "int", "0", "0=forward，1=update"), ("head_num", "int", "0", "forward 输出拆分 head；0 保持 BSH/TH"),
        ),
        "layouts": "输入 BSH/TH；head_num>0 时输出 BNSD/NTD",
        "dtype": "x/weight/bias/state/y 为 FP16/BF16；元数据 INT64",
        "modes": "定长/变长序列 forward、decode/update、投机接受 token",
        "limits": [
            "卷积宽度 W 仅支持 2/3/4，特征维 D 必须为 16 的倍数。",
            "activation_mode 仅支持 0/1，run_mode 仅支持 0/1。",
            "head_num>0 仅用于 forward，且必须整除 D；拆分后的 D/head_num 仍须为 16 的倍数。",
            "conv_states 的 L_s 至少为 W-1；rank-3 投机 update 还要求 L_s >= (W-1)+(T-1)。",
            "num_accepted_tokens 仅在 run_mode=1 且 W=4 时实现，值域为每条逻辑序列的 [0,token_count]。",
            "conv_states 为可变输入；非连续状态仅在 CANN >= 9.1 支持。",
            "rank-2 forward 必须提供 query_start_loc；query_start_loc、cache_indices、initial_state_mode 和接受数的长度必须与逻辑序列数一致。",
            "initial_state_mode 仅用于 run_mode=0 且元素只能为 0/1；cache_indices 只能选择有效状态槽或 pad_slot_id。",
        ],
        "task": "forward 按 batch/sequence/channel tile 划分，update 按有效 state slot/token 划分；pad_slot_id 不产生输出和状态提交。",
        "tiling_key": "使用模板化三维选择：runModeKey(2)、widthKey(runtime/2/3/4) 和 fnPlanKey(CUTBS/CUTBSD)。实际选择表仅 7 个实例；width 编译期展开用于消除热循环，update 保留 runtime width。B/T/D 不进入 key。",
        "flow": "forward 路径滚动加载 W 个 token、执行 depthwise FMA、可选 SiLU并更新尾状态；update 路径按 cache index 读取/滚动 state，依据接受 mask 提交。",
        "memory": "conv_states 常驻 GM；UB 双缓冲 x/weight/输出 tile，workspace 仅用于 initial-state 同步或 layout 转换。",
        "sync": "每个 state slot 的更新由唯一任务负责；MTE2/V/MTE3 事件保护滚动 buffer，存在 workspace 协作时使用明确 flag，`--cce-auto-sync=off`。",
        "errors": rows(
            ("x/weight/conv_states/y、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID"),
            ("W 不在 2/3/4、D 未按 16 对齐、shape 或 dtype 不一致", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("rank-2 forward 缺 query_start_loc，或元数据长度/值域非法", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("num_accepted_tokens 用于非 update、W!=4 或超出 token 数", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("head_num 用于 update、不能整除 D 或拆分维未按 16 对齐", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
        ),
        "python_sig": "causal_conv1d(x, weight, bias=None, conv_states=None, *, query_start_loc=None, cache_indices=None, initial_state_mode=None, num_accepted_tokens=None, activation_mode=0, pad_slot_id=-1, run_mode=0, head_num=0)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import causal_conv1d

            B, T, D, W = 2, 64, 128, 3
            x = torch.randn(B, T, D, device="npu", dtype=torch.bfloat16)
            weight = torch.randn(W, D, device="npu", dtype=torch.bfloat16)
            bias = torch.randn(D, device="npu", dtype=torch.bfloat16)
            state = torch.zeros(B, W, D, device="npu", dtype=torch.bfloat16)
            y = causal_conv1d(x, weight, bias, state, activation_mode=1, run_mode=0)
            torch.npu.synchronize()
            assert y.shape == x.shape
        """),
        "aclnn_call": "x, weight, bias, convStates, nullptr, nullptr, nullptr, nullptr, activationMode, padSlotId, runMode, headNum, y, &workspaceSize, &executor",
        "runner": "tests/operators/causal_conv1d/accuracy/backend.py",
        "reference": "torch_depthwise_causal_conv1d_reference",
        "direct_template": {"decl": "uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey", "args": "runModeKey, widthKey, fnPlanKey"},
        "case": {
            "shape": {"B": 2, "T": 64, "D": 128, "W": 3, "D_s": 2, "L_s": 3},
            "dtype": {"x_weight_state": "bfloat16", "metadata": "int64"}, "layout": "BSH",
            "attrs": {"activation_mode": 0, "pad_slot_id": -1, "run_mode": 0, "head_num": 0},
            "optional_inputs": {"bias": "present", "query_start_loc": None, "cache_indices": None, "initial_state_mode": None, "num_accepted_tokens": None},
            "variant_shape": {"B": 1, "T": 129, "D": 256, "W": 4, "D_s": 1, "L_s": 4},
            "variant_dtype": {"x_weight_state": "float16", "metadata": "int64"},
            "variant_attrs": {"activation_mode": 1, "pad_slot_id": -1, "run_mode": 0, "head_num": 4}, "variant_layout": "BNSD",
            "tail_shape": {"B": 3, "T": 7, "D": 128, "W": 4, "D_s": 3, "L_s": 3, "Q_a": 3},
            "tail_layout": "TH", "tail_attrs": {"activation_mode": 0, "pad_slot_id": -1, "run_mode": 1, "head_num": 0},
            "tail_optional_inputs": {"bias": None, "query_start_loc": [0, 3, 5, 7], "cache_indices": [0, 1, 2], "initial_state_mode": None, "num_accepted_tokens": [3, 1, 2]},
            "negative_shape": {"B": 1, "T": 64, "D": 130, "W": 3, "D_s": 1, "L_s": 3}, "negative_message": "multiple of 16",
            "extra_cases": [
                {
                    "id": "causal_conv1d_rank2_forward_requires_qsl", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "T": 64, "D": 128, "W": 3, "D_s": 2, "L_s": 2},
                    "dtype": {"x_weight_state": "float16", "metadata": "int64"}, "layout": "TH",
                    "attrs": {"activation_mode": 0, "pad_slot_id": -1, "run_mode": 0, "head_num": 0},
                    "optional_inputs": {"bias": None, "query_start_loc": None, "cache_indices": None, "initial_state_mode": None, "num_accepted_tokens": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "query_start_loc"},
                },
                {
                    "id": "causal_conv1d_accepted_tokens_requires_width4", "tags": ["negative", "boundary", "decode"],
                    "shape": {"B": 2, "T": 2, "D": 128, "W": 3, "D_s": 2, "L_s": 3},
                    "dtype": {"x_weight_state": "float16", "metadata": "int64"}, "layout": "BSH",
                    "attrs": {"activation_mode": 0, "pad_slot_id": -1, "run_mode": 1, "head_num": 0},
                    "optional_inputs": {"bias": None, "query_start_loc": None, "cache_indices": None, "initial_state_mode": None, "num_accepted_tokens": [1, 2]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "width=4"},
                },
            ],
        },
    },
    "causal_conv1d_bwd": {
        "root": "fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd",
        "title": "CausalConv1dBwd",
        "family": "gdn",
        "purpose": "CausalConv1d 的训练反向算子。它支持定长/变长序列和五种公开 layout，根据 x/weight/dy 及可选预激活 y、initial_state、dht 计算 dx/dw/db/dh0。",
        "math": dedent(r"""
            若启用 SiLU/Swish，先得到预激活梯度：

            ```text
            dz = dy                                      activation=0
            dz = dy * (sigmoid(y)+y*sigmoid(y)*(1-sigmoid(y))) activation=1/2
            dx[t-j,d] += dz[t,d] * weight[j,d]
            dw[j,d]   += dz[t,d] * x_or_state[t-j,d]
            db[d]     += sum_t dz[t,d]
            ```

            序列起点以前的 dx 贡献累积到 `dh0`，末状态梯度 `dht` 通过状态滚动关系反传。
        """),
        "inputs": rows(
            ("x", "必选", "[B,T,D] 或 [T,D]", "FP32/FP16/BF16", "逻辑 BSH/TH", "前向输入"),
            ("y", "条件可选", "由 input_layout 决定", "与 x 一致", "BSH/BSND/BNSD/TND/NTD", "预激活；activation=1/2 必选"),
            ("weight", "必选", "[W,D]", "与 x 一致", "ND", "卷积权重"), ("dy", "必选", "与 y 同形", "与 x 一致", "由 input_layout 决定", "上游梯度"),
            ("initial_state", "可选", "[B,W,D]", "与 x 一致", "ND", "前向初始状态"), ("dht", "可选", "[B,W,D]", "与 x 一致", "ND", "末状态梯度"),
            ("query_start_loc", "TND/NTD 必选", "[B+1]", "INT64", "ND", "变长序列边界"),
        ),
        "outputs": rows(("dx", "与逻辑 x 同形", "与 x 一致", "输入梯度"), ("dw", "[W,D]", "与 weight 一致", "权重梯度"),
                           ("db", "[D]", "与 weight 一致", "偏置梯度"), ("dh0", "[B,W,D]", "与 x 一致", "初始状态梯度")),
        "attrs": rows(("activation", "int", "0", "0=无激活，1=SiLU，2=Swish(同 SiLU)"),
                       ("input_layout", "str", "BSND", "BSH/BSND/BNSD/TND/NTD")),
        "layouts": "x/dx 始终逻辑 BSH/TH；y/dy 按 BSH、BSND、BNSD、TND 或 NTD",
        "dtype": "FP32/FP16/BF16，同次调用所有浮点输入一致",
        "modes": "定长/变长序列、无激活/SiLU、可选初末状态",
        "limits": [
            "weight 必须 `[W,D]` 且 W 仅支持 2/3/4；D 或拆分后的 V 必须为 16 的倍数。",
            "activation=1/2 时 y 必须提供且与 dy 同 layout/shape。",
            "TND/NTD 必须提供合法 query_start_loc；累计长度非递减、首项为 0、末项为总 T，且总 token 数必须大于 0。",
            "initial_state/dht/dh0 的逻辑 shape 为 `[B,W,D]`。",
        ],
        "task": "dx 按 token/channel tile 计算，dw/db 按 channel/width 分核并在 workspace 归约；变长序列不跨序列边界读取。",
        "tiling_key": "kernel 入口不按 activation 模板化，固定使用 key=0；activation=0/1/2、layout、W、shape 和切分全部由 tiling data 在运行时选择，因此不存在按 shape 扩张的 key 组合。",
        "flow": "AIV 生成 dz 并处理 layout，随后计算 dx/dh0；dw/db 使用分核局部累加和 workspace 归约，最终转换到输出 dtype。",
        "memory": "workspace 保存各 core 的 dw/db FP32 partial；UB 保存 x/dz/weight tile，initial_state 与 dht 按序列边界加载。",
        "sync": "局部 partial 写完后由归约 owner 读取，跨核阶段由明确计数/flag 管理；核内 MTE/V 事件成对，`--cce-auto-sync=off`。",
        "errors": rows(
            ("x/weight/dy/dx/dw/db/dh0、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID"),
            ("activation 不在 0/1/2，或启用激活时 y 缺失/shape 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("layout 不在 BSH/BSND/BNSD/TND/NTD，或物理 rank/shape 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("W 不在 2/3/4、D/V 未按 16 对齐、浮点 dtype 不一致", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("变长序列缺 query_start_loc 或累计长度非法", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
        ),
        "python_sig": "causal_conv1d_bwd(x, y, weight, dy, initial_state=None, dht=None, *, query_start_loc=None, activation=0, input_layout='BSND')",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import causal_conv1d_bwd

            B, T, D, W = 2, 64, 128, 3
            x = torch.randn(B, T, D, device="npu", dtype=torch.float16)
            weight = torch.randn(W, D, device="npu", dtype=torch.float16)
            y = torch.randn(B, T, 4, D // 4, device="npu", dtype=torch.float16)
            dy = torch.randn_like(y)
            dx, dw, db, dh0 = causal_conv1d_bwd(x, y, weight, dy, activation=1, input_layout="BNSD")
            torch.npu.synchronize()
            assert dx.shape == x.shape and dw.shape == weight.shape
        """),
        "aclnn_call": "x, y, weight, dy, initialState, dht, nullptr, activation, \"BNSD\", dx, dw, db, dh0, &workspaceSize, &executor",
        "runner": "tests/operators/causal_conv1d_bwd/accuracy/backend.py",
        "reference": "torch_autograd_causal_conv1d_reference",
        "case": {
            "shape": {"B": 2, "T": 64, "D": 128, "W": 3}, "dtype": {"floating": "float16", "metadata": "int64"}, "layout": "BSH",
            "attrs": {"activation": 0, "input_layout": "BSH"}, "optional_inputs": {"y": None, "initial_state": None, "dht": None, "query_start_loc": None},
            "variant_shape": {"B": 1, "T": 129, "H_v": 4, "V": 64, "D": 256, "W": 4}, "variant_dtype": {"floating": "bfloat16", "metadata": "int64"},
            "variant_layout": "BNSD", "variant_attrs": {"activation": 1, "input_layout": "BNSD"},
            "variant_optional_inputs": {"y": "present", "initial_state": "present", "dht": "present", "query_start_loc": None},
            "tail_shape": {"B": 2, "T": 193, "H_v": 4, "V": 32, "D": 128, "W": 2}, "tail_layout": "NTD",
            "tail_attrs": {"activation": 2, "input_layout": "NTD"}, "tail_optional_inputs": {"y": "present", "initial_state": "present", "dht": None, "query_start_loc": [0, 65, 193]},
            "negative_attrs": {"activation": 1, "input_layout": "BSND"}, "negative_optional_inputs": {"y": None, "initial_state": None, "dht": None, "query_start_loc": None},
            "negative_message": "y",
            "extra_cases": [
                {
                    "id": "causal_conv1d_bwd_invalid_width", "tags": ["negative", "boundary"],
                    "shape": {"B": 2, "T": 64, "D": 128, "W": 5},
                    "dtype": {"floating": "float16", "metadata": "int64"}, "layout": "BSH",
                    "attrs": {"activation": 0, "input_layout": "BSH"},
                    "optional_inputs": {"y": None, "initial_state": None, "dht": None, "query_start_loc": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "W"},
                },
                {
                    "id": "causal_conv1d_bwd_varlen_requires_offsets", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"N": 2, "T": 193, "H_v": 4, "V": 32, "D": 128, "W": 3},
                    "dtype": {"floating": "bfloat16", "metadata": "int64"}, "layout": "NTD",
                    "attrs": {"activation": 0, "input_layout": "NTD"},
                    "optional_inputs": {"y": None, "initial_state": None, "dht": None, "query_start_loc": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "query_start_loc"},
                },
                {
                    "id": "causal_conv1d_bwd_invalid_state_shape", "tags": ["negative", "boundary"],
                    "shape": {"B": 2, "T": 64, "D": 128, "W": 3},
                    "dtype": {"floating": "float16", "metadata": "int64"}, "layout": "BSH",
                    "attrs": {"activation": 0, "input_layout": "BSH"},
                    "optional_inputs": {"y": None, "initial_state": {"status": "present", "shape": "[B,W-1,D]"}, "dht": None, "query_start_loc": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "initial_state"},
                },
            ],
        },
    },
    "chunk_local_cumsum": {
        "root": "fla/ops/ascendc/gdn/gdn_preprocess/chunk_local_cumsum",
        "title": "ChunkLocalCumsum",
        "family": "gdn",
        "replaces_triton": True,
        "triton_baseline": "fla_npu.ops.triton.chunk_local_cumsum",
        "purpose": "在每个时间 chunk 内对 gate 或任意尾部特征做局部前缀/后缀累加。该 Ascend C 实现替换 Triton 预处理路径，并支持定长/变长序列与 scale。",
        "math": dedent(r"""
            将 token 后的尾部维展平为 P：

            ```text
            out[b,h,t,p] = scale * sum(k=chunk_start..t) g[b,h,k,p]       reverse=false
            out[b,h,t,p] = scale * sum(k=t..chunk_end-1) g[b,h,k,p]       reverse=true
            ```

            变长序列的 chunk_start/chunk_end 在每条逻辑序列内计算，不跨 `cu_seqlens` 边界。
        """),
        "inputs": rows(("g", "必选", "[B,H_v,T] 或 [B,H_v,T,...]", "FP32", "head-first", "待累加 gate/特征"),
                       ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度，Device tensor"),
                       ("chunk_indices_out", "可选", "[N_b,2] 或 [2*N_b]", "INT64", "ND", "变长序列内部处理块映射，Device tensor")),
        "outputs": rows(("out", "与 g 相同", "FP32", "chunk-local 累加结果")),
        "attrs": rows(("chunk_size", "int", "无", "2 的幂"), ("reverse", "bool", "false", "前缀/后缀"),
                       ("scale", "double", "1.0", "输出缩放"), ("head_first", "bool", "true", "当前必须 true"),
                       ("output_dtype", "str", "float32", "仅 float32/torch.float/torch.float32")),
        "layouts": "head-first `[B,H_v,T,...]`",
        "dtype": "输入输出仅 FP32；索引 INT64",
        "modes": "定长/变长序列、forward/reverse、任意连续尾部 P",
        "limits": [
            "g rank 至少 3，所有维度为正；head_first 当前必须 true。",
            "chunk_size 必须为 2 的幂，且结合 P 后能满足 UB tile；交付矩阵覆盖 16/32/64/128。",
            "output_dtype 仅支持 FP32 别名。",
            "变长序列物理 B=1，两个 Device 索引必须同时提供；cu_seqlens 首项为 0、末项为 T 且非递减。",
            "chunk_indices_out 必须按 sequence-major 完整列出内部处理块；处理块长度 B_T 由 UB、chunk_size 和 P 共同决定，不能直接按 chunk_size 构造。",
        ],
        "task": "定长按 `(B*H_v,chunk,tail_tile)` 分配；变长序列按 `(seq_id,local_block_id,head,tail_tile)` 分配，每个处理块内再按 chunk_size 完成 scan。",
        "tiling_key": "未使用多分支 tiling key；reverse、scale、tail P、定长/变长序列都由 tiling data 和 runtime 常量处理。",
        "flow": "MTE2 分段加载一个 chunk/tail tile，Vector 执行顺序或逆序 scan 与 scale，MTE3 写回；长 P 分 tile。",
        "memory": "UB 保存当前 scan tile和必要的 carry；不同 task 不共享输出区，无 user workspace 数据依赖。",
        "sync": "MTE2-V-MTE3 事件按双缓冲槽闭环，同一 scan 由单 core 顺序处理；`--cce-auto-sync=off`。",
        "errors": rows(
            ("g/out、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID"),
            ("g 非 FP32、rank<3、存在非正维或 head_first=false", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("chunk_size 非正 2 的幂，或 output_dtype 非 FP32 别名", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("两个变长序列索引未成对、非 INT64、shape/值/顺序非法或物理 B!=1", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
        ),
        "python_sig": "chunk_local_cumsum(g, chunk_size, *, cu_seqlens=None, chunk_indices_out=None, reverse=False, scale=1.0, head_first=True, output_dtype='float32')",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_local_cumsum

            B, H_v, T, chunk_size = 1, 4, 129, 64
            g = torch.randn(B, H_v, T, device="npu", dtype=torch.float32)
            out = chunk_local_cumsum(g, chunk_size, reverse=False, scale=1.0, head_first=True)
            torch.npu.synchronize()
            assert out.shape == g.shape and out.dtype == torch.float32
        """),
        "aclnn_call": "g, nullptr, nullptr, chunkSize, false, scale, true, const_cast<char *>(\"float32\"), out, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_local_cumsum/accuracy/backend.py",
        "reference": "torch_chunk_local_cumsum_reference",
        "case": {
            "shape": {"B": 2, "H_v": 4, "T": 128, "P": 1, "chunk_size": 64}, "dtype": {"g_out": "float32", "metadata": "int64"}, "layout": "BNS",
            "attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
            "optional_inputs": {"cu_seqlens": None, "chunk_indices_out": None},
            "variant_shape": {"B": 1, "H_v": 8, "T": 257, "P": 16, "chunk_size": 128}, "variant_attrs": {"chunk_size": 128, "reverse": True, "scale": 0.5, "head_first": True, "output_dtype": "torch.float32"},
            "tail_shape": {"B": 1, "N": 2, "H_v": 4, "T": 193, "P": 1, "chunk_size": 64, "N_b": 2},
            "tail_attrs": {"chunk_size": 64, "reverse": True, "scale": 0.5, "head_first": True, "output_dtype": "torch.float32"},
            "tail_optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices_out": [[0, 0], [1, 0]]},
            "negative_attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": False, "output_dtype": "float32"}, "negative_message": "head_first",
            "extra_cases": [
                {
                    "id": "chunk_local_cumsum_varlen_pair_required", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "N": 2, "H_v": 4, "T": 193, "P": 1, "chunk_size": 64},
                    "dtype": {"g_out": "float32", "metadata": "int64"}, "layout": "BNS",
                    "attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
                    "optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices_out": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "provided together"},
                },
                {
                    "id": "chunk_local_cumsum_invalid_dtype", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_v": 4, "T": 128, "P": 1, "chunk_size": 64},
                    "dtype": {"g_out": "float16", "metadata": "int64"}, "layout": "BNS",
                    "attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
                    "optional_inputs": {"cu_seqlens": None, "chunk_indices_out": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "FP32"},
                },
                {
                    "id": "chunk_local_cumsum_non_power_of_two", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_v": 4, "T": 128, "P": 1, "chunk_size": 48},
                    "dtype": {"g_out": "float32", "metadata": "int64"}, "layout": "BNS",
                    "attrs": {"chunk_size": 48, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
                    "optional_inputs": {"cu_seqlens": None, "chunk_indices_out": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "power of two"},
                },
                {
                    "id": "chunk_local_cumsum_varlen_requires_b1", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "N": 2, "H_v": 4, "T": 193, "P": 1, "chunk_size": 64, "N_b": 2},
                    "dtype": {"g_out": "float32", "metadata": "int64"}, "layout": "BNS",
                    "attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
                    "optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices_out": [[0, 0], [1, 0]]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "B=1"},
                },
                {
                    "id": "chunk_local_cumsum_invalid_block_order", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "N": 2, "H_v": 4, "T": 193, "P": 1, "chunk_size": 64, "N_b": 2},
                    "dtype": {"g_out": "float32", "metadata": "int64"}, "layout": "BNS",
                    "attrs": {"chunk_size": 64, "reverse": False, "scale": 1.0, "head_first": True, "output_dtype": "float32"},
                    "optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices_out": [[1, 0], [0, 0]]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "sequence-major"},
                },
            ],
        },
    },
    "chunk_scaled_dot_kkt": {
        "root": "fla/ops/ascendc/gdn/gdn_preprocess/chunk_scaled_dot_kkt",
        "title": "ChunkScaledDotKkt",
        "family": "gdn",
        "replaces_triton": True,
        "triton_baseline": "fla_npu.ops.triton.chunk_scaled_dot_kkt_fwd",
        "purpose": "构造 GDN WY 表示的 chunk-wise 严格下三角 KKT 矩阵。它融合 K@K^T、gate 差指数和 beta 行缩放，输出与 key head 对齐的 A。",
        "math": dedent(r"""
            令 `r=t mod chunk_size`、`s=t-r+c`：

            ```text
            A[b,h_k,t,c] = beta[b,h_k,t] * exp(clip(g[t]-g[s],-50,50))
                            * dot(k[b,h_k,t,:], k[b,h_k,s,:])  if c < r
                            0                                    otherwise
            ```

            输出严格下三角，不含对角线；H_v>H_k 时当前 KKT 阶段读取 g/beta 的前 H_k 个 head。
        """),
        "inputs": rows(("k", "必选", "[B,H_k,T,K]", "FP16/BF16", "BNSD", "Key"),
                       ("g", "必选", "[B,H_v,T]", "FP32", "BNS", "累积 gate"), ("beta", "必选", "[B,H_v,T]", "FP32", "BNS", "行缩放"),
                       ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"), ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "展平 chunk 索引")),
        "outputs": rows(("A", "[B,H_k,T,chunk_size]", "FP32", "严格下三角 scaled KKT")),
        "attrs": rows(("chunk_size", "int", "64", "16/32/64/128")),
        "layouts": "head-first BNSD/BNS",
        "dtype": "K 为 FP16/BF16；g/beta/A 为 FP32",
        "modes": "定长/变长序列、GVA 输入、整块/尾块",
        "limits": [
            "chunk_size 仅支持 16/32/64/128。",
            "必须满足 H_v % H_k == 0；A 的 head 维为 H_k。",
            "cu_seqlens/chunk_indices 必须同时提供或同时省略；变长序列物理 B 必须为 1。",
            "变长序列累计长度必须覆盖 [0,T]，chunk_indices 必须按 sequence-major 完整列出每个长度为 chunk_size 的 chunk。",
            "指数差固定 clip 到 [-50,50]；H_v>H_k 时当前实现读取 g/beta 的前 H_k 个 head。",
        ],
        "task": "按 `(batch,key_head,chunk)` 分配，Cube 计算 KKT，Vector 应用严格下三角、gate 和 beta。",
        "tiling_key": "采用模板化 `D_T_K`(FP16/BF16) 与 `CHUNK_KEY`(16/32/64/128)，共 8 个显式实例。不同 chunk_size 的 Cube/UB tile 必须编译期确定；B/H/T 不进入 key。",
        "flow": "AIC 计算 CxC KKT，AIV 转 FP32、应用 gate/beta 和严格下三角 mask，尾块写有效行并清零其余列。",
        "memory": "workspace 保存 Cube KKT tile，L1/L0 放 K tile，UB 放 FP32 score、gate 差、beta 与 mask。",
        "sync": "AIC 写 KKT 后通知 AIV，AIV 处理完释放槽位；ready/free 协议保护复用，`--cce-auto-sync=off`。",
        "errors": rows(
            ("k/g/beta/A、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID"),
            ("k 非 FP16/BF16，g/beta 非 FP32，或 B/T/H_v shape 不匹配", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("H_v 不能被 H_k 整除，或 chunk_size 不在 16/32/64/128", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
            ("变长序列索引未成对、累计长度/索引顺序非法或物理 B!=1", "ACLNN_ERR_PARAM_INVALID / Python RuntimeError"),
        ),
        "python_sig": "chunk_scaled_dot_kkt(k, g, beta, *, cu_seqlens=None, chunk_indices=None, chunk_size=64)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_scaled_dot_kkt

            B, H_k, H_v, T, K, chunk_size = 1, 2, 4, 129, 128, 64
            k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
            g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
            beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
            A = chunk_scaled_dot_kkt(k, g, beta, chunk_size=chunk_size)
            torch.npu.synchronize()
            assert A.shape == (B, H_k, T, chunk_size) and A.dtype == torch.float32
        """),
        "aclnn_call": "k, g, beta, nullptr, nullptr, chunkSize, A, &workspaceSize, &executor",
        "runner": "tests/operators/chunk_scaled_dot_kkt/accuracy/backend.py",
        "reference": "torch_chunk_scaled_dot_kkt_reference",
        "case": {
            "shape": {"B": 2, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "chunk_size": 64}, "dtype": {"k": "float16", "g_beta_A": "float32"}, "layout": "BNSD",
            "attrs": {"chunk_size": 64}, "optional_inputs": {"cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 1, "H_k": 4, "H_v": 4, "T": 257, "K": 128, "chunk_size": 128}, "variant_dtype": {"k": "bfloat16", "g_beta_A": "float32"}, "variant_attrs": {"chunk_size": 128},
            "tail_shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "chunk_size": 32, "N_c": 7}, "tail_attrs": {"chunk_size": 32},
            "tail_optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 3]},
            "negative_shape": {"B": 1, "H_k": 3, "H_v": 4, "T": 128, "K": 128, "chunk_size": 64}, "negative_message": "divisible by H_k",
            "extra_cases": [
                {
                    "id": "chunk_scaled_dot_kkt_varlen_requires_b1", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "chunk_size": 32, "N_c": 7},
                    "dtype": {"k": "float16", "g_beta_A": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 32},
                    "optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 3]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "physical B=1"},
                },
                {
                    "id": "chunk_scaled_dot_kkt_varlen_pair_required", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 1, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "chunk_size": 32},
                    "dtype": {"k": "float16", "g_beta_A": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 32}, "optional_inputs": {"cu_seqlens": [0, 65, 193], "chunk_indices": None},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "provided together"},
                },
            ],
        },
    },
    "chunk_kda_fwd": {
        "root": "fla/ops/ascendc/kda/chunk_kda_fwd",
        "title": "ChunkKdaFwd",
        "family": "kda",
        "purpose": "Kimi Delta Attention 正向主算子。它消费已经按 chunk 累加的 key gate `gk`，分阶段生成 chunk 内矩阵项、递推状态和最终输出，并可返回完整中间量用于训练链路与精度定位。",
        "math": dedent(r"""
            对每个 value head 映射到对应 key head，在一个 chunk 内定义：

            ```text
            Aqk[i,j] = tril(q_i @ k_j^T * exp2(gk_i-gk_j)) * scale
            Akk      = inv(I + tril((k_i @ k_j^T) * exp2(gk_i-gk_j) * beta_i, -1))
            w        = Akk @ (k * beta * exp2(gk))
            u        = Akk @ (v * beta)
            kg       = k * exp2(-gk)
            v_new    = u - w @ h_prev
            h_next   = exp2(gk_last) * h_prev + kg_state^T @ v_new
            o        = (qg @ h_prev + Aqk @ v_new) * scale
            ```

            `gk` 位于 log2 空间，因此 kernel 以 `exp(x*ln2)` 实现 `exp2(x)`。`final_state`
            固定为 FP32；partial chunk 的补齐行使用中性值参与固定 tile，公开输出的无效区域写零。
        """),
        "inputs": rows(
            ("q", "必选", "按 layout 为 [B,T,H_k,K]/[B,H_k,T,K]/[T,H_k,K]/[H_k,T,K]", "FP16/BF16", "BSND/BNSD/TND/NTD", "Query"),
            ("k", "必选", "与 q 相同", "与 q 相同", "与 q 相同", "Key"),
            ("v", "必选", "对应 [B,T,H_v,V]/[B,H_v,T,V]/[T,H_v,V]/[H_v,T,V]", "与 q 相同", "同 layout", "Value"),
            ("gk", "必选", "与 k 的 token/head/K 维对应", "FP32/BF16", "同 layout", "chunk 内 log2 累积 key gate"),
            ("beta", "必选", "去掉 K 维的 gk shape", "FP32/BF16", "同 layout", "Delta 更新系数"),
            ("initial_state", "可选", "[N,H_v,K,V]", "FP32", "ND", "每条逻辑序列的初始状态"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
            ("chunk_indices", "可选", "[2*N_c]", "INT64", "ND", "sequence-major chunk 二元组"),
        ),
        "outputs": rows(
            ("o", "与 v 相同", "与 v 相同", "KDA 输出"),
            ("final_state", "[N,H_v,K,V] 或空", "FP32", "output_final_state=false 时 Python 返回空 tensor"),
            ("g", "与 gk 相同", "FP32", "Python 返回槽：gk 转 FP32"),
            ("Aqk/Akk", "按 layout 为 [...,T,chunk_size]", "与 q 相同", "chunk 内因果矩阵；内部计算可使用 FP32"),
            ("w/qg/kg", "按 layout 为 [...,T,K]", "与 q 相同", "K 维中间量"),
            ("u/v_new", "与 v 相同", "与 v 相同", "V 维中间量"),
            ("h", "按 layout 为 [B,H_v,N_c,K,V] 或 [B,N_c,H_v,K,V]", "与 q 相同", "每个 chunk 的起始状态"),
            ("initial_state_out", "与 initial_state 相同或空", "FP32", "Python 预留透传槽"),
        ),
        "attrs": rows(
            ("layout", "str", "BSND", "只接受大写 BSND/BNSD/TND/NTD"),
            ("scale", "double", "无", "通常为 1/sqrt(K)"),
            ("chunk_size", "int", "无", "64 或 128"),
            ("output_final_state", "bool", "false", "是否返回有效 final_state"),
            ("return_intermediate", "bool", "false", "是否物化八个中间张量"),
            ("safe_gate", "bool", "false", "预留，当前必须 false"),
            ("transpose_state_layout", "bool", "false", "预留，当前必须 false"),
        ),
        "attr_ranges": {
            "layout": '{"BSND", "BNSD", "TND", "NTD"}',
            "scale": "-",
            "chunk_size": "{64, 128}",
            "output_final_state": "{false, true}",
            "return_intermediate": "{false, true}",
            "safe_gate": "{false}",
            "transpose_state_layout": "{false}",
        },
        "layouts": "BSND/BNSD/TND/NTD；BNSD/NTD 为内部性能布局，BSND/TND 通过 KdaLayoutSwap12 转换",
        "dtype": "q/k/v 为同一 FP16 或 BF16；gk/beta 为 FP32 或 BF16，当前实现在 L2 转 FP32；状态为 FP32",
        "modes": "定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量",
        "limits": [
            "chunk_size 仅支持 64/128；K/V 均须在 [16,256] 且为 16 的倍数；交付矩阵覆盖 K=128、V=128/256。",
            "H_k/H_v 必须在 [1,128] 且 H_v % H_k == 0；TND 仅支持 H_k=1，多 head rank3 使用 NTD。",
            "变长序列的 cu_seqlens 至少含首尾、非递减且末项等于 T；单次最多 1024 条逻辑序列。",
            "显式 chunk_indices 必须完整、合法并严格采用 sequence-major 规范顺序。",
            "safe_gate 与 transpose_state_layout 当前必须为 false；raw gate 应先调用 kda_gate_cumsum。",
        ],
        "architecture_note": dedent("""
            > **架构债务：** 当前 `stage=1/3/2` 与 L2 Cast 是待整改历史实现，不是新增算子的参考架构。
            > 整改目标和参考实现见 [设计文档](docs/design.md#51-算子边界与-l2l0-分工)。
        """).strip(),
        "l2_role": (
            "aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射；当前内部布局中间结果通过 "
            "`ViewCopy` 回写，BSND/TND 外部布局转换调用 `KdaLayoutSwap12`，这些现状均需随架构整改重新收口。"
        ),
        "direct_status": "历史诊断通路，待整改",
        "direct_readme": "`chunk_kda_fwd<<<...>>>`（历史诊断通路，待整改）",
        "architecture": dedent("""
            当前实现通过 `stage=1/3/2` 复用同一 L0，并由 L2 拼接 gate cast、阶段间 cast/scale 和状态 kernel；
            `<<<>>>` 调用者需要理解内部 stage，属于开发规范明确列出的反面样例。整改应优先收敛为一个完整入口下
            的两个语义 phase，在 L0 内闭合必要的全核同步和 `TPipe::Reset()`，并把输入、阶段间和输出 cast
            融入 kernel；若两个 phase 没有共同归并语义或片上复用价值，再拆成两个语义独立的 L0。

            正面结构参考 [ops-nn PR #4803 的 GroupNormSwishGrad A5 实现](https://gitcode.com/cann/ops-nn/pull/4803/diffs)：
            公共 kernel 不暴露 stage，第一 phase 生成输出和 workspace，`pipe.Reset()` 后在第二 phase 消费 workspace
            前执行 `SyncAll()`，再重建 buffer 并在 kernel 内 reduce/cast。KDA 重构必须结合自身参与核和状态依赖
            重新证明同步顺序，不能机械复制。
        """).strip(),
        "task": "stage1/3/2 按 (sequence,value_head,chunk) 分配无跨 chunk 的矩阵任务；状态传播复用 ChunkGatedDeltaRuleFwdH 并保持同一序列的 chunk 顺序。变长序列 tiling 保存每序列起止与累计 chunk offset，不按每个 chunk 膨胀。",
        "tiling_key": "公开接口只设置 key=1，用于选择 AIC:AIV=1:2 的 mixed task kernel 类型；它不编码 B/H/T/chunk_size/layout，也不产生 shape 组合。设备源码中的 key=0/key=2 是历史保留分支，host 已明确不可达。保留 key=1 的原因是当前 Ascend C mixed task 发射需要通过 tiling key 绑定任务类型，不能仅由普通 tiling data 替代。",
        "flow": "L2 先做 contiguous、layout 规范化与 gate cast；stage1 生成 Aqk/Akk/qg/kg/w seed，stage3 完成 Akk@W/U，GDN fwd_h 更新 h/v_new/final_state，stage2 计算 qg@h 与 Aqk@v_new 并合并 o。",
        "memory": "stage1 user workspace 包含每 core 两槽、三 plane 的 score scratch，以及每 core 5 个 chunk_size*chunk_size FP32 solve slot；stage2 使用两个 FP32 output plane。中间 tensor 由 executor 显式持有，不能把后一 stage 读取的数据只作为原地输出参数。",
        "sync": "stage1 的 AIV producer 与 AIC consumer 使用深度 2 的 ready/free 双向 cross-core flag；空 payload 也完成握手，队列排空后才复用 flag。MTE2/V/MTE3、Cube/Fixpipe 生命周期由事件闭环，`--cce-auto-sync=off`。",
        "python_sig": "chunk_kda_fwd(q, k, v, gk, beta, scale, chunk_size, *, layout='BSND', initial_state=None, output_final_state=False, cu_seqlens=None, chunk_indices=None, return_intermediate=False, safe_gate=False, transpose_state_layout=False)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import chunk_kda_fwd

            B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
            q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
            gk = -torch.rand(B, H_v, T, K, device="npu", dtype=torch.float32).cumsum(2)
            beta = torch.sigmoid(torch.randn(B, H_v, T, device="npu", dtype=torch.float32))
            outputs = chunk_kda_fwd(q, k, v, gk, beta, K ** -0.5, chunk_size,
                                    layout="BNSD", output_final_state=True)
            o, final_state = outputs[:2]
            torch.npu.synchronize()
            assert o.shape == v.shape and final_state.dtype == torch.float32
        """),
        "aclnn_call": "q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices, layout, scale, chunkSize, outputFinalState, totalChunks, o, finalState, aqk, akk, w, u, qg, kg, vNew, h, &workspaceSize, &executor",
        "direct_api": dedent("""
            > **架构债务：** 下述 stage launch 仅记录当前历史实现和诊断通路，不满足“`<<<>>>` 对外提供完整算子语义、
            > 不暴露内部 stage、L2 不拼接 Cast”的开发规范。新增算子不得照搬，KDA 后续必须按设计文档整改。

            `chunk_kda_fwd` 是复合算子，完整直调通路必须使用 host 为同一 case 生成的四组 launch 配置，
            并按同一 stream 串行执行以下流水；单独发射一次 `chunk_kda_fwd` 不构成公开算子语义：

            ```cpp
            // 输入已按公共 layout 契约规范化为内部 BNSD/NTD，并完成 contiguous/gate cast。
            chunk_kda_fwd<<<stage1.blockDim, nullptr, stream>>>(/* stage=1 参数、workspace、tiling */);
            ScaleAndCastAqkAndQg(aqkFp32, qg, scale, aqkScaled, qgScaled, stream);
            chunk_kda_fwd<<<stage3.blockDim, nullptr, stream>>>(/* stage=3 参数、workspace、tiling */);
            chunk_gated_delta_rule_fwd_h<<<state.blockDim, nullptr, stream>>>(
                kg, w, u, neutralG, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, state.workspace, state.tiling);
            chunk_kda_fwd<<<stage2.blockDim, nullptr, stream>>>(/* stage=2 参数、workspace、tiling */);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ```

            stage1 生成矩阵与预处理量，缩放/转换步骤与 aclnn L2 完全相同，stage3 生成 `w/u/kg`，
            GDN 状态 kernel 递推 `h/v_new/final_state`，stage2 合成 `o`。四组 `blockDim/workspace/tiling`
            不能混用，`stage` 分别固定为 1、3、GDN 状态阶段和 2。可编译的两个 kernel 原型及单 stage
            launch 包装见 `tests/operators/chunk_kda_fwd/routes/test_direct_chunk_kda_fwd.cpp`；直调执行器还必须
            负责公开 layout 的前后转换和中间张量生命周期。
        """),
        "direct_route": dedent("""
            // Canonical Ascend C stage-launch contract for ChunkKdaFwd.
            // A complete direct route is stage1 -> scale/cast -> stage3 -> state -> stage2.
            // Every launch configuration is derived from tests/op_cases/chunk_kda_fwd.json.
            #include "acl/acl.h"
            #include "kernel_operator.h"

            extern "C" __global__ __aicore__ void chunk_kda_fwd(
                GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initial_state,
                GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR stage_qg, GM_ADDR stage_aqk,
                GM_ADDR stage_v_new, GM_ADDR stage_h, GM_ADDR o, GM_ADDR final_state, GM_ADDR aqk,
                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
                GM_ADDR workspace, GM_ADDR tiling);

            extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(
                GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initial_state,
                GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                GM_ADDR workspace, GM_ADDR tiling);

            void LaunchChunkKdaFwdStage(
                uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
                GM_ADDR beta, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                GM_ADDR stage_qg, GM_ADDR stage_aqk, GM_ADDR stage_v_new, GM_ADDR stage_h, GM_ADDR o,
                GM_ADDR final_state, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
                GM_ADDR kg, GM_ADDR v_new, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
            {
                chunk_kda_fwd<<<blockDim, nullptr, stream>>>(
                    q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, stage_qg, stage_aqk,
                    stage_v_new, stage_h, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                    workspace, tiling);
            }

            void LaunchChunkKdaStateStage(
                uint32_t blockDim, aclrtStream stream, GM_ADDR kg, GM_ADDR w, GM_ADDR u,
                GM_ADDR neutral_g, GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens,
                GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                GM_ADDR workspace, GM_ADDR tiling)
            {
                chunk_gated_delta_rule_fwd_h<<<blockDim, nullptr, stream>>>(
                    kg, w, u, neutral_g, gk, initial_state, cu_seqlens, chunk_indices,
                    h, v_new, final_state, workspace, tiling);
            }
        """),
        "runner": "tests/operators/_shared/chunk_kda_backend.py",
        "reference": "tests/reference/chunk_kda_reference.py",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("q/k/v/gk/beta 或任一必选输出为空", "ACLNN_ERR_PARAM_INVALID（CheckParams 外层映射）"),
            ("layout、rank、shape、dtype、GVA、K/V、chunk_size 或 scale 不合法", "ACLNN_ERR_PARAM_INVALID"),
            ("变长序列累计长度、chunk 顺序、物理 B 或状态 shape 不合法", "ACLNN_ERR_PARAM_INVALID"),
            ("return_intermediate 与中间输出的全有/全无契约不一致", "ACLNN_ERR_PARAM_INVALID"),
            ("执行器创建、内部布局转换或 kernel 执行失败", "ACLNN_ERR_INNER/内部错误码"),
        ),
        "case": {
            "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
            "dtype": {"q_k_v": "float16", "gk_beta_state": "float32"}, "layout": "BNSD",
            "attrs": {"layout": "BNSD", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": True, "return_intermediate": True, "safe_gate": False, "transpose_state_layout": False},
            "optional_inputs": {"initial_state": "present", "cu_seqlens": None, "chunk_indices": None},
            "variant_shape": {"B": 1, "H_k": 4, "H_v": 8, "T": 256, "K": 128, "V": 256, "chunk_size": 128, "N_c": 2},
            "variant_dtype": {"q_k_v": "bfloat16", "gk_beta_state": "float32"}, "variant_layout": "BSND",
            "variant_attrs": {"layout": "BSND", "scale": 0.0883883476, "chunk_size": 128, "output_final_state": True, "return_intermediate": False, "safe_gate": False, "transpose_state_layout": False},
            "tail_shape": {"N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4}, "tail_layout": "NTD",
            "tail_attrs": {"layout": "NTD", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": True, "return_intermediate": True, "safe_gate": False, "transpose_state_layout": False},
            "tail_optional_inputs": {"initial_state": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
            "negative_shape": {"H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64}, "negative_layout": "TND",
            "negative_attrs": {"layout": "TND", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": False, "return_intermediate": False, "safe_gate": False, "transpose_state_layout": False},
            "negative_message": "TND layout with H > 1",
            "extra_cases": [
                {
                    "id": "chunk_kda_fwd_varlen_requires_b1", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
                    "dtype": {"q_k_v": "float16", "gk_beta_state": "float32"}, "layout": "BNSD",
                    "attrs": {"layout": "BNSD", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": True, "return_intermediate": True, "safe_gate": False, "transpose_state_layout": False},
                    "optional_inputs": {"initial_state": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [0, 0, 0, 1, 1, 0, 1, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "physical B=1"},
                },
                {
                    "id": "chunk_kda_fwd_intermediates_all_or_none", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "H_k": 2, "H_v": 4, "T": 128, "K": 128, "V": 128, "chunk_size": 64, "N_c": 2},
                    "dtype": {"q_k_v": "float16", "gk_beta_state": "float32"}, "layout": "BNSD",
                    "attrs": {"layout": "BNSD", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": True, "return_intermediate": "partial", "safe_gate": False, "transpose_state_layout": False},
                    "optional_inputs": {"initial_state": None, "cu_seqlens": None, "chunk_indices": None},
                    "run_on": ["aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "materialized together"},
                },
                {
                    "id": "chunk_kda_fwd_chunk_order", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"N": 2, "H_k": 2, "H_v": 4, "T": 193, "K": 128, "V": 128, "chunk_size": 64, "N_c": 4},
                    "dtype": {"q_k_v": "bfloat16", "gk_beta_state": "float32"}, "layout": "NTD",
                    "attrs": {"layout": "NTD", "scale": 0.0883883476, "chunk_size": 64, "output_final_state": True, "return_intermediate": True, "safe_gate": False, "transpose_state_layout": False},
                    "optional_inputs": {"initial_state": "present", "cu_seqlens": [0, 65, 193], "chunk_indices": [1, 0, 0, 0, 0, 1, 1, 1]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "sequence-major"},
                },
            ],
        },
    },
    "kda_gate_cumsum": {
        "root": "fla/ops/ascendc/kda/kda_gate_cumsum",
        "title": "KdaGateCumsum",
        "family": "kda",
        "purpose": "把 KDA 的逐 token gate 转换为每个 chunk 内的 log2 累积 gate `gk`。既支持已计算好的 step gate，也支持带 A_log/dt_bias 的 safe-gate raw 输入。",
        "math": dedent(r"""
            对每条序列、value head 和 K 维，累加在 chunk 边界重新置零：

            ```text
            step = g                                             use_gate_in_kernel=false
            x    = (g + dt_bias[h_v,k]) * exp(A_log[h_v])        raw safe-gate
            step = lower_bound * sigmoid(x)                      raw safe-gate
            gk[t] = sum(step[chunk_start:t]) / ln(2)
            ```

            因下游使用 `exp2(gk_i-gk_j)`，输出统一为 FP32。safe-gate 的逐步值位于
            `[lower_bound,0]`，合法的长 chunk 累积可达到较大负值，不能通过收窄输入范围掩盖写回或同步问题。
        """),
        "inputs": rows(
            ("g", "必选", "[B,T,H_v,K]/[B,H_v,T,K]/[T,H_v,K]/[H_v,T,K]", "FP16/BF16/FP32", "BSND/BNSD/TND/NTD", "step gate 或 raw gate"),
            ("A_log", "条件必选", "[H_v]", "FP32", "ND", "use_gate_in_kernel=true 时必选"),
            ("dt_bias", "可选", "[H_v*K] 或 [H_v,K]", "FP32", "ND", "safe-gate 偏置"),
            ("cu_seqlens", "可选", "[N+1]", "INT64", "ND", "变长序列累计长度"),
        ),
        "outputs": rows(("gk", "与 g 相同", "FP32", "chunk 内 log2 累积 gate")),
        "attrs": rows(
            ("chunk_size", "int", "无", "32/64/128"),
            ("use_gate_in_kernel", "bool", "false", "是否把 g 当作 raw gate"),
            ("safe_gate", "bool", "false", "raw gate 当前仅支持 true"),
            ("lower_bound", "double", "-5.0", "safe gate 下限，范围 [-5,0)"),
            ("layout", "str", "BSND", "大写 BSND/BNSD/TND/NTD"),
        ),
        "layouts": "rank4 使用 BSND/BNSD，rank3 使用 TND/NTD；layout 必须显式且不根据 shape 推导",
        "dtype": "g 为 FP16/BF16/FP32；A_log/dt_bias/gk 为 FP32；cu_seqlens 为 INT64",
        "modes": "定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块",
        "limits": [
            "K<=256，chunk_size 仅支持 32/64/128。",
            "use_gate_in_kernel=true 时 A_log 必须为 [H_v]、safe_gate 必须 true，dt_bias 若存在须为 [H_v*K] 或 [H_v,K]。",
            "lower_bound 仅支持 [-5,0)；use_gate_in_kernel=false 时 safe_gate 必须 false。",
            "use_gate_in_kernel=false 时 A_log 与 dt_bias 必须为空，避免未消费输入在不同通路产生歧义。",
            "rank4 变长序列物理 B 必须为 1；cu_seqlens 首项为 0、非递减且末项等于 T。",
        ],
        "task": "dense 按 (B,H_v,chunk) 分配，变长序列按 (sequence,H_v) 分配并在 core 内遍历该序列真实 chunk；每个 task 的 K 行按最多 256 元素的 UB 向量处理。",
        "tiling_key": "不使用 tiling key 组合。dataType 与 safeGate 由入口选择 `KdaGateCumsumKernel<T,SAFE_GATE>` 有限模板实例，shape/layout/chunk_size 保留在 tiling data；热循环内没有 dtype 分支。",
        "flow": "每个 task 清零 FP32 acc；逐 token MTE2 加载并转换 g，safe 模板可应用 dt_bias、exp(A_log) 和 sigmoid，再乘 1/ln2 累加，MTE3 写回当前行。chunk/序列切换时 acc 重新置零。",
        "memory": "UB 固定分配 row/acc/tmp/one 各 256 个 FP32、输入类型缓冲和两个 32-byte scalar 缓冲；无 user scratch，GM 输出按 task 不重叠。",
        "sync": "MTE2->V、V->MTE3、MTE3->MTE2 与 MTE3->V 均显式闭环。最后一项尤其保护下一 task 的 Duplicate 不覆盖仍被 MTE3 读取的 acc；`--cce-auto-sync=off`。",
        "python_sig": "kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None, cu_seqlens=None, use_gate_in_kernel=False, safe_gate=False, lower_bound=None, layout='BSND')",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import kda_gate_cumsum

            B, T, H_v, K, chunk_size = 1, 128, 4, 128, 64
            raw = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
            A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
            dt_bias = torch.randn(H_v, K, device="npu", dtype=torch.float32)
            gk = kda_gate_cumsum(raw, chunk_size, A_log=A_log, dt_bias=dt_bias,
                                 use_gate_in_kernel=True, safe_gate=True,
                                 lower_bound=-5.0, layout="BSND")
            torch.npu.synchronize()
            assert gk.shape == raw.shape and gk.dtype == torch.float32
        """),
        "aclnn_call": "g, aLog, dtBias, cuSeqlens, chunkSize, useGateInKernel, safeGate, lowerBound, layout, gk, &workspaceSize, &executor",
        "errors": rows(
            ("必选 tensor、workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("rank/shape/dtype/layout、chunk_size、lower_bound 或模式组合非法", "ACLNN_ERR_PARAM_INVALID"),
            ("step-gate 模式仍传入 A_log/dt_bias", "ACLNN_ERR_PARAM_INVALID"),
            ("Python 输入不是 NPU tensor 或 runtime/op_api 未加载", "RuntimeError"),
        ),
        "runner": "tests/operators/_shared/chunk_kda_backend.py",
        "reference": "_kda_gate_cumsum_reference",
        "case": {
            "shape": {"B": 1, "T": 128, "H_v": 4, "K": 128, "chunk_size": 64}, "dtype": {"g": "float32", "gk": "float32"}, "layout": "BSND",
            "attrs": {"chunk_size": 64, "use_gate_in_kernel": False, "safe_gate": False, "lower_bound": -5.0, "layout": "BSND"},
            "optional_inputs": {"A_log": None, "dt_bias": None, "cu_seqlens": None},
            "variant_shape": {"H_v": 4, "T": 1536, "K": 128, "chunk_size": 64}, "variant_dtype": {"g": "bfloat16", "A_log_dt_bias_gk": "float32"}, "variant_layout": "NTD",
            "variant_attrs": {"chunk_size": 64, "use_gate_in_kernel": True, "safe_gate": True, "lower_bound": -5.0, "layout": "NTD"},
            "variant_optional_inputs": {"A_log": "[H_v]", "dt_bias": "[H_v,K]", "cu_seqlens": None},
            "tail_shape": {"B": 1, "N": 2, "T": 193, "H_v": 4, "K": 128, "chunk_size": 32}, "tail_layout": "BSND",
            "tail_attrs": {"chunk_size": 32, "use_gate_in_kernel": False, "safe_gate": False, "lower_bound": -5.0, "layout": "BSND"},
            "tail_optional_inputs": {"A_log": None, "dt_bias": None, "cu_seqlens": [0, 65, 193]},
            "negative_attrs": {"chunk_size": 64, "use_gate_in_kernel": False, "safe_gate": True, "lower_bound": -5.0, "layout": "BSND"},
            "negative_message": "safeGate only takes effect",
            "extra_cases": [
                {
                    "id": "kda_gate_cumsum_raw_gate_requires_a_log", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "T": 128, "H_v": 4, "K": 128, "chunk_size": 64},
                    "dtype": {"g": "bfloat16", "gk": "float32"}, "layout": "BSND",
                    "attrs": {"chunk_size": 64, "use_gate_in_kernel": True, "safe_gate": True, "lower_bound": -5.0, "layout": "BSND"},
                    "optional_inputs": {"A_log": None, "dt_bias": None, "cu_seqlens": None},
                    "run_on": ["ascendc", "aclnn", "torch_ops_npu"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "A_log is required"},
                },
                {
                    "id": "kda_gate_cumsum_step_rejects_raw_inputs", "tags": ["negative", "boundary", "route"],
                    "shape": {"B": 1, "T": 128, "H_v": 4, "K": 128, "chunk_size": 64},
                    "dtype": {"g": "float32", "A_log_dt_bias_gk": "float32"}, "layout": "BSND",
                    "attrs": {"chunk_size": 64, "use_gate_in_kernel": False, "safe_gate": False, "lower_bound": -5.0, "layout": "BSND"},
                    "optional_inputs": {"A_log": "[H_v]", "dt_bias": "[H_v,K]", "cu_seqlens": None},
                    "run_on": ["ascendc", "aclnn", "torch_ops_npu"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "must be omitted"},
                },
                {
                    "id": "kda_gate_cumsum_varlen_requires_b1", "tags": ["negative", "boundary", "varlen"],
                    "shape": {"B": 2, "N": 2, "T": 193, "H_v": 4, "K": 128, "chunk_size": 32},
                    "dtype": {"g": "float32", "gk": "float32"}, "layout": "BNSD",
                    "attrs": {"chunk_size": 32, "use_gate_in_kernel": False, "safe_gate": False, "lower_bound": -5.0, "layout": "BNSD"},
                    "optional_inputs": {"A_log": None, "dt_bias": None, "cu_seqlens": [0, 65, 193]},
                    "run_on": ["ascendc", "aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "B=1"},
                },
            ],
            "tolerance": {"float16": {"rtol": 0.002, "atol": 0.002}, "bfloat16": {"rtol": 0.003, "atol": 0.003}, "float32": {"rtol": 0.0002, "atol": 0.0002}},
        },
    },
    "kda_layout_swap12": {
        "root": "fla/ops/ascendc/kda/kda_layout_swap12",
        "title": "KdaLayoutSwap12",
        "family": "kda",
        "purpose": "KDA 内部的连续布局转置算子。rank3 交换维 0/1，rank>=4 交换维 1/2，其余维保持顺序；可选 dependency 只建立 executor 调度依赖，不参与数值计算。",
        "math": dedent(r"""
            ```text
            rank(x) = 3: y[j,i,...]   = x[i,j,...]
            rank(x) >= 4: y[b,j,i,...] = x[b,i,j,...]
            ```

            这是精确的数据重排，不执行浮点算术。`dependency` 的 shape/dtype 不改变 y；它仅让 L2
            图显式等待前序生产者，防止临时 tensor 生命周期被 executor 提前复用。
        """),
        "inputs": rows(
            ("x", "必选", "[D_0,D_1,D_2,...]，rank>=3", "FP16/BF16/FP32", "ND", "连续化后参与重排"),
            ("dependency", "可选", "任意 tensor", "任意", "ND", "仅调度依赖，不读取值"),
        ),
        "outputs": rows(("y", "rank3: [D_1,D_0,D_2]；rank>=4: [D_0,D_2,D_1,...]", "与 x 相同", "连续转置结果")),
        "attrs": rows(("无", "-", "-", "无公开属性")),
        "layouts": "ND；rank3 交换维 0/1，rank>=4 交换维 1/2",
        "dtype": "FP16/BF16/FP32，输入输出一致",
        "modes": "rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy",
        "limits": [
            "x rank 必须至少为 3；y rank、dtype 及交换后的每一维必须与 x 精确对应。",
            "输入在 L2 中先 contiguous；当前 API 总是创建连续 y。",
            "dependency 只表达执行顺序，不提供数据、dtype 或 shape 语义，不得依赖其值改变输出。",
            "aclnn/Python 通过 executor 输入依赖排序；`<<<>>>` 直调不读取 dependency，调用者必须在同一 stream 上先发射其生产者。",
            "所有维度必须为正；host 在进入 tiling 前拒绝空维度，kernel 的 usedCoreNum 至少为 1。",
        ],
        "task": "将交换前的 (batch,firstDim,secondDim) 行空间展平后按 core 轮转；当整行 32-byte 对齐且 stride 可编码时，把最多 64 行组合成一次 strided MTE2，再连续 MTE3 写回。",
        "tiling_key": "必须保留 3 个有限 dtype key：0=FP32、1=BF16、2=FP16，因为元素类型决定 DataCopy 长度和 kernel 模板实例。key 不编码 rank、shape 或 dependency，组合固定为 3；这是使用 tiling key 的必要原因。",
        "flow": "host 折叠维 3 之后的 tailDim；kernel 优先 grouped row copy。不能 grouped 时，每行按 8192 元素 UB tile 搬入并写到交换后的 GM offset，非 32-byte 对齐使用 DataCopyPad。",
        "memory": "每 core 仅使用 8192 个元素的 UB copyBuf；无 user scratch。x/y 地址区间由交换映射一一对应，各 core 的输出行不重叠。",
        "sync": "每个 tile 使用 MTE2_MTE3 event 0 和 MTE3_MTE2 event 1，写回完成后才复用 copyBuf。无 Vector 计算和跨核共享区，`--cce-auto-sync=off`。",
        "python_sig": "kda_layout_swap12(x, *, dependency=None)",
        "python_example": dedent("""
            import torch
            from fla_npu.ops.ascendc import kda_layout_swap12

            B, T, H_v, K = 2, 129, 4, 128
            x = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
            y = kda_layout_swap12(x)
            torch.npu.synchronize()
            assert y.shape == (B, H_v, T, K)
            torch.testing.assert_close(y.cpu(), x.permute(0, 2, 1, 3).contiguous().cpu())
        """),
        "aclnn_call": "x, dependency, y, &workspaceSize, &executor",
        "runner": "tests/operators/_shared/chunk_kda_backend.py",
        "reference": "torch_permute_contiguous_reference",
        "errors": rows(
            ("workspaceSize 或 executor 为空", "ACLNN_ERR_PARAM_NULLPTR"),
            ("x 或 y 为空", "ACLNN_ERR_PARAM_INVALID（CheckParams 外层映射）"),
            ("x rank 小于 3、存在空维或 dtype 不在 FP16/BF16/FP32", "ACLNN_ERR_PARAM_INVALID"),
            ("y 的 shape/dtype 不符合维 1/2 交换契约", "ACLNN_ERR_PARAM_INVALID"),
            ("executor 创建、contiguous、内部 op 或 kernel 执行失败", "ACLNN_ERR_INNER/内部错误码"),
        ),
        "case": {
            "shape": {"B": 2, "T": 129, "H_v": 4, "K": 128}, "dtype": {"x_y": "float16"}, "layout": "ND-rank4", "attrs": {}, "optional_inputs": {"dependency": None},
            "variant_shape": {"T": 193, "H_v": 8, "K": 128}, "variant_dtype": {"x_y": "bfloat16"}, "variant_layout": "ND-rank3", "variant_optional_inputs": {"dependency": "present"},
            "tail_shape": {"B": 1, "N_c": 3, "H_v": 4, "K": 17, "V": 7}, "tail_dtype": {"x_y": "float32"}, "tail_layout": "ND-rank5", "tail_optional_inputs": {"dependency": None},
            "negative_shape": {"T": 128, "K": 128}, "negative_layout": "ND-rank2", "negative_message": "rank must be at least 3",
            "extra_cases": [
                {
                    "id": "kda_layout_swap12_empty_dimension", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "T": 0, "H_v": 4, "K": 128},
                    "dtype": {"x_y": "float16"}, "layout": "ND-rank4", "attrs": {},
                    "optional_inputs": {"dependency": None}, "run_on": ["ascendc", "aclnn"],
                    "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "positive"},
                },
                {
                    "id": "kda_layout_swap12_invalid_dtype", "tags": ["negative", "boundary"],
                    "shape": {"B": 1, "T": 128, "H_v": 4, "K": 128},
                    "dtype": {"x_y": "int32"}, "layout": "ND-rank4", "attrs": {},
                    "optional_inputs": {"dependency": None}, "run_on": ["ascendc", "aclnn"],
                    "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "float32, float16 or bfloat16"},
                },
                {
                    "id": "kda_layout_swap12_invalid_output_shape", "tags": ["negative", "boundary", "route"],
                    "shape": {"B": 1, "T": 128, "H_v": 4, "K": 128},
                    "dtype": {"x_y": "float16"}, "layout": "ND-rank4", "attrs": {},
                    "optional_inputs": {"dependency": None}, "output_override": {"y": "[B,T,H_v,K]"},
                    "run_on": ["aclnn"], "reference": "error_contract",
                    "expect": {"return_code": "ACLNN_ERR_PARAM_INVALID", "message_contains": "y shape"},
                },
            ],
            "tolerance": {"float16": {"rtol": 0.0, "atol": 0.0}, "bfloat16": {"rtol": 0.0, "atol": 0.0}, "float32": {"rtol": 0.0, "atol": 0.0}},
        },
    },
})


def table(headers, data):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(item) + " |" for item in data)
    return "\n".join(lines)


def input_table(spec):
    return table(
        ("名称", "必选/可选", "Shape", "Dtype", "Layout", "说明"),
        [(f"`{a}`", b, f"`{c}`", d, e, f) for a, b, c, d, e, f in spec["inputs"]],
    )


def output_table(spec):
    return table(
        ("名称", "Shape", "Dtype", "说明"),
        [(f"`{a}`", f"`{b}`", c, d) for a, b, c, d in spec["outputs"]],
    )


def attr_table(spec):
    attr_ranges = spec.get("attr_ranges")
    if attr_ranges is None:
        return table(
            ("名称", "类型", "默认值", "说明"),
            [(f"`{a}`", b, f"`{c}`", d) for a, b, c, d in spec["attrs"]],
        )
    return table(
        ("名称", "类型", "默认值", "取值范围", "说明"),
        [
            (f"`{a}`", b, f"`{c}`", f"`{attr_ranges[a]}`" if attr_ranges[a] != "-" else "-", d)
            for a, b, c, d in spec["attrs"]
        ],
    )


def has_input(spec, name):
    return any(item[0] == name for item in spec["inputs"])


def model_symbol_link(spec, *, from_docs):
    if spec["family"] == "gdn":
        return "../../../README.md#model-shape-symbols" if from_docs else "../../README.md#model-shape-symbols"
    return "../../README.md#model-shape-symbols" if from_docs else "../README.md#model-shape-symbols"


def varlen_note(spec):
    if not has_input(spec, "cu_seqlens"):
        return ""
    chunk_note = ""
    if has_input(spec, "chunk_indices"):
        chunk_note = (
            "`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；"
            "其条目数和当前调用的 `N_c` 一致。"
        )
    elif has_input(spec, "chunk_indices_out"):
        chunk_note = (
            "`chunk_indices_out` 按 sequence-major 保存内部处理块 `(seq_id, local_block_id)`；"
            "处理块长度 `B_T` 由 tiling 根据 `chunk_size` 和尾部维 `P` 计算，不等同于算子属性 `chunk_size`。"
        )
    return (
        "变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。"
        + chunk_note
        + "定长与变长序列、尾块与整块遵循同一数学定义。"
    )


def api_error_table(spec):
    errors = spec.get("errors")
    if errors is None:
        errors = rows(
            ("必选 tensor、workspaceSize 或 executor 为空", "以 op_api 的 CheckNotNull/外层映射为准"),
            ("rank/shape/dtype/layout 或属性不符合 README", "ACLNN_ERR_PARAM_INVALID"),
            ("Python 输入不是 NPU tensor 或 runtime/op_api 未加载", "RuntimeError"),
        )
    return table(("条件", "返回码/异常"), errors)


def legacy_op_name(op, spec):
    configured = spec.get("legacy_op")
    if configured:
        return configured
    schema = ROOT / "torch_custom" / "fla_npu" / "npu_custom.yaml"
    name = f"npu_{op}"
    if schema.is_file() and re.search(rf"\b{re.escape(name)}\s*\(", schema.read_text(encoding="utf-8")):
        return name
    return None


def aclnn_names(spec):
    return f'aclnn{spec["title"]}GetWorkspaceSize', f'aclnn{spec["title"]}'


def aclnn_signature(op, spec):
    get_name, run_name = aclnn_names(spec)
    root = ROOT / spec["root"]
    candidates = list(root.glob("op_host/op_api/aclnn_*.h")) + list(root.glob("docs/aclnn*.md"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        match = re.search(rf"(?:ACLNN_API\s+)?aclnnStatus\s+{get_name}\s*\(.*?\)\s*;?", text, re.S)
        if match:
            first = re.sub(r"\n\s*", "\n", match.group(0)).strip()
            return first + f"\n\naclnnStatus {run_name}(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);"
    return f"aclnnStatus {get_name}(/* 参数见本页公共参数表 */, uint64_t *workspaceSize, aclOpExecutor **executor);\n\naclnnStatus {run_name}(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);"


def tiling_fields(spec):
    root = ROOT / spec["root"]
    fields = []
    scalar_pattern = re.compile(r"TILING_DATA_FIELD_DEF\(([^,]+),\s*([^\)]+)\);?(?:\s*//\s*(.*))?")
    array_pattern = re.compile(r"TILING_DATA_FIELD_DEF_ARR\(([^,]+),\s*([^,]+),\s*([^\)]+)\);?(?:\s*//\s*(.*))?")
    tiling_headers = list(root.glob("op_host/**/*tiling*.h")) + list(root.glob("op_kernel/**/*tiling*.h"))
    for header in sorted(set(tiling_headers)):
        for line in header.read_text(encoding="utf-8").splitlines():
            array_match = array_pattern.search(line)
            if array_match:
                dtype, count, name, comment = array_match.groups()
                fields.append((f"`{name.strip()}`", f"`{dtype.strip()}[{count.strip()}]`", (comment or "host 计算并由 kernel 消费").strip()))
                continue
            scalar_match = scalar_pattern.search(line)
            if scalar_match:
                dtype, name, comment = scalar_match.groups()
                fields.append((f"`{name.strip()}`", f"`{dtype.strip()}`", (comment or "host 计算并由 kernel 消费").strip()))
    struct_headers = list(root.glob("op_host/**/*tiling*.h")) + list(root.glob("op_kernel/**/*_struct.h"))
    struct_pattern = re.compile(r"struct\s+\w*TilingData\s*\{(.*?)\};", re.S)
    field_pattern = re.compile(
        r"^\s*((?:u?int(?:8|16|32|64)_t)|float|double|bool)\s+(\w+)\s*;\s*(?://\s*(.*))?$",
        re.M,
    )
    for header in sorted(set(struct_headers)):
        text = header.read_text(encoding="utf-8")
        for block in struct_pattern.findall(text):
            for dtype, name, comment in field_pattern.findall(block):
                fields.append((f"`{name}`", f"`{dtype}`", (comment or "host 计算并由 kernel 消费").strip()))
    fields = list(dict.fromkeys(fields))
    if not fields:
        fields = [("`shape/layout fields`", "整数", "输入 shape、任务数、尾块与模式信息"),
                  ("`workspace fields`", "整数", "workspace 偏移和阶段 buffer 大小")]
    return table(("字段", "类型", "含义"), fields)


def memory_lifetime_table(spec):
    configured = spec.get("memory_lifetime")
    if configured:
        return table(("层级/资源", "生命周期与所有权"), configured)
    if "L1/L0" in spec["memory"] or "L0" in spec["memory"]:
        rows_ = [
            ("GM/Workspace", "host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应"),
            ("L1/L0A/L0B/L0C", "当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用"),
            ("UB", "当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放"),
            ("事件/flag", "按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用"),
        ]
    else:
        rows_ = [
            ("GM/Workspace", "输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点"),
            ("UB", "按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放"),
            ("MTE2/V/MTE3 事件", "每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用"),
        ]
    return table(("层级/资源", "生命周期与所有权"), rows_)


def kernel_signature(spec):
    root = ROOT / spec["root"]
    source = next(root.glob("op_kernel/*.cpp"))
    text = source.read_text(encoding="utf-8")
    name = source.stem
    matches = list(re.finditer(rf'(?:extern\s+"C"\s+)?__global__\s+__aicore__\s+void\s+{name}\s*\((.*?)\)\s*\{{', text, re.S))
    if not matches:
        return name, "GM_ADDR tiling", ["tiling"]
    params = matches[-1].group(1)
    params = re.sub(r"//[^\n]*", "", params)
    params = re.sub(r"/\*.*?\*/", "", params, flags=re.S)
    params = re.sub(r"\s+", " ", params).strip()
    names = []
    for param in params.split(","):
        names.append(param.strip().split()[-1].replace("*", ""))
    return name, params, names


def render_readme(op, spec):
    limits = "\n".join(f"- {item}" for item in spec["limits"])
    model_link = model_symbol_link(spec, from_docs=False)
    support_note = varlen_note(spec)
    reference = spec.get("reference", "仓内 PyTorch/CPU reference")
    coverage = spec.get("coverage", spec["modes"])
    architecture_note = spec.get("architecture_note", "")
    if spec.get("replaces_triton"):
        default_performance = (
            f"使用 msopprof 在相同 shape/dtype/layout、warmup 和迭代配置下对比 "
            f"`{spec['triton_baseline']}`；Ascend C 必须更快，仓内主 example 已切换到 Ascend C。"
        )
    else:
        default_performance = (
            "使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。"
        )
    performance = spec.get("performance_target", default_performance)
    legacy_name = legacy_op_name(op, spec)
    legacy_api = f"`torch.ops.npu.{legacy_name}`" if legacy_name else "未实现"
    direct_readme = spec.get("direct_readme", f"`{op}<<<blockDim, l2ctrl, stream>>>(...)`")
    return dedent(f"""
        # {spec['title']}

        ## 1. 功能概述

        {spec['purpose']}

        ## 2. 数学定义

        {spec['math'].strip()}

        ## 3. 输入、输出和属性

        本文使用的 Shape 符号统一引用[{spec['family'].upper()} 模型符号表]({model_link})，不在算子 README 中重复定义。

        ### 3.1 输入

        {input_table(spec)}

        ### 3.2 输出

        {output_table(spec)}

        ### 3.3 属性

        {attr_table(spec)}

        ## 4. 支持范围

        | 项目 | 支持范围 |
        | --- | --- |
        | SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
        | Dtype | {spec['dtype']} |
        | Format/Layout | {spec['layouts']} |
        | 模式 | {spec['modes']} |

        {support_note}

        {architecture_note}

        ## 5. 调用入口

        实现类型：`ascendc`

        | 入口 | API |
        | --- | --- |
        | Python 主入口 | `fla_npu.ops.ascendc.{op}` |
        | aclnn | `{aclnn_names(spec)[0]}` / `{aclnn_names(spec)[1]}` |
        | Ascend C `<<<>>>` | {direct_readme} |
        | legacy（可选） | {legacy_api} |

        表中正式入口必须表达一次完整算子语义，不要求调用者理解或传入内部 stage 编号，类型转换也不由 L2 Cast 组成调用前置步骤；现有实现不满足时必须明确标为架构债务和待整改通路。

        完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

        ## 6. 精度与性能

        - 主精度入口：`tests/operators/{op}/accuracy/test_{op}.py`，主调用使用 `fla_npu.ops.ascendc`。
        - 用例规格：`tests/op_cases/{op}.json`；覆盖 {coverage}。
        - 参考实现：`{reference}`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
        - 性能：{performance}

        ## 7. 已知限制

        {limits}

        ## 8. 构建与验证

        ```bash
        FLA_NPU_SOC=ascend910b FLA_NPU_OPS={op} python -m pip wheel --no-build-isolation --no-deps . -w dist
        pytest -q tests/operators/{op}/accuracy/test_{op}.py
        python scripts/check_operator_compliance.py
        ```

        A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
        `tests/operators/{op}/routes/`，均使用同一份 JSON 规格。
    """).strip() + "\n"


def render_design(op, spec):
    limits = "\n".join(f"- {item}" for item in spec["limits"])
    family = FAMILY_INFO[spec["family"]]
    model_link = model_symbol_link(spec, from_docs=True)
    if has_input(spec, "cu_seqlens"):
        boundary = (
            "定长尾块与变长序列尾段均按每条逻辑序列的有效长度计算，任何补齐元素在参与指数、"
            "矩阵乘或归约前使用中性值或 mask，并按公开输出语义写零。非法累计长度和索引由 host 拦截。"
        )
    else:
        boundary = (
            "边界 shape、空维、非对齐搬运和可选输入组合严格按 README 的已知限制处理；host 在 launch 前拦截"
            "不支持组合，kernel 不用越界读取、静默截断或 fallback 改变公开语义。"
        )
    boundary = spec.get("boundary", boundary)
    precision = spec.get(
        "precision_design",
        "FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。"
        "逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。",
    )
    if spec.get("replaces_triton"):
        default_performance = (
            f"以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待；"
            f"在相同输入和测量配置下逐项对比 `{spec['triton_baseline']}`，Ascend C 必须更快。"
        )
    else:
        default_performance = (
            "以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，"
            "并对比当前主线 Ascend C 基线检查性能回退。"
        )
    performance = spec.get("performance_design", default_performance)
    architecture = spec.get(
        "architecture",
        (
            "公共接口只表达完整算子语义，不暴露内部 stage 编号。强生产消费关系可在一个 L0 kernel 内拆成"
            "语义 phase，并在阶段边界闭合异步生命周期、`TPipe::Reset()` 和必要的 `SyncAll`；只有 GM 数据依赖且"
            "没有共同归并语义或片上复用价值的计算拆成独立 L0。L2 只负责编排、校验和 workspace/executor，"
            "输入、阶段间及输出 cast 均在 kernel 内完成。"
        ),
    )
    l2_role = spec.get(
        "l2_role",
        "aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射。",
    )
    return dedent(f"""
        # {spec['title']} 设计方案

        ## 1. 背景

        {spec['purpose']} 本实现通过统一 aclnn 与 ctypes 稳定入口接入 {family['name']} 链路。

        ## 2. 目标与非目标

        ### 2.1 目标

        - README 所列 `{spec['modes']}` 场景精度与仓内参考实现一致。
        - A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
        - 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
        - aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

        ### 2.2 非目标

        - 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
        - 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

        ## 3. 能力边界

        实现类型：`ascendc`。Dtype：{spec['dtype']}。Layout：{spec['layouts']}。模式：{spec['modes']}。
        Shape 符号统一引用[{spec['family'].upper()} 模型符号表]({model_link})。

        ## 4. 数学与接口语义

        {spec['math'].strip()}

        Python、aclnn 与直调入口均以同一逻辑 shape 解释输入。接口层转换只处理连续性、描述符或文档明确的
        layout 规范化，不改变公式、边界 mask、head 映射或可选输入语义。

        ## 5. 整体架构

        1. `op_host/*_def.cpp` 注册输入输出、dtype、属性和 A2/A3/A5。
        2. InferShape、op_api 与 tiling host 共同按 README 校验必选参数、shape、dtype、layout、属性和可选输入组合，并构造或核对输出。
        3. tiling processor 计算任务数、边界块、workspace 偏移和模板实例。
        4. `op_kernel/` 按本算子的计算流程完成搬运、计算、同步和写回。
        5. {l2_role}
        6. `fla_npu.ops.ascendc` 仅通过 ctypes 调用 aclnn，不依赖 torch_npu dispatcher。

        ### 5.1 算子边界与 L2/L0 分工

        {architecture}

        ## 6. Tiling 设计

        ### 6.1 任务划分

        {spec['task']}

        ### 6.2 Tiling Data

        {tiling_fields(spec)}

        ### 6.3 模板化方案与 tiling key

        {spec['tiling_key']}

        ## 7. Kernel 设计

        ### 7.1 计算流程

        {spec['flow']}

        ### 7.2 内存规划

        {spec['memory']}

        {memory_lifetime_table(spec)}

        ### 7.3 流水与同步

        {spec['sync']}

        ### 7.4 边界处理

        {boundary}

        ## 8. 平台设计

        | 平台 | SOC | 路径 | 验证要求 |
        | --- | --- | --- | --- |
        | A2 | `ascend910b` | 公共实现 | 构建、全量精度、性能、通路 |
        | A3 | `ascend910_93` | 公共实现 | 构建、全量精度、性能、通路 |
        | A5 | `ascend950` | `arch35/` 存在时使用特化，否则公共实现 | 构建、全量精度、性能、通路 |

        平台差异不得改变公开 shape/dtype/layout 语义；若某模板在平台上不可用，应在 host 明确报错并同步更新 README 与 JSON。

        ## 9. 精度设计

        {precision} 参考实现与阈值由 `tests/op_cases/{op}.json` 固定。

        ## 10. 性能设计

        {performance} 不得使用 Python wall time 下结论。

        ## 11. 测试设计

        `tests/op_cases/{op}.json` 是唯一 case 规格。`tests/operators/{op}/accuracy/` 运行主精度与泛化矩阵，
        `routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
        A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

        ## 12. 已知限制与演进计划

        {limits}

        后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
    """).strip() + "\n"


def legacy_example_code(op, spec):
    legacy_name = legacy_op_name(op, spec)
    if legacy_name is None:
        return None
    code = spec["python_example"].strip()
    stable_import = f"from fla_npu.ops.ascendc import {op}"
    code = code.replace(stable_import, "import fla_npu")
    code = code.replace(
        "import fla_npu\n",
        "import fla_npu\n\nfla_npu.load_legacy_torch_ops()\n",
        1,
    )
    return re.sub(
        rf"\b{re.escape(op)}\(",
        f"torch.ops.npu.{legacy_name}(",
        code,
    )


def render_api(op, spec):
    get_name, run_name = aclnn_names(spec)
    kernel_name, _, kernel_args = kernel_signature(spec)
    direct_args = ", ".join(kernel_args)
    direct_name = f"{kernel_name}<{spec['direct_template']['args']}>" if "direct_template" in spec else kernel_name
    limits = "\n".join(f"- {item}" for item in spec["limits"])
    legacy_name = legacy_op_name(op, spec)
    legacy_status = "支持（显式加载）" if legacy_name else "未实现"
    direct_status = spec.get("direct_status", "支持")
    model_link = model_symbol_link(spec, from_docs=True)
    if legacy_name:
        legacy_code = legacy_example_code(op, spec)
        legacy_example = f"```python\n{legacy_code}\n```"
    else:
        legacy_example = "当前未注册 `torch.ops.npu` 入口；调用方使用 `fla_npu.ops.ascendc`。"
    direct_api = spec.get("direct_api")
    if direct_api is None:
        direct_api = dedent(f"""
            `blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

            ```cpp
            {direct_name}<<<blockDim, nullptr, stream>>>({direct_args});
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ```

            可编译的参数声明和 launch 包装位于 `tests/operators/{op}/routes/test_direct_{op}.cpp`。
        """).strip()
    return dedent(f"""
        # {spec['title']} API 与调用示例

        ## 1. API 总览

        | 通路 | API/入口 | 支持情况 |
        | --- | --- | --- |
        | Python 主入口 | `fla_npu.ops.ascendc.{op}` | 支持 |
        | aclnn | `{get_name}` / `{run_name}` | 支持 |
        | Ascend C `<<<>>>` | `{kernel_name}<<<blockDim, nullptr, stream>>>` | {direct_status} |
        | legacy | `{legacy_name or '-'}` | {legacy_status} |

        表中标记为“支持”的正式入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义；待整改诊断通路不据此宣称已经满足完整公开语义。

        公共 API 应只表达完整算子语义，不要求调用者传入或理解内部 stage 编号；类型转换应由 kernel 完成，不由 L2 Cast 组成调用前置步骤。现有实现不满足时，必须像本页一样明确标为架构债务和待整改通路。

        ## 2. 公共参数与约束

        Shape 符号统一引用[{spec['family'].upper()} 模型符号表]({model_link})。

        ### 2.1 输入

        {input_table(spec)}

        ### 2.2 输出

        {output_table(spec)}

        ### 2.3 属性

        {attr_table(spec)}

        ## 3. aclnn API

        ### 3.1 接口签名

        ```cpp
        {aclnn_signature(op, spec)}
        ```

        `GetWorkspaceSize` 完成校验并创建 executor；第二段在传入 stream 上异步执行。输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

        ### 3.2 调用示例

        ```cpp
        int32_t deviceId = 0;
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(deviceId));
        aclrtStream stream = nullptr;
        ACL_CHECK(aclrtCreateStream(&stream));

        // 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus status = {get_name}(
            {spec['aclnn_call']});
        ACLNN_CHECK(status);
        void *workspace = nullptr;
        if (workspaceSize != 0) {{
            ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }}
        ACLNN_CHECK({run_name}(workspace, workspaceSize, executor, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (workspace != nullptr) {{
            ACL_CHECK(aclrtFree(workspace));
        }}
        // 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        ```

        `tests/operators/{op}/routes/test_aclnn_{op}.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
        参数表构造，不能用物理补齐 shape 替代逻辑 shape。

        ## 4. `fla_npu.ops.ascendc` API

        ### 4.1 接口签名

        ```python
        {spec['python_sig']}
        ```

        稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

        ### 4.2 调用示例

        ```python
        {spec['python_example'].strip()}
        ```

        ## 5. Ascend C `<<<>>>` 直调

        {direct_api}

        ## 6. `torch.ops.npu` API（可选）

        {legacy_example}

        ## 7. 平台支持

        | 平台 | SOC | 状态 |
        | --- | --- | --- |
        | A2 | `ascend910b` | 支持 |
        | A3 | `ascend910_93` | 支持 |
        | A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

        ## 8. 已知限制

        {limits}

        ## 9. 异常与返回码

        {api_error_table(spec)}

        负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/{op}.json`，修改拦截时必须同步更新。

        ## 10. 文档自检

        - [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
        - [x] Shape 使用模型符号，固定值仅列在已知限制。
        - [x] A2/A3/A5、{spec['modes']} 与错误码均有说明。
        - [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
    """).strip() + "\n"


def render_test_readme(op, spec):
    legacy_name = legacy_op_name(op, spec)
    legacy_line = (
        f"- legacy 通路：`torch.ops.npu.{legacy_name}`，由主 route case 验证显式加载。"
        if legacy_name
        else "- legacy 通路：未实现，不生成 `torch.ops.npu` 测试。"
    )
    return dedent(f"""
        # {spec['title']} 测试归档

        ## 1. 唯一用例规格

        `tests/op_cases/{op}.json` 统一保存 shape、dtype、layout、属性、可选输入、SOC、运行通路、随机种子、
        参考实现、容差和预期返回码。新 case 必须先进入 JSON，再由执行端按 case ID 消费。

        ## 2. 归档内容

        | 路径 | 内容 |
        | --- | --- |
        | `common/case_matrix.py` | 本算子 JSON 加载、tag/route 筛选和 case ID 环境变量 |
        | `accuracy/test_{op}.py` | `fla_npu.ops.ascendc` 主精度、泛化、边界和回归入口 |
        | `routes/test_aclnn_{op}.cpp` | aclnn 两段式接口签名、workspace/executor/stream 契约 |
        | `routes/test_direct_{op}.cpp` | host tiling 结果驱动的 `<<<>>>` 参数和 launch 契约 |
        | `ut/op_host/test_contract.py` | manifest、SOC、返回码、host 负向用例静态契约 |
        | `ut/op_kernel/test_contract.py` | kernel 入口、tiling key 说明和 direct launch 静态契约 |
        | `performance/profile.py` | 读取 performance tag 并通过 msopprof 运行设备侧 profiling |
        | `st/test_example.py` | example tag 与仓内数值执行后端的 ST 入口 |

        {legacy_line}

        现有数值/reference 后端：`{spec['runner']}`。该后端由 canonical 入口传入
        `FLA_NPU_CASE_MANIFEST`、`FLA_NPU_CASE_IDS` 和 `FLA_NPU_OPERATOR`；关键 shape、dtype、属性组合不在
        canonical 脚本中重复定义。

        ## 3. 执行命令

        ```bash
        pytest -q tests/operators/{op}/accuracy/test_{op}.py
        FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/{op}/accuracy/test_{op}.py
        FLA_NPU_CASE_TAGS=generalization FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/{op}/accuracy/test_{op}.py
        pytest -q tests/operators/{op}/ut
        python tests/operators/{op}/performance/profile.py --dry-run
        FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/{op}/st/test_example.py
        ```

        A2/A3/A5 通过 `FLA_NPU_SOC` 选择。精度逐项比较全部公开输出并检查 NaN/Inf；性能只使用 msopprof
        设备侧结果，按 JSON 的 `expect.requirement` 对比 Triton 或当前主线基线。报告记录平台、case 总数、通过数和
        失败 case ID，不记录本地环境路径。
    """).strip() + "\n"


def render_case_matrix(op):
    return dedent(f'''\
        """JSON selectors shared by {op} tests."""

        from __future__ import annotations

        import os

        from tests.operators._shared.cases import load_cases, select_cases


        OP = "{op}"


        def manifest():
            return load_cases(OP)


        def cases(*, tag=None, route=None):
            old_tags = os.environ.get("FLA_NPU_CASE_TAGS")
            if tag is not None:
                os.environ["FLA_NPU_CASE_TAGS"] = tag
            try:
                return select_cases(OP, route=route)
            finally:
                if tag is not None:
                    if old_tags is None:
                        os.environ.pop("FLA_NPU_CASE_TAGS", None)
                    else:
                        os.environ["FLA_NPU_CASE_TAGS"] = old_tags


        def case_ids(*, tag=None, route=None):
            return [case["id"] for case in cases(tag=tag, route=route)]
    ''')


def render_op_host_contract(op):
    return dedent(f'''\
        """Static op_host contract for {op}; device execution lives in accuracy/routes."""

        from tests.operators.{op}.common.case_matrix import manifest


        def test_host_contract_has_platform_and_negative_matrix():
            data = manifest()
            assert set(data["capability"]["soc"]) >= {{"ascend910b", "ascend910_93", "ascend950"}}
            negatives = [case for case in data["cases"] if "negative" in case["tags"]]
            assert negatives
            for case in negatives:
                assert case["expect"]["return_code"] != "ACLNN_SUCCESS"
                assert case["expect"].get("message_contains")
                assert "aclnn" in case["run_on"] or case["expect"]["return_code"] == "RuntimeError"


        def test_route_case_uses_one_shape_definition():
            data = manifest()
            route_cases = [case for case in data["cases"] if "route" in case["tags"]]
            assert route_cases
            assert any({{"ascendc", "aclnn", "direct_launch"}} <= set(case["run_on"]) for case in route_cases)
    ''')


def render_op_kernel_contract(op, spec):
    root = spec["root"]
    return dedent(f'''\
        """Static kernel/tiling contract for {op}."""

        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[5]
        OP_ROOT = ROOT / "{root}"


        def test_direct_launch_matches_a_real_kernel_entry():
            direct = ROOT / "tests/operators/{op}/routes/test_direct_{op}.cpp"
            kernel_sources = list((OP_ROOT / "op_kernel").glob("*.cpp"))
            assert direct.is_file() and kernel_sources
            text = direct.read_text(encoding="utf-8")
            assert "<<<blockDim" in text and "workspace" in text and "tiling" in text
            assert any("__global__" in source.read_text(encoding="utf-8") for source in kernel_sources)


        def test_tiling_key_has_design_rationale():
            design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
            assert "模板化方案与 tiling key" in design
            assert "组合" in design or "不使用 tiling key" in design
    ''')


def render_performance_runner(op, spec):
    runner = spec["runner"]
    return dedent(f'''\
        #!/usr/bin/env python3
        """Profile JSON performance cases for {op} with msopprof."""

        from __future__ import annotations

        import argparse
        import os
        import shlex
        import subprocess
        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[4]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from tests.operators.{op}.common.case_matrix import case_ids


        RUNNER = ROOT / "{runner}"


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--dry-run", action="store_true")
            parser.add_argument("--output", default="outputs/{op}_msopprof")
            args = parser.parse_args()
            ids = case_ids(tag="performance", route="ascendc")
            if not ids:
                raise RuntimeError("no performance case is defined")
            env = os.environ.copy()
            env.update({{
                "FLA_NPU_OPERATOR": "{op}",
                "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/{op}.json"),
                "FLA_NPU_CASE_IDS": ",".join(ids),
            }})
            application = f"{{shlex.quote(sys.executable)}} {{shlex.quote(str(RUNNER))}}"
            command = [
                "msopprof", f"--application={{application}}", f"--output={{args.output}}",
                "--aic-metrics=BasicInfo", "--launch-count=20", "--warm-up=5", "--kill=off",
            ]
            if args.dry_run:
                print(" ".join(shlex.quote(part) for part in command))
                return
            subprocess.run(command, cwd=RUNNER.parent, env=env, check=True)


        if __name__ == "__main__":
            main()
    ''')


def render_st_test(op, spec):
    runner = spec["runner"]
    return dedent(f'''\
        """Example-tagged ST entry for {op}."""

        from __future__ import annotations

        import os
        import subprocess
        import sys
        from pathlib import Path

        import pytest

        ROOT = Path(__file__).resolve().parents[4]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from tests.operators.{op}.common.case_matrix import case_ids


        RUNNER = ROOT / "{runner}"


        @pytest.mark.npu
        def test_example_cases():
            if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
                pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
            ids = case_ids(tag="example", route="ascendc")
            assert ids and RUNNER.is_file()
            env = os.environ.copy()
            env.update({{
                "FLA_NPU_OPERATOR": "{op}",
                "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/{op}.json"),
                "FLA_NPU_CASE_IDS": ",".join(ids),
            }})
            subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
    ''')


def render_legacy_route(op, spec):
    legacy_name = legacy_op_name(op, spec)
    code = legacy_example_code(op, spec)
    if legacy_name is None or code is None:
        raise ValueError(f"{op} does not implement a legacy route")
    return dedent(f'''\
        """Explicit torch.ops.npu route for {op}."""

        import os

        import pytest

        from tests.operators.{op}.common.case_matrix import case_ids
        from tests.operators._shared.route_requirements import require_legacy_route

        require_legacy_route()


        @pytest.mark.npu
        def test_legacy_route_case():
            if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
                pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
            assert case_ids(route="torch_ops_npu")
{indent(code, '            ')}
    ''')


def render_accuracy_test(op, spec):
    return dedent(f'''\
        """Canonical JSON entry for {spec['title']} accuracy/generalization tests."""

        from __future__ import annotations

        import os
        import subprocess
        import sys
        from pathlib import Path

        import pytest

        from tests.operators._shared.cases import load_cases, select_cases
        from tests.operators._shared.npu_generalization import run_generalization_cases


        OP = "{op}"
        ROOT = Path(__file__).resolve().parents[4]
        RUNNER = ROOT / "{spec['runner']}"


        def test_case_manifest_covers_required_matrix():
            manifest = load_cases(OP)
            cases = manifest["cases"]
            tags = {{tag for case in cases for tag in case["tags"]}}
            assert {{"accuracy", "generalization", "boundary", "negative", "route", "performance", "example"}} <= tags
            assert {{"ascend910b", "ascend910_93", "ascend950"}} <= set(manifest["capability"]["soc"])
            assert all("ascendc" in case["run_on"] for case in cases if "accuracy" in case["tags"])


        def test_selected_case_ids_are_unique():
            selected = select_cases(OP)
            ids = [case["id"] for case in selected]
            assert len(ids) == len(set(ids))


        @pytest.mark.npu
        def test_json_generalization_cases():
            if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
                pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
            cases = select_cases(
                OP,
                tags=("generalization",),
                route="ascendc",
                include_negative=False,
            )
            assert cases, f"{{OP}} has no executable generalization cases"
            run_generalization_cases(OP, cases)


        @pytest.mark.npu
        def test_main_ascendc_accuracy_backend():
            if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
                pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
            assert RUNNER.is_file(), RUNNER
            env = os.environ.copy()
            env["FLA_NPU_OPERATOR"] = OP
            env["FLA_NPU_CASE_MANIFEST"] = str(ROOT / "tests" / "op_cases" / f"{{OP}}.json")
            accuracy_cases = select_cases(OP, tags=("accuracy",), route="ascendc", include_negative=False)
            env["FLA_NPU_CASE_IDS"] = ",".join(case["id"] for case in accuracy_cases)
            subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
    ''')


def render_manifest(op, spec):
    case = spec["case"]
    all_socs = ["ascend910b", "ascend910_93", "ascend950"]
    success = {"return_code": "ACLNN_SUCCESS"}
    main_routes = ["ascendc", "aclnn", "direct_launch"]
    if legacy_op_name(op, spec):
        main_routes.append("torch_ops_npu")
    performance_reference = spec.get("triton_baseline", "current_mainline_ascendc_profile")
    performance_requirement = "faster_than_triton" if spec.get("replaces_triton") else "no_regression"
    cases = [
        {
            "id": f"{op}_main_accuracy",
            "tags": ["accuracy", "example", "route"],
            "shape": case["shape"],
            "dtype": case["dtype"],
            "layout": case["layout"],
            "attrs": case["attrs"],
            "optional_inputs": case.get("optional_inputs", {}),
            "soc": all_socs,
            "run_on": main_routes,
            "reference": spec.get("reference", "torch_cpu_reference"),
            "expect": success,
        },
        {
            "id": f"{op}_dtype_layout_generalization",
            "tags": ["accuracy", "generalization", "boundary"],
            "shape": case.get("variant_shape", case["shape"]),
            "dtype": case.get("variant_dtype", case["dtype"]),
            "layout": case.get("variant_layout", case["layout"]),
            "attrs": case.get("variant_attrs", case["attrs"]),
            "optional_inputs": case.get("variant_optional_inputs", case.get("optional_inputs", {})),
            "soc": all_socs,
            "run_on": ["ascendc"],
            "reference": spec.get("reference", "torch_cpu_reference"),
            "expect": success,
        },
        {
            "id": f"{op}_tail_or_varlen",
            "tags": ["accuracy", "generalization", "boundary", "tail"],
            "shape": case.get("tail_shape", case.get("variant_shape", case["shape"])),
            "dtype": case.get("tail_dtype", case.get("variant_dtype", case["dtype"])),
            "layout": case.get("tail_layout", case.get("variant_layout", case["layout"])),
            "attrs": case.get("tail_attrs", case.get("variant_attrs", case["attrs"])),
            "optional_inputs": case.get("tail_optional_inputs", case.get("optional_inputs", {})),
            "soc": all_socs,
            "run_on": ["ascendc"],
            "reference": spec.get("reference", "torch_cpu_reference"),
            "expect": success,
        },
        {
            "id": f"{op}_performance",
            "tags": ["performance", "route"],
            "shape": case.get("perf_shape", case["shape"]),
            "dtype": case.get("perf_dtype", case["dtype"]),
            "layout": case.get("perf_layout", case["layout"]),
            "attrs": case.get("perf_attrs", case["attrs"]),
            "optional_inputs": case.get("perf_optional_inputs", case.get("optional_inputs", {})),
            "soc": all_socs,
            "run_on": ["ascendc", "direct_launch"],
            "reference": performance_reference,
            "expect": {
                "return_code": "ACLNN_SUCCESS",
                "metric": "msopprof",
                "requirement": performance_requirement,
            },
        },
        {
            "id": f"{op}_negative_contract",
            "tags": ["negative", "boundary"],
            "shape": case.get("negative_shape", case["shape"]),
            "dtype": case.get("negative_dtype", case["dtype"]),
            "layout": case.get("negative_layout", case["layout"]),
            "attrs": case.get("negative_attrs", case["attrs"]),
            "optional_inputs": case.get("negative_optional_inputs", case.get("optional_inputs", {})),
            "soc": all_socs,
            "run_on": ["ascendc", "aclnn"],
            "reference": "error_contract",
            "expect": {
                "return_code": case.get("negative_code", "ACLNN_ERR_PARAM_INVALID"),
                "message_contains": case.get("negative_message", "invalid"),
            },
        },
    ]
    for extra in case.get("extra_cases", []):
        item = dict(extra)
        item.setdefault("soc", all_socs)
        item.setdefault("run_on", ["ascendc"])
        item.setdefault("reference", spec.get("reference", "torch_cpu_reference"))
        item.setdefault("expect", success)
        if "negative" in item.get("tags", ()):
            item["expect"] = dict(item["expect"])
            item["expect"].setdefault("return_code", "ACLNN_ERR_PARAM_INVALID")
            item["expect"].setdefault("message_contains", "invalid")
            if item["expect"]["return_code"] != "RuntimeError" and "aclnn" not in item["run_on"]:
                item["run_on"] = [*item["run_on"], "aclnn"]
        cases.append(item)
    for index, item in enumerate(cases):
        item.setdefault("seed", 2026 + index)
        if "accuracy" in item.get("tags", ()):
            item["expect"] = dict(item["expect"])
            item["expect"].setdefault("finite_outputs", True)
            item["expect"].setdefault("compare_outputs", [output[0] for output in spec["outputs"]])

    capability_dtypes = set()
    capability_layouts = set()
    for item in cases:
        if "negative" in item.get("tags", ()):
            continue
        dtype = item.get("dtype")
        if isinstance(dtype, dict):
            capability_dtypes.update(str(value) for value in dtype.values())
        elif dtype is not None:
            capability_dtypes.add(str(dtype))
        capability_layouts.add(str(item.get("layout")))
    route_case_ids = [item["id"] for item in cases if "route" in item["tags"]]
    accuracy_case_ids = [item["id"] for item in cases if "accuracy" in item["tags"]]
    negative_case_ids = [item["id"] for item in cases if "negative" in item["tags"]]
    performance_case_ids = [item["id"] for item in cases if "performance" in item["tags"]]
    manifest = {
        "schema_version": 1,
        "op": op,
        "implementation": "ascendc",
        "model": spec["family"],
        "symbol_table_version": f"{spec['family']}-shape-v1",
        "capability": {
            "soc": all_socs,
            "dtypes": sorted(capability_dtypes),
            "layouts": sorted(capability_layouts),
            "features": mode_features(spec),
            "routes": ["ascendc", "aclnn", "direct_launch"],
            "optional_routes": ["torch_ops_npu"] if legacy_op_name(op, spec) else [],
        },
        "data_generation": spec.get(
            "data_generation",
            {
                "floating_inputs": {"distribution": "uniform", "min": -1.0, "max": 1.0},
                "positive_inputs": {"distribution": "uniform", "min": 0.0, "max": 1.0},
                "structured_inputs": "由 reference 名称对应的 runner 构造 gate、三角矩阵、状态和索引",
            },
        ),
        "coverage_requirements": {
            "public_modes": mode_features(spec),
            "known_limits": spec["limits"],
            "required_outputs": [output[0] for output in spec["outputs"]],
            "accuracy_case_ids": accuracy_case_ids,
            "generalization_case_ids": [item["id"] for item in cases if "generalization" in item["tags"]],
            "negative_case_ids": negative_case_ids,
            "route_case_ids": route_case_ids,
            "performance_case_ids": performance_case_ids,
        },
        "tolerance": case.get(
            "tolerance",
            {"float16": {"rtol": 0.003, "atol": 0.003}, "bfloat16": {"rtol": 0.006, "atol": 0.006}},
        ),
        "cases": cases,
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def enrich_existing_manifest(op, spec, manifest):
    """Add the common schema to hand-authored matrices without replacing their cases."""
    cases = manifest["cases"]
    all_socs = ["ascend910b", "ascend910_93", "ascend950"]
    performance_requirement = "faster_than_triton" if spec.get("replaces_triton") else "no_regression"
    for index, item in enumerate(cases):
        item.setdefault("seed", 2026 + index)
        item.setdefault("soc", all_socs)
        if "accuracy" in item.get("tags", ()):
            item["expect"] = dict(item["expect"])
            item["expect"].setdefault("finite_outputs", True)
            item["expect"].setdefault("compare_outputs", [output[0] for output in spec["outputs"]])
        if "negative" in item.get("tags", ()):
            item["expect"].setdefault("message_contains", "invalid")
        if "performance" in item.get("tags", ()):
            item["expect"].setdefault("metric", "msopprof")
            item["expect"].setdefault("requirement", performance_requirement)
            item["expect"].setdefault(
                "baseline",
                spec.get("triton_baseline", "current_mainline_ascendc_profile"),
            )

    capability = manifest.setdefault("capability", {})
    capability.setdefault("soc", all_socs)
    capability.setdefault("dtypes", [spec["dtype"]])
    capability.setdefault("layouts", [spec["layouts"]])
    capability["features"] = mode_features(spec)
    capability.setdefault("routes", ["ascendc", "aclnn", "direct_launch"])
    capability.setdefault("optional_routes", ["torch_ops_npu"] if legacy_op_name(op, spec) else [])
    manifest["data_generation"] = manifest.get(
        "data_generation",
        {
            "floating_inputs": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            "positive_inputs": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "structured_inputs": "由 case.optional_inputs 和 reference runner 构造 gate、状态及变长序列索引",
        },
    )
    manifest["coverage_requirements"] = {
        "public_modes": mode_features(spec),
        "known_limits": spec["limits"],
        "required_outputs": [output[0] for output in spec["outputs"]],
        "accuracy_case_ids": [item["id"] for item in cases if "accuracy" in item["tags"]],
        "generalization_case_ids": [item["id"] for item in cases if "generalization" in item["tags"]],
        "negative_case_ids": [item["id"] for item in cases if "negative" in item["tags"]],
        "route_case_ids": [item["id"] for item in cases if "route" in item["tags"]],
        "performance_case_ids": [item["id"] for item in cases if "performance" in item["tags"]],
        "legacy_regression_case_ids": [item["id"] for item in cases if "legacy" in item],
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def render_aclnn_route(op, spec):
    get_name, run_name = aclnn_names(spec)
    return dedent(f"""
        // Canonical aclnn route contract for {spec['title']}.
        // Runtime inputs are defined by tests/op_cases/{op}.json.
        #include "aclnn_{op}.h"

        namespace {{
        [[maybe_unused]] auto *const kGetWorkspace = &{get_name};
        [[maybe_unused]] auto *const kRun = &{run_name};
        }} // namespace
    """).lstrip()


def render_direct_route(op, spec):
    if "direct_route" in spec:
        return spec["direct_route"].lstrip()
    name, params, names = kernel_signature(spec)
    template = spec.get("direct_template")
    declaration_prefix = f"template <{template['decl']}>\n" if template else "extern \"C\" "
    launch_prefix = f"template <{template['decl']}>\n" if template else ""
    launch_name = f"{name}<{template['args']}>" if template else name
    return (
        f"// Canonical Ascend C direct-launch contract for {spec['title']}.\n"
        f"// blockDim/workspace/tiling are produced from tests/op_cases/{op}.json.\n"
        '#include "acl/acl.h"\n'
        '#include "kernel_operator.h"\n\n'
        f"{declaration_prefix}__global__ __aicore__ void {name}({params});\n\n"
        f"{launch_prefix}void Launch{spec['title']}(uint32_t blockDim, aclrtStream stream, {params})\n"
        "{\n"
        f"    {launch_name}<<<blockDim, nullptr, stream>>>({', '.join(names)});\n"
        "}\n"
    )


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".md":
        content = re.sub(r"(?m)^ {8}", "", content)
    path.write_text(content, encoding="utf-8")


def preserve_test_migration_tail(path, content):
    """Keep the per-operator migration ledger and commands across regeneration."""
    if not path.is_file():
        return content
    existing = path.read_text(encoding="utf-8")
    migration_marker = "\n## 3. 历史资产迁移\n"
    generated_command_marker = "\n## 3. 执行命令\n"
    if migration_marker not in existing:
        return content
    if generated_command_marker not in content:
        raise ValueError(f"cannot preserve migration section in {path}")
    return content.split(generated_command_marker, 1)[0] + existing[existing.index(migration_marker):]


def merge_migrated_cases(path, rendered):
    """Append archived case objects when a generated manifest is refreshed."""
    generated = json.loads(rendered)
    if not path.is_file():
        return rendered
    existing = json.loads(path.read_text(encoding="utf-8"))
    migrated = [case for case in existing.get("cases", []) if "legacy" in case]
    if not migrated:
        return rendered
    generated["cases"] = [case for case in generated["cases"] if "legacy" not in case] + migrated
    generated["coverage_requirements"]["legacy_regression_case_ids"] = [case["id"] for case in migrated]
    return json.dumps(generated, ensure_ascii=False, indent=2) + "\n"


def generate(op, spec):
    root = ROOT / spec["root"]
    write(root / "README.md", render_readme(op, spec))
    write(root / "docs" / "design.md", render_design(op, spec))
    write(root / "docs" / "api.md", render_api(op, spec))
    for legacy in root.glob("docs/aclnn*.md"):
        legacy.unlink()

    test_root = ROOT / "tests" / "operators" / op
    test_readme = test_root / "README.md"
    rendered_test_readme = preserve_test_migration_tail(test_readme, render_test_readme(op, spec))
    write(test_readme, rendered_test_readme)
    write(test_root / "common" / "__init__.py", "")
    write(test_root / "common" / "case_matrix.py", render_case_matrix(op))
    write(test_root / "accuracy" / f"test_{op}.py", render_accuracy_test(op, spec))
    write(test_root / "routes" / f"test_aclnn_{op}.cpp", render_aclnn_route(op, spec))
    write(test_root / "routes" / f"test_direct_{op}.cpp", render_direct_route(op, spec))
    legacy_route = test_root / "routes" / f"test_legacy_{op}.py"
    if legacy_op_name(op, spec):
        write(legacy_route, render_legacy_route(op, spec))
    elif legacy_route.exists():
        legacy_route.unlink()
    write(test_root / "ut" / "op_host" / "test_contract.py", render_op_host_contract(op))
    write(test_root / "ut" / "op_kernel" / "test_contract.py", render_op_kernel_contract(op, spec))
    write(test_root / "performance" / "profile.py", render_performance_runner(op, spec))
    write(test_root / "st" / "test_example.py", render_st_test(op, spec))
    manifest = ROOT / "tests" / "op_cases" / f"{op}.json"
    if "case" in spec:
        rendered_manifest = merge_migrated_cases(manifest, render_manifest(op, spec))
        write(manifest, rendered_manifest)
    else:
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        write(manifest, enrich_existing_manifest(op, spec, existing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="*", default=sorted(OPS))
    args = parser.parse_args()
    unknown = sorted(set(args.ops) - set(OPS))
    if unknown:
        parser.error(f"unknown operators: {unknown}")
    for op in args.ops:
        generate(op, OPS[op])
        print(f"generated {op}")


if __name__ == "__main__":
    main()
