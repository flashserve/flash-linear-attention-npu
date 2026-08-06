# ChunkKdaFwd 设计

## 目标

1. 顶层接口对齐非 CP 的 FLA `chunk_kda_fwd`。
2. 每个阶段是独立 L0 算子，不在一个 kernel 中用 stage 跨越另一个算子。
3. A2/A3/A5 使用同一数学定义；A5 保留 regbase 双发射特化。
4. 输入 layout 与输出 layout 解耦。
5. FwdH 同时服务 KDA 与 GDN，并支持可选 scalar gate、key-wise gate 和 `state_v_first`。

## L2 调度

```text
raw g
  -> KdaGateCumsum
  -> ChunkKdaFwdPrepare
  -> ChunkKdaFwdPostWu
  -> ChunkGatedDeltaRuleFwdH
  -> ChunkKdaFwdFinalize
  -> attn_out
```

BSND/TND 输入先由 `l0op::Transpose` 物化为连续 BNSD/NTD。仓内不再维护独立 swap 算子。
每个 kernel launch 在同一 stream 上建立阶段依赖。

## 阶段职责

### KdaGateCumsum

将 raw/已激活 gate 转为 FP32 chunk-local log2 累计值：

```text
gk = cumsum(gate) / ln(2)
```

该算子同时保留独立 L2 接口供 GDN2 调用，输入和输出固定为 BNSD/NTD。

### ChunkKdaFwdPrepare

只读取 `q/k/v/gk/beta` 及变长元数据，产生：

```text
Aqk, Akk, qg, qg_scaled, w_seed, u_seed
```

矩阵计算和三角求逆使用 FP32 累积；公开中间量在写回时转为 q dtype。

### ChunkKdaFwdPostWu

只读取 `k/gk/w_seed/Akk/u_seed`，产生：

```text
w, u, kg, v_new_seed
```

`Akk` 的 head 循环按 `H_v` 执行，GQA 映射只在读取 q/k head 时换算，避免按 `H_k` 重复或漏算。

### ChunkGatedDeltaRuleFwdH

读取 `kg/w/u/gk` 和可选 `initial_state`，计算 chunk 间递推：

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
```

`h` 与 `v_new` 必选于该 L0；`final_state` 真正可空。共享算子还支持 scalar `g`，其自然指数路径不再暴露
`use_exp2` 属性；key-wise `gk` 固定使用 `exp2`。

### ChunkKdaFwdFinalize

只读取 `qg_scaled/Aqk/v_new/h`，计算：

```text
attn_out = qg_scaled @ h + Aqk @ v_new
```

kernel 内直接按 BSND/TND 写出 `attn_out`。供反向使用的中间量保持 BNSD/NTD。

## 状态布局

内部递推统一使用 `[...,K,V]`。`state_v_first=true` 时，L2 在进入 FwdH 前转置 initial state。
内部 `hCompute` 始终保持 head-major 供 Finalize 消费；公开 `hOut` 在 L2 导出边界转为
sequence-major，并按 `state_v_first` 决定末两维顺序。`final_state` 按序列排列，与 FLA 顶层
输出一致。

## 重计算策略

L2 不理解 autograd 重计算策略。`final_state/gk/w/u/qg/kg/v_new/h` 是相互独立的
`OPTIONAL_OUTPUT`；非空指针表示导出，空指针表示不公开该结果。`w/u/qg/kg/v_new/h` 的 L0
阶段固定写内部 compute tensor，L2 通过 `ViewCopy` 导出非空输出，不能依赖公开可选输出来
承接或延长内部生命周期。`gkOut` 非空时直接复用为 `gkCompute`，避免目标场景额外复制整张
FP32 gate。

Python/legacy 包装层对齐 fla-org `chunk_kda_fwd` 提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

- `Aqk/Akk` 始终返回。
- `disable_recompute=false` 时不保留 `w/u/qg/kg/v_new`。
- `disable_recompute=true` 或 `return_intermediate_states=true` 时保留公开 `hOut`。
- `use_gate_in_kernel=false` 或 `disable_recompute=true` 时保留 `gk`。
- `final_state` 只在 `output_final_state=true` 时创建公开输出。

内部 `hCompute` 与公开 `hOut` 是两个生命周期：`hCompute` 是 FwdH 到 Finalize 的必需
head-major 阶段结果，L2 无论 `hOut` 是否为空都必须提供；`hOut` 仅表示是否将该结果转为
sequence-major 并通过 `ViewCopy` 导出到调用方。该规则对齐非 CP 的低层 12 返回值接口；
第 12 项 `initial_state` 由 Python 层原对象透传。

## 模板化方案与 tiling key

Prepare、PostWu 和 Finalize 是三个独立算子，各自维护 tiling 数据、workspace 偏移和固定
`tiling key=1`，运行时 shape、变长序列信息和任务数只通过本阶段 tiling 传递。数据类型、
safe-gate 数值路径和 A2/A3/A5 向量实现由编译期模板选择，不用运行时 stage 分支跨算子。

共享 FwdH 同样是独立算子，仅用 tiling key 区分 V=128 与 V=256 的 tile shape；scalar `g`
固定按自然指数处理，key-wise `gk` 固定按 `exp2` 处理。该划分限制模板组合数量，也避免
把其他阶段的属性或 workspace 生命周期带入 FwdH。

## 性能设计

- Prepare 的右矩阵在 L1 驻留，避免 K/K^T 重复搬运和重复转置。
- AIC 使用 L1/L0 双缓冲组织 MTE2、MTE1、Cube、Fixpipe。
- AIV 使用输入 staging ping-pong，使下一 tile MTE2 与当前 tile VEC 重叠。
- A5 VEC 路径使用 regbase 双发射；数值主计算仍保持 FP32。
- inter-sub-chunk 合并使用独立 workspace 区域，避免阻塞主 tile 流水。

性能结论只使用 `msopprof`。目标回归 case 定义在 `tests/op_cases/chunk_kda_fwd.json`。

## 验证矩阵

- 平台：A2/A3/A5。
- dtype：FP16/BF16。
- layout：BSND/BNSD/TND/NTD。
- gate：raw/已激活、safe true/false。
- Shape：K=128，V=128/256，chunk=64/128，dense/varlen/tail/GQA。
- 属性：final state、重计算策略、`state_v_first`。
