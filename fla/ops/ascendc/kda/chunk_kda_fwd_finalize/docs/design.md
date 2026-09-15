# ChunkKdaFwdFinalize 设计

## 阶段边界

Prepare 已生成按 HV 展开的 `qg_scaled` 和 `Aqk`，FwdH 已生成
`v_new` 和每个 chunk 的前态 `h`。Finalize 只消费这四项并写出
`attn_out`，不读取原始 `q/k/v/g/beta`，不输出反向检查点或
`h/final_state`。公开约束见[算子 README](../README.md#输入输出)。

## 数学与精度

对每个 batch、value head、序列及其第 `c` 个 chunk，设有效长度为
`M<=64`，物理 `h_c` 按 `state_v_first` 解释为逻辑 `[K,V]`：

```text
P_c = fp32(qg_scaled_c[M,128]) @ fp32(h_c[128,128])
R_c = fp32(Aqk_c[M,M]) @ fp32(v_new_c[M,128])
attn_out_c = bf16(P_c + R_c)
```

两个矩阵乘均以 BF16 输入、FP32 累加，只有最终和落盘时转 BF16。
`qg_scaled` 和 `Aqk` 在 Prepare 边界已经舍入且已应用 scale，
本阶段不能重新缩放，也不能从未舍入的原始 q/k 重算。

`Aqk` 的物理列数为 64，但尾 chunk 只读前 M 列；`v_new` 同样只读
有效 token。不同序列不会共享 chunk，`h` 的 chunk 轴按规范的
sequence-major 顺序映射，不按 tensor 的 T 维简单除以 64。

## 调度与存储

各 chunk 不存在跨 chunk 数据依赖，先按 chunk 分核；仅当 chunk
任务少于可用核时增加完整 value head 的分区。一个 AIC workgroup
一轮最多四个 value head，两个 AIV 各处理两个 head；输出写回
地址按 `(batch,chunk,head)` 分区，不与其他 workgroup 重叠。

| Stage | 核 | 本轮计算 | 下一阶段依赖 |
| --- | --- | --- | --- |
| C0 | Cube | 两个互不依赖的 MMAD：`P=qg_scaled@h`、`R=Aqk@v_new`，各 FP32 `[64,128]` | 两个结果均 ready |
| V1 | Vector | Arch35 一次 VF 完成 FP32 `P+R` 及最终 BF16 cast；Arch22 同一 Stage 先 `Add`、再 `Cast`；均仅写有效 token 行 | 输出 `attn_out` |

Arch22 的两条向量 API 属于同一 Vector Stage，不拆分 pass，也不重复从
GM 读取同一份中间结果；它不满足 Arch35 的“一次 VF 调用”细则，
是 A2/A3 路径需单独核查的架构兼容差异。

每个 head 的 L1 操作数地址固定：`qg_scaled` 16 KiB、`h` 32 KiB、
`Aqk` 8 KiB、`v_new` 16 KiB，共 72 KiB；四个 head 为 288 KiB，
小于每 AIC 的 512 KiB。C0 读完对应 head 的操作数后才能复用其
L1 槽。`h` 的 32 KiB 是 BF16 `[128,128]`，不能和两个 FP32
计算结果的空间混同。

C0 在 Stage 入口先搬完四个 GM 输入，再开始矩阵乘。两项乘积的
L0A 区域分别为 `qg_scaled` 16 KiB 和 `Aqk` 8 KiB，共 24 KiB；
L0B 分别为 `h` 32 KiB 和 `v_new` 16 KiB，共 48 KiB；L0C 的
`P/R` 各占 32 KiB，共 64 KiB。两次 MMAD 分别写自己的 L0C
区域，两个 Fixpipe 结果也不共用目标地址；Arch35 以
PIPE_MTE1/PIPE_M/PIPE_FIX 的 Mutex 约束相应区域的读写，
Arch22 使用对应的 HardEvent。

A2/A3 的尾块不足 16 行时，L1 左操作数按每个 K 分形只填零无效的
M 行，MMAD 的物理 M 补到 16；Fixpipe 与输出仍只写有效行。
填零和输入搬运都由 MTE2 完成，现有 MTE2 到 MTE1 的事件覆盖两者，
不会读取未初始化的 L0A 行，也不会改变有效行的计算语义。

每个 AIV 的 UB 按两个 head slot 静态划分；每 slot 保存独立
FP32 `P` 32 KiB、FP32 `R` 32 KiB 和 BF16 输出 16 KiB，共
80 KiB。两个 slot 占 160 KiB，小于 248 KiB；两个 FP32 plane
在 V1 完成读取前均不能覆盖，其余 88 KiB 不与这两个槽位重叠。
UB 不在 Stage 间搬位；输出所在的 16 KiB 区域不能与尚未完成的
`P/R` 异步写入地址重叠。

Arch35 的 Fixpipe 将 C0 的两份结果直达配对 AIV 的两个 UB plane；
Host 除库 API workspace 外不预留结果 relay。Arch22 的 C0 先写
每个 used AIC 私有 GM relay：`4 heads * 2 planes * 32 KiB =
256 KiB`；AIV 的 MTE2 将两个 plane 都搬入自己的 UB slot 后即可
归还该 head 的 relay 空间，后续 V1 只读取 UB。Host 另加当前平台的
库 API workspace。relay 的两个 plane 必须是独立地址。

跨核同步为 ready/free 双向握手：C0 两个 FP32 plane 完成后
通知 AIV。Arch35 的 free 在 V1 完成及输出 MTE3 读完 UB 后发出；
Arch22 的 free 在 AIV 的 MTE2 搬完该 head 的 GM relay 后发出，
该 AIV 继续从 UB 消费 `P/R`。AIC 必须等待对应 free 才能覆盖
结果槽位。Arch35 的 ready 为 AIV local `0/1`（AIC 映射
`0/1/16/17`），free 为 AIV local `4/5`（AIC 映射
`4/5/20/21`）；Arch22 的 ready 为 `0/1`，free 为 `2/3`。
Arch22 的 UB slot 另以 MTE3 到 MTE2 的核内事件保护复用，
不能以核间 relay free 代替该 UB 生命周期同步。

## layout 与元数据

四个输入始终 head-major。`qg_scaled/Aqk` 为 rank-4 时允许
`BSND/BNSD` 输出，为 rank-3 时允许 `TND/NTD` 输出；packed 模式的
FwdH 主路径的 `v_new/h` 仍保留 rank-4/rank-5 首维 1，不能按
`qg_scaled` 的 rank 自动删掉；独立调用也允许 rank-3 `v_new`。
输入的物理形状不随 `output_layout` 变化。
`state_v_first=true` 时，读取 `h` 时交换末两维语义，避免变更公开
输入的物理存储。

变长 `cu_seqlens` 严格递增且覆盖全部 T token。若提供
`chunk_indices`，它必须与 `cu_seqlens` 同时存在，且恰为所有
`(sequence_id, local_chunk_id)` 的规范 sequence-major 枚举。输出
始终保持输入 token 的相对顺序；`BSND/BNSD` 交换 rank-4 的
token/head 轴，`TND/NTD` 交换 rank-3 的 token/head 轴。

## 兼容边界

独立算子不接收 HK；GVA 复用发生在 Prepare，Finalize 只能验证
已展开为 HV 的数据。`h/final_state` 的公开保留策略属于 FwdH 或
完整 forward，Finalize 必须实际收到内部 `h`，即使调用者不请求
公开 intermediate states。本 PR 不修改共享 Python 注册代码，
因此算子私有 ATK 直调不等价于稳定 `fla_npu` 主入口验证。
