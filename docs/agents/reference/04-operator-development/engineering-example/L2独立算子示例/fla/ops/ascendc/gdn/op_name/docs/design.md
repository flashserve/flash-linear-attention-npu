# OpName 设计

<!--
示例文件：fla/ops/ascendc/gdn/op_name/docs/design.md

注意事项：
  1. 开头记录方案设计规则版本（与 docs/agents/03-方案设计.md 的版本对齐），否则后续迭代无法判断
     本文是否还适用。
  2. 写清 Stage 划分、每个 Stage 的执行单元、输入落点、输出落点、同步条件——这几项是 lv3 检视的
     唯一依据，缺一项就等于没有详设。
  3. 内存分配要给出各层 buffer 的份数与容量结论，workspace 要给出区域表（offset/size/使用方/生命周期）。
  4. 模板化方案要写清每个模板参数的含义、取值域与选择条件，并说明 TilingKey 只标场景族。
  5. 公开接口语义不在本文重复，链接 docs/api.md 与算子 README。
-->

> 方案设计规则版本：`V1`（对应仓库 `docs/agents/03-方案设计.md` 的当前版本）
>
> 接口与支持范围：[`api.md`](api.md)；输入 shape 与已知限制：[`../README.md`](../README.md)

## 1. Stage 划分

| Stage | 执行单元 | 计算 | 输入 | 输出 | 同步 |
| --- | --- | --- | --- | --- | --- |
| S0 | MTE2 + VEC | 计算 `norm`（L2 或恒等） | `x`、`epsilon` | UB `normBuf` | MTE2→VEC 事件 |
| S1 | Cube | `partial = x^T @ (scan * norm)` | UB/L1 上的 `x`、`scan` | L0C `partial` | S0 完成事件 → Cube |
| S2 | Fixpipe + VEC | 写回 `y`；`save` 档额外写 `state`、`x_norm` | L0C、UB | GM | Cube→Fixpipe 事件、VEC→MTE3 事件 |

同一 core 连续处理多个 chunk 时按 S0→S1→S2 顺序推进；每个 chunk 的 `scan` 需要前一个 chunk 的前缀和，
因此单核内按 chunk 顺序串行，跨 chunk 的依赖不做核间同步。

## 2. 任务与分核

1. host 把 fixed/varlen 输入统一转换为 `(seq, chunk)` 任务，附带 token 起点、有效长度、head 与
   workspace offset。
2. 按 head 分核：`headsPerCore = ceil(H / aicCoreNum)`，余数依次分给前面的 core；每核内的 chunk 顺序
   执行。
3. `usedCoreNum = min(aicCoreNum, H)`；不按 chunk 分核，避免同一序列的状态被多个核拆分。

## 3. 内存分配

| 资源 | 用途 | 份数 | 容量结论 |
| --- | --- | --- | --- |
| UB `xBuf` | 当前 chunk 的 `x` tile | 2（ping/pong） | 按 `tileT * D * sizeof(dtype)` 计算，两份之和不超过 arch22/arch35 的 UB 规格 |
| UB `normBuf` | 归一化系数 | 1 | `tileT * sizeof(float)` |
| L1 `xCache` | Cube 左矩阵 | 2 | `tileT * D * sizeof(dtype) * 2` |
| L0C `partial` | Cube 输出 | 1 | `tileT * tileD * sizeof(float)` |

workspace 区域表：

| 区域 | size | 使用方 | 生命周期 |
| --- | --- | --- | --- |
| `stateSlot` | `usedCoreNum * chunkPerCore * D * sizeof(dtype)` | S2（`save` 档） | kernel 结束释放；`none` 档不分配 |
| `normScratch` | `usedCoreNum * tileT * sizeof(float)` | S0 | 单 chunk 内有效，下个 chunk 覆盖前必须等待 VEC 完成 |

## 4. 模板化方案与 TilingKey

| 模板参数 | 含义 | 取值域 | 选择条件 |
| --- | --- | --- | --- |
| `D_T_X` | `x` 的存储类型 | BF16 / FP16 | 输入 dtype |
| `D_T_G` | `g` 的存储类型 | BF16 / FP32 | 输入 dtype |
| `NORM_MODE` | 归一化模式 | L2 / Identity | `epsilon` 是否参与（由 host 依据 `a_log` 是否为空推导） |
| `USE_STATE` | 是否读初始状态 | 0 / 1 | `initial_state` 是否提供 |
| `OUTPUT_MODE` | 公开输出档位 | none / save | L2 的输出指针组合 |

TilingKey 只由上述 dtype 与模式决定，**不编码平台**：A2/A3/A5 生成同样的 key，平台在 kernel 入口按
`__CCE_AICORE__` 选择 `arch22` 或 `arch35` 实现。`chunk_size`（64/128）作为独立场景族，通过
tiling 数据传入并在同一 key 内分支。

## 5. 平台差异

| 平台 | tiling 差异 | kernel 差异 |
| --- | --- | --- |
| A2/A3（`arch22`） | `HEAD_PER_CORE` 与 tile 取 `..._ARCH22_*` 常量 | `op_kernel/arch22/op_name_{cube,vec}.h` |
| A5（`arch35`） | tile 更大、`LibApiWorkSpaceSize` 计入 workspace | `op_kernel/arch35/op_name_{cube,vec}.h` |

host 侧用 `platform.GetCurNpuArch() == NpuArch::DAV_3510` 判定 A5，不把平台写进 TilingKey。
