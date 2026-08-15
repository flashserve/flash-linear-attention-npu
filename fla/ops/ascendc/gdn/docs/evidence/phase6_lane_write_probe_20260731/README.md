# Phase 6 `g_cumsum` BTH 写回探针（2026-07-31）

## 1. 结论

在 A2 / DAV220 / CANN `9.1.0.beta1` 上，多个 AIV 按 head 分担同一 BTH 行的
4-byte `DataCopyPad` 写回会互相覆盖，**不能用于 Phase 6 P0a**。这不是越界写：
所有 case 的 guard 区均未变，错误发生在同一 32-byte GM block 内的 payload lane。

可行路线是为每个公开 BTH 行设定唯一 owner：owner 读取 BHT tile，在 UB 内
转为完整 BTH 行后一次写回。独立探针中，owner 路径在所有 `117/117` case 上
bit-exact，guard 区 `0` 个不一致。

## 2. 环境和身份

- 设备：A2 device 7（Ascend 910B）
- CANN：`/opt/chw/zhengbao/9.1.0.beta1/cann-9.1.0-beta.1`
- `ccec`：`clang version 15.0.5`
- 探针基线 commit：`57c3ba03a3c15a797cedb9a712f02d3957de94f2`
- 远程工作目录：`/opt/chw/codex-gdn-phase6-lane-probe-20260731/flash-linear-attention-npu`
- 探针源码 SHA256：`c520a5f47ee88a72d86bbcf9897a2640a58e3f21a379961f0eec85f49443bcfe`
- `_C.abi3.so` SHA256：`b31e605cceb01f525d5b7c2cb689354e32fd443f7109da5d8660ea41733b00dd`
- kernel object SHA256：`f61f2c7378cdf99f49a7a1e1f6b9c81aaa3bd71cffee5cd503ff12b42ef2a1df`
- 原始结果：[`probe_result.json`](probe_result.json)，SHA256
  `833d38bff6d58597ba93932343d1ec2c2a2ea5d508029f2c2d5379c383d49e0c`

构建产物已确认包含实际设备 kernel 符号
`ascend_ops::GdnLaneWriteProbe::lane_write_probe_kernel(...)`。

## 3. 方法和结果

探针用有限 FP32 的 raw bit 模式做逐位比较，覆盖 `H=4/8/16`、
`rows=1/2/3/7/8/9/31/64/65/127/128/1025/4096` 和 3 个 seed。`rows` 是展平后的
`B*T`。lane 路径另加 20 轮压力测试。

| 模式 | 写回所有权 | exact / cases | payload 不一致 | guard 不一致 |
| --- | --- | ---: | ---: | ---: |
| lane | 每个 AIV 写一个 head lane | `17/137` | `557400` | `0` |
| whole-row control | 输入已是 BTH，单 owner 写完整行 | `117/117` | `0` | `0` |
| owner-transpose | 单 owner 将 BHT tile 转为 BTH 完整行再写回 | `117/117` | `0` | `0` |

`H=8` 的独立 kernel NPU Event 中位数如下。这些数据只判断写回形式的
可行性，不直接充当 Phase 6 完整 core 收益。

| rows | lane | whole-row | owner-transpose | owner - whole-row |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.036796 ms | 0.014542 ms | 0.014204 ms | -0.339 us |
| 1025 | 0.177578 ms | 0.015573 ms | 0.015424 ms | -0.149 us |
| 4096 | 0.634048 ms | 0.012083 ms | 0.026778 ms | +14.694 us |

owner 路径对长序列的多 tile 循环需要在每个 tile 末尾执行本 owner 的
`PipeBarrier<PIPE_ALL>()` 排空；加入后 `rows=1025/4096` 均 bit-exact。这是单核流水
约束，不是 ABC/DEF 之间的全核 barrier。

## 4. Phase 6 约束

1. 禁止在当前 A2 证据范围内复用“多 AIV 对同一 BTH 行分 lane 4-byte 写回”。
2. P0a 必须使用 chunk/row owner 完整行写回，不保留独立 transpose task。
3. 生产集成应按 chunk 并行分配 owner，不能照搬独立探针中“单 owner 串行整个
   sequence”的调度。
4. 未决项仍是 owner 与 Phase 6 统一 scheduler 的集成、每 chunk 的流水排空代价和
   完整 core 临界路径性能，必须在 P0a/P0c 用真实融合 kernel 复验。

## 5. 备注

早期两次失败分别来自 MTE3 源地址未对齐和 VEC 访问 UB 未对齐，均已修正；
本文只引用对齐语义与现有 cumsum 标量写路径一致后的最终全量结果。
