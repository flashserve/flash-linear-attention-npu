# A2/A3 六阶段私有实现

本目录承载 `DAV_2201`（A2/A3）的 `ChunkGatedDeltaRuleFwd` 私有实现。代码最初从
冻结提交 `chw@e0fb8942336efcf3b2e13d7463e63b5b5d38f341` 的独立算子路径复制；当前
目录按计算阶段、组合流水和共享资源组织。

## 入口与目录职责

`chunk_gated_delta_rule_fwd_arch22.cpp` 是总入口，负责六阶段调用和跨阶段衔接；
`chunk_gated_delta_rule_fwd_arch22_struct.h` 保存统一 tiling 结构及相关常量。

| 路径 | 职责 |
| --- | --- |
| `stages/cumsum/` | Cumsum Prepare，准备前段使用的累计门控数据。 |
| `stages/kkt/` | `kkt_cube.h` 计算原始 K×Kᵀ；`kkt_epilogue.h` 完成 KKT 向量后处理。 |
| `stages/solve/` | FP32 Solve 流水与布局暂存；`compat/` 保留原 Solve 的 Cube、Vector 和公共定义。 |
| `stages/recompute_wu/` | W/U 重计算及其 host tiling 辅助定义。 |
| `stages/fwd_h/` | H 状态递推及其 host tiling 辅助定义。 |
| `stages/fwd_o/` | O 输出计算。 |
| `pipeline/` | `cumsum_kkt_solve.cpp` 组合前段实现，`wu_ho/` 组合 WU/H/O，HO 协议头提供调度通知原语。 |
| `common/` | `gdn_raw_front_resources.h` 提供共享资源与偏移视图，`qkv_input_layout.h` 提供 QKV 输入布局定义。 |

组合实现由总入口以原有 `IMPL_ONLY` 宏方式引入；目录层次不增加计算阶段。
主路径使用 FP32 Solve，`stages/solve/compat/` 中的实现继续保留原有架构条件分支。

## 维护约定

- A2/A3 私有实现的修复进入本目录；A5 实现保持独立。
- 保持公共 ABI、输入输出顺序、tiling 序列化布局及兼容实现。
- 修改前读取当前工程的 `HANDOFF.md`，验收遵循当前工程约定和本轮明确的测试范围；
  复用冻结的用例、输入、参考实现与阈值，并绑定实际源码和构建产物。
- 目录整理只修改文件位置、include 和构建路径；计算、同步、模板参数、宏条件、类名及调用顺序保持原样。
