# ChunkGatedDeltaRuleBwdDhu ATK 工程

本目录提供 `chunk_gated_delta_rule_bwd_dhu` 的 ATK 单算子工程，包含 `executor_chunk_gated_delta_rule_bwd_dhu.py`、`gen_chunk_gated_delta_rule_bwd_dhu.py`、`chunk_gated_delta_rule_bwd_dhu.yaml`、`atk_chunk_gated_delta_rule_bwd_dhu.json`。

## 输入约束

- `q/k` 必须为 `[B,HK,T,K]`，且二者形状完全一致。
- `w` 必须为 `[B,HV,T,K]`；`dO/dv/dv2` 必须为 `[B,HV,T,V]`，且 `dO` 与 `dv` 形状完全一致。
- `g` 与 `gk` 必须二选一：`g=[B,HV,T]`，`gk=[B,HV,T,K]`；门控 dtype 需要为 `FLOAT` 或与 `q/k` 一致。
- `dh` 固定为 `[B,NT,HV,K,V]`；`state_v_first` 仅控制 `h0/dht/dh0` 的末两维布局。
- `q/k` 与 `w/dO/dv/g` 的 `B`、`T` 必须一致；`HV % HK == 0`。
- tiling 要求 `K=128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`。
- 变长模式下 `cu_seqlens` 与 `chunk_indices` 必须同时提供，`chunk_indices` 长度为正偶数，且 `B=1`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/README.md; torch_custom/fla_npu/test/test_npu_chunk_gated_delta_rule_bwd_dhu.py

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_gated_delta_rule_bwd_dhu.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

本目录 `atk_chunk_gated_delta_rule_bwd_dhu.json` 内置 200 条泛化用例（100 个 shape × bf16/fp16），
标准为 `mixed_tolerance_bm`，SOC 为 `ascend950`。覆盖 noGVA（`HK==HV`）与 GVA（`HV>HK`）。

示例（BF16 首条）：
`{"dtype": "bf16", "B": 64, "HK": 8, "HV": 8, "T": 1024, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_bwd_dhu", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## 非零 dht 回归（#924）

`gen_chunk_gated_delta_rule_bwd_dhu.py --dht-output <输出JSON>` 额外生成 384 条用例，
覆盖 BF16/FP16、V=128/256、chunk=64/128、KV/VK 状态布局、定长/变长、g/gK、
h0 存在/缺失的 128 种组合，每种组合使用 3 个固定种子。
输入仍使用原有 `_randn`、`_gate`、`_kda_gate` 规则，标准仍为模板中的 `mixed_tolerance_bm`。
CPU 标杆复用 `torch_custom/fla_npu/test/test_bwd_dhu.py`，不修改标杆公式或容差。

在本目录运行生成后的 JSON：

```bash
atk node --backend npu --devices 0 node --backend cpu task \
  -c "$DHT_CASES_JSON" --task accuracy --bm_device cpu \
  -p executor_chunk_gated_delta_rule_bwd_dhu.py -to 2000
```

增加 `--dht-minimal` 可生成 32 条代表用例，用于确定性和内存检查；覆盖两种 dtype、
两种 V、两种布局、定长/变长以及 g/gK，保留原始 case id。
executor 从 ATK case metadata 读取 `case_spec`，避免 `non_param` 被移除后丢失可选参数。

算子目录下的 `test/test_dht.py` 另提供 66 条独立解析回归：令 q/w/dO/dv/gate 为零、k 为可追踪 one-hot，
验证非零 dht 在每个 chunk 的 dh、最终 dh0 以及 dv2 中的精确传播，同时验证 None 与零 dht 等价。
其中包含超过核心数的 seq/head-window task，检查跨任务缓冲复用。该测试不依赖矩阵乘标杆。

### 验证结果

被测版本为基于 `47d7c97` 的本次 #924 修复；CPU 标杆和原有输入值域、精度标准保持一致。

| 测试项 | 结果 |
| --- | --- |
| A5 非调试构建 | 通过，运行包的 Python wrapper、opapi 与 kernel 一致性检查通过 |
| A2 通用路径交叉构建 | 通过；未执行 A2/A3 实机测试 |
| 独立解析回归 | 66/66 通过，None/零/非零 dht 的输出均精确符合预期 |
| 原有 ATK 精度 | 200/200 通过，执行失败 0，精度失败 0 |
| 非零 dht ATK 精度 | 384/384 通过，执行失败 0，精度失败 0 |
| ATK 确定性 | 32/32 通过，每条重复 50 次 |
| ATK memcheck | 32 条执行完成、报告通过；日志告警见下文 |

memcheck 日志包含原有 `dh0` 清零 `AscendC::Fill` 调用的 GM 越界 WARNING。
未修改的 `47d7c97` 包也可复现；带行号构建将告警定位到该调用，核对报出的地址及长度均位于
已分配的 `dh0` 缓冲内。32 条检查中告警均来自这一调用，未发现指向新增 dht 加载或转置路径的告警。
保留该工具告警记录，不将“报告通过”等同于无告警。

性能按原有 `_perf.json` 的 BF16 模型用例单轮采集，`B=1, HK=HV=4, T=512, K=V=128, chunk=64`，
`dht=None`。取 `op_summary` 中该算子的 `Task Duration(us)`：基线 `39.136 us`，修复后 `38.972 us`；
pipe 字段完整。此处只记录单轮数据。

## A5 pre-gated qg direct（独立 c9e2+Dhu 分支）

### 适用边界

`USE_GK=1 && HRatio==1` 时，q 已是 pre-gated qg。AIC 在当前 head 的 `DH_READY`
等待前直接将 q 从 GM 装载到该 head 原来的 qg L1A slot，AIV 不再执行 GM→UB、恒等 cast 和
UB→L1 staging。GEMM 输入内容、布局和计算顺序不变。direct 使用本地 MTE2/MTE1 ready/free，
最后一个 K tile 的 L1→L0A 后释放 L1，不增加跨核旗、跨 task 预取或 L1 峰值。
GVA 与 `USE_GK=0` 保留原 AIV staging、共享与 gate 路径，公开接口和跨核旗不变。
共享 kernel 入口给 A2/A3/A5 的 Cube `Init` 都传入 q；A2/A3 不使用该参数，原数据路径不变。

本独立分支只包含 c9e2 基线上的 Dhu qg direct 和测试 `ct` 延迟导入，不包含 Finalize、
公共 V2 ATK README、Python 路由或其它算子的改动。

### 仓内 direct 持久回归

`atk_chunk_gated_delta_rule_bwd_dhu_direct.json` 和
`atk_chunk_gated_delta_rule_bwd_dhu_direct_mss.json` 由仓内 generator 生成，不依赖外部实验目录。
在仓库根目录执行：

```bash
python tests/atk/chunk_gated_delta_rule_bwd_dhu/gen_chunk_gated_delta_rule_bwd_dhu.py \
  --direct-output tests/atk/chunk_gated_delta_rule_bwd_dhu/atk_chunk_gated_delta_rule_bwd_dhu_direct.json \
  --direct-mss-output tests/atk/chunk_gated_delta_rule_bwd_dhu/atk_chunk_gated_delta_rule_bwd_dhu_direct_mss.json
```

精度文件共 196 条：先从 dht profile 复制 192 条 `use_gk=true` 组合并改为 `HK=HV=4`，保留原有
BF16/FP16、V128/V256、C64/C128、KV/VK 布局、dense/varlen、h0 和固定种子分布；再增加
case id 384–387，覆盖 1/2/3/4-head window、3-head/V256、尾长 1/17/65 以及多 task 输入。
独立 `_mss.json` 共 36 条，包括 C64/C128 各 16 条代表组合和上述 4 条边界用例；其中
B=48/65/96 的用例用于覆盖跨 task 及奇偶 workspace slot 复用。

所有 direct 用例均满足 `use_gk=true && HK==HV`，并保留现有 `case_spec`、executor 和
`mixed_tolerance_bm`。模板中的 `outputs: null` 表示比较 executor 返回的完整 `(dh, dh0, dv2)`；
`h0` 存在时同时比较非空 `dh0` golden，可选 `dht/h0` 的存在与缺失均保留。
`USE_GK=0` 的 raw-gate 和 `HK!=HV` 的 GVA 仍走原 AIV staging 回退路径；默认 200 条和
dht 384 条 profile 不被 direct 生成入口改写。

direct 精度复现命令：

```bash
atk node --backend npu --devices 0 node --backend cpu task \
  -c tests/atk/chunk_gated_delta_rule_bwd_dhu/atk_chunk_gated_delta_rule_bwd_dhu_direct.json \
  --task accuracy --bm_device cpu \
  -p tests/atk/chunk_gated_delta_rule_bwd_dhu/executor_chunk_gated_delta_rule_bwd_dhu.py -to 2000
```

仓内 direct 196 条已在当前生产实现上完成真实 ATK accuracy，**196/196 PASS**，报告见
`runs/pr_final_20261029/pr995/atk/direct196/case_detail.json`。case 384 是单 task 两 chunk；
case 385–387 才具有跨 task 输入设计。当前尚未记录实际 headsPerTask/tiling 选择，不能把输入设计
直接写成全部实机分核证据。独立 direct_mss36 的内存、竞争和确定性检查另行列出，不由精度结果代替。
下文历史 direct 192/192 结果不涵盖新增边界用例和独立 `_mss.json`。

### 历史口径与独立复测范围

参考工作树中的早期 standalone 实验和联合 V2 环境记录仅作历史参考，不作为本独立分支的
最终验收证据。其它 PR（包括 PR913）或“前三项优化”联合环境的布局、精度和性能结果均不
归入本 PR。以下是本次 c9e2+Dhu 独立分支的真实复测结果，不转录联合环境结果。

| 项目 | 本次独立分支结果 | 证据 |
| --- | --- | --- |
| A5 build | 通过 | `runs/pr_final_20261029/pr995/build.log` |
| A2/A3 交叉编译 | 均通过；未执行 A2/A3 实机测试 | `runs/pr_delivery_20261028/dhu/a2_dhu_build.log`；`runs/pr_delivery_20261028/dhu/a3_dhu_build.log` |
| wheel 安装匹配 | **1318/1318** 个 `fla_npu` 文件匹配，`wheel_install_mismatches=[]`；两个 Dhu 对象与已验收生产对象一致 | `runs/pr_final_20261029/pr995/install_verify.log` |
| fresh 基线对象比较 | 仅 **2 个 Dhu 对象**变化；其余 **185** 个对象名称/hash 全同，scoped 比较通过 | `runs/pr_delivery_20261028/base/object_compare_final.log` |
| smoke | **20/20 bit-exact**（saved 10/10、recompute 10/10） | `runs/pr_final_20261029/pr995/regression_r2/smoke_compare.json` |
| 非零 dht ATK | **384/384 PASS** | `runs/pr_final_20261029/pr995/regression_r2/atk/dht384/case_detail.json` |
| 默认 ATK | **200/200 PASS** | `runs/pr_final_20261029/pr995/regression_r2/atk/default200/case_detail.json` |
| 仓内 direct ATK | **196/196 PASS**，含新增 384–387 边界用例 | `runs/pr_final_20261029/pr995/atk/direct196/case_detail.json` |
| direct 确定性 | **36/36 × 50 PASS** | `runs/pr_final_20261029/pr995/determinism/check.log` |
| minimal 输出对照 | 16 case、**48/48** 张量逐位一致 | `runs/pr_final_20261029/pr995/regression_r2/minimal/compare.json` |
| ct 收集与导入 | 收集 2 个测试；无 ct 导入与委托检查通过；ct 未安装，未执行实际 `ct.single` 比较 | `runs/pr_final_20261029/pr995/ct/collect.log`；`runs/pr_final_20261029/pr995/ct/import_and_delegation.log` |
| C64/C128 memcheck | C64 **19/19**、C128 **17/17** 报告 Pass；两侧 WARNING 签名一致，新增 0 | `runs/pr_final_20261029/pr995/mss/memcheck_c64_compare.json`；`runs/pr_final_20261029/pr995/mss/memcheck_c128_compare.json` |
| C64/C128 racecheck | C64 **19/19**、C128 **17/17** `No error detected`；两侧 WARNING 签名一致，新增 0 | `runs/pr_final_20261029/pr995/mss/racecheck_c64_compare.json`；`runs/pr_final_20261029/pr995/mss/racecheck_c128_compare.json` |

这里的“新增 WARNING=0”是候选相对基线的 `new_warnings=[]`，**不是总告警数为零**。
新 direct_mss36 边界下，memcheck 在基线/候选两侧的 WARNING 数分别为 C64=936、C128=64262；
racecheck 两侧均为 0。四份 audit 同时确认 `identical_warning_signatures=true`、`missing_warnings=[]`、`errors=[]`、`pass=true`。
这些结果只表示当前 Dhu direct 子集没有新增签名，既有告警仍需跟踪，不构成整链内存验收通过结论。

saved/recompute 性能只取本独立分支的 `runs/pr_delivery_20261028/dhu/perf/perf_saved.json` 和
`runs/pr_delivery_20261028/dhu/perf/perf_recompute.json`。两份 JSON 各 6 轮；Dhu 一栏的配对差
中位数是逐轮 `candidate - baseline` 的中位数，比值变化直接取 `kernel_delta_percent`
（两侧 kernel 中位数的比值减 1），不将两种统计量混用。

| 路径 | Dhu 基线中位数 (µs) | Dhu 候选中位数 (µs) | Dhu 配对差中位数 (µs) | Dhu 比值变化 | bwd 配对差中位数 (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| saved | 1438.9415 | 1302.9905 | -136.9220 | -9.447987% | -143.7635 |
| recompute | 1330.3625 | 1293.8535 | -36.7550 | -2.744290% | -25.5505 |

上述 direct、回退路径和 C64/C128 单算子检查均不构成联合环境或整链验收结论。
**整链 msSanitizer 尚未闭环**；A2/A3 仅完成交叉编译，**未跑 A2/A3 实机，整体 ST 未跑**。
当前 head 的 7 项 A2+A5 NPU CI 未执行，CI 契约测试已通过；既有 memcheck WARNING 仍需跟踪，不能写成零告警。
