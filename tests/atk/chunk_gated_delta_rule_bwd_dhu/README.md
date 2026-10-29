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
