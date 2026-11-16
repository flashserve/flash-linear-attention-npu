# kda_gate_cumsum ATK 工程

本目录按 `chunk_bwd_dqkwg` 的显式输入工程组织，完成 Phase A 的配置、生成器、executor 与 CPU reference 准备。CPU reference 使用 FP64 计算、返回 FP32；尚未执行 NPU 精度验收或值域校准。

## 接口和 shape

实际 ACLNN 顺序为 `g, a_log, dt_bias, cu_seqlens, chunk_size, use_gate_in_kernel, safe_gate, lower_bound -> gk`；`cu_seqlens` 在 ACLNN 中是 `aclIntArray`，torch adapter 负责转换。`g` 为 head-major `[B,H,T,K]` 或 `[H,T,K]`，支持 FP16/BF16/FP32，输出 `gk` 与 `g` 同 shape、固定 FP32。`a_log=[H]`、`dt_bias=[H*K]` 固定 FP32。格式为 ND。

YAML 中 `varlen/use_rank3/with_bias` 是测试控制参数，executor 消费后再调用实际接口。`use_rank3` 在生成器中将 4D 候选修正为 `[H,T,K]`；varlen 将 B 修正为 1。`cu_seqlens` 显式张量槽位固定 shape `[3]`，executor 按 T 构造 `[0,T//2,T]`，并以 host int-array 传递；T=1 时首序列为空，符合非递减契约。dense 传 None。门控关闭时 `a_log/dt_bias=None`；门控打开时始终提供 a_log，按 `with_bias` 决定 bias 是否存在。浮点输入直接保留 ATK 样本，不在 executor 重新随机生成。

## 约束来源清单

源码路径均相对仓库根，前缀 `S=fla/ops/ascendc/kda/kda_gate_cumsum`。

| 约束 | 来源 | 配置/修正 |
| --- | --- | --- |
| g rank 3/4；B/H/T/K>0；K<=256 | C++ `S/op_host/op_api/aclnn_kda_gate_cumsum.cpp:87-98`、`S/op_host/kda_gate_cumsum_tiling.cpp:49-51` | YAML 4D 候选 + use_rank3；generator 只修正维度与 K 上限 |
| chunk_size=32/64/128 | C++ `S/op_host/op_api/aclnn_kda_gate_cumsum.cpp:85` | YAML 离散枚举 |
| g FP16/BF16/FP32；辅助 FP32；cu INT64；gk FP32 同 shape | C++ `S/op_host/kda_gate_cumsum_def.cpp`、ACLNN 输出校验；torch `torch_custom/fla_npu/csrc/src/stable_kda_gate_cumsum.cpp:47-65` | 显式 dtypes；CPU 返回固定 FP32 |
| a_log=[H]；可选 bias=[H*K]；use_gate 时必须有 a_log | C++ `S/op_host/op_api/aclnn_kda_gate_cumsum.cpp:106-127` | generator 修正辅助 shape；executor 处理 optional |
| safe 且启用 gate 时 lower_bound∈[-5,0) | 同上 `:113-116` | YAML [-5.0,-1.0] |
| cu 至少两项，以0起、T终、非递减；4D varlen B=1 | 同上 `:58-74,101-105` | generator 将 B 修正为1；executor 生成合法 offsets |
| Gate 后按 chunk 重置 cumsum，输出为 log2 域 | C++ `S/op_kernel/kda_gate_cumsum_kernel.h` 的 ApplyGate/ProcessChunk | 独立 CPU reference，不加载算子 Python 实现 |
| 总元素数不超过2^34 | ATK 精度标准 §1.2 | YAML max_length=17179869184；当前离散候选远低于此上限 |

C++ tiling 的 task/shape 运算为 int64，kernel 主要地址运算为 uint64；本工程未加入超过 INT32 的超大 shape 专项，不宣称完成大规模索引覆盖。

## 覆盖与限制

候选包含 T=1/31/32/33/63/64/65/127/128/129、K=1/31/32/127/128/255/256、chunk 32/64/128、3种 dtype、rank3/rank4、dense/varlen、gate开关、safe开关、bias存在/缺失。候选为有限设计空间，不表示所有组合已被抽样穷举。正态/均匀分布各50%，值域和分布参数来自本次读取的 ATK 标准；非有限值边界可生成。空 tensor/标量不符合 positive logical dimensions 与 rank 契约，因此禁用。当前未加入预期错误用例。

SoC 注册为 ascend910b / ascend910_93 / ascend950；源码接口含弃用提示，executor 使用当前已存在接口。真实 NPU/CPU 差异、特殊值传播与容差仍需 Phase B 验证。未提供用户模型 case，因此不创建性能 JSON；未执行全部 TilingKey 的真机选择记录，因此不创建 mss 验收 JSON。

## 运行

在正确安装 ATK 的环境，仓库根目录执行：

```bash
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -scope=gen_cases
```

小规模 dry-run 和正式候选生成：

```bash
cd tests/atk/kda_gate_cumsum
timeout 60 atk case -f kda_gate_cumsum.yaml -p gen_kda_gate_cumsum.py -dt 1 -en 2
timeout 60 atk case -f kda_gate_cumsum.yaml -p gen_kda_gate_cumsum.py -s 20261030
```

输出为 `result/kda_gate_cumsum/json/all_kda_gate_cumsum.json`，审核后固化为 `atk_kda_gate_cumsum.json`。`result/` 和日志不提交。CPU 结构检查可通过 ATK 发起（不代表 NPU 精度通过）：

```bash
atk node -b cpu task -c atk_kda_gate_cumsum.json -p executor_kda_gate_cumsum.py -s 0 -e 1 -tk accuracy
```

本次状态：Phase A；无 NPU 验收或性能结论。

ATK 合成 upper_border 可能越过 YAML 离散候选并取 INT32_MAX。generator 读取 YAML 每维候选最大值，将该类边界投影回本工程声明的有限测试域；不在 constraint 中另设候选范围。大于该测试域的索引溢出场景需另行设计。

## 本次 Phase A 验证记录

- 环境：CPU Docker，ATK 26.9.8、PyTorch 2.7.1；未执行 NPU。
- ATK `case -dt 1 -en 0 -s 20261030` dry-run 通过；正式按 YAML `dtype_numbers=32`、`extra_numbers=3` 生成 100 条候选；按用户要求仅保留首条到 `atk_kda_gate_cumsum.json`（共 1 条）。
- JSON 由 ATK 产生，归档时统一设置 `default_seed=20261030`；输入 shape/dtype/属性保持生成结果。`standard.acc=mixed_tolerance_bm` 遵循仓库测试契约。该包是 Phase A 候选，未满足正式验收三类用例包与每场景3固定种子的要求。
- `atk node -b cpu task -s 0 -e 1 -tk accuracy` 实测执行1条，日志为 `Total Task: 1, success 1, failed 0`。仅确认 CPU 路径能执行；未产生 CPU/NPU 精度通过率。
- 用归档 JSON 经 ATK `OpsDataset` 还原最小样本，核对实际 CPU 输出的 shape、输出数与 FP32 dtype。Python AST 与 YAML 解析检查通过。
