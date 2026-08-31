# ChunkFwdH ATK 工程

本目录提供 `chunk_fwd_h` 的 ATK 单算子交付件。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 被测入口

Executor 只通过稳定 ctypes 入口调用被测算子：

```python
from fla_npu.ops.ascendc import chunk_fwd_h
```

本工程不加载 legacy dispatcher，也不为可选兼容命名空间保留测试分支。CPU 节点只执行
独立参考实现，NPU 节点只执行上述稳定入口。

## 输入限制

- `k/w/u` 均为 BF16、BNSD rank-4 tensor。
- `k=[B,HK,T,128]`，`w/u=[B,HV,T,128]`，当前仅支持 `K=V=128`。
- `chunk_size` 固定为 `64`，`save_new_value` 固定为 `true`。
- `g` 与 `gk` 必须且只能提供一个。
- g-only：`g=[B,HV,T]`，`k` 为 raw key，要求 `HV >= HK` 且 `HV % HK == 0`。
- gk-only：`gk=[B,HV,T,128]`，`k` 为已准备好的 `kg`，要求 `HK=HV`。
- `g/gk` 支持 BF16/FP32；`initial_state` 支持空、BF16 或 FP32。
- `state_v_first=false` 时 state 末两维为 `[K,V]`，为 `true` 时为 `[V,K]`。
- 变长模式使用 BNSD 容器且要求 `B=1`；`cu_seqlens` 从 0 开始、以 `T` 结束并严格递增。
- `chunk_indices` 为空时由稳定入口自动生成；非空时必须等于 sequence-major 的规范序列。
- 标准连续 tensor 和非连续 ND view 均有正向用例；私有 NPU format 不属于本接口范围。

## CPU 标杆

`executor_chunk_fwd_h.py` 内的参考实现独立展开每个 sequence、value head 和 chunk。输入先按
接口 dtype 量化，标杆再复现 kernel 的关键精度边界：

1. rolling state 写入 `h` 前转 BF16；
2. `W @ H` 和 `K^T @ right` 使用 FP32 累加；
3. Stage0 结果按 state dtype 转换后再计算 `v_new`；
4. `v_new` 与 g-only 的 right 在进入第二次矩阵乘前转 BF16；
5. `use_exp2=true` 按 kernel 的 `x * ln(2) -> Exp` 路径计算；
6. rolling state 每个 chunk 后按 BF16/FP32 state dtype 回写。

因此该标杆不是把整个递推直接提升到 FP64，而是复现公开计算语义中的 cast 点。CPU 与 NPU
节点使用相同 seed 和量化后的输入，ATK 统一使用 `mixed_tolerance_bm` 比较所有可见输出。

## 精度与反向矩阵

`atk_chunk_fwd_h.json` 共 49 条，只包含正向精度用例。其中 0-16 保留功能、边界、
分核和变长专项场景，17-48 是完整的 32 项顶层模板矩阵。

正向 case id 按以下分组：

| case id | 覆盖内容 |
| --- | --- |
| 0 | 单 token、单 head、末 chunk 不输出 final state，覆盖跳过 Stage2/Stage3 |
| 1 | 总任务数 33、g-only `HK:HV=1:3`、BF16 initial state |
| 2 | 总任务数 56；20/24 核时预期为 3-head，28/32 核时为 2-head；另覆盖 g-only `HK:HV=1:7` |
| 3 | 总任务数 65 且跨 5 个 sequence；24/28/32 核时预期为 3-head，20 核时 `headsPerCore=4` 并覆盖 sequence 边界拆分 |
| 4 | 总任务数 160，覆盖 4-head work unit、跨 round 和 raw K 复用 |
| 5 | gk-only 总任务数 160，覆盖每个 head 独立 prepared kg |
| 6、9 | g/gk、整 chunk、BF16 initial state、final state |
| 7、8、10 | tail=63/tail=1、FP32 常驻 state、lookahead、末 chunk 分支 |
| 11 | 非连续 `u` ND view |
| 12 | 17 个 chunk，覆盖流水 credit 复用 |
| 13、14 | 变长显式/自动 `chunk_indices`，覆盖 g/gk 与跨 sequence chunk |
| 15、16 | dense/varlen 总任务数 64；28 核预期为 22 block，即 `21x3 + 1x1` |
| 17-48 | BF16/FP32 gate x g/gk x exp/exp2 x BF16/FP32 initial state x `[K,V]/[V,K]` 完整 32 组合；每条 `T=129`，同时覆盖 full chunk 与 tail=1 |

分核覆盖由整个矩阵共同完成，不假设同一 case 在所有 SoC 上有相同的 `activeHeadCount`：

| 物理 AIC 核数 `C` | 1-head | 2-head | 3-head | 4-head |
| --- | --- | --- | --- | --- |
| 20 | case 0 | case 1 | case 2 | case 4/5 |
| 24 | case 0 | case 1 | case 2/3 | case 4/5 |
| 28 | case 0 | case 1/2 | case 3、15/16 | case 4/5 |
| 32 | case 0 | case 1/2、15/16 | case 3 | case 4/5 |

上表依据 `headsPerCore=ceil(totalHeadTasks/C)` 和每 work unit 最多四个 head 推导，属于预期
映射；实际 `blockDim` 和 active head 记录仍需在对应 SoC 运行后回填。

`atk_chunk_fwd_h_negative.json` 共 25 条，case 0-24 逐项覆盖稳定入口公开拦截，包括 gate 二选一、固定属性、rank/dtype/shape、
K/V、g-only head 比例、gk-only prepared kg head 数、gate/state dtype、state layout、变长
batch 与 `cu_seqlens/chunk_indices` 规范序列。

反向 case 使用 NPU-only `run`，不进入精度比较。每条 case_spec 都显式配置
`expected_return_code=0`、`expected_exception=RuntimeError` 和 `expected_message`。这些公开参数
校验发生在稳定 wrapper 调用 aclnn 之前，因此返回码 0 表示 executor 严格核验异常后正常结束，
不是 aclnn 返回码。executor 同时核验异常的精确类型和消息子串，其他异常或意外成功都会使
该 case 执行失败；共享结果检查器还会核对 ATK 报告中 `total>0`、`exec_pass=total` 且
`exec_fail=0`。

## 模板实例与 TilingKey 覆盖

Kernel 顶层入口模板参数依次为 gate dtype、`V_DIM`、g/gk、exp/exp2、state dtype 和 state
layout。`V_DIM` 当前固定为 128，其余五项各有两个取值，因此注册
`2 x 1 x 2 x 2 x 2 x 2 = 32` 个可达实例。canonical matrix 均显式传入 BF16 或 FP32
`initial_state`，避免把 `initial_state=None` 的 FP32 默认规则与 state dtype 模板覆盖混在一起；
专项 case 仍覆盖无 initial state 路径。

下表的 `K00`-`K31` 是 ATK 交付件中的符号编号，不是 CANN 生成的数值 TilingKey。数值 key
由目标 CANN 版本编码，不能写入用例或作为跨版本稳定语义。每条 accuracy matrix case 的
`T=129` 同时包含两个 full chunk 和一个 tail=1，因此普通与边界栏使用同一个 case id。

| 符号实例 | gate dtype | gate | 指数 | state dtype | state layout | accuracy 普通/边界 | performance | mss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K00 | BF16 | g | exp | BF16 | `[K,V]` | 17 | 7 | 0 |
| K01 | BF16 | g | exp | BF16 | `[V,K]` | 18 | 8 | 1 |
| K02 | BF16 | g | exp | FP32 | `[K,V]` | 19 | 9 | 2 |
| K03 | BF16 | g | exp | FP32 | `[V,K]` | 20 | 10 | 3 |
| K04 | BF16 | g | exp2 | BF16 | `[K,V]` | 21 | 11 | 4 |
| K05 | BF16 | g | exp2 | BF16 | `[V,K]` | 22 | 12 | 5 |
| K06 | BF16 | g | exp2 | FP32 | `[K,V]` | 23 | 13 | 6 |
| K07 | BF16 | g | exp2 | FP32 | `[V,K]` | 24 | 14 | 7 |
| K08 | BF16 | gk | exp | BF16 | `[K,V]` | 25 | 15 | 8 |
| K09 | BF16 | gk | exp | BF16 | `[V,K]` | 26 | 16 | 9 |
| K10 | BF16 | gk | exp | FP32 | `[K,V]` | 27 | 17 | 10 |
| K11 | BF16 | gk | exp | FP32 | `[V,K]` | 28 | 18 | 11 |
| K12 | BF16 | gk | exp2 | BF16 | `[K,V]` | 29 | 19 | 12 |
| K13 | BF16 | gk | exp2 | BF16 | `[V,K]` | 30 | 20 | 13 |
| K14 | BF16 | gk | exp2 | FP32 | `[K,V]` | 31 | 21 | 14 |
| K15 | BF16 | gk | exp2 | FP32 | `[V,K]` | 32 | 22 | 15 |
| K16 | FP32 | g | exp | BF16 | `[K,V]` | 33 | 23 | 16 |
| K17 | FP32 | g | exp | BF16 | `[V,K]` | 34 | 24 | 17 |
| K18 | FP32 | g | exp | FP32 | `[K,V]` | 35 | 25 | 18 |
| K19 | FP32 | g | exp | FP32 | `[V,K]` | 36 | 26 | 19 |
| K20 | FP32 | g | exp2 | BF16 | `[K,V]` | 37 | 27 | 20 |
| K21 | FP32 | g | exp2 | BF16 | `[V,K]` | 38 | 28 | 21 |
| K22 | FP32 | g | exp2 | FP32 | `[K,V]` | 39 | 29 | 22 |
| K23 | FP32 | g | exp2 | FP32 | `[V,K]` | 40 | 30 | 23 |
| K24 | FP32 | gk | exp | BF16 | `[K,V]` | 41 | 31 | 24 |
| K25 | FP32 | gk | exp | BF16 | `[V,K]` | 42 | 32 | 25 |
| K26 | FP32 | gk | exp | FP32 | `[K,V]` | 43 | 33 | 26 |
| K27 | FP32 | gk | exp | FP32 | `[V,K]` | 44 | 34 | 27 |
| K28 | FP32 | gk | exp2 | BF16 | `[K,V]` | 45 | 35 | 28 |
| K29 | FP32 | gk | exp2 | BF16 | `[V,K]` | 46 | 36 | 29 |
| K30 | FP32 | gk | exp2 | FP32 | `[K,V]` | 47 | 37 | 30 |
| K31 | FP32 | gk | exp2 | FP32 | `[V,K]` | 48 | 38 | 31 |

生成器会对 accuracy、performance 和 mss 的 canonical matrix 做集合相等与去重断言。实际 key
证据必须在 A2、A3、A5 clean build 后从 `binary_info_config.json` 确认 32 个唯一 key，并从
编译 wrapper/dump 建立“数值 key 到六个模板参数”的一一映射；运行时再将每条 ATK case 的
host tiling 记录与该映射核对。未取得对应 SoC 的 build/runtime 记录前，不将输入条件推断写成
实际选择结论。

## 性能矩阵

`atk_chunk_fwd_h_perf.json` 共 44 条，只在 NPU 节点运行 `performance_device`，输入在
`init_by_input_data` 阶段生成并缓存，不把随机输入构造计入重复 launch。

- 3 条小型 smoke：g、GVA、gk。
- `B=2,HK=16,HV=32,T=11264` 原目标场景及对应 `B=1` 场景。
- 同一 `B=1,HK=16,HV=32,T=11264` 下 BF16/FP32 initial state 对照。
- `B=1,HK=HV=32,T=11264` 下完整 32 项顶层模板矩阵，共 32 条。
- 64 个变长 sequence、`T=65536`，显式规范 `chunk_indices`。
- `B/HK/HV/T` 为 `4/96/96/128`、`1/32/32/160`、`6/6/6/1084`、
  `1/12/12/1084` 的模型场景。

性能结论以 ATK profile 产物为准，本 README 不预填未在当前提交上实测的耗时。

## 确定性与内存检测

`atk_chunk_fwd_h_mss.json` 共 32 条，每个顶层模板签名恰好一条。同一文件供 `accuracy_dc`
与 mssanitizer 使用。默认 shape 为 `B=HK=HV=1,T=129`；为兼顾同步和分核路径，以下签名在
不改变模板参数的前提下使用专项 shape：

- mss 0/1：dense/varlen 总任务数 64，覆盖跨 sequence 和不同核数下的分核映射；
- mss 3：无 initial/final state 的单 token VEC-only 路径；
- mss 4：`T=1025` 的流水 credit 复用；
- mss 7/18：FP32 state 常驻、lookahead、tail=1/tail=63；
- mss 16/17/23/24：2/3/4-head、跨 round、raw K 复用和 gk 独立 key；
- mss 19：变长显式 `chunk_indices` 和跨 sequence chunk；
- mss 31：gk、FP32 state、exp2、`[V,K]` 的 tail=1 路径。

单条 case 的 active head 数会随物理核数变化；实际 `blockDim` 和 active head 记录仍需在对应
SoC 运行后回填。

使用 sanitizer 前必须确认当前 OPP 为 sanitizer 编译版本，且运行日志明确显示目标 kernel
已启用对应工具；仅看到未激活提示不能作为内存无异常结论。

## 运行

在仓库根目录准备好 ATK、CANN、OPP 和 `fla_npu` Python 包后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=negative
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=mssanitizer
```

按统一入口重新生成 ATK case 模板：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=gen_cases
```

直接重建仓内冻结 JSON：

```bash
python3 tests/atk/chunk_fwd_h/gen_chunk_fwd_h.py \
  --output-dir tests/atk/chunk_fwd_h \
  --summary
```

生成后的 case id、`case_key` 和 seed 是确定的；修改矩阵后应重建四份 JSON 并做 JSON、YAML
和 Python 静态校验。ATK 输出目录、profile、XLSX、sanitizer 日志和 Python 缓存不得提交。
