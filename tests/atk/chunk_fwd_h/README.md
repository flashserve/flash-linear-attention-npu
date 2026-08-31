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

`atk_chunk_fwd_h.json` 共 25 条，只包含正向精度用例。

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
| 17-24 | g/gk x BF16/FP32 initial state x exp/exp2 完整 8 组合 |

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

Kernel 入口以 `V_DIM` 为编译期模板参数。当前模板声明仅注册 `V_DIM=128`，host 在完成
`V=128` 校验后调用 `GET_TPL_TILING_KEY(128)`。A2、A3、A5 的 clean build metadata 均确认
唯一生成 key 为 `0`。该数值是当前目标 CANN 版本的模板编码细节，不写入用例或公共接口，
也不作为跨版本稳定语义。

| 模板实例 | 生成 key | 选择条件 | 普通用例 | 边界/分核用例 | SoC | build/runtime 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `V_DIM=128` | `0` | `K=V=128`、`chunk_size=64` | 6/9/17-24 | 0-5/7-16；mss 0-11 | A2/A3/A5 | A2/A3/A5 clean build metadata 均仅含 key 0；A2 稳定入口 g(exp2、无 initial) 与 gk(exp、BF16 initial) 的 65-token tail smoke 已闭环 host/key 0 launch；A5 dump wrapper 为 `chunk_fwd_h<128>` 且 host/runtime 实际 key 0；A3 仅有 build 证据，无实机 runtime 结果 |

所有正向 case 都选择同一个 `V_DIM=128` 模板实例。gate dtype、g/gk、exp/exp2、state
dtype 和 state layout 是该 binary 内的运行时分支，不是额外的模板 key。A3 当前结论只覆盖
clean build 生成结果，不声称已完成 runtime 选择验证。

## 性能矩阵

`atk_chunk_fwd_h_perf.json` 共 20 条，只在 NPU 节点运行 `performance_device`，输入在
`init_by_input_data` 阶段生成并缓存，不把随机输入构造计入重复 launch。

- 3 条小型 smoke：g、GVA、gk。
- `B=2,HK=16,HV=32,T=11264` 原目标场景及对应 `B=1` 场景。
- 同一 `B=1,HK=16,HV=32,T=11264` 下 BF16/FP32 initial state 对照。
- `B=1,HK=HV=32,T=11264` 下 g/gk x BF16/FP32 initial state x exp/exp2 共 8 条。
- 64 个变长 sequence、`T=65536`，显式规范 `chunk_indices`。
- `B/HK/HV/T` 为 `4/96/96/128`、`1/32/32/160`、`6/6/6/1084`、
  `1/12/12/1084` 的模型场景。

性能结论以 ATK profile 产物为准，本 README 不预填未在当前提交上实测的耗时。

## 确定性与内存检测

`atk_chunk_fwd_h_mss.json` 共 12 条。矩阵整体按上表分别覆盖 20/24/28/32 核时可达的
1/2/3/4-head work unit；单条 case 的 active head 数会随物理核数变化。其余覆盖包括 g/gk、
BF16/FP32 state、tail、FP32 常驻/lookahead、dense/varlen、sequence 边界和总任务数 64 的
分核路径。同一文件供 `accuracy_dc` 与 mssanitizer 使用。

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
