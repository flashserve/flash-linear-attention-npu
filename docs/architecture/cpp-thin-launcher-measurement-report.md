# fla_npu C++ 薄层：方案、实测与 ctypes 差异报告

> 关联 PR：#496；设计文档：[cpp-thin-launcher-design.md](./cpp-thin-launcher-design.md)
> 关联 issue：[#491](https://github.com/flashserve/flash-linear-attention-npu/issues/491)
> 日期：2026-09-08

## 1. 结论摘要

1. 薄层把算子 host enqueue 从"Python/ctypes 解释层"搬进"编译型 C++"，同
   kernel、同 OPP，输出/state 与 ctypes **逐位一致**。
2. recurrent 实测：ctypes public ~0.52-0.65 ms → 薄层 public（M4 后）
   ~0.10 ms，薄层 C++ 直连 ~0.043 ms；vllm custom public ~0.073 ms。
3. conv1d（PR #390 ABI 验证环境）：ctypes ~0.67-0.89 ms → 薄层 ~0.074 ms，
   parity 逐位一致。
4. 兼容性：有效输入路径与 ctypes 完全等价（签名/mutation/数值/stream 顺序）；
   非法输入只保证"会报错"，不保证异常类型与 ctypes 一致（有意为之，薄层不做
   完整 Python 前置校验）。
5. 薄层默认编译、默认启用（编译成功且算子已适配时），未适配/未编译自动回退
   ctypes；编译期不依赖 torch_npu。

## 2. 方案回顾

```text
fla_npu.ops.ascendc.npu_xxx(...)        # 公共 API（不变）
   └─ _thin.py（默认启用；FLA_NPU_THIN_LAUNCHER=0 关闭）
        └─ fla_npu._C_thin（C++，默认编译；FLA_NPU_BUILD_THIN=0 关闭编译）
             ├─ Runtime：dlopen 包内 libcust_opapi.so/libopapi.so，符号缓存
             ├─ TensorDesc：ND + storage 基址/offset/strides（与 ctypes 同语义）
             └─ 两段式 aclnn*GetWorkspaceSize + aclnn* launch
未适配算子 / 扩展缺失 → 自动回退 ctypes
```

关键决策（详见设计文档）：

- 不依赖 torch_npu 头/库（编译期），stream 由 Python 侧传入；
- workspace 仍每调用现分配（vllm custom 同款），per-stream 池留 M4 后续；
- 不复刻 vllm-ascend 的 NPUBridge/static_cast，descriptor 全部自建；
- 保留 ctypes fallback 与 mutation/autograd 契约；
- 每算子只需一份 ABI 适配（~50-100 行），按 profile 增量接入。

## 3. 实测方法与环境

环境：192.168.9.221（Ascend 910B3，CANN 9.1.0，fzy：Python 3.11 /
torch 2.9.0 / torch_npu 2.9.0.post2）。

方法：隔离子进程分别加载 FLA 与 vllm custom（避免同名 aclnn 符号冲突）；
`time.perf_counter()` 包住单次调用（不 synchronize）统计 host enqueue，
排序取 P50；NPU event 用 `torch.npu.Event` 统计。warmup 10-20 次，
正式 100-200 次。

## 4. 实测结果

### 4.1 RecurrentGatedDeltaRule（非连续 state、q=1、200 次）

shape：state `(blocks=1448, 16, 128, 128)`，stride gap=16384，offset=12288。

| 路径 | host P50 | event P50 | 说明 |
|---|---|---|---|
| ctypes public（原实现） | ~0.52-0.65 ms | ~0.14-0.34 ms | batch 32-128；共享机波动 |
| 薄层 C++ 直连 | ~0.043-0.053 ms | 同 kernel | 不含 Python wrapper |
| 薄层 public（M4 前） | ~0.165-0.17 ms | 同 kernel | wrapper：current_stream + bind |
| 薄层 public（M4 后） | ~0.098-0.108 ms | 同 kernel | raw stream 查询 + mutation 快路径 |
| vllm custom public | ~0.073 ms | ~0.14-0.28 ms | 对照 |

结论：host 从 ctypes 的 ~6-7x custom 降到 M4 后的 ~1.4-1.5x；C++ 执行体
本身已快于 custom。event（device）两侧基本一致，未复现 issue 中的 +22-31%
device 差距（隔离单算子场景）。

### 4.2 CausalConv1d（PR #390 ABI 验证环境，update 形态）

shape：x `[8, 4096]` bf16，weight `[4, 4096]`，conv_state `(blocks, 6, 4096)`
非连续（gap=565248），activation=silu，run_mode=1，预分配 out。

| 路径 | host P50 | parity |
|---|---|---|
| FLA ctypes（#390 wheel） | ~0.67-0.89 ms | 基准 |
| 薄层（#390 ABI，直接写 out） | ~0.074 ms | out/state max_abs = 0 |

说明：#390 未合入 upstream，conv1d 薄层当前只在验证环境按 #390 ABI 跑通；
正式落地等 #390 合入后重写 `csrc_thin` 的 conv1d 适配。

### 4.3 正确性 / 回归

- recurrent：ctypes vs thin 在 q=1/q=4、连续/非连续 state 下 out/state
  **逐位一致**（max_abs = 0）。
- mutation：state `_version` 递增、requires_grad 拒绝（与 ctypes 同契约）。
- determinism / stream 顺序：重复调用 bit 一致；后续依赖 kernel 顺序正确。
- 自动化测试：`test_thin_launcher.py` + `test_thin_customer_compat.py`
  （7 passed + 3 subtests）。

## 5. 与 ctypes 的差异对照

| 维度 | 原 ctypes | 薄层（thin） |
|---|---|---|
| host 执行 | Python 校验 + ctypes 对象管理 + 两段式调用 | C++ descriptor 建销 + 两段式调用 |
| 数值/语义（合法输入） | 基准 | 与 ctypes 逐位一致 |
| 每调用 Python 校验 | 完整 dtype/device/shape 校验 | 少量校验（性能路径） |
| 非法输入报错 | 多为 `TypeError`/`RuntimeError`（Python 校验产生） | 常为 aclnn `RuntimeError`；类型不保证一致 |
| stream | 每调用 `torch.npu.current_stream()` | 每调用 `_npu_getCurrentRawStream`（~1us；无进程级缓存，多线程安全） |
| workspace | 每调用 `torch.empty` | 每调用 C++ `at::empty`（暂同策略） |
| 编译依赖 | 无（纯 Python） | 编译期仅 torch/CANN；不依赖 torch_npu |
| 默认行为 | 默认 | 默认编译、默认启用（可 `=0` 关闭），缺失自动回退 |
| in-place/autograd | 经 `_wrap_mutable_direct_op` | 同（mutation wrapper 增加位置参数快路径） |
| 新增算子成本 | 1 个 Python wrapper（+可选 argtypes） | C++ 适配 + pybind + `_thin.py` + 白名单 |
| 适用范围 | 全部算子 | 白名单内算子；其余回退 ctypes |

### 已知且有意的差异

1. 非法输入报错类型不保证一致（薄层为速度跳过 Python 前置校验，只保证会报错）。
2. 白名单外算子不享受薄层加速（仍 ctypes）。
3. conv1d（旧 ABI 的 `npu_causal_conv1d` 与 #390 的 `causal_conv1d_update`）在
   #390 合入前均**不启用** thin，始终回退 ctypes。

### 修复记录：vLLM 多线程多 stream 崩溃（2026-09-10）

- 现象：vLLM 单 curl 崩溃（设备非法地址）。根因是 `_thin.py` 早期提交
  （`cd206c27 perf: cache current stream ...`）用**进程级全局变量**缓存
  stream 指针并 monkeypatch `torch.npu.set_stream`；vLLM 多 worker 线程各用
  独立 stream 时，线程 A 写入的缓存被线程 B 读到，kernel 下发到错误 stream，
  破坏跨 stream 依赖并访问未就绪内存。
- 修复：删除全局缓存与 monkeypatch，改为每调用经
  `torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device())` 读取
  调用线程当前 stream 的原始指针（实测 ~1.2us/次，对比对象路径 ~23.6us），
  老版本 torch_npu 无该接口时回退 `torch.npu.current_stream()`。
- 回归：新增 `test_thin_stream_interleaving.py::test_no_process_global_stream_cache`
  与 `::test_threads_use_their_own_streams`（4 线程 × 4 stream，barrier 对齐后
  各自下发，事件须落在本线程 stream 且结果与 ctypes 逐位一致）。

#### vLLM 侧 A/B（80.5.9.126，512-token 单请求，异步）

| 配置 | 结果 | 用时 |
|---|---|---|
| Recurrent thin + Conv update thin（原全局缓存） | 失败，约 257 token 设备异常 | - |
| Recurrent thin + Conv update thin（原缓存，同步） | 200，512 token | 71 s |
| 两个算子都走 ctypes | 200，512 token | 41 s |
| 仅 Recurrent thin | 200，512 token | 38 s |
| 仅 Conv update thin | 200，512 token | 42 s |
| 两个算子都 thin，取消 stream 缓存 | 200，512 token | 36 s |

plog 首个可观测故障是 MoE kernel 的 GM 非法地址，`aclnnFlaCausalConv1d
361001` 是 stream 进入异常态后的连带错误；同步与 thin 拆分实验证明 MoE 不是
必要条件。修复后用同一 512-token 请求通过。

#### Launcher 侧 A/B（221，910B3）

同一个 512 验证 wheel（`causal_conv1d` + `recurrent_gated_delta_rule`，仅编
这两个算子约 5 min），只替换 `_thin.py`：

| `_thin.py` | 4 线程 × 4 stream 交替 Recurrent+Conv | 24 次轮换 stream 反复调用 |
|---|---|---|
| 旧（全局 `_CURRENT_STREAM_PTR` 缓存） | FAIL：线程 2/3 读到同一个缓存指针（1447952608），自身 stream 为 1507399680/1513441440 | PASS* |
| 新（每调用 raw stream） | PASS | PASS |

\* 旧实现对“单线程轮换 stream”恰好不暴露问题（每次 `set_stream` 都会刷新缓存），
只有多线程并发才能稳定复现，这正是 vLLM worker 的形态。

stream 查询开销（同机 2000 次 P50）：缓存 0 ms、raw accessor 0.0012 ms、
`torch.npu.current_stream().npu_stream` 0.0233 ms。thin recurrent host P50
0.1103 ms（缓存）→ 0.1167 ms（raw）。按每 decode step ~30 次 recurrent/conv
调用估算：raw 方案约 +0.04-0.2 ms/step，若改用每调用 `current_stream()` 的
最小修法则约 +0.7 ms/step；因此采用 raw accessor 兼顾正确性与 host 开销。

> 待办：vLLM 侧仍需按验收清单完成重复 20 次、并发 8/32/64、TTFT/TPOT 复核；
> 以及此前观察到的 Conv1d 多 batch 输出不一致问题（与 stream 修复无关，需单独
> 做算子精度回归）。

## 6. 变更文件清单

- `torch_custom/fla_npu/csrc_thin/`：runtime / tensor_desc / ops_recurrent_gdn /
  ops_causal_conv1d（占位）/ pybind
- `torch_custom/fla_npu/setup.py`：`_C_thin` 默认编译分支
- `fla_npu/ops/ascendc/_thin.py`：薄层 Python 适配（每调用 raw stream 查询）
- `fla_npu/ops/ascendc/__init__.py`：默认启用 + 白名单 + mutation 快路径
- `torch_custom/fla_npu/test/test_thin_launcher.py`、
  `test_thin_customer_compat.py`：dispatch/parity/兼容性测试
- `docs/architecture/cpp-thin-launcher-design.md`：设计文档

## 7. 后续计划

- 剩余 public wrapper 差距（~0.03 ms）：pybind 位置参数直传等（可选 M4.2）；
- conv1d：等 #390 合入后按新 ABI 重写并启用；
- device/event 差距：在服务级 profiling 场景复核是否仍存在；
- 按 profile 增量接入其他高频算子。

## 8. mutation 包装层快路径（2026-09-10）

公共入口 `fla_npu.ops.ascendc.*` 会对 `MUTATED_ARGUMENTS` 里的算子包一层
mutation wrapper（拒绝 `requires_grad` 的 state、launch 后
`increment_version`）。旧实现只要该算子有 predicate 就每调用执行
`inspect.signature().bind()` + `apply_defaults()`，而
`npu_recurrent_kda` 恰好有 predicate，因此**每次调用**都付这笔钱，且调用方
无法规避。

改法：把 predicate 从 lambda 改成声明
`MUTATION_FLAGS = {"npu_recurrent_kda": ("inplace_final_state", True)}`，
wrapper 在 wrap 时算好 plan（校验默认值与真签名一致），热路径直接从
args/kwargs 读取 mutated tensor 与 flag。

### 8.1 同一进程 A/B（dummy op + 真签名，排除进程间噪声；us/次）

| 调用形态 | 旧 wrapper | 新 wrapper | 差值 |
| --- | --- | --- | --- |
| KDA `initial_state` 位置、`inplace_final_state=True` | 38.52 | **6.40** | −32.1 |
| KDA `initial_state` 位置、`inplace_final_state=False` | 36.19 | **4.42** | −31.8 |
| KDA `initial_state` 关键字 | 38.82 | 39.15 | +0.3（仍走 bind，按设计不变） |
| GDR `state` 位置 | 5.94 | 5.05 | −0.9 |
| GDR `state` 关键字 | 27.62 | 26.38 | −1.2 |

碎片测量（221，20 参数签名）：`hash(signature)` **30.4us**、
`signature.bind+apply_defaults` **40.6us**、`_mutation_plan`（仅 wrap 时）
9.9us、新热路径判定 **1.23us**。第一条解释了为什么 plan **不能**用
`lru_cache(signature)` 缓存——那样每次调用都要付 30us 的签名哈希，比它省下
的还贵（本改动第一版就踩了这个坑，A/B 显示"优化后更慢"，随后改为闭包持有
plan）。

### 8.2 真实算子（`npu_recurrent_kda`，batch 100×T=1，host P50，910b）

| 调用形态 | 旧 wrapper | 新 wrapper |
| --- | --- | --- |
| `initial_state` 位置、`inplace_final_state=True` | 0.155 ms | **0.092 ms** |
| `initial_state` 位置、`inplace_final_state=False` | 0.153 ms | **0.087 ms** |
| `initial_state` 位置、flag 省略（默认 True） | 0.154 ms | **0.092 ms** |
| `initial_state` 关键字 | 0.154 ms | 0.151 ms（不变） |

### 8.3 正确性门禁

- `torch_custom/fla_npu/test/test_ascendc_mutation_contract.py`：纯 Python
  （不需要 torch/NPU），对 11 种调用形态断言"快路径结果 ≡ 旧 bind 参考实现"，
  并用 `mock.patch.object(inspect.Signature, "bind", side_effect=...)` 硬性
  证明位置参数路径**确实没有**调用 bind；另测 flag 默认值与签名不一致、
  flag 名字写错都会在 wrap 时立刻报错。
- `tests/regression_mutation_contract.py`：安装态 device 回归，
  `inplace=True`→version+1、`inplace=False`→version+0 且返回 scratch state、
  `requires_grad` 在 inplace=True 时拒绝 / inplace=False 时放行、
  ctypes vs thin public 数值逐位一致（910b 上 11 项全 PASS）。
