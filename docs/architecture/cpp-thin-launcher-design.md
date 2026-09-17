# fla_npu C++ 薄层执行器设计（thin launcher）

> 分支：`feat/fla-npu-thin-launcher`（基于最新 main）
> 状态：设计稿 / 落地准备（M0）
> 关联 issue：[flash-linear-attention-npu#491](https://github.com/flashserve/flash-linear-attention-npu/issues/491)

## 1. 背景与结论

在 vLLM-Ascend 服务中，同一个算子（`RecurrentGatedDeltaRule`、`CausalConv1d`）分别走
`fla_npu`（Python ctypes 直调 OPP）和 vLLM-Ascend custom（编译型 C++ torch op）时，
FLA 路径存在稳定的 host enqueue 回退。

复现（192.168.9.221，CANN 9.1.0，SOC 910B3/ascend910b，fzy env：
torch 2.9.0 + torch_npu 2.9.0.post2；FLA 为已安装 26.7.0.dev0 wheel）：

RecurrentGatedDeltaRule，非连续 state，q=1，200 次，blocks=1448，
gap=16384 elements，offset=12288 elements：

| Batch | FLA host P50 | Custom host P50 | host 倍数 | FLA event P50 | Custom event P50 |
|---|---|---|---|---|---|
| 64 | 0.645 ms | 0.075 ms | ~8.6x | 0.144 ms | 0.143 ms |
| 100 | 0.655 ms | 0.074 ms | ~8.9x | 0.212 ms | 0.214 ms |
| 128 | 0.656 ms | 0.072 ms | ~9.1x | 0.272 ms | 0.275 ms |

输出/state 两侧逐位一致（max_abs = 0），说明对比语义有效。

结论：

1. **主因是 host 侧**：FLA wrapper 每次调用做全套 Python 校验、descriptor 建/销、
   int-array 转换、stream 查询、workspace 现分配，host 时间（~0.65 ms）反超
   device 时间，是 issue #491 的核心。
2. **device/event +22~31% 在隔离单算子场景未复现**（两侧 event 几乎相同）。该差距
   更可能来自 FLA 版本差异或服务并发下 host 造成的 stream 空洞，落地后需用服务级
   profiling 复测，不在此设计的前置路径上。
3. `CausalConv1d`（update 变体）同源问题，复现脚本见
   `torch_custom/fla_npu` 测试目录外的服务侧 benchmark（conv1d update 入口在
   当前上游 main 尚未导出，见 §8 open questions）。

## 2. 目标与非目标

目标：

- 把 `recurrent_gated_delta_rule` 与 `causal_conv1d`（含 update/decode 形态）的
  host enqueue 降到与 vLLM-Ascend custom 路径同一量级（~0.05-0.08 ms/调用）。
- 编译型薄层**默认编译、编译成功且算子已适配时默认启用**（可通过
  `FLA_NPU_BUILD_THIN=0` 关闭编译、`FLA_NPU_THIN_LAUNCHER=0` 关闭运行时分发）；
  kernel 继续使用 fla_npu 自带 OPP（`libcust_opapi.so`）。未适配/未编译时自动
  回退 ctypes，保证功能可用。
- 薄层 .so **不依赖 torch_npu 头文件/库/ABI**（编译期），只依赖 torch(C++ ABI)
  + CANN acl；stream 由调用方显式传入。

兼容性说明：薄层是性能路径，只保证**输入合法**时与 ctypes 逐位等价；非法输入的
报错类型/文案不保证与 ctypes 一致（ctypes 会先做 Python 校验抛 TypeError 等，
薄层通常表现为 aclnn RuntimeError）。已通过客户视角兼容性测试
（`test_thin_customer_compat.py`）覆盖有效输入签名/数值/mutation/stream 顺序。

非目标（本阶段）：

- 不改 kernel、不改 `libcust_opapi.so` 的 aclnn ABI。
- 不做"零 torch/torch_npu 调用"的 standalone runtime（那是后续独立方向）。
- 不做跨调用 descriptor/executor 缓存（vllm 也没做；收益小、生命周期复杂）。

## 3. 架构决策

### 3.1 形态：pybind 扩展模块 `fla_npu._C_thin`

```text
fla_npu.ops.ascendc.npu_xxx(...)          # Python 入口（保持不变）
  └─ FLA_NPU_THIN_LAUNCHER=1 时走 _thin.py
       └─ fla_npu._C_thin (pybind C++)
            ├─ ThinRuntime：dlopen libcust_opapi.so/libopapi.so，dlsym 缓存
            ├─ TensorDesc → aclCreateTensor（ND + strides/storage 语义）
            ├─ per-stream workspace 复用（低优先级，M4）
            └─ 两段式 aclnn*GetWorkspaceSize + aclnn* launch
```

关键决策（相对早前草案的修订）：

1. **workspace 池降级为可选优化**。vllm custom 路径同样每调用 `at::empty`
   workspace，性能仍然远好于 ctypes，证明主因不是 workspace 分配，而是
   Python/ctypes 解释层。M1 薄层第一版刻意保留"每调用现分配 workspace"
   （C++ `at::empty`），用于验证主因假设；M4 再决定是否移植 per-stream 单块池
   （参考 torch_npu `NPUWorkspaceAllocator` 的 2 MiB 对齐单块模式）。
2. **不复制 vllm-ascend 的 NPUBridge/static_cast 思路**。descriptor 一律由
   薄层自建 TensorDesc 生成：`sizes/strides/storage_offset/data_ptr/dtype/
   storage_numel`，format 固定 ND（与现 ctypes `nd_tensor` 语义一致），不读
   torch_npu storage 描述，避免 ABI 静默耦合。
3. **stream 由 Python 调用侧按“每次调用”解析后传入**（优先
   `torch_npu._C._npu_getCurrentRawStream(device)` 的原始指针，缺失时回退
   `torch.npu.current_stream().npu_stream`），薄层不查询、不创建 stream。
   这样薄层 .so 不链接 torch_npu，同时与调用线程的当前 stream 严格保序；
   **不做进程级缓存**——vLLM 等多线程 server 会在不同线程使用不同 stream，
   全局缓存会把某线程的 stream 泄漏给其它线程（kernel 下发到错误 stream，
   表现为非法地址/顺序破坏）。
4. **符号解析运行时进行**：复用现有 `fla_npu.load_ascendc_opapi_libraries()`
   的路径约定（`FLA_NPU_OP_API_LIB` 指向包内 `libcust_opapi.so`），C++ 侧
   `dlopen` 同一路径并缓存 `dlsym` 结果，等价 ctypes `_AclnnRuntime.symbol()`。
5. **ctypes 作为 fallback**：薄层未编译、import 失败或算子未实现时，自动回退
   现有 ctypes 路径，保证任何环境不破坏现状（默认开关见目标节）。

> 注意：conv1d（旧 ABI 的 `npu_causal_conv1d`）当前**不在** thin 白名单内，
> 始终回退 ctypes；待 PR #390 合入并按新 ABI 完成适配后再启用。

### 3.2 每个算子的 ABI 事实（落地前必须逐字核对 OPP 头）

ABI 唯一权威是包内 OPP 的 `op_api/include/aclnnop/aclnn_*.h`（当前仓库源码不含
该头，随 run 包安装）；开发期以 `_aclnn_ctypes.py::_GET_WORKSPACE_ARGTYPES` 和
`_runtime.py` 为镜像基线。

- `aclnnRecurrentGatedDeltaRule`：query/key/value/state/beta/actual_seq_lengths/
  ssm_state_indices/g/gk/num_accepted_tokens 为 tensor（可选以空描述符表示），
  `scale` 为 **float**，输出 out 为 tensor。
- `aclnnCausalConv1d`（上游 main 旧形态）：x/weight/bias/conv_states +
  `query_start_loc/cache_indices/initial_state_mode/num_accepted_tokens` 四组
  int-array + activation/pad_slot/run_mode/head_num 标量 + out。
- `aclnnCausalConv1d`（**PR #390 新形态**，FLA 服务实际使用，见
  [flash-linear-attention-npu#390](https://github.com/flashserve/flash-linear-attention-npu/pull/390)）：
  x/weight/bias/conv_states + 四组 device tensor metadata
  （query_start_loc/cache_indices/has_initial_state/num_accepted_tokens）+
  对应 `*_cpu` int-array + `activation` 为 char* 字符串 +
  pad_slot_id/null_block_id/run_mode/head_num/max_query_len 标量 + out。
  `causal_conv1d_update` 即该 ABI 的 update 形态（run_mode=1，preallocated
  out）。vLLM-Ascend custom 侧对应 PR #8256 kernel。
- vLLM-Ascend 最新 main 的同名 aclnn 为另一套 ABI（四组 metadata 是可选
  tensor），二者不可混用；薄层必须跟 **FLA 自己 OPP** 的签名。

### 3.3 目录与文件规划

新增（放在 fla_npu 独立源码目录，不进入纯 Python wheel 的默认打包路径）：

```text
torch_custom/fla_npu/csrc_thin/
  include/thin_launcher/
    runtime.h        # dlopen/dlsym 缓存、错误转换
    tensor_desc.h    # at::Tensor -> aclTensor（RAII）
    workspace.h      # per-stream workspace 池（M4）
    ops.h            # 算子适配器声明 + aclnn 函数指针 typedef
  src/
    runtime.cpp
    tensor_desc.cpp
    workspace.cpp
    ops_recurrent_gdn.cpp
    ops_causal_conv1d.cpp
    pybind.cpp
```

修改：

- `setup.py`：新增 `FLA_NPU_BUILD_THIN=1` 分支，用 `torch.utils.cpp_extension`
  编 `fla_npu._C_thin`（include：torch + CANN acl；不 include torch_npu）。
- `fla_npu/ops/ascendc/__init__.py`：`_get_direct_op` 增加薄层优先 + fallback。
- 新增 `fla_npu/ops/ascendc/_thin.py`：薄层适配层（签名对齐、mutation 契约）。
- 新增测试与 benchmark（见 §5）。

## 4. 落地顺序（M0-M5）

- M0（当前）：本设计稿 + 分支 `feat/fla-npu-thin-launcher`。
- M1：`csrc_thin` 骨架 + runtime/tensor_desc + `aclnnRecurrentGatedDeltaRule`
  打通；host enqueue 对比目标 ≤0.10 ms/调用；与 ctypes 输出做 bit 级一致校验。
- M2：`aclnnCausalConv1d`（fwd/update 形态；先解决 §8 的入口/ABI 问题）。
- M3：构建/打包（wheel 内嵌 .so 与 OPP 布局兼容）、dispatch/fallback、CI 化。
- M4：可选优化：per-stream workspace 池、预分配输出、shape 稳定 fast-path、
  GIL 释放评估。
- M5：服务级复测（decode step / TP 通信同步），决定是否继续 recurrent q=1
  kernel 专项。

## 5. 验收与回归协议

1. 数值一致性：薄层 vs ctypes 同一 OPP kernel，输出应 bit 级一致（dtype/shape/
   strides 完全相同时）。
2. host enqueue benchmark：复用服务侧两个对比脚本的隔离子进程方案：
   - recurrent：非连续 state `(blocks=1448/2574, Hv=16, D=128)`，gap/offset 按
     issue（16384/12288 或脚本默认 32768/12288），q=1/q=4，batch 32/64/100/120/
     128，warmup≥20、iterations≥200，P50 host enqueue + NPU event。
   - conv1d：conv state `(blocks, 6, 4096)` bf16，gap=565248，offset=0，
     x `[B, 4096]`，weight `[4, 4096]`，activation=silu、run_mode=1。
3. 正确性对照：FLA vs vLLM-Ascend custom 输出/state allclose（现有脚本 rtol/atol
   5e-2；实测可达 max_abs=0）。
4. 服务级：decode step wall / TP AllReduce 等待时间消融（host 修复后 collective
   对齐应改善）。

## 6. 构建与打包

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或本机 CANN 路径
cd torch_custom/fla_npu
python setup.py build_ext --inplace    # 默认编译薄层；FLA_NPU_BUILD_THIN=0 关闭
```

- OPP（`libcust_opapi.so`）仍由 run 包流程安装到
  `fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/`，薄层运行时按
  `FLA_NPU_OP_API_LIB` / 包内路径解析，不重复携带。
- wheel 中 `.so` 命名 `_C_thin*.so`，放 `fla_npu/` 包目录，通过
  `import fla_npu._C_thin` 加载；未携带时不阻塞 import。

## 7. 性能预算与验收红线

- M1/M2 验收红线：host enqueue P50 ≤ 0.10 ms/调用（目标 0.05-0.08），相对
  vLLM custom ≤ 1.5x；数值一致通过；ctypes fallback 行为不变。

## 8. Open Questions

1. `causal_conv1d_update` 入口来源已确认：**flash-linear-attention-npu PR #390**
   （feat(causal_conv1d): add fn/update APIs and device metadata，未合入）。
   它把 `aclnnCausalConv1d` 升级为"device tensor + `*_cpu` int-array 双通道 +
   activation 字符串 + null_block_id/max_query_len"的 ABI。FLA 侧用 #390、
   vLLM-Ascend 侧用 #8256。因此 M2 的 conv1d 薄层必须以 #390 ABI 为准
   （当前 `ops_causal_conv1d.cpp` 只是上游旧形态占位，M2 需重写），且依赖
   #390 合入后对应 OPP 重建。
2. device/event +22~31% 未在隔离复现中出现：落地后用同版本 FLA wheel + 服务级
   profiling 复核；若仍存在再开 kernel 专项。
3. torch_npu 版本差异（复现机 2.9.0.post2 vs issue 2.10.0.post4）可能影响 FLA
   host 绝对值（复现中 FLA host ~0.65 ms 高于 issue 的 ~0.30 ms），验收以
   "薄层相对同一环境 ctypes 的下降倍数" 为准，避免跨环境绝对比较。
