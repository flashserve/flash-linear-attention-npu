# 热路径 host 开销：本方案 vs vLLM-Ascend（2026-09-15 重测，910B3）

环境：221，`env_full`（OPP 由 main 现编），launcher 由本分支现场编译
（`libfla_npu_stable_623.so`），卡 6。两侧跑同一组对比脚本
（`bench_recurrent_variants.py` / `bench_conv1d_vllm.py`）与同进程拆解脚本，形状与
计时口径一致：host 时间是 Python 调用把活交给 aclnn 的墙钟时间（计时区间内不
synchronize），device 时间是 NPU event。

> 本文件先前（2026-09-14）记的是 0.1665 / 0.1797 ms 那一组，当时 221 的 load 很高。
> 在空闲卡上按同一脚本重测后绝对值明显不同，**conv1d 的“0.79×（我们更快）”不复现**；
> 下文是重测结果。机器负载会移动绝对值，同一次运行内的比值才是可比量。

## recurrent GatedDeltaRule：batch 128、q=1、非连续 state、100 次

同进程四层拆解（300 次，改前 → 改后）：

| 层 | 改前 host P50 | 改后 host P50 |
| --- | --- | --- |
| 公共 API（dispatcher + `_stable.py` wrapper） | 0.152 ms | **0.101 ms** |
| `_stable.py` wrapper（跳过 dispatcher） | 0.097 ms | **0.076 ms** |
| `torch.ops` 直调 + 每次查 stream | 0.071 ms | 0.074 ms |
| `torch.ops` 直调 + 固定 stream（C++/boxed 底） | 0.062 ms | 0.061 ms |

与 vLLM / ctypes 对照（同卡，脚本模式）：

| 路径 | host P50 | host mean | device P50 |
| --- | --- | --- | --- |
| FLA stable（公共 API） | **0.101 ms** | — | 0.276–0.298 ms |
| FLA stable（`_stable` wrapper 直调） | 0.076 ms | — | — |
| FLA ctypes（参考实现） | 0.6198 ms | 0.9346 | 0.3751 ms |
| vLLM-Ascend custom | **0.0748 ms** | 0.0797 | 0.2730 ms |

- 公共 API / vLLM = **1.35×**；`_stable` wrapper / vLLM ≈ **1.0×**；
  ctypes / vLLM = 8.3×。
- device 侧 = **1.01–1.09×**（同一个 kernel），差异全在 host。
- 改后仍剩下的 0.101 − 0.076 = 0.025 ms 是 dispatcher 记账 + mutation 包装；
  0.076 − 0.061 = 0.015 ms 是每次 stream 查询与 boxed 拆栈。

## causal_conv1d update：batch 100、连续 state、200 次

去掉两处多余拷贝之后的最后一次测量（state 是带 12288 元素 storage offset 的
contiguous view，`scenario_conv1d_update_offset_state` 覆盖的就是它）：

| 路径 | host P50 | device P50 |
| --- | --- | --- |
| FLA stable（`out=`，直接写调用方缓冲） | **0.1203 ms** | **0.1910 ms** |
| FLA stable（无 `out=`，原地语义） | 0.2031 ms | 0.3015 ms |
| vLLM-Ascend custom | 0.0849 ms | 0.1407 ms |
| FLA ctypes（参考实现） | 0.607–0.641 ms | 0.739 ms |
| PR #512 的 pybind 薄层（历史对照） | 0.1907 ms | 0.290 ms |

- `out=` 路径 / vLLM = **1.42×**（host）/ **1.36×**（device）；原地路径 = 2.4× / 2.1×。
  这个差值就是"原地语义必须多一次输出回写"的代价。
- FLA / ctypes = **0.2–0.3×**（快 3.3–5 倍）。
- 去掉的两处拷贝：（1）`_dense_conv_state` 对"contiguous 但带 storage offset"的
  conv_state 做 dense staging + 回写 —— 实测不需要，直接交 view 是逐位一致的；
  （2）`out=` 情况下适配层自分配输出、wrapper 再 `copy_` 回写 —— 现在把调用方的
  `out` 直接当 aclnn 的输出（`out` 必须是 x 的 shape/dtype/device，否则 C++ 侧拒绝）。
  原地路径的这次回写**去不掉**：kernel 边读 x 边写输出，把输出别名到 x 会让
  `conv1d_spec_decode(conv_states)` 直接对不上（实测 diff=240560）。
- conv1d 剩下的 host 差是 wrapper 自己的 3 次 `_host_ints` 与 dispatcher 记账；
  device 上剩下的差来自 OPP 里的 kernel 版本（FLA 的 19 参数版 vs vLLM vendored 的
  12 参数版），不是适配层。

## 为什么 stream 查询留在 Python

把"取当前 stream"下沉进 C++ 是**不可行**的，三条路都验过：

- Stable ABI 的 `aoti_torch_get_current_stream` / `Stream::id()` 在 torch_npu 上返回
  `96`，而同一时刻 Python 侧 `_npu_getCurrentRawStream` 返回 `1570479984`——它不对应
  NPU stream；
- torch_npu 的 `_C.*.so` 没有导出任何 stream 相关的 C 符号，无法 `dlsym`；
- ACL 只有 `aclrtCtxGetCurrentDefaultStream`（context 默认 stream），不是线程
  current stream，而线程隔离正是当初崩溃/路由错误的那个区分。

因此 stream 仍是**每次调用现取**（不再缓存，缓存正是 vLLM 崩溃的根因），隔离测量
约 1.7 µs，在完整 wrapper 里约 9 µs。

## 结论

- vLLM 实际调用的两个算子都在**同一量级**：recurrent 公共 API 1.35×（wrapper 口径
  1.0×），conv1d 2.3–2.6×；device 侧 recurrent 与 vLLM 同 kernel（1.0×），conv1d 的
  device 差异来自两套 kernel。
- 被替代的 ctypes 路径要贵 3.2–8.3 倍。
- 公共 API 上还剩的两段开销来自 dispatcher 记账/mutation 包装与 boxed 拆栈，是迁移
  计划里 R19 记录项；要再压到 1.0× 只能让热路径直接调注册 op（跳过 `_stable.py`）。
- 机器负载会移动绝对值；同一次运行内同形状的比值才是可比量。
