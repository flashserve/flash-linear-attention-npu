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
- 去掉的两处拷贝：（1）`_dense_conv_state` 的 dense staging + 回写 —— 现在只留给
  runtime 无法寻址的 state（见下一节）：带 storage offset 的 contiguous view 和
  分块（paged）view 都直接交描述符，实测逐位一致；
  （2）`out=` 情况下适配层自分配输出、wrapper 再 `copy_` 回写 —— 现在把调用方的
  `out` 直接当 aclnn 的输出（`out` 必须是 x 的 shape/dtype/device，否则 C++ 侧拒绝）。
  原地路径的这次回写**去不掉**：kernel 边读 x 边写输出，把输出别名到 x 会让
  `conv1d_spec_decode(conv_states)` 直接对不上（实测 diff=240560）。
- conv1d 剩下的 host 差是 wrapper 自己的 3 次 `_host_ints` 与 dispatcher 记账；
  device 上剩下的差来自 OPP 里的 kernel 版本（FLA 的 19 参数版 vs vLLM vendored 的
  12 参数版），不是适配层。

### paged（分块）state：去掉 staging 之后（CANN 9.2.0，batch 100，200 次）

state = 2574 块 × `(5, 4096)`，只有 100 块在用、由 `conv_state_indices` 寻址，
block stride 589824 元素；同一个环境里三臂对照（host P50 / device P50，ms）：

| 臂 | FLA 原地 | FLA `out=` | vLLM-Ascend custom |
| --- | --- | --- | --- |
| method 1（state 直接交 view） | 0.1977 / 0.2748 | **0.1399 / 0.2163** | 0.0829 / 0.1442 |
| 强制 staging（`FLA_NPU_CONV1D_VIEW_STATE=0`） | 0.2723 / 0.3635 | 0.2413 / 0.3256 | 0.0767 / 0.1334 |
| CANN 9.1.0（只能 staging） | 0.2565 / 0.3411 | 0.2332 / 0.3219 | 0.0772 / 0.1119 |

- 去掉 staging 的收益（`out=` 是与 custom 同形的那条路）：0.2413 → 0.1399，即
  **−0.10 ms/次（−42%）**；原地 0.2723 → 0.1977（−0.075 ms/次）。device 侧同步降
  0.11 / 0.09 ms，因为那次 `contiguous()` + `copy_` 本身就是设备工作。
- 精度不是交换来的：view 与 stage 的输出、state **逐位一致**（maxdiff 0.0），与
  vLLM custom 也逐位一致（0.0）。
- 剩下的 0.1399 vs 0.0767–0.0829 ≈ **1.7–1.8×**（host）、0.2163 vs 0.13–0.14 ≈
  **1.5–1.6×**（device）不是这一层再拷贝：host 是公共 API 的 3 次 `_host_ints` 与
  dispatcher / mutation 记账，device 是两条路交给 tiling 的元数据形态不同（FLA 传
  NPU tensor + host int[]，custom 只传 host int[]），tiling 因此选了不同的配置。
  空载机器上这个比值更低（前一次记录 `out=` 为 1.42× / 1.36×）。

## conv_state 的非连续能力由 CANN 版本决定

同一份 OPP（同一份源码、同一个 toolkit 编出来的）、只换 `ASCEND_HOME_PATH`：

| conv_state 布局 | CANN 9.1.0-beta.1 | CANN 9.2.0-beta.1 |
| --- | --- | --- |
| dense，或带 storage offset 的 contiguous view | ✅ | ✅ | 
| 分块（paged，stride=(576,64,1)） | ❌ 错值，回写还落进块间 gap | ✅ 与 dense 逐位一致 |
| dim 内 stride≠1（转置） | ❌ 不报错、静默错值 | ✅ 被算子拒绝（561002） |

算子自己在 tiling 里读 `context->GetInputStride(convStates)` 并优先用它
(`causal_conv1d_tiling_validation.h`)，但 9.1.0 的 aclnn 单算子路径不把描述符的
view 信息交给 tiling（算子日志 `isview=0 / stride_null=1`），于是回退到 dense
stride。所以 `_runtime.conv1d_view_state_supported()` 按 toolkit 版本判定：
≥ 9.2.0 直接交 view，更老或读不到版本的 runtime 仍走 dense staging + 回写，
两种情况下正确性都由上述探针（`conv1d_state_layout_parity.py`）验收。
`FLA_NPU_CONV1D_VIEW_STATE=1/0` 可覆盖判定。

这道门禁是承重的，两个方向都验过：9.1.0 + `=1`（强制 view）在 outer-gap /
inner-gap 上算错（out maxdiff 1.2e5 / 1.4e5，且有 192 个 padding 元素被写进
buffer）；9.2.0 + `=0`（强制 staging）仍然逐位正确。所以"直接交 view"只在
≥ 9.2.0 成立，更老的 runtime 必须留着 staging。

## 为什么 stream 查询曾经留在 Python（2026-09-16 已被推翻，见下一节）

把"取当前 stream"下沉进 C++ 是**不可行**的，三条路都验过：

- Stable ABI 的 `aoti_torch_get_current_stream` / `Stream::id()` 在 torch_npu 上返回
  `96`，而同一时刻 Python 侧 `_npu_getCurrentRawStream` 返回 `1570479984`——它不对应
  NPU stream；
- torch_npu 的 `_C.*.so` 没有导出任何 stream 相关的 C 符号，无法 `dlsym`；
- ACL 只有 `aclrtCtxGetCurrentDefaultStream`（context 默认 stream），不是线程
  current stream，而线程隔离正是当初崩溃/路由错误的那个区分。

因此 stream 仍是**每次调用现取**（不再缓存，缓存正是 vLLM 崩溃的根因），隔离测量
约 1.7 µs，在完整 wrapper 里约 9 µs。

> 下面三条只对 `aoti_torch_get_current_stream` 成立：它返回的是 `Stream::id()`，
> 在 torch_npu 上不是底层句柄。这也正是当时判"下沉不可行"的原因——找错了符号。

## 2026-09-16：stream 已经下沉进 launcher

torch_npu 自己用的是**另一个**稳定 ABI 符号：`aoti_torch_get_current_npu_stream`，
它返回的正是 `_npu_getCurrentRawStream` 的原始指针。`runtime.cpp` 现在
`dlsym(RTLD_DEFAULT)` 找它（找不到再 `dlopen("libtorch_npu.so")`），**每次调用真查、
不缓存**，Python 侧确认符号可用后一律传哨兵 `-1`。`FLA_NPU_STABLE_STREAM=python`
是留给现场二分定位的逃生阀，强制回到 Python 取 stream。

同一次运行内的空队列分层（batch 100、dim 4096、910B3，臂之间轮转）：

| 层 | host P50 |
| --- | --- |
| `aten.is_contiguous`（参考） | 0.0034 ms |
| boxed dispatch（不调 aclnn） | 0.0041 ms |
| raw `torch.ops` conv1d update | 0.0462 ms |
| + `_stable.py` wrapper | 0.0489 ms |
| + mutation wrapper | 0.0560 ms |
| 公共 `causal_conv1d_update` | 0.0606 ms |

三点修正：

- stream 解析在 raw 层已经≈0：raw+sentinel（0.0462）与 raw+调用方 stream（0.0445）
  同值，说明省下的是 Python 取 stream 的那一段；
- boxed 拆栈不是瓶颈：0.0041 对 `aten` 0.0034，参数从 1 个涨到 13 个只多约 5 µs；
- 剩下的开销是 **mutation wrapper ~10 µs + kwargs 调用形态 ~4 µs**，
  而不是 stream 查询。

## 结论

- vLLM 实际调用的两个算子都在**同一量级**：recurrent 公共 API 1.35×（wrapper 口径
  1.0×）；conv1d 空队列下 0.0606 / 0.0366 = **1.65×**（2026-09-16 口径，见上一节），
  device 侧 recurrent 与 vLLM 同 kernel（1.0×），conv1d 的 device 差异来自两套 kernel。
- 被替代的 ctypes 路径要贵 3.2–8.3 倍。
- 公共 API 上还剩的两段开销是 **mutation wrapper（~10 µs）与 kwargs 调用形态
  （~4 µs）**，boxed 拆栈只占 ~1 µs；要再压下去，一是让 schema 用 `Tensor(a!)`
  标注可变参数、交给 torch 自己记版本，二是让热路径直接调注册 op（跳过 `_stable.py`）。
  两项都还没做，都记在迁移计划的 R19 里。
- 机器负载会移动绝对值；同一次运行内同形状的比值才是可比量。
