# pre_process_fwd_kernel_merged 接口与计算约定

本算子实现线性 Attention 在**上下文并行（CP, context parallel）**下的前处理步骤：为当前
rank 的**一个序列窗口（一个 part）**计算窗口边界状态 `h` 与仿射链矩阵 `m`，并按上游布局
打包写回 `hm`。上游 `merge_fwd_bwd_kernel` 负责在 rank 间合并，不由本算子承担。

工作流：`catlass-linear-attention-v1`（CANNBot Linear Attention 专用流程，五阶段：
01 接口确认 → 02 标杆生成 → 03 方案设计 → 04 算子开发 → 05 算子测试）。

## 0. 修订记录

| 日期 | 修订 | 触发 |
| --- | --- | --- |
| 2026-09-21 | `K` 收敛为 `<= 128`、`V` 收敛为 `<= 128`、`chunk_size` 固定 64；布局定为 BNSD（与仓内其他 AscendC 算子一致）；GVA（`HK` 与 `HV` 成倍数）保留 | 用户明确要求 |
| 2026-09-21 | 进一步把 `K`、`V` 都**写死为 128**（与仓内 `chunk_fwd_h` 等同规格），host 拦截非 128；TilingKey 随之收敛到 6 个 | 用户侧核查结论（仓内所有 AscendC 算子的 `K` 都是 128，模型 case 的 `Kdim` 全是 128） |
| 2026-09-21 | **恢复标准的序列表达**：定长 = `cu_seqlens` 缺省且序列数 `= B`；变长 = `B = 1` + `cu_seqlens` 多段、序列数 `= len(cu_seqlens)-1`。取消"单窗口 `N = 1`"的收窄 | 用户澄清；与仓内 `chunk_fwd_h`、上游 `chunk_delta_h` 的 `N = B if cu_seqlens is None else len(cu_seqlens)-1` 一致 |
| 2026-09-22 | 定稿：`hm` 前导维 = **链条数 `Nseq`**（定长 `= B`；变长 `= len(cu_seqlens)-1`）；`M_c@m` 取 **FP32 原生**；DPLR **注册 6 个 TilingKey / 本轮验收 4 个**；新增 §3.5 Python 调用示例 | 用户确认（结合 CP 语义实测与竞品源码复核） |
| 2026-09-22 | **收掉 `B > 1`**：序列表达只保留 varlen 打包窗口（`B ≡ 1`、`cu_seqlens` 必给、`Nseq = len(cu_seqlens)-1`）；等长 batch 由调用方打包成等长多段。与竞品 CP 契约（"CP expects `B == 1` for varlen"）完全一致；`hm` 前导维保留（= 链条数），`Nseq=1` 时与竞品跨卡调用逐字节相同 | 用户要求"全量对齐竞品" |
| 2026-09-23 | **支持"子区间窗口"**：`cu_seqlens` 允许 `0 ≤ cu[0] < cu[-1] ≤ T`（即 `bos > 0`、`eos < T` 都合法），算子在**整根张量**上只处理该子区间 —— 与竞品调用形态（`cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]`）**1:1 一致**，零拷贝零浪费；相应放宽 host 校验（仍拒绝 `B != 1`、零长段、越界、非递增） | 用户要求"用法与竞品保持一致"；内核尚未实现，此时纳入成本最低 |
| 2026-09-30 | **去掉 DPLR**：`bg` / `v` 退回"仅占 ABI 槽位"，非空值在 host tiling、aclnn、ctypes 与 stable 四个入口**一致地直接拒绝**（不再存在"bg 配上 gk 就放行"的通道）；算法族收敛为 GDN（`g`）+ KDA（`gk`）；TilingKey 可达 **4 个**（`USE_G`/`USE_GK` × gate `BF16`/`FP32`），`USE_BG` 位保留但 host 永不产生 | PR 评审意见（DPLR 分支多入口判据不一致）+ 用户确认"不需要支持 DPLR" |

## 1. 参考资料与固定版本

| 项目 | 内容 |
| --- | --- |
| 参考实现仓库 | `fla-org/flash-linear-attention`（GitHub 上游） |
| 固定 commit | `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`（`main`，2026-09-17 读取） |
| 参考文件 | `fla/ops/cp/chunk_delta_h.py`，`pre_process_fwd_kernel_merged`（第 42 行起） |
| 参考文件 SHA256 | `a6ed6aaaf0bc6c7dc8c9235a5a9346d14a0bbeb3630b8baf8f1c2bc7ac92435c` |
| 许可证 | MIT（上游仓库根目录 `LICENSE`） |
| 本地归档 | 开发机上按上面的 commit 固定了一份只读快照（`PIN.txt` + 逐文件 SHA256），不在仓库内 |
| 用户需求来源 | 用户指定该 kernel 作为本 AscendC 算子的对标接口 |

同一 commit 下的调用方（决定各分支的实际语义）：

| 调用方 | 传参特征 | 生效分支 |
| --- | --- | --- |
| KDA `fla/ops/kda/chunk_fwd.py` | `k=kg, w=w, u=u, gk=g, v=None` | `USE_GK`，`v` 复用 `u` |
| GDN `fla/ops/gated_delta_rule/chunk.py` | `k=k, w=w, u=u, g=g, v=None` | `USE_G`，`v` 复用 `u`，`HK` 可小于 `HV` |
| DPLR `fla/ops/generalized_delta_rule/dplr/chunk.py` | `k=kg, w=w, u=u, gk=gi, bg=bg, v=v` | `USE_GK` + `USE_BG`，`v`/`u` 为不同张量 —— **本算子不支持**（见 §2、§6） |

上游 Python 包装的窗口切分：`cp_context.layout == 'zigzag'` 时按 `front`/`back` 两个 part
分别取 `cu_seqlens[fns-1:fns+1]` 与 `cu_seqlens[-2:]`，对每个 part 调用一次本 kernel
（`is_last_by_part` 为真时跳过）。**每次 kernel 调用处理一个窗口**，这正是本算子的粒度。

## 2. 本次范围

**做**：一次调用处理**一个打包窗口**（窗口内可含多段序列，段边界由 `cu_seqlens` 给出），
给定该窗口的 `k/w/u/g/gk/cu_seqlens`，输出完整 `hm[Nseq, HV, K, V+K]`
（`Nseq` = 链条数，定义见 §3.4）。上游的 `all_gather_into_tensor` + `merge_fwd_bwd_kernel`
是**编排层**的事，不由本算子承担（§2"不做"）。
算法族覆盖 **GDN（`g`）与 KDA（`gk`）两条路径**。

**不做**（由框架侧或后续版本承担）：

- **DPLR（`gk` + `bg`，`v` 与 `u` 分离）**：上游 kernel 的 `USE_BG` 分支、`bg` 的 L1/L0 槽位与
  `M_c` 的 `+` 号都不在本版本内。`bg` / `v` 两个入参**保留在 ABI 参数位上但必须为空**：
  传非空会在这四个入口被拒绝（host tiling / aclnn / ctypes / stable），
  不会静默按 GDN/KDA 计算，也不会下发未实现的 TilingKey 3；
- CP 多卡编排：`all_gather_into_tensor`、`merge_fwd_bwd_kernel`、zigzag 的 part 选择与
  `is_last_by_part` 判定；
- `MULTI_SEQS` 路径（上游包装恒传 `False`）；
- `state_v_first`（它只影响包装里 `initial_state` 的物理布局，不影响本 kernel 的 `hm` 输出）；
  merge kernel 的注释也明确写了这件事：`STATE_V_FIRST: tl.constexpr = False,  # When True,
  h0/h use [V, K] layout; ag_hm always [K, V+K]`（`fla/ops/cp/chunk_delta_h.py:355`）——
  即**本算子写出的 `hm` 恒为 `[K, V+K]`**，与布局开关无关。仓内下游
  `npu_chunk_gated_delta_rule_fwd_h` 的 `transpose_state_layout` 也是"预留、必须 False"，
  两边同样锁定在 `[K,V]`，因此本算子不需要该参数。
- `AFFINE_CHAIN_PRECISION = tf32x3`（NVIDIA 专用；NPU 固定 `ieee`，即 FP32 累加）。

## 3. 接口

### 3.1 算子名与层级

| 项目 | 取值 |
| --- | --- |
| 算子名（snake） | `pre_process_fwd_kernel_merged` |
| OP TYPE | `PreProcessFwdKernelMerged` |
| aclnn 接口 | `aclnnPreProcessFwdKernelMerged` |
| 工程目录 | `fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/`（CP 算子统一放在 `fla/ops/ascendc/cp/`，与 `chunk_delta_h_bwd_preprocess` 同目录） |
| Python 调用 | `from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged`（同时提供 `npu_pre_process_fwd_kernel_merged`） |
| 接入方式 | **走 ctypes（2026-09-28 指定）**：仓内 AscendC 算子 + `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 的 Python ctypes wrapper + `ops/ascendc/__init__.py` 的 `_ASCENDC_OPS` 注册。**三处改动**见 §3.1.1 |
| 目标 SoC | Ascend950（`NpuArch=3510`，`CATLASS_ARCH=3510`） |

`_ASCENDC_OPS` 注册时会同时导出 `npu_pre_process_fwd_kernel_merged` 与去掉前缀的
`pre_process_fwd_kernel_merged`，与 `npu_chunk_fwd_h` / `chunk_fwd_h` 的现有约定一致。

#### 3.1.1 ctypes 接入的三处改动（已落地并通过 ABI 单测）

| # | 文件 | 改动 |
| --- | --- | --- |
| ① | `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` | 加 `_GET_WORKSPACE_ARGTYPES["aclnnPreProcessFwdKernelMerged"]`（8 个 descriptor 指针 + `int64 chunkSize` + `hmOut` + `workspaceSize*` + `executor*`）；加 `def npu_pre_process_fwd_kernel_merged(k, w, u, g=None, *, gk=None, bg=None, v=None, cu_seqlens=None, chunk_size=64)`（`bg`/`v` 只保留签名位，传非空抛 `NotImplementedError`），内部做参数契约校验、用 `_zeros` 预分配 `hm[Nseq,HV,K,V+K]` fp32、经 `_call_aclnn` 两段式调用 |
| ② | `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` | 在 `_ASCENDC_OPS` 元组里加 `"npu_pre_process_fwd_kernel_merged"`（自动导出带/不带前缀两个名字） |
| ③ | `torch_custom/fla_npu/test/test_pre_process_fwd_kernel_merged.py` | 新增 **11 条离线**单测（FakeTensor/FakeCallContext，不需要 NPU）：子区间 `[40,512]`→`hm[1,…]`、多段 `[0,88,188,512]`→`hm[3,…]`、`B!=1`/缺 `cu_seqlens`/门控冲突/越界拒绝、**`bg` 与 `v` 传非空被拒（DPLR 不支持）**、`ARGTYPES` 与签名核对 |

验证命令与结果（本地 Windows，无 NPU 也可跑）：

```bash
cd torch_custom/fla_npu/test
python -m unittest test_pre_process_fwd_kernel_merged    # Ran 11 tests ... OK
python -m unittest test_aclnn_ctypes_abi                 # 回归：Ran 8 tests ... OK
```

**顺带关闭了 §5.2 第 7 项**：ctypes 路由下 `hm` 由 Python 侧 `_zeros` 预分配、作为输出张量传给
aclnn 再返回 —— 既与竞品"调用方预分配 `hm` buffer"的用法一致（`k.new_zeros(HV,K,V+K)`），
又与仓内其它 ctypes 算子（如 `npu_chunk_gated_delta_rule_fwd_h` 的 `h_out/v_new_out`）一致，
所以**不额外引入 `hm_out` 参数**。

### 3.2 输入

`B` 为 batch，`T` 为每序列的 token 数；`K` 为 key 维，`V` 为 value 维，`HK` 为 key 侧 head 数，
`HV` 为 value 侧 head 数。布局沿用本仓 AscendC 算子约定 `[B, H, T, D]`（BNSD），与 `chunk_fwd_h` 一致。

**序列表达只有一种（2026-09-22 定稿）：varlen 打包窗口**；**窗口可以是张量 T 轴的子区间**
（2026-09-23 定稿，用于与竞品调用形态一致）。

| 项 | 取值 |
| --- | --- |
| `B` | **恒为 1**（与竞品 CP 契约一致：`fla/ops/cp/README.md` "CP expects `B == 1` for varlen"，局部输入 `[1, T_local, D]`） |
| `cu_seqlens` | **必给**，给出本窗口内的段边界（`[N+1]`，严格递增，`0 ≤ cu[0] < cu[-1] ≤ T`） |
| 序列数 | `Nseq = len(cu_seqlens) - 1 >= 1` |
| 整窗单段 | `cu_seqlens = [0, T]`（`Nseq = 1`，窗口 = 整个张量） |
| **子区间单段（竞品用法）** | `cu_seqlens = [bos, eos]`（`Nseq = 1`，`bos > 0` 或 `eos < T` 均合法）：算子**只处理张量的这一小段**，窗口外不读 —— 这正是竞品跨卡调用 `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` 的形态 |
| 等长 batch | 由**调用方打包**：`B` 条 `T` 长的序列 ⇔ 一个打包窗口 `cu_seqlens=[0, T, 2T, …, B·T]`、`T_win = B·T`；每段的 chunk 仍从各自起点对齐，结果与逐条独立调用逐位相同（打包写法与"不打包就逐条调 B 次"的取舍见 §3.5 示例 1b） |

本算子的并行度按序列总数算（`Nbase = Nseq`，见 `docs/design.md` 2.1.4）。上游同一处写法是
`N = B if cu_seqlens is None else len(cu_seqlens) - 1`——竞品 CP 路径**恒走 varlen 分支**
（`cu_seqlens` 为 None 时它自己的包装会崩），所以只有等价于 `len(cu_seqlens)-1` 的那一支。

**子区间窗口为什么必须支持**：竞品的每一次调用都是"整根张量 + 2 元素子区间"
（`cu_seqlens[-2:]` 取的末段可能在窗口中间，`cu_seqlens[fns-1:fns+1]` 取的 front 末段起点常常 > 0）。
若只允许"窗口 = 张量 T 轴"，调用方就得先把那段**拷贝成独立连续张量**（模型 case 下一段的搬运
约 77 µs，比该 rank 算子自身 ~35 µs 还贵）；而"整窗多段一次算"又会在大 `Nseq` 下按波次白算。
**支持子区间 = 零拷贝 + 零浪费 + 与竞品调用 1:1**。实现上段基址按 `bos` 相对张量起点算、
h 维 stride 用 `shape[2]`，内核地址表达式不变形（见 `docs/design.md` 1.4.1 C1）。

**head 约定**：`HK` 与 `HV` **可以不一致，但必须成倍数**（`HV >= HK` 且 `HV % HK == 0`），
这正是 GVA 的形态：`k` 在 `HK` 维，`w/u/g` 在 `HV` 维，算子内部按
`hk = hv // (HV/HK)` 取对应的 key head。输出 `hm` 与两个状态都在 `HV` 维。
`gk` 路径（KDA）的 gate **按 value head 给**（`[1, HV, T, K]`），`k` **仍按 `HK` 头**，
因此 `HK < HV`（GVA）在 gk 路径同样合法 —— 与竞品 `gk` 的 head 语义一致。

| 名称 | 必选 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| `k` | 是 | `[1, HK, T, K]`（g 与 gk 路径同形；GVA 时 `HK < HV`） | BF16 | raw key，按 `hk = hv // (HV/HK)` 复用；`gk` 路径同样按 `HK` 头，gate 侧才按 `HV` |
| `w` | 是 | `[1, HV, T, K]` | BF16 | erase/WY 输出，h 与 m 的左矩阵 |
| `u` | 是 | `[1, HV, T, V]` | BF16 | 取值来源（GDN/KDA 的 `v` 与 `u` 为同一张量，故只收 `u`） |
| `v` | **否** | `[1, HV, T, V]` | BF16 | **DPLR 专用位，本版本不支持**：必须传 `None`（或省略），传非空直接拒绝 |
| `g` | 二选一 | `[1, HV, T]` | FP32 或 BF16 | 标量 gate，**base-2 的 chunk 内累积对数衰减**；与 `gk` 互斥 |
| `gk` | 二选一 | `[1, HV, T, K]` | FP32 或 BF16 | 逐 K gate，同为 base-2 chunk 内累积量；与 `g` 互斥 |
| `bg` | **否** | `[1, HK, T, K]` | BF16 | **DPLR 专用位，本版本不支持**：必须传 `None`（或省略），传非空直接拒绝 |
| `cu_seqlens` | **是** | `[N+1]` | **Python `list[int]`**（torch schema `int[]?`） | 本窗口内的段边界（T 轴打包多段），`N = Nseq >= 1`；严格递增，`0 ≤ cu[0] < cu[-1] ≤ T`（**允许子区间**：`cu[0] > 0` 或 `cu[-1] < T`） |

`cu_seqlens` 与仓内其它 AscendC 算子一致：**是 host 侧的整型数组，不是张量**（Python `list[int]`
→ `at::OptionalIntArrayRef` → aclnn 的 `aclIntArray`）。因此它**不占 GM 带宽、也不需要 device
张量**，边界校验与"每段的 `bos/T_win/NT` 展开"全部在 host tiling 里做（这也是仓内算子还要
`chunk_indices` 的原因）。

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `64` | 本轮固定 `64`，其它值在 host 侧拒绝 |
| `AFFINE_CHAIN_PRECISION` | - | `ieee` | 固定 FP32 累加；`tf32x3` 不支持 |

### 3.4 输出

| 名称 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `hm` | `[Nseq, HV, K, V+K]` | FP32 | 每条链 × 每个 head 的 `[K, V+K]` 矩阵：左 `[0, V)` 为 `h`，右 `[V, V+K)` 为 `m` |

**`Nseq` 不是入参**，而是由 `cu_seqlens` 推导出来的**链条数**（= 上游的 `N`）：
`Nseq = len(cu_seqlens) - 1`。单段窗口 `[0, T_win]` → `Nseq = 1`；打包 2 段 `[0,256,512]`
→ `Nseq = 2`。它同时是并行度的乘数（`Nwork = Nseq × HV`，见 `design.md` 2.1.4）。

固定规格下 `K = V = 128`，所以 `hm` 展开就是 `[Nseq, HV, 128, 256]`；例如 `Nseq=3, HV=32`
→ `(3, 32, 128, 256)` FP32（约 12.6 MB），其中 `hm[0]` 是本地第一段的链、`hm[Nseq-1]` 是本地
末段的链（即竞品 `cu_seqlens[-2:]` 那一条）。单段窗口时是 `[1, HV, 128, 256]`——去掉 size-1 维
就与竞品跨卡调用的 `[HV, K, V+K]` 逐字节相同。

**`h` / `m` 在 `hm` 里的 shape**（`K=V=128`）：

```text
hm[i, hv]            : [K, V+K] = [128, 256]   FP32   ← 一条链、一个 head
  ├─ h = hm[i,hv][:, 0:V]    : [K, V] = [128, 128]    行 = key 维、列 = value 维
  └─ m = hm[i,hv][:, V:V+K]  : [K, K] = [128, 128]    传递矩阵，初值 I
```

即 **`h` 不是独立张量，而是 `hm` 最后两维的左半边**（行间隔是 `V+K`，所以单独取 `h` 是带 stride
的视图、不是连续块）。竞品 kernel 用同一套写法：`stride_hm_kv = K + V`（`cp/chunk_delta_h.py:243`）、
h 落列 `[0,V)`（`:244` `p_h = hm + o_k*stride_hm_kv + o_vb`）、m 落列 `[V,V+K)`
（`:316` `p_m = hm + V + row*stride_hm_kv + col`）。这也是本设计里 S3 按"`h` 列段"写 GM、
S4 按 `[V, V+K)` 写 `m` 的原因（`design.md` 2.5/2.6）。

**前导维 = 链条数 `Nseq`**（2026-09-22 定稿）：`Nseq = len(cu_seqlens) - 1`（`B ≡ 1`，
唯一形态）。语义等价说法：
**竞品一次调用产出一份 `hm`，我们一次调用产出 `Nseq` 份，第 i 份与竞品针对第 i 段单独调用
一次逐位相同**（竞品 kernel 的 `MULTI_SEQS` 与逐段调用实测逐位相等）。
CP 场景下本 rank 的窗口常是单段（`Nseq = 1`），此时去掉 size-1 维后
与竞品的 `[HV, K, V+K]` 内存布局逐字节相同。

**竞品 `hm` 的形状（逐处核对源码）**：

| 场景 | 上游 `hm` | 本算子 `hm` |
| --- | --- | --- |
| **跨卡 CP 的每一次调用**（contiguous / zigzag 都一样） | `[HV, K, V+K]`，**没有前导维**——`cp/chunk_delta_h.py:873` `hm = k.new_zeros(HV, K, (V + K))`，`MULTI_SEQS=False`；kernel 里每个 head 写一块 `[K,V+K]`（`:74` `hm += i_h*K*(K+V)`） | `Nseq = 1` → `[1, HV, K, V+K]`（去掉 size-1 维即逐字节相同） |
| zigzag 的 `hm` **缓冲区** | `[2, HV, K, V+K]`（`cp/chunk_delta_h.py:804`）——但这个 `2` 是 **part 维**：包装层对 front/back 各调一次 kernel，每次传 `hm=hm[part]`（`MULTI_SEQS` 仍为 `False`），**不是"一次调用产出两份"** | 两次调用各 `Nseq = 1`，或（若调用方愿意）一次调用吃两段 `Nseq = 2` |
| **卡内 CP（推理切段）** | `[S_split, HV, K, V+K]`（`fla/ops/common/intracard_cp.py:268`），`grid=(列块, HV, S_split)`、`MULTI_SEQS=True`；kernel 的段间跨距是 `HV*K*(K+V)`（`cp/chunk_delta_h.py:71`） | `Nseq = S_split` → 同形 |

**与竞品"一次一份"形态的换算**：竞品 CP 包装层对每个 part 只喂"该 part 的末段"
（`cu_seqlens[fns-1:fns+1]` 或 `cu_seqlens[-2:]`），因此它一次只产一份；我们一次可以产出
`Nseq` 份（第 i 份 == 竞品对第 i 段单独调用）。下游 `all_gather` / `merge` 需要的分片语义
不变，只按 `part` ↔ `Nseq` 映射即可。

> **竞品的 `hm` 确定没有 batch 维。** kernel 里对 `hm` 只有两个前导偏移：
> `hm += i_h * K*(K+V)`（head）与 `hm += i_n * HV*K*(K+V)`（段号，仅 `MULTI_SEQS=True`），
> **没有任何 batch 偏移**（`cp/chunk_delta_h.py:68-74`）；`initial_state` 的分配也只按
> `N = len(cu_seqlens)-1`（段数）算，与 batch 无关。原因是 CP 下局部输入恒为 `B = 1`（varlen
> 打包），batch 已经被折进 token 轴。所以本算子的前导维**不是 `B`、而是链条数 `Nseq`**——
> 单段时它与竞品逐字节相同（`[1,…]` 去掉 size-1 维），多段时它是"每段一条链"的表达；
> **不要把它当 batch 去 broadcast**（`B>1` 的定长输入必须先打包，见 §3.2/§3.5）。

**为什么带前导维**（而不是照抄上游的 `[HV, K, V+K]`）：

1. **输入输出对称**：`k/w/u/g/gk` 都是 `[B, H, T, (D)]`，输出不带前导维则"一次调用处理多条
   序列 / 多段"无法表达。
2. **与仓内约定一致**：仓内 `npu_chunk_fwd_h` 的输出 `h` 是 `[B, HV, NT, K, V]`，带前导维；
   本算子接在它之前，接口形态保持一致，落点无需额外适配（见 `docs/design.md` 1.4）。
3. **上游"无前导维"是它调用粒度的产物，不是设计原则**：上游跨卡包装一次只算一段窗口，所以
   不需要；而它一用 zigzag（每 rank 两个 part）就不得不加一个 `2` 维，卡内 CP 的
   `intracard_pre_scan` 更是直接用 `[S_split, HV, K, V+K]`（`MULTI_SEQS=True`）。
4. **`Nseq > 1` 是本算子并行度的来源**：并行度 = `Nseq × HV`（`Nseq = len(cu_seqlens)-1`）。
   没有前导维就只能靠调用方按段多次调用，固定开销翻倍（见 `docs/design.md` 2.1.4）。

### 3.5 Python 调用示例（落地形态）

注册与导入：安装后 `import fla_npu` 即把算子注册到 `torch.ops.npu`；
`fla_npu.ops.ascendc` 同时导出带前缀与不带前缀两个名字（与仓内 `npu_chunk_fwd_h` /
`chunk_fwd_h` 的约定一致）：

```python
import torch
import torch_npu            # noqa: F401  （NPU 运行时）
import fla_npu              # noqa: F401  （注册 torch.ops.npu.*）
from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged
# 等价：from fla_npu.ops.ascendc import npu_pre_process_fwd_kernel_merged
# 等价：torch.ops.npu.npu_pre_process_fwd_kernel_merged(...)
```

签名（输入布局统一 BNSD `[B, H, T, D]`，`k` 在 `HK` 维、`w/u/g/gk` 在 `HV` 维）：

```python
hm = pre_process_fwd_kernel_merged(
    k, w, u,
    g=None,          # 与 gk 二选一（GDN 路径）
    gk=None,         # 与 g 二选一（KDA 路径；gate 按 HV 头给，k 仍按 HK 头）
    bg=None,         # DPLR 专用（不支持）：必须为 None，传非空直接抛 NotImplementedError
    v=None,          # DPLR 专用（不支持）：必须为 None，取值走 u
    cu_seqlens=None, # 必给：list[int]（host 数组，非张量），[N+1]，严格递增，0 <= cu[0] < cu[-1] <= T
                     #       允许子区间（cu[0] > 0 或 cu[-1] < T），即竞品"整根张量 + [bos,eos]"的用法
    chunk_size=64,   # 固定 64
)
# 返回 hm: [Nseq, HV, K, V+K] FP32（左 [0,V) 为 h，右 [V, V+K) 为 m）
# Nseq = len(cu_seqlens) - 1（B ≡ 1）
```

**示例 1：单段窗口 + GVA（`HK=2, HV=4, T_win=256`）**

```python
HK, HV, T, K, V = 2, 4, 256, 128, 128
dev, dt = "npu:0", torch.bfloat16

k = torch.randn(1, HK, T, K, device=dev, dtype=dt)               # B ≡ 1
w = torch.randn(1, HV, T, K, device=dev, dtype=dt)
u = torch.randn(1, HV, T, V, device=dev, dtype=dt)
g = torch.randn(1, HV, T,    device=dev, dtype=torch.float32)   # base-2 的 chunk 内累积对数衰减

hm = pre_process_fwd_kernel_merged(k, w, u, g=g, cu_seqlens=[0, T])
assert hm.shape == (1, HV, 128, 256) and hm.dtype == torch.float32
# hm[0] 是这条链的 (h | m)；下游 all_gather + merge 由框架侧做
```

**示例 1b：定长 batch（2 条 256 长的序列）→ 打包成一次调用**

定长输入 `[B,H,T,D]` **不能直接喂进来**（`B ≡ 1`），要先按 token 轴打包：token 序 =
批次 0 的全部 token、批次 1 的全部 token……（各段在自己的起点重新开始 chunk，所以与逐条
独立调用逐位相同）。

```python
B, HK, HV, T, K, V = 2, 2, 4, 256, 128, 128
dev, dt = "npu:0", torch.bfloat16

# 原始定长输入（BNSD）
k = torch.randn(B, HK, T, K, device=dev, dtype=dt)
w = torch.randn(B, HV, T, K, device=dev, dtype=dt)
u = torch.randn(B, HV, T, V, device=dev, dtype=dt)
g = torch.randn(B, HV, T,    device=dev, dtype=torch.float32)

def pack_bt(x):                 # [B,H,T,(D)] -> [1,H,B*T,(D)]
    B, H = x.shape[:2]
    return x.permute(1, 0, *range(2, x.dim())) \
            .reshape(1, H, B * x.shape[2], *x.shape[3:]).contiguous()

cu = [t * T for t in range(B + 1)]             # [0, 256, 512]
hm = pre_process_fwd_kernel_merged(pack_bt(k), pack_bt(w), pack_bt(u),
                                   g=pack_bt(g), cu_seqlens=cu)
assert hm.shape == (B, HV, K, V + K)           # 每段一条链；hm[b] == 逐条调用第 b 次
```

> 打包是一次 `permute + contiguous`（付一份搬运）。若不想付，也可以**逐条调用 B 次**
> （每次 `cu_seqlens=[0, T]`、`hm` 为 `[1, HV, K, V+K]`），代价是 B 倍固定开销与 B 倍
> `all_gather`/`merge` 通信量。竞品 CP 侧的调用方本来就是按打包形态准备数据的
> （`fla/ops/cp/README.md`：局部输入 `[1, T_local, D]`），所以实际没有这次额外搬运。
>
> **这一段只属于"拿定长 dense 输入硬要一次调用"的情形**。真实 CP 流程里调用方持有的就是
> 打包后的局部窗口，`pack_bt` 无需执行。另外注意本算子契约是 **BNSD `[B,H,T,D]`**，与竞品
> 的 token-major `[B,T,H,D]` 不同——布局这一层的取舍见 `docs/design.md` 5.2（结论：由调用方
> 吸收，前提是上游张量本来就是 BNSD；若框架侧只有 token-major 数据，需要先与我们对齐）。

**示例 2：变长打包（`B=1`，3 段，段长 256 / 256 / 512）**

```python
T = 1024
k = torch.randn(1, 32, T, K, device=dev, dtype=dt)
w = torch.randn(1, 32, T, K, device=dev, dtype=dt)
u = torch.randn(1, 32, T, V, device=dev, dtype=dt)
gk = torch.randn(1, 32, T, K, device=dev, dtype=torch.float32)   # KDA：逐 K gate（按 HV 头给）

hm = pre_process_fwd_kernel_merged(k, w, u, gk=gk, cu_seqlens=[0, 256, 512, 1024])
assert hm.shape == (3, 32, 128, 256)      # Nseq = 3，每段一条链
```

**示例 3：CP 场景的两种形态（与竞品一一对应）**

```python
# ① 严格对标形态：与竞品跨卡包装层完全相同的调用 —— 整根张量 + 2 元素"子区间"
#    （张量仍是本 rank 的整根本地 buffer；窗口只是它的一段，可以不从 0 开始）
bos, eos = 1408, 2816                     # 本地 token 下标（相对本 rank 窗口/张量起点）
hm = pre_process_fwd_kernel_merged(k_loc, w_loc, u_loc, gk=gk_loc, cu_seqlens=[bos, eos])
# hm: [1, HV, K, V+K]，去掉 size-1 维后与竞品 `hm` 逐字节相同 → 可直接和 H20 基线逐元素比
#     竞品对应写法：cu_win = cu_seqlens[-2:]（contiguous）或 cu_seqlens[fns-1:fns+1]（zigzag front）

# ② 吃满并行度形态：把本 rank 整个窗口（含多段）一次传进来，Nseq = 段数
window_cu = [0, 88, 188, 512]             # 本 rank 局部窗口的段边界（相对窗口起点）
hm = pre_process_fwd_kernel_merged(k_loc, w_loc, u_loc, gk=gk_loc, cu_seqlens=window_cu)
# hm[i] == 竞品针对第 i 段单独调用一次的结果（kernel 侧 MULTI_SEQS 已实测逐位相等）
```

> CP 相关的取舍（`layout` 感知、`is_first_rank` / `is_last_rank` 的跳过判断、`all_gather`
> 与 `merge`、`compress_h0`）都在**编排层**，不在本算子接口里。

### 3.6 与竞品 kernel 参数的一一对应

竞品 kernel 形参（22 个）在本算子里的归宿——**"不暴露"不等于"能力缺失"**：

| 竞品形参 | 本算子 | 说明 |
| --- | --- | --- |
| `k`, `v`, `w`, `u` | 输入 ✓ | 本算子收 `k`/`w`/`u`；`v` 是 DPLR 专用位（本版本必须为 `None`）。GDN/KDA 下 `v` 与 `u` 是同一张量，故只收 `u`——与上游包装 `v = u if v is None else v` 等价 |
| `g`, `gk`, `bg` | 输入 ✓ | `g`/`gk` 二选一（推导 `USE_G`/`USE_GK`）；`bg` 是 DPLR 专用位（本版本必须为 `None`），因此 `USE_BG` 不会被选中 |
| `cu_seqlens` | 输入 ✓ | 竞品是 device tensor、kernel 用 `tl.load` 读；本算子是 **host `list[int]`**（§3.2）：边界校验与每段 `bos/T_win/NT` 的展开在 host tiling 做，不占 GM 带宽 |
| `hm` | **返回值** | 竞品的 `hm` 是包装内部的预分配 buffer（`k.new_zeros(...)`；zigzag 用 `hm=hm[part]` 复用）；本算子返回 `[Nseq,HV,K,V+K]`——**与仓内其它 AscendC 算子一致**（`npu_chunk_fwd_h` 也是返回 `(h, v_new, final_state)`）。若框架侧要求"直接写进 `all_gather` 的目标 buffer"，再加可选 `hm_out` 作为扩展（5.2 第 7 项） |
| `T` | **不暴露** | varlen 分支里 kernel 用 `eos-bos` 覆盖它；本算子只走 varlen，窗口长 = 张量的 T 轴长度，每段长度由 `cu_seqlens` 给出（校验见 §6） |
| `H`, `HV`, `K`, `V`, `BT` | **不暴露** | `HK`/`HV` 由张量形状给出；`K=V=128`、`BT=64` 是固定规格（编译期，host 拦截其它值） |
| `BLOCK_SIZE`, `BK1` | **不暴露** | 上游的列分块宽与 `next_power_of_2(K)`；本设计不拆列段，这两个量在 host tiling 里推导（`design.md` 3.1/3.4） |
| `USE_G`, `USE_GK`, `USE_BG` | **不暴露**（= `TilingKey`） | 由 `g`/`gk` 是否给推导，**可达 4 个 `TilingKey`**（2 个门控 × gate `BF16`/`FP32`，`design.md` 3.2.1）；`USE_BG` 位保留但 host 永不产生（`bg` 非空被拒） |
| `IS_VARLEN` | **不暴露**（恒 True） | 竞品有定长/变长两个分支；本算子只有 varlen 打包窗口（§3.2），恒等价于 `IS_VARLEN=True` 那一支 |
| `MULTI_SEQS` | **不暴露**（恒 True） | 竞品跨卡包装恒传 `False`（一次一段）；本算子恒按"一次可多段"实现，`hm[Nseq,...]` 的段前导维就是它的产物。**这不是我们额外发明的语义**——竞品 kernel 的 `MULTI_SEQS` 分支与卡内 `intracard_pre_scan` 用的就是这一支 |
| `AFFINE_CHAIN_PRECISION` | **不暴露**（固定 FP32 原生） | 竞品可为 `tf32x3`/`None`；本算子固定 `ieee`（FP32 原生），见 `design.md` 2.8 |

**zigzag 的 `for part in (front, back)` 循环怎么吸收——不需要搬进算子。**

**`IS_VARLEN` / `MULTI_SEQS` 这两个"旋钮"为什么不用传，以及怎么按竞品形态调用：**

它们在竞品里是 **kernel 的 `tl.constexpr`**，由**包装层**（不是模型/用户）在 launch 时决定：
`IS_VARLEN` 来自 `cu_seqlens is not None`（`@triton.heuristics` 自动推导），`MULTI_SEQS` 由
跨卡/卡内两条路径硬编码。我们的契约把输入形态收敛成"唯一 varlen 打包窗口"后，两者都成了常量，
因此**不必出现在入参里**；而且若暴露出来，反而多出"参数与数据不一致"的非法组合
（例如 `MULTI_SEQS=False` 却给了 5 个元素的 `cu_seqlens`）。**调用方仍然能精确表达竞品的两种形态**：

| 想要竞品的哪种调用 | 我们怎么调 | 等价关系 |
| --- | --- | --- |
| 跨卡（`IS_VARLEN=True, MULTI_SEQS=False`） | `cu_seqlens=[bos, eos]`（2 个元素） | `Nseq=1` → `hm [1,HV,K,V+K]`，与竞品那块**逐字节相同** |
| 卡内切段（`IS_VARLEN=True, MULTI_SEQS=True`） | `cu_seqlens` 给整条 `[0,s1,…,T]` | `Nseq=N` → `hm [N,HV,K,V+K]`，第 i 条 == 竞品对第 i 段单独调用 |
| 定长（`IS_VARLEN=False`，`B>1`） | **不支持** | CP 下不可达（竞品 CP 路径恒 varlen；非 CP 下它直接 `return`） |

如果框架侧确实希望"照抄竞品调用、显式声明这两个 flag"，可以在 `fla_npu.ops.ascendc` 的
Python 适配层加一个同形 shim（`multi_seqs: bool / is_varlen: bool` → 转成上面两种调用形态），
**不改算子契约**。

竞品那段循环做的是：对每个 part 取**该 part 的末段**（`cu_seqlens[fns-1:fns+1]` / `cu_seqlens[-2:]`）、
各调一次 kernel（`MULTI_SEQS=False`、`hm=hm[part]`），即"**两个 part = 两次调用、每次一条链**"。
我们等价的做法是**把整个本地 buffer 当一个窗口**：

```python
# 本地 buffer = [front; back] 拼在一根 token 轴上（T_local = 2 * part_len）
hm = pre_process_fwd_kernel_merged(k_loc, w_loc, u_loc, gk=gk_loc,
                                   cu_seqlens=ctx.cu_seqlens_cpu)   # 覆盖全部段，含 part 边界
# hm: [Nseq, HV, K, V+K]，每段一条链
hm_front_last = hm[context.front_num_seqs - 1]   # front part 的末段（= 竞品 part 0 那一条）
hm_back_last  = hm[Nseq - 1]                     # back part 的末段（= 竞品 part 1 那一条）
```

成立的前提只有一条：**任何 part 边界都必须出现在 `cu_seqlens` 里**——zigzag 天然满足，因为
`get_cp_cu_seqlens` 构造的本地 `cu_seqlens = cat([front_cu, back_cu[1:] + part_len])`，而
`front_cu[-1] == part_len` 恒成立，即 part 边界本来就是其中一个元素。因此**不需要新增
`part_offsets`、也不需要把 `cu_seqlens` 升成 `[B,N+1]`**；多出来的 `Nseq-2` 条链是"用段数补
并行度"的代价（不取用即可，见 `design.md` 2.1.4）。

> 若框架侧出于别的原因仍希望"每个 part 单独调一次"（最保守对标竞品），那就需要窗口是张量
> T 轴的**子区间**（`bos>0` 或 `eos<T`），见 `design.md` 5.2 第 12 项；两条路都在文档里留了。

**为什么 contiguous 分支没有那个 for 循环**——两个布局的"每 rank part 数"不同：

| 布局 | 每 rank 的 part | 竞品导出方式 | 竞品 `hm` | 我们 |
| --- | --- | --- | --- | --- |
| contiguous（默认） | **1 个**（区间 `[r·part_len, (r+1)·part_len)`） | 直接取本地 `cu_seqlens[-2:]`（该窗口的末段）→ **1 次调用**，用标量 `is_last_rank` 跳过 | `[HV, K, V+K]`，无前导维 | 1 次调用：窗口 = 该 rank 的整段 token，`Nseq = 段数` |
| zigzag | **2 个**（front + back，链序分别为 `r` 与 `2W-1-r`） | 对 front 取 `cu_seqlens[fns-1:fns+1]`、对 back 取 `cu_seqlens[-2:]` → **2 次调用**，用逐 part 的 `is_last_by_part` 跳过 | `[2, HV, K, V+K]`，前导维 = part | **1 次调用**：窗口 = `[front; back]` 整块，`Nseq = 全部段数`，调用方取 `hm[fns-1]` / `hm[Nseq-1]` |

所以 for 循环不是"语义需要"，而是"**一个 rank 有两个窗口要各自导出末段**"的产物；我们的
"一个窗口可含多段、每段一条链"天然把它吸收掉——而且 **part 边界必然出现在 `cu_seqlens` 里**
（`front_cu[-1] == part_len`，构造时 `back_cu[0]=0` 被丢掉、只保留一次 `part_len`），所以段枚举
不会把两个 part 的相邻段误并成一段。

## 4. 数学语义

按一个 value head `i_h`、一个窗口描述，`T` 为该窗口 token 数，`BT = 64`，
`NT = ceil(T / BT)`。累加一律 FP32。`E(x) = exp2(x)`（**全部衰减量都是 base-2 对数域**）。

状态：`h` 为 `K x V`，初值全零；`m` 为 `K x K`，初值单位阵。

对第 `c` 个 chunk（`c = 0 .. NT-1`），记 `last = min((c+1)*BT, T) - 1`：

```text
# 1) 用上一 chunk 遗留的 h（未按本 chunk 衰减）算 v 的衰减项，dot 前 h 先降到 BF16
v_decay = W_c @ bf16(h)

# 2) 形成本 chunk 的“新 v”
GDN/KDA:  v_new = V_c - v_decay

# 3) chunk 内逐 token 衰减
USE_G:    v_new *= E(g[last] - g[t])   # 逐行，t 为 chunk 内全局 token 下标
          h     *= E(g[last])          # 整个状态按本 chunk 的总衰减缩放
USE_GK:   h[k, :] *= E(gk[last, k])    # 逐 K 行衰减（此时 v_new 不再单独缩放）

# 4) 累积本 chunk 的贡献（右矩阵降为 BF16 后做 Cube 乘累加）
GDN/KDA:  h += K_c^T @ bf16(v_new)

# 5) 仿射链 m 的推进（全 FP32）
USE_G:    K_c = K_c * E(g[last] - g[t])[:, None]
          M_c = diag(E(g[last])) - K_c^T @ W_c
USE_GK:   M_c = diag(E(gk[last, :])) - K_c^T @ W_c
m = M_c @ m        # 按 chunk 顺序左乘，m 初值为 I
```

其中 `K_c = k_c`（g-only 时 `k_c` 复用的是 `HV` 侧展开后的 key head
`hk = i_h // (HV / HK)`）。上游 DPLR 分支的 `bg_c` 与 `M_c` 的 `+` 号不在本版本内（§2）。

写回：

```text
hm[i_h, 0:K, 0:V]     = fp32(h)
hm[i_h, 0:K, V:V+K]   = m
```

**必须保持的舍入点**（与上游逐位对齐的关键）：

1. `v_decay` 的 dot 中 `h` **先降到 BF16 再入 dot**；
2. `h` 的累积 `h += K^T @ bf16(v_new)` 在 FP32 累加器上进行，不被中途舍入；
3. `m` 的整条链在 FP32 内完成，不在 chunk 之间降精度；
4. `hm` 为 FP32 输出，不做 BF16 舍入。

## 5. 支持矩阵（本轮）

| 维度 | 范围 |
| --- | --- |
| 算法族 | **GDN（`g`）+ KDA（`gk`）**；DPLR（`gk` + `bg`）**不支持**：`bg` / `v` 必须在接口上为空，非空在这里（以及 host/aclnn/ctypes/stable）直接拒绝 |
| `K` | **固定 128**（编译期规格，host 拦截其它值） |
| `V` | **固定 128**（同上） |
| `BT` | **固定 64**（同上） |
| `B` | **恒为 1**（CP 契约）。需要处理多条序列时由调用方打包进 T 轴（§3.2） |
| 序列 | **唯一形态：varlen 打包窗口**（`B ≡ 1` + `cu_seqlens` 必给，`Nseq = len(cu_seqlens) - 1 >= 1`），支持单段、等长多段、不等长多段、尾块与不满一个 chunk 的尾部；`hm` 前导维 = 链条数 `Nseq`（§3.4） |
| CP 场景 | 竞品在 CP 下**恒为 `B = 1`（varlen 打包）**：一个 rank 的局部窗口是打包轴上的切片，可能含多段；每段各自从零状态开始，只有"被左边界切开的那条"需要非零初始状态。本算子只负责产出各段的链，接续/复合由编排层做 |
| dtype | `k/w/u` BF16；`g/gk` FP32 或 BF16；`hm` FP32 |
| head | `HV >= HK` 且 `HV % HK == 0`；GVA 支持，`k` 在 `HK` 维、其余在 `HV` 维 |
| SoC | Ascend950 / `NpuArch=3510` |

## 6. 边界与异常

- 空窗口（`T = 0`）：host 侧拒绝，返回非零错误码。
- `K != 128`：host 侧拒绝。
- `V != 128`：host 侧拒绝。
- `chunk_size != 64`：host 侧拒绝。
- `g` 与 `gk` 同时提供或同时缺失：host 侧拒绝。
- 提供 `bg`（DPLR 的 K 侧项）：host 侧拒绝（`DPLR is not implemented in this release`）；
  提供 `v`：同样拒绝（它是 DPLR 专用位，GDN/KDA 的取值来自 `u`）。
- `cu_seqlens` 非严格递增、`cu_seqlens[i] >= T`（该段起点越界）、或 `cu_seqlens[-1] > T`：host 侧拒绝。
  **注意 `cu_seqlens[0] > 0` 与 `cu_seqlens[-1] < T` 都是合法的**（子区间窗口 = 竞品用法，见 §3.2）。
- 变长模式下存在**零长段**（`cu_seqlens[i] == cu_seqlens[i+1]`）：host 侧拒绝（要处理多段不等于允许空段）。
- `B != 1`：host 侧拒绝（CP 契约固定 `B = 1`）。**定长 `[B,T,H,D]`（`B>1`）不是本算子的输入模式**：
  要么按 §3.5 示例 1b 打包成 `[1,H,B·T,D]` + `cu_seqlens=[0,T,2T,…]`（一次调用拿 B 条链），
  要么逐条调用 B 次（每次 `cu_seqlens=[0,T]`，无打包搬运、但 B 倍固定开销与通信量）。
  竞品同样如此——它的 CP 路径恒要求 `B=1`（README 原文 "CP expects `B == 1` for varlen"），
  非 CP 定长根本不调 pre_process。
- `cu_seqlens` 缺失（`nullptr`）：host 侧拒绝（唯一形态要求必给）。
- `HK > HV` 或 `HV % HK != 0`：host 侧拒绝。
- K 或 V 非 `BLOCK_SIZE` 整数倍：允许，由尾块掩码处理，`hm` 越界区域不写。
- `hm` 的非有效区域（`k >= K` 或列越界）不写、不参与比较。

## 7. 性能目标

**对标 1.0 倍 H20**：在约定的模型 case 上，本算子的 kernel 耗时必须达到 H20 上同一
`pre_process_fwd_kernel_merged` kernel 的 1.0 倍以内（即 `T_npu <= 1.0 x T_h20`）。

| 项目 | 取值 |
| --- | --- |
| 对标对象 | NVIDIA H20 上运行的上游 Triton kernel `pre_process_fwd_kernel_merged` |
| 目标倍率 | 1.0x |
| 模型 case（shape / dtype / SoC） | **待用户提供**（见 §8） |
| H20 基线数值与采集口径 | **待用户提供**（预热次数、采样次数、统计量、是否含上下游 kernel） |
| 判定指标 | `msprof` `op_summary` 的 `Task Duration(us)`，与本算子自身基线同条件比较 |

本机没有 NVIDIA 设备，H20 基线无法在本地复现，只能采用用户提供的实测数据；因此该 case 的
shape、采集口径与基线数值必须在 03 阶段进入设计前固定下来。

## 8. 尚未解决、需要用户选择的问题

1. **H20 基线数据**：需要给出对标模型 case 的完整 shape（`B / HK / HV / T / K / V /
   chunk_size`、dtype、是否 varlen）、H20 上的实测耗时，以及采集口径（预热/采样次数、
   统计量、是否只统计该 kernel）。这是 03 阶段性能设计的输入，缺失时 `stage` 不能进入 03。
2. **V 维是否合并**：**已定稿为合并**。03 阶段按昇腾的算力/带宽脊点（252，H20 为 37）论证：
   上游式在昇腾上是带宽受限，合并读 `k`/`w` 后回到计算受限侧；落地方式是共享同一个
   `kᵀ`（`[ΔH | Kw]` 用一份 L0A），详见 `docs/design.md` 2.4 与 3.4。
3. **命名门禁偏离**：本流程的 Step 0/Step 1 要求算子名包含 `catlass`，而用户指定的
   `pre_process_fwd_kernel_merged` 不含该子串（为与上游 kernel 对齐而有意偏离）。已按用户
   明确要求执行，`select_operator_workflow.py` 的 `legacy` 分支判断在新建目录时为
   `route=pending`，不构成阻塞，但该偏离需在交付说明中记录。
4. **`g`/`gk` 数值域**：上游无条件使用 `exp2`（base-2 chunk 内累积）；本仓 `chunk_fwd_h` 有
   `use_exp2` 开关、默认自然对数。本算子暂固定 base-2（与上游一致），请确认是否需要在
   Python 适配层做换算。
5. **`u`/`v` 接口形态**：**已定稿** —— `v` 随 DPLR 一起退场（本版本不支持 DPLR），
   取值统一由 `u` 给出；`v` / `bg` 只保留 ABI 参数位并对非空值报错（§0 2026-09-30、§6）。
6. **测试落点**：本流程默认在算子工程内用 `test/` + `scripts/compare_precision.py`；本仓另有
   `tests/atk/<op>/` 的 ATK 验收工程。需要在 02/05 阶段确定两套入口的分工（建议：开发期用
   流程自带入口，最终验收补 `tests/atk/pre_process_fwd_kernel_merged/`）。
