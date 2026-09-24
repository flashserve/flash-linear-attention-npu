# ChunkDeltaHBwdPreprocess

## 功能

把本 rank 边界序列的反向状态递推压缩成仿射摘要：

```text
dH_start = P_r @ dH_end + E_r
```

算子一次 launch 处理**一个** `[bos, eos)` segment，输出一个 FP32 摘要张量 `dhm`：

```text
dhm[hv, :, 0:V]   = E_r    # 与末端 dH 无关的本地常量项
dhm[hv, :, V:V+K] = P_r    # 末端 dH 到起始 dH 的反向线性映射（FP32 链）
```

`E_r` 不是完整 `dh0`，`P_r` 也不是参数梯度。两者的组合与跨 rank 合并由框架侧 all-gather 与后续的
`ChunkDeltaHBwdMerge` 完成。

## 输入 / 输出

| 名称 | 类型 | Shape | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `q` | BF16/FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `q`；KDA/GDN2 传已应用 key gate 的 `qg` |
| `k` | BF16/FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `k`；KDA/GDN2 传 `kg` |
| `w` | BF16/FP16 | `[B, Hv, T, K]` | 是 | WY 辅助量 `w_wy`（不含公开输入 `w_gate`） |
| `d_o` | BF16/FP16 | `[B, Hv, T, V]` | 是 | 输出梯度 `dO` |
| `dv` | BF16/FP16 | `[B, Hv, T, V]` | 是 | 不含未来状态项的本地 `dV`（`dv_local`） |
| `g` | BF16/FP16/FP32 | `[B, Hv, T]` | 否 | GDN 的 chunk-local 累计标量 log2 gate |
| `gk` | BF16/FP16 | `[B, Hv, T, K]` | 否 | KDA/GDN2 的 chunk-local 累计逐 K log2 gate |
| `cu_seqlens` | INT64 | `[N+1]` | 否 | 给出本 segment 的 `[bos, eos)`；dense 时不传，取 `[0, T)` |
| `dhm` | FP32 | `[Hv, K, V+K]` | 输出 | 摘要张量，前 `V` 列为 `E_r`，后 `K` 列为 `P_r` |

属性：

| 名称 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `scale` | float | 1.0 | 只乘 `Q̄ᵀ dO` 项，通常为 `K^-0.5` |
| `chunk_size` | int | 64 | 必须与 gate cumsum、WY 生成、正式 backward 一致 |

## 支持的场景

| 维度 | 支持范围 |
| --- | --- |
| gate 模式 | `USE_G`（GDN 标量 gate）、`USE_GK`（KDA/GDN2 逐 K gate）、无门控 |
| 序列 | dense（`B=1`，整段 `[0,T)`）；varlen（一次处理 `cu_seqlens[0:2]` 给出的一个 segment） |
| 状态行维 | `K <= 256`，K 方向按 64 行分组，最多 4 组 |
| 状态列维 | `V` 任意，列方向 tile 宽度 `BLOCK_SIZE = 32 if K <= 64 else 64` |
| head | `Hk == Hv` 或 `Hv % Hk == 0`（GVA，`hk = hv // (Hv/Hk)`） |
| layout | `[B, H, T, D]`（BSND）；TND/NTD 需由调用方或 L2 侧做 layout sweep |
| dtype | `q/k/w/d_o/dv` 为 BF16 或 FP16；`dhm` 与内部链固定 FP32 |

## 不支持（本版显式拦截）

| 场景 | 行为 |
| --- | --- |
| `USE_BG`（DPLR：用 `bg` 顶替 `k`、加号链、无 `scale` 的 query 项） | 不支持；DPLR 需要独立分支 |
| `AFFINE_CHAIN_PRECISION`（`tf32x3` / `ieee` 等链精度开关） | 不支持；`P` 链固定 FP32 |
| `g` 与 `gk` 同时非空 | host 直接拦截（上游没有 assert，本仓不沿用这个缺口） |
| `K > 256` | host 拦截 |
| `Hv % Hk != 0` | host 拦截 |
| `B > 1` 且未传 `cu_seqlens` | host 拦截：一次 launch 只处理一个 segment |
| 非 64 的 `chunk_size` | 与本版 tiling 不匹配，host 拦截 |

## 调用

```python
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import chunk_delta_h_bwd_preprocess

dhm = chunk_delta_h_bwd_preprocess(
    q=qg, k=kg, w=w_wy, d_o=do, dv=dv_local, gk=gk,
    scale=K ** -0.5, chunk_size=64, cu_seqlens=cu_seqlens,
)
```

aclnn 与 `<<<>>>` 直调见 [`docs/api.md`](docs/api.md)。

## 当前实现状态

| 部分 | 状态 |
| --- | --- |
| `op_host`（def / tiling / op_api） | 已实现：shape 校验、gate 互斥拦截、分核规则、workspace 规划、tilingKey 分发 |
| `op_kernel` | 已实现：六个 Stage 的真实计算体（Catlass `BlockMmadTla` 五路矩阵乘 + Vector 手写事件对），A2/A5 精度均已打通 |
| 构建接入 | 已完成：`op_host/CMakeLists.txt` 走 `op_host_aclnnExc` + `ACLNNTYPE aclnn_exclude`（**aclnn 手写，不走自动生成**），A2/A5 均已编出 OPP 运行包 |
| Python `fla_npu.ops.ascendc` 入口 | 待接入（当前分支缺少 `ops/ascendc/_runtime.py` 适配层，接入步骤见 `docs/design.md`） |
| 测试 | `tests/op_cases/chunk_delta_h_bwd_preprocess.json`（16 正向 + 10 反向）已作为唯一用例来源；`tests/operators/chunk_delta_h_bwd_preprocess/` 提供 CPU 标杆、aclnn 取数程序、逐平面核对与精度矩阵执行脚本 |

### 已验证（本版）

| 项 | 证据 |
| --- | --- |
| A5 / ascend950 编译 | 六个 Stage 的真实实现落地后 `build.sh --pkg --soc=ascend950 ... --ops=chunk_delta_h_bwd_preprocess` 成功，**0 编译错误**，产出 `build_out/fla-npu-fla_npu_linux-x86_64.run` 与 `build/binary/ascend950/bin/chunk_delta_h_bwd_preprocess` |
| A2 / ascend910b 编译 | 同一实现 `--soc=ascend910b` 成功，**0 编译错误**，产出 `build_out/fla-npu-fla_npu_linux-aarch64.run` |
| OPP 安装内容 | 运行包安装后含 `op_api/include/aclnnop/aclnn_chunk_delta_h_bwd_preprocess.h`、`op_api/lib/libcust_opapi.so`、`op_impl/ai_core/tbe/kernel/ascend950/chunk_delta_h_bwd_preprocess`（kernel bin + config + tiling py） |
| aclnn 手写路径 | 构建产物含 `build/autogen/exc/aclnnExc_chunk_delta_h_bwd_preprocess.cpp/.h` 与 `aic-ascend950-ops-info.ini`，确认走的是 aclnn 排除自动生成的 exc 通路 |
| 参考实现数学口径 | 纯 Python 数值校验：三种 gate 模式下 `dh0 == P_r @ dht + E_r` 的最大绝对误差 ~1e-16 |
| 设备侧精度（A5 / ascend950） | 按 `tests/op_cases/chunk_delta_h_bwd_preprocess.json` 的 16 条正向用例在设备上执行：**16/16 PASS**。`E` 面 `rel_norm ≤ 1.2e-2`、`P` 面 `rel_norm ≤ 1.1e-2`（单 task 场景 ~2e-3；`pos_13` 32 个 chunk 的长链 1.2e-2 为最大）；无 NaN/Inf |
| 设备侧精度（A2 / ascend910b） | 同一 16 条用例在同一实现上执行：**16/16 PASS**，误差量级与 A5 一致（`E ≤ 1.5e-2`、`P ≤ 1.1e-2`） |
| 数值稳定性 | 关键用例（`pos_15` 96 head 多 task、`pos_13` 32 chunk 长链、`gate_gk`）重复执行结果逐位一致；此前出现过的"随负载时好时坏"已定位到跨核 wait 的 pipe 语义并修复 |
| 设备侧反向拦截 | 按 `op_cases` 的 10 条反向用例在 A2/A5 上执行：**10/10 PASS**，实际返回码与 `expected_return_code` 一致（`ACLNN_ERR_PARAM_INVALID`=161001），覆盖 `g`/`gk` 互斥、`K>256`、`Hv%Hk!=0`、dense/varlen `B>1`、`chunk_size!=64`、`g` shape 不匹配、`gk` FP32、`cu_seqlens` 过短、空 tensor |
| A3 / ascend910_93 | `def.cpp` 已声明该平台配置，但**本版未在 A3 设备上执行验证**（无可用设备）；编译与精度结论只覆盖 A2/A5 |

### 尚未完成（下一步）

1. **Python 入口**：在具备 `ops/ascendc/_runtime.py`、`_aclnn_ctypes.py` 适配层的分支上接入 ctypes/aclnn 主入口
   （默认路径不依赖 `torch_npu` dispatcher）。
2. **列 tile 展平分核（`BY_TILE`）**：本版 host 固定"仅按 head 连续分核"，`Hv` 不足核数时也不会按列 tile 展平
   （`pos_16_tile_split_partition` 当前实际按 `BY_HEAD` 执行，结果仍与标杆一致，但并行度未展开）。
3. **1AIC:2AIV 与流水重叠**：本版 `KERNEL_TYPE_MIX_AIC_1_1`（1:1 配对）且同 chunk 内 AIV/AIC 严格交替，
   没有跨 chunk 的 ping-pong 重叠；1:2 需要两个 AIV 分工并各自协调 flag。
4. **sanitizer**：按仓库规范补做 `mssanitizer`（`racecheck`/`memcheck`/`initcheck`/`synccheck`），并确认运行命中的是
   sanitizer 版本对象。
5. **官方 ATK / CI 接入**：把本目录的精度矩阵与反向拦截用例并入 `tests/atk`、`ci/` 的既有流程。
6. **精度判据口径**：本版 `E_r/P_r` 的目标是"跨 rank 仿射摘要"，链上用模型 dtype 传递状态（`Pc`/`PBf`/`dHBf` 都是
   bf16/fp16），因此判据采用"相对参考幅值"（详见测试 README），不是逐元素绝对阈值。

### 精度取数链路（已验证）

```bash
INSTALL=<install root> CASE_DIR=<case dir> bash tests/operators/chunk_delta_h_bwd_preprocess/harness/run_accuracy.sh \
  --dtype bf16 --gate gk --Hk 4 --Hv 4 --T 256 --K 128 --V 128
```

- 输入/期望值由 `make_case.py` 生成（固定种子，落盘后两侧共用同一份 bytes），`run_case` 用 aclnn 直调取数，
  `compare.py` 分别给出 `E_r` / `P_r` 平面的 max_abs / max_rel。
- **对照实验**：把 kernel 编成空实现（`-DCDHP_DEBUG_EMPTY`，仅用于定位，不作为交付）后，同一用例立即返回、
  `dhm` 全 0、比对流程正常结束 ⇒ aclnn 启动、workspace、D2H 均可用，挂死只可能来自 kernel 内部。

### 编译期踩坑记录（A2/A5 都要用）

- Catlass 的架构实现由 **`CATLASS_ARCH`** 选择（`2201` = atlasa2/A2，`3510` = ascend950/A5）。不定义该宏时
  `gemm/tile/tile_copy.hpp` 里 `ScaleGranularity` / `CopyToGM` 会直接编译失败；定义成与
  `__CCE_AICORE__` 不匹配的值会把两个架构实现同时拉进来，报重定义。
- **不要引用仓内 `common/kernel_utils/block/block_mmad_pingpong_tla.hpp`**：它经由
  `kernel_utils/tile/copy_l0c_to_ub.hpp` 无条件包含 `catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp`，
  在 A2（910b）构建里必然报 950 实现重定义。本算子改用 catlass 原生 `BlockMmadTla`。

### 运行期踩坑记录（已修，含设备报错原文）

- **`CrossCoreWaitFlag(flag)` 的默认模板参数是 `pipe = PIPE_S`**：在 A5（`dav_3510`）上 `WaitEventImpl` 会按 pipe
  分派，默认值只会阻塞标量 pipe，随后的 `MTE2`（GM→UB/L1）载入不会被拦在 flag 之后，会读到 cube 还没落盘的
  中间平面；表现为误差随设备负载时好时坏（历史症状：`wterm` 出现整块旧值、`P` 链尾部为 0）。**必须显式写
  `CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag)`**，让消费数据的 pipe 真正等待。
- **`CrossCoreSetFlag` 之前要让本 pipe 排空**：`PipeBarrier<PIPE_FIX>()`（cube）/ `PipeBarrier<PIPE_MTE3>()`（vector）
  之后再 set flag，否则生产侧还有未完成的落盘，消费者可能读到半成品平面。
- **跨 task 复用 UB 需要 `MTE3→V`、`MTE3→S`**：`InitState` 用 `Duplicate`/`SetValue` 复写 `s0F32_/s1F32_`，而上一个
  task 的 `WriteOutput` 可能还有一笔搬出（MTE3）在读同一块 UB；`PipeBarrier<PIPE_MTE3>` 只约束 MTE3 自己，
  拦不住随后的 V/S。漏掉这一步时，**只有每个工作组最后一个 task 的输出是对的**，前面 task 的 `P` 面尾部 16 行
  会写成 `InitState` 刚写入的单位阵内容（`Hv > 核数` 时必然踩到）。
- **门控载入复用 `s0DT_` 前要等三向**：`S_MTE2`、`V_MTE2`、`MTE3_MTE2`（`GuardScratchRewrite`）。`gk` 分支的
  `gkLast` 载入曾把 `InitState` 正在落盘的 `dHBf` 覆盖掉，导致首个 chunk 的 `dV_pre` 用到被污染的 dH 操作数，
  误差随 chunk 数放大。
- **cube 侧 `BlockMmadTla` 每次构造都会重新 `SetFlag`**（`MTE1_MTE2`/`M_MTE1`/`FIX_M`），相当于默认"上一轮已排空"。
  每次 `RunGemm` 前必须自己 `SetFlag+WaitFlag` 把上一轮的 `FIX_M`/`M_MTE1`/`MTE1_MTE2` 真正等掉，否则新一轮
  MMAD 会覆盖上一轮 Fixpipe 仍在读的 L0C（或 L0A/L0B、L1），上一轮写出的平面会出现整块旧值。
- **workspace 平面必须按工作组切分**：slot 与所有中间平面都是"每工作组一份"（`dh`/`p` 每工作组再分 parity 两份）。
  只切分 slot 而共用中间平面时，后写的 head 会覆盖先写的 head；`PlaneAt` 的切片数必须和 host 规划一致，
  否则会越过平面边界写到相邻平面。
- **输出写回需要显式 `MTE2→MTE3`**：`WriteOutput` 是 GM→UB→GM，缺 `MTE2_MTE3` 时搬出会读到 UB 里的上一份内容
  （历史症状：`dhm` 前 16 行、前 16 列全 0）。
- **P 初值（单位阵）注入需要 `V→S`**：`Duplicate` 清零（V）与对角 `SetValue`（S）之间缺 `V_S` 时，标量写入的 1.0
  会被随后的向量清零覆盖，`P` 初值全 0，`P_r` 整链归零。
- **`g` 为 FP32 时不能做 `Cast<float,float>`**：`USE_G` 且 `g` 是 FP32 时直接搬进 FP32 缓冲（`if constexpr` 分支），
  走 `Cast` 会得到错误数值（历史症状：`E_r` 相对误差 ~50%）。
- **`gk` 的 dtype 校验必须取 `gk` 自己的输入描述**：`g` 缺失时 `g` 的 dtype 会落到缺省 FP32，用它校验 `gk` 会把
  合法的 bf16/fp16 `gk` 拦成 `ACLNN_ERR_PARAM_INVALID`（161001）。
- **aclnn 的 `cu_seqlens` 是 `aclIntArray`**，不是 `aclTensor`：取数程序里要用 `aclCreateIntArray` 传入，
  否则编译期类型不匹配、或运行期 varlen 语义不生效。
- **取数程序用 sentinel 初始化输出缓冲**：用全 0 预清零无法区分"kernel 没写"和"kernel 写了 0"，改用非零 sentinel
  初值后，未写区域会直接暴露。
- **逐个 16 行 tile 的输出写回 + `PipeBarrier<PIPE_MTE3>` 不够**：MTE3 的 barrier 只保证 MTE3 内部顺序，
  跨 pipe（V/S 复用同一 UB）必须用事件对，见上面的 `MTE3→V`/`MTE3→S`。
- **`MIX_AIC_1_2` 不能配裸 flag 握手**：2 个 AIV 会各跑一遍 AIV 分支、各自 set 同一 flag，而 AIC 只 wait 一次，
  必然挂死（NPU 100% 空转）。本版用 `KERNEL_TYPE_MIX_AIC_1_1`（1:1 配对），1:2 分工留作后续优化。
- **`BlockMmadTla` 必须每次调用构造**：把 `BlockMmad` 对象构造一次长期复用（我最初的写法）会让 AIC 挂死；
  与仓内 `chunk_bwd_dv_local_cube.h` 一致改成"每次 `RunGemm` 内部构造、共用同一 `Resource`"后，五路矩阵乘全部跑通。
- **UB 上的向量指令必须 32B 对齐**：对角注入最初写成 `Adds(t1F[r*kDim+(r0+r)], ..., 1)`，设备报
  `errcode:(340) The address for VEC to access UB is not aligned. subErrType: 0x4`（`ACL_ERROR_RT_AICORE_EXCEPTION 507015`）。
  改为标量读写 `SetValue/GetValue` 后消除。
- **`--cce-auto-sync=off` 下所有跨 pipe 依赖都必须手写事件对**：我的 staging 搬运最初只写了
  `PipeBarrier`，导致 `Cast`(V) 与 `DataCopyPad`(MTE3) 之间缺少 `V_MTE3`、GM→UB 与 `Cast` 之间缺少 `MTE2_V`，
  slot 的多次 store 复用同一 UB 暂存区时**拷出了上一份数据**（表现为"Q̄s/do 里出现 k"、P 不是单位阵、
  误差量级 6e3）。补齐 `MTE2_V / V_MTE2 / V_MTE3 / S_MTE3 / V_S / S_V` 后误差降到 8.4e1。
- **ping-pong parity 与首轮初值**：`C1` 消费的是模型 dtype 的 dH 操作数平面，首轮必须在 `InitState` 里一并清零；
  `V4` 读上一轮 P 应使用 `(chunkIdx+1)%2` 而不是本轮的 parity。

## 已知限制

- 一次 launch 只处理一个 segment；多 segment 需要 host 多次 launch，或在后续版本增加 segment 维 grid。
- `state_v_first` 与本算子无关：本算子只读 token-major 的 `q/k/w/do/dv`，只写固定 `[Hv, K, V+K]` 的 `dhm`。
- `dv` 必须是 `dv_local`。若调用方传入已经加入 `K̄ @ dH` 的 `dv`，状态贡献会被累计两次。
- 本版未覆盖 DPLR（`USE_BG`）与链精度开关，二者需要独立设计与用例。
