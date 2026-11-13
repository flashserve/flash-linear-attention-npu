# GDN / KDA CP 示例

本目录分别提供 [GDN](chunk_gdn_cp.py) 和 [KDA](chunk_kda_cp.py) 的 CP 前后向调用示例。加载 CANN，并激活安装了匹配平台 `torch`、`torch_npu` 和 `fla_npu` 的环境，在仓库根目录执行下面的命令；设备号替换为分配到的空闲卡。

| 示例 | CP 调用方式 | 切分及长度要求 |
| --- | --- | --- |
| [GDN](#gdn) | `--communication none/allgather/ptp`，CP 卡数由 `torchrun` 决定 | 连续或 zigzag，支持 packed；连续全局长度整除 `N`，zigzag 整除 `2 × N`，无需局部长度按 64 对齐 |
| [KDA](#kda) | `--communication none/allgather/ptp`，CP 卡数由 `torchrun` 决定 | 连续或 zigzag，支持 packed；连续全局长度整除 `N`，zigzag 整除 `2 × N`，无需局部长度按 64 对齐 |

两者都沿 token 维切分，每卡保留全部 head。以下分别说明各自的输入参数、接口和支持范围。

## GDN

### 环境与支持范围

使用包含主线 `34851b76c9986aeb9dd9826a4ca57c3ba0c205f1` 的匹配平台算子包；该版本已合入 A2/A3 的独立 Prepare。加载 CANN 后，激活安装了 `torch`、`torch_npu`、`fla_npu` 的环境，安装说明见[开发者指南](../../docs/开发者指南.md)。

```bash
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HCCL_EXEC_TIMEOUT=180
```

使用包的默认 Python 后端。需要 `chunk_gated_delta_rule_fwd_prepare`、`chunk_fwd_h`、`chunk_fwd_o`、`chunk_bwd_dv_local`、`chunk_gated_delta_rule_bwd_dhu`、`chunk_gated_delta_rule_bwd_finalize`，以及 `pre_process_fwd_kernel_merged`、`pre_process_bwd_kernel_merged`、`merge_fwd_bwd_kernel`。

- 接口覆盖 A2/A3/A5；各平台实机验证结果需分别确认。本次先验证 A2，不能替代 A3/A5 验证。
- B=1、K=V=128、BF16 Q/K/V/do、FP32 g/beta、chunk_size=64。保留 GVA，示例范围为 `Hv/Hk ∈ {1,2,3,4}`。
- Q/K 在进入示例计算前已做 L2Norm，beta 是已激活值；g 是自然对数域逐 token gate。返回的是对这些输入的梯度，不包含前置 L2Norm/sigmoid 的反向。
- 保存前向 A、W/U、h、v_new，反向直接使用，不重新计算前向中间量。
- 支持连续等长或 zigzag 切分，以及 packed/空序列边界；不支持外部初始状态、末状态损失和不等长分片。多机、整层卷积/FSDP 不在本次验证范围。

### 运行：不开 CP、AllGather、PTP

`--tokens` 是切分前的全局 token 数，`torchrun --nproc_per_node` 决定本机 CP 卡数。以下设备号需替换为分配到的空闲卡。

```bash
# 不开 CP：仍走相同小算子链，跳过 CP 预处理和通信。
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_gdn_cp.py \
  --communication none --accuracy-check

# AllGather，CP=4。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_gdn_cp.py --communication allgather --accuracy-check

# PTP，CP=4，GVA：Hk=2、Hv=6。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_gdn_cp.py --communication ptp \
  --heads 2 --value-heads 6 --accuracy-check

# PTP + zigzag + packed，每卡持有两段，每段 65 token。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_gdn_cp.py --communication ptp --partition zigzag \
  --tokens 520 --heads 2 --value-heads 6 \
  --cu-seqlens 0,0,1,64,65,130,131,259,260,519,520 --accuracy-check

# 仅 CPU 参考，不执行 NPU 或通信。
python examples/cp/chunk_gdn_cp.py --tokens 128 --cpu-reference-only
```

连续切分要求 `tokens >= CP` 且整除 CP；zigzag 要求 `tokens >= 2*CP` 且整除 `2*CP`。局部长度和单条 packed 序列无需按 64 对齐，head 数无需整除 CP。分片规则及 AllGather/PTP 的摘要扫描与下文 KDA 相同；GVA 的状态/摘要使用 Hv 个 head。

### 前后向调用链与输入

`run_gdn(x, plan, communication)` 中直接列出全部阶段，A2/A3/A5 通过相同公共接口调用：

```text
前向：FwdPrepare → pre_process_fwd → exchange_boundary → FwdH(h0) → FwdO
反向：DvLocal → pre_process_bwd → exchange_boundary → Dhu(dht) → BwdFinalize
```

`exchange_boundary` 选择 AllGather + Merge 或 PTP 递归倍增扫描。AllGather 收集各段摘要后合并；PTP 在通信轮次间合成摘要，反向按相反的序列顺序扫描。两种方式都只把同一条 packed 序列的状态传到对应边界。FwdH 接收 FP32 h0，Dhu 接收 BF16 dht；不开 CP 时使用 FP32 零初态，以保持基准和 CP 的状态精度一致。

Prepare 输出以 2 为底的分块累积门控，供 FwdH、CP 预处理、Dhu 和 Finalize 使用。DvLocal 内部使用自然指数，因此接收 `gc * ln(2)`。A2/A3 FwdO 也使用自然指数并输出 BNSD，示例随后转为 BSND 视图；A5 FwdO 直接使用 exp2 并输出 BSND。直接调用 `run_gdn` 时用 `is_a5=True` 选择后者，CLI 自动识别设备。GVA 的共享 Q/K 梯度由 Finalize 在组内求和。

`plan` 由 `build_partition(global_cu, global_tokens, rank, world, partition)` 生成；输入按 `plan["parts"]` 顺序取本地片段，zigzag 为前段后接后段。设本地总长度为 `L=tokens/world`：

| `x` 键 | dtype | 本地 shape | 含义 |
| --- | --- | --- | --- |
| `q`、`k` | BF16 | `[1,Hk,L,128]` | 已归一化的 Q/K |
| `v`、`do` | BF16 | `[1,Hv,L,128]` | value 和上游输出梯度 |
| `g` | FP32 | `[1,Hv,L]` | 自然对数域逐 token gate，未做 cumsum |
| `beta` | FP32 | `[1,Hv,L]` | 已激活的更新系数，非 logits |

返回 `o/dq/dk/dv/dbeta/dg`；token 输出为 BSND，beta/g 梯度为 BSN。CP 模式额外返回 `h0/dht`，shape 为 `[本地逻辑序列数,Hv,128,128]`，无跨卡依赖的边界为零。`dq/dk` 使用 Hk，其他输出使用 Hv。

这是显式前后向示例，不是 autograd 层。直接调用前需初始化 NPU；CP 模式需初始化 HCCL。模型集成应使用模型的 CP 通信组，不能直接把默认进程组当作所有并行场景的 CP 组。

### 精度检查、计时与 profiling

`--accuracy-check` 使用独立 FP32 递推/autograd 标杆，检查输出、五项梯度以及 CP 的 h0/dht。CPU 标杆限制 `tokens * value_heads <= 8192`，这是参考计算的资源限制，不是算子输入限制。输出、dq/dk/dv、gate 梯度、边界的 atol/rtol 默认分别为 0.005、0.008、0.02、0.02，同时检查有限性和余弦相似度；不通过时返回非零退出码。

大用例使用 `--save-result` 保存同包不开 CP 的 NPU 结果，再用 `--reference` 比较；这不能替代独立 CPU 精度验收。计时排除输入生成、CPU 参考和文件写入，包含完整前后向及 CP 通信，报告最慢 rank 的平均耗时。`--profile` 单独采集一轮预热后的执行。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_gdn_cp.py \
  --communication none --tokens 131072 --heads 32 \
  --cu-seqlens 0,70290,129130,131072 --warmup 5 --iterations 20 \
  --save-result /path/to/gdn-no-cp.pt --output-dir /path/to/gdn-none --profile

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_gdn_cp.py --communication ptp \
  --tokens 131072 --heads 32 --cu-seqlens 0,70290,129130,131072 \
  --warmup 5 --iterations 20 --reference /path/to/gdn-no-cp.pt \
  --output-dir /path/to/gdn-ptp --profile
# 将 ptp 改成 allgather，并使用独立 output-dir，可采集 AllGather。
```

每卡结果在 `<output-dir>/rankN/result.json`；profiling 在对应的 `profiling/PROF_*` 下。导出原始 CSV：

```bash
msprof --export=on --output=/path/to/PROF_directory --summary-format=csv
```

保留每卡的 `op_summary*.csv`、结果、源码版本和 wheel hash。PTP 的主要区别是摘要存储和传递方式，并不保证耗时比 AllGather 低。


## KDA

The verified operator baseline is main commit `2bb7e74ecf4876acbc4527070e403bffeef06e75`
(PR #993). Use a matching platform wheel containing the staged KDA and CP
operators; do not mix the old backward ABI with this example.

### Supported path

- A2/A3/A5, B=1, Hk=Hv, K=V=128, BF16 Q/K/V, FP32 raw gate/beta/parameters.
- Equal contiguous or zigzag token shards, including packed and empty sequences.
- Fused QK L2Norm and safe gate (`lower_bound=-5`); beta sigmoid mode depends on the platform.
- Saved intermediates (`disable_recompute=True` / Prepare `backward_mode="save"`).
- Zero initial state, no final-state loss; explicit forward/backward calls.
- Output and all five token gradients are BSND (beta gradient is BSN).

`--beta-sigmoid auto` (default) uses fused sigmoid on A2/A3. A5's fused
sigmoid backward is reserved, so A5 computes `beta_eff = sigmoid(beta)` in
FP32 before the operator and applies `d_beta = d_beta_eff * beta_eff *
(1-beta_eff)` after backward. Both no-CP and CP use this same rule. The selected
mode is recorded in every result; timing/profiling includes these operations.
`--beta-sigmoid external` also allows comparison on A2/A3. Explicit `fused`
on A5 is rejected. This adapts the example, not the A5 kernel's capabilities.

### CP input contract

Partition metadata follows upstream FLA commit
[`f2ca71c0afb3`](https://github.com/fla-org/flash-linear-attention/blob/f2ca71c0afb3254b564471e152e73ba5d1cddcaa/fla/ops/cp/context.py).
The caller partitions the sequence dimension; each rank retains every head.

| Condition | This example and upstream FLA |
| --- | --- |
| Contiguous partition | Total tokens >= CP size and divisible by CP size |
| Zigzag partition | Total tokens >= 2 * CP size and divisible by 2 * CP size |
| Local length / individual packed sequence length | No multiple-of-64 requirement; partial chunks use real sequence boundaries |
| Head count | No divisibility requirement with CP size |
| Packed input | B=1; global cumulative lengths become local boundaries; empty segments are omitted |
| External initial state / final-state loss | Not supported in CP |

`--partition zigzag` assigns global intervals `r` and `2*world-1-r` to rank
`r`, concatenated as `[front; back]`. Each interval has `tokens/(2*world)`
tokens. An explicit local boundary separates the two intervals, even if they
belong to the same global sequence: the CP state carries their dependencies.
AllGather reorders summaries into global sequence order before merge. PTP
scans the same interval chain, copying same-rank edges locally and matching
cross-rank messages in global edge order. In either scan direction, PTP retains
only the local intervals' running summaries rather than gathering all summaries.

The example also validates its benchmark options (positive iteration count,
an output directory for profiling, and a size cap for the CPU autograd
reference). The CPU reference cap is not an operator input limit: use a saved
no-CP `--reference` for larger cases.

#### Operator capabilities outside this CP change

CP partition compatibility does not imply that all upstream KDA tensor modes
are implemented in the current AscendC operators. This change does not extend
the following capabilities:

| Capability | Upstream FLA | Current AscendC example |
| --- | --- | --- |
| Head dimensions | K <= 256 at the public entry, subject to backend/kernel constraints | K=V=128; also enforced by the CP pre-process kernels |
| Chunk size | 32 or 64 at the public entry | 64 |
| Grouped value heads | HV divisible by H | Hk=Hv in this backward composition |
| Q/K/V dtype | Backend-dependent, including FP16/BF16 | BF16 in the KDA backward stages |
| Gate / normalization options | Configurable | Fused L2Norm and safe gate with lower_bound=-5; A5 sigmoid is external |
| State representation | K-first or V-first | K-first internally |
| Recompute | Configurable | Saved intermediates, as requested for this example |
| Platforms / graph capture | Backend-dependent; Ascend CP graph mode currently rejected | A2/A3/A5 eager execution |

These are remaining operator or example-mode differences, not new restrictions
on sequence partitioning. Do not remove kernel guards to claim broader support.

### Communication selection

```bash
# Single NPU, total forward/backward public interfaces.
python examples/cp/chunk_kda_cp.py --communication none --accuracy-check

# Four NPUs, staged forward/backward with boundary exchange.
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication allgather --accuracy-check

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication ptp --accuracy-check

# Two 65-token intervals per rank, including unaligned packed boundaries.
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication ptp --partition zigzag \
  --tokens 520 --heads 3 --cu-seqlens 0,0,1,64,65,130,131,259,260,519,520 --accuracy-check
```

AllGather collects one `[E | M]` FP32 summary per interval (two per rank for
zigzag) and calls
`merge_fwd_bwd_kernel` on same-sequence predecessors (forward) or successors
(backward). PTP implements the inclusive/exclusive scan in
[MindSpeed issue #4, section 4.6](https://gitcode.com/Liz_/MindSpeed/issues/4):
distances 1, 2, 4, ...; each round sends the previous inclusive summary and
updates the local exclusive summary. Backward reverses the logical interval order.
It uses PyTorch FP32 matrix multiplication for summary composition; it is not
a new fused AscendC merge kernel and does not guarantee lower latency.

Packed resets zero the outgoing affine matrix when an interval contains a sequence
boundary; only its first local sequence can have incoming forward state, and
only its last can have incoming backward state. State exchange returns FP32
`h0` to FwdH and BF16 `dht` to Dhu. Shared parameter gradients are summed across
ranks in both communication modes.

### Current backward interface

The operator calls are kept together in `run_kda()`:

```text
forward:  FwdPrepare -> pre_process_fwd -> exchange_boundary -> FwdH(h0) -> FwdFinalize
backward: BwdPrepare -> pre_process_bwd -> exchange_boundary -> Dhu(dht) -> BwdFinalize
```

`exchange_boundary()` selects AllGather + Merge or PTP summary composition.
It produces the boundary state; FwdH/Dhu then propagate it through local chunks.
After backward, `run_kda()` sums `dA_log` and `ddt_bias` across the default
process group. Model integration must use its CP group and avoid reducing these
gradients twice. This example uses explicit backward calls, not an autograd layer.

`run_kda(x, plan, communication, fuse_beta_sigmoid=...)` takes local contiguous
head-major inputs. `plan` comes from `build_partition()` using global packed
offsets and the global token count; inputs are sliced in `plan["parts"]` order.

| `x` entry | dtype | Local shape | Meaning |
| --- | --- | --- | --- |
| `q`, `k`, `v`, `do` | BF16 | `[1,H,L,128]` | Q/K are not pre-normalized; `do` is the output gradient |
| `beta` | FP32 | `[1,H,L]` | Raw sigmoid logits, including external sigmoid mode |
| `g` | FP32 | `[1,H,L,128]` | Raw gate, before activation and cumsum |
| `A_log` | FP32 | `[H]` | Shared across CP ranks |
| `dt_bias` | FP32 | `[H,128]` | Shared across CP ranks |

Here `L = tokens / world`, including both local intervals in zigzag mode.
Direct A5 calls must set `fuse_beta_sigmoid=False`; the CLI selects this automatically.

The total backward (no CP) uses `layout="BSND"`: V and output token gradients
are BSND; `do`, raw gate and original beta are always token-major. Saved
normalized Q/K, reciprocal norms and `beta_eff` stay head-major. Both original
beta and FP32 `beta_eff` must be passed with fused sigmoid enabled. Without
fused sigmoid, pass the effective beta as `beta` and set `beta_eff=None`.

The CP path must use staged backward to inject `dht` into Dhu. Small Prepare uses
`layout="NTD"` for A5 variable-length support (also supported on A2/A3): remove
the unit batch axis from Aqk, Vnew, do and h, then restore it on the outputs.
These are views without data copies. Small Finalize
takes head-major tensors and **saved beta_eff**, with `output_layout="BSND"`
controlling all five token gradients. These are different contracts: do not
copy the total backward argument order/layout into small Finalize. No custom
ctypes ABI or model-side autograd adapter is used by this example.

### Packed validation and profiling

For CPU reference validation, keep `tokens * heads <= 8192`. The recurrence
checks forward, all seven gradients, and cross-rank h0/dht. Example:

```bash
torchrun --standalone --nproc_per_node=4 examples/cp/chunk_kda_cp.py \
  --communication ptp --tokens 512 --heads 2 --cu-seqlens 0,0,130,257,512 \
  --accuracy-check
```

Large cases use the same-seed no-CP NPU result as a separate comparison (not an
independent CPU accuracy certification). Warmup, timing and profiling exclude
input generation, accuracy checks, file writes and CPU copies. Timing includes
both forward/backward and shared parameter-gradient AllReduce, and reports the
maximum rank time; profiler collection is a separate warmed iteration.
Error norms and dot products accumulate in FP64 to avoid inaccurate FP32
reductions on long vectors. The original example's mixed tolerances and cosine
thresholds remain unchanged; failures are recorded and return a nonzero status.

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python examples/cp/chunk_kda_cp.py \
  --communication none --tokens 131072 --heads 32 \
  --cu-seqlens 0,70290,129130,131072 --warmup 5 --iterations 20 \
  --save-result /path/to/no_cp.pt --output-dir /path/to/none --profile

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication allgather \
  --tokens 131072 --heads 32 --cu-seqlens 0,70290,129130,131072 \
  --warmup 5 --iterations 20 --reference /path/to/no_cp.pt \
  --output-dir /path/to/allgather --profile
# Repeat the second command with --communication ptp and a separate output-dir.
```

`--profile` writes CANN data under `<output-dir>/rankN/profiling`. Export each
`PROF_*` directory with the installed CANN tool:

```bash
msprof --export=on --output=/path/to/PROF_directory --summary-format=csv
```

Keep `op_summary*.csv` for every rank, `result.json`, source revision and wheel
hash. Never sum overlapping HCCL/AICPU activity durations into wall time.
