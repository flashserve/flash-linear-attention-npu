# chunk_kda_bwd_finalize ATK

This is a **Phase A ATK scaffold** modeled on `../chunk_bwd_dqkwg`.
The CPU function in `scripts/chunk_kda_bwd_finalize_cpu.py` intentionally returns **all zeros**.
Its output count, order, shape and dtype follow the ACLNN and stable-torch ABI;
it is **not a numerical golden**. CPU execution checks plumbing only. Do not use
NPU-versus-CPU results as accuracy acceptance, calibration or performance evidence.

## Inputs and constraints

Input token tensors are always head-major `[B,H,T,D]` / `[H,T,D]`.
q/k/v/v_new/dv_scan use BF16 and D=128; akk uses BF16 and D=64.
gk/raw_g/dq_raw use FP32 and D=128; d_aqk uses FP32 and D=64.
h/dh are BF16 NT-first `[B,Nc,H,128,128]` / `[Nc,H,128,128]`.
beta is `[B,H,T]` / `[H,T]`, BF16 or FP32. a_log is FP32 `[H]`,
dt_bias is FP32 `[H,128]`. Optional q_rstd/k_rstd are FP32 and match beta
shape, and are either both present or both absent (`with_rstd` test control).

Ordered outputs are:

| Output | Shape | dtype |
| --- | --- | --- |
| dq, dk, dv | `[B,H,T,128]` for BNSD; `[B,T,H,128]` for BSND; `[H,T,128]` for NTD; `[T,H,128]` for TND | BF16 |
| d_beta | **always** `[B,T,H]` / `[T,H]` | same as beta |
| d_g | **always** `[B,T,H,128]` / `[T,H,128]` | FP32 |
| d_a_log | same shape as a_log | FP32 |
| d_dt_bias | same shape as dt_bias | FP32 |

| Constraint | Source under fla/ops/ascendc/kda/chunk_kda_bwd_finalize | Mapping |
| --- | --- | --- |
| Required dtypes and contiguous tensors; a_log FP32 on A2/A3 | op_host/op_api/aclnn_chunk_kda_bwd_finalize.cpp::GetWorkspaceSize | YAML (portable FP32 a_log) |
| C64; safe_gate/use_gate/use_exp2 true; K-first state; finite scale and -5<=lower_bound<0 | same ACLNN; op_host/op_tiling/chunk_kda_bwd_finalize_tiling.cpp | YAML fixed/finite attrs |
| NQ=NV; K=V128; positive B/H/T; NT-first h/dh and Nc | op_tiling/chunk_kda_bwd_finalize_tiling_processor.h::Process | generator shared shapes, state shape |
| raw_g/gk shape, beta/rstd shape, a_log H elements, dt_bias H*128 | tiling.cpp::ValidateFinalize | generator dependent shapes |
| rstd pair; canonical packed metadata; B=1 in packed mode | ACLNN; op_api/chunk_kda_bwd_finalize.cpp; tiling processor | executor optional pair and canonical metadata |
| Output rank/layout, token-major d_beta/d_g, preserved a_log/bias shapes | ValidateFinalize; torch_custom/fla_npu/csrc/src/stable_chunk_kda_bwd_finalize.cpp | CPU output contract |
| allow_neg_eigval requires beta-sigmoid; A5 beta-sigmoid unavailable | ACLNN and ValidateFinalize | both attrs false in portable YAML |
| Overflow/row-stride guards | tiling.cpp::ValidateFinalize and Tiling4ChunkKdaBwdFinalize | bounded candidates; no address-space stress cases |

The portable matrix uses beta-sigmoid=false, allow_neg_eigval=false and FP32
a_log; A2/A3 additional sigmoid/eigenvalue modes and A5 BF16 a_log are not
included. TilingKey before arch22 specialization is `dense/packed + 2*hasRstd`
(1..4); this is source evidence, not measured device selection. Host workspace
and address arithmetic use int64/size_t with guards. Oversized overflow stress
cases are not generated or silently repaired into small shapes.


The YAML declares explicit tensor inputs and discrete shape/attribute candidates.
`gen_chunk_kda_bwd_finalize.py` only repairs cross-input relationships and required fixed dimensions;
it does not replace the YAML candidate space. Seed token axes are B/H/T/D in the
YAML; the constraint materializes the selected public layout in JSON.
All computation tensors retain ATK-generated values. Only metadata tensors are
replaced by deterministic canonical host arrays: packed `[0,0,T//2,T]` includes
an empty sequence, and indices are flattened sequence-major `(seq, local_chunk)`
pairs. `is_varlen` and `explicit_chunk_indices` are test controls removed before
calling the public operator; dense cases pass both metadata inputs as `None`.
Optional metadata tensors in candidate JSON are placeholders for those arrays.

Tensor values follow the current skill precision standard: 50% uniform `[-5,5]`
and 50% normal with mean in `[-5,5]` and standard deviation in `[0.1,2]`.
These are initial candidate ranges, not calibrated accuracy ranges. Empty token
tensors and scalar ranks are excluded because the host requires positive B/H/T
and specific ranks. Nonfinite-input/error cases and full upper limits have not
been validated. The bounded matrix covers small B/H, odd head counts, the
16-row boundary and T=63/64/65/127/128/129 chunk tails. This does not claim full
coverage of every supported dimension or runtime branch.

## Generate and check

Use ATK >=26.8.8 and PyTorch. From this directory:

```bash
atk case -f chunk_kda_bwd_finalize.yaml -p gen_chunk_kda_bwd_finalize.py -dt 1 -en 0 -s 20261030
# Actual output: result/chunk_kda_bwd_finalize/json/all_chunk_kda_bwd_finalize.json
python scripts/check_shapes.py
atk node -b cpu task -c result/chunk_kda_bwd_finalize/json/all_chunk_kda_bwd_finalize.json \
  -p executor_chunk_kda_bwd_finalize.py -s 0 -e 1 -tk accuracy
```

The first command is a small generator dry-run, not formal acceptance. To expand
candidates use `bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_finalize -scope=gen_cases`
from the repository root, after preparing the environment described in the parent
README. The plugin imports NPU bindings lazily, so CPU use needs no NPU package.

No `_perf.json` is invented: user model cases were not provided. No `_mss.json`,
actual TilingKey selection proof, NPU execution, three-seed numerical golden
validation or formal acceptance is claimed. Before Phase B accuracy acceptance,
replace the zero placeholder with an independently validated numerical golden,
calibrate the value range, and complete the three acceptance packages and mappings
required by `tests/atk/README.md`.

## Phase A validation record (2026-10-30)

ATK 26.9.8 / CPU PyTorch 2.7.1 parsed the YAML and registered the generator.
A `-dt 1 -en 0` dry-run completed. Candidate generation used YAML
`dtype_numbers=96`, `extra_numbers=0`, generator seed 20261030.
`python scripts/archive_cases.py` copies only ATK-generated cases, removes
identical input configurations and sets `default_seed=20261030`; the checked-in
`atk_chunk_kda_bwd_finalize.json` has 1 retained candidate. This single-seed candidate archive
is not the three-seed formal acceptance baseline.

Independent `scripts/check_shapes.py` passed 160 contract examples.
Every archived candidate was instantiated through the CPU executor initialization,
and its output count/order/shape/dtype and metadata were checked against the
source ABI. All public layouts in this matrix, dense/packed modes and the declared
T=1/15/16/17/63/64/65/127/128/129 candidates are represented. The CPU zero
placeholder remains unsuitable for numerical acceptance.

Reproduce the candidate archive after reviewing the generated JSON:

```bash
atk case -f chunk_kda_bwd_finalize.yaml -p gen_chunk_kda_bwd_finalize.py -s 20261030
python scripts/archive_cases.py
```

Both beta dtypes and all eight `(dense/packed, rstd absent/present, beta dtype)`
combinations are represented. The source A5 tiling keys 1..4 each have candidates;
actual hardware key selection has not been checked.

ATK CPU CLI smoke (`atk node -b cpu task`, JSON slice `[0,1)`) completed
with `success 1, failed 0`. This verifies dataset creation and plugin execution;
no NPU comparison or numerical accuracy conclusion was made.

This project uses the `api_type` Python executor for CPU/NPU dispatch. It does
not implement a direct `pyaclnn` executor: test-control attrs and canonical host
metadata must be processed by `executor_chunk_kda_bwd_finalize.py` before dispatch.

## Issue #1156 定向复检（2026-11-12，A2）

本记录使用用户提供的外部 `fail.json` 和真实 FP32/FP64 双标杆 executor，
没有使用上面的 Phase A 零输出 scaffold。该结果仅覆盖 issue 指定用例及定向回归，
不表示本目录的全量正式验收已经完成。

基线为 main 每日包 `0.dev20261111+main.4fc94af`，修复基于
`4fc94af96fc181274cdfc5c1760a2bb0dc9d76f2`。硬件 Ascend910B3，CANN 9.2.0，
ATK 26.9.24，CT 0.11.1。executor SHA256：
`cd1b1726110b011f95d2230fbda8d12e6885403d068ec009442e52be22fe8e54`。

| case | seed | 原始 dA_log 绝对误差 | 修复后绝对误差 | 最终复检 |
|---|---:|---:|---:|---|
| 4547 | 20265560 | 1.59368e-5 | 9.76026e-7 | 100 次，七输出 CT PASS |
| 6404 | 20267417 | 5.96195e-5 | 2.83122e-7 | 100 次，七输出 CT PASS |
| 9110 | 20270123 | 1.03936e-6 | 1.11759e-8 | 100 次，七输出 CT PASS |

每轮实际调用修复后的 NPU 算子；不变的 CPU FP32/FP64 标杆可以缓存。
每条真实 ATK JSON 含 100 次记录，每个输出有 100 个有效样本，再执行
`ct dual analyze <ATK_JSON>`。三条用例的 21 项统计指标分别全部 PASS，
CT 退出码均为 0，ATK 精度失败也为 0。没有修改输入值域、seed 或比较阈值。

复现命令的结构为：

```bash
python -m atk node --port "$PORT" --backend pyaclnn --devices 0 -o "$OUT/results" \
  node --port "$PORT" --backend cpu task -c "$CASE_JSON" --task accuracy \
  -p "$EXECUTOR_DIR" -cp -mt 1 --loop_nums 100 --save_report_json -to 600
ct dual analyze "$ATK_JSON"
```

两节点使用同一 PORT；每条任务使用独立 PORT、可见设备、输出目录。运行前在独立开发环境
配置本轮 OPP 和 ATK_CUSTOM_OPP_PATH，核对已安装 libcust_opapi.so 及六个 kernel .o
与本轮构建一致。工程及日志根目录为
`/home/chenlinxin/wys/issue1156-20261111/fix`，各 case 的完整记录位于
`ct100-<case>/results`、`ct100-<case>/ct-analyze.log`、`ct100-<case>/summary.json`。

性能口径：同一物理卡 0、相同输入、20 次预热、50 次采样，取 msprof 目标 kernel
`Task Duration(us)` 中位数；两版均为一个 Cube block。

| case | 基线 / μs | 修复 / μs | 耗时变化 |
|---|---:|---:|---:|
| 4547 | 34651.93 | 35250.29 | +1.73% |
| 6404 | 8684.85 | 8785.29 | +1.16% |
| 9110 | 709.57 | 712.13 | +0.36% |

精度复检目标达到；零性能回退目标未达到。保留双 head 调度、原有核数和流水，仍存在上述
小幅代价。物理卡 1 上另测 B=1/T=1024/BSND/norm=False：H=3 耗时增加 2.11%，H=4
增加 4.17%，也不能写成零回退。

八组定向回归覆盖 H=1/2/3/4、dense/varlen、空序列、奇数 head、norm、三种 beta mode
及三种输出布局。七组七输出通过 CT 单次 L1；另一组的 dq/dv 与原 main 包同样存在
BF16 小误差，最大绝对误差分别为 1.90735e-6、3.81470e-6，没有新增失败项。
多核调度、A3 设备及完整上界/异常输入矩阵未作为本次通过结论。
