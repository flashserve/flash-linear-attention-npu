# chunk_kda_bwd_prepare ATK

This is a **Phase A ATK scaffold** modeled on `../chunk_bwd_dqkwg`.
The CPU function in `scripts/chunk_kda_bwd_prepare_cpu.py` intentionally returns **all zeros**.
Its output count, order, shape and dtype follow the ACLNN and stable-torch ABI;
it is **not a numerical golden**. CPU execution checks plumbing only. Do not use
NPU-versus-CPU results as accuracy acceptance, calibration or performance evidence.

## Inputs and constraints

All required inputs are contiguous BF16; C=64 and K=V=128.

| Public layout | aqk | v_new / d_o | h | Outputs |
| --- | --- | --- | --- | --- |
| BNSD | [B,H,T,64] | [B,H,T,128] | [B,Nc,H,128,128] | head-major rank 4 |
| BSND | [B,T,H,64] | [B,T,H,128] | [B,Nc,H,128,128] | head-major rank 4 |
| NTD | [H,T,64] | [H,T,128] | [Nc,H,128,128] | head-major rank 3 |
| TND | [T,H,64] | [T,H,128] | [Nc,H,128,128] | head-major rank 3 |

Ordered outputs: `d_aqk` FP32 with last dim 64; `dv` BF16 and `dq_raw` FP32
with last dim 128. Head-major output axes are `[B,H,T,D]` / `[H,T,D]` even
when inputs are token-major. Both state orientations are candidate attributes.
Dense Nc=ceil(T/64); packed Nc=sum(ceil(seq_len/64)), and packed rank-4 B=1.

| Constraint | Source under fla/ops/ascendc/kda/chunk_kda_bwd_prepare | Mapping |
| --- | --- | --- |
| BF16 inputs, contiguous, finite FP32 scale, C=64 | op_host/op_api/aclnn_chunk_kda_bwd_prepare.cpp::Check | YAML dtypes/attrs; ATK contiguous inputs |
| Layout/rank, positive B/H/T, matching feature shapes, fixed 64/128 widths | same Check; op_host/chunk_kda_bwd_prepare_validation.h::ParseTokenShape | generator token_shape and dependent shapes |
| NT-first h; dense/packed Nc; packed B=1 | Check; op_host/op_tiling/chunk_kda_bwd_prepare_tiling_processor.h::Process | generator state shape |
| Nondecreasing cu from 0 to T, canonical indices; no indices alone | ACLNN ValidateMetadata; L0 op_api/chunk_kda_bwd_prepare.cpp | executor canonical metadata |
| Head-major output shapes and FP32/BF16/FP32 order | ACLNN Check; torch_custom/fla_npu/csrc/src/stable_chunk_kda_bwd_prepare.cpp | CPU placeholder output contract |
| Overflow-safe work/shape multiplication, positive dimensions | tiling processor Process and ParseTokenShape | bounded YAML candidates; no oversized allocation cases |

A2/A3 implement all listed layout/mode combinations. **A5 currently supports
fixed BNSD and packed NTD only**; filter candidates accordingly before A5 use.
Source TilingKey is selected by `(inputLayout,isVariable)`; no device selection
has been observed. Arithmetic scan found explicit int64/uint64 guards and
size_t workspace multiplication; this bounded generation is not an overflow test.


The YAML declares explicit tensor inputs and discrete shape/attribute candidates.
`gen_chunk_kda_bwd_prepare.py` only repairs cross-input relationships and required fixed dimensions;
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
atk case -f chunk_kda_bwd_prepare.yaml -p gen_chunk_kda_bwd_prepare.py -dt 1 -en 0 -s 20261030
# Actual output: result/chunk_kda_bwd_prepare/json/all_chunk_kda_bwd_prepare.json
python scripts/check_shapes.py
atk node -b cpu task -c result/chunk_kda_bwd_prepare/json/all_chunk_kda_bwd_prepare.json \
  -p executor_chunk_kda_bwd_prepare.py -s 0 -e 1 -tk accuracy
```

The first command is a small generator dry-run, not formal acceptance. To expand
candidates use `bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_prepare -scope=gen_cases`
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
`dtype_numbers=64`, `extra_numbers=0`, generator seed 20261030.
`python scripts/archive_cases.py` copies only ATK-generated cases, removes
identical input configurations and sets `default_seed=20261030`; the checked-in
`atk_chunk_kda_bwd_prepare.json` has 1 retained candidate. This single-seed candidate archive
is not the three-seed formal acceptance baseline.

Independent `scripts/check_shapes.py` passed 80 contract examples.
Every archived candidate was instantiated through the CPU executor initialization,
and its output count/order/shape/dtype and metadata were checked against the
source ABI. All public layouts in this matrix, dense/packed modes and the declared
T=1/15/16/17/63/64/65/127/128/129 candidates are represented. The CPU zero
placeholder remains unsuitable for numerical acceptance.

Reproduce the candidate archive after reviewing the generated JSON:

```bash
atk case -f chunk_kda_bwd_prepare.yaml -p gen_chunk_kda_bwd_prepare.py -s 20261030
python scripts/archive_cases.py
```

ATK CPU CLI smoke (`atk node -b cpu task`, JSON slice `[0,1)`) completed
with `success 1, failed 0`. This verifies dataset creation and plugin execution;
no NPU comparison or numerical accuracy conclusion was made.

This project uses the `api_type` Python executor for CPU/NPU dispatch. It does
not implement a direct `pyaclnn` executor: test-control attrs and canonical host
metadata must be processed by `executor_chunk_kda_bwd_prepare.py` before dispatch.
