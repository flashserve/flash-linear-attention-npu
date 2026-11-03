# chunk_kda_bwd_intra ATK

This is a **Phase A ATK scaffold** modeled on `../chunk_bwd_dqkwg`.
The CPU function in `scripts/chunk_kda_bwd_intra_cpu.py` intentionally returns **all zeros**.
Its output count, order, shape and dtype follow the ACLNN and stable-torch ABI;
it is **not a numerical golden**. CPU execution checks plumbing only. Do not use
NPU-versus-CPU results as accuracy acceptance, calibration or performance evidence.

## Inputs and constraints

q/k are contiguous BF16. beta is BF16 or FP32. gk, dAqk, dAkk,
dq, dk, db, dg and all outputs are FP32. C=64 and safe_gate=true.

| Mode/layout | q/k/gk/dq/dk/dg | beta/db | dAqk/dAkk | K |
| --- | --- | --- | --- | --- |
| dense BNSD | [B,H,T,K] | [B,H,T] | [B,H,T,64] | 64,128,256 |
| dense BSND | [B,T,H,K] | [B,T,H] | [B,T,H,64] | 64,128,256 |
| packed TND | [T,H,128] | [T,H] | [T,H,64] | 128 |
| packed BSND | [1,T,H,128] | [1,T,H] | [1,T,H,64] | 128 |

Ordered outputs are `(dq_out,dk_out,db_out,dg_out)`, each FP32 and exactly
the shape of its corresponding input gradient. Output layout equals input layout.
For TND the generator enables packed mode; packed BNSD candidates are corrected
to BSND and packed K is corrected to 128.

| Constraint | Source under fla/ops/ascendc/kda/chunk_kda_bwd_intra | Mapping |
| --- | --- | --- |
| Dtypes, contiguity, C=64, safe_gate=true | op_host/op_api/aclnn_chunk_kda_bwd_intra.cpp::Check | YAML dtypes/fixed attrs |
| Layout/mode/rank rules, positive B/H/T, dense K / packed K | same Check; op_host/chunk_kda_bwd_intra_tiling_processor.h::CheckSpec | generator layout and K repair |
| All vector shapes equal q; scalar beta/db; dA width64 | Check / CheckSpec | generator dependent shapes |
| Packed metadata 2..65 cu entries, 0..T, nondecreasing, canonical indices | ACLNN ValidateVarLenMetadata | executor fixed four-entry cu; canonical indices |
| Output shapes/dtypes equal corresponding gradients | ACLNN Check; torch_custom/fla_npu/csrc/src/stable_chunk_kda_bwd_intra.cpp | CPU output contract |
| No dense metadata; packed B=1 | Check / CheckSpec | generator B repair and executor metadata |

The host registers arch22 and arch35 paths. TilingKey encodes dense K=64/128/256
or packed K=128, safe-gate, mode and beta dtype. This scaffold does not assert
which key a device selected. Workspace uses uint64 calculations; no int32
size/offset assignment was found in the inspected host path. Full address-space
overflow coverage is outside this bounded Phase A matrix.


The YAML declares explicit tensor inputs and discrete shape/attribute candidates.
`gen_chunk_kda_bwd_intra.py` only repairs cross-input relationships and required fixed dimensions;
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
atk case -f chunk_kda_bwd_intra.yaml -p gen_chunk_kda_bwd_intra.py -dt 1 -en 0 -s 20261030
# Actual output: result/chunk_kda_bwd_intra/json/all_chunk_kda_bwd_intra.json
python scripts/check_shapes.py
atk node -b cpu task -c result/chunk_kda_bwd_intra/json/all_chunk_kda_bwd_intra.json \
  -p executor_chunk_kda_bwd_intra.py -s 0 -e 1 -tk accuracy
```

The first command is a small generator dry-run, not formal acceptance. To expand
candidates use `bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -scope=gen_cases`
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
`atk_chunk_kda_bwd_intra.json` has 1 retained candidate. This single-seed candidate archive
is not the three-seed formal acceptance baseline.

Independent `scripts/check_shapes.py` passed 80 contract examples.
Every archived candidate was instantiated through the CPU executor initialization,
and its output count/order/shape/dtype and metadata were checked against the
source ABI. All public layouts in this matrix, dense/packed modes and the declared
T=1/15/16/17/63/64/65/127/128/129 candidates are represented. The CPU zero
placeholder remains unsuitable for numerical acceptance.

Reproduce the candidate archive after reviewing the generated JSON:

```bash
atk case -f chunk_kda_bwd_intra.yaml -p gen_chunk_kda_bwd_intra.py -s 20261030
python scripts/archive_cases.py
```

The raw 128-case generator output had eight identical input configurations
(repetition metric 6.25%, generator `result:false` with CLI exit 0); static
deduplication retained all 120 unique configurations. Both beta dtypes and all
eight `(dense/packed, K, beta dtype)` source branches remain represented.

ATK CPU CLI smoke (`atk node -b cpu task`, JSON slice `[0,1)`) completed
with `success 1, failed 0`. This verifies dataset creation and plugin execution;
no NPU comparison or numerical accuracy conclusion was made.

This project uses the `api_type` Python executor for CPU/NPU dispatch. It does
not implement a direct `pyaclnn` executor: test-control attrs and canonical host
metadata must be processed by `executor_chunk_kda_bwd_intra.py` before dispatch.
