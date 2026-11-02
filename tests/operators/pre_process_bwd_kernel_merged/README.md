# CP Host Boundary UT

## Current Targeted Patch

For the current A2/A5 error-code-only acceptance, use the minimal entry below.
It runs 14 subcases: output pointer and propagation4; shape/value/missing input/
rank/dtype/descriptor6; actual native executor4. It does not run accuracy,
performance, a model matrix, or kernels.

```bash
source /data/Ascend/cann/cann-9.1.0/set_env.sh
source /home/s30080348/dfx-cp-596f808/venv/bin/activate
cd /home/s30080348/flash-linear-attention-npu
export ASCEND_RT_VISIBLE_DEVICES=1 DFX_DEVICE=0 DFX_BUILD_DIR="$PWD/build"
bash tests/operators/pre_process_bwd_kernel_merged/run_dfx_smoke.sh
```

Both platforms passed 14/14 with the same production source revision. A5 was
rebuilt on 2026-11-02 in `/home/s30080348/cp-dfx-a5-validation-20261102` using
CANN9.2.0; its `validation.exit` is0. `artifacts.log` verifies the installed API
library, both kernel objects and Python wrapper files against the fresh package.
`dfx-smoke.log`, `boundary-evidence.json` and `native-evidence.json` record the
14 scenarios. These logs are also archived in the A2 evidence directory under
`pr-aligned-results/a5/`.

The subsequent released-opbase review closed the redundant post-construction
executor null check as REMOVED, supported by v9.1.0/v9.2.0 source, actual A2
binary disassembly and4 passing native subcases. Evidence is under
`a2-a5-validation/opbase-source-closure/`. The frozen42-item ledger now has
29FIXED/3REMOVED/10BLOCKED; this does not close every allocator failure path.

For the A5 minimal entry:

```bash
source /home/s30080348/miniforge3/envs/fla-cann920/Ascend/cann-9.2.0/set_env.sh
source /home/s30080348/cp-dfx-a5-validation-20261102/venv/bin/activate
cd /home/s30080348/cp-dfx-a5-validation-20261102/source
export ASCEND_RT_VISIBLE_DEVICES=3 DFX_DEVICE=0 DFX_BUILD_DIR="$PWD/build"
bash tests/operators/pre_process_bwd_kernel_merged/run_dfx_smoke.sh
```

The remainder documents extended tests available for investigation; they are
not all required by this minimal error-reporting acceptance.

Following the reference PRs, current production code preserves the baseline
public execution-failure status561000 while logging the native callee rc and
retaining its first error. No new C++ catch/function-try-block is part of the
targeted patch; previous exception-injection results are historical only.

Current host UT contains **5 methods / 24 scenarios**:7 Contiguous parameters
with/without simulated first error(14), conversion(2), launcher with/without
segment(4), execution propagation(2), output pointer guards(2).
Native executor tests expect direct161001 and CP-public561000, with the same
single nativeEZ1001. This distinction is intentional public compatibility.

The production metadata helper checks missing input shape first(EZ0004), then
descriptor missing despite a present shape(EZ1008), including supplied g/gk.
`tests/atk/pre_process_bwd_kernel_merged/scripts/test_tiling_dfx.py` exercises
47 parameter/resource/metadata cases and checks exact native code/message.
The public-builder omission cases remain evidence of missing required inputs,
not proof of independent descriptor loss while a tensor is present.

`test_normal_tiling_compatibility.py` compares 64 original/current processor
configurations(2 dtypes x4 gates x4 lengths x2 dense/varlen), including block
dimension, workspace, processor-derived template key and all tiling bytes.
This is processor compatibility, not a full GE success-path comparison.
The public ATK regression separately exercises the production callback/key
selection in the current installed package.

Latest logs are under the A2 evidence directory `pr-aligned-results/`; earlier
counts and raw-rc return expectations below describe superseded test versions.
Run the normal comparison with the original source checkout:

```bash
export CP_BASELINE_REPO=/home/s30080348/dfx-cp-596f808/a2-a5-validation/baseline-source
python tests/operators/pre_process_bwd_kernel_merged/test_normal_tiling_compatibility.py -v
python tests/atk/pre_process_bwd_kernel_merged/scripts/test_tiling_dfx.py -v
```

These tests compile the current `pre_process_bwd_kernel_merged` production
sources with their real CMake target flags and installed CANN libraries.
They do not modify the installed operator or introduce production fault hooks.

## Run

On A2 (`179.183.2.2`):

```bash
source /data/Ascend/cann/cann-9.1.0/set_env.sh
source /home/s30080348/dfx-cp-596f808/venv/bin/activate
cd /home/s30080348/flash-linear-attention-npu
export ASCEND_RT_VISIBLE_DEVICES=6
export DFX_BUILD_DIR="$PWD/build" DFX_DEVICE=0
export CP_BOUNDARY_EVIDENCE=/home/s30080348/dfx-cp-596f808/a2-a5-validation/boundary-strict-evidence.json
python tests/operators/pre_process_bwd_kernel_merged/test_host_fault_contract.py -v
export CP_GE_EVIDENCE=/home/s30080348/dfx-cp-596f808/a2-a5-validation/ge-boundary-strict-evidence.json
python tests/operators/common/test_ge_tiling_context_mre.py -v
export CP_NATIVE_EVIDENCE=/home/s30080348/dfx-cp-596f808/a2-a5-validation/native-executor-evidence.json
python tests/operators/pre_process_bwd_kernel_merged/test_native_executor_contract.py -v
```

The latest A5 follow-up ran only the minimal entry above, including four native
executor scenarios and six processor scenarios. The extended GE omission
fixture and full host suite remain A2-only for the current patch. The previous
A5 `dfx-991-xyDKNtce/cp-validation/current` source and venv were removed; use the
new isolated directory above. The existing A5 main checkout was not modified.

Confirm the selected physical device is free before running. The existing
build must have `compile_commands.json`, `libcust_opapi.so`, and current CANN
libraries. If compile commands have not been exported, run this after any
active build has finished:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Historical Host Cases (Before Scope Reconciliation)

| Boundary | Cases | Exact expected behavior |
| --- | ---: | --- |
| Constructor bad_alloc/std/unknown | 3 | Valid descriptors reach the constructor once; statuses 561101/561000/561000; no lower calls or published outputs |
| q/k/w/d_o/dv/g/gk Contiguous | 14 | Each parameter fails at its own pointer; exact call order; status 161002 |
| ConvertToTensor | 2 | A valid aclIntArray reaches conversion once; no launcher call; status 161001 |
| Launcher, with/without segment | 4 | Real L0 is reached, launcher wrapper hit once; existing L2 null-result mapping returns 161001 |
| Execution rc | 2 | Injected 361001 is returned unchanged |
| Execution std/unknown | 2 | Status 561000, one exception-detail payload in the target EZ9999 output |
| workspaceSize/executor output pointers | 2 | Direct production guards return 161001/EZ0037 before executor construction |
| Total | 29 | Seven unittest methods, all injection counts and parameter sequences checked |

Lower failure cases are run twice: the double reports no formal error, or the
double reports one explicitly named `CP_UT_SIMULATED_LOWER` diagnostic through
the real CANN ErrorManager. The tests verify no new outer diagnostic in the
first case and preservation of exactly one EZ0037 in the second.

The constructor catches currently log only PLOG, so the no-report constructor
cases correctly observe an empty ErrorManager message. The test does not
invent a formal allocator error. Launcher status is not claimed to be publicly
preserved: current L0 returns null and L2 retains its historical 161001 mapping.

Both real L2 and L0 files are compiled into the test library. The library uses
`--wrap`, `-Bsymbolic-functions`, and `-z,defs`, plus symbol checks, to avoid
silently invoking a stale installed implementation. Input descriptors have
contiguous strides and live backing storage; output shape/dtype is valid.

## GE Cases

The MRE compiles the current tiling source and links native `libtiling_api.a`
and GE/CANN libraries. It uses the public `OpTilingContextBuilder`, with no
test-only platform method definitions or private ABI casts.

Four tests pass on the inspected A2/CANN9.1 and A5/CANN9.2 installations:

- Resource control: typed tilingData, attrs, platform are all non-null.
- Insufficient tiling capacity: typed tilingData is null; attrs/platform remain non-null.
- Null raw tilingData: typed tilingData is null; attrs/platform remain non-null.
- Null TilingContext: actual production callback returns GRAPH_FAILED/EZ1008.

The A2 follow-up additionally constructs five required-descriptor omission
cases with public `IOInstanceNum`: retain 0/1/2/3/4 required input instances,
omit remaining required and optional instances, and provide exactly that many
non-null tensors. This is a malformed operator invocation in a real,
well-formed builder context, not a null-Tensor or ABI-mutation probe.

The fixture asserts the actual input count, all preceding descriptors present,
the first missing descriptor null, and all tiling/attrs/platform resources
present before invoking the production callback. The q/k/w/d_o/dv cases each
return GRAPH_FAILED and one EZ0004 with the correct parameter.

A2 now has 7 GE unittest methods: 5 pass and 2 skip; the passing descriptor
method contains 5 subcases. That is **9 successful scenarios and 2 skipped
scenarios**, not 7/7 methods passed.

Two A2 methods remain skipped and are not counted as passed:

- Missing attrs: this public builder supplies non-null RuntimeAttrs even without appended attrs.
- Missing platform: this public builder rejects the context before the callback.

The earlier A5 result remains 4 passed/3 skipped; the new omission fixture has
not been run there. No crashing null-Tensor probe is run on either platform.

The context control uses builder-supported placeholder attributes only to
check resource structure. It is not a valid full operator success case because
the public builder has no double attribute overload for the operator's scale.
The tested failure callbacks stop before dereferencing those values.

## Historical A2 Native Executor Evidence

`test_native_executor_contract.py` compiles unchanged L2/L0 with no wrappers
or simulated reporting, then calls installed CANN9.1 `CommonOpExecutorRun`:

- Null executor, direct native call.
- Null executor, through the real CP execution entry.
- Null workspace with workspaceSize=1 and a valid native executor, direct call.
- The same workspace error through the real CP execution entry.

All four return161001 with exactly one native EZ1001, correct parameter text,
and no outer duplicate report. `dladdr` records the actual nnopbase owner and
the JSON records its SHA256. This proves first reporting for these two native
parameter-failure branches and representative CP propagation; it does not
cover every native runtime/device/allocator error.

The isolated opbase UT dependencies under the A2 task directory are available:
gtest1.14.0, mockcpp2.7-h2, Boost1.87.0. Functional same-executable mock/reset
and ABI-0 smoke tests pass. The mockcpp far-DSO replacement capability test
fails on AArch64 and was not applied to CANN. Those dependencies are optional
for the three scripts above, which use Python unittest and target CANN.

## Evidence Boundary

Passing these UTs proves the operator's response to specified test inputs and
test-double outcomes. It does not prove native Contiguous/ConvertToTensor/
launcher/allocator first reporting, full GE graph execution, or sanitizer
acceptance. Existing full-chain BLOCKED dispositions remain separate.

The authoritative result and corrected disposition counts are maintained in
the operator's `docs/error_code_return_status.md`. Earlier reports of three GE
resource omissions and untargeted lower-fault coverage are superseded by these
tests and their explicit pass/skip evidence.

GE attrs/platform null-resource fixtures are excluded from this CP error-code remediation by user instruction. Their historical skip evidence is retained for any future GE-specific work.
