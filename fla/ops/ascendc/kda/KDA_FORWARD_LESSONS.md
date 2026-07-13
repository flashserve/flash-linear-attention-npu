# KDA Forward Development Lessons

This document is intentionally written as a handoff note for a future engineer or AI agent. Read it before changing KDA forward. It records the design traps already encountered so the same mistakes are not repeated.

## 1. Start Here

If you enter this project from a fresh conversation, assume the current KDA forward PR has these constraints:

- Focus on forward only.
- `K=128, V=128, chunk_size=64` dense BSND and TND have been validated.
- BNSD/NTD are the direct performance layouts when upstream causal conv has already converted memory order. Do not route these layouts back through `KdaLayoutSwap12`.
- `V=256` is not the same template. Do not "just increase V" in the `V=128` path.
- High `K/V` non-chunk-aligned varlen still needs a real partial-chunk design.
- `final_state` is `float32`, not the dtype of `q`.
- Target matrix work must use Catlass cube. Do not reintroduce scalar dot-product loops for target shapes.

## 2. Do Not Repeat These Mistakes

### Mistake 1: Treating `exp2` as a scalar helper

Bad pattern:

```cpp
for (...) {
    float gate = Exp2Scalar(x);
}
```

Why it is wrong:

- It turns a vector operation into scalar pipe work.
- It blocks the intended AIV SIMD style.
- It hides performance problems until large shapes are tested.

Correct direction:

- Use vector `Exp` on `x * ln2`.
- Keep `gk` rows in UB as vector tensors.
- Use `DataCopy`/`DataCopyPad` to move rows, then apply vector math.

### Mistake 2: Computing `q @ k^T` with per-element loops

Bad pattern:

```cpp
for i:
  for j:
    for d:
      acc += q[i, d] * k[j, d] * gate[i, j, d]
```

Why it is wrong:

- Cube is much faster than vector/scalar for matrix products.
- The whole point of the KDA forward split is to put `Aqk/Akk`, post-WU, and output matmuls on cube.

Correct direction:

- Prepare `qg`, `kg`, and `w` rows on AIV.
- Use Catlass cube GEMM for full target chunks.
- Keep scalar fallback only for non-target correctness paths, never for target performance.

### Mistake 3: Using `GetValue` and `SetValue`

Do not use `GetValue`/`SetValue` for hot paths. They produce scalarized code and usually indicate the kernel has fallen out of the AscendC SIMD programming model.

Correct direction:

- Move full rows or tiles with `DataCopy`/`DataCopyPad`.
- Process vector chunks with `Mul`, `Muls`, `Add`, `Sub`, `Cast`, `Duplicate`, `Brcb`, `Exp`.
- Use row-level helper functions that still compile to vector operations.

### Mistake 4: Forgetting `final_state` dtype

fla-org and GDN forward-h semantics keep state in `float32`. KDA must follow this:

```text
initial_state: optional float32
final_state:  float32
h:            q dtype
```

Symptoms when this is wrong:

- PyTorch reference comparison fails only on dtype.
- L0 op registration can reject the call with executor/nullptr because `final_state` was registered as q dtype.

Correct files to check:

- `chunk_kda_fwd_def.cpp`: `initial_state` and `final_state` must be `DT_FLOAT`.
- `FLANpuOpApi.cpp`: allocate `final_state_work` as `at::kFloat`.
- `chunk_kda_fwd.cpp`: `initialState_` and `finalState_` should be `GlobalTensor<float>`.

### Mistake 5: Assuming op_def dtype lists are harmless

The op_def dtype list is not documentation only. It participates in graph/operator matching.

If the kernel writes `float32 final_state` but op_def says `final_state` follows `q` dtype, the call can fail before useful kernel diagnostics.

Checklist:

- Every required output dtype in op_def must match the actual tensor allocated by L2.
- Optional inputs with fixed semantics, such as `initial_state`, should not reuse broad `q/k/v` dtype lists.
- If L2 casts `gk/beta` before L0, L0 op_def only needs to match the post-cast dtype.

### Mistake 6: Pairing AIC/AIV through task count instead of core count

Cross-core flag protocols require both sides to participate with matching counts. Launching only `taskNum` blocks for half/bfloat16 `K>=16` can leave some AIC/AIV pairs unmatched.

Symptom:

- Small dense shapes may pass.
- Varlen or partial chunk shapes timeout in `torch.npu.synchronize`.

Correct direction:

- For half/bfloat16 target cube paths, launch full AICore count.
- Derive logical `coreIdx` from `GetBlockIdx() / GetSubBlockNum()` on AIV.
- Make every `CrossCoreSetFlagWithReverse` have a matching wait on the paired side.

### Mistake 7: Treating partial chunks like full chunks

This is the biggest current trap.

Full `BT=64` paths can use cube solve and fixed scratch slots. A partial chunk, for example a sequence segment of length 40, has different requirements:

- Some AIV subblocks may have no valid rows.
- AIC may still expect a ready flag for a chunk that AIV skipped.
- Catlass tiles may read dirty rows if full tile cleanup is not designed.
- `chunkIdx` is a scratch slot index, while `start` is a token offset. Do not mix the two.

Do not claim high `K/V` varlen support until this is fixed.

Correct direction for a future fix:

1. Separate full chunk and partial chunk scheduling.
2. For partial chunks, either:
   - use a vector-only correctness path with no cube handshake, or
   - pad/clean UB/GM scratch to full `BT` and still balance all AIC/AIV flags.
3. Make `ResolveFlatChunk` return both:
   - `globalTokenStart` for q/k/v/gk/beta/o reads and writes.
   - `chunkSlot` for h/scratch/final state slots.
4. Add timeout tests for `cu_seqlens` where at least one segment length is not divisible by `chunk_size`.

### Mistake 8: Retrofitting `V=256` into the `V=128` template

`V=256` doubles the row payload and changes whether `w/u/o/h_next` can stay in UB. A template that is acceptable for `V=128` can become scalar-bound, MTE-bound, or simply exceed practical UB staging when stretched.

Correct direction:

- Build a separate `V=256` template.
- Recompute UB arena partitioning.
- Decide which rows are resident and which are streamed.
- Recheck Catlass tile shapes and post-WU/output cube paths.

### Mistake 9: Overusing `SyncAll`

`SyncAll` may hide lifecycle bugs while destroying performance. It is not a substitute for a clear producer-consumer protocol.

Correct direction:

- Prefer independent output regions.
- For AIC/AIV cooperation, use named cross-core flags with reverse/credit behavior.
- Keep flag ids centralized and avoid magic-number reuse.
- Use kernel-stage separation when a dependency is naturally stage-wide.

### Mistake 10: Misreading `ct viz` slowness

`ct viz` can be slow for large tensors, so use sampling:

```bash
ct viz real.npy expect.npy --out_dir out --name name --diff_thd 0.03 --spatial -sc 4096
```

But if the program hangs before `.npy` files are produced, the problem is not `ct viz`. It is usually:

- kernel timeout,
- deadlocked cross-core flags,
- out-of-bounds/invalid memory access,
- or an op launch/runtime error.

Always distinguish:

```text
Python produced .npy, ct viz is slow  -> use --sc
torch.npu.synchronize times out      -> debug kernel
```

### Mistake 11: Reusing a UB tile without closing the VEC-to-MTE2 lifecycle

This caused a real fixed-input, non-bitwise-stable bug in the `hasGk` path.

Bad pattern:

```cpp
// hUpdateUbTensor is used as a temporary broadcast buffer.
Broadcast(hUpdateUbTensor, gkScale, ...);
Mul(calcUbTensor, calcUbTensor, hUpdateUbTensor, ...);

// Later the same hUpdateUbTensor is reused as the MTE2 destination.
WaitFlag<V_MTE2>(oldEvent);
DataCopy(hUpdateUbTensor, hUpdateInput, ...);
Add(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, ...);
```

Why it is wrong:

- The old event only says the previous owner released the tile.
- It does not say the current VEC writes to `hUpdateUbTensor` are finished before MTE2 overwrites it.
- With fixed inputs, outputs can change between runs because `Add` may see the stale broadcast value instead of the copied `hUpdateInput`.

Correct direction:

1. Before using the tile as a VEC temporary, consume the free event for that tile.
2. After the VEC temporary use finishes, set the matching `V_MTE2` event.
3. Only then let the later MTE2 `DataCopy` wait on that event and overwrite the tile.

For `chunk_gated_delta_rule_fwd_h`, the `hasGk` HUpdate path must keep:

```cpp
WaitFlag<V_MTE2>(EVENT_ID0 + pingpongFlag);
...
Mul(calcUbTensor, calcUbTensor, hUpdateUbTensor, ...);
PipeBarrier<PIPE_V>();
SetFlag<V_MTE2>(EVENT_ID0 + pingpongFlag);
...
WaitFlag<V_MTE2>(EVENT_ID0 + pingpongFlag);
DataCopy(hUpdateUbTensor, hUpdateInputThisSubBlock, ...);
```

### Mistake 12: Running layout swap after upstream conv already converted layout

KDA's kernel layout is BNSD. If upstream causal conv already produces BNSD/NTD-contiguous tensors, adding `KdaLayoutSwap12` inside KDA is pure overhead.

Symptoms:

- `msopprof` shows `KdaLayoutSwap12` dominating the target cases.
- The short-head long-sequence case spends more time moving layout than computing KDA.

Correct direction:

- Treat BSND/TND as compatibility layouts.
- Treat BNSD/NTD as the hot path.
- For NTD, reshape to `[1, H, T, D]`/`[1, HV, T, D]` views in L2 and call the BNSD kernel directly.
- `npu_kda_gate_cumsum` must preserve the input layout too; BNSD input returns BNSD `gk`, NTD input returns NTD `gk`.

### Mistake 13: Using user outputs as split-path raw intermediates

The split KDA forward path computes raw `o` and raw `Aqk`, then applies scale with elewise L0 ops. It also feeds `w/u/kg/h/v_new` between custom L0 ops.

Bad pattern:

```cpp
// For BNSD direct input, oBnsd/aqkBnst point to user outputs.
ChunkKdaFwd(..., oBnsd, aqkBnst, wBnsd, uBnsd, ...);
auto aqkScaled = Muls(aqkBnst, scale, executor);
auto hResult = ChunkGatedDeltaRuleFwdH(kgBnsd, wBnsd, uBnsd, ...);
```

Why it is wrong:

- Executor-owned temporaries and user outputs do not behave the same in producer-consumer L0 chains.
- Using user output tensors as raw intermediate inputs can trigger elewise tiling shape errors or invalid workspace inference, including impossible allocations.

Correct direction:

- In split forward, allocate executor-owned BNSD temporaries for `o/Aqk/Akk/w/u/qg/kg/v_new/h`.
- Feed these temporaries through `ChunkKdaFwd`, elewise scale, and `ChunkGatedDeltaRuleFwdH`.
- At the end, `ViewCopy` the temporary tensors to user outputs in the same BNSD/NTD layout. This is not a layout swap.
- Keep `KdaLayoutSwap12` only for BSND/TND compatibility paths.

## 3. KDA Forward Debug Checklist

Before running long shapes:

1. Build only the necessary operators:
   - `chunk_kda_fwd`
   - `chunk_gated_delta_rule_fwd_h`
   - `kda_layout_swap12`
   - `kda_gate_cumsum`
2. Source the custom package environment.
3. Confirm Python imports the installed `fla_npu` extension, not the source package without a compiled `.so`.
4. Start with dense `T=64` and `T=128`.
5. Compare `o` and `final_state` against `tests/reference/chunk_kda_reference.py`.
6. Run `ct viz --sc` after outputs exist.
7. Only then move to TND, varlen, and long sequence performance.

## 4. Shape Ladder

Use this progression to avoid wasting time:

```text
1. Dense BNSD, fp16, K=128, V=128, T=64
2. Dense BNSD, fp16, K=128, V=128, T=128
3. Dense NTD,  fp16, K=128, V=128, T=128
4. Dense BNSD, fp16, K=128, V=128, T=512
5. Varlen aligned cu_seqlens, all segment lengths divisible by chunk_size
6. Varlen non-aligned cu_seqlens
7. Long sequence performance
8. V=256 only after a separate template exists
```

If step 6 times out, do not keep trying larger varlen shapes. Fix partial chunk scheduling first.

## 5. What A Future Partial-Chunk Fix Must Prove

A correct high `K/V` varlen implementation must prove:

- Every AIC wait has a corresponding AIV set.
- Every AIV wait has a corresponding AIC set.
- Subblocks with no valid rows still participate in the flag protocol if their paired AIC expects them.
- Dirty rows outside `curT` are either never read or explicitly padded to neutral values.
- `Aqk/Akk` rows outside `curT` do not affect solve/output.
- Scratch slots are indexed by compact chunk slot, not by absolute token start.
- `h` output remains semantically ordered as `[chunk, HV, K, V]` for TND and `[B, chunk, HV, K, V]` for BSND public outputs.

## 6. Public Reporting Rules

When writing PR descriptions or test reports:

- Do not include server names, IPs, usernames, absolute paths, temporary directories, package install directories, or log paths.
- Report only what was tested and whether it passed.
- Say "`ct viz --sc 4096` found no sampled point over threshold" instead of publishing artifact paths.
- If a scenario times out, say the scenario and failure type, not where it was run.

## 7. Minimal Known-Good Claims

At the time this note was written, these claims were validated:

- Custom package build passed for KDA forward and required helper/GDN operators.
- Dense BSND `K=128, V=128, chunk_size=64` passed reference comparison.
- Dense TND `K=128, V=128, chunk_size=64` passed reference comparison.
- Dense BNSD and NTD direct layouts passed reference comparison.
- Sampled `ct viz` with `-sc 4096` showed no points over `0.03` threshold for `o` and `final_state`.
- Fixed-input repeated runs are bitwise stable for:
  - standalone `chunk_gated_delta_rule_fwd_h` `hasGk`,
  - KDA special-value dense case,
  - KDA random dense case with `H_V=32`.
- Sampled long-shape reference checks passed for:
  - BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64`,
  - BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64`,
  - NTD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64`.
- `msopprof` BasicInfo measured no `KdaLayoutSwap12` on the BNSD target path.

These claims must not be made without more work:

- `V=256` template support.
- Optimized high `K/V` non-aligned varlen support.
- Backward KDA support.
- Sanitizer-clean memory/race/sync result for the new high `K/V` varlen path.
