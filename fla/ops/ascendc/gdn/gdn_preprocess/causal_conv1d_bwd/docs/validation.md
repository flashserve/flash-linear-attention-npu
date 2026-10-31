# CausalConv1dBwd Host validation repair

## Scope and design

Base: `A6-dev` at `aece2620062e1fc159fdadd083599eb4a5200956`.
The existing contract is documented in `aclnnCausalConv1dBwd.md`: floating
inputs have identical dtype and activation is one of 0, 1, 2. The ATK CPU
reference also rejects other activation values. No new mathematical or ABI
contract is introduced.

- Check that x uses FLOAT, FLOAT16 or BFLOAT16.
- Check required weight/dy descriptors and compare their dtype with x.
- Check y/initial_state/dht only when their optional descriptors exist.
- Read activation as its declared int64 type, validate before narrowing to
  uint32, and preserve default 0 when the attribute is omitted.
- Return GRAPH_FAILED on invalid inputs before kernel scheduling. Keep the
  existing queryStartLoc INT64 check separate.
- No change to kernel stages, tiling keys, workspace, layouts or schema.
  This repair does not add A6 registration/kernel support from PR #1000.

## Local evidence

The actual modified validation blocks were extracted from the source and
compiled with MSVC C++17 against mocked CANN descriptors, attributes and
logging. All 157 host cases passed: three supported dtypes, all optional
descriptor presence combinations, each mismatching floating input, missing
required descriptors, activation defaults and values 0/1/2, -1, 3, 2^32,
2^32+1, 2^32+2, and INT64_MIN/MAX. Invalid cases checked failure and diagnostic
parameter/value text. This isolates the new guards; it does not execute the
full tiling function, CANN logging, or the existing y-required check.

## Pending device acceptance and known limitations

- Linux/CANN builds, ATK, NPU negative tests and Example/ST have not run;
  the development host is Windows without CANN/NPU.
- Validate legal FP32/FP16/BF16, optional absence/presence, mismatching each
  input dtype, and the full activation boundary set through the real entry.
- Diagnostics use the existing OP_LOGE convention. Parameter and actual /
  expected values are logged, but structured parameter-level reporting and
  aclGetRecentErrMsg propagation remain unverified (DFX-MESSAGE limitation).
- A6 device acceptance additionally requires the separate operator adaptation.
- Formal test acceptance is incomplete. Retain this development record until
  real build/device results can be archived in the ATK README.
