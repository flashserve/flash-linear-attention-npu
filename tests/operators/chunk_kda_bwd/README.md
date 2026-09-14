# chunk_kda_bwd tests

The accuracy regression compares one packed varlen invocation with independent
per-sequence invocations. Cases are defined in
`tests/op_cases/chunk_kda_bwd.json` and include the original issue #544 shape.
Ascend 950 cases use the same saved forward tensors for both backward paths,
initialize the native packed backward workspace to `0xFF`, and check the
8/24/40/56-token boundaries as well as the original issue shape.

Build and install `chunk_kda_fwd,chunk_kda_bwd` for Ascend 910B or Ascend 950,
then run:

```bash
python -m pytest -q \
  tests/operators/chunk_kda_bwd/accuracy/test_chunk_kda_bwd_varlen.py
```

Each NPU case skips on the other SoC. The Ascend 950 case checks that the
packed backward reaches a single native ACLNN invocation even though the
public A5 partial-tail wrapper normally splits it into dense calls; the test
does not change the production wrapper. The manifest contract test remains
runnable without an NPU.
