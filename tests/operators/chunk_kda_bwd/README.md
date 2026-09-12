# chunk_kda_bwd tests

The accuracy regression compares one packed varlen invocation with independent
per-sequence invocations. Cases are defined in
`tests/op_cases/chunk_kda_bwd.json` and include the original issue #544 shape.

Build and install `chunk_kda_fwd,chunk_kda_bwd` for Ascend 910B, then run:

```bash
python -m pytest -q \
  tests/operators/chunk_kda_bwd/accuracy/test_chunk_kda_bwd_varlen.py
```

The NPU cases skip when the active device is not an Ascend 910B. The manifest
contract test remains runnable without an NPU.
