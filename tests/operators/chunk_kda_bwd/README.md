# chunk_kda_bwd tests

The accuracy regression covers the Ascend 950 dense fused-Gate path. Cases are
defined in `tests/op_cases/chunk_kda_bwd.json`.

Run all three cases on an Ascend 950 system with the operator installed:

```bash
python -m pytest -q tests/operators/chunk_kda_bwd/accuracy/test_chunk_kda_bwd.py
```

The test skips only when the current device is not Ascend 950. The executed
set includes the full `B=1, H=96, S=4096, K=V=128` regression.
