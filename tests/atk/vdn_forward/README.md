# VDN forward tests

The public device entry is `fla_npu.ops.vdn.vdn_forward`. Run against an installed A2 package in a shell with CANN initialized. The 910B optimized measurements require the VDN native extension. The benchmark reports which implementation it loaded.

## Oracle and scope

`reference.py` solves the symmetric positive definite system directly on CPU for each head, after computing FP32 Grams. For BF16 it rounds the weighted values, value Gram, state used for readout and output at the feature boundary. It computes each window's two decay products directly. The device code uses factorized updates and log-prefix differences. CPU tests include a three-frame scalar example with an independently computed middle-frame answer, a full-window clip with zero and nonzero text seeds, and strided head slices.

Device tests cover both update rules, all four SiLU/L2 settings, missing/present text state, 1/5/9 frames, one and multiple heads, strided features, padding across an interior tile, and FP32 depthwise spatial/temporal convolution with a temporal halo crossing a tile boundary. Invalid windows, unknown update rules and invalid frame tiles must raise. Separate tests check the FP32 inverse residual for the identity, rank-one Gram with top eigenvalue 405, and a rotated dense spectrum. Full-output limits are relative L2 below 2e-4 (FP32) or 0.007 (BF16) and maximum absolute error normalized by the reference peak below 4e-4 or 0.04. The inverse test uses normalized Frobenius residual below 2e-5.

```bash
python -m unittest discover -s tests/atk/vdn_forward -p test_reference.py -v
VDN_TEST_DEVICE=0 python -m unittest discover -s tests/atk/vdn_forward -v
python tests/atk/vdn_forward/bench_forward.py --device 0 --heads 7 14 28 56
python tests/atk/vdn_forward/bench_forward.py --device 0 --rule vdn_solve
```

Each benchmark call validates all output elements against the CPU oracle before eight warm-ups and 31 synchronized samples. The shape is 87 frames, 405 tokens, width 128 and BF16 features by default. Peak allocated and reserved memory are reported in MiB. Incremental peak subtracts live inputs and other persistent allocations before the measured loop. Reserved memory also includes prior caching.

## ATK

Use the repository's [ATK setup and commands](../README.md). The JSON cases use two solve rules at each of four configurations. IDs 0–1 use a single FP32 frame and token, 2–3 use a five-frame tail, 4–5 use nine BF16 frames without text, and 6–7 use the 87-frame, seven-head BF16 configuration. The performance subset contains 6–7. MSS includes 0–1 and 4–5. The generator varies seeds within the reviewed matrix.

```bash
bash tests/atk/run_test_cpu.sh -op=vdn_forward -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=vdn_forward -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=vdn_forward -npu_device_id=0 -scope=mssanitizer
```

ATK CPU and device nodes generate identical seeded CPU inputs, and only the device node transfers them to the NPU. Use the standalone benchmark above to time the public forward with input generation and transfers outside the measurement.


## A2 verification

On Ascend 910B2C with CANN 8.5.0, an installed wheel built from this change and the 26.10.0 A2 base OPP passed all eight unittest methods, including the three native stage checks. With 87 frames, 405 tokens and width 128, the installed native Newton path produced these synchronized measurements after checking the full output. Each row uses 31 calls; times are milliseconds and incremental peak allocated memory is MiB.

| Heads | Frame tile | P10 | P50 | P90 | Incremental peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 64 | 4.027 | 4.042 | 4.056 | 383.97 |
| 14 | 32 | 7.664 | 7.681 | 7.727 | 488.44 |
| 28 | 16 | 14.004 | 14.049 | 14.110 | 640.35 |
| 56 | 8 | 27.051 | 27.098 | 27.160 | 1225.41 |

The seven-head Cholesky path on the same input passed the same accuracy gate. The other supported installation and hardware combinations require separate validation.
