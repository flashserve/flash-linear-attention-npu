# chunk_kda_bwd_prepare GVA development validation

Active scope: accuracy exactly 200 cases, one fixed seed per case; determinism
20 cases; memory sanitizer the same 20 cases; performance 2 supplied cases.
The user explicitly replaced the design's three-seed suggestion with one seed.
Original supplied reference and cases remain intact at /home/yqw/atk/prepare.

R=2..16, HK=1/3, HV=HK*R, four layouts, fixed/variable lengths and both state
orientations; maximum accuracy shape H=48, T=257. MSS20 IDs are in mss20.json.
FP32 Bench and FP64 Golden use the user's formula in the single reference.py.
The input distribution and double-benchmark thresholds are unchanged.

Generate/check: python gen_chunk_kda_bwd_prepare_a3.py --kind full --seed-count 1
(or --check). Native ATK generation uses chunk_kda_bwd_prepare_a3.yaml.
For double benchmark, NPU DUT is node 1 (Actual), CPU Bench is node 2 (Benchmark),
with --bm_device cpu providing automatic FP64 Golden. The comparison report column carries the second node name; check Actual/Benchmark mapping and precision rows
and all three output comparison records; an ATK exit code alone is insufficient.

Current hardware: A2 910B3, CANN 9.3.0. The _a3 suffix is inherited asset naming;
no A3/A5 hardware acceptance is claimed. Validation completed on 2026-11-24: accuracy 200/200, DC20/20, MSS20/20.

Final validation: accuracy 200/200 (one seed), determinism 20/20, memcheck 20/20; extra peak 19494400 bytes.
msprof medians for the design shapes: M1 2473.7195 us; M2 authorized 16x1024 probe 3610.0920 us; 5 rounds of warmup10 + samples20 each.
MSS retains 690 FFTS_BASE_ADDR register warnings; no memory errors were reported. See validation_report.md for limits and raw report paths.

2026-11-25 性能纠正：M1 2473.7195us，M2用户授权16×1024探针3610.0920us（中位，各100样本）。正式性能JSON已改为HK16/HV128的M1与M2探针；旧H96结果仅作历史记录。见design_model_performance_report.md。

Model-shape accuracy: M1 and M2 authorized probe both pass all three outputs using the performance seeds and attributes. See model_accuracy_report.md.

PR preparation (2026-11-25): ported onto upstream main fc0bc08e, retaining upstream A6 support and structured diagnostics. Host ratio/key test and 200/20/2 manifest checks passed after port. Full NPU rebuild/regression has not been repeated on the ported tree; device results above belong to the original isolated implementation based on c4ca9684.
