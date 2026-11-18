# 验证记录

## 02 标杆运行

| 范围 | 标杆运行 | 计算语义 | 精度策略 | 证据 | 结论 |
|---|---|---|---|---|---|
| CPU FP64，固定 seed 419 的 ragged/GQA、BF16/FP32 checkpoint、padding、空请求和无效页 | PASS | PASS | PASS | `python test/test_reference.py` 6 项通过；与 PR 20485 固定提交中的 CPU 标杆逐元素一致；BF16 `atol=rtol=2^-6`，FP32 `atol=2^-16, rtol=2^-10`，matched ratio 0.999 | 通过 |
| A5 NPU，BF16/FP32 checkpoint，seed 419 ragged 与空请求 | PASS | PASS | PASS | NPU 单算子测试 3 项通过；输出最大绝对误差 `2.25e-4`，replay correction/key/decay 最大绝对误差分别为 `2.23e-7`、`3.06e-8`、`5.85e-8` | 通过 |
| CPU FP64 PR 20485 标杆，`BS=32/64/72`、`HV=32/48/128`、`TPB=1..16`、L2Norm on/off | PASS | PASS | PASS | 288/288 通过；L2Norm on/off 各 144 项；输出最大绝对误差 `0.2481`，replay correction/key/decay 最大绝对误差分别为 `1.96e-4`、`6.74e-8`、`1.62e-7` | 通过 |
| A5 性能，`BS=32/64/72`、`HV=32/48/128`、`TPB=1/2/4/8` | PASS | PASS | PASS | 36/36 完成采集；35/36 优于 RecurrentGDN；几何平均耗时比为 `0.477`；仅 `BS32/TPB1/HV32` 略慢，耗时比 `1.074` | 通过 |
