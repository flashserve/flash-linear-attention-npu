# 验证记录

## 02 标杆运行

| 范围 | 标杆运行 | 计算语义 | 精度策略 | 证据 | 结论 |
|---|---|---|---|---|---|
| CPU FP64，固定 seed 419 的 ragged/GQA、BF16/FP32 checkpoint、padding、空请求和无效页 | PASS | PASS | PASS | `python test/test_reference.py` 6 项通过；与 PR 20485 固定提交中的 CPU 标杆逐元素一致；BF16 `atol=rtol=2^-6`，FP32 `atol=2^-16, rtol=2^-10`，matched ratio 0.999 | 通过 |
| A5 NPU，BF16/FP32 checkpoint，seed 419 ragged、空 batch 与 padding 请求混排 | PASS | PASS | PASS | NPU 单算子测试 4/4 通过；新增 `Q/K=[1,7,8,128]`、`V/out=[1,7,24,128]`、10 个空请求和 2 个非空请求混排用例，5 秒内完成，无卡死；输出和 replay 与 CPU FP64 标杆一致 | 通过 |
| CPU FP64 双测试套件回归 | PASS | PASS | PASS | 算子参考测试和 Stable-ABI 参考测试各 6/6 通过，覆盖空请求、padding、BF16/FP32 checkpoint 与 PR 20485 固定标杆对齐 | 通过 |
| CPU FP64 PR 20485 标杆，`BS=32/64/72`、`HV=32/48/128`、`TPB=1..16`、L2Norm on/off、BF16/FP32 checkpoint | PASS | PASS | PASS | 576/576 通过；输出最大绝对误差 `0.2397`，replay correction/key/decay 最大绝对误差分别为 `1.27e-4`、`6.93e-8`、`1.69e-7` | 通过 |
| A5 性能，`BS=32/64/72`、`HV=32/48/128`、`TPB=1/2/4/8` | PASS | PASS | PASS | 36/36 完成采集；35/36 优于 RecurrentGDN；几何平均耗时比为 `0.473`；仅 `BS32/TPB1/HV32` 略慢，耗时比 `1.017` | 通过 |
