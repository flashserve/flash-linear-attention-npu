# 验证记录

## 02 标杆运行

| 范围 | 运行结果 | 计算语义 | 精度策略 | 证据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| CPU FP64，固定 seed，常规 final commit、boundary prefix、零长度、null page、同页 final/boundary、BF16 state | PASS | PASS | PASS | `test/test_reference.py` 6 项全部通过 | 通过 |

## 04 算子开发

| 检查项 | 结果 |
| --- | --- |
| Python 语法检查 | PASS |
| Stable-ABI 源码审计 | PASS |
| Stable-ABI 覆盖检查（严格模式） | PASS，1 个调用点，0 mismatch/missing header |
| Ascend C host/API 编译 | PASS |
| Ascend C kernel 编译 | PASS，BF16/FP32 与 `align_mode` 模板组合均已实际生成 |
| run 包封装 | PASS |
| wheel 打包 | PASS，scoped overlay 仅导出 `npu_gdn_recoverssm_commit` |

## 单算子精度验收

| 测试项 | 结果 |
| --- | --- |
| CPU FP64 参考测试 | PASS，6 项全部通过 |
| FP32 state + align + 多层 + boundary | PASS |
| BF16 state + align + 多层 + boundary | PASS |
| 空 batch、非 align 路径 | PASS |
| 单算子 NPU 精度测试总数 | PASS，9 项全部通过 |

本阶段验收结论：编译、Stable-ABI 契约、单算子精度和窗口矩阵性能均通过。
性能口径使用 `msprof` 的 `Task Duration(us)`。

## PR #20485 窗口矩阵性能

5 个受支持的 `K=V=128` 配置在同一轮 `msprof` 采集中执行，结果位于同一份
`op_summary`。输入数据全部在 CPU 构造后拷贝到 NPU，profiling 期间除目标
算子外不调用其他 NPU 计算算子；`op_statistic` 中
`GdnRecoverssmCommit` 的占比为 100%。

| HV | Window | State dtype | Batch | 初始(us) | 双行VF(us) | 拆分循环(us) | Tiling裁剪(us) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | FP32 | 12 | 24.734 | 18.786 | 13.699 | 13.557 |
| 8 | 2 | BF16 | 12 | 28.199 | 22.328 | 13.013 | 11.393 |
| 8 | 5 | FP32 | 12 | 47.016 | 33.849 | 14.977 | 14.619 |
| 24 | 9 | BF16 | 4 | 73.661 | 51.990 | 20.425 | 20.035 |
| 3 | 5 | FP32 | 3 | 16.841 | 11.827 | 6.880 | 6.665 |

`K=33/V=47` 不属于当前算子 `K=V=128` 的支持范围，因此不产生 kernel task，
性能记为 N/A。5 个 case 总耗时从 `190.451 us` 降至 `68.994 us`，相对初始
版本提升 `63.8%`；相对双行 VF 版本再提升 `50.3%`。以上为单轮
`Task Duration(us)`，未使用 Python wall time。

TilingData 裁剪后，使用同一矩阵执行 6 轮 `msprof`，逐 case 取中位数，总耗时
为 `66.269 us`，相对拆分循环版本再降低 `3.95%`。5 个 case 均未退化，
`op_statistic` 中目标算子耗时占比为 100%。
