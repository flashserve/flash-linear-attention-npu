# KDA 正向 C128/V256 精度与性能验证报告

## 1. 验证范围

本报告记录 Kimi Delta Attention（KDA）正向 AscendC 算子对以下能力的增量验证：

- `chunk_size=128`：C9-C12、C26、C30。
- `Vdim=256`：C21-C24、C27-C29、C31-C33。
- BF16、`Kdim=128`、BSND/NTD、定长/变长序列、随机非对齐尾块。
- Stage 1 AIV/Cube 双槽流水和 Stage 3 双 AIV 大块路径的性能收益。

本轮不使用双标杆。精度标杆使用 CPU FP32 中间计算：汇总指标比较 NPU BF16 输出转 FP32后的值与 CPU FP32 结果；CT 可视化前再把 CPU 结果 cast 到 BF16 输出边界，与 NPU BF16 输出保持同一公开 dtype。每条用例使用：

```text
ct viz <npu.pt> <cpu_fp32.pt> -wl 1 -sc 100000
```

长序列从代表性的 `B/HV` 对中抽样，V 维覆盖首列、中列和末列，最多均匀保留 100000 点。所有 NPU 输出先做全量 finite 检查。相对误差定义为：

```text
relative_error = abs(npu - cpu_fp32) / max(abs(cpu_fp32), 1e-6)
```

因此接近 0 的输出会放大逐点相对误差；结论同时参考 `rel_p99`、绝对误差、cosine 和 CT 图中是否存在结构性偏差。

## 2. 精度结果

| 用例 | Layout | B | H_K | H_V | T | V | Chunk | 序列数 | 采样点 | Abs mean | Abs max | Rel mean | Rel p99 | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C9 | BSND | 64 | 8 | 8 | 2048 | 128 | 128 | dense | 24576 | 2.146e-6 | 1.180e-5 | 0.0198 | 0.2451 | 0.9999924 |
| C10 | BSND | 32 | 16 | 16 | 4096 | 128 | 128 | dense | 49152 | 2.163e-6 | 1.322e-5 | 0.0203 | 0.2611 | 0.9999883 |
| C11 | BSND | 16 | 32 | 32 | 8192 | 128 | 128 | dense | 98304 | 2.172e-6 | 1.338e-5 | 0.0208 | 0.2707 | 0.9999872 |
| C12 | BSND | 8 | 32 | 32 | 16384 | 128 | 128 | dense | 98304 | 2.199e-6 | 1.548e-5 | 0.0210 | 0.2765 | 0.9999855 |
| C21 | NTD | 1 | 16 | 32 | 16384 | 256 | 64 | 128 | 98304 | 1.392e-6 | 1.248e-5 | 0.0168 | 0.2255 | 0.9999871 |
| C22 | NTD | 1 | 16 | 32 | 16384 | 256 | 64 | 128 | 98304 | 1.413e-6 | 1.198e-5 | 0.0168 | 0.2251 | 0.9999858 |
| C23 | NTD | 1 | 21 | 63 | 16384 | 256 | 64 | 1 | 98304 | 2.225e-6 | 1.492e-5 | 0.0210 | 0.2779 | 0.9999845 |
| C24 | NTD | 1 | 8 | 32 | 65536 | 256 | 128 | 172 | 100000 | 1.823e-6 | 1.381e-5 | 0.0190 | 0.2590 | 0.9999875 |
| C26 | NTD | 1 | 4 | 32 | 7178 | 128 | 128 | 17 | 86136 | 1.872e-6 | 1.457e-5 | 0.0190 | 0.2503 | 0.9999880 |
| C27 | NTD | 1 | 2 | 64 | 11202 | 256 | 64 | 32 | 100000 | 1.882e-6 | 1.420e-5 | 0.0194 | 0.2484 | 0.9999871 |
| C28 | BSND | 1 | 16 | 32 | 4096 | 256 | 64 | dense | 49152 | 2.195e-6 | 1.227e-5 | 0.0213 | 0.2787 | 0.9999868 |
| C29 | BSND | 16 | 21 | 63 | 2048 | 256 | 64 | dense | 24576 | 2.209e-6 | 1.260e-5 | 0.0212 | 0.2738 | 0.9999909 |
| C30 | BSND | 711 | 4 | 32 | 196 | 128 | 128 | dense | 2352 | 1.596e-6 | 9.607e-6 | 0.0193 | 0.2460 | 0.9999909 |
| C31 | BSND | 176 | 2 | 64 | 24 | 256 | 64 | dense | 288 | 6.119e-7 | 5.213e-6 | 0.0103 | 0.0749 | 0.9999936 |
| C32 | NTD | 1 | 16 | 48 | 16387 | 256 | 64 | 667 | 98322 | 6.709e-7 | 9.465e-6 | 0.0132 | 0.1873 | 0.9999900 |
| C33 | NTD | 1 | 16 | 48 | 8999 | 256 | 128 | 13 | 100000 | 1.985e-6 | 1.335e-5 | 0.0192 | 0.2529 | 0.9999902 |

汇总结论：

- 16 条用例全量 NPU 输出均为 finite。
- `abs_max` 为 `5.21e-6~1.55e-5`，cosine 为 `0.9999845~0.9999936`。
- `rel_p99` 为 `0.0749~0.2787`；高值集中在 CPU 结果接近 0 的点，未伴随绝对误差突增。
- 逐条 CT 图均紧贴 `y=x`，未观察到固定 chunk、固定 head、尾块或 `cu_seqlens` 边界相关的块状/条纹状结构性误差。

## 3. 回归结果

当前源码完成以下重点单测回归：

```text
test_chunk_kda_fwd_matches_reference
test_chunk_kda_fwd_fp16_matches_reference
test_chunk_kda_fwd_vdim256_matches_reference
test_chunk_kda_fwd_chunk128_matches_reference
test_chunk_kda_fwd_bsnd_export_dependency_matches_reference
test_chunk_kda_fwd_without_intermediate_matches_export_and_reference
test_chunk_kda_fwd_bnsd_direct_matches_reference
test_chunk_kda_fwd_ntd_direct_matches_reference
```

重点覆盖 FP16/BF16、C64/C128、V128/V256、四种公开 layout 和 `return_intermediate` 两种模式。`return_intermediate=False/True` 在相同输入下逐位一致，专门看护 split L0 中间量的输入依赖和 workspace 生命周期。

另对 NTD、`chunk_size=128`、`Vdim=256`、4 条非对齐变长序列固定输入连续执行 10 次；全部公开输出逐元素二进制一致，用于看护 Stage 1 双槽 `ready/free` 同步协议的确定性。

完整组合包还执行了 NTD 模型形状回归：`T=131072`、`H_K=H_V=2`、`Kdim=Vdim=128`、8 条非对齐序列、`initial_state=None`。`o`、`final_state` 和全部中间量均为 finite；`o` 与 BF16 CPU 参考一致，FP32 `final_state` 最大绝对误差约 `4e-4`。

## 4. 性能结果

性能使用 `msopprof --aic-metrics=BasicInfo`，只统计设备侧 kernel duration：

| 用例 | 优化前 | 优化后 | 降幅 | 优化后主要耗时 |
| --- | ---: | ---: | ---: | --- |
| BNSD `B=1,H_K=1,H_V=2,T=16384,K=V=128,C=64` | 4.751 ms | 3.531 ms | 25.7% | stage 1 1.512 ms，fwd_h 1.385 ms |
| NTD `B=1,H_K=H_V=32,T=65536,K=V=128,C=64` | 206.142 ms | 127.648 ms | 38.1% | stage 1 93.123 ms，stage 2 21.991 ms |
| BSND C9 `B=64,H_K=H_V=8,T=2048,K=V=128,C=128` | 114.572 ms | 68.035 ms | 40.6% | stage 1 42.374 ms，layout 10.011 ms |
| NTD C33 `B=1,H_K=16,H_V=48,T=8999,K=128,V=256,C=128` | 48.621 ms | 27.210 ms | 44.0% | stage 1 19.064 ms，stage 2 5.114 ms |

优化项与证据：

- Stage 3 使用两个 AIV subblock 分摊连续行，并把逐行搬运改为 UB 预算内的大块搬运/向量处理，耗时降低 `84.8%~92.2%`。
- Stage 1 使用两槽 `ready/free` 生产者消费者队列，使 AIV factor 准备与 AIC Catlass score GEMM 重叠，耗时降低 `10.0%~31.7%`。
- `fwd_h` 的变长序列 chunk offset 元数据由逐任务 GM 标量读取改为一次 `DataCopyPad` 批量搬入 UB，随后只在 UB 中索引；长序列 `fwd_h` 从 `6.545 ms` 降至 `5.505 ms`。
- 三个长序列/大 shape 的 Stage 1 AIC `wait_id4` 由 `29.94/20.43/12.26 ms` 降到 `14.92/7.20/3.40 ms`。
- score workspace 按 core 数和固定队列深度分配，不随 T 或 chunk 数线性增长。

## 5. 当前边界与后续优化

- 当前交付验证范围为 `Kdim=128`、`Vdim=128/256`、`chunk_size=64/128`、FP16/BF16。
- 变长序列 partial chunk 已保证正确性，当前使用完整 tile 补中性值并只回写有效区；仍可增加专用 partial 性能模板。
- Stage 1 仍占长序列链路约 `62%~72%`，后续优先降低 score scratch 往返、score block 控制开销和 solve 串行段。
- BSND 大 head 场景还受到 layout swap 影响；上游已提供 BNSD/NTD 时应直接使用性能 layout。
- 本报告不覆盖 KDA 反向算子，也不把未执行的 sanitizer 检查写成通过结论。
