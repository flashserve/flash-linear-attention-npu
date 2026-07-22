# KDA 正向 C128/V256 精度与性能验证报告

## 1. 验证范围

本报告记录 Kimi Delta Attention（KDA）正向 AscendC 算子对以下能力的增量验证：

- `chunk_size=128`：C9-C12、C26、C30。
- `Vdim=256`：C21-C24、C27-C29、C31-C33。
- BF16、`Kdim=128`、BSND/NTD、dense/varlen、随机非对齐尾块。
- Prepare AIV/Cube 双槽流水和 PostWu 双 AIV 大块路径的性能收益。

第 2 至 5 节记录的 C128/V256 历史矩阵不使用双标杆。该矩阵的精度标杆使用 CPU FP32 中间计算：汇总指标比较 NPU BF16 输出转 FP32 后的值与 CPU FP32 结果；CT 可视化前再把 CPU 结果 cast 到 BF16 输出边界，与 NPU BF16 输出保持同一公开 dtype。PR228 四阶段拆分后的 fla-org 三方对标见第 6 节。每条历史用例使用：

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

另对 NTD、`chunk_size=128`、`Vdim=256`、4 条非对齐变长序列固定输入连续执行 10 次；全部公开输出逐元素二进制一致，用于看护 Prepare 双槽 `ready/free` 同步协议的确定性。

完整组合包还执行了 NTD 模型形状回归：`T=131072`、`H_K=H_V=2`、`Kdim=Vdim=128`、8 条非对齐序列、`initial_state=None`。`o`、`final_state` 和全部中间量均为 finite；`o` 与 BF16 CPU 参考一致，FP32 `final_state` 最大绝对误差约 `4e-4`。

## 4. 性能结果

性能使用 `msopprof --aic-metrics=BasicInfo`，只统计设备侧 kernel duration：

| 用例 | 优化前 | 优化后 | 降幅 | 优化后主要耗时 |
| --- | ---: | ---: | ---: | --- |
| BNSD `B=1,H_K=1,H_V=2,T=16384,K=V=128,C=64` | 4.751 ms | 3.531 ms | 25.7% | Prepare 1.512 ms，fwd_h 1.385 ms |
| NTD `B=1,H_K=H_V=32,T=65536,K=V=128,C=64` | 206.142 ms | 127.648 ms | 38.1% | Prepare 93.123 ms，Finalize 21.991 ms |
| BSND C9 `B=64,H_K=H_V=8,T=2048,K=V=128,C=128` | 114.572 ms | 68.035 ms | 40.6% | Prepare 42.374 ms，layout 10.011 ms |
| NTD C33 `B=1,H_K=16,H_V=48,T=8999,K=128,V=256,C=128` | 48.621 ms | 27.210 ms | 44.0% | Prepare 19.064 ms，Finalize 5.114 ms |

优化项与证据：

- PostWu 使用两个 AIV subblock 分摊连续行，并把逐行搬运改为 UB 预算内的大块搬运/向量处理，耗时降低 `84.8%~92.2%`。
- Prepare 使用两槽 `ready/free` 生产者消费者队列，使 AIV factor 准备与 AIC Catlass score GEMM 重叠，耗时降低 `10.0%~31.7%`。
- `fwd_h` 的 varlen chunk offset 元数据由逐任务 GM 标量读取改为一次 `DataCopyPad` 批量搬入 UB，随后只在 UB 中索引；长序列 `fwd_h` 从 `6.545 ms` 降至 `5.505 ms`。
- 三个长序列/大 shape 的 Prepare AIC `wait_id4` 由 `29.94/20.43/12.26 ms` 降到 `14.92/7.20/3.40 ms`。
- score workspace 按 core 数和固定队列深度分配，不随 T 或 chunk 数线性增长。

## 5. 当前边界与后续优化

- 当前交付验证范围为 `Kdim=128`、`Vdim=128/256`、`chunk_size=64/128`、FP16/BF16。
- varlen partial chunk 已保证正确性，当前使用完整 tile 补中性值并只回写有效区；仍可增加专用 partial 性能模板。
- Prepare 仍占长序列链路约 `62%~72%`，后续优先降低 score scratch 往返、score block 控制开销和 solve 串行段。
- BSND 大 head 场景还受到 layout swap 影响；上游已提供 BNSD/NTD 时应直接使用性能 layout。
- 本报告不覆盖 KDA 反向算子，也不把未执行的 sanitizer 检查写成通过结论。

## 6. PR228 四阶段拆分增量复测

PR228 将正向通路拆为 `ChunkKdaFwdPrepare`、`ChunkKdaFwdPostWu`、独立
`ChunkGatedDeltaRuleFwdH` 和 `ChunkKdaFwdFinalize`。A2/A5 使用
fla-org/flash-linear-attention 的 Triton 实现作为实际三方参考，固定目标用例为 NTD
`[32,8192,128]`、`K=V=128`、`chunk_size=64`、FP16：

| 平台 | 输出 | max abs | mean abs | bad ratio |
| --- | --- | ---: | ---: | ---: |
| A2 | `o` | 4.76837e-7 | 2.19373e-8 | 0 |
| A2 | `final_state` | 6.35767e-6 | 8.23598e-7 | 0 |
| A2 | `g` | 0 | 0 | 0 |
| A5 | `o` | 4.76837e-7 | 2.19394e-8 | 0 |
| A5 | `final_state` | 6.40936e-6 | 8.23737e-7 | 0 |
| A5 | `g` | 0 | 0 | 0 |

A2/A5 的小 shape 全中间量对比覆盖 `safe_gate=false/true`，所有输出 `bad_ratio=0`；
独立 `fwd_h` 的 `h`、`v_new`、`final_state` 也与三方 `exp2` 公式对齐。A2/A5 另覆盖
TND/BNSD、FP16/BF16、V=256 和极端负 gate；A3 完成目标 FP16 `fwd_h` 和全部
Prepare tiling key 的编译验证。

本轮继续完成两项热点优化：

- Aqk/Akk 的两次 MMAD 共用单槽 L1 B buffer，右矩阵 `K^T` 首次搬入后保持驻留，
  第二次 MMAD 只更新左矩阵；同时使用 MMAD 自身的 pre/final event 协议替代两次
  `PIPE_ALL`。
- A5 `fwd_h` 将 gate cast/`ln(2)`/exp、`u - workspace`、state cast/row-scale/add
  分别融合进 regbase VF，并以两条独立寄存器链处理相邻向量或行。

目标用例执行 `msopprof --aic-metrics=BasicInfo`，设备侧均值如下：

| 平台 | 阶段 | 优化前 | 优化后 | 变化 |
| --- | --- | ---: | ---: | ---: |
| A2 | Prepare | 9.708 ms | 9.604 ms | -1.1% |
| A2 | PostWu | 0.716 ms | 0.722 ms | +0.8% |
| A2 | FwdH | 0.800 ms | 0.800 ms | 0.0% |
| A2 | Finalize | 0.744 ms | 0.745 ms | +0.2% |
| A2 | 合计 | 11.968 ms | 11.871 ms | -0.8% |
| A5 | Prepare | 7.964 ms | 7.847 ms | -1.5% |
| A5 | PostWu | 0.606 ms | 0.607 ms | +0.2% |
| A5 | FwdH | 1.162 ms | 1.157 ms | -0.4% |
| A5 | Finalize | 0.600 ms | 0.600 ms | 0.0% |
| A5 | 合计 | 10.332 ms | 10.211 ms | -1.2% |

对照 Triton 同用例约 11 ms，A5 AscendC 四阶段合计时延降低约 7.2%。profile 中一次
`ZerosLike` 来自测试输入 `k = torch.zeros_like(q)` 的准备过程，不属于 KDA 四阶段通路。
