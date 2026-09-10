# ChunkKdaFwdPrepare ATK 工程

本目录通过稳定入口 `fla_npu.ops.ascendc.chunk_kda_fwd_prepare` 验证 Prepare
算子，不加载 `torch_npu` dispatcher 或 `torch.ops.npu` legacy 入口。算子固定接收
9 个输入并返回以下 13 个公开输出：

```text
gk, aqk, akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

## 标杆语义

`executor_chunk_kda_fwd_prepare.py` 独立实现 CPU 标杆，覆盖 Q/K L2Norm、chunk
内 gate cumsum、S=4 的 16 行局部参考缩放、因果 `Aqk/Lkk`、32x32 分块三角逆、
`W/U` 和 post-WU 输出。标杆显式保留 `qg_scaled` 与 `K_beta_g` 对已舍入 BF16
中间值的二次消费，避免把两次 cast 错误合并为一次。

输入从相同 seed 的 FP32 随机数生成，并先量化到接口声明的 BF16/FP32，再分别送入
CPU 和 NPU 节点。ATK 使用 `mixed_tolerance_bm` 比较全部 13 个输出。

## 用例矩阵

| 文件 | 用例数 | 覆盖内容 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_prepare.json` | 171 | 四种 layout、dense/varlen、tail、GVA 和 144 个编译组合 |
| `atk_chunk_kda_fwd_prepare_perf.json` | 7 | 模型 shape、跨 wave GVA、layout 与 varlen |
| `atk_chunk_kda_fwd_prepare_mss.json` | 144 | 每个编译组合的双 chunk 同步与内存检查 |

144 个编译组合来自：

```text
2 gate dtype x 2 beta dtype x 2 norm mode x
3 beta mode x 3 gate mode x 2 exp domain = 144
```

其中 beta mode 为 raw、sigmoid、two-sigmoid；gate mode 为预计算 step、softplus、
safe sigmoid。GVA 用例包含大于 4 的比例以及 `HV=160`，只要求
`HV >= HK` 且 `HV % HK == 0`，不施加额外 head 数上限。
变长边界用例还覆盖 1025 条空/非空混合序列，不施加额外序列数上限。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=mssanitizer
```

重建三份冻结 JSON：

```bash
python3 tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py \
  --output-dir tests/atk/chunk_kda_fwd_prepare \
  --summary
```

A2、A3、A5 使用同一份 `soc=all` 精度矩阵；实际设备测试结果应分别记录，不能用
任一平台的结果替代其他平台。
