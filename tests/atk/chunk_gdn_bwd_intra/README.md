# ChunkGdnBwdIntra ATK 工程

本目录提供 `chunk_gdn_bwd_intra` 的逐 Stage 混合容差测试交付件。开发期间通过
`stage=0/1/2` 逐步开放输出；泛化精度矩阵统一使用 `stage=2` 比较正式输出。
本文件是最终测试交付入口，集中维护标杆、输入生成、覆盖矩阵、执行方式和验收结论。

## CPU 标杆与输入生成

- CPU 标杆语义来源于用户提供并确认的参考材料；仓内版本位于
  `executor_chunk_gdn_bwd_intra.py`，与测试文件同提交。`_stage0_ref`、`_stage1_ref`、
  `_stage2_ref` 分别实现三个开发 Stage，正式精度测试使用 `_stage2_ref`。
- CPU 节点以 FP64 计算。NPU 与 CPU 使用相同 seed；`q/k/d_o` 使用标准差 `0.20` 的
  确定性正态分布，`v/A` 使用标准差 `0.05` 的确定性正态分布，`beta` 使用
  `[0.1,0.9)` 均匀分布，`g` 使用公共 gate 输入生成器。
- 变长 profile 提供 `cu_seqlens`，executor 据此生成 canonical `chunk_indices`；CPU 和 NPU
  使用相同 sequence 边界切分，不跨 sequence 计算。
- 精度标准为 ATK `mixed_tolerance_bm`。`scripts/smoke_stage.py --compare` 仅用于开发期
  定位，额外输出 `rtol=5e-3, atol=5e-3` 的 `torch.allclose` 结果，不替代正式 ATK 结论。

## 正式接口约束

- `q/k` 为 `[B,HK,T,128]`，`v/d_o` 为 `[B,HV,T,128]`。
- `g/beta` 为 `[B,HV,T]`，各自支持 BF16/FP32；不支持 FP16。
- `A` 为 `[B,HV,T,64]`，主 dtype 与 `q` 相同。
- `q/k/v/A/d_o` 支持 BF16/FP16，`HV/HK` 为 `1/2/3/4`。
- `chunk_size` 固定为 `64`，`use_exp2` 默认 `true`。
- 变长场景由 profile 提供 `cu_seqlens`，executor 生成 canonical `chunk_indices`，
  两者同时传入 NPU，并在 CPU 标杆中按相同 sequence 边界切分。

## TilingKey 覆盖

目标 SoC 为 A5（`ascend950`）。TilingKey 由定长/变长、主 dtype、`g` dtype 和
`beta` dtype 四个维度组成；下表 16 个 profile 对四个维度做笛卡尔积。`G`、尾块、
`use_exp2` 和 `scale` 是 key 内运行时分支，由表中用例同时覆盖。

| ID | 场景 | 主 dtype | g/beta dtype | G | B/T 或 sequence 长度 | exp | 额外覆盖 |
| ---: | --- | --- | --- | ---: | --- | --- | --- |
| 0 | 定长 | BF16 | BF16/BF16 | 1 | B=2,T=64 | exp2 | `HV=5`，部分 CG |
| 1 | 定长 | BF16 | BF16/FP32 | 2 | B=1,T=65 | exp | 尾长 1 |
| 2 | 定长 | BF16 | FP32/BF16 | 3 | B=1,T=96 | exp2 | 尾长 32 |
| 3 | 定长 | BF16 | FP32/FP32 | 4 | B=1,T=97 | exp | 尾长 33，`scale=0.03125` |
| 4 | 定长 | FP16 | BF16/BF16 | 4 | B=1,T=128 | exp2 | 两个完整 chunk |
| 5 | 定长 | FP16 | BF16/FP32 | 3 | B=1,T=65 | exp | 尾长 1 |
| 6 | 定长 | FP16 | FP32/BF16 | 2 | B=1,T=96 | exp2 | 尾长 32 |
| 7 | 定长 | FP16 | FP32/FP32 | 1 | B=1,T=97 | exp | 尾长 33 |
| 8 | 变长 | BF16 | BF16/BF16 | 4 | 0/1/64/33 | exp2 | 空 sequence、尾长 1/33 |
| 9 | 变长 | BF16 | BF16/FP32 | 3 | 32/64 | exp | 尾长 32 |
| 10 | 变长 | BF16 | FP32/BF16 | 2 | 65/64 | exp2 | 多 chunk、尾长 1 |
| 11 | 变长 | BF16 | FP32/FP32 | 1 | 33/64/64 | exp | 尾长 33 |
| 12 | 变长 | FP16 | BF16/BF16 | 1 | 64/1 | exp2 | 完整 chunk、尾长 1 |
| 13 | 变长 | FP16 | BF16/FP32 | 2 | 32/64/64 | exp | 尾长 32 |
| 14 | 变长 | FP16 | FP32/BF16 | 3 | 33/64 | exp2 | 尾长 33 |
| 15 | 变长 | FP16 | FP32/FP32 | 4 | 128/96 | exp | 多 chunk、`scale=0.125` |

上述输入条件是预期选择条件；实际 TilingKey 选择以 A5 host tiling 或运行时记录为准，
没有选择记录时不把对应 key 标记为已完成覆盖。

## 当前实测结果

2026-09-05 使用 ATK 26.8.8 的 `mixed_tolerance_bm`，以 FP64 CPU 标杆比较 Stage 2 的
`w/u/dv_local`。16 个用例分四批执行，每批 4/4 通过，合计 16/16 通过。测试时关闭
GM 初始化，避免测试框架的额外显存占用影响算子验证。

固定长度诊断组合 `B=1,T=65,HK=3,HV=6,K=V=128` 的 w/u/dv_local 最大绝对误差
分别为 `4.8314e-4`、`1.2095e-4`、`4.8804e-4`。该组合的算子调用和设备同步均正常返回。

2026-09-05 使用 CT v0.9.1 对 `q/k/d_o` 标准差 `0.10/0.15/0.20/0.25` 进行值域扫描。
`0.10` 无法在 BF16 case 拦截全零 `dv_local`，`0.15` 的最弱 case 仅刚好越过 99% 匹配率
门槛；选择 `0.20` 后，16 个 profile 的真实 `w/u/dv_local` 均满足混合容差，且全零
`dv_local` 的匹配率降至 BF16 最高 `78.57%`、FP16 最高 `6.51%`，所有 profile 均能
拦截全零输出。CT 分布图未发现系统偏移、分叉或按 head/chunk 聚集的误差。

## 执行方式

```bash
GEN_CASES_DTYPE_NUMBERS=8 bash tests/atk/run_test_cpu.sh \
  -op=chunk_gdn_bwd_intra -scope=gen_cases
bash tests/atk/run_test_cpu.sh \
  -op=chunk_gdn_bwd_intra -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

单独排查 Stage 时可继续使用 `scripts/smoke_stage.py`；正式泛化精度结论只使用完整
`w/u/dv_local` 输出。
