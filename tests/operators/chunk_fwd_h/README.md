# chunk_fwd_h 测试

用例定义唯一来源为 `tests/op_cases/chunk_fwd_h.json`。精度脚本通过独立稳定入口
`fla_npu.ops.ascendc.chunk_fwd_h` 调用算子，不加载旧算子或 legacy `torch.ops.npu` 接口。

```bash
python tests/operators/chunk_fwd_h/accuracy/test_chunk_fwd_h.py
```

接口反向校验：

```bash
pytest -q tests/operators/chunk_fwd_h/ut/test_chunk_fwd_h_validation.py
```

可通过 `CHUNK_FWD_H_CASE_IDS` 仅运行逗号分隔的指定 case：

```bash
CHUNK_FWD_H_CASE_IDS=g_ratio_1_to_6_cross_round \
python tests/operators/chunk_fwd_h/accuracy/test_chunk_fwd_h.py
```

覆盖范围包括：每轮 1/2/3/4 个 head、`HK:HV=1:2/1:3/1:5/1:6/1:7`、跨 round
重读 raw K、奇数 tail pair、长序列 credit 复用、g/gk、BF16/FP32 rolling state、
`state_v_first` 两种布局、`exp/exp2`、dense 多 batch、定长/变长、显式/自动
chunk indices、尾 chunk 和最终 `v_new-only` 分支。A5 另覆盖 FP32 state 跨 chunk 常驻、
单 head W/U/K/g 双 bank lookahead，以及 FP32/BF16 gate 的 tail63/tail1 边界。
反向用例覆盖 gate 二选一、输入/输出 shape 与 dtype、固定 chunk size、g/gk head
约束、varlen 元数据、canonical chunk indices、state shape/layout、ND/连续输出、可选
final-state 物理存在性和必选 tensor 空指针，并校验对应 aclnn 返回码。

性能用例同样定义在 `tests/op_cases/chunk_fwd_h.json` 的 `performance_cases`，runner
只负责 warmup 和重复 launch。性能结论使用 `msprof` 的目标 kernel 记录，不使用 Python
wall time：

```bash
python tests/operators/chunk_fwd_h/performance/run_chunk_fwd_h.py \
  --case-id a5_g_h4_t512 --warmup 5 --iterations 50
```

六条 A5 性能场景的 case id 为：`a5_b2_hk16_hv32_t11264`、
`a5_varlen_h32_t65536_s64`、`a5_b4_hk96_hv96_t128`、
`a5_b1_hk32_hv32_t160`、`a5_b6_hk6_hv6_t1084` 和
`a5_b1_hk12_hv12_t1084`。补充的 batch 对照场景为
`a5_b1_hk16_hv32_t11264`；其 BF16/FP32 initial-state 对照分别为
`a5_b1_hk16_hv32_t11264_bf16_initial` 和
`a5_b1_hk16_hv32_t11264_fp32_initial`。变长场景的 65 个 `cu_seqlens` 边界由 seed 202
随机生成后固化在用例 JSON 中，runner 根据这些边界生成 sequence-major canonical
`chunk_indices`。
