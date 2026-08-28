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

覆盖范围包括：每轮 1/2/3/4 个 head、`HK:HV=1:2/1:3/1:6`、跨 round
重读 raw K、g/gk、BF16/FP32 rolling state、`state_v_first` 两种布局、
`exp/exp2`、定长/变长、尾 chunk 和最终 `v_new-only` 分支。
反向用例覆盖 gate 二选一、固定 chunk size、g/gk head 约束、varlen batch 和
`state_v_first` 对应的 state shape。
