# ChunkGatedDeltaRuleBwd ATK 测试说明

本目录验证 `chunk_gated_delta_rule_bwd` 融合接口。NPU 节点只调用一次公开融合接口；
CPU 节点使用同一组冻结输入，独立串联四个阶段的 CPU 标杆，比较最终有效梯度。

## 标杆与输出

CPU 标杆依次计算 `ChunkGdnBwdIntra`、`ChunkFwdH`、
`ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize`。阶段边界按照
NPU 接口的数据类型量化，避免把 NPU 中间结果作为标杆。

有效输出顺序为 `dq, dk, dv, d_beta, d_g`；传入 `initial_state` 时额外比较
`dh0`。其中 `d_beta` 和 `d_g` 固定使用 BSND `[B,T,HV]`，其余有效输出跟随
对应输入布局。保留接口输出 `d_a_log` 和 `d_dt_bias` 当前为空，不参与数值比较。

## 覆盖范围

- Ascend 950，输入主类型为 BF16，`K=V=128`，`chunk_size=64`。
- 覆盖 `BNSD/BSND/NTD/TND`，定长与变长，以及完整块、尾块和多块。
- 覆盖 `G=HV/HK` 的 `1/2/3/4`。
- 覆盖 BF16/FP32 标量输入、初始状态、末状态梯度和 `state_v_first`。
- 覆盖 Q/K L2Norm backward、beta sigmoid backward 及保留输入。
- `use_exp2=true`、`use_gate_in_kernel=false`。

## 用例矩阵

`gen_chunk_gated_delta_rule_bwd.py` 定义 12 个确定性 profile。矩阵以较小 shape
覆盖接口分支和四阶段组合语义，避免大 shape 使 CPU 标杆成为测试瓶颈。

## 执行

```bash
GEN_CASES_DTYPE_NUMBERS=12 bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_bwd -scope=gen_cases
bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_bwd -soc=ascend950 -npu_device_id=0 -scope=accuracy
```
