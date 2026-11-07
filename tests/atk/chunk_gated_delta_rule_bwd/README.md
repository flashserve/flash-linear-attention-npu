# ChunkGatedDeltaRuleBwd ATK 测试说明

本目录验证 `chunk_gated_delta_rule_bwd` 融合接口。NPU 节点只调用一次公开融合接口；
CPU 节点使用同一组冻结输入，独立串联四个阶段，并提供 FP64 golden 与 FP32
同精度 benchmark 两条 CPU 标杆；NPU DUT 与两条标杆均比较最终有效梯度。

## 标杆与输出

CPU 标杆依次计算 `ChunkGdnBwdIntra`、`ChunkFwdH`、
`ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize`。FP64 golden 保留
双精度阶段计算，FP32 benchmark 按 BF16 输入、FP32 累加和阶段落盘规则复现同精度链，
避免把 NPU 中间结果作为标杆。

精度使用 ATK 原生 `cv_fused_double_benchmark`，采用 L1 误差比标准：
`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5`，不使用
`mixed_tolerance_bm`。

有效输出顺序为 `dq, dk, dv, d_beta, d_g`；传入 `initial_state` 时额外比较
`dh0`。其中 `d_beta` 和 `d_g` 固定使用 BSND `[B,T,HV]`，其余有效输出跟随
对应输入布局。保留接口输出 `d_a_log` 和 `d_dt_bias` 当前为空，不参与数值比较。

## 覆盖范围

- A2/A3 与 A5，输入主类型为 BF16，`K=V=128`，`chunk_size=64`。
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

# CPU 双标杆：NPU DUT + CPU FP32 benchmark + CPU FP64 golden
bash tests/atk/chunk_gated_delta_rule_bwd/scripts/run_double_benchmark.sh 0
```

双标杆脚本会检查三路角色数量和 XLSX 中的逐用例精度结果；任一执行或精度失败均返回非零。

布局改造另用同一组输入对比 BNSD/BSND、NTD/TND 的逐字节结果，同时检查输出 shape、dtype、
连续性和有限值。补充场景覆盖 T=1/63/64/65、G=3、多 batch、空序列、混合标量 dtype、
两种指数模式、T=H、非连续输入和可追踪数值；这项对照不替代 CPU 双标杆精度测试。

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --device 0 --save-dir <new_result_dir>
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --emit-supplemental-cases <supplemental_json>
GDN_ATK_CASE_JSON=<supplemental_json> \
  bash tests/atk/chunk_gated_delta_rule_bwd/scripts/run_double_benchmark.sh 0
```
