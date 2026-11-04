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

## Issue #795 典型 shape 验证

本次新增范围使用独立用例生成入口，保留原小 shape 矩阵。生成四个算子各两例：
`B=1, HK=16, HV=128, K=V=128, chunk_size=64`，主类型 BF16，g/beta FP32，
BNSD；定长 `T=11264`，变长 `T=16384`、16 段各 1024。其余用例待后续提供。

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/gen_gva_typical.py --output-dir /tmp/gva_typical
# 在对应 tests/atk/<op> 目录，沿用 CPU 单标杆及 mixed_tolerance_bm：
atk node --name npu_dut --backend npu --devices 0 \
  node --name cpu_golden --backend cpu task \
  -c /tmp/gva_typical/atk_<op>_gva_typical.json --task accuracy --bm_device cpu \
  -p ./executor_<op>.py --gm_init_flag 4096 -to 14400
```

2026-11-04 验证：Ascend950PR（设备4），CANN 9.2.0-beta.2，torch/torch_npu 2.12.0，
ATK 提交 45cc4061（单 CPU worker、solo pool，避免 PyTorch prefork 等待）。
CPU 单标杆及原 mixed_tolerance_bm 阈值不变，四个算子各两例全部通过（8/8）。
case_spec 必须以 string 属性传入；non_param 会被 ATK 从 executor kwargs 排除，
不能据此声称验证了指定变长分段。

A5 Host/Kernel/wheel 构建及安装产物 MD5 校验通过；A2 的 intra/finalize Host/Kernel
编译及 2 处 ABI 检查通过。构建脚本的多 SoC 参数被拒绝，因此分别编译 A5/A2；
本轮不代表 A2/A3 硬件回归。实际 runtime leader 辅助函数的 G=5..16、17/31/32/64
映射检查通过；NPU 精度仅覆盖上表两个 G=8 典型 shape。

显存口径：G>4 旧版本被 Host 拒绝，无法直接实测同 shape 修改前峰值。资源审计中
intra 不新增 GM workspace；finalize 保留 7R，仅增加两个 C×65536 字节的 FP32 区。
C=28 时用户 workspace 从 24.5 MiB 增至 28 MiB，新增 3.5 MiB；父链张量及四节点
生命周期未改动。allocator 分桶可能造成少量取整差异，不能把总峰值当作新增占用。

完整调用边界使用同一份 BNSD 输入；所需 transpose/contiguous、四节点及输出转换
均包含在计时内。预热5次、采样20次，表中取中位数；普通包、设备4。

| 典型 shape | NPU event (ms) | 同步 wall (ms) | peak allocated (MiB) |
| --- | ---: | ---: | ---: |
| 定长 T=11264 | 12.721 | 12.793 | 5089.000 |
| 变长 T=16384，16×1024 | 18.095 | 18.150 | 7350.261 |

性能门槛尚未判定：设计指定的 FLA cf9f520fcfef873534132e65e61e12d36b77ab62，
在真实248 KiB UB配置下，Triton-Ascend 3.6.0及3.6.0.post2均在
prepare_wy_repr_bwd_npu 编译时报 UB 超限（需要352.75 KiB）。一次缺失TBE路径
而回退64 KiB资源预算的运行不作为有效标杆，不计算或宣称≤1.25×FLA。
需在可运行该 FLA 提交的同设备环境补齐比较。

MSS同步检测未通过：带行号调试包在 G=8 和既有 G=4 对照中，均报告 intra Vector
共享 L1 复用的 CrossCoreWaitFlag（chunk_gdn_bwd_intra_vector.h:400）等待。
该同步点未在本次变更中修改；现有证据不能区分工具兼容问题与既有同步缺陷，
故不凭猜测改写流水，也不以正常精度通过代替 MSS 通过结论。
