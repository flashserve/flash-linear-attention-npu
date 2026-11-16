# ChunkGdnBwdIntra ATK 工程

本目录提供 `chunk_gdn_bwd_intra` 的混合容差测试交付件，泛化精度矩阵比较完整的
`w/u/dv_local` 正式输出。
本文件是最终测试交付入口，集中维护标杆、输入生成、覆盖矩阵、执行方式和验收结论。

## CPU 标杆与输入生成

- CPU 标杆语义来源于用户提供并确认的参考材料；仓内版本位于
  `executor_chunk_gdn_bwd_intra.py`，与测试文件同提交，由 `_cpu_ref` 生成完整输出。
- 普通 CPU 节点（`FunctionApi.__call__`）使用 `run_cpu(high_precision=True)`：
  输入先量化到与 DUT 相同的主/辅助 dtype，再提升为 FP64 计算；`A_bg/A_beta/D`
  保留设备工作区的主 dtype 量化，最终输出转回主 dtype。
- 显式设备匹配参考 `FunctionApi.cpu` / `run_cpu(high_precision=False)` 使用 FP32
  累加，保留工作区量化，返回 FP32 输出；它不是普通 CPU 节点的 golden 入口。
- 独立数学标杆 `run_cpu_benchmark` 使用相同已量化输入，提升为 FP64 后不再量化
  `A_bg/A_beta/D` 或最终输出；`FunctionApi.cpu_benchmark` 仅在交给 ATK 时将输出
  归一为 FP32。ATK benchmark 任务通过 `task_result.is_benchmark_task` 在
  `__call__` 中显式分派，不依赖不存在的 BaseApi 自动 `cpu_benchmark` 钩子。
- 本工程正式配置仍使用单标杆 `mixed_tolerance_bm`，不是
  `cv_fused_double_benchmark`；其普通 CPU 节点的 `is_benchmark_task=False`，
  实际走上述 FP64 计算、保留工作区量化并返回主 dtype 的 golden 路径，不调用独立
  数学标杆。不因提供独立数学标杆而切换比较器或验收标准。
  NPU 与 CPU 使用相同 seed；`q/k/d_o` 使用标准差 `0.20` 的
  确定性正态分布，`v/A` 使用标准差 `0.05` 的确定性正态分布，`beta` 使用
  `[0.1,0.9)` 均匀分布，`g` 使用公共 gate 输入生成器。
- 变长 profile 提供 `cu_seqlens`，executor 据此生成 canonical `chunk_indices`；CPU 和 NPU
  使用相同 sequence 边界切分，不跨 sequence 计算。
- 精度标准为 ATK `mixed_tolerance_bm`。`scripts/smoke_operator.py --compare` 仅用于开发期
  定位，额外输出 `rtol=5e-3, atol=5e-3` 的 `torch.allclose` 结果，不替代正式 ATK 结论。

## 正式接口约束

- `q/k` 为 `[B,HK,T,128]`，`v/d_o` 为 `[B,HV,T,128]`。
- `g/beta` 为 `[B,HV,T]`，各自支持 BF16/FP32；不支持 FP16。
- `A` 为 `[B,HV,T,64]`，主 dtype 与 `q` 相同。
- `q/k/v/A/d_o` 支持 BF16/FP16。`HV/HK`：A5（`ascend950`）为 `1/2/3/4`；
  A2（`ascend910b`）支持 `1~16`，其中 `5~16` 走运行时 G 桶（TilingKey `G_KEY=0`）。
- `chunk_size` 固定为 `64`，`use_exp2` 默认 `true`。
- 变长场景由 profile 提供 `cu_seqlens`，executor 生成 canonical `chunk_indices`，
  两者同时传入 NPU，并在 CPU 标杆中按相同 sequence 边界切分。

## TilingKey 覆盖

主目标 SoC 为 A5（`ascend950`）；A2/A3（`ascend910b`/`ascend910_93`，同为
DAV_2201）的运行时 G 桶覆盖另附独立交付件（见下文）。TilingKey 由定长/变长、`G`、
主 dtype、`g` dtype 和 `beta` dtype 五个维度组成；`G` 直接选择编译期模板参数
（`G=1/2/3/4`），A2/A3 上 `G>4` 统一落入 `G_KEY=0` 的运行时 G 实例（比例经
`tiling.groupSize` 传入）。下表 18 个 A5 profile 覆盖五个维度的全部取值及
`G=1/2/3/4` 执行路径。尾块、`use_exp2` 和 `scale` 是 key 内运行时分支，由表中用例
同时覆盖。

A2/A3 运行时 G 桶（`G_KEY=0`）由独立交付件覆盖，不混入 950 主流程：

- 生成器：`gen_chunk_gdn_bwd_intra.py` 的 `ARCH22_PROFILES`（5 条，G=5/6/8/12/16，
  定长/变长、跨 hk 切片与部分 CG 切片）与注册名
  `generator_chunk_gdn_bwd_intra_a2`；
- 用例文件：`chunk_gdn_bwd_intra_a2.yaml`（soc 枚举 `ascend910b/ascend910_93`）；
- 快照：`atk_chunk_gdn_bwd_intra_arch22.json`（A2 用例集，5 条 `ascend910b`，
  G=5/6/8/12/16）。A3（ascend910_93）与 A2 同为 DAV_2201、共用同一构建产物，
  归入 A2 用例集，不再单独维护 ascend910_93 用例；需要复核时在对应 SoC
  机器上执行同一快照；
  `python gen_chunk_gdn_bwd_intra.py` 重新生成该快照；
- 执行入口：`scripts/run_a2_validation.sh`（A2 直接运行；A3 机器上原样
  执行同一脚本，命令见"执行方式"）；
- 负向拦截：`scripts/smoke_operator.py --expect-reject`（构造 GVA 超限输入，
  断言 aclnn 参数校验层拒绝且错误文本携带实际 HV/HK 与平台范围；如
  A5 上 `--group 5 --expected-arch 3510 --expect-reject` 应被拒，A2/A3 上
  `--group 17 --expected-arch 2201 --expect-reject` 应被拒）；
- 占位参数说明：本算子 ATK 交付件沿用存量的 `low_precision_marker/case_spec`
  占位参数模式承载 dtype/shape/attr 空间（历史契约如此）；新增 A2/A3 用例
  为与存量快照结构一致继续沿用，向真实算子签名对齐的迁移另行安排。

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
| 16 | 定长 | BF16 | FP32/FP32 | 1 | B=1,T=1793 | exp2 | 29 个 work；尾长 1 复用首核 q/k L1 |
| 17 | 变长 | FP16 | BF16/FP32 | 1 | 1792/1 | exp | 29 个 work；尾长 1 复用首核 q/k L1 |

A2/A3 运行时 G 桶用例（对应 `atk_chunk_gdn_bwd_intra_arch22.json`；A3 与 A2
同为 DAV_2201，共用本用例集，不再单独设 ascend910_93 用例）：

| ID | 场景 | 主 dtype | g/beta dtype | G | B/T 或 sequence 长度 | exp | 额外覆盖 |
| ---: | --- | --- | --- | ---: | --- | --- | --- |
| 18 | A2 定长 | BF16 | BF16/BF16 | 5 | B=1,T=64 | exp2 | HK=2,HV=10；运行时 G 桶；CG=4 的第二切片跨 hk，末切片仅 2 个 head（部分 CG） |
| 19 | A2 定长 | FP16 | FP32/BF16 | 6 | B=1,T=65 | exp | HK=1,HV=6；尾长 1；部分 CG，不跨 hk |
| 20 | A2 定长 | BF16 | FP32/FP32 | 8 | B=2,T=96 | exp2 | HK=2,HV=16；尾长 32；四个完整 CG，hk 边界与切片对齐 |
| 21 | A2 变长 | FP16 | BF16/FP32 | 16 | 33/64 | exp2 | HK=1,HV=16；尾长 33 |
| 22 | A2 变长 | BF16 | FP32/BF16 | 12 | 65/64 | exp | HK=1,HV=12；多 chunk、尾长 1 |

上述输入条件是预期选择条件；实际 TilingKey 选择以 A5 host tiling 或运行时记录为准，
没有选择记录时不把对应 key 标记为已完成覆盖。

## 当前实测结果

2026-09-12 使用 ATK 26.8.8 的 `mixed_tolerance_bm`，以 FP64 CPU 标杆比较完整的
`w/u/dv_local`。使用 `20260904..20260921` 的 18 个固定 seed 用例一次完整执行，执行
成功 18/18、精度通过 18/18、失败 0。用例 16/17 分别覆盖定长和变长场景中
`Nwork=29>blockDim=28`、末尾 chunk 仅 1 token 的跨 work q/k L1 复用。

固定长度诊断组合 `B=1,T=65,HK=3,HV=6,K=V=128` 的 w/u/dv_local 最大绝对误差
分别为 `4.8314e-4`、`1.2095e-4`、`4.8804e-4`。该组合的算子调用和设备同步均正常返回。

### A2/A3 运行时 G 桶（G_KEY=0）验证结论

在 ascend910b 上安装含本特性的 wheel 后执行
`atk_chunk_gdn_bwd_intra_arch22.json` 的全部 5 条 ascend910b 用例（G=5/6/8/12/16）：

- **精度标准**：与主 json/主 yaml 相同的 `mixed_tolerance_bm`（仓上本算子
  统一验收口径）；
- **执行与判定**：5/5 执行成功、5/5 精度通过（`acc_pass_result: Pass`，
  通过率 100%），无 hang、无 aicore 异常；
- **误差水平**：bf16 用例大元素（|golden|>0.01）最大相对误差 1.1%~5.9%，
  fp16 用例 0.1%~0.8%，与同仓 950 上 G=1~4 的误差水平一致，未见随 G 增长的
  系统性偏差；

A3（ascend910_93）与 A2 同为 DAV_2201、共用同一构建产物，已归入上述 A2
用例集，不再单独维护 ascend910_93 用例；需要复核时在 A3 机器上执行同一
快照即可。

2026-09-05 使用 CT v0.9.1 对 `q/k/d_o` 标准差 `0.10/0.15/0.20/0.25` 进行值域扫描。
`0.10` 无法在 BF16 case 拦截全零 `dv_local`，`0.15` 的最弱 case 仅刚好越过 99% 匹配率
门槛；选择 `0.20` 后，16 个 profile 的真实 `w/u/dv_local` 均满足混合容差，且全零
`dv_local` 的匹配率降至 BF16 最高 `78.57%`、FP16 最高 `6.51%`，所有 profile 均能
拦截全零输出。CT 分布图未发现系统偏移、分叉或按 head/chunk 聚集的误差。

## 执行方式

950 主流程（18 条 A5 用例）：

```bash
GEN_CASES_DTYPE_NUMBERS=9 bash tests/atk/run_test_cpu.sh \
  -op=chunk_gdn_bwd_intra -scope=gen_cases
bash tests/atk/run_test_cpu.sh \
  -op=chunk_gdn_bwd_intra -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

A2/A3 运行时 G 桶（`atk_chunk_gdn_bwd_intra_arch22.json`，须在对应 SoC 机器上运行）：

```bash
# 需先安装含本算子的 wheel 并激活 CANN 环境
python tests/atk/chunk_gdn_bwd_intra/gen_chunk_gdn_bwd_intra.py   # 重新生成 arch22 快照

# A2（ascend910b）；A3（ascend910_93）机器上原样执行同一命令复核
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./a2_results \
  bash tests/atk/chunk_gdn_bwd_intra/scripts/run_a2_validation.sh accuracy
```

负向拦截验证（断言超限 GVA 被参数校验层拒绝且报错含实际值）：
`--expected-arch` 在负向模式必填，独立声明期望平台范围，不从被测报文反推。
负向模式强制使用 `FLA_NPU_STABLE_LAUNCH=inline`，避免延迟启动或设备同步错误
被误判为参数拒绝；`--compare` 是布尔开关，不支持 `--compare=false`，无需比较时省略。

```bash
# A2 上 G=17（超上限）：期望被拒
python tests/atk/chunk_gdn_bwd_intra/scripts/smoke_operator.py \
  --group 17 --expected-arch 2201 --expect-reject
# A5 上 G=5（超该平台上限）：期望被拒
python tests/atk/chunk_gdn_bwd_intra/scripts/smoke_operator.py \
  --group 5 --expected-arch 3510 --expect-reject
```

开发期快速排查可使用 `scripts/smoke_operator.py`；正式泛化精度结论使用完整
`w/u/dv_local` 输出。

本地回归（不连接 NPU）：

```bash
python tests/atk/chunk_gdn_bwd_intra/scripts/test_review_regressions.py
python tests/atk/chunk_gdn_bwd_intra/scripts/test_cpu_reference.py
```

后者需要可用的 CPU PyTorch，不依赖 ATK 安装；覆盖独立 FP64 数学标杆、设备匹配
工作区量化、相同输入提升和 ATK 普通/benchmark 任务分派。PyTorch 无法导入时显式
skip，不计入数值验证通过。
