# A2/A3 泛化开发期验证

分支 feat/fla-compat，开发基线 fc0bc08e。本记录只证明已完成的前向范围；不作为反向、A2 或完整规格验收。

## 实现与追溯

- Prepare：arch22/chunk_kda_fwd_prepare_general.h，FP32 norm/gate/score/三角逆/W/U。general launcher 按 chunk/head 分派，元数据省略 chunk_indices 时从 cu_seqlens 定位。
- FwdH：arch22/chunk_fwd_h_general.h，K 全维、V32 列片的 FP32 递推；先完整 WH/Vnew，再更新状态；h 是更新前快照。StateT 只用于初态读取和末态转换。
- Finalize：arch22/chunk_kda_fwd_finalize_general.h，FP32 qg_scaled@h 与 Aqk@Vnew 分别累加后合并，输出 cast FP16/BF16。状态地址使用真实 K/V。
- 组合 V2：新路径中间缓存 FP32，旧 BF16/128 缓存类型保留。A2/A3 Python 前向仅序列化/路由，公开规格检查由 ACLNN 执行。
- 新模板为 AIV-only；深度 1 TQue 管理 DMA，逐任务末尾 PIPE_ALL。未宣称跨核流水重叠或性能收益。

## 构建与运行环境

A3 独立开发服务器，任务目录 <task-root>，CANN 9.3.0.P009，ascend910_93/dav-2201。构建使用 -j$(nproc)，本机为 640。wheel 安装在本任务独立 venv，未修改系统环境；测试使用空闲物理设备 8。

前向三个独立算子包、组合 wheel 和增量 run 包均构建成功。第一版 wheel 为 flash_linear_attention_npu_a3-26.10.0+unknown-py3-none-manylinux_2_34_aarch64.whl；最后增量包覆盖本任务 wheel layer，op_api MD5=5cec75fcb054e3981698a4ef6b294a4f。

原 verify_libcust_opapi_md5.py 不识别 wheel_layers，误选 base OPP。开发辅助 verify_active_overlay.py 使用已安装 runtime_plan 的首选 binding，保留原脚本 hash 比较：op_api、全部 30 个 kernel .o、Python wrapper 全部 [OK]。未改仓库检查脚本或使用外部 OPP 覆盖。

## 上游原始用例

固定 fla-org/flash-linear-attention 提交 c55dc553edf44701bf703715ed052d811f8a6376。

public_forward_plugin.py 只将被测 chunk_kda 调用转换到安装后的 npu_chunk_kda_fwd，转换参数拼写和 A_log/dt_bias 的平铺视图，不修改输入值、标杆、精度阈值或断言；不包含 CPU/Triton 计算回退。

| 原始用例 | 最终结果 |
| --- | --- |
| test_chunk_varlen_prefill，FP16，D60 | PASS |
| test_chunk_varlen_prefill，FP16，D64 | PASS |
| test_chunk_varlen_prefill，FP16，D100 | PASS |
| test_chunk_varlen_prefill，FP16，D128 | PASS |
| test_chunk_varlen_prefill，FP16，D256 | PASS |

每个进程命令为 `ASCEND_RT_VISIBLE_DEVICES=8 python -m pytest tests/ops/test_kda.py -p public_forward_plugin -k 'test_chunk_varlen_prefill and D<dim>' -q -s`，外层 timeout 60s。五个最终日志 <task-root>/public-final-d{60,64,100,128,256}.log。kernel 路由经组合 V2 及新增 FP32 缓存三段分派。

D256 首次因预激活 gate + safe_gate=True 被旧 Host 校验拒绝；修正新通用路径校验后，原用例原开关通过。该标志不导致预激活 gate 被重复激活。

## 尚未覆盖

- A2 设备、K64/V32、BF16 新维度、完整 norm/sigmoid/allow_neg_eigval 矩阵及性能。
- Recompute/Dhu/BwdFinalize、完整反向和保存/重算精度。
- 原范围排除的组合 h0/dht/dh0、GVA、C32、Graph、A5 新功能。依赖这些特性的原始用例记未覆盖，不删除断言或改变输入凑通过。
- 不使用本地 ATK。当前开发期记录不能替代最终上游验收报告。
