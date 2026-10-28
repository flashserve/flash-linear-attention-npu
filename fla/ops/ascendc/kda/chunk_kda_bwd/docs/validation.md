# A2/A3 总接口输出适配：开发验证

## 修改范围

基于 main `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`，分支 `fix/kda-bwd-output-contract`。
只修改 A2/A3 的 dq/dk 输出分配与 dG/dBeta 布局适配；Python/ACLNN 签名、
小算子 kernel、A5 ACLNN 实现和其他仓的模型接入均未修改。接口与设计见 api.md、design.md。

## 2026-10-28 验证

| 检查 | 结果 |
| --- | --- |
| Python AST、git diff --check | 通过 |
| Python/ACLNN 签名与基线对比 | 不变 |
| A5 ACLNN 实现与基线对比 | 不变；未进行 A5 设备测试 |
| A2 wheel 构建、独立环境安装 | 通过；筛选 chunk_kda_bwd，包含编排依赖和 Stable 扩展 |
| 输出分配测试 | 48 组通过，见 tests/stable_abi/test_chunk_kda_bwd_output_contract.py |
| NPU 定向回归 | 20 条 × Stable/ctypes 两入口，共 40 次通过 |
| 非法输出检查 | FP32 dq、FP32 dk、head-major db、head-major dg 均在 ACLNN 返回 161002 |

设备为 Ascend910B3。覆盖 dense/packed、四种布局、
保存/重计算、norm、普通 beta/sigmoid/2-sigmoid、奇数 head、尾块、空序列和 H=T。

NPU 对比使用旧 fa5493f wheel：将旧 dq/dk 转 BF16、旧 head-major db/dg 转
BSND/TND 后，与新输出按 dtype、shape 和数值逐元素比较，rtol=atol=0，全部一致。
已核对 fa5493f 到本次 main 的相关反向小算子源码没有变化。该检查验证输出适配等价性，
不替代独立 CPU 双标杆验收。

wheel SHA256：`2b1ccc197e56bbba71ee589bcf78d1f8dde3f565f3350467278a3825e79a528d`。

## 模型 case 1 profiling

B=1、H=4、T=131046、K=V=128，579 条变长序列；norm/sigmoid 融合开启，
disable_recompute=True。独立脚本按模型数据布局调用总接口，未修改模型仓。
每组预热 5 次，采集 2 轮 × 10 次；前向缓存生成、加载和正确性检查均不计时。
七个模型梯度与旧链路按新输出约定对齐后逐元素一致。

| g/A_log/dt_bias dtype | 每次任务数 | 任务耗时合计 | 设备时间跨度 |
| --- | --- | --- | --- |
| BF16 | 10：3 个计算核、1 次 dO 转置、6 次 gate Cast | 11.615 ms | 11.843 ms |
| FP32 | 4：3 个计算核、1 次 dO 转置 | 10.968 ms | 11.195 ms |

原始 op_summary 确认没有 dq/dk 往返 Cast 和 dG/dBeta 布局往返。
这两组是 dtype 配置对照，不是修改前后版本的性能 A/B；不据此声明版本提速比例。

## 待验证

本轮尚未运行完整 200 条 ATK、msprof 性能对比和 A5 设备回归，不据此声明全量验收完成。
CPU 标杆公式和数值阈值不变；后续 ATK 应按新契约更新返回布局与 benchmark 输出 dtype，
保留独立 FP64 Golden。模型接入按用户要求留在后续任务处理。
