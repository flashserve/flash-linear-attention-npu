# Finalize 输出布局直写

设计规则版本：V2。接口见 [api.md](api.md)。本次只改变末阶段 dg/db 的 GM
输出地址及行间 stride，保留数学计算、UB 排布、workspace、核任务和同步事件。

A2/A3 使用现有 outputTokenMajor：token-major 地址为 `(b*T+t)*H+h`，
head-major 地址为 `(b*H+h)*T+t`。dg 按 D=128 展开，db 为标量。
token-major 行间跳过 H-1 个 head；head-major 行间无间隔。
db 的 UB 每行保持 32 字节对齐，用已有 DataCopyPad 写回，无额外 Cast/Transpose。

A5 的 GatePost 和 dBeta 写回复用 FinalizeOutputOffset，并使用相同的布局标记
选择 GM stride。总接口 V2 删除原先 token-major 到 head-major 的适配。

验证覆盖 dense/packed、两种输出顺序、非对齐尾块、奇数 head、H=T、BF16/FP32 beta、
norm/sigmoid 分支。预期值由原有 CPU 计算或已验证基线按目标布局排列；不得改变
数学公式、输入值域或精度阈值。原版 CI example 不修改，用于检查旧调用方式。

## 定向验证

2026-10-28，在 Ascend910B3 / CANN 9.2.0 上重建 wheel 后，24 组独立 Finalize
用例的七个输出与旧实现按目标布局排列后的结果逐元素一致，12 组默认布局检查通过。
覆盖 dense B=2、packed 含空序列、T=65 尾块、H=3、H=T=4、norm 开关及 BF16/FP32 beta。
这是写回布局等价性验证，不替代完整 ATK 或 A5 设备验收。
