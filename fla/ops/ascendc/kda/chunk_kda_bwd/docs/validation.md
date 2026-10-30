# do 总接口转置：开发验证

基线：PR #993 的 `3eef55d4`；本轮验证源码：`cebb1e1a`。用户确认：do 固定 BSND/TND，由 KDA ACLNN 转为 BNSD/NTD，
不修改 Dhu 接口。Prepare 同样复用转换结果，撤回新增的 do 布局扩展。

数学公式、CPU 标杆、输出布局/dtype、保存/重计算和 beta_eff 契约不变。

| 检查 | 状态 |
|---|---|
| Dhu/Prepare 与 PR 基线 `a7c608f2` 一致 | 通过：15 个回退文件，两个目录相对基线无 diff |
| A2/A3、A5 V2 的 Prepare/Dhu 共用单次 do 转置 | 通过静态检查；4 处 Dhu、3 处 Prepare 调用与原 L0 参数数一致 |
| A2：定长、三维变长、B=1 四维变长，四种 layout | 77 组配置、412 次 Stable/ctypes 调用与基线逐元素一致 |
| A2：异常及独立小接口回归 | 21 项拒绝检查、8 组独立小接口对比通过；融合时原始 beta 仅提供 descriptor 的检查通过 |
| A2：真实前向→反向 | 48 组通过独立 CPU 递推标杆，覆盖保存/重算、L2Norm 和 BF16/FP32 gate |
| 静态检查、格式、GDN 输出契约单测 | 静态工具 0 error/0 warning，diff 检查通过，2 项输出契约单测通过 |
| CI 契约单测 | Linux + Node.js 24.22.1 下 165 项通过；替代此前 Windows/旧 Node.js 环境失败记录 |
| A2 构建及运行产物 | Ascend910B3、CANN 9.2.0，256 并行编包及独立环境安装通过；op-api、38 个 kernel .o 与 Python wrapper 一致性检查通过 |
| A2 原 CI KDA Example | 4/4 通过：定长/变长精度、尾块、空段 guard |
| A2 模型 profiling | BSND：4 Cast、1 do Transpose；BNSD：4 Cast、2 前向 Transpose、1 do Transpose；无新增 Cast |
| A5 本轮实卡 | 未验证，先前版本的 A5 结果不能代替本轮结果 |

本轮复用既有值域、种子和精度阈值，未调整标杆以适配结果。
profiling 使用 T=131072、H=4、D=128、31 段/2066 chunks，B=1 四维变长，保存模式，
L2Norm 与 beta sigmoid 融合；g FP32，原始 beta/A_log/dt_bias BF16，beta_eff FP32。
预热 5 次、采样 10 次。do 转置为 `[1,131072,4,128]→[1,4,131072,128]`，
BSND/BNSD 下平均耗时分别为 202.152/201.494 μs，每次反向仅一份，供 Prepare/Dhu 共用。
4 次 Cast 均来自模型边界的 A_log/dt_bias 输入及梯度转换。采样包含 KDA 与接入层，
不含卷积、投影和通信；设备任务耗时之和不等于端到端延迟。
本记录为定向开发验证，不代表完整 ATK/sanitizer、A3/A5 实卡或双平台 CI 合入门禁通过。
