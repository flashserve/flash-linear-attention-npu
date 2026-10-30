# do 总接口转置：开发验证

基线：PR #993 的 `3eef55d4`。用户确认：do 固定 BSND/TND，由 KDA ACLNN 转为 BNSD/NTD，
不修改 Dhu 接口。Prepare 同样复用转换结果，撤回新增的 do 布局扩展。

数学公式、CPU 标杆、输出布局/dtype、保存/重计算和 beta_eff 契约不变。

| 检查 | 状态 |
|---|---|
| Dhu/Prepare 与 PR 基线 `a7c608f2` 一致 | 通过：15 个回退文件，两个目录相对基线无 diff |
| A2/A3、A5 V2 的 Prepare/Dhu 共用单次 do 转置 | 通过静态检查；4 处 Dhu、3 处 Prepare 调用与原 L0 参数数一致 |
| 定长、三维变长、B=1 四维变长的形状/寻址 | 代码检查通过；实卡待验证 |
| 静态检查、格式、GDN 输出契约单测 | 静态工具 0 error/0 warning，diff 检查通过，2 项输出契约单测通过 |
| 全量 CI 契约单测 | 未通过：Windows 首轮 165 项中 23 error、49 skip；2 项编码错误在 UTF-8 下复跑通过，其余缺少 Node.js，未作通过结论 |
| 构建、CPU 标杆/小算子链路、profiling | 未执行：A2、A5 验证服务器 SSH 连接超时 |

实卡恢复后复用已有独立 CPU 标杆和小算子链路验证，重建包含 Prepare/Dhu 和总接口的配套包。
覆盖双 Python 后端、四种 layout、保存/重算、
L2Norm、尾块和门控 dtype；profiling 确认只有一份 do 转置供两个消费者复用。
先前版本的实卡结果不能作为本轮通过依据。本记录不代表 ATK/sanitizer 正式验收通过。
