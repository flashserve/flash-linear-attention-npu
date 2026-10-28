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

## Gradient dtype writeback

All gate arithmetic, chunk partials and parameter reductions remain FP32.
The A2/A3 final gate stage casts dg with CAST_RINT into the retired raw-gate
input slot before DMA, then waits for MTE3 before that slot is reused by MTE2.
The A5 gate stage uses its retired raw-gate tile at UB[176,208) KiB for the
BF16 result; the existing stage7 MTE3-to-MTE2 event protects its lifetime.
No UB allocation or workspace size changes are needed.

Stage 12 casts dA only after the FP32 reduction. A2/A3 uses sum+8 (a spare
32-byte-aligned slot); A5 uses daOut+16, within the existing 70 KiB buffer.
BF16 output ownership groups 16 adjacent heads per core to avoid competing
writes within one 32-byte DMA block. dt_bias partials and outputs stay FP32.
The output descriptors select writeback types; no extra launch is introduced
in A2/A3 or A5 V2. A5 legacy retains its FP32 device ABI and L2 converts its
outputs as a compatibility fallback; it does not gain the V2 fusion benefit.

Validation compares all eight raw_g/beta/A_log dtype combinations against
FP32-output baseline results rounded once to each requested output dtype.
Cover both output orders, saved/recompute, dense/packed, empty sequence and
partial chunks. Separately compare the independent CPU mathematical golden.

### A2 dtype 定向验证（2026-10-28）

179.166.1.3 / Ascend910B3 / CANN 9.2.0，独立 wheel 构建、安装通过。
72 组用例在 Stable 与 ctypes 两入口共 144 次调用通过；比较全部有效输出，
仅将旧版基线的 db/dg/dA 转换为新约定 dtype，结果逐元素一致。
覆盖 8 种 gate/beta/A_log 类型组合、dense/packed、保存/重计算、
BNSD/BSND/NTD/TND、尾块、空序列、norm、sigmoid/2-sigmoid 和 H=17。
3 项错误梯度 dtype 均被 ACLNN 拒绝；真实前向缓存接反向的公开接口
回归 16/16 通过。独立 CPU 标杆的递推、自梯度、beta 模式与布局自检通过。

验证目录为 `/home/chenlinxin/wys/pr993-dtype-20261028`，wheel SHA256 为
`a8f99a088a716236a61e4a4323582238725e47910736418c3d45e6aa99e05cec`。
本轮按用户要求只验证 A2，未验证 A3/A5 设备，未执行完整 ATK、性能或 sanitizer 验收。
