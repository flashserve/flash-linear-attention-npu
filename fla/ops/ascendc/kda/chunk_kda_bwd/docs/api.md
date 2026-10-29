# KDA 反向优化接口

## A2/A3 总接口输出约定

A2/A3 的所有 implementation 模式统一使用原总 ACLNN 接口编排小算子。
本次不新增接口或参数，返回顺序仍为 `(dq, dk, dv, db, dg, dh0, dA, dbias)`。

| 输出 | dtype | 布局 |
| --- | --- | --- |
| dq/dk/dv | 分别跟随 q/k/v，当前仅支持 BF16 输入 | output_layout：BNSD/BSND/NTD/TND |
| db/dg | 分别跟随 beta/raw_g 的输入 dtype | 跟随 output_layout；db 省略末尾 D 维 |
| dA | FP32 | 不受 output_layout 控制 |
| dh0/dbias | dh0 为 None；dbias 为 FP32，未传时为 None | 不受 output_layout 控制 |

直接调用 ACLNN 时，dqOut/dkOut 必须与 q/k 类型相同，不能再传 FP32 输出。
dbOut/dgOut 必须分别匹配 beta/raw_g 的 dtype，dAOut/dbias 固定 FP32；dbOut/dgOut 跟随 output_layout。
无论是否重计算、是否融合 norm/sigmoid，均使用以上输出规则。A_log/dt_bias 固定 FP32，其余输入限制不变。

总接口保留原有输出布局约定，旧调用方的 head-major→token-major 转置不变。
dq/dk 从 FP32 改为跟随输入 dtype，调用方需更新相关 dtype 断言。
签名和返回数量不变，新版绑定与 op-api 库需配套部署。
A5 原版和 V2 同样采用 db/dg 跟随输入 dtype、dA/dbias 固定 FP32 的规则；布局和路由不变。
这是明确的非兼容 dtype 变更，调用方需更新原先固定 FP32 的断言和输出分配。
下文优化路径说明针对 A5。

Finalize 小接口的 output_layout 统一控制 dq/dk/dv/db/dg，由 kernel 直接按目标
布局写回。默认 dense 为 BNSD、packed 为 NTD。直接使用 #962 Finalize 的调用方
不能再假设 db/dg 恒为 token-major，应按所选 output_layout 处理这五个输出。
总接口无需在 Finalize 后转置；A2/A3 和 A5 V2 直接按目标 dtype 写入返回张量。
数学计算、参数梯度的分块累加和最终归约仍为 FP32，仅最终写回转为输入 dtype。
A5 legacy 保留 FP32 kernel ABI，由 L2 转换输出，仍会有独立 Cast。
sigmoid 模式下 beta 指接口实际接收的 FP32 beta_eff，db 跟随该张量；
接口不会推断模型中生成 beta_eff 之前的投影 dtype。

## 调用方式

沿用 Python 入口 `chunk_kda_bwd`，新增三个可选参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| use_gate_in_kernel | True | 仅支持 True，在反向算子内完成门控激活的链式求导，需要提供 raw_g 和 A_log |
| implementation | "auto" | 可选 auto、legacy、optimized |
| q_rstd | None | 前向 Q 归一化保存的 FP32 标准差倒数 |
| k_rstd | None | 前向 K 归一化保存的 FP32 标准差倒数 |

普通保存中间量调用仍走原路径。auto 在传入完整 rstd 对或
`disable_recompute=False` 时选择 V2；legacy 不接受 rstd。
不支持的优化配置在启动前报错，运行失败后不自动回退。

```python
from fla_npu.ops.ascendc import chunk_kda_bwd

dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
    q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale,
    raw_g=raw_g, A_log=A_log, dt_bias=dt_bias,
    use_gate_in_kernel=True, safe_gate=True, use_exp2=True,
    disable_recompute=True, implementation="optimized",
    q_rstd=q_rstd, k_rstd=k_rstd,
)
```

未使用前向 Q/K 归一化时省略 rstd。使用时，q/k 必须是前向保存的归一化
BF16 张量；Finalize 已融合归一化反向，调用方不应重复计算。

## 输入与输出

| 项目 | 优化路径约束 |
|---|---|
| 平台与维度 | Ascend950；Hq=Hv，K=V=128，chunk_size=64，B/H/T 为正 |
| Token 布局 | 连续 dense [B,H,T,D] 或 packed [H,T,D] |
| h 布局 | 与前向一致：[B,Nc,H,K,V] 或 [Nc,H,K,V]，连续存储 |
| q/k/v/dO、Aqk/Akk、Token/状态缓存 | BF16；gk 为 FP32 |
| beta / db | BF16 或 FP32，输入输出类型一致 |
| raw_g | BF16 或 FP32，Finalize 在片上转为 FP32 |
| A_log | FP32 |
| dt_bias | 可选 FP32，共 H×128 个元素 |
| q_rstd / k_rstd | 成对提供，FP32，形状为 Token 形状去掉最后一维 |
| dq/dk/dv | BF16 |
| dg | 跟随 raw_g 的 dtype |
| dA | FP32 |
| dbias | FP32；未传时为 None |
| 状态 | 不支持 initial_state/dht；dh0 为 None，state_v_first=False |
| Gate | safe_gate、use_gate_in_kernel、use_exp2 均为 True；-5≤lower_bound<0 |

前向公开接口的 h 可直接传入，无需转置。旧调用方需删除 h 的 head-major 转换；
H=Nc 时仅检查 shape 无法发现旧布局，调用方仍须同步更新。内部 dh 同样为 NT-first。
元数据使用 Host INT64、按序列排列的规范 chunk 顺序；Python 层压缩空序列并
重排序列编号，直接调用 V2 时须自行提供该形式，不接受设备端元数据或 T=0。

重计算时设置 `disable_recompute=False`，将 gk/w/qg/kg/v_new/h 设为 None，
仍须提供 Aqk/Akk。当前要求 H≤256 且为 8 的倍数。
内部 FwdH 直接产生 NT-first 的 h，无需转置。

## 实现位置

Stable ABI 与 ctypes 共用 `_kda_policy.py` 的校验，各自负责启动。
Stable ABI 扩展现有 `npu_chunk_kda_bwd` 注册，由 C++ 选择原 ACLNN 或 V2；
无新增公开 Python 算子，升级时需同步重编译扩展。

V2 是独立 L2 符号，保留旧 ABI；声明见
[aclnn_chunk_kda_bwd_v2.h](../op_host/op_api/aclnn_chunk_kda_bwd_v2.h)。
设计见 [设计说明](design.md)。

## Finalize Cast 融合

A2/A3 staged、A5 V2 和独立 Finalize 支持 raw_g/beta 的 4 种 BF16/FP32
组合；A_log/dt_bias 固定 FP32。saved 模式将原始输入直接传给 Finalize，
取消 raw_g 的 GM Cast；调用方需保证 A_log/dt_bias 为 FP32。
重算分支仍为 Recompute 单独创建 FP32 临时参数，随后 Finalize 使用原始输入。
A5 legacy 的 dt_bias 仍要求 FP32。本次不改变前向的参数 dtype 支持范围。
