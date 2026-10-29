# KDA 反向接口

## A2/A3 总接口：单 layout 与原始 beta / beta_eff

本节为当前 A2/A3 实验接口契约；替代旧 input_layout/output_layout 以及融合时 beta 参数承载缓存的约定。
Python `npu_chunk_kda_bwd` 新增 keyword-only `beta_eff=None`。返回顺序仍为
`(dq, dk, dv, db, dg, dh0, dA, dbias)`。ACLNN 在 kRstdOptional 后、layout 前新增 betaEffOptional；新版绑定和 OPP 必须配套更新。

- layout=BSND/BNSD/TND/NTD 只控制 qkv 输入及 dq/dk/dv 输出；传成对 rstd 时 q/k 为固定 head-major 的 q_hat/k_hat。
- do/raw_g/beta 和 dg/db 固定 token-major；beta 的 scalar shape 为 [B,T,H] 或 [T,H]，支持 BF16/FP32。
- sigmoid 关闭：beta 是前向实际使用的有效 beta，beta_eff 必须为空。
- sigmoid 开启：beta 是前向接口的原始 logits，beta_eff 必须为前向保存的 FP32 head-major [B,H,T] 或 [H,T]。
- beta_eff 必须连续、与 q 同 device、rank/shape 匹配；仅传缓存而缺少原始 beta、缺失缓存、关闭 sigmoid 却传缓存均报错。
- db 的 dtype 始终跟随原始 beta，shape 始终为 token-major；融合时 db 是对原始 logits 的梯度。
- dg 跟随 raw_g dtype；dq/dk/dv 跟随 BF16 q/k/v；A_log/dt_bias 及梯度固定 FP32；dh0=None。
- 保存和重计算路径选择同一份有效 beta：融合时使用 beta_eff，关闭时使用 beta 的一次 head-major 转置。融合时不读取或转置原始 beta 的数值。
- 普通 sigmoid 导数因子为 beta_eff*(1-beta_eff)；allow_neg_eigval=True 时为 beta_eff*(1-beta_eff/2)，仅在融合模式允许。
- 缓存保持 head-major；四维变长支持 B=1。A5 尚不支持此实验总接口，保持明确拒绝，不扩大平台支持声明。

独立 Finalize Python/ACLNN 接口语义保持不变，融合时仍传 FP32 有效 beta、默认返回 FP32 db。
总接口内部的 Finalize model_layout 路径允许 FP32 有效 beta 直接写出 BF16 db；不在算子外增加 Cast。

以下历史优化路径说明针对 A5，并不覆盖上述实验 A2/A3 接口。

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
