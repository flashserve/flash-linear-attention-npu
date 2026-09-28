# KDA 反向优化接口

## 调用方式

沿用 Python 入口 `chunk_kda_bwd`，新增三个可选参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
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
| raw_g / A_log | BF16 或 FP32，执行器按需转为 FP32 |
| dt_bias | 可选 FP32，共 H×128 个元素 |
| q_rstd / k_rstd | 成对提供，FP32，形状为 Token 形状去掉最后一维 |
| dq/dk/dv | BF16 |
| dg/dA/dbias | FP32；未传 dt_bias 时 dbias 为 None |
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
