# KDA 反向总接口

A2/A3 与 A5 V2 使用相同的单 `layout` 接口。返回顺序为
`(dq, dk, dv, db, dg, dh0, dA, dbias)`。

## 布局和 dtype

`layout` 支持 `BSND/BNSD/TND/NTD`，默认 `BNSD`。下表中的 head-major 为
BNSD/NTD，token-major 为 BSND/TND；beta/rstd 没有最后的 D 维。

| 输入 / 输出 | 布局 | dtype |
|---|---|---|
| q/k，无 rstd | 随 layout | BF16 |
| q/k，有成对 rstd | 前向保存的 q_hat/k_hat，固定 head-major | BF16 |
| v；dq/dk/dv | 随 layout | BF16 |
| do | 固定 token-major | BF16 |
| raw_g；dg | 固定 token-major | BF16/FP32；dg 跟随 raw_g |
| beta；db | 固定 token-major | BF16/FP32；db 跟随 beta |
| beta_eff | 可选，固定 head-major | FP32 |
| Aqk/Akk、w/qg/kg/v_new | 固定 head-major | BF16 |
| gk | 固定 head-major | FP32 |
| q_rstd/k_rstd | 成对传入，固定 head-major | FP32 |
| h | [B,Nc,H,128,128] 或 [Nc,H,128,128] | BF16 |
| A_log；dA | [H] | FP32 |
| dt_bias；dbias | 连续 H×128 个元素；Python 保留输入 shape | FP32；可不传，dbias 随之为 None |

所有张量须连续、同设备、使用普通 ND/NCHW/NCDHW/NCL 存储格式。
定长使用四维布局；变长可用三维 TND/NTD，也可用 B=1 的四维 BSND/BNSD。
变长元数据为 Host INT64，chunk_indices 按序列、chunk 排列；Python 压缩空段。

## beta 语义与平台差异

| 模式 | beta | beta_eff | db 对应的梯度 |
|---|---|---|---|
| 不融合 sigmoid | 模型提供的有效 beta | 必须为 None | 有效 beta |
| 融合 sigmoid（仅 A2/A3） | 原始 logits | 必传前向保存的 FP32 有效 beta | 原始 logits |

融合时只读取 beta 的描述信息以分配 db；计算使用 beta_eff。
普通 sigmoid 导数因子为 `beta_eff*(1-beta_eff)`；两倍 sigmoid 为
`beta_eff*(1-beta_eff/2)`，后者要求 `allow_neg_eigval=True`。

**A5 V2 仅预留 sigmoid 反向接口**：`beta_eff=None`、
`use_beta_sigmoid_in_kernel=False`、`allow_neg_eigval=False`，其他值在
Python 和 ACLNN 层拒绝。前反向须配套：A5 若需要 sigmoid，应在算子外完成前反向。

## 保存与重计算

- `disable_recompute=True`：传入前向保存的 gk/w/qg/kg/v_new/h。
- `disable_recompute=False`：以上六个缓存必须为 None，仍传 Aqk/Akk。
  Recompute/FwdH 恢复缓存，再执行 Prepare → Dhu → Finalize。
- 有 rstd 时，两条路径均使用前向保存的 q_hat/k_hat；Finalize 完成 L2Norm 反向。
- beta 在 ACLNN 内只转一次 head-major，供 Recompute/Finalize 共用；do/raw_g/qkv
  按实际布局直接读入片上，梯度按约定布局写出，不增加完整 token 张量的 Transpose。
- Finalize 在片上转换 raw_g/beta，dg/db 直接按输入 dtype 写出。
  重计算分支仍可能先为 Recompute 创建 FP32 raw_g 临时张量。

## A5 V2 范围

- Ascend950，Hq=Hv，K=V=128，chunk_size=64，B/H/T 为正。
- `implementation="auto"` 和 `"optimized"` 均走 V2；`"legacy"` 拒绝新契约，不自动回退。
- safe_gate/use_gate_in_kernel/use_exp2 必须为 True，`-5<=lower_bound<0`。
- state_v_first=False；不支持 initial_state/dht，dh0=None。
- **保留现有重计算限制：H≤256 且 H 为 8 的倍数**；保存路径不受该限制。
- 直接调用 V2 ACLNN 时须提供无空段的规范变长元数据。

## 接口升级

这是非兼容升级。Python 总接口使用 `layout`，不再接收 input_layout/output_layout。
ACLNN 的 A2/A3 总接口与 A5 V2 均在 kRstdOptional 后按顺序接收
`betaEffOptional, layout, useBetaSigmoidInKernel, allowNegEigval`。
绑定、op-api、OPP/kernel 必须配套更新，不能与旧 V2 ABI 混用。

独立 Prepare/Recompute/Dhu/Finalize 的公开默认契约保持不变；
独立 Finalize 的 output_layout 仍控制全部五个梯度。总接口通过内部属性实现混合布局。
本次不修改前向。实现见 [A5 V2 ACLNN](../op_host/op_api/aclnn_chunk_kda_bwd_v2.h)，
流程见 [设计说明](design.md)。
