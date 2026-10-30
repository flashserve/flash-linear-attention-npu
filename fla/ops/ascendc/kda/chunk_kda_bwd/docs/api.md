# KDA 反向总接口

## 版本变更：26.9.1 → main / 26.10.0

**本次反向总接口是非兼容变更。** 本文描述 main 分支本次变更后的接口；
26.10.0 后续回合同一变更，26.9.1 保持旧接口。升级时须同步调整模型接入，
不能只替换算子包后沿用原调用。

| 要点 | 26.9.1 旧接口 | main / 26.10.0 新接口 |
|---|---|---|
| 布局参数 | 无 layout，也无 input_layout/output_layout | 新增单一 layout，支持 BSND/BNSD/TND/NTD，默认 BNSD |
| q/k/v；dq/dk/dv | 固定 head-major（BNSD/NTD） | 随 layout；有成对 rstd 时，输入 q/k 固定为 head-major 的 q_hat/k_hat |
| do/raw_g/beta；dg/dbeta | 固定 head-major | 固定 token-major（BSND/TND），不随 layout |
| beta 与 sigmoid 反向 | beta 传有效值，无独立 beta_eff 和 sigmoid 反向标志 | 不融合时仍传有效 beta；融合时 beta 传原始 logits，另传 FP32 head-major 的 beta_eff；dbeta 对应原始 beta |
| dq/dk/dbeta/dg dtype | A2/A3 原总接口均为 FP32；A5 V2 的 dq/dk 为 BF16、dbeta 跟随 beta、dg 为 FP32 | dq/dk 跟随 q/k；dbeta/dg 分别跟随 beta/raw_g，支持 BF16/FP32 门控输入 |
| q/k/v dtype | 原总接口支持 FP16/BF16；A5 V2 为 BF16 | 统一只支持 BF16，dv 也为 BF16 |
| gate 模式 | use_gate_in_kernel 默认 False | 默认且仅支持 True，必须提供 raw_g 和 FP32 A_log；dt_bias 可选且固定 FP32 |
| 变长输入 | cu_seqlens 对应三维 head-major 张量 | 支持三维 NTD/TND，也支持 B=1 的四维 BNSD/BSND |
| A5 路由 | auto 按条件选择 legacy/V2 | auto/optimized 统一走 V2，legacy 拒绝新接口 |

A2/A3 支持融合 beta sigmoid 反向；**A5 仅预留接口**，当前必须保持
beta_eff=None、use_beta_sigmoid_in_kernel=False、allow_neg_eigval=False。
A_log/dt_bias 及其梯度在新接口中均固定 FP32；A5 旧 V2 接受 BF16 A_log 的调用也须调整。
前向接口和反向返回顺序不变，Python 绑定、ACLNN 与 OPP/kernel 必须配套升级。

迁移时重点检查：即使 layout 仍用默认 BNSD，do/raw_g/beta 与 dg/dbeta 也已改为
BSND；融合 sigmoid 时保存并传入原始 beta 与 beta_eff；删除接入层中已不再需要的梯度 Cast/Transpose。

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
- A5 V2 的 do 固定为 token-major，由总接口 ACLNN 直接传给 Prepare/Dhu；两个小算子通过内部可选标志按实际行距读取，
  独立小接口仍默认 head-major。三维变长为 TND，四维为 BSND，BNSD/NTD 的内部缓存布局不变。
  A2/A3 仍在总接口转置为 head-major 后复用。
- 普通 beta 仍在 ACLNN 内转一次 head-major，供 Recompute/Finalize 共用；融合时直接复用 beta_eff。
  raw_g/qkv 按实际布局直接读入片上，梯度按约定布局写出。
- Finalize 在片上转换 raw_g/beta，dg/db 直接按输入 dtype 写出。
  重计算分支仍可能先为 Recompute 创建 FP32 raw_g 临时张量。

## 接口升级

这是非兼容升级。Python 总接口仅使用 `layout`，不接收 input_layout/output_layout。
ACLNN 的 A2/A3 总接口与 A5 V2 均在 kRstdOptional 后按顺序接收
`betaEffOptional, layout, useBetaSigmoidInKernel, allowNegEigval`。
绑定、op-api、OPP/kernel 必须配套更新，不能与旧 V2 ABI 混用。

独立 Prepare/Recompute/Dhu/Finalize 的公开默认契约保持不变；
独立 Finalize 的 output_layout 仍控制全部五个梯度。总接口通过内部属性实现混合布局。
本次不修改前向。实现见 [A5 V2 ACLNN](../op_host/op_api/aclnn_chunk_kda_bwd_v2.h)，
流程见 [设计说明](design.md)。
