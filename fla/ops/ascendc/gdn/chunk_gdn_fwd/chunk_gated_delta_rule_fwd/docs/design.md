# GDN 前向归一化结果导出

规则版本 V2。本次仅变更 [API 返回合同](api.md)，复用现有 prepare → H → O 计算 DAG。

1. Python 开启归一化时分配 q_hat/k_hat 及 FP32 `[B,Hk,T]` rstd，将 descriptor 传给已有 ACLNN 输出槽位。
2. prepare 产生的内部 qHat/kHat 继续供 H/O 消费，并按输入 layout 导出。内部 rstd 原本就是 `[B,Hk,T]`，导出时直接 ViewCopy，取消 sequence-major 转置。
3. Python 关闭归一化时返回输入 q/k 对象，ACLNN 四个槽位保持空，不增加 kernel、分配或复制。
4. workspace、UB/L1 物理布局和 Stage 同步均不变。内部 tensor 仍由 executor 管理，公开输出独立持有；关闭归一化的 hats 由原输入存储持有。
5. 返回值追加到原六项之后，更新全部仓内解包调用。前后向示例将 hats 和 rstd 直接传给反向。

回归覆盖四种 layout、归一化开关、训练/推理、尾块、GVA、变长以及相同输入下 prepare 导出结果的一致性。
