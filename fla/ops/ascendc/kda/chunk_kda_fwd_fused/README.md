# chunk_kda_fwd_fused — KDA forward chunk32 大融合(issue #829)

编译器 demo(`kda-fusion/kda_fusion_demo_compiler/`,chunk32 原生)的忠实翻译,
双源 mix kernel:AIC 裸指令 + AIV MicroAPI,一次 launch 完成 prepare+forward+finalize。

## 契约(与生产语义对齐处)

- 输入:`q/k/v/g/beta` bf16,**[H,T,K] 头连续 3 维**(非生产 BSND);`a_log` fp32 [H]
  **原始对数门控,核内 exp**(入仓适配;demo 是 host 预 expf,两者 ulp 级差异);
  `dt_bias` fp32 [H*K]
- 输出:`o` bf16 [H,T,128];**不写 final_state**(demo 无此输出)
- 约束:K=V=128、T%128==0、单 batch;不支持 varlen / initial_state
- blockDim = H(一核一头);tiling key 恒 1
- kernel 每拍处理 128 token(内部 4 个 32 行 chunk 两级迭代)
- gate 语义:`lower_bound·sigmoid((g+dt_bias)·exp(A_log))`(生产 safe-gate 变体);
  beta 核内 sigmoid;q/k 核内 L2 归一

## 已知缺口(后续 Step 处理)

1. 布局:生产链路是 BSND [B,S,H,K],融合 kernel 要 [H,T,K];接入 dispatch 时需
   在 aclnn 层转置或扩展 kernel 寻址。
2. final_state:训练/续跑需要状态输出,demo 不写;需在 Step3 数值评审后补。
3. initial_state:demo 从零状态起算;decode 续跑不支持。
4. 数值:demo 相对 golden 有约 19% 偏差(compare-bed 有证据链),翻译原样保留,
   修复在 Step3 评审后另行批准。

## 验证

- 逐位对齐 demo:`runs/20261010-chunk32-fused/trans/verify_annot.sh`(净卡 5/5 biteq=1.0)
- 性能对齐 demo:同 harness wall 2.3~3.0ms vs demo 2.4~3.0ms @H96/T8192
