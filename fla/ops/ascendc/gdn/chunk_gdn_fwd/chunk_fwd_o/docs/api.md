# ChunkFwdO 接口与 A6 适配范围

本次新增 ascend960dt（DAV_9201）实现，公开签名、默认值和 ABI 沿用 [算子说明](../README.md) 及 op_host/op_api/aclnn_chunk_fwd_o.h，不增加参数。

| 路径 | 输出 | 主 dtype / g dtype | K / V / BT | 属性 |
| --- | --- | --- | --- | --- |
| A5/A6 优化 | BSND / TND | BF16 / BF16 或 FP32 | 128 / 128 / 64 | use_exp2、state_v_first 均支持 true/false；HV/HK=1..16 |
| A5/A6 兼容 | BNSD / NTD | BF16 或 FP16 / FP32 或与主 dtype 相同 | 128 / 128 或 256 / 64 或 128 | use_exp2=false、state_v_first=false |

输入 q/k=[B,HK,T,K]，v=[B,HV,T,V]，g=[B,HV,T]，HV 是 HK 的整数倍。
h 定长为 [B,ceil(T/BT),HV,K,V]，变长为 [1,totalChunks,HV,K,V]；state_v_first=true 时末两维交换。
输出分别为 [B,HV,T,V]、[B,T,HV,V]、[HV,T,V]、[T,HV,V]，与 layout 对应。
变长同时提供 INT64 cu_seqlens 和扁平化 [seq_id,local_chunk_id] chunk_offsets，B=1。
输出 dtype 与 v 相同；scale 显式传入。空输入沿用 ACLNN 原有返回约定，不扩大 tiling 接受范围。

优化路径 A=BF16(causal((Q@K.T)*E(g_i-g_j)))，O=BF16((E(g_i)*(Q@H)+A@V)*scale)。
E 在 use_exp2=true 时为 exp(x*ln2)，否则为 exp(x)。兼容路径保留 A5 的 min(g_i-g_j,0) 后 exp、cast 和 scale 顺序。
A6 支持状态为源码适配待设备验收；A5/A2/A3 原有行为保持。
