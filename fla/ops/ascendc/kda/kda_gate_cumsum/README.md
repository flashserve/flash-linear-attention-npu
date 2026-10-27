# KdaGateCumsum 算子说明

> **[已弃用]** 本算子已弃用，将在2027年1月15日移除。请使用融合算子 `npu_chunk_kda_fwd` 替代。

`KdaGateCumsum` 用于 KDA (K-head Delta Attention) 中门控张量的 log-cumsum 计算，支持 safe gate 和 lower_bound 限制，可选 A_log 和 dt_bias 输入。
