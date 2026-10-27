# PrepareWyReprBwd 算子说明

> **[已弃用]** 本算子已弃用，将在后续版本中移除。请使用融合算子 `npu_chunk_gated_delta_rule_bwd` 替代。

`PrepareWyReprBwd` 用于 Gated Delta Rule 反向传播中 WY 表示的预处理，根据 Key、Value、Beta、A 矩阵以及反向传入的 dw、du 梯度和 Gate，计算 Key（k）、Value（v）、Beta 和 Gate 的梯度，并内部产生 dA。
