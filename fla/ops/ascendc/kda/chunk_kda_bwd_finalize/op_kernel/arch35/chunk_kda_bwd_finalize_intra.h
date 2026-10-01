/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_INTRA_H
#define CHUNK_KDA_BWD_FINALIZE_INTRA_H

// 本文件在 KDA 命名空间内、寄存器读写辅助函数之后包含。
// 同一特征的指数平移在 GEMM 两侧抵消，用于控制分带操作数的数值范围。

// safe-gate 输入是非正 gate 增量的 chunk 累积和，各特征沿行不增。
// 因此当前带首末行就是最大/最小值，中心取二者均值，无须逐行求极值。
__simd_vf__ inline void FinalizeIntraCenterVF(
    __ubuf__ float *g, __ubuf__ float *center, uint16_t rows)
{
    // 1. 分两组覆盖 128 个特征。
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t half = 0; half < 2; ++half) {
        RegTensor<float> first, last, value;

        // 2. 读取当前带首末行，center=(first+last)/2，保存供两侧指数平移使用。
        LoadAlign(first, g + half * 64);
        LoadAlign(last, g + (rows - 1U) * 128 + half * 64);
        Add(value, first, last, mask);
        Muls(value, value, 0.5f, mask);
        StoreAlign(center + half * 64, value, mask);
    }
}

#endif
