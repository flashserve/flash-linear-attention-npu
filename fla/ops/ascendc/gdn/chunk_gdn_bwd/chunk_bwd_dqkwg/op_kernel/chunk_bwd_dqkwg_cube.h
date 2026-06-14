/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_cube.h
 */

 #ifndef CHUNK_BWD_DQKWG_CUBE_H
 #define CHUNK_BWD_DQKWG_CUBE_H
 
 #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
     #define CATLASS_ARCH 3510
 #include "chunk_bwd_dqkwg_common.h"
 #include "catlass/arch/arch.hpp"
 #include "catlass/catlass.hpp"
 #include "catlass/gemm/block/block_mmad.hpp"
 #include "catlass/gemm/block/block_swizzle.hpp"
 #include "catlass/gemm/device/device_gemm.hpp"
 #include "catlass/gemm/dispatch_policy.hpp"
 #include "catlass/gemm/gemm_type.hpp"
 #include "catlass/layout/layout.hpp"
 #include "catlass/status.hpp"
 #include "tla/layout.hpp"
 #include "tla/tensor.hpp"
  using _0 = tla::Int<0>;
       using _1 = tla::Int<1>;
       using _2 = tla::Int<2>;
       using _4 = tla::Int<4>;
       using _8 = tla::Int<8>;
       using _16 = tla::Int<16>;
       using _32 = tla::Int<32>;
       using _64 = tla::Int<64>;
       using _128 = tla::Int<128>;
       using _256 = tla::Int<256>;
       using _512 = tla::Int<512>;
       using _1024 = tla::Int<1024>;
       using _2048 = tla::Int<2048>;
       using _4096 = tla::Int<4096>;
       using _8192 = tla::Int<8192>;
       using _16384 = tla::Int<16384>;
       using _32768 = tla::Int<32768>;
       using _65536 = tla::Int<65536>;
       #else
       #define CATLASS_ARCH 2201
       #include "chunk_bwd_dqkwg_common.h"
       #include "catlass/arch/arch.hpp"
       #include "catlass/catlass.hpp"
       #include "catlass/gemm/block/block_mmad.hpp"
       #include "catlass/gemm/block/block_swizzle.hpp"
       #include "catlass/gemm/device/device_gemm.hpp"
       #include "catlass/gemm/dispatch_policy.hpp"
       #include "catlass/gemm/gemm_type.hpp"
       #include "catlass/layout/layout.hpp"
       #include "catlass/status.hpp"
       #include "tla/layout.hpp"
       #include "tla/tensor.hpp"
       #endif
 using namespace tla;
 using namespace Catlass;
 using namespace AscendC;
 
namespace Catlass::Gemm::Kernel {

    /**
     * TileGemmDirect: Single-shot tile-level GEMM replacing BlockMmadTla.
     * Directly performs GM→L1→L0→Mmad→FixPipe→GM without tiling loops or pingpong buffers.
     * Optimal when all matrix dimensions fit in a single L0 tile (M≤128, N≤128, K≤128).
     *
     * Compared to BlockMmadTla<MmadPingpong>:
     *  - No multi-stage L1/L0 buffers (1 stage vs 2)
     *  - No nested tiling loops (kL1×mL0×kL0×nL0 all = 1 for our sizes)
     *  - Minimal event overhead: 4 Set + 4 Wait per GEMM vs ~20+ in Block
     *  - Half the L1/L0 memory usage (no pingpong double-buffering)
     */
    template <class ArchTag_, class ElementC_, class TileCopy_>
    struct TileGemmDirect {
        using ArchTag = ArchTag_;
        using TileCopy = TileCopy_;
        using ElementA = typename TileCopy::ElementA;
        using ElementB = typename TileCopy::ElementB;
        using ElementC = ElementC_;
        using ElementAccumulator = typename TileCopy::ElementAccumulator;
    
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmadType = Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1A>;
    
        // Single L0 tile dimensions (max supported by AtlasA2 Cube)
        static constexpr uint32_t TILE_M = 128;
        static constexpr uint32_t TILE_N = 128;
        static constexpr uint32_t TILE_K = 128;
    
        // Single-stage buffer sizes (no pingpong)
        static constexpr uint32_t L1A_TILE_SIZE = TILE_M * TILE_K * sizeof(ElementA);
        static constexpr uint32_t L1B_TILE_SIZE = TILE_K * TILE_N * sizeof(ElementB);
    
        // Static L1 layouts for tile-sized buffers
        static constexpr auto L1A_LAYOUT =
            tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
        static constexpr auto L1B_LAYOUT =
            tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});
    
        /// Construct: allocate single-stage buffers, initialize events
        CATLASS_DEVICE
        TileGemmDirect(Arch::Resource<ArchTag> &resource)
        {
            if ASCEND_IS_AIC {
                AscendC::SetMMLayoutTransform(true);
    
                // Allocate single-stage buffers from resource pools
                l1ABuf = resource.l1Buf.template GetBufferByByte<ElementA>(0);
                l1BBuf = resource.l1Buf.template GetBufferByByte<ElementB>(L1A_TILE_SIZE);
                l0ABuf = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
                l0BBuf = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
                l0CBuf = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    
                // Initialize events: buffers are initially free
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);  // L1 free for MTE2 write
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);     // L0A/L0B free for MTE1 write
            }
        }
    
        /// Destructor: drain pending events
        CATLASS_DEVICE
        ~TileGemmDirect()
        {
            if ASCEND_IS_AIC {
                AscendC::SetMMLayoutTransform(false);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            }
        }
    
        /// Perform a single-tile GEMM: C = A @ B
        template <class TensorA, class TensorB, class TensorC>
        CATLASS_DEVICE
        void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord const &actualShape)
        {
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
            using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
    #if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
            using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<TensorC>;
    #endif
    #if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
            using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<TensorC>;
    #endif
    
            CopyGmToL1A copyGmToL1A;
            CopyGmToL1B copyGmToL1B;
            CopyL1ToL0A copyL1ToL0A;
            CopyL1ToL0B copyL1ToL0B;
            CopyL0CToGm copyL0CToGm;
            TileMmadType tileMmad;
    
            uint32_t m = actualShape.m();
            uint32_t k = actualShape.k();
            uint32_t n = actualShape.n();
    
            // Avoid gemv mode on AtlasA2
            uint32_t mActual = m;
            if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                if (mActual == 1) mActual = 16;
            }
    
            // Create L1 tensor views (static tile-size layout)
            auto tensorL1A = tla::MakeTensor(l1ABuf, L1A_LAYOUT, Arch::PositionL1{});
            auto tensorL1B = tla::MakeTensor(l1BBuf, L1B_LAYOUT, Arch::PositionL1{});
    
            // ---- Step 1: GM → L1 (MTE2) ----
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);  // Wait L1 free
            copyGmToL1A(tensorL1A, tensorA);
            copyGmToL1B(tensorL1B, tensorB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);   // Signal L1 data ready
    
            // ---- Step 2: L1 → L0A/L0B (MTE1) ----
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);  // Wait L1 data ready
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);     // Wait L0A/L0B free
    
            auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(mActual, k);
            auto tensorL0A = tla::MakeTensor(l0ABuf, layoutL0A, Arch::PositionL0A{});
            auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, k));
            copyL1ToL0A(tensorL0A, tensorTileL1A);
    
            auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
            auto tensorL0B = tla::MakeTensor(l0BBuf, layoutL0B, Arch::PositionL0B{});
            auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
            copyL1ToL0B(tensorL0B, tensorTileL1B);
    
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);   // Signal L1 free
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);      // Signal L0 data ready
    
            // ---- Step 3: Mmad (Cube) ----
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    
            auto layoutL0C = tla::MakeLayoutL0C(mActual, n);
            auto tensorL0C = tla::MakeTensor(l0CBuf, layoutL0C, Arch::PositionL0C{});
    
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, n, k, true, 0b11);
    
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);      // Signal L0A/L0B free
    
            // ---- Step 4: L0C → GM (FixPipe, auto-triggered by unitFlag=0b11) ----
            copyL0CToGm(tensorC, tensorL0C, 0b11);
        }
    
    private:
        AscendC::LocalTensor<ElementA> l1ABuf;
        AscendC::LocalTensor<ElementB> l1BBuf;
        AscendC::LocalTensor<ElementA> l0ABuf;
        AscendC::LocalTensor<ElementB> l0BBuf;
        AscendC::LocalTensor<ElementAccumulator> l0CBuf;
    };
    
    /**
     * ChunkBwdDqkwg Cube Kernel 模板类
     * 
     * 模板参数:
     * - BlockMmadPart1_: Part 1 的矩阵乘法 (dv @ h^T -> dw)
     * - BlockMmadPart2_: Part 2 的矩阵乘法 (q @ k^T -> mm5)
     * - BlockMmadPart3_: Part 3 的矩阵乘法 (do @ v^T -> ds)
     * - BlockMmadPart4_: Part 4 的矩阵乘法 (do @ h^T -> dq)
     * - BlockMmadPart5_: Part 5 的矩阵乘法 (v @ dh -> dk)
     * - BlockMmadPart6_: Part 6 的矩阵乘法 (ds @ k -> dq+=)
     * - BlockMmadPart7_: Part 7 的矩阵乘法 (ds^T @ q -> dk+=)
     */
    template <
        class BlockMmadPart1_,  // dv @ h^T -> dw
        class BlockMmadPart2_,  // q @ k^T -> mm5
        class BlockMmadPart3_,  // do @ v^T -> ds
        class BlockMmadPart4_,  // do @ h^T -> dq
        class BlockMmadPart5_,  // v @ dh -> dk
        class BlockMmadPart6_,  // ds @ k -> dq
        class BlockMmadPart7_   // ds^T @ q -> dk
    >
    class ChunkBwdDqkwgTla {
    public:
        using BlockMmadPart1 = BlockMmadPart1_;
        using BlockMmadPart2 = BlockMmadPart2_;
        using BlockMmadPart3 = BlockMmadPart3_;
        using BlockMmadPart4 = BlockMmadPart4_;
        using BlockMmadPart5 = BlockMmadPart5_;
        using BlockMmadPart6 = BlockMmadPart6_;
        using BlockMmadPart7 = BlockMmadPart7_;
        
        using ArchTag = typename BlockMmadPart1::ArchTag;
        
        // 数据类型定义 (基于 Part 2: q @ k^T)
        using ElementA = typename BlockMmadPart2::ElementA;
        using ElementB = typename BlockMmadPart2::ElementB;
        using ElementC = typename BlockMmadPart2::ElementC;
        
        // Layout 定义
        using LayoutRowMajor = layout::RowMajor;
        using LayoutColMajor = layout::ColumnMajor;
        
        /// Parameters structure
        struct Params {
            // 输入指针
            GM_ADDR ptrQ;      // [B, H, T, K]
            GM_ADDR ptrK;      // [B, H, T, K]
            GM_ADDR ptrV;      // [B, H, T, V]
            GM_ADDR ptrG;      // [B, H, T]
            GM_ADDR ptrH;      // [B, num_chunks, H, K, V]
            GM_ADDR ptrDo;     // [B, H, T, V]
            GM_ADDR ptrDh;     // [B, num_chunks, H, K, V]
            GM_ADDR ptrDv;     // [B, H, T, V]
            //varlen
            GM_ADDR ptrCuSeqLens;
            GM_ADDR ptrChunkIndices;
            
            // 输出指针
            GM_ADDR ptrDq;     // [B, H, T, K]
            GM_ADDR ptrDk;     // [B, H, T, K]
            GM_ADDR ptrDw;     // [B, H, T, K]
            GM_ADDR ptrDg;     // [B, H, T]
            
            // Workspace 指针
            GM_ADDR ptrWorkspace;
            
            // Workspace 偏移
            uint64_t wsDwOffset;
            uint64_t wsDgLastOffset;
            uint64_t wsMm5Offset;
            uint64_t wsDsTempOffset;
            uint64_t wsMm6Offset;
            uint64_t wsMm7Offset;
            
            // 形状参数
            uint64_t B;// = CONST_B;
            uint64_t H;// = CONST_H;
            uint64_t T;// = CONST_T;
            uint64_t K;// = CONST_K;
            uint64_t V;// = CONST_V;
            uint64_t BT;// = CONST_BT;
            uint64_t numChunks;// = CONST_NUM_CHUNKS;
            // uint64_t chunkSize = 64;
            uint64_t isVarLen;
            
            // 其他参数
            float scale;
            
            CATLASS_DEVICE
            Params() {}
            
            CATLASS_DEVICE
            Params(
                GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
                GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices,
                GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
                GM_ADDR workspace,
                uint64_t B, uint64_t H, uint64_t T, uint64_t K, uint64_t V, uint64_t BT, uint64_t numChunks, 
                uint64_t wsDw, uint64_t wsDgLast, uint64_t wsMm5, uint64_t wsDsTemp, uint64_t wsMm6, uint64_t wsMm7,
                float s, uint64_t isVarLen
            ) : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h),
                ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLens(cu_seqlen), ptrChunkIndices(chunk_indices),
                ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg),
                ptrWorkspace(workspace),
                wsDwOffset(wsDw), wsDgLastOffset(wsDgLast),
                wsMm5Offset(wsMm5), wsDsTempOffset(wsDsTemp), wsMm6Offset(wsMm6), wsMm7Offset(wsMm7),
                scale(s), B(B), H(H), T(T), K(K), V(V), BT(BT), numChunks(numChunks), isVarLen(isVarLen) {}
        };
        
        CATLASS_DEVICE
        ChunkBwdDqkwgTla() {}
        
        template <int32_t CORE_TYPE = g_coreType>
        CATLASS_DEVICE
        void operator()(Params const &params);
        
        /**
         * AIC (Cube) 执行入口
         */
        template <>
        CATLASS_DEVICE
        void operator()<AscendC::AIC>(Params const &params) {
            Arch::Resource<ArchTag> resource;
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            
            uint32_t coreLoops = params.B * params.numChunks;
            
            // Layout 创建
            auto layoutBTxK = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.K);
            auto layoutKxBT = LayoutColMajor::MakeLayout<ElementA>(params.K, params.BT);
            auto layoutBTxV = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.V);
            auto layoutVxBT = LayoutColMajor::MakeLayout<ElementA>(params.V, params.BT);
            auto layoutBTxBT = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.BT);
            auto layoutBTxBT_T = LayoutColMajor::MakeLayout<ElementA>(params.BT, params.BT);
            auto layoutKxV = LayoutRowMajor::MakeLayout<ElementA>(params.K, params.V);
            auto layoutVxK = LayoutColMajor::MakeLayout<ElementA>(params.V, params.K);
    
            uint32_t bos = 0;
            uint32_t eos = 0;
    
            // ========== Part 1: b_dw = b_dv @ b_h^T ==========
            {
                BlockMmadPart1 blockMmadPart1(resource);
                auto layoutH = LayoutColMajor::MakeLayout<ElementA>(params.K, params.V);
                // printf("[cube]coreLoops %d, H %d\n",coreLoops,params.H);
                for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                                    params.BT, loopIdx, bos, eos);
                    uint32_t actual_chunk_len = eos-bos;
                    uint32_t bIdx = loopIdx / params.numChunks;
                    uint32_t chunkIdx = loopIdx % params.numChunks;
    
                    GemmCoord actualBlockShape{
                        static_cast<uint32_t>(actual_chunk_len),
                        static_cast<uint32_t>(params.K),
                        static_cast<uint32_t>(params.V)
                    };
    
                    for (uint32_t h = 0; h < params.H; h++) {
    
                        // 设置 GM 地址
                        // dv: [B, H, T, V] -> offset = ((b * H + h) * T + chunk * BT) * V
                        // uint64_t dvOffset = ((bIdx * params.H + h) * params.T + bos) * params.V;
                        uint64_t dvOffset = (h * params.T + bos) * params.V;
    
                        // h: [B, H, num_chunks, K, V] -> offset = ((b * numChunks + chunk) * H + h) * K * V
                        uint64_t hOffset = ((bIdx * params.H + h) * params.numChunks + chunkIdx) * params.K * params.V;
    
                        // dw output (workspace): [B, H, T, K]
                        // uint64_t dwOffset = ((bIdx * params.H + h) * params.T + bos) * params.K;
                        uint64_t dwOffset = (h * params.T + bos) * params.K;
    
                        GlobalTensor<ElementA> gmDv;
                        gmDv.SetGlobalBuffer((__gm__ ElementA *)params.ptrDv + dvOffset);
                        // gmDv.SetGlobalBuffer((__gm__ ElementA *)params.ptrDv);
                        GlobalTensor<ElementA> gmH;
                        gmH.SetGlobalBuffer((__gm__ ElementA *)params.ptrH + hOffset);
                        // gmH.SetGlobalBuffer((__gm__ ElementA *)params.ptrH);
                        GlobalTensor<ElementC> gmDw;
                        gmDw.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsDwOffset) + dwOffset);
    
                        auto tensorDv = tla::MakeTensor(gmDv, MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                        auto tensorH = tla::MakeTensor(gmH, MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});  // h^T
                        auto tensorDw = tla::MakeTensor(gmDw, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                        
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
    
                        auto tensorBlockDv = GetTile(tensorDv, tla::MakeCoord(0, 0), 
                                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), 
                                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                        blockMmadPart1(tensorBlockDv, tensorBlockH, tensorBlockDw, actualBlockShape);

                        AscendC::PipeBarrier<PIPE_FIX>();

                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);

                    }
                }
                // 最终同步(后移)
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
            }
            AscendC::SyncAll<false>();

            // ========== Part 2: mm5 = q @ k^T (纯 Cube) ==========
            {
                BlockMmadPart2 blockMmadPart2(resource);
                
                for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                                    params.BT, loopIdx, bos, eos);
                    uint32_t actual_chunk_len = eos-bos;
                    uint32_t bIdx = loopIdx / params.numChunks;
                    uint32_t chunkIdx = loopIdx % params.numChunks;
                    
                    GemmCoord actualBlockShape{
                        static_cast<uint32_t>(actual_chunk_len),
                        static_cast<uint32_t>(actual_chunk_len),
                        static_cast<uint32_t>(params.K)
                    };
                    
                    for (uint32_t h = 0; h < params.H; h++) {
                        // q, k: [B, H, T, K]
                        uint64_t qkOffset = (h * params.T + bos) * params.K;
                        // mm5: workspace [B, H, T, BT]
                        uint64_t mm5Offset = (h * params.T + bos) * params.BT;
                        
                        GlobalTensor<ElementA> gmQ;
                        gmQ.SetGlobalBuffer((__gm__ ElementA *)params.ptrQ + qkOffset);
                        GlobalTensor<ElementA> gmK;
                        gmK.SetGlobalBuffer((__gm__ ElementA *)params.ptrK + qkOffset);
                        GlobalTensor<ElementC> gmMm5;
                        gmMm5.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsMm5Offset) + mm5Offset);
    
                        auto tensorQ = tla::MakeTensor(gmQ, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                        auto tensorK = tla::MakeTensor(gmK, MakeLayoutFromTag(layoutKxBT), Arch::PositionGM{});  // k^T
                        auto tensorMm5 = tla::MakeTensor(gmMm5, MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});
    
                        auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), 
                                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), 
                                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockMm5 = GetTile(tensorMm5, tla::MakeCoord(0, 0), 
                                                       tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        
                        blockMmadPart2(tensorBlockQ, tensorBlockK, tensorBlockMm5, actualBlockShape);
    
                    }
                }
    
            }
            AscendC::SyncAll<false>();

            // ========== Part 3: b_ds = b_do @ b_v^T ==========
            {
                // AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                
                BlockMmadPart3 blockMmadPart3(resource);
                
                for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                                    params.BT, loopIdx, bos, eos);
                    uint32_t actual_chunk_len = eos-bos;
                    uint32_t bIdx = loopIdx / params.numChunks;
                    uint32_t chunkIdx = loopIdx % params.numChunks;
                    
                    GemmCoord actualBlockShape{
                        static_cast<uint32_t>(actual_chunk_len),
                        static_cast<uint32_t>(actual_chunk_len),
                        static_cast<uint32_t>(params.V)
                    };
                    
                    for (uint32_t h = 0; h < params.H; h++) {
                        // do, v: [B, H, T, V]
                        // uint64_t dvOffset = ((bIdx * params.H + h) * params.T + bos) * params.V;
                        uint64_t dvOffset = (h * params.T + bos) * params.V;
                        // ds_temp: workspace [B, H, T, BT]
                        // uint64_t dsOffset = ((bIdx * params.H + h) * params.T + bos) * params.BT;
                        uint64_t dsOffset = (h * params.T + bos) * params.BT;
                        
                        GlobalTensor<ElementA> gmDo;
                        gmDo.SetGlobalBuffer((__gm__ ElementA *)params.ptrDo + dvOffset);
                        GlobalTensor<ElementA> gmV;
                        gmV.SetGlobalBuffer((__gm__ ElementA *)params.ptrV + dvOffset);
                        GlobalTensor<ElementC> gmDs;
                        gmDs.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsDsTempOffset) + dsOffset);
                        
                        auto tensorDo = tla::MakeTensor(gmDo, MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                        auto tensorV = tla::MakeTensor(gmV, MakeLayoutFromTag(layoutVxBT), Arch::PositionGM{});  // v^T
                        auto tensorDs = tla::MakeTensor(gmDs, MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});
                        
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                        // AscendC::PipeBarrier<PIPE_MTE2>();
                        auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0), 
                                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), 
                                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockDs = GetTile(tensorDs, tla::MakeCoord(0, 0), 
                                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        
                        blockMmadPart3(tensorBlockDo, tensorBlockV, tensorBlockDs, actualBlockShape);
    
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
                    }
                }
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
            }
            AscendC::SyncAll<false>();

            // ========== Fused Part 4+6: b_dq = b_do @ b_h^T, then mm6 = b_ds_temp @ b_k ==========
            // 每次迭代先计算 dq, 再计算 mm6, Vector 端在 UB 中完成 dq += mm6 避免 GM 往返
            {
                for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                                    params.BT, loopIdx, bos, eos);
                    uint32_t actual_chunk_len = eos-bos;
                    uint32_t bIdx = loopIdx / params.numChunks;
                    uint32_t chunkIdx = loopIdx % params.numChunks;
    
                    for (uint32_t h = 0; h < params.H; h++) {
                        // --- Sub-part 4: dq = do @ h^T ---
                        {
                            GemmCoord actualBlockShape{
                                static_cast<uint32_t>(actual_chunk_len),
                                static_cast<uint32_t>(params.K),
                                static_cast<uint32_t>(params.V)
                            };
                            uint64_t doOffset = (h * params.T + bos) * params.V;
                            uint64_t hOffset = ((bIdx * params.H + h) * params.numChunks + chunkIdx) * params.K * params.V;
                            uint64_t dqOffset = (h * params.T + bos) * params.K;
    
                            GlobalTensor<ElementA> gmDo;
                            gmDo.SetGlobalBuffer((__gm__ ElementA *)params.ptrDo + doOffset);
                            GlobalTensor<ElementA> gmH;
                            gmH.SetGlobalBuffer((__gm__ ElementA *)params.ptrH + hOffset);
                            GlobalTensor<ElementC> gmDq;
                            gmDq.SetGlobalBuffer((__gm__ ElementC *)params.ptrDq + dqOffset);
    
                            auto tensorDo = tla::MakeTensor(gmDo, MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                            auto tensorH = tla::MakeTensor(gmH, MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                            auto tensorDq = tla::MakeTensor(gmDq, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
    
                            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
    
                            auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0),
                                                          tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                            auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                                         tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                            auto tensorBlockDq = GetTile(tensorDq, tla::MakeCoord(0, 0),
                                                          tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
    
                            BlockMmadPart4 blockMmadPart4(resource);
                            blockMmadPart4(tensorBlockDo, tensorBlockH, tensorBlockDq, actualBlockShape);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
    
                        // --- Sub-part 6: mm6 = ds_temp @ k ---
                        {
                            GemmCoord actualBlockShape{
                                static_cast<uint32_t>(actual_chunk_len),
                                static_cast<uint32_t>(params.K),
                                static_cast<uint32_t>(actual_chunk_len)
                            };
                            uint64_t dsOffset = (h * params.T + bos) * params.BT;
                            uint64_t kOffset = (h * params.T + bos) * params.K;
    
                            GlobalTensor<ElementA> gmDsTemp;
                            gmDsTemp.SetGlobalBuffer((__gm__ ElementA *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsDsTempOffset) + dsOffset);
                            GlobalTensor<ElementA> gmK;
                            gmK.SetGlobalBuffer((__gm__ ElementA *)params.ptrK + kOffset);
                            GlobalTensor<ElementC> gmMm6;
                            gmMm6.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsMm6Offset) + kOffset);
    
                            auto tensorDsTemp = tla::MakeTensor(gmDsTemp, MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});
                            auto tensorK = tla::MakeTensor(gmK, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                            auto tensorMm6 = tla::MakeTensor(gmMm6, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
    
                            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
    
                            auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                              tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                            auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                         tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                            auto tensorBlockMm6 = GetTile(tensorMm6, tla::MakeCoord(0, 0),
                                                           tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
    
                            BlockMmadPart6 blockMmadPart6(resource);
                            blockMmadPart6(tensorBlockDsTemp, tensorBlockK, tensorBlockMm6, actualBlockShape);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
                    }
                }
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
            }
            AscendC::SyncAll<false>();

            // ========== Fused Part 5+7: b_dk = b_v @ b_dh, then mm7 = b_ds_temp^T @ b_q ==========
            {
                for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                                    params.BT, loopIdx, bos, eos);
                    uint32_t actual_chunk_len = eos-bos;
                    uint32_t bIdx = loopIdx / params.numChunks;
                    uint32_t chunkIdx = loopIdx % params.numChunks;
    
                    for (uint32_t h = 0; h < params.H; h++) {
                        // --- Sub-part 5: dk = v @ dh ---
                        {
                            GemmCoord actualBlockShape{
                                static_cast<uint32_t>(actual_chunk_len),
                                static_cast<uint32_t>(params.K),
                                static_cast<uint32_t>(params.V)
                            };
                            uint64_t vOffset = (h * params.T + bos) * params.V;
                            uint64_t dhOffset = ((bIdx * params.H + h) * params.numChunks + chunkIdx) * params.K * params.V;
                            uint64_t dkOffset = (h * params.T + bos) * params.K;
    
                            GlobalTensor<ElementA> gmV;
                            gmV.SetGlobalBuffer((__gm__ ElementA *)params.ptrV + vOffset);
                            GlobalTensor<ElementA> gmDh;
                            gmDh.SetGlobalBuffer((__gm__ ElementA *)params.ptrDh + dhOffset);
                            GlobalTensor<ElementC> gmDk;
                            gmDk.SetGlobalBuffer((__gm__ ElementC *)params.ptrDk + dkOffset);
    
                            auto tensorV = tla::MakeTensor(gmV, MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                            auto tensorDh = tla::MakeTensor(gmDh, MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                            auto tensorDk = tla::MakeTensor(gmDk, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
    
                            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
    
                            auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                                         tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                            auto tensorBlockDh = GetTile(tensorDh, tla::MakeCoord(0, 0),
                                                          tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                            auto tensorBlockDk = GetTile(tensorDk, tla::MakeCoord(0, 0),
                                                          tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
    
                            BlockMmadPart5 blockMmadPart5(resource);
                            blockMmadPart5(tensorBlockV, tensorBlockDh, tensorBlockDk, actualBlockShape);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
    
                        // --- Sub-part 7: mm7 = ds_temp^T @ q ---
                        {
                            GemmCoord actualBlockShape{
                                static_cast<uint32_t>(actual_chunk_len),
                                static_cast<uint32_t>(params.K),
                                static_cast<uint32_t>(actual_chunk_len)
                            };
                            uint64_t dsOffset = (h * params.T + bos) * params.BT;
                            uint64_t qOffset = (h * params.T + bos) * params.K;
    
                            GlobalTensor<ElementA> gmDsTemp;
                            gmDsTemp.SetGlobalBuffer((__gm__ ElementA *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsDsTempOffset) + dsOffset);
                            GlobalTensor<ElementA> gmQ;
                            gmQ.SetGlobalBuffer((__gm__ ElementA *)params.ptrQ + qOffset);
                            GlobalTensor<ElementC> gmMm7;
                            gmMm7.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t*)params.ptrWorkspace + params.wsMm7Offset) + qOffset);
    
                            auto tensorDsTemp = tla::MakeTensor(gmDsTemp, MakeLayoutFromTag(layoutBTxBT_T), Arch::PositionGM{});
                            auto tensorQ = tla::MakeTensor(gmQ, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                            auto tensorMm7 = tla::MakeTensor(gmMm7, MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
    
                            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
    
                            auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                              tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                            auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                                         tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                            auto tensorBlockMm7 = GetTile(tensorMm7, tla::MakeCoord(0, 0),
                                                           tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
    
                            BlockMmadPart7 blockMmadPart7(resource);
                            blockMmadPart7(tensorBlockDsTemp, tensorBlockQ, tensorBlockMm7, actualBlockShape);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
                    }
                }
    
                // 最终同步
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
            }
    
        }
    };
    
    } // namespace Catlass::Gemm::Kernel
    
    /**
     * Cube Process 类 (AIC 端入口)
     */
    template <typename DataType, typename GType>
    class ChunkBwdDqkwgCubeProcess {
    public:
        __aicore__ inline ChunkBwdDqkwgCubeProcess(
            GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
            GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices,
            GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
            GM_ADDR workspace
        ) : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h),
            ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen), ptrChunkIndices(chunk_indices),
            ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg),
            ptrWorkspace(workspace) {}
        
        __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling);
        __aicore__ inline void Process();
        
    private:
        // 输入输出指针
        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrG;
        GM_ADDR ptrH;
        GM_ADDR ptrDo;
        GM_ADDR ptrDh;
        GM_ADDR ptrDv;
        GM_ADDR ptrCuSeqLen;
        GM_ADDR ptrChunkIndices;
        GM_ADDR ptrDq;
        GM_ADDR ptrDk;
        GM_ADDR ptrDw;
        GM_ADDR ptrDg;
        GM_ADDR ptrWorkspace;
        
        // Tiling 参数
        uint64_t B = CONST_B;
        uint64_t H = CONST_H;
        uint64_t T = CONST_T;
        uint64_t K = CONST_K;
        uint64_t V = CONST_V;
        uint64_t BT = CONST_BT;
        uint64_t numChunks = CONST_NUM_CHUNKS;
        float scale;
        uint64_t isVarLen;
        
        // Workspace 偏移
        uint64_t wsDwOffset;
        uint64_t wsDgLastOffset;
        uint64_t wsMm5Offset;
        uint64_t wsDsTempOffset;
        uint64_t wsMm6Offset;
        uint64_t wsMm7Offset;
    };
    
    template <typename DataType, typename GType>
    __aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling) {
    
    /*
        // 设置 workspace 偏移
        // wsDwOffset = 0;
        wsDgLastOffset = 0;//B * H * numChunks * sizeof(float);
        // wsDgLastOffset = ((wsDgLastOffset + 31) / 32) * 32;
        wsMm5Offset = wsDgLastOffset + ((B * H * numChunks * sizeof(float) + 31) / 32) * 32;
        // wsMm5Offset = wsMm5Offset;
        wsDsTempOffset = wsMm5Offset + B * H * T * BT * sizeof(DataType);
        wsMm6Offset = wsDsTempOffset + B * H * T * BT * sizeof(DataType);
        wsMm7Offset = wsMm6Offset + B * H * T * K * sizeof(DataType);
    */
        // printf("[cube] wsDgLastOffset %d,wsMm5Offset %d,wsDsTempOffset %d, wsMm6Offset %d wsMm7Offset %d\n",wsDgLastOffset,wsMm5Offset,wsDsTempOffset,wsMm6Offset, wsMm7Offset);
    ////////////////////tiling/////
        scale = tiling.scale;
        B = tiling.B;
        H = tiling.H;
        T = tiling.T;
        K = tiling.K;
        V = tiling.V;
        BT = tiling.BT;
        numChunks = tiling.numChunks;
        wsDwOffset = tiling.wsDwOffset;
        wsDgLastOffset = tiling.wsDgLastOffset;
        // wsDgLastOffset = ((wsDgLastOffset + 31) / 32) * 32;
        wsMm5Offset = tiling.wsMm5Offset;
        // wsMm5Offset = wsMm5Offset;
        wsDsTempOffset = tiling.wsDsTempOffset;
        wsMm6Offset = tiling.wsMm6Offset;
        wsMm7Offset = tiling.wsMm7Offset;
        isVarLen = tiling.isVarLen;
    // printf("[cube] DTYPE_G %d, float %d\n",sizeof(DTYPE_G),sizeof(float));
    }
    
    template <typename DataType, typename GType>
    __aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Process() {
        using namespace Catlass;
        using namespace Catlass::Gemm;
        
        // Layout 类型定义
        using LayoutRowMajor = layout::RowMajor;
        using LayoutColMajor = layout::ColumnMajor;
        
        // 架构和策略定义
        using ArchTag = Arch::AtlasA2;
    
        // Part 1: dv @ h^T -> dw  [BT, V] @ [V, K] -> [BT, K]
        using TileCopyPart1 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart1 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart1>;
        
        // Part 2: q @ k^T -> mm5  [BT, K] @ [K, BT] -> [BT, BT]
        using TileCopyPart2 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart2 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart2>;
        
        // Part 3: do @ v^T -> ds  [BT, V] @ [V, BT] -> [BT, BT]
        using TileCopyPart3 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart3 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart3>;
        
        // Part 4: do @ h^T -> dq  [BT, V] @ [V, K] -> [BT, K]
        using TileCopyPart4 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart4 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart4>;
        
        // Part 5: v @ dh -> dk  [BT, V] @ [V, K] -> [BT, K]
        using TileCopyPart5 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart5 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart5>;
        
        // Part 6: ds @ k -> dq+=  [BT, BT] @ [BT, K] -> [BT, K]
        using TileCopyPart6 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutRowMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart6 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart6>;
        
        // Part 7: ds^T @ q -> dk+=  [BT, BT] @ [BT, K] -> [BT, K]
        using TileCopyPart7 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutColMajor, DataType, LayoutRowMajor, DataType, LayoutRowMajor>;
        using BlockMmadPart7 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart7>;
        
        // Kernel 实例
        using MatmulKernel = Kernel::ChunkBwdDqkwgTla<
            BlockMmadPart1, BlockMmadPart2, BlockMmadPart3, BlockMmadPart4,
            BlockMmadPart5, BlockMmadPart6, BlockMmadPart7
        >;
        
        MatmulKernel kernel;
        typename MatmulKernel::Params params(
            ptrQ, ptrK, ptrV, ptrG, ptrH,
            ptrDo, ptrDh, ptrDv, ptrCuSeqLen, ptrChunkIndices,
            ptrDq, ptrDk, ptrDw, ptrDg,
            ptrWorkspace, B, H, T, K, V, BT, numChunks,
            wsDwOffset, wsDgLastOffset, wsMm5Offset, wsDsTempOffset, wsMm6Offset, wsMm7Offset,
            scale, isVarLen
        );
        
        kernel(params);
    }

    #endif  // CHUNK_BWD_DQKWG_CUBE_H
