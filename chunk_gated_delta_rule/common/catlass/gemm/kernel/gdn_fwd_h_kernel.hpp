#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_scheduler_gdn_fwd_h.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "kernel_operator.h"
using namespace Catlass;

namespace Catlass::Gemm::Kernel {

template<
    class CubeScheduler, 
    class VecScheduler, 
    class BlockMmadWH,
    class BlockMmadKV,
    class EpilogueGDNFwdHVnew,
    class EpilogueGDNFwdHUpdate
>
class GDNFwdHKernel {
public:
    
    using ArchTag = Arch::AtlasA2;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = typename BlockMmadKV::ElementA;
    using ElementW = typename BlockMmadWH::ElementA;
    using ElementU = typename BlockMmadKV::ElementB;
    using ElementG = float;
    using ElementH = typename BlockMmadWH::ElementB;
    using ElementV = typename BlockMmadKV::ElementB;
    using ElementVWork = float;
    using ElementHWork = float;

    using L1TileShape = typename BlockMmadWH::L1TileShape;
    
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementH> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementH> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    
    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, 
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementH *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementH *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
        }
    }
    
    __aicore__ inline void Process() {

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource);
            BlockMmadKV blockMmadKV(resource);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * kHeadDim * PING_PONG_STAGES, vHeadDim);

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();
                // step 1: v_work = w @ h[i]
                GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetStage1Offsets();
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);
                if (cubeBlockScheduler.NeedProcessStage1()) {
                    int64_t cube1OffsetW = cube1Offsets.wOffset;
                    int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                    int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                    auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape {cube1Offsets.blockTokens, vHeadDim, kHeadDim};
                    auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    blockMmadWH.preSetFlags();
                    blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                    blockMmadWH.finalWaitFlags();
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);

                if (cubeBlockScheduler.iterId > 1) {
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                    GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetStage2Offsets();
                    if (cubeBlockScheduler.NeedProcessStage2()) {
                        // step 3: h[i+1] = k.T @ v_work
                        int64_t cube2OffsetK = cube2Offsets.wkOffset;
                        int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                        int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                        auto tensorK = tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                        auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                        auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube2Shape{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
                        auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                        auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                        auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                        blockMmadKV.preSetFlags();
                        blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                        blockMmadKV.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                }
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

        }

        if ASCEND_IS_AIV {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();
                // step 2:
                GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetStage1Offsets();
                // gmV = gmU - gmVWorkspace
                // g_buf = gmG[-1] - gmG
                // g_buf = exp(g_buf)
                // gmVWorkspace = g_buf * gmV
                if (vecBlockScheduler.NeedProcessStage1()) {
                    epilogueGDNFwdHVnew(
                        gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset], 
                        gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset], 
                        vec1Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube1Done
                    );
                } else {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);

                if (vecBlockScheduler.iterId > 1) {
                    GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetStage2Offsets();
                    if (vecBlockScheduler.NeedProcessStage2()) {
                        // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                        AscendC::GlobalTensor<ElementH> gmVec2Out = vec2Offsets.isFinalState ? gmFinalState : gmH;
                        EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
                        epilogueGDNFwdHUpdate(
                            gmVec2Out[vec2Offsets.hDstOffset],
                            gmG[vec2Offsets.gOffset],
                            gmH[vec2Offsets.hSrcOffset],
                            gmHWorkspace[vec2Offsets.hWorkOffset],
                            vec2Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube2Done
                        );
                    } else {
                        Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
                }
            }

        }
    }

};

}