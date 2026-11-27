#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace JyShallowNewton {
using namespace Catlass;
constexpr uint32_t D = 128;
constexpr uint32_t ELEMENTS = D * D;
constexpr uint32_t MATRIX_BYTES = ELEMENTS * sizeof(float);
constexpr uint32_t TILE_ROWS = 64;
constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * D;
constexpr uint32_t TILE_BYTES = TILE_ELEMENTS * sizeof(float);
constexpr uint32_t BIG_TILE_ROWS = D;
constexpr uint32_t BIG_TILE_ELEMENTS = BIG_TILE_ROWS * D;
constexpr uint32_t BIG_TILE_BYTES = BIG_TILE_ELEMENTS * sizeof(float);
constexpr uint16_t X_READY = 0;
constexpr uint16_t P_READY = 1;
constexpr uint16_t R_READY = 2;
constexpr uint16_t Q_DONE = 3;

using MatrixType = Gemm::GemmType<float, layout::RowMajor>;
using Copies = Gemm::Tile::TileCopy<Arch::AtlasA2, MatrixType, MatrixType, MatrixType>;
using L1Layout = typename Copies::CopyL1ToL0A::LayoutSrc;
using L0ALayout = typename Copies::CopyL1ToL0A::LayoutDst;
using L0BLayout = typename Copies::CopyL1ToL0B::LayoutDst;
static_assert(std::is_same_v<L1Layout, typename Copies::CopyL1ToL0B::LayoutSrc>,
    "Resident X needs identical L1 source layout for both operand positions");

// Explicit serial panel schedule using the pinned Catlass tile implementations,
// not the high-level Matmul/KFC service. No hidden GM/UB workspace or flags.
// All L1 tensors use the same FP32 zN layout, allowing X to be either operand.
template<uint32_t K_TILE>
class ResidentGemm {
public:
    CATLASS_DEVICE explicit ResidentGemm(Arch::Resource<Arch::AtlasA2>& resource)
        : l0a_(resource.l0ABuf.template GetBufferByByte<float>(0)),
          l0b_(resource.l0BBuf.template GetBufferByByte<float>(0)),
          l0c_(resource.l0CBuf.template GetBufferByByte<float>(0))
    {
        static_assert(D * K_TILE * sizeof(float) <= Arch::AtlasA2::L0A_SIZE);
        static_assert(D * K_TILE * sizeof(float) <= Arch::AtlasA2::L0B_SIZE);
        static_assert(MATRIX_BYTES <= Arch::AtlasA2::L0C_SIZE);
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        AscendC::SetHF32TransMode(AscendC::HF32TransMode::NEAREST_EVEN);
    }

    CATLASS_DEVICE void Load(AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src)
    {
        // Previous L1 readers are drained before any bank is overwritten.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        copyGm_(dst, src, L1Layout::template MakeLayout<float>(D, D), layout::RowMajor{D, D});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
    }

    CATLASS_DEVICE void Multiply(AscendC::LocalTensor<float> a,
        AscendC::LocalTensor<float> b, AscendC::GlobalTensor<float> output)
    {
        const auto l1Layout = L1Layout::template MakeLayout<float>(D, D);
        const auto aLayout = L0ALayout::template MakeLayout<float>(D, K_TILE);
        const auto bLayout = L0BLayout::template MakeLayout<float>(K_TILE, D);
        const auto cLayout = layout::zN::MakeLayoutInL0C(MakeCoord(D, D));
        // Previous Fixpipe must stop reading L0C before the next initialization.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        for (uint32_t k = 0; k < D; k += K_TILE) {
            // Single L0 panel: complete previous Mmad before reusing A/B.
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            copyA_(l0a_, a[l1Layout.GetOffset(MatrixCoord{0, k})], aLayout, l1Layout);
            copyB_(l0b_, b[l1Layout.GetOffset(MatrixCoord{k, 0})], bLayout, l1Layout);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID4);
            // FP32 operands, FP32 accumulator, increasing K panels; no bias,
            // no partial GM accumulation, no HF32. Exactness remains a gate.
            mad_(l0c_, l0a_, l0b_, D, D, K_TILE, k == 0, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID4);
        copyOut_(output, l0c_, layout::RowMajor{D, D}, cLayout);
    }

    CATLASS_DEVICE void DrainStore()
    {
        // FIX -> MTE2 covers Xnext GM feedback, including the GM load itself.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID4);
    }

private:
    AscendC::LocalTensor<float> l0a_, l0b_, l0c_;
    typename Copies::CopyGmToL1A copyGm_;
    typename Copies::CopyL1ToL0A copyA_;
    typename Copies::CopyL1ToL0B copyB_;
    typename Copies::CopyL0CToGm copyOut_;
    Gemm::Tile::TileMmad<Arch::AtlasA2, MatrixType, MatrixType, void> mad_;
};
// Private K32 specialization; the original generic above remains unchanged.
class ResidentGemmK32Pipeline {
private:
    static constexpr uint32_t K_TILE = 32;
    static constexpr uint32_t BANK_BYTES = D * K_TILE * sizeof(float);
public:
    CATLASS_DEVICE explicit ResidentGemmK32Pipeline(Arch::Resource<Arch::AtlasA2>& resource)
        : l0a_(resource.l0ABuf.template GetBufferByByte<float>(0)),
          l0b_(resource.l0BBuf.template GetBufferByByte<float>(0)),
          l0c_(resource.l0CBuf.template GetBufferByByte<float>(0)),
          l0aNext_(resource.l0ABuf.template GetBufferByByte<float>(BANK_BYTES)),
          l0bNext_(resource.l0BBuf.template GetBufferByByte<float>(BANK_BYTES))
    {
        static_assert(D * K_TILE * sizeof(float) <= Arch::AtlasA2::L0A_SIZE);
        static_assert(D * K_TILE * sizeof(float) <= Arch::AtlasA2::L0B_SIZE);
        static_assert(MATRIX_BYTES <= Arch::AtlasA2::L0C_SIZE);
        static_assert(BANK_BYTES == 16384);
        static_assert(2 * BANK_BYTES <= Arch::AtlasA2::L0A_SIZE);
        static_assert(2 * BANK_BYTES <= Arch::AtlasA2::L0B_SIZE);
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        AscendC::SetHF32TransMode(AscendC::HF32TransMode::NEAREST_EVEN);
    }

    CATLASS_DEVICE void Load(AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src)
    {
        // Previous L1 readers are drained before any bank is overwritten.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        copyGm_(dst, src, L1Layout::template MakeLayout<float>(D, D), layout::RowMajor{D, D});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
    }

    CATLASS_DEVICE void Multiply(AscendC::LocalTensor<float> a,
        AscendC::LocalTensor<float> b, AscendC::GlobalTensor<float> output)
    {
        const auto l1Layout = L1Layout::template MakeLayout<float>(D, D);
        const auto aLayout = L0ALayout::template MakeLayout<float>(D, K_TILE);
        const auto bLayout = L0BLayout::template MakeLayout<float>(K_TILE, D);
        const auto cLayout = layout::zN::MakeLayoutInL0C(MakeCoord(D, D));
        // Previous Fixpipe must stop reading L0C before the next initialization.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        // Each call starts with exactly one binary FREE token per bank.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        Preload(a, b, l1Layout, aLayout, bLayout, 0);
        for (uint32_t k = 0; k < D; k += K_TILE) {
            if (k + K_TILE < D) {
                Preload(a, b, l1Layout, aLayout, bLayout, k + K_TILE);
            }
            const bool bank1 = ((k / K_TILE) & 1u) != 0;
            const int32_t event = bank1 ? EVENT_ID1 : EVENT_ID0;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event);
            auto currentA = bank1 ? l0aNext_ : l0a_;
            auto currentB = bank1 ? l0bNext_ : l0b_;
            // Same four ascending FP32 Mmad calls, init only at k==0.
            mad_(l0c_, currentA, currentB, D, D, K_TILE, k == 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // Gate PIPE_M on completion of both asynchronous MTE1 drains before
        // the next call can reseed FREE. All READY generations are consumed.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID4);
        copyOut_(output, l0c_, layout::RowMajor{D, D}, cLayout);
    }

    // The injection is the initial value for both directional scans.
    CATLASS_DEVICE void StoreAgain(AscendC::GlobalTensor<float> output) {
        copyOut_(output, l0c_, layout::RowMajor{D,D},
                 layout::zN::MakeLayoutInL0C(MakeCoord(D,D)));
    }

    CATLASS_DEVICE void DrainStore()
    {
        // FIX -> MTE2 covers Xnext GM feedback, including the GM load itself.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID4);
    }

private:
    AscendC::LocalTensor<float> l0a_, l0b_, l0c_;
    AscendC::LocalTensor<float> l0aNext_, l0bNext_;

    CATLASS_DEVICE void Preload(AscendC::LocalTensor<float> a,
        AscendC::LocalTensor<float> b, const L1Layout& l1Layout,
        const L0ALayout& aLayout, const L0BLayout& bLayout, uint32_t k)
    {
        const bool bank1 = ((k / K_TILE) & 1u) != 0;
        const int32_t event = bank1 ? EVENT_ID1 : EVENT_ID0;
        auto nextA = bank1 ? l0aNext_ : l0a_;
        auto nextB = bank1 ? l0bNext_ : l0b_;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event);
        copyA_(nextA, a[l1Layout.GetOffset(MatrixCoord{0, k})], aLayout, l1Layout);
        copyB_(nextB, b[l1Layout.GetOffset(MatrixCoord{k, 0})], bLayout, l1Layout);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event);
    }

    typename Copies::CopyGmToL1A copyGm_;
    typename Copies::CopyL1ToL0A copyA_;
    typename Copies::CopyL1ToL0B copyB_;
    typename Copies::CopyL0CToGm copyOut_;
    Gemm::Tile::TileMmad<Arch::AtlasA2, MatrixType, MatrixType, void> mad_;
};

template<uint32_t K_TILE>
using ActiveResidentGemm = std::conditional_t<K_TILE == 32, ResidentGemmK32Pipeline, ResidentGemm<K_TILE>>;

class VectorPhases {
public:
    CATLASS_DEVICE explicit VectorPhases(Arch::Resource<Arch::AtlasA2>& resource)
        : residentEye_(resource.ubBuf.template GetBufferByByte<float>(0)),
          bigInput_(resource.ubBuf.template GetBufferByByte<float>(BIG_TILE_BYTES)),
          bigResidual_(resource.ubBuf.template GetBufferByByte<float>(2 * BIG_TILE_BYTES)),
          input_(resource.ubBuf.template GetBufferByByte<float>(BIG_TILE_BYTES)),
          residual_(resource.ubBuf.template GetBufferByByte<float>(BIG_TILE_BYTES + TILE_BYTES)),
          result_(resource.ubBuf.template GetBufferByByte<float>(BIG_TILE_BYTES + 2 * TILE_BYTES))
    {
        static_assert(BIG_TILE_BYTES + 2 * BIG_TILE_BYTES <= Arch::AtlasA2::UB_SIZE,
            "Resident eye plus two whole-matrix regions must fit the UB reservation");
        static_assert(BIG_TILE_BYTES + 3 * TILE_BYTES <= Arch::AtlasA2::UB_SIZE,
            "Resident eye plus three 64-row debug regions must fit the UB reservation");
    }

    // The two-eye constant is identical for every matrix and iteration, so it
    // is loaded once per kernel and reused by every call.
    CATLASS_DEVICE void LoadEye(AscendC::GlobalTensor<float> eye)
    {
        AscendC::DataCopy(residentEye_, eye[0], BIG_TILE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
    }

    CATLASS_DEVICE void Run(AscendC::GlobalTensor<float> input,
        AscendC::GlobalTensor<float> eye, AscendC::GlobalTensor<float> output,
        float coefficient, bool first, GM_ADDR pDebug = nullptr, GM_ADDR rDebug = nullptr)
    {
        AscendC::GlobalTensor<float> pGm, rGm;
        if (pDebug) pGm.SetGlobalBuffer((__gm__ float*)pDebug);
        if (rDebug) rGm.SetGlobalBuffer((__gm__ float*)rDebug);
        // The first step with debug copies needs three separate 64-row regions
        // beside the resident eye; every other call uses two whole-matrix
        // regions. The first step reuses the input region for X1 after Sub.
        if (first && (pDebug || rDebug)) {
            for (uint32_t offset = 0; offset < ELEMENTS; offset += TILE_ELEMENTS) {
                AscendC::DataCopy(input_, input[offset], TILE_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::Muls(input_, input_, coefficient, TILE_ELEMENTS);
                AscendC::PipeBarrier<PIPE_V>(); // P0=r32(M*c32)
                AscendC::Sub(residual_, residentEye_[offset], input_, TILE_ELEMENTS);
                AscendC::PipeBarrier<PIPE_V>(); // R=r32(positive-two-eye - P)
                AscendC::Muls(result_, residual_, coefficient, TILE_ELEMENTS);
                AscendC::PipeBarrier<PIPE_V>(); // X1=r32(R0*c32)
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                AscendC::DataCopy(output[offset], result_, TILE_ELEMENTS);
                if (pDebug) AscendC::DataCopy(pGm[offset], input_, TILE_ELEMENTS);
                if (rDebug) AscendC::DataCopy(rGm[offset], residual_, TILE_ELEMENTS);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            return;
        }
        const unsigned part=AscendC::GetSubBlockIdx()*TILE_ELEMENTS;
        AscendC::DataCopy(bigInput_, input[part], TILE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        if (first) {
            AscendC::Muls(bigInput_, bigInput_, coefficient, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>(); // P0=r32(M*c32)
        }
        AscendC::Sub(bigResidual_, residentEye_[part], bigInput_, TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>(); // R=r32(positive-two-eye - P)
        if (first) {
            AscendC::Muls(bigInput_, bigResidual_, coefficient, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>(); // X1=r32(R0*c32) into the input region
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::DataCopy(output[part], first ? bigInput_ : bigResidual_, TILE_ELEMENTS);
        if (pDebug) AscendC::DataCopy(pGm[part], bigInput_, TILE_ELEMENTS);
        if (rDebug) AscendC::DataCopy(rGm[part], bigResidual_, TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    AscendC::LocalTensor<float> residentEye_;
    AscendC::LocalTensor<float> bigInput_, bigResidual_;
    AscendC::LocalTensor<float> input_, residual_, result_;
};

struct Params {
    GM_ADDR m;
    GM_ADDR x;
    GM_ADDR eye;
    GM_ADDR workspace;
    GM_ADDR output;
    GM_ADDR pDebug;
    GM_ADDR rDebug;
    uint32_t matrices, iterations;
    float coefficient;
};

template<uint32_t K_TILE>
__global__ __aicore__ void GemmKernel(Params p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
    Arch::Resource<Arch::AtlasA2> resource;
    ActiveResidentGemm<K_TILE> mm(resource);
    auto a = resource.l1Buf.template GetBufferByByte<float>(0);
    auto b = resource.l1Buf.template GetBufferByByte<float>(MATRIX_BYTES);
    AscendC::GlobalTensor<float> gmA, gmB, gmY;
    gmA.SetGlobalBuffer((__gm__ float*)p.m);
    gmB.SetGlobalBuffer((__gm__ float*)p.x);
    gmY.SetGlobalBuffer((__gm__ float*)p.output);
    for (uint32_t q = AscendC::GetBlockIdx(); q < p.matrices; q += AscendC::GetBlockNum()) {
        const int64_t off = static_cast<int64_t>(q) * ELEMENTS;
        mm.Load(a, gmA[off]);
        mm.Load(b, gmB[off]);
        mm.Multiply(a, b, gmY[off]);
        mm.DrainStore();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

__global__ __aicore__ void FirstKernel(Params p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
    Arch::Resource<Arch::AtlasA2> resource;
    VectorPhases vec(resource);
    AscendC::GlobalTensor<float> gmM, gmEye, gmY;
    gmM.SetGlobalBuffer((__gm__ float*)p.m);
    gmEye.SetGlobalBuffer((__gm__ float*)p.eye);
    gmY.SetGlobalBuffer((__gm__ float*)p.output);
    vec.LoadEye(gmEye);
    for (uint32_t q = AscendC::GetBlockIdx(); q < p.matrices; q += AscendC::GetBlockNum()) {
        const int64_t off = static_cast<int64_t>(q) * ELEMENTS;
        vec.Run(gmM[off], gmEye, gmY[off], p.coefficient, true,
            p.pDebug ? p.pDebug + off * sizeof(float) : nullptr,
            p.rDebug ? p.rDebug + off * sizeof(float) : nullptr);
    }
#endif
}

// Two matrices are pipelined per AIC/AIV pair: P0 and P1 are both issued before
// the first R wait, so the AIV's R0 computation overlaps P1 and R1 overlaps
// Xnext0. The odd tail matrix keeps the original single-matrix schedule.
// Flags stay pairwise counted; the pairing and order are identical on both cores.
template<uint32_t K_TILE, bool FIRST_SPECIALIZED>
__aicore__ inline void RunNewton(Params p) {
#if defined(__DAV_CUBE__)
    const uint32_t owner = AscendC::GetBlockIdx();
    const uint32_t workers = AscendC::GetBlockNum();
    Arch::Resource<Arch::AtlasA2> resource;
    ActiveResidentGemm<K_TILE> mm(resource);
    auto m0 = resource.l1Buf.template GetBufferByByte<float>(0);
    auto x0a = resource.l1Buf.template GetBufferByByte<float>(MATRIX_BYTES);
    auto x0b = resource.l1Buf.template GetBufferByByte<float>(2 * MATRIX_BYTES);
    auto r0 = resource.l1Buf.template GetBufferByByte<float>(3 * MATRIX_BYTES);
    auto m1 = resource.l1Buf.template GetBufferByByte<float>(4 * MATRIX_BYTES);
    auto x1a = resource.l1Buf.template GetBufferByByte<float>(5 * MATRIX_BYTES);
    auto x1b = resource.l1Buf.template GetBufferByByte<float>(6 * MATRIX_BYTES);
    auto r1 = resource.l1Buf.template GetBufferByByte<float>(7 * MATRIX_BYTES);
    static_assert(8 * MATRIX_BYTES <= Arch::AtlasA2::L1_SIZE);
    AscendC::GlobalTensor<float> gmM, gmX, gmY, scratch;
    gmM.SetGlobalBuffer((__gm__ float*)p.m);
    gmX.SetGlobalBuffer((__gm__ float*)p.x);
    gmY.SetGlobalBuffer((__gm__ float*)p.output);
    scratch.SetGlobalBuffer((__gm__ float*)p.workspace);
    const int64_t slot = static_cast<int64_t>(owner) * 6 * ELEMENTS;
    auto wp0 = scratch[slot], wr0 = scratch[slot + ELEMENTS], wx0 = scratch[slot + 2 * ELEMENTS];
    auto wp1 = scratch[slot + 3 * ELEMENTS], wr1 = scratch[slot + 4 * ELEMENTS], wx1 = scratch[slot + 5 * ELEMENTS];
    uint32_t q = owner;
    while (q < p.matrices) {
        const uint32_t q2 = q + workers;
        if (q2 >= p.matrices) {
            // Odd tail: original single-matrix schedule.
            const int64_t off = static_cast<int64_t>(q) * ELEMENTS;
            mm.Load(m0, gmM[off]);
            if constexpr (FIRST_SPECIALIZED) {
                AscendC::CrossCoreWaitFlag<2>(X_READY);
                mm.Load(x0a, wx0);
            } else {
                mm.Load(x0a, gmX[off]);
            }
            for (uint32_t iteration = FIRST_SPECIALIZED ? 1 : 0; iteration < p.iterations; ++iteration) {
                mm.Multiply(m0, x0a, wp0);
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(P_READY);
                AscendC::CrossCoreWaitFlag<2>(R_READY);
                mm.Load(r0, wr0);
                const bool final = iteration + 1 == p.iterations;
                mm.Multiply(x0a, r0, final ? gmY[off] : wx0);
                mm.DrainStore();
                if (!final) {
                    mm.Load(x0b, wx0);
                    auto old = x0a; x0a = x0b; x0b = old;
                }
            }
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(Q_DONE);
            break;
        }
        const int64_t off0 = static_cast<int64_t>(q) * ELEMENTS;
        const int64_t off1 = static_cast<int64_t>(q2) * ELEMENTS;
        mm.Load(m0, gmM[off0]);
        mm.Load(m1, gmM[off1]);
        if constexpr (FIRST_SPECIALIZED) {
            AscendC::CrossCoreWaitFlag<2>(X_READY);
            mm.Load(x0a, wx0);
            AscendC::CrossCoreWaitFlag<2>(X_READY);
            mm.Load(x1a, wx1);
        } else {
            mm.Load(x0a, gmX[off0]);
            mm.Load(x1a, gmX[off1]);
        }
        for (uint32_t iteration = FIRST_SPECIALIZED ? 1 : 0; iteration < p.iterations; ++iteration) {
            mm.Multiply(m0, x0a, wp0);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(P_READY);
            mm.Multiply(m1, x1a, wp1);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(P_READY);
            const bool final = iteration + 1 == p.iterations;
            AscendC::CrossCoreWaitFlag<2>(R_READY);
            mm.Load(r0, wr0);
            mm.Multiply(x0a, r0, final ? gmY[off0] : wx0);
            mm.DrainStore();
            if (!final) {
                mm.Load(x0b, wx0);
                auto old = x0a; x0a = x0b; x0b = old;
            }
            AscendC::CrossCoreWaitFlag<2>(R_READY);
            mm.Load(r1, wr1);
            mm.Multiply(x1a, r1, final ? gmY[off1] : wx1);
            mm.DrainStore();
            if (!final) {
                mm.Load(x1b, wx1);
                auto old = x1a; x1a = x1b; x1b = old;
            }
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(Q_DONE);
        q = q2 + workers;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#elif defined(__DAV_VEC__)
    const uint32_t owner = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    const uint32_t workers = AscendC::GetBlockNum();
    Arch::Resource<Arch::AtlasA2> resource;
    VectorPhases vec(resource);
    AscendC::GlobalTensor<float> gmM, gmEye, scratch;
    gmM.SetGlobalBuffer((__gm__ float*)p.m);
    gmEye.SetGlobalBuffer((__gm__ float*)p.eye);
    scratch.SetGlobalBuffer((__gm__ float*)p.workspace);
    const int64_t slot = static_cast<int64_t>(owner) * 6 * ELEMENTS;
    auto wp0 = scratch[slot], wr0 = scratch[slot + ELEMENTS], wx0 = scratch[slot + 2 * ELEMENTS];
    auto wp1 = scratch[slot + 3 * ELEMENTS], wr1 = scratch[slot + 4 * ELEMENTS], wx1 = scratch[slot + 5 * ELEMENTS];
    vec.LoadEye(gmEye);
    uint32_t q = owner;
    while (q < p.matrices) {
        const uint32_t q2 = q + workers;
        if (q2 >= p.matrices) {
            const int64_t off = static_cast<int64_t>(q) * ELEMENTS;
            if constexpr (FIRST_SPECIALIZED) {
                vec.Run(gmM[off], gmEye, wx0, p.coefficient, true);
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(X_READY);
            }
            for (uint32_t iteration = FIRST_SPECIALIZED ? 1 : 0; iteration < p.iterations; ++iteration) {
                AscendC::CrossCoreWaitFlag<2>(P_READY);
                vec.Run(wp0, gmEye, wr0, 0.f, false,
                    p.pDebug ? p.pDebug + off * sizeof(float) : nullptr,
                    p.rDebug ? p.rDebug + off * sizeof(float) : nullptr);
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(R_READY);
            }
            AscendC::CrossCoreWaitFlag<2>(Q_DONE);
            break;
        }
        const int64_t off0 = static_cast<int64_t>(q) * ELEMENTS;
        const int64_t off1 = static_cast<int64_t>(q2) * ELEMENTS;
        if constexpr (FIRST_SPECIALIZED) {
            vec.Run(gmM[off0], gmEye, wx0, p.coefficient, true);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(X_READY);
            vec.Run(gmM[off1], gmEye, wx1, p.coefficient, true);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(X_READY);
        }
        for (uint32_t iteration = FIRST_SPECIALIZED ? 1 : 0; iteration < p.iterations; ++iteration) {
            AscendC::CrossCoreWaitFlag<2>(P_READY);
            vec.Run(wp0, gmEye, wr0, 0.f, false,
                p.pDebug ? p.pDebug + off0 * sizeof(float) : nullptr,
                p.rDebug ? p.rDebug + off0 * sizeof(float) : nullptr);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(R_READY);
            AscendC::CrossCoreWaitFlag<2>(P_READY);
            vec.Run(wp1, gmEye, wr1, 0.f, false,
                p.pDebug ? p.pDebug + off1 * sizeof(float) : nullptr,
                p.rDebug ? p.rDebug + off1 * sizeof(float) : nullptr);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(R_READY);
        }
        AscendC::CrossCoreWaitFlag<2>(Q_DONE);
        q = q2 + workers;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


 } // namespace JyShallowNewton
