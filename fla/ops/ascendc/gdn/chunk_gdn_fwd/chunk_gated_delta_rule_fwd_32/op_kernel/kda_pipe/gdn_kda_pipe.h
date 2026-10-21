/**
 * GDN forward scheduled like kda_32_var.
 *
 * One step is 32 tokens. AIV0 owns even chunks, AIV1 owns odd chunks: that AIV
 * does L2, sigmoid, cumsum and the 32x32 inverse. Both AIVs take part in every
 * chunk. v_new and o are split 16/16 (subblock 0 gets rows 0..15). State is
 * split by K: each AIV holds 64 of the 128 rows and adds its D slices as the
 * fixpipe lands. The next chunk's P and o_inter use the updated state.
 *
 * Cube order inside a chunk matches KDA's consumers: kkt/qk, then A@vb and
 * w@H, then v_new, then o and the four D slices. L2 of chunk c+2 runs after
 * the H upload, so the next w@H is not waiting out that L2.
 * H stays in L0B across q@H. q_exp is loaded before the v_new wait, so
 * q@H does not move H a second time. A@kbg uses the other L0 bank.
 * In-loop Q/K publish splits L2 the same way as the prologue: the K flag
 * is set before q's L2, so K@K is not waiting on q.
 */

#ifndef GDN_KDA_PIPE_H
#define GDN_KDA_PIPE_H

#include "../arch35/chunk_gated_delta_rule_fwd_32_matmul.h"
#include "../arch35/chunk_gated_delta_rule_fwd_32_tiling_data.h"
#include "../arch35/chunk_gated_delta_rule_fwd_32_vf.h"

namespace GdnKdaPipe {

using namespace AscendC;
using namespace AscendC::MicroAPI;
using GdnFwd32::kBt;
using GdnFwd32::kDim;
using GdnFwd32::kFixUbNz2Nd;

constexpr uint16_t kAiv1 = 16;
constexpr uint16_t kFlagQk = 0;
constexpr uint16_t kFlagQ = 9;
constexpr uint16_t kFlagKkt = 1;
constexpr uint16_t kFlagInv = 2;
constexpr uint16_t kFlagH = 3;
constexpr uint16_t kFlagP = 4;
constexpr uint16_t kFlagV = 5;
constexpr uint16_t kFlagO = 6;
constexpr uint16_t kFlagD = 7;
constexpr uint16_t kFlagD2 = 8;
constexpr uint16_t kFlagAttn = 10;

constexpr uint32_t kUbState = 0;
constexpr uint32_t kUbQ = 16 * 1024;
constexpr uint32_t kUbK = 24 * 1024;
constexpr uint32_t kUbV = 32 * 1024;
constexpr uint32_t kUbQnz = 40 * 1024;
constexpr uint32_t kUbKnz = 48 * 1024;
constexpr uint32_t kUbKbg = 56 * 1024;
constexpr uint32_t kUbQexp = 64 * 1024;
constexpr uint32_t kUbVnz = 72 * 1024;
constexpr uint32_t kUbG = 80 * 1024;
constexpr uint32_t kUbBeta = 81 * 1024;
constexpr uint32_t kUbBe = 82 * 1024;
constexpr uint32_t kUbGp = 83 * 1024;
constexpr uint32_t kUbEg = 84 * 1024;
constexpr uint32_t kUbKkt = 88 * 1024;
constexpr uint32_t kUbQk = 92 * 1024;
constexpr uint32_t kUbL = 96 * 1024;
constexpr uint32_t kUbA = 100 * 1024;
constexpr uint32_t kUbAbf = 104 * 1024;
constexpr uint32_t kUbAttn = 106 * 1024;
constexpr uint32_t kUbU = 108 * 1024;
constexpr uint32_t kUbP = 116 * 1024;
constexpr uint32_t kUbVn = 124 * 1024;
constexpr uint32_t kUbVg = 128 * 1024;
constexpr uint32_t kUbO = 132 * 1024;
// Two 32-row D slices per AIV, so the cube can land both before the saxpy.
constexpr uint32_t kUbD = 176 * 1024;
constexpr uint32_t kUbEye16 = 136 * 1024;
constexpr uint32_t kUbEye = 144 * 1024;
constexpr uint32_t kUbSh = 148 * 1024;
constexpr uint32_t kUbScratch = 152 * 1024;
constexpr uint32_t kUbG2 = 160 * 1024;
constexpr uint32_t kUbGp2 = 161 * 1024;
constexpr uint32_t kUbEg2 = 162 * 1024;

constexpr uint32_t kL1H = 0;
constexpr uint32_t kL1Slot = 40 * 1024;
constexpr uint32_t kL1Stride = 56 * 1024;
constexpr uint32_t kOffQ = 0;
constexpr uint32_t kOffK = 8 * 1024;
constexpr uint32_t kOffVb = 16 * 1024;
constexpr uint32_t kOffKbg = 24 * 1024;
constexpr uint32_t kOffQexp = 32 * 1024;
constexpr uint32_t kOffA = 40 * 1024;
constexpr uint32_t kOffAttn = 42 * 1024;
constexpr uint32_t kOffW = 44 * 1024;
constexpr uint32_t kL1Vn = 160 * 1024;
constexpr uint32_t kL1Vg = 168 * 1024;

__aicore__ inline uint16_t Seen(uint16_t id, uint32_t sub)
{
    return sub == 0 ? id : static_cast<uint16_t>(id + kAiv1);
}

// [m, n] L0C, m is a multiple of 32. Each call writes m/2 rows to one AIV.
// The second half starts one 16x16 fractal in, and srcStride stays the full
// height, matching kda's l0c_ub_bf16. dualDstCtl cannot be combined with bf16.
__aicore__ inline void FixpipeHalfBf16(LocalTensor<bfloat16_t> ub, LocalTensor<float> l0C, int m, int n)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m / 2);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(n);
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    p.subBlockId = false;
    Fixpipe<bfloat16_t, float, kFixUbNz2Nd>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<bfloat16_t, float, kFixUbNz2Nd>(ub, l0C[16 * 16], p);
}

// 16 ND rows. u/p are the half this AIV received from the dual fixpipe.
// global row = rowBase + r, so the gate reads that lane of eg.
__aicore__ inline void VnewHalfVF(LocalTensor<bfloat16_t> u, LocalTensor<bfloat16_t> p, LocalTensor<float> eg,
                                  LocalTensor<bfloat16_t> vNew, LocalTensor<bfloat16_t> vNewG, uint32_t rowBase)
{
    __ubuf__ bfloat16_t *uAddr = (__ubuf__ bfloat16_t *)u.GetPhyAddr();
    __ubuf__ bfloat16_t *pAddr = (__ubuf__ bfloat16_t *)p.GetPhyAddr();
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)vNew.GetPhyAddr();
    __ubuf__ bfloat16_t *gAddr = (__ubuf__ bfloat16_t *)vNewG.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> last;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(last, eAddr + (kBt - 1));
        for (uint16_t r = 0; r < 16; ++r) {
            RegTensor<bfloat16_t> a, b, glo, ghi;
            RegTensor<float> ri, gate;
            LoadAlign(a, uAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(b, pAddr + static_cast<uint32_t>(r) * kDim);
            Sub(a, a, b, m128);
            // Tight NZ, panel height 16. PublishHalf then needs one copy, not eight.
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(nAddr + static_cast<uint32_t>(r) * 16, a, 16,
                                                                  m128);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ri, eAddr + 32 + rowBase + r);
            Mul(gate, last, ri, pregAll);
            Cast<bfloat16_t, float, GdnFwd32::kCastB322B16>(glo, gate, m16);
            Cast<bfloat16_t, float, GdnFwd32::kCastB322B16Hi>(ghi, gate, m16);
            Or((RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)ghi, m16);
            Mul(a, a, glo, m128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(gAddr + static_cast<uint32_t>(r) * 16, a, 16,
                                                                  m128);
        }
    }
}

__aicore__ inline void ScaleHalfVF(LocalTensor<bfloat16_t> src, LocalTensor<bfloat16_t> dst, float scale)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)src.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> sc;
        Duplicate(sc, scale, pregAll);
        for (uint16_t r = 0; r < 16; ++r) {
            RegTensor<float> even, odd;
            GdnFwd32::LoadEvenOdd(sAddr + static_cast<uint32_t>(r) * kDim, even, odd, m16);
            Mul(even, even, sc, pregAll);
            Mul(odd, odd, sc, pregAll);
            RegTensor<bfloat16_t> lo, hi;
            Cast<bfloat16_t, float, GdnFwd32::kCastB322B16>(lo, even, m16);
            Cast<bfloat16_t, float, GdnFwd32::kCastB322B16Hi>(hi, odd, m16);
            Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
            StoreAlign(dAddr + static_cast<uint32_t>(r) * kDim, lo, m128);
        }
    }
}

// state = decay * state + d over one AIV's 64x128 tile. Both buffers are the
// NZ panel image (8 contiguous [64, 16] panels), so this is a flat saxpy.
__aicore__ inline void SaxpyStateVF(LocalTensor<bfloat16_t> state, LocalTensor<bfloat16_t> d, float decay)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)state.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)d.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> dec;
        Duplicate(dec, decay, preg);
        for (uint16_t i = 0; i < 64; ++i) {
            const uint32_t off = static_cast<uint32_t>(i) * 128;
            RegTensor<float> s0, s1, d0, d1;
            GdnFwd32::LoadCastBf16(sAddr + off, s0, preg);
            GdnFwd32::LoadCastBf16(sAddr + off + 64, s1, preg);
            GdnFwd32::LoadCastBf16(dAddr + off, d0, preg);
            GdnFwd32::LoadCastBf16(dAddr + off + 64, d1, preg);
            Mul(s0, s0, dec, preg);
            Mul(s1, s1, dec, preg);
            Add(s0, s0, d0, preg);
            Add(s1, s1, d1, preg);
            GdnFwd32::StoreCastBf16(sAddr + off, s0, preg);
            GdnFwd32::StoreCastBf16(sAddr + off + 64, s1, preg);
        }
    }
}

// [64, 128] ND <-> 8 contiguous [64, 16] panels. Used at the head and the tail
// only; the per-chunk state stays in panels.
__aicore__ inline void NdToPanels(LocalTensor<bfloat16_t> nz, LocalTensor<bfloat16_t> nd)
{
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)nd.GetPhyAddr();
    __ubuf__ bfloat16_t *zAddr = (__ubuf__ bfloat16_t *)nz.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t r = 0; r < 64; ++r) {
            RegTensor<bfloat16_t> row;
            LoadAlign(row, nAddr + static_cast<uint32_t>(r) * kDim);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(zAddr + static_cast<uint32_t>(r) * 16, row, 64,
                                                                  m128);
        }
    }
}

__aicore__ inline void PanelsToNd(LocalTensor<bfloat16_t> nd, LocalTensor<bfloat16_t> nz)
{
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)nd.GetPhyAddr();
    __ubuf__ bfloat16_t *zAddr = (__ubuf__ bfloat16_t *)nz.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t r = 0; r < 64; ++r) {
            RegTensor<bfloat16_t> row;
            LoadAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(row, zAddr + static_cast<uint32_t>(r) * 16, 64, m128);
            StoreAlign(nAddr + static_cast<uint32_t>(r) * kDim, row, m128);
        }
    }
}

class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR initial, GM_ADDR o,
                                GM_ADDR finalState, GM_ADDR workspace, const ChunkGatedDeltaRuleFwd32TilingData &td)
    {
        (void)workspace;
        sub_ = GetSubBlockIdx();
        batchSize_ = td.B;
        valueHeadNum_ = td.HV;
        seqLen_ = td.T;
        scale_ = td.scale;
        hasInitial_ = td.hasInitial;
        uint32_t aicIdx = GetBlockIdx();
        if ASCEND_IS_AIV {
            const uint32_t subNum = GetSubBlockNum();
            if (subNum > 0) {
                aicIdx /= subNum;
            }
        }
        aicNum_ = td.coreNum == 0 ? 1 : td.coreNum;
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        isActive_ = valueHeadNum_ != 0 && aicIdx < totalHeads;
        headBegin_ = isActive_ ? aicIdx : 0;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(o));
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(finalState));
        if (hasInitial_ != 0 && initial != nullptr) {
            initGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initial));
        }

        ubState_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbState);
        ubQ_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQ);
        ubK_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbK);
        ubV_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbV);
        ubQnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQnz);
        ubKnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbKnz);
        ubKbg_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbKbg);
        ubQexp_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQexp);
        ubVnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVnz);
        ubG_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbG);
        ubBeta_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbBeta);
        ubBe_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBe);
        ubGp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGp);
        ubEg_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEg);
        ubKkt_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbKkt);
        ubQk_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbQk);
        ubL_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbL);
        ubA_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbA);
        ubAbf_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAbf);
        ubAttn_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAttn);
        ubU_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbU);
        ubP_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbP);
        ubVn_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVn);
        ubVg_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVg);
        ubO_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbO);
        ubD_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbD);
        ubEye_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEye);
        ubEye16_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEye16);
        ubSh_ = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kUbSh);
        ubScratch_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbScratch);
        l1H_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1H);
        l1Vn_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Vn);
        l1Vg_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Vg);
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t base = kL1Slot + s * kL1Stride;
            l1Q_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffQ);
            l1K_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffK);
            l1Vb_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffVb);
            l1Kbg_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffKbg);
            l1Qexp_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffQexp);
            l1A_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffA);
            l1Attn_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffAttn);
            l1W_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffW);
        }
        // 32KB banks. H is 128*128 bf16 and fills bank 0; bank 1 never holds H.
        l0A_[0] = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(0);
        l0A_[1] = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(32 * 1024);
        l0B_[0] = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(0);
        l0B_[1] = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(32 * 1024);
        l0C0_ = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0C1_ = l0C0_[static_cast<uint32_t>(kBt * kDim)];
        ubG2_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbG2);
        ubGp2_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGp2);
        ubEg2_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEg2);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    __aicore__ inline int64_t TokenBase() const
    {
        return static_cast<int64_t>((static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * seqLen_);
    }

    __aicore__ inline bool Owns(uint32_t chunk) const
    {
        return (chunk & 1U) == sub_;
    }

    __aicore__ inline void ZeroState()
    {
        constexpr uint32_t nElem = 64 * 128;
        Duplicate(ubState_, static_cast<bfloat16_t>(0), nElem);
        SetFlag<HardEvent::V_MTE3>(1);
    }

    __aicore__ inline void InitState()
    {
        constexpr uint32_t nElem = 64 * 128;
        if (hasInitial_ != 0) {
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(sub_) * nElem);
            DataCopy(ubD_, initGm_[elem], nElem);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            NdToPanels(ubState_, ubD_);
        } else {
            Duplicate(ubState_, static_cast<bfloat16_t>(0), nElem);
        }
        SetFlag<HardEvent::V_MTE3>(1);
    }

    __aicore__ inline void LoadGate(uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        DataCopy(ubG2_, gGm_[token], kBt);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        GdnFwd32::CumsumExp2VF<true>(ubG2_, ubGp2_, ubEg2_, ubSh_);
    }

    // One vector register. Used to keep this chunk's eg while PrepareUb fills ubEg_.
    __aicore__ inline void Copy64(LocalTensor<float> dst, LocalTensor<float> src)
    {
        __ubuf__ float *sAddr = (__ubuf__ float *)src.GetPhyAddr();
        __ubuf__ float *dAddr = (__ubuf__ float *)dst.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg preg = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> v;
            LoadAlign(v, sAddr);
            StoreAlign(dAddr, v, preg);
        }
    }
    __aicore__ inline float DecayOf(LocalTensor<float> eg)
    {
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return eg.GetValue(kBt - 1);
    }

    // GM loads only. Issued before the inverse VF so the L2 wait is not an MTE2 stall.
    __aicore__ inline void PrepareLoad(uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const int32_t nRow = kBt * kDim;
        DataCopy(ubQ_, qGm_[token * kDim], nRow);
        DataCopy(ubK_, kGm_[token * kDim], nRow);
        DataCopy(ubV_, vGm_[token * kDim], nRow);
        DataCopy(ubG_, gGm_[token], kBt);
        DataCopy(ubBeta_, betaGm_[token], kBt);
        SetFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void WaitLoads()
    {
        WaitFlag<HardEvent::MTE2_V>(0);
        // Previous PublishQk may still be reading the NZ buffers this VF overwrites.
        if (nzArmed_ != 0) {
            WaitFlag<HardEvent::MTE3_V>(3);
            nzArmed_ = 0;
        }
    }

    __aicore__ inline void PrepareCompute()
    {
        WaitLoads();
        GdnFwd32::BetaSigmoidVF(ubBeta_, ubBe_);
        GdnFwd32::CumsumExp2VF<true>(ubG_, ubGp_, ubEg_, ubSh_);
        GdnFwd32::L2InplaceQKVF(ubQ_, ubK_, ubKbg_, ubQexp_, ubBe_, ubEg_, ubQnz_, ubKnz_);
        GdnFwd32::ScaleRowsVF<true>(ubV_, ubV_, ubBe_, reinterpret_cast<__ubuf__ bfloat16_t *>(ubVnz_.GetPhyAddr()));
    }

    __aicore__ inline void PrepareUb(uint32_t chunk)
    {
        PrepareLoad(chunk);
        PrepareCompute();
    }

    __aicore__ inline void PublishQk(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        GdnFwd32::CopyNzToL1(l1K_[slot], ubKnz_, kBt, kDim);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQk);
        GdnFwd32::CopyNzToL1(l1Q_[slot], ubQnz_, kBt, kDim);
        // K@K only needs K. Q is a second flag so the prologue can overlap q's L2
        // with K@K. Both flags are set here once the two NZ images are queued.
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQ);
        GdnFwd32::CopyNzToL1(l1Vb_[slot], ubVnz_, kBt, kDim);
        GdnFwd32::CopyNzToL1(l1Kbg_[slot], ubKbg_, kBt, kDim);
        GdnFwd32::CopyNzToL1(l1Qexp_[slot], ubQexp_, kBt, kDim);
        SetFlag<HardEvent::MTE3_V>(3);
        nzArmed_ = 1;
    }

    // K's L2 and the K copy complete before q's L2, and K@K runs under that
    // second L2. vb/kbg/qexp still go out after both flags.
    __aicore__ inline void PublishStaged(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        WaitLoads();
        GdnFwd32::BetaSigmoidVF(ubBeta_, ubBe_);
        GdnFwd32::CumsumExp2VF<true>(ubG_, ubGp_, ubEg_, ubSh_);
        GdnFwd32::L2InplaceQKVF<2>(ubQ_, ubK_, ubKbg_, ubQexp_, ubBe_, ubEg_, ubQnz_, ubKnz_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        GdnFwd32::CopyNzToL1(l1K_[slot], ubKnz_, kBt, kDim);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQk);
        GdnFwd32::L2InplaceQKVF<1>(ubQ_, ubK_, ubKbg_, ubQexp_, ubBe_, ubEg_, ubQnz_, ubKnz_);
        GdnFwd32::ScaleRowsVF<true>(ubV_, ubV_, ubBe_, reinterpret_cast<__ubuf__ bfloat16_t *>(ubVnz_.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        GdnFwd32::CopyNzToL1(l1Q_[slot], ubQnz_, kBt, kDim);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQ);
        GdnFwd32::CopyNzToL1(l1Vb_[slot], ubVnz_, kBt, kDim);
        GdnFwd32::CopyNzToL1(l1Kbg_[slot], ubKbg_, kBt, kDim);
        GdnFwd32::CopyNzToL1(l1Qexp_[slot], ubQexp_, kBt, kDim);
        SetFlag<HardEvent::MTE3_V>(3);
        nzArmed_ = 1;
    }

    // Gate and inverse stay in UB. The L1 copy is separate so this can run
    // while the cube is still reading this slot for the previous chunk's o.
    __aicore__ inline void InverseUb(uint32_t chunk)
    {
        (void)chunk;
        // L only. qk is still in flight on the prologue; attn is filled in InverseL1.
        GdnFwd32::GateAttnVF<1>(ubEg_, ubBe_, ubKkt_, ubQk_, ubL_, ubAttn_);
        GdnFwd32::InvertAVF(ubL_, ubA_, ubAbf_, ubEye_);
    }

    __aicore__ inline void InverseL1(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        // A is 2KB. The cube starts A@v on this flag. attn is only read at o.
        GdnFwd32::CopyNdToL1(l1A_[slot], ubAbf_, kBt, kBt);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagInv);
        CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
        GdnFwd32::GateAttnVF<2>(ubEg_, ubBe_, ubKkt_, ubQk_, ubL_, ubAttn_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        GdnFwd32::CopyNdToL1(l1Attn_[slot], ubAttn_, kBt, kBt);
    }

    __aicore__ inline void UploadH()
    {
        WaitFlag<HardEvent::V_MTE3>(1);
        // State is already 8 contiguous [64, 16] panels. Each panel is one
        // 2KB copy into its slot of the 128-row L1 image.
        constexpr uint32_t kRows = 64;
        constexpr uint32_t kFull = 128;
        constexpr uint16_t kBurst = static_cast<uint16_t>(kRows * 16 * sizeof(bfloat16_t) / 32);
        for (uint32_t fc = 0; fc < kDim / 16; ++fc) {
            DataCopy(l1H_[fc * kFull * 16 + sub_ * kRows * 16], ubState_[fc * kRows * 16],
                     DataCopyParams(1, kBurst, 0, 0));
        }
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH);
    }

    // 16 tight-NZ rows into a height-32 L1 panel. The other AIV's half is the dst gap.
    __aicore__ inline void CopyHalfNz(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nz, uint32_t row0)
    {
        constexpr uint16_t nFrac = static_cast<uint16_t>(kDim / 16);
        constexpr uint16_t burst = static_cast<uint16_t>(16 * 16 * sizeof(bfloat16_t) / 32);
        constexpr uint16_t gap = static_cast<uint16_t>((kBt - 16) * 16 * sizeof(bfloat16_t) / 32);
        DataCopy(l1[row0 * 16], nz, DataCopyParams(nFrac, burst, 0, gap));
    }

    __aicore__ inline void PublishHalf(uint32_t chunk)
    {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const uint32_t row0 = sub_ * 16;
        CopyHalfNz(l1Vn_, ubVn_, row0);
        CopyHalfNz(l1Vg_, ubVg_, row0);
        (void)chunk;
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagV);
    }

    __aicore__ inline void StoreO(uint32_t chunk)
    {
        ScaleHalfVF(ubO_, ubO_, scale_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt + static_cast<int64_t>(sub_) * 16;
        DataCopy(oGm_[token * kDim], ubO_, 16 * kDim);
    }

    __aicore__ inline void ApplyD(float decay)
    {
        // Slice 0 and slice 1 use different flags. A second set of the same id
        // would be dropped if the first slice's flag is still pending.
        CrossCoreWaitFlag<0x4, PIPE_V>(kFlagD);
        SaxpyStateVF(ubState_, ubD_, decay);
        SetFlag<HardEvent::V_MTE3>(1);
    }

    __aicore__ inline void ProcessAiv()
    {
        if (!isActive_) {
            return;
        }
        const uint32_t nChunk = seqLen_ / static_cast<uint32_t>(kBt);
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        bool tables = false;
        for (uint32_t head = headBegin_; head < totalHeads; head += aicNum_) {
            batchIdx_ = valueHeadNum_ == 0 ? 0 : head / valueHeadNum_;
            headIdx_ = valueHeadNum_ == 0 ? 0 : head - batchIdx_ * valueHeadNum_;
            const bool prime = sub_ < nChunk;
            // Loads first, then the table VF, so the old scalar SetValue loop is
            // not sitting in front of MTE2. No initial state on this path.
            if (prime && hasInitial_ == 0) {
                PrepareLoad(sub_);
            }
            if (!tables) {
                GdnFwd32::BuildTablesVF(ubSh_, ubEye_);
                tables = true;
            }
            if (hasInitial_ != 0) {
                InitState();
                if (prime) {
                    PrepareUb(sub_);
                    PublishQk(sub_);
                }
            } else {
                ZeroState();
                if (prime) {
                    PublishStaged(sub_);
                }
            }
            uint32_t invDone = 0xFFFFFFFFU;
            // 1 once the updated state is already in L1 for this chunk.
            uint32_t uploaded = 0;
            for (uint32_t c = 0; c < nChunk; ++c) {
                const bool owner = Owns(c);
                if (owner && invDone != c) {
                    CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                    InverseUb(c);
                    InverseL1(c);
                }
                // H before the gate. The non-owner cumsum was the cube's H bubble.
                if (uploaded == 0) {
                    UploadH();
                }
                uploaded = 0;
                // Chunk 1's kkt is ready before this loop (Q/K came from the prologue).
                // Invert it here, under chunk 0's A@v, instead of after v_new.
                if (!owner && c == 0 && c + 1 < nChunk) {
                    CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                    InverseUb(c + 1);
                    InverseL1(c + 1);
                    invDone = c + 1;
                }
                if (!owner) {
                    LoadGate(c);
                }
                // eg of this chunk is saved before the next PrepareUb, which reuses ubEg_.
                // L2 is after the H upload: the next chunk's w@H was waiting on it.
                const uint32_t nxt = c + 2;
                LocalTensor<float> egNow = owner ? ubEg_ : ubEg2_;
                if (owner && nxt < nChunk) {
                    Copy64(ubScratch_, ubEg_);
                    egNow = ubScratch_;
                }
                // Next chunk's GM load overlaps v_new, o and the saxpy. It used to
                // sit inside PublishStaged, in front of the cube's K@K / V wait.
                if (owner && nxt < nChunk) {
                    PrepareLoad(nxt);
                }
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
                const float decay = DecayOf(egNow);
                VnewHalfVF(ubU_, ubP_, egNow, ubVn_, ubVg_, sub_ * 16);
                PublishHalf(c);
                if (!owner && c + 1 < nChunk && invDone != c + 1) {
                    CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                    InverseUb(c + 1);
                    InverseL1(c + 1);
                    invDone = c + 1;
                }
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagO);
                StoreO(c);
                ApplyD(decay);
                // Next H before PublishQk. Q/K copies were sitting in front of the H flag.
                if (c + 1 < nChunk) {
                    UploadH();
                    uploaded = 1;
                }
                // K's L2 and the K flag go out before q's L2, so K@K is not behind q.
                if (owner && nxt < nChunk) {
                    PublishStaged(nxt);
                }
            }
            WaitFlag<HardEvent::V_MTE3>(1);
            if (nzArmed_ != 0) {
                WaitFlag<HardEvent::MTE3_V>(3);
                nzArmed_ = 0;
            }
            constexpr uint32_t nElem = 64 * 128;
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(sub_) * nElem);
            PanelsToNd(ubD_, ubState_);
            SetFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(1);
            DataCopy(stateGm_[elem], ubD_, nElem);
        }
    }

    __aicore__ inline LocalTensor<float> L0COf(uint8_t bank)
    {
        return bank == 0 ? l0C0_ : l0C1_;
    }

    // One L0 bank. M_MTE1(bank) and M_FIX(bank) are set; FIX_M(bank) is consumed.
    __aicore__ inline void Mad(LocalTensor<bfloat16_t> a, LocalTensor<bfloat16_t> b, int m, int n, int k,
                               bool transposeB, uint8_t bank)
    {
        WaitFlag<HardEvent::M_MTE1>(bank);
        GdnFwd32::LoadL1NzToL0AB(a, b, l0A_[bank], l0B_[bank], m, n, k, transposeB, 0, bank);
        GdnFwd32::MmadL0ToL0C(l0A_[bank], l0B_[bank], L0COf(bank), m, n, k, true, bank);
        SetFlag<HardEvent::M_MTE1>(bank);
        SetFlag<HardEvent::M_FIX>(bank);
    }

    // K@K only. The inverse starts on this flag, so it is issued before the H wait
    // and the H stall covers the mad. Q@K stays later, under v_new.
    __aicore__ inline void Kkt(uint32_t chunk, uint8_t bank)
    {
        const uint32_t owner = chunk & 1U;
        const uint32_t slot = owner;
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagQk, owner));
        Mad(l1K_[slot], l1K_[slot], kBt, kBt, kDim, false, bank);
        WaitFlag<HardEvent::M_FIX>(bank);
        GdnFwd32::FixpipeFp32Ub(ubKkt_, L0COf(bank), kBt, kBt, static_cast<uint8_t>(owner));
        SetFlag<HardEvent::FIX_M>(bank);
        CrossCoreSetFlag<0x4, PIPE_FIX>(Seen(kFlagKkt, owner));
    }

    __aicore__ inline void Qk(uint32_t chunk, uint8_t bank)
    {
        const uint32_t owner = chunk & 1U;
        const uint32_t slot = owner;
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagQ, owner));
        Mad(l1Q_[slot], l1K_[slot], kBt, kBt, kDim, false, bank);
        WaitFlag<HardEvent::M_FIX>(bank);
        GdnFwd32::FixpipeFp32Ub(ubQk_, L0COf(bank), kBt, kBt, static_cast<uint8_t>(owner));
        SetFlag<HardEvent::FIX_M>(bank);
        CrossCoreSetFlag<0x4, PIPE_FIX>(Seen(kFlagAttn, owner));
    }

    // bothBanks: K@K on bank 0 and Q@K on bank 1, so the Q load overlaps the K mad.
    // Otherwise both gemms stay on bank 1 and bank 0 (the live H) is left alone.
    __aicore__ inline void KktQk(uint32_t chunk, bool bothBanks)
    {
        const uint8_t b0 = bothBanks ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1);
        const uint8_t b1 = bothBanks ? static_cast<uint8_t>(1) : b0;
        Kkt(chunk, b0);
        Qk(chunk, b1);
    }

    __aicore__ inline void ProcessAic()
    {
        if (!isActive_) {
            return;
        }
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        const uint32_t nChunk = seqLen_ / static_cast<uint32_t>(kBt);
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        for (uint32_t head = headBegin_; head < totalHeads; head += aicNum_) {
            if (nChunk > 0) {
                KktQk(0, true);
            }
            // Chunk 1's Q/K are already published. Its kkt overlaps chunk 0's inverse.
            if (nChunk > 1) {
                KktQk(1, true);
            }
            for (uint32_t c = 0; c < nChunk; ++c) {
                const uint32_t owner = c & 1U;
                const uint32_t slot = owner;
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagInv, owner));

                // w's load overlaps u's mad. Different L0C, so u's fixpipe overlaps w's mad.
                Mad(l1A_[slot], l1Vb_[slot], kBt, kDim, kBt, true, 0);
                Mad(l1A_[slot], l1Kbg_[slot], kBt, kDim, kBt, true, 1);
                WaitFlag<HardEvent::M_FIX>(0);
                FixpipeHalfBf16(ubU_, l0C0_, kBt, kDim);
                SetFlag<HardEvent::FIX_M>(0);
                WaitFlag<HardEvent::M_FIX>(1);
                GdnFwd32::FixpipeBf16L1(l1W_[slot], l0C1_, kBt, kDim);
                SetFlag<HardEvent::FIX_M>(1);
                SetFlag<HardEvent::FIX_MTE1>(0);
                WaitFlag<HardEvent::FIX_MTE1>(0);

                // H is the updated state. A@v above overlaps the upload.
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagH, 0));
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagH, 1));
                Mad(l1W_[slot], l1H_, kBt, kDim, kDim, true, 0);
                WaitFlag<HardEvent::M_FIX>(0);
                FixpipeHalfBf16(ubP_, l0C0_, kBt, kDim);
                SetFlag<HardEvent::FIX_M>(0);
                CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagP);
                CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagP + kAiv1));

                // H stays in l0B_[0]. q_exp is in L1 already; load it before v_new so
                // q@H does not move H again. Bank 1 holds the next kkt.
                WaitFlag<HardEvent::M_MTE1>(0);
                GdnFwd32::LoadL1NzToL0A(l1Qexp_[slot], l0A_[0], kBt, kDim, false);
                SetFlag<HardEvent::MTE1_M>(0);
                if (c >= 1 && c + 1 < nChunk) {
                    KktQk(c + 1, false);
                }

                CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagV, 0));
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(Seen(kFlagV, 1));

                WaitFlag<HardEvent::MTE1_M>(0);
                GdnFwd32::MmadL0ToL0C(l0A_[0], l0B_[0], l0C0_, kBt, kDim, kDim, true, 0);
                SetFlag<HardEvent::M_MTE1>(0);
                WaitFlag<HardEvent::M_MTE1>(1);
                GdnFwd32::LoadL1NzToL0AB(l1Attn_[slot], l1Vn_, l0A_[1], l0B_[1], kBt, kDim, kBt, true, 0, 1);
                GdnFwd32::MmadL0ToL0C(l0A_[1], l0B_[1], l0C0_, kBt, kDim, kBt, false, 0);
                SetFlag<HardEvent::M_MTE1>(1);
                SetFlag<HardEvent::M_FIX>(0);
                WaitFlag<HardEvent::M_FIX>(0);
                FixpipeHalfBf16(ubO_, l0C0_, kBt, kDim);
                SetFlag<HardEvent::FIX_M>(0);
                CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagO);
                CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagO + kAiv1));

                // One 64-row panel per AIV. ubD_ is 64*128 bf16, so both saxpy slices
                // land in one fixpipe. The two L0C tiles are 32KB apart.
                constexpr int kDm = 64;
                constexpr uint8_t dBank = 0;
                WaitFlag<HardEvent::M_MTE1>(dBank);
                GdnFwd32::LoadL1NzToL0AB(l1K_[slot], l1Vg_, l0A_[0], l0B_[0], kDm, kDim, kBt, true, 0, dBank, true,
                                         false);
                for (int part = 0; part < 2; ++part) {
                    const uint8_t ev = static_cast<uint8_t>(part);
                    LocalTensor<float> l0c = l0C0_[static_cast<uint32_t>(part) * static_cast<uint32_t>(kDm * kDim)];
                    WaitFlag<HardEvent::MTE1_M>(dBank);
                    GdnFwd32::MmadL0ToL0C(l0A_[0], l0B_[0], l0c, kDm, kDim, kBt, true, ev);
                    SetFlag<HardEvent::M_FIX>(ev);
                    SetFlag<HardEvent::M_MTE1>(dBank);
                    if (part + 1 < 2) {
                        WaitFlag<HardEvent::M_MTE1>(dBank);
                        auto kPart = l1K_[slot][static_cast<uint32_t>(2 * 2 * kBt * 16)];
                        GdnFwd32::LoadL1NzToL0A(kPart, l0A_[0], kDm, kBt, true);
                        SetFlag<HardEvent::MTE1_M>(dBank);
                    }
                    WaitFlag<HardEvent::M_FIX>(ev);
                    const uint8_t dstSub = static_cast<uint8_t>(part);
                    GdnFwd32::FixpipeBf16UbNz(ubD_, l0c, kDm, kDim, dstSub);
                    SetFlag<HardEvent::FIX_M>(ev);
                    CrossCoreSetFlag<0x4, PIPE_FIX>(Seen(kFlagD, dstSub));
                }
            }
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
    }

    OnChipBuffer buf;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_, betaGm_, oGm_, stateGm_, initGm_;
    GlobalTensor<float> gGm_;
    LocalTensor<bfloat16_t> ubState_, ubQ_, ubK_, ubV_, ubQnz_, ubKnz_, ubKbg_, ubQexp_, ubVnz_;
    LocalTensor<bfloat16_t> ubBeta_, ubAbf_, ubAttn_, ubU_, ubP_, ubVn_, ubVg_, ubO_, ubD_;
    LocalTensor<float> ubG_, ubBe_, ubGp_, ubEg_, ubKkt_, ubQk_, ubL_, ubA_, ubEye_, ubEye16_, ubScratch_;
    LocalTensor<float> ubG2_, ubGp2_, ubEg2_;
    LocalTensor<uint32_t> ubSh_;
    LocalTensor<bfloat16_t> l1H_, l1Vn_, l1Vg_;
    LocalTensor<bfloat16_t> l1Q_[2], l1K_[2], l1Vb_[2], l1Kbg_[2], l1Qexp_[2], l1A_[2], l1Attn_[2], l1W_[2];
    LocalTensor<bfloat16_t> l0A_[2], l0B_[2];
    LocalTensor<float> l0C0_, l0C1_;

    uint32_t sub_ = 0;
    uint32_t aicNum_ = 1;
    uint32_t headBegin_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t valueHeadNum_ = 0;
    uint32_t seqLen_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t headIdx_ = 0;
    uint32_t hasInitial_ = 0;
    uint32_t nzArmed_ = 0;
    float scale_ = 0.f;
    bool isActive_ = false;
};

}  // namespace GdnKdaPipe

#endif
