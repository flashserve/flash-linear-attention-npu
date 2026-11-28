/**
 * MegaRecurrentKDA v2 Vector-side (AIV) stage: Stage R.
 *
 * One AIV owns a (sequence, head) pair through its last token and retains H
 * in UB across projection tiles. Each pair performs:
 *
 *   1. conv1d + silu for the head's 384 components (q/k/v of this head),
 *      sliding over three history rows plus this call's raw qkv rows;
 *   2. L2 normalisation of q and k inside D = 128, q additionally scaled;
 *   3. sigmoid(beta_raw), sigmoid(gate_raw), the decay chain
 *      exp(lower_bound * sigmoid(exp(A_log) * (decay_raw + dt_bias)));
 *   4. the S-step KDA recursion on H [V, K] (per-K-column decay, matvec with k,
 *      rank-1 update, matvec with q) writing replay_u / replay_k / replay_decay;
 *   5. RMSNorm + output gate -> `gated`;
 *   6. the in-place conv_state update of this head's 384 columns.
 *
 * Cost model: per token the mandatory work is two 128x64 element-wise products
 * plus two row reductions (H@k, H@q), one rank-1 update and one per-column
 * scale of H.  Everything that only depends on the chunk (conv, normalisation
 * scales, decay, ogate) is hoisted out of the per-token loop, and full-pipe
 * barriers are only placed where a scalar or an MTE operation reads V output.
 */
#ifndef MEGA_RECURRENT_KDA_V2_VECTOR_H
#define MEGA_RECURRENT_KDA_V2_VECTOR_H

#include "catlass/catlass.hpp"
#include "mega_recurrent_kda_v2_sync.h"
#include "mega_recurrent_kda_v2_tiling.h"

namespace MegaRecurrentKdaV2 {
namespace RDetail {

__aicore__ inline uint32_t Align16(uint32_t value)
{
    return (value + 15u) & ~15u;
}

/// Vector-pipe fence in front of a count-form (level-2) vector call.  Profiling
/// knob bit6 (64) drops them so their cost can be measured.
__aicore__ inline void VBar(const V2Params &params)
{
    if ((params.skipSync & 64u) == 0u) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}

constexpr uint32_t kD = 128;
constexpr uint32_t kHalf = 64;

// Splitting the 128 value rows guarantees at most 64 local state rows, leaving
// room to prepare eight independent input rows while preserving token-order updates.
template <bool SplitValues>
struct VectorLayout {
    static constexpr uint32_t kChunk = SplitValues ? 8 : 4;                      // tokens per conv/norm chunk
    static constexpr uint32_t kHalfFloats = (SplitValues ? kD / 2 : kD) * kHalf; // 8192
    static constexpr uint32_t kHalfBytes = kHalfFloats * 4u;                     // 32768

    // UB layout (bytes).
    static constexpr uint32_t kUbh0 = 0; // [128][64] fp32
    static constexpr uint32_t kUbh1 = kUbh0 + kHalfBytes;
    static constexpr uint32_t kUbScratch = kUbh1 + kHalfBytes;               // prod / rep, [128][64] fp32
    static constexpr uint32_t kUbRawB = kUbScratch + kHalfBytes;             // [CH][384] bf16
    static constexpr uint32_t kUbRawF = kUbRawB + kChunk * 384u * 2u;        // [CH+3][384] fp32
    static constexpr uint32_t kUbConv = kUbRawF + (kChunk + 3u) * 384u * 4u; // [CH][384] fp32
    static constexpr uint32_t kUbConvB = kUbConv + kChunk * 384u * 4u;       // [CH][384] bf16
    static constexpr uint32_t kUbRepW = kUbConvB + kChunk * 384u * 2u;       // [CH][384] fp32
    static constexpr uint32_t kUbConvW = kUbRepW + kChunk * 384u * 4u;       // [4][3][128] bf16
    static constexpr uint32_t kUbCache = kUbConvW + 4u * 3u * kD * 2u;       // [3][384] fp32
    static constexpr uint32_t kUbQ = kUbCache + 3u * 384u * 4u;              // [CH][128] fp32
    static constexpr uint32_t kUbK = kUbQ + kChunk * kD * 4u;
    static constexpr uint32_t kUbV = kUbK + kChunk * kD * 4u;
    static constexpr uint32_t kUbDec = kUbV + kChunk * kD * 4u;
    static constexpr uint32_t kUbOg = kUbDec + kChunk * kD * 4u;
    static constexpr uint32_t kUbDrB = kUbOg + kChunk * kD * 4u;            // [CH][128] bf16
    static constexpr uint32_t kUbGoB = kUbDrB + kChunk * kD * 2u;           // [CH][128] bf16
    static constexpr uint32_t kUbBetaB = kUbGoB + kChunk * kD * 2u;         // [CH*128] bf16
    static constexpr uint32_t kUbBeta = kUbBetaB + kChunk * kD * 2u;        // [CH*128] fp32
    static constexpr uint32_t kUbOnes = kUbBeta + kChunk * kD * 4u;         // 1536 fp32
    static constexpr uint32_t kUbGamma = kUbOnes + kChunk * 384u * 4u;      // 128 fp32
    static constexpr uint32_t kUbBias = kUbGamma + kD * 4u;                 // 128 fp32
    static constexpr uint32_t kUbTmp = kUbBias + kD * 4u;                   // [CH][384] fp32
    static constexpr uint32_t kUbVecs = kUbTmp + kChunk * 384u * 4u;        // small vectors/staging
    static constexpr uint32_t kUbOnesD = kUbVecs + 512u;                    // 128 fp32 ones
    static constexpr uint32_t kUbCorrBuf = kUbOnesD + kD * 4u;              // [CH][128] fp32
    static constexpr uint32_t kUbKHalfBuf = kUbCorrBuf + kChunk * kD * 4u;  // [CH][128] fp16
    static constexpr uint32_t kUbGatedBuf = kUbKHalfBuf + kChunk * kD * 2u; // [CH][128] bf16
    static constexpr uint32_t kUbSVec = kUbGatedBuf + kChunk * kD * 2u;     // 128 fp32
    static constexpr uint32_t kUbRVec = kUbSVec + kD * 4u;                  // 128 fp32
    static constexpr uint32_t kUbRedTmp = kUbRVec + kD * 4u;
    static constexpr uint32_t kUbStg = kUbRedTmp + 2048u;          // 8 int32 publish staging
    static constexpr uint32_t kUbRBuf = kUbStg + 64u;              // [CH][128] fp32 r
    static constexpr uint32_t kUbR2 = kUbRBuf + kChunk * kD * 4u;  // [CH][128] fp32 r^2
    static constexpr uint32_t kUbPack = kUbR2 + kChunk * kD * 4u;  // 2 x [CH][64] fp32
    static constexpr uint32_t kUbSum = kUbPack + kChunk * kD * 4u; // 16 fp32
    static constexpr uint32_t kUbBytes = kUbSum + 64u;
};
static_assert(VectorLayout<true>::kUbBytes <= 192u * 1024u);
constexpr uint32_t kUbBytes = VectorLayout<false>::kUbBytes;
constexpr uint32_t kUbStg = VectorLayout<false>::kUbStg;

// Small-vector offsets inside kUbVecs (floats).
constexpr uint32_t kVecS1 = 0;  // [8]
constexpr uint32_t kVecS2 = 16; // [8]
constexpr uint32_t kVecR1 = 32; // [8]
constexpr uint32_t kVecR2 = 48; // [8]
constexpr uint32_t kVecM1 = 64; // [8]
constexpr uint32_t kVecM2 = 80; // [8]
constexpr uint32_t kVecSq = 96; // [16]

/// Row sums of a packed [128][64] FP32 tile into 128 contiguous values.
///
/// Two-level block reduce: the 8 column blocks (8 values each) of a row are
/// reduced to 8 partial sums, which the hardware packs into one 32 B block, so
/// the intermediate is a compact [128][8] tile; the second call reduces those 8
/// partials per row into the 32 B block that holds the final row sum.  The
/// summation order is the same 8-then-8 grouping the whole-repeat form uses,
/// but block-reduce is ~4.6x cheaper than whole-reduce on this core (measured
/// 112 ns vs 520 ns per 128-row reduce), which is ~25% of the per-token cost.
__aicore__ inline void RowReduce64(AscendC::LocalTensor<float> const &dst, AscendC::LocalTensor<float> const &src,
                                   AscendC::LocalTensor<float> const &tmp, uint32_t rows)
{
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(tmp, src, 64, rows, 1, 1, 8);
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(dst, tmp, 64, rows / 8, 1, 1, 8);
}

/// Row sums of a packed [tokens][128] FP32 tile into `tokens` values (the two
/// 64-wide K halves of each token row are joined).  Both levels use the block
/// reduce, which packs eight 8-value block sums per 32 B block: the first level
/// writes `tokens` blocks per K half (half-major), the second folds each block
/// into one value in a 32 B block of its own, and the final Add joins the two
/// halves.  Same 8-groups-then-8 summation order as the AR whole-reduce path.
/// `dst2` must be a 32 B aligned scratch block next to `dst`.
__aicore__ inline void RowReduce128(AscendC::LocalTensor<float> const &dst, AscendC::LocalTensor<float> const &dst2,
                                    AscendC::LocalTensor<float> const &src, AscendC::LocalTensor<float> const &part,
                                    uint32_t tokens)
{
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(part, src, 64, tokens, 1, 1, 16);
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(part[8u * tokens], src[kHalf], 64, tokens, 1, 1, 16);
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(dst, part, static_cast<int32_t>(8u * tokens), 1, 1, 1, 8);
    AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(dst2, part[8u * tokens], static_cast<int32_t>(8u * tokens), 1, 1,
                                                       1, 8);
    AscendC::Add(dst, dst, dst2, static_cast<int32_t>(tokens));
}

/// conv1d_weight normalised form: [3][N][4][D].  The host also accepts the other
/// two API layouts and normalises them before the launch.
__aicore__ inline void LoadConvWeights(AscendC::LocalTensor<bfloat16_t> const &convW, GM_ADDR convWeight, uint32_t p,
                                       uint32_t head, uint32_t layout)
{
    AscendC::GlobalTensor<bfloat16_t> gmW;
    gmW.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(convWeight));
    const uint32_t dBlocks = kD * 2u / 32u; // 8
    if (layout == 0) {
        // [tap, Q/K/V, head, D]: consecutive head slices are P elements apart.
        const uint16_t gap = static_cast<uint16_t>((p - kD) / 16u);
        AscendC::DataCopyParams copy{12, static_cast<uint16_t>(dBlocks), gap, 0};
        AscendC::DataCopy(convW, gmW[head * kD], copy);
    } else {
        // Per-head and shared weights both store the four taps contiguously.
        AscendC::DataCopyParams copy{4, static_cast<uint16_t>(dBlocks), 0, static_cast<uint16_t>(2 * dBlocks)};
        for (uint32_t g = 0; g < 3; ++g) {
            const size_t group = layout == 1 ? g * (p / kD) + head : g;
            AscendC::DataCopy(convW[g * kD], gmW[group * 4 * kD], copy);
        }
    }
}

} // namespace RDetail

struct HeadConstants {
    uint32_t head = UINT32_MAX;
    float alpha = 0.f;
};

// Bias, norm weight and the four convolution taps occupy dedicated UB slices.
// They stay valid until this AIV switches to another head.
template <bool SplitValues>
CATLASS_DEVICE void PrepareHeadConstants(const V2Params &p, const AscendC::LocalTensor<uint8_t> &ub, uint32_t head,
                                         HeadConstants &cached)
{
    using namespace RDetail;
    using Layout = VectorLayout<SplitValues>;
    if (cached.head == head) {
        return;
    }
    const uint32_t D = kD, P = p.p;
    AscendC::LocalTensor<bfloat16_t> convW = ub[Layout::kUbConvW].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<float> gammaV = ub[Layout::kUbGamma].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> biasV = ub[Layout::kUbBias].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> convWf = ub[Layout::kUbRepW].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> tmp = ub[Layout::kUbTmp].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> ones = ub[Layout::kUbOnes].template ReinterpretCast<float>();
    AscendC::GlobalTensor<float> gmALog;
    gmALog.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.aLog));
    AscendC::GlobalTensor<float> gmDt;
    gmDt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.dtBias));
    AscendC::GlobalTensor<float> gmGamma;
    gmGamma.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.gamma));
    // ---- per-chain constants ---------------------------------------------
    AscendC::DataCopy(gammaV, gmGamma, static_cast<int32_t>(D));
    AscendC::DataCopy(biasV, gmDt[static_cast<size_t>(head) * D], static_cast<int32_t>(D));
    AscendC::PipeBarrier<PIPE_ALL>();
    // gate_logits = exp(A_log) * (decay_raw + dt_bias): the scalar exp(A_log) is
    // evaluated on the vector unit (one 8-lane op) because the scalar unit has no
    // transcendental instruction.
    float alpha = gmALog.GetValue(head);
    {
        AscendC::Duplicate(tmp, alpha, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmp, tmp, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        alpha = tmp.GetValue(0);
    }

    // ---- conv weights for this head (FP32, reused by every chunk) --------
    // The old form rebuilt a [CH][384] repeated-weight tile per tap and per
    // chunk with doubling UB copies plus a fence after each one.  Keeping the
    // 4x3x128 casted weights resident lets the conv1d add every tap with a
    // row-broadcast operand (src1 repeat stride 0) and no per-tap fence at all.
    if ((p.skipSync & 2048u) == 0u) {
        LoadConvWeights(convW, p.convWeight, P, head, p.convLayout);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(convWf, convW, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(4u * 3u * kD));
        VBar(p);
    }
    AscendC::LocalTensor<float> oneDInit = ub[Layout::kUbOnesD].template ReinterpretCast<float>();
    AscendC::Duplicate(ones, 1.0f, static_cast<int32_t>(Layout::kChunk * 384u));
    AscendC::Duplicate(oneDInit, 1.0f, static_cast<int32_t>(kD));
    AscendC::PipeBarrier<PIPE_ALL>();
    if ((p.skipSync & 4096u) != 0u) {
        AscendC::Duplicate(biasV, 0.0f, static_cast<int32_t>(D));
        AscendC::Duplicate(gammaV, 1.0f, static_cast<int32_t>(D));
    }
    cached.head = head;
    cached.alpha = alpha;
}

/// Each AIV maintains complete K rows for its assigned value range, in token order.
/// A single owner covers all 128 values. T02 waits and publishes at tile boundaries;
/// T01 batches credits in its caller. Both keep the FP32 state in UB until the sequence ends.
template <bool StreamTiles = false, bool SplitValues = false>
CATLASS_DEVICE void ProcessSequence(V2Params const &p, AscendC::LocalTensor<uint8_t> const &ub, uint32_t b,
                                    uint32_t head, uint32_t seqStart, uint32_t seqEnd, HeadConstants &cached,
                                    uint32_t firstValue = 0, uint32_t rows = 128)
{
    using namespace RDetail;
    using Layout = VectorLayout<SplitValues>;
    const uint32_t valueRows = SplitValues ? rows : 128;
    const uint32_t valueStart = SplitValues ? firstValue : 0;
    const uint32_t N = p.n;
    const uint32_t P = p.p;
    const uint32_t D = kD;
    const uint32_t threeP = 3u * P;

    AscendC::GlobalTensor<int32_t> gmIdx;
    gmIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.stateIndices));
    AscendC::GlobalTensor<int32_t> gmAck;
    gmAck.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.accepted));

    if (seqEnd <= seqStart) {
        return;
    }
    const uint32_t n = seqEnd - seqStart;
    const int32_t id = gmIdx.GetValue(b);
    const uint32_t slot = p.sourceConvStride && (id < 0 || uint32_t(id) >= p.blockNum) ? 0u : uint32_t(id);
    const uint32_t record = p.sourceConvStride ? b : slot;
    const uint32_t count = static_cast<uint32_t>(gmAck.GetValue(b));
    const uint32_t accepted = p.sourceConvStride && (id <= 0 || count > p.convLength-2) ? 1u : count;
    const uint32_t hs = accepted > 1u ? accepted - 1u : 0u;

    // ---- tensor views -----------------------------------------------------
    AscendC::LocalTensor<float> h0 = ub[Layout::kUbh0].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> h1 = ub[Layout::kUbh1].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> scratch = ub[Layout::kUbScratch].template ReinterpretCast<float>();
    AscendC::LocalTensor<bfloat16_t> rawB = ub[Layout::kUbRawB].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<float> rawF = ub[Layout::kUbRawF].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> conv = ub[Layout::kUbConv].template ReinterpretCast<float>();
    AscendC::LocalTensor<bfloat16_t> convB = ub[Layout::kUbConvB].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<float> repW = ub[Layout::kUbRepW].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> cache = ub[Layout::kUbCache].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> qV = ub[Layout::kUbQ].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> kV = ub[Layout::kUbK].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> vV = ub[Layout::kUbV].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> decV = ub[Layout::kUbDec].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> ogV = ub[Layout::kUbOg].template ReinterpretCast<float>();
    AscendC::LocalTensor<bfloat16_t> drB = ub[Layout::kUbDrB].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<bfloat16_t> goB = ub[Layout::kUbGoB].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<bfloat16_t> betaB = ub[Layout::kUbBetaB].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<float> betaV = ub[Layout::kUbBeta].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> ones = ub[Layout::kUbOnes].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> corrBuf = ub[Layout::kUbCorrBuf].template ReinterpretCast<float>();
    AscendC::LocalTensor<half> kHalfBuf = ub[Layout::kUbKHalfBuf].template ReinterpretCast<half>();
    AscendC::LocalTensor<bfloat16_t> gatedBuf = ub[Layout::kUbGatedBuf].template ReinterpretCast<bfloat16_t>();
    AscendC::LocalTensor<float> sVecBig = ub[Layout::kUbSVec].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> rVecBig = ub[Layout::kUbRVec].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> tmp = ub[Layout::kUbTmp].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> vecs = ub[Layout::kUbVecs].template ReinterpretCast<float>();
    AscendC::LocalTensor<uint8_t> redTmp = ub[Layout::kUbRedTmp];
    AscendC::LocalTensor<float> rBuf = ub[Layout::kUbRBuf].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> r2Buf = ub[Layout::kUbR2].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> packBuf = ub[Layout::kUbPack].template ReinterpretCast<float>();
    AscendC::LocalTensor<int32_t> stg = ub[Layout::kUbStg].template ReinterpretCast<int32_t>();

    AscendC::GlobalTensor<bfloat16_t> gmQkv;
    gmQkv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wQkv));
    AscendC::GlobalTensor<bfloat16_t> gmDr;
    gmDr.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wDecayRaw));
    AscendC::GlobalTensor<bfloat16_t> gmGo;
    gmGo.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wGateRaw));
    AscendC::GlobalTensor<bfloat16_t> gmBeta;
    gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wBetaRaw));
    AscendC::GlobalTensor<bfloat16_t> gmGated;
    gmGated.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wGated));
    AscendC::GlobalTensor<float> gmRu;
    gmRu.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.replayU));
    AscendC::GlobalTensor<half> gmRk;
    gmRk.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(p.replayK));
    AscendC::GlobalTensor<float> gmRd;
    gmRd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.replayDecay));
    AscendC::GlobalTensor<bfloat16_t> gmCs;
    gmCs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.convState));

    const uint32_t packParamsRaw[6] = {1, 1, 1, 8, 8, 0};
    const AscendC::BinaryRepeatParams packParams(packParamsRaw[0], packParamsRaw[1], packParamsRaw[2], packParamsRaw[3],
                                                 packParamsRaw[4], packParamsRaw[5]);
    // Row-broadcast form: src0 block stride 0 (one 32 B block holds the per-row
    // scalar eight times), src0 repeat stride 1 block, src1 row stride 0.
    const AscendC::BinaryRepeatParams bcastParams(1, 0, 1, 8, 1, 0);

    PrepareHeadConstants<SplitValues>(p, ub, head, cached);
    const float alpha = cached.alpha;
    AscendC::LocalTensor<float> convWf = ub[Layout::kUbRepW].template ReinterpretCast<float>();

    // Load the two K halves once and update them in place throughout recurrence.
    if ((p.skipSync & 1024u) == 0u) {
        AscendC::GlobalTensor<float> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.checkpoint) + slot * p.checkpointStride + head * D * D);
        // H[V][K] rows are 512 B; each K half is one contiguous 256 B run, so the
        // two packed [128][64] halves are two 2D copies (128 x 256 B blocks).
        AscendC::DataCopyParams hLoad;
        hLoad.blockCount = valueRows;
        hLoad.blockLen = 8;
        hLoad.srcStride = 8;
        hLoad.dstStride = 0;
        AscendC::DataCopy(h0, g[valueStart * 128], hLoad);
        AscendC::DataCopy(h1, g[valueStart * 128 + kHalf], hLoad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ---- sliding raw history: 3 rows before the sequence --------------------------
    {
        const uint16_t dBlocks = static_cast<uint16_t>(D * 2u / 32u);
        const uint16_t gap = static_cast<uint16_t>((P - D) / 16u);
        const size_t headBase = record * p.convStride + static_cast<size_t>(head) * D;
        AscendC::DataCopyParams load{9, dBlocks, gap, 0};
        if constexpr (SplitValues) {
            AscendC::GlobalTensor<bfloat16_t> history;
            history.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(p.out + uint64_t(p.tokens) * P * sizeof(float)));
            AscendC::DataCopy(rawB, history[(uint64_t(b) * N + head) * 9 * D], 9 * D);
        } else {
            AscendC::GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.sourceConvStride ? p.sourceConv : p.convState));
        const size_t sourceBase = slot * (p.sourceConvStride ? p.sourceConvStride : p.convStride) + size_t(head)*D;
        AscendC::DataCopy(rawB, source[sourceBase + static_cast<size_t>(hs) * threeP], load);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(cache, rawB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(3u * 384u));
        AscendC::PipeBarrier<PIPE_ALL>();
        // conv_state[slot, 0:2] = history[1:3]
        AscendC::Cast(rawB, cache[384], AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(2u * 384u));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyParams store{6, dBlocks, 0, gap};
        if (valueStart == 0)
            AscendC::DataCopy(gmCs[headBase], rawB, store);
    }

    // ---- chunk loop -------------------------------------------------------
    // beta_raw is BF16-rounded in stage P; sigmoid it per chunk.  The load must
    // be *inside* the chunk loop (the previous version loaded only the group's
    // first CH tokens, so every chunk after the first reused them).
    const bool betaBatch = (N <= 128u);
    const uint64_t replayBase = static_cast<uint64_t>(head) * p.replayCapacity * D;
    // The step is the chunk length actually executed (`cl`), not Layout::kChunk: a chunk
    // clamped at a tile edge is shorter and the remaining tokens of the range
    // must still be processed (T02 small-S ranges can cross one tile edge).
    // The first chunk's up-front raw load writes rawB, so the preamble's history
    // write-back (an MTE3 read of rawB) must have retired first.
    AscendC::PipeBarrier<PIPE_ALL>();
    uint32_t t0 = 0u;
    while (t0 < n) {
        const uint32_t gs = seqStart + t0;
        if constexpr (StreamTiles) {
            if ((t0 == 0u || gs % kTileRows == 0u) && (p.skipSync & 1u) == 0u) {
                Sync::WaitEq(p.counts, CounterByteOffset(0u, gs / kTileRows, p.tiles), p.nProj);
            }
        }
        const uint32_t tileRowEnd = (gs / kTileRows + 1u) * kTileRows;
        uint32_t cl = (n - t0 < Layout::kChunk) ? (n - t0) : Layout::kChunk;
        if (gs + cl > tileRowEnd) {
            // Raw rings are tile-indexed, so a chunk never crosses a tile edge.
            cl = tileRowEnd - gs;
        }
        const uint32_t rowSlot = (gs / kTileRows) % p.slotsPR;
        const uint32_t ringRow = gs % kTileRows;
        const uint32_t m384 = cl * 384u;
        const uint32_t mD = cl * D;

        // ---- fetch everything this chunk needs with a single fence --------
        // Four independent GM reads (beta_raw slice, this chunk's raw qkv rows,
        // decay_raw rows, gate_raw rows) plus the sliding-window UB copy are all
        // issued here.  One fence then covers the whole latency, instead of the
        // four separate fences the old per-reader form needed.  The conv_state
        // write-back is issued *after* the fence (it only reads rawB and writes
        // GM, so it never blocks the vector work).
        const uint32_t dBlocks16 = static_cast<uint16_t>(D * 2u / 32u);
        const uint16_t srcGapP = static_cast<uint16_t>((P - D) * 2u / 32u);
        const size_t rowSrcBase = (static_cast<size_t>(rowSlot) * kTileRows + ringRow);
        if (betaBatch) {
            const uint32_t betaWords = cl * N;
            const uint32_t copyElems = Align16(betaWords);
            const size_t src = (rowSlot * kTileRows + ringRow) * N;
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint16_t>(copyElems * 2u / 32u);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopy(betaB, gmBeta[src], params);
        }
        const bool loadRaw = (p.skipSync & 8192u) == 0u;
        if (loadRaw) {
            AscendC::DataCopyParams copy{static_cast<uint16_t>(3u * cl), static_cast<uint16_t>(dBlocks16), srcGapP, 0};
            AscendC::DataCopy(rawB, gmQkv[rowSrcBase * threeP + static_cast<size_t>(head) * D], copy);
        }
        {
            const size_t src = rowSrcBase * P + static_cast<size_t>(head) * D;
            AscendC::DataCopyParams params;
            params.blockCount = static_cast<uint16_t>(cl);
            params.blockLen = dBlocks16;
            params.srcStride = srcGapP;
            params.dstStride = 0;
            AscendC::DataCopy(drB, gmDr[src], params);
            if constexpr (!SplitValues)
                AscendC::DataCopy(goB, gmGo[src], params);
        }
        AscendC::DataCopy(rawF, cache, static_cast<int32_t>(3u * 384u));
        AscendC::PipeBarrier<PIPE_ALL>();
        if (loadRaw) {
            // write this call's raw rows back into conv_state[slot, 2+t]
            const uint32_t csRow = 2u + t0;
            const size_t dst = record * p.convStride + static_cast<size_t>(csRow) * threeP + static_cast<size_t>(head) * D;
            AscendC::DataCopyParams copy{static_cast<uint16_t>(3u * cl), static_cast<uint16_t>(dBlocks16), 0, srcGapP};
            if (valueStart == 0)
                AscendC::DataCopy(gmCs[dst], rawB, copy);
        }

        // ---- sliding window: cache rows + this chunk's raw rows -----------
        if (loadRaw) {
            AscendC::Cast(rawF[3u * 384u], rawB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(m384));
            VBar(p);

            // ---- conv1d over the 384 components, all CH tokens at once --------
            // conv[t][c] = sum_tap rawF[(tap+t)*384 + c] * convWf[tap][c]; the weight
            // row is broadcast across the cl token rows (src1 repeat stride 0).
            const AscendC::BinaryRepeatParams rowParams48(1, 1, 1, 48, 48, 0);
            for (uint32_t j = 0; j < 6u; ++j) {
                AscendC::Mul(conv[j * kHalf], rawF[j * kHalf], convWf[j * kHalf], 64, cl, rowParams48);
            }
            for (uint32_t tap = 1u; tap < 4u; ++tap) {
                for (uint32_t j = 0; j < 6u; ++j) {
                    AscendC::MulAddDst(conv[j * kHalf], rawF[tap * 384u + j * kHalf], convWf[tap * 384u + j * kHalf],
                                       64, cl, rowParams48);
                }
            }
            // silu + BF16 rounding of the convolved value (golden: convolved = BF16(silu(.)))
            VBar(p);
            AscendC::Muls(tmp, conv, -1.0f, static_cast<int32_t>(m384));
            VBar(p);
            AscendC::Exp(tmp, tmp, static_cast<int32_t>(m384));
            VBar(p);
            AscendC::Adds(tmp, tmp, 1.0f, static_cast<int32_t>(m384));
            VBar(p);
            AscendC::Div(tmp, ones, tmp, static_cast<int32_t>(m384));
            VBar(p);
            AscendC::Mul(conv, conv, tmp, static_cast<int32_t>(m384));
            VBar(p);
            // Round-trip through BF16 in rawB (the raw rows were already consumed by
            // the cast above, so the buffer is free here).  The fence first makes sure
            // the conv_state write-back finished reading rawB.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(rawB, conv, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(m384));
            VBar(p);
            AscendC::Cast(conv, rawB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(m384));
            VBar(p);

            // ---- split q / k / v ---------------------------------------------
            AscendC::DataCopyParams split;
            split.blockCount = static_cast<uint16_t>(cl);
            split.blockLen = static_cast<uint16_t>(D * 4u / 32u);
            split.srcStride = static_cast<uint16_t>((384u - D) * 4u / 32u);
            split.dstStride = 0;
            AscendC::DataCopy(qV, conv, split);
            AscendC::DataCopy(kV, conv[D], split);
            AscendC::DataCopy(vV, conv[2u * D], split);
            VBar(p);
        }

        // ---- L2 normalisation of q and k (batched row sums) ---------------
        // The two 64-wide K halves of every row are reduced with the two-level
        // block reduce (no packing copies, no fence), then added.
        {
            AscendC::LocalTensor<float> s1 = vecs[kVecS1].template ReinterpretCast<float>();
            AscendC::LocalTensor<float> s2 = vecs[kVecS2].template ReinterpretCast<float>();
            AscendC::LocalTensor<float> r1 = vecs[kVecR1].template ReinterpretCast<float>();
            AscendC::LocalTensor<float> r2 = vecs[kVecR2].template ReinterpretCast<float>();
            AscendC::LocalTensor<float> part = ub[Layout::kUbPack].template ReinterpretCast<float>();
            for (uint32_t pass = 0; pass < 2u; ++pass) {
                AscendC::LocalTensor<float> srcv = (pass == 0u) ? qV : kV;
                AscendC::LocalTensor<float> dstS = (pass == 0u) ? s1 : r1;
                AscendC::LocalTensor<float> dstS2 = (pass == 0u) ? s2 : r2;
                AscendC::Mul(tmp, srcv, srcv, static_cast<int32_t>(mD));
                VBar(p);
                RowReduce128(dstS, dstS2, tmp, part, cl);
                VBar(p);
            }
            // inv = rsqrt(sum + 1e-6) with 8-wide vector ops
            const uint32_t vecs8 = 8u;
            AscendC::Duplicate(ones, 1.0f, static_cast<int32_t>(vecs8));
            VBar(p);
            AscendC::Adds(s1, s1, 1e-6f, static_cast<int32_t>(cl));
            AscendC::Adds(r1, r1, 1e-6f, static_cast<int32_t>(cl));
            VBar(p);
            AscendC::Sqrt(s1, s1, static_cast<int32_t>(vecs8));
            AscendC::Sqrt(r1, r1, static_cast<int32_t>(vecs8));
            VBar(p);
            AscendC::Div(s1, ones, s1, static_cast<int32_t>(vecs8));
            AscendC::Div(r1, ones, r1, static_cast<int32_t>(vecs8));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t t = 0; t < cl; ++t) {
                const float invq = s1.GetValue(t) * p.scale;
                const float invk = r1.GetValue(t);
                AscendC::Muls(qV[static_cast<uint32_t>(t) * D], qV[static_cast<uint32_t>(t) * D], invq,
                              static_cast<int32_t>(D));
                AscendC::Muls(kV[static_cast<uint32_t>(t) * D], kV[static_cast<uint32_t>(t) * D], invk,
                              static_cast<int32_t>(D));
            }
            VBar(p);
        }

        // ---- decay_raw -> dec (per token, D lanes) ------------------------
        {
            AscendC::Cast(decV, drB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(mD));
            VBar(p);
            const auto bias = ub[Layout::kUbBias].template ReinterpretCast<float>();
            const AscendC::BinaryRepeatParams broadcast(1, 1, 1, 16, 16, 0);
            AscendC::Add(decV, decV, bias, 64, cl, broadcast);
            AscendC::Add(decV[64], decV[64], bias[64], 64, cl, broadcast);
            VBar(p);
            AscendC::Muls(decV, decV, -alpha, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Exp(decV, decV, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Adds(decV, decV, 1.0f, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Div(decV, ones, decV, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Muls(decV, decV, p.lowerBound, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Exp(decV, decV, static_cast<int32_t>(mD));
            VBar(p);
        }

        // ---- gate_raw -> sigmoid(ogate) ----------------------------------
        if constexpr (!SplitValues) {
            AscendC::Cast(ogV, goB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Muls(ogV, ogV, -1.0f, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Exp(ogV, ogV, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Adds(ogV, ogV, 1.0f, static_cast<int32_t>(mD));
            VBar(p);
            AscendC::Div(ogV, ones, ogV, static_cast<int32_t>(mD));
            VBar(p);
            // fold output_norm_weight into the gate tile (saves one token op)
            const auto gamma = ub[Layout::kUbGamma].template ReinterpretCast<float>();
            const AscendC::BinaryRepeatParams broadcast(1, 1, 1, 16, 16, 0);
            AscendC::Mul(ogV, ogV, gamma, 64, cl, broadcast);
            AscendC::Mul(ogV[64], ogV[64], gamma[64], 64, cl, broadcast);
            VBar(p);
        }

        // ---- beta (sigmoid of the BF16 raw value) ------------------------
        float betaLocal[Layout::kChunk];
        if (betaBatch) {
            AscendC::Cast(betaV, betaB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(Align16(cl * N)));
            VBar(p);
            AscendC::Muls(betaV, betaV, -1.0f, static_cast<int32_t>(Align16(cl * N)));
            VBar(p);
            AscendC::Exp(betaV, betaV, static_cast<int32_t>(Align16(cl * N)));
            VBar(p);
            AscendC::Adds(betaV, betaV, 1.0f, static_cast<int32_t>(Align16(cl * N)));
            VBar(p);
            AscendC::Div(betaV, ones, betaV, static_cast<int32_t>(Align16(cl * N)));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t t = 0; t < cl; ++t) {
                betaLocal[t] = betaV.GetValue(t * N + head);
            }
        } else {
            // Large N: read one 32 B aligned block around each beta element.
            for (uint32_t t = 0; t < cl; ++t) {
                const size_t rowRing = static_cast<size_t>(rowSlot) * kTileRows + ringRow + t;
                const size_t byteOff = rowRing * static_cast<size_t>(N) * 2u + static_cast<size_t>(head) * 2u;
                const size_t blockOff = byteOff & ~static_cast<size_t>(31u);
                const uint32_t inBlock = static_cast<uint32_t>(byteOff - blockOff) / 2u;
                AscendC::GlobalTensor<bfloat16_t> one;
                one.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.wBetaRaw + blockOff));
                AscendC::DataCopyParams bp;
                bp.blockCount = 1;
                bp.blockLen = 1;
                bp.srcStride = 0;
                bp.dstStride = 0;
                AscendC::DataCopy(betaB, one, bp);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(betaV, betaB, AscendC::RoundMode::CAST_NONE, 16);
                VBar(p);
                AscendC::Muls(betaV, betaV, -1.0f, 16);
                VBar(p);
                AscendC::Exp(betaV, betaV, 16);
                VBar(p);
                AscendC::Adds(betaV, betaV, 1.0f, 16);
                VBar(p);
                AscendC::Div(betaV, ones, betaV, 16);
                AscendC::PipeBarrier<PIPE_ALL>();
                betaLocal[t] = betaV.GetValue(inBlock);
            }
        }

        // ---- per-token recursion -----------------------------------------
        // Everything that only depends on the chunk is hoisted above; r = H@q is
        // written straight into the per-chunk r buffer and the RMSNorm + output
        // gate epilogue is batched after the loop, so no full-pipe scalar fence
        // sits between two consecutive steps of the recursion.
        AscendC::Cast(kHalfBuf, kV, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(mD));
        for (uint32_t t = 0; t < cl; ++t) {
            const uint32_t kvOff = t * D;
            AscendC::LocalTensor<float> kRow = kV[kvOff];
            AscendC::LocalTensor<float> qRow = qV[kvOff];
            AscendC::LocalTensor<float> vRow = vV[kvOff];
            AscendC::LocalTensor<float> decRow = decV[kvOff];
            const float beta = betaLocal[t];

            VBar(p);
            AscendC::Mul(h0, h0, decRow, 64, valueRows, packParams);
            AscendC::Mul(h1, h1, decRow[64], 64, valueRows, packParams);
            VBar(p);
            AscendC::LocalTensor<float> sVec = sVecBig;
            // s = (H*dec) @ k
            AscendC::Mul(scratch, h0, kRow, 64, valueRows, packParams);
            VBar(p);
            AscendC::MulAddDst(scratch, h1, kRow[64], 64, valueRows, packParams);
            VBar(p);
            if ((p.skipSync & 131072u) == 0u) {
                RowReduce64(sVec, scratch, tmp, valueRows);
            }
            VBar(p);
            // Keep the correction in its replay slot for the state update and the chunk store.
            AscendC::Sub(tmp, vRow[valueStart], sVec, valueRows);
            VBar(p);
            AscendC::Muls(corrBuf[t * D + valueStart], tmp, beta, valueRows);
            VBar(p);
            // H += outer(corr, k): Brcb already produces a [128][8] tile whose
            // 32 B row v holds corr[v] eight times, so the product can be added
            // with block stride 0 (all eight 32 B blocks of a row read that one
            // row) instead of materialising a full [128][64] row-repeat tile.
            // Drops the 32 KiB broadcast write and its two reads per token.
            AscendC::Brcb(tmp, corrBuf[t * D + valueStart], valueRows / 8, AscendC::BrcbRepeatParams(1, 8));
            VBar(p);
            AscendC::MulAddDst(h0, tmp, kRow, 64, valueRows, bcastParams);
            AscendC::MulAddDst(h1, tmp, kRow[64], 64, valueRows, bcastParams);
            VBar(p);
            // r = H @ q, stashed per token for the batched epilogue
            AscendC::Mul(scratch, h0, qRow, 64, valueRows, packParams);
            VBar(p);
            AscendC::MulAddDst(scratch, h1, qRow[64], 64, valueRows, packParams);
            VBar(p);
            if ((p.skipSync & 262144u) == 0u) {
                RowReduce64(rBuf[t * D + valueStart], scratch, tmp, valueRows);
            }
        }

        if constexpr (SplitValues) {
            // Each AIV owns disjoint value rows. Only the first writer publishes head-wide K and decay.
            AscendC::GlobalTensor<float> records;
            records.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.out));
            AscendC::Adds(corrBuf, corrBuf, 0.f, cl * D);
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t offset = uint64_t(gs) * P + uint64_t(head) * D + valueStart;
            AscendC::DataCopyParams emit{uint16_t(cl), uint16_t(valueRows / 8), uint16_t((D - valueRows) / 8),
                                         uint16_t((P - valueRows) / 8)};
            AscendC::DataCopy(records[offset], rBuf[valueStart], emit);
            emit.dstStride = uint16_t((D - valueRows) / 8);
            AscendC::DataCopy(gmRu[uint64_t(slot) * p.uStride + replayBase + uint64_t(t0) * D + valueStart],
                              corrBuf[valueStart], emit);
            if (valueStart == 0) {
                AscendC::DataCopy(gmRk[uint64_t(slot) * p.keyStride + replayBase + uint64_t(t0) * D], kHalfBuf, cl * D);
                AscendC::DataCopy(gmRd[uint64_t(slot) * p.decayStride + replayBase + uint64_t(t0) * D], decV, cl * D);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            // ---- batched RMSNorm + output gate over the chunk ------------------
            // mean_t = sum(r_t^2)/D + eps, inv_t = rsqrt(mean_t), gated = r * inv *
            // (sigmoid(ogate) * gamma), rounded to BF16.  The per-token scalar fence
            // of the old form is replaced by one fence per chunk.
            {
                const uint32_t clD = cl * D;
                if ((p.skipSync & 8388608u) == 0u) {
                    AscendC::Mul(r2Buf, rBuf, rBuf, static_cast<int32_t>(clD));
                    VBar(p);
                    AscendC::LocalTensor<float> sumA = ub[Layout::kUbSum].template ReinterpretCast<float>();
                    AscendC::LocalTensor<float> part = ub[Layout::kUbPack].template ReinterpretCast<float>();
                    RowReduce128(sumA, sumA[8], r2Buf, part, cl);
                    VBar(p);
                    AscendC::Muls(sumA, sumA, 0.0078125f, 8); // 1 / D, D = 128
                    VBar(p);
                    AscendC::Adds(sumA, sumA, p.rmsNormEps, 8);
                    VBar(p);
                    AscendC::Sqrt(sumA, sumA, 8);
                    VBar(p);
                    AscendC::Div(sumA, ones, sumA, 8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    float inv[Layout::kChunk];
                    for (uint32_t t = 0; t < cl; ++t) {
                        inv[t] = sumA.GetValue(t);
                        if (inv[t] > 1e6f) {
                            inv[t] = 1e6f; // rms(0) clamp, same as the scalar path
                        }
                    }
                    for (uint32_t t = 0; t < cl; ++t) {
                        AscendC::Muls(rBuf[static_cast<uint32_t>(t) * D], rBuf[static_cast<uint32_t>(t) * D], inv[t],
                                      static_cast<int32_t>(D));
                    }
                    VBar(p);
                    AscendC::Mul(rBuf, rBuf, ogV, static_cast<int32_t>(clD));
                    VBar(p);
                    AscendC::Cast(gatedBuf, rBuf, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(clD));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // ---- batched epilogue: replay_* and gated --------------------------
            if ((p.skipSync & 16777216u) == 0u) {
                // Preserve the replay format's positive zero after using the raw correction to update H.
                AscendC::Adds(corrBuf, corrBuf, 0.0f, static_cast<int32_t>(mD));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(gmRu[record * p.uStride + replayBase + static_cast<size_t>(t0) * D], corrBuf,
                                  static_cast<int32_t>(cl * D));
                AscendC::DataCopy(gmRk[record * p.keyStride + replayBase + static_cast<size_t>(t0) * D], kHalfBuf,
                                  static_cast<int32_t>(cl * D));
                AscendC::DataCopy(gmRd[record * p.decayStride + replayBase + static_cast<size_t>(t0) * D], decV,
                                  static_cast<int32_t>(cl * D));
                const uint32_t row0 = gs;
                const uint32_t slotG = (row0 / kTileRows) % p.slotsGated;
                const size_t gatedBase =
                    (static_cast<size_t>(slotG) * kTileRows + (row0 % kTileRows)) * P + static_cast<size_t>(head) * D;
                // One 2D copy for the whole chunk instead of one per token: the D
                // columns of a token are contiguous in both tiles, only the row
                // stride differs (P in the ring).
                AscendC::DataCopyParams gp;
                gp.blockCount = static_cast<uint16_t>(cl);
                gp.blockLen = static_cast<uint16_t>(D * 2u / 32u);
                gp.srcStride = 0;
                gp.dstStride = static_cast<uint16_t>((P - D) * 2u / 32u);
                AscendC::DataCopy(gmGated[gatedBase], gatedBuf, gp);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        // ---- cache update: next chunk needs rows cl .. cl+2 ----------------
        if (t0 + cl < n) {
            AscendC::DataCopy(cache, rawF[cl * 384u], static_cast<int32_t>(3u * 384u));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        t0 += cl;
        if constexpr (StreamTiles) {
            if (t0 == n || gs + cl == tileRowEnd) {
                Sync::AivAdd(stg, p.counts, CounterByteOffset(1u, gs / kTileRows, p.tiles), 1u);
            }
        }
    }
    // The last chunk's conv_state write-back is still in flight here; drain it so
    // the next sequence cannot overwrite rawB under the DMA reader.
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// Stage R driver: walks the global unit list, publishes per-tile credits.
CATLASS_DEVICE uint32_t RunStageR(V2Params const &p, AscendC::LocalTensor<uint8_t> const &ub)
{
    using namespace RDetail;
    const uint32_t rank = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum()) * 2u;
    AscendC::LocalTensor<int32_t> stg = ub[kUbStg].template ReinterpretCast<int32_t>();

    const uint32_t total = p.batch * p.n;
    HeadConstants cached;
    uint32_t lastTile = 0xFFFFFFFFu;
    uint32_t credits = 0u;
    // T01 puts each sequence entirely inside one tile. The global unit index
    // therefore enumerates (sequence, head) directly in prefix order.
    for (uint32_t u = rank; u < total; u += aivNum) {
        const uint32_t b = u / p.n, head = u % p.n;
        const uint32_t first = SequenceStart(p, b);
        const uint32_t end = SequenceStart(p, b + 1u);
        const uint32_t tile = first / kTileRows;
        if (tile != lastTile) {
            if (lastTile != 0xFFFFFFFFu) {
                Sync::AivAdd(stg, p.counts, CounterByteOffset(1u, lastTile, p.tiles), credits);
            }
            credits = 0u;
            lastTile = tile;
            if ((p.skipSync & 1u) == 0u) {
                Sync::WaitEq(p.counts, CounterByteOffset(0u, tile, p.tiles), static_cast<int32_t>(p.nProj),
                             (p.skipSync & 4u) != 0u ? 2000000u : 0u);
            }
        }
        ProcessSequence(p, ub, b, head, first, end, cached);
        ++credits;
    }
    if (lastTile != 0xFFFFFFFFu) {
        Sync::AivAdd(stg, p.counts, CounterByteOffset(1u, lastTile, p.tiles), credits);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return lastTile == 0xFFFFFFFFu ? 0u : 1u;
}

} // namespace MegaRecurrentKdaV2

#endif // MEGA_RECURRENT_KDA_V2_VECTOR_H
