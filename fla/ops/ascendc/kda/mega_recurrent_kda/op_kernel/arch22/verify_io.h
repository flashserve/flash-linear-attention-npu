#pragma once
namespace MegaRecurrentKdaV2 {
// A full width decode batch is already BSH. Captured graphs can replay with
// shorter requests or dummy rows; decide on device for the current replay.
CATLASS_DEVICE bool IsDenseVerifyBatch(const V2Params &p, uint32_t inputTokens)
{
    if (inputTokens != p.tokens) return false;
    auto starts = reinterpret_cast<__gm__ const int32_t *>(p.verifyStarts);
    auto slots = reinterpret_cast<__gm__ const int32_t *>(p.stateIndices);
    for (uint32_t b = 0; b < p.batch; ++b) {
        if (starts[b] != int32_t(b * p.sequence) || slots[b] <= 0) return false;
    }
    return starts[p.batch] == int32_t(inputTokens);
}

template<bool Restore>
CATLASS_DEVICE void CopyVerifyRows(const V2Params &p, GM_ADDR src, GM_ADDR dst, uint32_t inputTokens)
{
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, kHidden * sizeof(uint16_t));
        auto row = buffer.Get<uint16_t>();
        auto in = Int8Qkv::Global<uint16_t>(src);
        auto out = Int8Qkv::Global<uint16_t>(dst);
        auto starts = reinterpret_cast<__gm__ const int32_t *>(p.verifyStarts);
        auto slots = reinterpret_cast<__gm__ const int32_t *>(p.stateIndices);
        for (uint32_t t = AscendC::GetBlockIdx(); t < (Restore ? inputTokens : p.tokens); t += p.aivNum) {
            int32_t origin = -1;
            if constexpr (Restore) {
                for (uint32_t b = 0; b < p.batch; ++b) {
                    if (slots[b] > 0 && int32_t(t) >= starts[b] && int32_t(t) < starts[b+1]) {
                        origin = b*p.sequence + t - starts[b];
                        break;
                    }
                }
            } else {
                const uint32_t index = uint32_t(starts[t/p.sequence]) + t%p.sequence;
                origin = index < inputTokens ? index : inputTokens-1;
            }
            if (origin >= 0 && uint32_t(origin) < (Restore ? p.tokens : inputTokens)) {
                AscendC::DataCopy(row, in[uint64_t(origin)*kHidden], kHidden);
            } else {
                AscendC::Duplicate(row, uint16_t(0), kHidden);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(out[uint64_t(t)*kHidden], row, kHidden);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}
} // namespace MegaRecurrentKdaV2
