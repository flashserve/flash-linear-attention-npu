// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. A2 GDN Prepare 的 host/device 共用地址合同。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_POLICY_H
#define GDN_ARCH22_UNIFIED_PREPARE_POLICY_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define GDN_UP_CONSTEXPR __aicore__ constexpr
#else
#define GDN_UP_CONSTEXPR constexpr
#endif

namespace GdnUnifiedPrepare {

// 内部消费窗口独立于外层 HK task；缩小窗口不能重复生产共享 raw KKT。
constexpr uint32_t kHeadsPerWindow = 4;
// Host/device 共用的外层 HK task token-head 预算。
constexpr uint32_t kTaskTokenBudget = 512;

GDN_UP_CONSTEXPR uint64_t Align512(uint64_t x) { return (x + 511) / 512 * 512; }
GDN_UP_CONSTEXPR uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
GDN_UP_CONSTEXPR uint64_t Max(uint64_t a, uint64_t b) { return a > b ? a : b; }
GDN_UP_CONSTEXPR uint64_t CeilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

// 所有计量均为字节。每个 head 独占 record；换义不改变两个 AIV 的地址归属。
// X/d0/d1/result 与 Kbg/Vb 使用独立区间，允许完整 RHS 在最终 Solve 等待前写入。
struct WorkspaceLayout {
    uint64_t taskHeadCapacity;
    uint64_t slotHeadCapacity;
    uint64_t hkPerTask;
    uint64_t xOffset;
    uint64_t d0Offset;
    uint64_t d1Offset;
    uint64_t resultOffset;
    uint64_t kbgOffset;
    uint64_t vbOffset;
    uint64_t headBytes;
    uint64_t slotBytes;
    uint64_t scoreBytes;
    uint64_t slotsOffset;
    uint64_t tempOffset;
    uint64_t coreBytes;
};

GDN_UP_CONSTEXPR WorkspaceLayout MakeWorkspaceLayout(uint64_t bt, uint64_t v, uint64_t g)
{
    WorkspaceLayout p{};
    p.taskHeadCapacity = kTaskTokenBudget / bt;
    p.slotHeadCapacity = kHeadsPerWindow;
    p.hkPerTask = Max(1, p.taskHeadCapacity / g);
    p.xOffset = 0;
    p.d0Offset = bt * bt * sizeof(float);
    p.d1Offset = p.d0Offset + bt * (bt == 64 ? 16 : 64) * sizeof(float);
    p.resultOffset = p.d1Offset + bt * 32 * sizeof(float);
    const uint64_t solveEnd = p.resultOffset + bt * bt / 4 * sizeof(float);
    p.kbgOffset = Align512(solveEnd);
    p.vbOffset = p.kbgOffset + bt * 128 * 2;
    const uint64_t rhsEnd = p.vbOffset + bt * v * 2;
    p.headBytes = Align512(rhsEnd);
    p.slotBytes = p.slotHeadCapacity * p.headBytes;
    // 固定按最大 HK 数预留，host 的每组步长只依赖 BT/V；实际只写 hkPerTask 份。
    p.scoreBytes = Align512(p.taskHeadCapacity * bt * bt * sizeof(float));
    p.slotsOffset = p.scoreBytes;
    p.tempOffset = p.slotsOffset + 2 * p.slotBytes;
    p.coreBytes = Align512(p.tempOffset + bt * bt); // 一个最大 (BT/2)^2 FP32 GM 桥。
    return p;
}

GDN_UP_CONSTEXPR uint64_t WorkspaceBytes(uint64_t bt, uint64_t v, uint64_t g, uint64_t cores)
{
    return MakeWorkspaceLayout(bt, v, g).coreBytes * cores;
}

GDN_UP_CONSTEXPR uint64_t WorkspaceBytes(uint64_t bt, uint64_t v, uint64_t cores)
{
    return MakeWorkspaceLayout(bt, v, 1).coreBytes * cores;
}

GDN_UP_CONSTEXPR uint64_t WorkItems(uint64_t batch, uint64_t nt, uint64_t hk, uint64_t g, uint64_t bt)
{
    return batch * nt * CeilDiv(hk, Max(1, (kTaskTokenBudget / bt) / g));
}

// 新 Prepare 只读取自身需要的输入和最终跨 kernel 输出，不接受旧 stage/tiling 对象。
// Address 在 device 实例化为 GM_ADDR；纯 host 可用 uint8_t* 或 uintptr_t 检查合同。
template <class Address>
struct Arguments {
    Address k;
    Address v;
    Address beta;          // kernel 有效调用已转换为 FP32 BHT。
    Address rawG;          // kPrepared=true: FP32 BTH；false: FP32 BHT。
    Address cuSeqlens;
    Address chunkIndices;  // canonical [NT,2]，varlen 时 B=1。
    Address gBht;          // FP32 [B,Hv,T]，H/O 公开跨 kernel 输入。
    Address gBth;          // 可空，FP32 [B,T,Hv]，outputG 控制。
    Address gDecay;        // kPrepared=true: exp(g_last-g)，FP32 BHT。
    Address gLastExp;      // kPrepared=true: [B*NT,alignUp(Hv,8)]。
    Address a;             // InputT [B,Hv,T,BT]。
    Address w;             // InputT [B,Hv,T,128]。
    Address u;             // InputT [B,Hv,T,V]。
    Address scratch;       // 独立连续区：WorkspaceBytes(BT,V,Hv/Hk,GetBlockNum())。
    uint64_t batch;
    uint64_t hk;
    uint64_t hv;
    uint64_t tokens;
    uint64_t nt;
    uint64_t isVarlen;
    uint64_t outputG;
    // 仅Norm模板读取；k在该模板中指向低精度Khat，原始K单独保留。
    Address normQ{};
    Address normK{};
    Address qHat{};
    Address qRstd{};
    Address kRstd{};
    // 新特性独立下传；标量参数与有效beta均保留FP32，betaEff为内部BHT。
    Address aLog{};        // useGate=true时[Hv]。
    Address dtBias{};      // 可空[Hv]，只在useGate=true时读取。
    Address betaEff{};     // 可空FP32 [B,Hv,T]，每个head由原owner唯一写入。
    uint64_t useGate{};
    uint64_t useBetaSigmoid{};
    uint64_t allowNegEigval{};
    uint64_t useExp2{};
};

namespace Local {
constexpr uint32_t kUbBytes = 184 * 1024;
constexpr uint32_t kL1Bytes = 496 * 1024;
constexpr uint32_t kL0Bytes = 64 * 1024;
constexpr uint32_t kL0CBytes = 128 * 1024;
// UB 常驻 context 共 24 KiB。前 4 KiB 为 g/beta，剩余空间用于 prefix 转置和索引。
constexpr uint32_t kG = 0;
constexpr uint32_t kBeta = 2048;
constexpr uint32_t kContextEnd = 24 * 1024;
constexpr uint32_t kLeaf = kContextEnd;
// 这是物理空间包络，不是活跃叶批量；BT128 最终 InputT A 仍需完整 32 KiB。
constexpr uint32_t kLeafBytes = 32 * 256 * sizeof(float);
constexpr uint32_t kArena = kLeaf + kLeafBytes;
// leaf 存活时 arena 作为 prod/epilogue 工作区；leaf 发布后 [kLeaf,kUbBytes) 换义。
constexpr uint32_t kProdBytes = kLeafBytes;
constexpr uint32_t kLeafRow = kArena + kProdBytes;
constexpr uint32_t kLeafRowBytes = 32 * 16 * sizeof(float);

// L1 前 240 KiB 属于通用 GEMM lane；后 256 KiB 是 C7 专用双 head RHS/A lane。
// 两个区域从不在一轮 work 中换义重叠；rhsLaneFree 在最后一次 MTE1 读后返回。
constexpr uint32_t kGemmA = 0;
constexpr uint32_t kGemmB = 64 * 1024;
constexpr uint32_t kRhsA = 240 * 1024;
constexpr uint32_t kRhsK = kRhsA + 64 * 1024;
constexpr uint32_t kRhsV = kRhsK + 64 * 1024;
} // namespace Local

} // namespace GdnUnifiedPrepare

#undef GDN_UP_CONSTEXPR
#endif
