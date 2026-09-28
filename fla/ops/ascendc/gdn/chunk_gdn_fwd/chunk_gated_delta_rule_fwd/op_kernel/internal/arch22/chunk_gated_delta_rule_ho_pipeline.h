/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_HO_PIPELINE_H
#define CHUNK_GATED_DELTA_RULE_HO_PIPELINE_H

#include <cstdint>
#include "kernel_operator.h"

// H/O 空闲流水通知协议（arch22 私有组件，仅协议原语，由 arch22 统一入口接入）。
// 契约：A2 mixed 下 IBSet/IBWait<false> 仅 AIV 可用，槽 GM 索引为
// blockNum*8*eventID + blockIdx*8（blockNum = 2*GetBlockNum()），每个 32B 槽
// （8 个 int32）对应一个物理 AIV；IBWait 消费标志后清零，故每槽单消费者、
// 单次发布。eventID 固定 0，chunk 维度用 bank 子视图区分。ready 为 GM 张量，
// 容量 readyBankCount*2*C*32B，由 host 独立预留；scratch 为本核 UB 内 32B，
// 由调用者在其安全区提供。本组件不分配 UB/TPipe、不占事件号、不扫 cu_seqlens。
namespace GdnHoPipeline {

using namespace AscendC;

constexpr uint32_t kSlotInt32 = 8;  // 32B 槽 = 8 个 int32
constexpr uint32_t kEventId = 0;    // 固定 eventID 0；chunk 维度走 bank 子视图

// 真实准入与其余前置由统一 caller 在 enabled 前明确负责（本组件不重复、不靠
// 数值猜测）：hasMultiChunkSequence、无 gk、V 块命中 V128/V256、BT 命中
// BT64/BT128、GM ready 与 UB scratch 安全区已由 host 独立预留/分离。
struct HoPipelineConfig {
    bool enabled;             // caller 完成全部准入后置 true
    uint32_t cubeCoreNum;     // C = GetBlockNum()；物理 AIV 编号域 [0, 2C)
    uint32_t sequenceNum;     // S：非空（紧凑化后）序列数
    uint32_t headNum;         // Hv
    uint32_t vBlockNum;       // VB：H 侧 vBlock 切分数
    uint32_t producerCount;   // P = S*Hv*VB；首版要求 0 < P < C
    uint32_t readyBankCount;  // bank 数（>= 1）；每个 localChunk 一个 bank
};

// 结构前置：关闭态恒真；启用时 C>0、0 < P = S*Hv*VB < C、readyBankCount >= 1。
// bank 数仅为结构下界，不证明存在多 chunk 序列（多条单 chunk 序列同样满足）。
__aicore__ inline bool HoPipelineValid(const HoPipelineConfig &cfg)
{
    if (!cfg.enabled) {
        return true;
    }
    const uint64_t p = static_cast<uint64_t>(cfg.sequenceNum) * cfg.headNum * cfg.vBlockNum;
    return cfg.cubeCoreNum > 0 && p > 0 && p == cfg.producerCount && p < cfg.cubeCoreNum &&
           cfg.readyBankCount >= 1;
}

// 仅 AIV：A2 mixed 下 AIV 的 GetBlockIdx() 即物理 AIV 编号 [0, 2C)。
__aicore__ inline uint32_t HoPhysicalAivIdx()
{
    return GetBlockIdx();
}

// 仅 AIV：物理核组号（配对 AIC 索引）= 物理 AIV 编号 / GetSubBlockNum()。
__aicore__ inline uint32_t HoCoreGroupIdx()
{
    return GetBlockIdx() / GetSubBlockNum();
}

// 通知协议类：H（生产者）与 O（消费者）共同调用。caller 排布约定：H 任务 t
// （vBlock-major：t = vBlock*(S*Hv) + seq*Hv + head）排在核组 [0, P) 的组 t，
// 其子核 s 的物理 AIV 编号即发布槽 2*t + s；O 任务经逻辑消费者 id 排在空闲
// 核组后缀 [P, C)，物理位置不参与槽身份。所有方法仅 AIV、且 enabled &&
// HoPipelineValid(cfg) 时调用；本组件不做局部跳过（局部忽略发布会让消费者
// 永久等待）。
class HoNotifyPipeline {
public:
    __aicore__ inline HoNotifyPipeline(const HoPipelineConfig &config, const GlobalTensor<int32_t> &readyGm)
        : cfg(config), ready(readyGm)
    {
    }

    __aicore__ inline bool Enabled() const
    {
        return cfg.enabled;
    }

    __aicore__ inline uint32_t ProducerCount() const
    {
        return cfg.producerCount;
    }

    // H 侧查询：生产者子核，物理 AIV 编号 [0, 2P)；关闭态恒 false。
    __aicore__ inline bool IsProducerAiv() const
    {
        return cfg.enabled && HoPhysicalAivIdx() < cfg.producerCount * 2u;
    }

    // O 侧查询：消费者子核，空闲后缀物理 AIV 编号 [2P, 2C)，与生产者互斥且
    // 并集覆盖全部物理 AIV；关闭态恒 false。
    __aicore__ inline bool IsConsumerAiv() const
    {
        return cfg.enabled && HoPhysicalAivIdx() >= cfg.producerCount * 2u;
    }

    // 初始化（仅 AIV）：本 AIV 把所有 bank 的自身 32B 槽清零。Duplicate 后、
    // 首个 DataCopy 前用 PIPE_ALL 闭合 V->MTE3 跨 pipe 依赖，末尾 PIPE_ALL 等
    // 待本核 MTE3 写完。本函数不做 SyncAll：跨核“先清零、后收发”顺序由
    // caller 随后的统一原 H 初始化屏障（覆盖全部 AIC/AIV）建立。
    __aicore__ inline void InitOwnSlots(const LocalTensor<int32_t> &scratch)
    {
        Duplicate(scratch, static_cast<int32_t>(0), kSlotInt32);
        PipeBarrier<PIPE_ALL>();
        const uint64_t own = static_cast<uint64_t>(HoPhysicalAivIdx()) * kSlotInt32;
        for (uint32_t bank = 0; bank < cfg.readyBankCount; ++bank) {
            DataCopy(ready[BankOffset(bank) + own], scratch, kSlotInt32);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // 发布（仅 AIV，H 生产者子核）：槽号 = 2*producerTask + 本子核 s；首版每
    // 槽只发一次、不复用；合法范围由统一配置与 caller 保证，不做局部忽略。
    __aicore__ inline void Publish(uint32_t compactSequence, uint32_t head, uint32_t vBlock,
                                   uint32_t localChunk, const LocalTensor<int32_t> &scratch)
    {
        const uint32_t producerTask = vBlock * (cfg.sequenceNum * cfg.headNum) +
                                      compactSequence * cfg.headNum + head;
        const uint32_t producerAiv = producerTask * 2u + GetSubBlockIdx();
        IBSet<false>(ready[BankOffset(localChunk)], scratch, producerAiv, kEventId);
    }

    // 等待（仅 AIV，O 消费者子核 s）：只等同名 (seq, head) 的 VB 个 vBlock 槽，
    // 各 IBWait 一次，两个 O 子核不重复等全部槽；其会合仍由原 O barrier +
    // vec1Done 完成，本组件不替代该边。每槽单消费者、单次消费。
    __aicore__ inline void Wait(uint32_t compactSequence, uint32_t head, uint32_t localChunk,
                                const LocalTensor<int32_t> &scratch)
    {
        const uint32_t name = compactSequence * cfg.headNum + head;
        const uint32_t stride = cfg.sequenceNum * cfg.headNum;
        for (uint32_t vBlock = 0; vBlock < cfg.vBlockNum; ++vBlock) {
            const uint32_t producerTask = vBlock * stride + name;
            IBWait<false>(ready[BankOffset(localChunk)], scratch, producerTask * 2u + GetSubBlockIdx(),
                          kEventId);
        }
    }

private:
    // ready 为 GM 张量，容量 readyBankCount*2*C*32B 由 host 独立预留（与本核
    // UB scratch 安全区无大小比较关系）；本组件不校验 GM 越界。偏移先提升 64 位。
    __aicore__ inline uint64_t BankOffset(uint32_t localChunk) const
    {
        return static_cast<uint64_t>(localChunk) * cfg.cubeCoreNum * 2u * kSlotInt32;
    }

    HoPipelineConfig cfg;
    GlobalTensor<int32_t> ready;
};

}  // namespace GdnHoPipeline

#endif  // CHUNK_GATED_DELTA_RULE_HO_PIPELINE_H
