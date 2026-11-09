/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_PLATFORM_H
#define CHUNK_KDA_FWD_PREPARE_PLATFORM_H

#include <cstdint>

namespace KdaPrepare {

// 平台 tier 标识，数值与 host 侧 tiling_processor.h 的 PrepareKernelTier 对齐
// （Arch22=0 / Arch35=1 / Arch92=2），作为 host 表驱动与 kernel 布局契约之间的
// 共同标识。新增平台只增不改。
enum class KdaPlatform : uint32_t {
    Arch22 = 0,
    Arch35 = 1,
    Arch92 = 2,
};

// 三段门控的唯一判定点：全工程散点 __NPU_ARCH__==9201 / __CCE_AICORE__==310
// 收敛到这一个编译期结构体。shared policy.h 里的 Shape/Workspace/L1/Arch35Ub
// 一律从这里取值，不再各自 #if。
struct PreparePlatform {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
    static constexpr KdaPlatform id = KdaPlatform::Arch92;  // A6(1:1)
    static constexpr uint32_t headsPerGroup = 3;
    static constexpr uint32_t aivPerAic = 1;
    static constexpr uint32_t headsPerAiv = 3;
    static constexpr uint32_t computeSlotCount = 3;
    static constexpr uint32_t slotWavesPerGroup = 1;
    static constexpr uint32_t ubCapacity = 0x5E000;        // 376 KiB
    static constexpr uint32_t computeSlotBytes = 0x1C000;
    static constexpr uint32_t stateBytes = 0x03000;
    static constexpr uint32_t workspaceSlotCount = 3;
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr KdaPlatform id = KdaPlatform::Arch35;  // A5(1:2)
    static constexpr uint32_t headsPerGroup = 4;
    static constexpr uint32_t aivPerAic = 2;
    static constexpr uint32_t headsPerAiv = 2;
    static constexpr uint32_t computeSlotCount = 2;
    static constexpr uint32_t slotWavesPerGroup = 1;
    static constexpr uint32_t ubCapacity = 0x3E000;        // 248 KiB
    static constexpr uint32_t computeSlotBytes = 0x1C000;
    static constexpr uint32_t stateBytes = 0x03000;
    static constexpr uint32_t workspaceSlotCount = 4;
#else
    static constexpr KdaPlatform id = KdaPlatform::Arch22;  // A2/A3
    static constexpr uint32_t headsPerGroup = 4;
    static constexpr uint32_t aivPerAic = 2;
    static constexpr uint32_t headsPerAiv = 2;
    static constexpr uint32_t computeSlotCount = 2;
    static constexpr uint32_t slotWavesPerGroup = 1;
    // arch22 使用独立的 Arch22Ub（192 KiB），此处仅为 trait 完整性/静态断言占位。
    static constexpr uint32_t ubCapacity = 0x3E000;
    static constexpr uint32_t computeSlotBytes = 0x1C000;
    static constexpr uint32_t stateBytes = 0x03000;
    static constexpr uint32_t workspaceSlotCount = 4;
#endif
};

// UB 布局模板：把「容量、计算槽/状态槽数量与起始地址」作为编译期常量集中表达，
// 供 arch35/arch92 共用（arch22 使用独立的 Arch22Ub）。各 Stage 内偏移是三维
// 架构共享的常数（见 policy.h 的 Arch35Ub），只有槽数/槽基地址随平台变化。
//
// 槽基地址用 constexpr 函数派生而非 static 数组成员：后者在 C++14 下被 odr 使用
// 时需要额外给出类外定义，模板类里写起来繁琐；共享消费方 Arch35Ub 用命名空间级
// constexpr 变量接住这些值（const 内置数组天然内部链接，每个 TU 都有副本）。
template <typename Platform>
struct UbLayout {
    static constexpr uint32_t kCapacity = Platform::ubCapacity;
    static constexpr uint32_t kComputeSlotCount = Platform::computeSlotCount;
    static constexpr uint32_t kComputeSlotBytes = Platform::computeSlotBytes;
    static constexpr uint32_t kStateBytes = Platform::stateBytes;
    static constexpr uint32_t kMaxSlots = 4;

    // 计算槽从 0 依次排开，每槽 kComputeSlotBytes；state 槽紧跟其后，按 slot 等距。
    static constexpr uint32_t ComputeSlotBase(uint32_t slot)
    {
        return slot * kComputeSlotBytes;
    }

    static constexpr uint32_t StateBase(uint32_t slot)
    {
        return kComputeSlotCount * kComputeSlotBytes + slot * kStateBytes;
    }

    // 槽数上限：表最多容纳 4 槽，覆盖 arch22/35/92 全部场景。
    static_assert(kComputeSlotCount <= kMaxSlots, "UB 计算槽数超过最大槽数 4");
    // 总容量不越界：全部计算槽 + 全部 state 槽必须落在 kCapacity 内。
    static_assert(
        kComputeSlotCount * kComputeSlotBytes +
            kComputeSlotCount * kStateBytes <=
        kCapacity,
        "UB 计算槽与 state 槽总容量越界");
    // 槽尺寸 4 KiB 对齐（0x1C000/0x03000 均为 0x1000 整数倍）。
    static_assert(kComputeSlotBytes % 0x1000 == 0 && kStateBytes % 0x1000 == 0,
                  "UB 槽尺寸须 4 KiB 对齐");
};

// arch35/arch92 实际使用的 UB 布局实例。
using PrepareUbLayout = UbLayout<PreparePlatform>;

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_PLATFORM_H