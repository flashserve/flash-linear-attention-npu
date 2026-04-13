/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file platform_stubs.cpp
 * \brief
 */

#include "tests/utils/platform.h"
#include "platform/platform_info.h"

using Platform = ops::adv::tests::utils::Platform;

namespace fe {

uint32_t PlatFormInfos::GetCoreNum(void) const
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        uint32_t coreNum = 0;
        return platform->socSpec.Get("SoCInfo", "ai_core_cnt", coreNum) ? coreNum : 0;
    } else {
        return 0;
    }
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        return platform->socSpec.Get(label.c_str(), key.c_str(), val);
    } else {
        return false;
    }
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val)
{
    return GetPlatformRes(label, key, val);
}

void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        std::string memType = std::to_string(static_cast<int32_t>(mem_type));
        (void)platform->socSpec.Get("LocalMemSize", memType.c_str(), size);
    } else {
        size = 0;
    }
}

void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        std::string memType = std::to_string(static_cast<int32_t>(mem_type));
        (void)platform->socSpec.Get("LocalMemBw", memType.c_str(), bw_size);
    } else {
        bw_size = 0;
    }
}

PlatformInfoManager& PlatformInfoManager::Instance() {
    static PlatformInfoManager pf;
    return pf;
}

uint32_t PlatformInfoManager::GetPlatformInstanceByDevice(const uint32_t &device_id, PlatFormInfos &platform_infos)
{
    return 0;
}

uint32_t PlatformInfoManager::InitializePlatformInfo()
{
    return 0;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatformInfo &platform_info,
                                                               OptionalInfo &opti_info) {
    opti_info.soc_version = "910B";
    return 0;
}

PlatformInfoManager::PlatformInfoManager() {}
PlatformInfoManager::~PlatformInfoManager() {}

} // namespace fe


namespace op {
// add stub func for GetCurrentPlatformInfo()&GetSocVersion() in aclnn
enum class SocVersion {
    ASCEND910B = 1,
    ASCEND910_95 = 3,
    ASCEND310P = 6
};

class PlatformInfo {
public:
    SocVersion GetSocVersion() const;
    fe::PlatFormInfos *GetPlatformInfos() const;
};

SocVersion PlatformInfo::GetSocVersion() const
{
    auto *platform = Platform::GetGlobalPlatform();
    std::string socVersion("Ascend910B");
    if (platform != nullptr) {
        platform->socSpec.Get("version", "Short_SoC_version", socVersion);
    }
    if (socVersion == "Ascend310P") {
        return SocVersion::ASCEND310P;
    }
    if (socVersion == "Ascend910_95")
    {
        return SocVersion::ASCEND910_95;
    }

    return SocVersion::ASCEND910B;
}

static PlatformInfo platformInfo;

const PlatformInfo &GetCurrentPlatformInfo() {
    return platformInfo;
}

fe::PlatFormInfos *PlatformInfo::GetPlatformInfos() const {
    fe::PlatFormInfos *platformOriInfo = nullptr;
    return platformOriInfo;
}
} // namespace op
