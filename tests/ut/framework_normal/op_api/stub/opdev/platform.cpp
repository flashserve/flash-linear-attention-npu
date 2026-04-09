/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "platform.h"

namespace op {

SocVersion g_socVersion = SocVersion::ASCEND910B;
PlatformInfo *g_platformInfo = new PlatformInfo();

bool PlatformInfo::CheckSupport(SocSpec socSpec, SocSpecAbility ability) const
{
    return true;
}

PlatformInfo::~PlatformInfo()
{
    if (impl_ != nullptr) {
        delete impl_;
    }
}

bool PlatformInfo::Valid() const
{
    return valid_;
}

void PlatformInfo::SetPlatformImpl(PlatformInfoImpl *impl)
{
    impl_ = impl;
    valid_ = true;
}

SocVersion PlatformInfo::GetSocVersion() const
{
    return g_socVersion;
}

const std::string& PlatformInfo::GetSocLongVersion() const
{
    return "";
}

int32_t PlatformInfo::GetDeviceId() const
{
    return 0;
}

int64_t PlatformInfo::GetBlockSize() const
{
    return 0;
}

uint32_t PlatformInfo::GetCubeCoreNum() const
{
    return 0;
}

uint32_t PlatformInfo::GetVectorCoreNum() const
{
    return 0;
}

bool PlatformInfo::GetFftsPlusMode() const
{
    return true;
}

fe::PlatFormInfos* PlatformInfo::GetPlatformInfos() const
{
    return nullptr;
}

const PlatformInfo& GetCurrentPlatformInfo()
{
    return *g_platformInfo;
}

ge::AscendString ToString(SocVersion socVersion)
{
    static const std::map<SocVersion, std::string> kSocVersionMap = {
        {SocVersion::ASCEND910, "Ascend910"},
        {SocVersion::ASCEND910B, "Ascend910B"},
        {SocVersion::ASCEND910_93, "Ascend910_93"},
        {SocVersion::ASCEND910_95, "Ascend910_95"},
        {SocVersion::ASCEND910E, "Ascend910E"},
        {SocVersion::ASCEND310, "Ascend310"},
        {SocVersion::ASCEND310P, "Ascend310P"},
        {SocVersion::ASCEND310B, "Ascend310B"},
        {SocVersion::ASCEND310C, "Ascend310C"},
        {SocVersion::ASCEND610LITE, "Ascend610LITE"},
        {SocVersion::KIRINX90, "KirinX90"},
        {SocVersion::RESERVED_VERSION, "UnknowSocVersion"},
    };
    static const std::string reserved("UnknowSocVersion");
    const auto it = kSocVersionMap.find(socVersion);
    if (it != kSocVersionMap.end()) {
        return ge::AscendString((it->second).c_str());
    } else {
        return ge::AscendString(reserved.c_str());
    }
}

void SetPlatformSocVersion(SocVersion socVersion)
{
    g_socVersion = socVersion;
}

} // namespace op