/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file platform.h
 * \brief 提供平台相关接口的打桩及辅助功能.
 */

#pragma once

#include <cstdint>
#include <map>
#include <any>
#include <string>
#include <dlfcn.h>

namespace ops::adv::tests::utils {

class Platform {
public:
    enum class SocVersion {
        Ascend910B1,
        Ascend910B2,
        Ascend910B3,
        Ascend310P3,
        Ascend910_9591,
		SocVersionBottom,
    };
    class SocSpec {
    public:
        std::map<std::string, std::map<std::string, std::any>> spec;

        bool Get(const char *label, const char *key, std::any &value) const;
        bool Get(const char *label, const char *key, std::string &value) const;
        bool Get(const char *label, const char *key, uint32_t &value) const;
        bool Get(const char *label, const char *key, uint64_t &value) const;
    };
    SocSpec socSpec;

    static void SetGlobalPlatform(Platform *platform);
    static Platform *GetGlobalPlatform();

    bool InitArgsInfo(int argc, char **argv);
    const char *GetExeAbsPath();

    Platform &SetSocVersion(const SocVersion &socVersion);
    [[maybe_unused]] [[nodiscard]] uint32_t GetCoreNum() const;
    [[maybe_unused]] [[nodiscard]] int64_t GetBlockDim() const;

    [[maybe_unused]] static void *LoadSo(const char *absPath, int mode = RTLD_NOW | RTLD_GLOBAL);
    [[maybe_unused]] static bool UnLoadSo(void *hdl);
    [[maybe_unused]] [[nodiscard]] static void *LoadSoSym(void *hdl, const char *name);

    [[maybe_unused]] bool LoadOpTilingSo();
    [[maybe_unused]] bool UnLoadOpTilingSo();
    [[maybe_unused]] [[nodiscard]] void *LoadOpTilingSoSym(const char *name);

    [[maybe_unused]] bool LoadOpProtoSo();
    [[maybe_unused]] bool UnLoadOpProtoSo();

private:
    void *tilingSoHdl_ = nullptr;
    void *protoSoHdl_ = nullptr;
    std::string exeAbsPath_;
};

} // namespace ops::adv::tests::utils
