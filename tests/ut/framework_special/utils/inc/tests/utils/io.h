/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file io.h
 * \brief
 */

#pragma once

#include <string>

namespace ops::adv::tests::utils {

[[maybe_unused]] bool FileExist(const std::string &filePath);

[[maybe_unused]] bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize);

[[maybe_unused]] bool WriteFile(const std::string &filePath, const void *buffer, size_t size);

} // namespace ops::adv::tests::utils
