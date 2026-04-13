/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file io.cpp
 * \brief
 */

#include "tests/utils/io.h"
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include "tests/utils/log.h"

[[maybe_unused]] bool ops::adv::tests::utils::FileExist(const std::string &filePath)
{
    struct stat s {};
    return stat(filePath.c_str(), &s) == 0;
}

[[maybe_unused]] bool ops::adv::tests::utils::ReadFile(const std::string &filePath, size_t &fileSize, void *buffer,
                                                       size_t bufferSize)
{
    struct stat sBuf {};
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        LOG_ERR("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG_ERR("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        LOG_ERR("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        LOG_ERR("file(%s)' size(%zu) is larger than buffer size(%zu)", filePath.c_str(), size, bufferSize);
        file.close();
        return false;
    }

    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), static_cast<std::streamsize>(size));
    fileSize = size;
    file.close();
    return true;
}

[[maybe_unused]] bool ops::adv::tests::utils::WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        LOG_ERR("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_ERR("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        LOG_ERR("Write file Failed.");
        return false;
    }

    return true;
}
