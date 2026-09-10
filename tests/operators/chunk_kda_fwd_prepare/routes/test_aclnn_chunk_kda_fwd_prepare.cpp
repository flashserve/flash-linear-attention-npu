/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of the BSD 3-Clause License (the "License").
 */

// 编译本文件可锁定 aclnn 两段式接口的公开函数签名；设备执行由同目录的
// Python 用例通过稳定 ctypes 入口完成。
#include "aclnn_chunk_kda_fwd_prepare.h"

namespace {

[[maybe_unused]] auto *const kGetWorkspace =
    &aclnnChunkKdaFwdPrepareGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkKdaFwdPrepare;

} // namespace
