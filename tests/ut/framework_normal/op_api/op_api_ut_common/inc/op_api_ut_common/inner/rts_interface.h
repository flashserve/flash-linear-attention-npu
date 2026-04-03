/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
 
#ifndef _OP_API_UT_COMMON_RTS_INTERFACE_H_
#define _OP_API_UT_COMMON_RTS_INTERFACE_H_

//#include "runtime/stream.h"

void * MallocDeviceMemory(unsigned long size);
void FreeDeviceMemory(void * device_mem_ptr);
int MemcpyToDevice(const void* host_mem, void* dev_mem, unsigned long size);
int MemcpyFromDevice(void* host_mem, const void* dev_mem, unsigned long size);
int RtsInit();
void RtsUnInit();
int SynchronizeStream();

#endif