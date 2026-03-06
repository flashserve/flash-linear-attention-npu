/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHdq WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#ifndef __USE_LARGEFILE64    
#define __USE_LARGEFILE64
#endif

#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_chunk_bwd_dqkwg.h"

#include <sstream>
#include <unordered_map>
#include <cctype>
using std::string;
using std::cout;
std::unordered_map<std::string, std::string> parse_config(const std::string& filename) {
    cout << "[test] reading config:" << filename << "\n";
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释行
        if (line.empty() || line[0] == '#') continue;
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) continue;
        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);
        // 去除首尾空白字符
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        config[key] = value;
    }
    return config;
}


#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
void PrintVector(const std::vector<T> &arrays, const std::string &name) {
    std::cout << "[test] " << name << ": ";
    for(int i = 0; i < arrays.size(); ++i) {
    std::cout << arrays[i] << " ";
    }
    std::cout << " , sizeof(T) = " << sizeof(T);
    std::cout << std::endl;

}

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    // std::cout << "[test] Reading " << filePath << "\n";
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
    	std::cout << filePath << " fileSize " << fileSize << ", sizeof(T) " << sizeof(T) << ", fileSize/sizeof(T) " << fileSize/sizeof(T) << std::endl;
        std::cerr << filePath << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << filePath <<"读取文件失败" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    std::cout << "[test] writing " << filePath << ", size " << size << ", sizeof(T) " << sizeof(T) << std::endl;
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_LARGEFILE | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        ERROR_LOG("Write file Failed.");
	std::cout << "[test] Wrote file size : " << writeSize << ", not equal to expected size " << size * sizeof(T) << std::endl;
        return false;
    }

    return true;
}


template <typename T>
bool WriteLargeFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    const size_t BLOCK_SIZE = 1024 * 1024 * 1024;  // 1G
    const size_t TOTAL_SIZE = size * sizeof(T);
    std::cout << "[test] writing " << filePath << ", size " << size << ", sizeof(T) " << sizeof(T) << std::endl;
    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_LARGEFILE, 0666);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    // void *buf = malloc(BLOCK_SIZE);  // 1MB
    // if (!buf) {
    //     perror("malloc error in writing file");
    //     close(fd);
    //     return 1;
    // }
    //memset(buf, 0, BLOCK_SIZE); // 填充缓冲区

    //size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));

    size_t written = 0;
    while (written < TOTAL_SIZE) {
        size_t to_write = BLOCK_SIZE;
        if (TOTAL_SIZE - written < BLOCK_SIZE) {
            to_write = TOTAL_SIZE - written;
        }
        ssize_t ret = write(fd, reinterpret_cast<char*>(hostData.data())+written, to_write);
        if (ret < 0) {
            perror("write error");
            break;
        }
        written += ret;
    }

    //free(buf);
    (void)close(fd);
    if (written != size * sizeof(T)) {
        ERROR_LOG("Write file Failed.");
	std::cout << "[test] Wrote file size : " << written << ", not equal to expected size " << size * sizeof(T) << std::endl;
        return false;
    }

    return true;
}

// void SeqLensCreate(aclTensor** cu_seqlens, void** cu_seqlensDeviceAddr, aclTensor** chunk_indices, void** chunk_indicesDeviceAddr, int chunk_size, int num_chunks) {
//     // aclTensor* cu_seqlens = nullptr;
//     // void* cu_seqlensDeviceAddr = nullptr;
//     std::vector<int64_t> cu_seqlensShape = {seqlen_nums};
//     std::vector<int64_t> cu_seqlensHostData = {0, 1023, 1024, 1024+512, 2048};
//     ret = CreateAclTensor(cu_seqlensHostData, cu_seqlensShape, cu_seqlensDeviceAddr, aclDataType::ACL_INT64, cu_seqlens);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);

//     // aclTensor* chunk_indices = nullptr;
//     // void* chunk_indicesDeviceAddr = nullptr;
//     // int num_chunks 
//     std::vector<int64_t> chunk_indicesShape = {num_chunks, 2};
//     std::vector<int64_t> chunk_indicesHostData(num_chunks * 2, 0);
//     ret = CreateAclTensor(chunk_indicesHostData, chunk_indicesShape, chunk_indicesDeviceAddr, aclDataType::ACL_INT64, chunk_indices);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// }

template <typename T = short>
void PrintResult(std::vector<int64_t>& shape, void** deviceAddr, const string & path)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    if(size < 2147479552 / sizeof(T)) WriteFile(path, size, resultData);
    else WriteLargeFile(path, size, resultData);
    // for (int64_t i = 0; i < size; i++) {
    //     LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    // }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // std::cout << "size: " << size << ", GetShapeSize" << GetShapeSize(shape)  << "\n";
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main(int argc, char* argv[])
{
    string DATAPATH;
    // string GENPATH;
    if (argc >= 2) {
        DATAPATH = argv[1];
    } else {
        DATAPATH = "/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/result/cpu_for_test";
    }
    DATAPATH += "/gen/";
    std::cout << "[test] setting GENPATH: " << DATAPATH << "\n";
    auto config = parse_config(DATAPATH + "config.cfg");
    // std::
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 2;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int16_t HALF_ONE = 0x3C00;
    const int16_t HALF_TWO = 0x4000;

    // std::cout << config["B"] << "\n";
    int B = std::stoi(config["B"]);
    int H = std::stoi(config["H"]);
    int T = std::stoi(config["T"]);
    int K = std::stoi(config["K"]);
    int V = std::stoi(config["V"]);
    int chunk_size = std::stoi(config["chunk_size"]);
    int num_chunks = std::stoi(config["num_chunks"]);
    int isVarLen = std::stoi(config["isVarLen"]);
    int seqlen_nums = 0;
    if (isVarLen) seqlen_nums = std::stoi(config["seqlen_nums"]);
    float scale = std::stof(config["scale"]);
    
    string str_datatype = config["datatype"];
    string str_gtype = config["gtype"];
    auto datatype = aclDataType::ACL_FLOAT16;
    auto gtype = aclDataType::ACL_FLOAT;
    if (str_datatype == "torch.bfloat16") {
        datatype = aclDataType::ACL_BF16;
    }
    if (str_gtype == "torch.bfloat16") {
        gtype = aclDataType::ACL_BF16;
    } else if (str_gtype == "torch.float16") {
        gtype = aclDataType::ACL_FLOAT16;
    }

    std::cout << "[test] B: " << B << "\n"
            << "[test] H: " << H << "\n"
            << "[test] T: " << T << "\n"
            << "[test] K: " << K << "\n"
            << "[test] V: " << V << "\n"
            << "[test] chunk_size: " << chunk_size << "\n"
            << "[test] num_chunks: " << num_chunks << "\n"
            << "[test] seqlen_nums: " << seqlen_nums << "\n"
            << "[test] scale: " << scale << "\n"
            << "[test] isVarLen: " << isVarLen << "\n"
            << "[test] DATAPATH: " << DATAPATH << "\n"
            << "[test] datatype: " << str_datatype << ", gtype: " << str_gtype << "\n"
            << "[test] deviceId: " << deviceId << "\n";

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    aclTensor* q = nullptr;
    void* qDeviceAddr = nullptr;
    std::vector<int64_t> qShape = {B, H, T, K};
    std::vector<int16_t> qHostData(B * H * T * K, HALF_ONE); // 2048：创建包含32*4*4*4=2048个元素的向量
    ReadFile(DATAPATH + "q.bin",qShape,qHostData);
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, datatype, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* k = nullptr;
    void* kDeviceAddr = nullptr;
    std::vector<int64_t> kShape = {B, H, T, K};
    std::vector<int16_t> kHostData(B * H * T * K, HALF_ONE);
    ReadFile(DATAPATH + "k.bin",kShape,kHostData);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, datatype, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* v = nullptr;
    void* vDeviceAddr = nullptr;
    std::vector<int64_t> vShape = {B, H, T, V};
    std::vector<int16_t> vHostData(B * H * T * V, HALF_ONE);
    ReadFile(DATAPATH + "v.bin",vShape,vHostData);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, datatype, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* w = nullptr;
    void* wDeviceAddr = nullptr;
    std::vector<int64_t> wShape = {B, H, T, K};
    std::vector<int16_t> wHostData(B * H * T * K, HALF_ONE);
    ReadFile(DATAPATH + "w.bin",wShape,wHostData);
    ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, datatype, &w);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* g = nullptr;
    void* gDeviceAddr = nullptr;
    std::vector<int64_t> gShape = {B, H, T};
    if (str_gtype == "torch.float32") {
        std::vector<float> gHostDataF32(B * H * T, 1.0);
        ReadFile(DATAPATH + "g.bin",gShape,gHostDataF32);
        ret = CreateAclTensor(gHostDataF32, gShape, &gDeviceAddr, gtype, &g);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    else {
        std::vector<int16_t> gHostDataF16(B * H * T, HALF_ONE);
        ReadFile(DATAPATH + "g.bin",gShape,gHostDataF16);
        ret = CreateAclTensor(gHostDataF16, gShape, &gDeviceAddr, gtype, &g);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    aclTensor* h = nullptr;
    void* hDeviceAddr = nullptr;
    std::vector<int64_t> hShape = {B, H, num_chunks, K, V};
    std::vector<int16_t> hHostData(B * H * num_chunks * K * V, HALF_TWO);
    ReadFile(DATAPATH + "h.bin",hShape,hHostData);
    ret = CreateAclTensor(hHostData, hShape, &hDeviceAddr, datatype, &h);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    aclTensor* dv = nullptr;
    void* dvDeviceAddr = nullptr;
    std::vector<int64_t> dvShape = {B, H, T, V};
    std::vector<int16_t> dvHostData(B * H * T * V, HALF_ONE);
    ReadFile(DATAPATH + "dv.bin",dvShape,dvHostData);
    ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, datatype, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* do_ = nullptr;
    void* doDeviceAddr = nullptr;
    std::vector<int64_t> doShape = {B, H, T, V};
    std::vector<int16_t> doHostData(B * H * T * V, HALF_ONE);
    ReadFile(DATAPATH + "do.bin",doShape,doHostData);
    ret = CreateAclTensor(doHostData, doShape, &doDeviceAddr, datatype, &do_);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* dh = nullptr;
    void* dhDeviceAddr = nullptr;
    std::vector<int64_t> dhShape = {B, H, num_chunks, K, V};
    std::vector<int16_t> dhHostData(B * H * num_chunks * K * V, HALF_ONE);
    // std::cout << "B * H * num_chunks * K * V: "<< B * H * num_chunks * K * V << "\n";
    ReadFile(DATAPATH + "dh.bin",dhShape,dhHostData);
    ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, datatype, &dh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
/*
    std::vector<int64_t> tuningConfigData = {512};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);
*/
    aclIntArray* cu_seqlens = nullptr;
    void* cu_seqlensDeviceAddr = nullptr;
    if (isVarLen == 1) {
        std::vector<int64_t> cu_seqlensShape = {seqlen_nums};
        std::vector<int64_t> cu_seqlensHostData(seqlen_nums, 0);
        ReadFile(DATAPATH + "cu_seqlens.bin",cu_seqlensShape,cu_seqlensHostData);
        // ret = CreateAclTensor(cu_seqlensHostData, cu_seqlensShape, &cu_seqlensDeviceAddr, aclDataType::ACL_INT64, &cu_seqlens);
        cu_seqlens = aclCreateIntArray(cu_seqlensHostData.data(), cu_seqlensShape[0]);
        PrintVector(cu_seqlensHostData, "cu_seqlens");
        // CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
        std::cout << "isVarLen == FALSE!\n";
    }

    aclIntArray* chunk_indices = nullptr;
    void* chunk_indicesDeviceAddr = nullptr;
    std::vector<int64_t> chunk_indicesShape = {num_chunks, 2};
    std::vector<int64_t> chunk_indicesHostData(num_chunks * 2, 0);
    if (isVarLen == 1) {
        ReadFile(DATAPATH + "chunk_indices.bin",chunk_indicesShape,chunk_indicesHostData);
        // ret = CreateAclTensor(chunk_indicesHostData, chunk_indicesShape, &chunk_indicesDeviceAddr, aclDataType::ACL_INT64, &chunk_indices);
        chunk_indices = aclCreateIntArray(chunk_indicesHostData.data(), num_chunks * 2);
        // CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // aclTensor* down_tri = nullptr;
    // void* downTriDeviceAddr = nullptr;
    // std::vector<int64_t> downTriShape = {chunk_size, chunk_size / 8};
    // std::vector<uint8_t> downTriHostData(B * H * num_chunks * K * V, 0);
    // // std::cout << "B * H * num_chunks * K * V: "<< B * H * num_chunks * K * V << "\n";
    // ReadFile(DATAPATH + "down_tri.bin",downTriShape,downTriHostData);
    // ret = CreateAclTensor(downTriHostData, downTriShape, &downTriDeviceAddr, aclDataType::ACL_UINT8, &down_tri);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* dq = nullptr;
    void* dqDeviceAddr = nullptr;
    std::vector<int64_t> dqShape = {B, H, T, K};
    std::vector<int16_t> dqHostData(B * H * T * K, 0);
    ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, datatype, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* dk = nullptr;
    void* dkDeviceAddr = nullptr;
    std::vector<int64_t> dkShape = {B, H, T, K};
    std::vector<int16_t> dkHostData(B * H * T * K, 0);
    ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, datatype, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* dw = nullptr;
    void* dwDeviceAddr = nullptr;
    std::vector<int64_t> dwShape = {B, H, T, K};
    std::vector<int16_t> dwHostData(B * H * T * K, 0);
    ret = CreateAclTensor(dwHostData, dwShape, &dwDeviceAddr, datatype, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* dg = nullptr;
    void* dgDeviceAddr = nullptr;
    std::vector<int64_t> dgShape = {B, H, T};
    if (str_gtype == "torch.float32") {
        std::vector<float> dgHostData(B * H * T, 0);
        ret = CreateAclTensor(dgHostData, dgShape, &dgDeviceAddr, gtype, &dg);
    } else {
        std::vector<int16_t> dgHostData(B * H * T, 0);
        ret = CreateAclTensor(dgHostData, dgShape, &dgDeviceAddr, gtype, &dg);
    }
    
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. 调用aclnnAddExample第一段接口
    ret = aclnnChunkBwdDqkwgGetWorkspaceSize(q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, scale,chunk_size, dq, dk, dw, dg, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkBwdDqkwgGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnAddExample第二段接口
    ret = aclnnChunkBwdDqkwg(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkBwdDqkwg failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintResult(dqShape, &dqDeviceAddr, DATAPATH + "dq_npu.bin");
    PrintResult(dkShape, &dkDeviceAddr, DATAPATH + "dk_npu.bin");
    PrintResult(dwShape, &dwDeviceAddr, DATAPATH + "dw_npu.bin");
    if (str_gtype == "torch.float32") {
        PrintResult<float>(dgShape, &dgDeviceAddr, DATAPATH + "dg_npu.bin");
    } else {
        PrintResult(dgShape, &dgDeviceAddr, DATAPATH + "dg_npu.bin");
    }

    // 7. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(w);
    aclDestroyTensor(g);
    aclDestroyTensor(h);
    aclDestroyTensor(dv);
    aclDestroyTensor(do_);
    aclDestroyTensor(dh);
    aclDestroyIntArray(cu_seqlens);
    aclDestroyIntArray(chunk_indices);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dw);
    aclDestroyTensor(dg);

    // 8. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(wDeviceAddr);
    aclrtFree(gDeviceAddr);
    aclrtFree(hDeviceAddr);
    aclrtFree(dvDeviceAddr);
    aclrtFree(doDeviceAddr);
    aclrtFree(dhDeviceAddr);
    aclrtFree(cu_seqlensDeviceAddr);
    aclrtFree(chunk_indicesDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dkDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(dgDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 9. acl去初始化
    aclFinalize();

    return 0;
}
