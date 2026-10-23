#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_recompute_w_u_fwd_experimental.h"

struct DeviceTensor {
    void *data = nullptr;
    aclTensor *descriptor = nullptr;
};

static int64_t ElementCount(const std::vector<int64_t> &shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

template <typename T>
static bool MakeTensor(const std::vector<T> &values, const std::vector<int64_t> &shape,
                       aclDataType dtype, DeviceTensor &out)
{
    const size_t bytes = values.size() * sizeof(T);
    if (ElementCount(shape) != static_cast<int64_t>(values.size()) ||
        aclrtMalloc(&out.data, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(out.data, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    out.descriptor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                     ACL_FORMAT_ND, shape.data(), shape.size(), out.data);
    return out.descriptor != nullptr;
}

int main(int argc, char **argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int64_t valueWidth = argc > 2 ? std::atoll(argv[2]) : 128;
    const int64_t t = argc > 3 ? std::atoll(argv[3]) : 64;
    if ((valueWidth != 128 && valueWidth != 256) || (t != 64 && t != 128)) {
        std::fprintf(stderr, "Only V=128/256 and chunk=64/128 are supported\n");
        return 2;
    }
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(device) != ACL_SUCCESS) {
        std::fprintf(stderr, "ACL initialization failed for device %d\n", device);
        return 1;
    }

    constexpr int64_t keyWidth = 128;
    const std::vector<int64_t> keyShape = {1, 1, t, keyWidth};
    const std::vector<int64_t> valueShape = {1, 1, t, valueWidth};
    const std::vector<int64_t> scalarShape = {1, 1, t};
    const std::vector<int64_t> aShape = {1, 1, t, t};
    std::vector<uint16_t> identity(t * t, 0);
    for (int64_t i = 0; i < t; ++i) {
        identity[i * t + i] = 0x3c00;
    }

    DeviceTensor k, v, beta, a, g, w, u;
    if (!MakeTensor(std::vector<uint16_t>(t * keyWidth, 0x3c00), keyShape, ACL_FLOAT16, k) ||
        !MakeTensor(std::vector<uint16_t>(t * valueWidth, 0x3c00), valueShape, ACL_FLOAT16, v) ||
        !MakeTensor(std::vector<float>(t, 1.0f), scalarShape, ACL_FLOAT, beta) ||
        !MakeTensor(identity, aShape, ACL_FLOAT16, a) ||
        !MakeTensor(std::vector<float>(t, 0.0f), scalarShape, ACL_FLOAT, g) ||
        !MakeTensor(std::vector<uint16_t>(t * keyWidth, 0), keyShape, ACL_FLOAT16, w) ||
        !MakeTensor(std::vector<uint16_t>(t * valueWidth, 0), valueShape, ACL_FLOAT16, u)) {
        std::fprintf(stderr, "Tensor allocation or descriptor creation failed\n");
        return 1;
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "Stream creation failed\n");
        return 1;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnRecomputeWUFwdExperimentalGetWorkspaceSize(
        k.descriptor, v.descriptor, beta.descriptor, a.descriptor, g.descriptor, nullptr,
        nullptr, nullptr, t, w.descriptor, u.descriptor, &workspaceSize, &executor);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "GetWorkspaceSize failed: %d\n", status);
        return 1;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "Workspace allocation failed: %llu bytes\n",
                     static_cast<unsigned long long>(workspaceSize));
        return 1;
    }
    status = aclnnRecomputeWUFwdExperimental(workspace, workspaceSize, executor, stream);
    if (status != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "Experimental ACLNN launch failed: %d\n", status);
        return 1;
    }

    std::vector<uint16_t> wHost(t * keyWidth), uHost(t * valueWidth);
    if (aclrtMemcpy(wHost.data(), wHost.size() * sizeof(uint16_t), w.data,
                    wHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(uHost.data(), uHost.size() * sizeof(uint16_t), u.data,
                    uHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::fprintf(stderr, "Output copy failed\n");
        return 1;
    }
    for (size_t i = 0; i < wHost.size(); ++i) {
        if (wHost[i] != 0x3c00) {
            std::fprintf(stderr, "w mismatch at %zu: 0x%x\n", i, wHost[i]);
            return 1;
        }
    }
    for (size_t i = 0; i < uHost.size(); ++i) {
        if (uHost[i] != 0x3c00) {
            std::fprintf(stderr, "u mismatch at %zu: 0x%x\n", i, uHost[i]);
            return 1;
        }
    }

    for (DeviceTensor *tensor : {&k, &v, &beta, &a, &g, &w, &u}) {
        aclDestroyTensor(tensor->descriptor);
        aclrtFree(tensor->data);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    std::printf("PASS: experimental ACLNN V=%lld chunk=%lld produced the expected w/u values\n",
                static_cast<long long>(valueWidth), static_cast<long long>(t));
    return 0;
}
