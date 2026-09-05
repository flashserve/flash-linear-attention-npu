/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
#include "aclnnop/aclnn_chunk_bwd_dv_local.h"
#include "aclnnop/aclnn_recompute_w_u_fwd.h"
#endif
#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
#include "aclnnop/aclnn_chunk_gdn_bwd_intra.h"
#endif

namespace {

struct TensorHandle {
    void *device = nullptr;
    aclTensor *tensor = nullptr;
};

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (int64_t dim : shape) {
        result *= dim;
    }
    return result;
}

bool CreateZeroTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                      size_t elementSize, TensorHandle &handle)
{
    const size_t bytes = static_cast<size_t>(Numel(shape)) * elementSize;
    if (aclrtMalloc(&handle.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemset(handle.device, bytes, 0, bytes) != ACL_SUCCESS) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    handle.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), handle.device);
    return handle.tensor != nullptr;
}

void DestroyTensor(TensorHandle &handle)
{
    if (handle.tensor != nullptr) {
        aclDestroyTensor(handle.tensor);
    }
    if (handle.device != nullptr) {
        aclrtFree(handle.device);
    }
}

template <typename Prepare, typename Launch>
float Measure(aclrtStream stream, int warmup, int repeats, Prepare prepare, Launch launch)
{
    for (int i = 0; i < warmup; ++i) {
        aclOpExecutor *executor = nullptr;
        if (!prepare(&executor) || !launch(executor)) {
            return -1.0F;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return -1.0F;
    }
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&end) != ACL_SUCCESS) {
        return -1.0F;
    }
    float total = 0.0F;
    for (int i = 0; i < repeats; ++i) {
        aclOpExecutor *executor = nullptr;
        if (!prepare(&executor) || aclrtRecordEvent(start, stream) != ACL_SUCCESS ||
            !launch(executor) || aclrtRecordEvent(end, stream) != ACL_SUCCESS ||
            aclrtSynchronizeEvent(end) != ACL_SUCCESS) {
            return -1.0F;
        }
        float elapsed = 0.0F;
        if (aclrtEventElapsedTime(&elapsed, start, end) != ACL_SUCCESS) {
            return -1.0F;
        }
        total += elapsed;
    }
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    return total / static_cast<float>(repeats);
}

} // namespace

int main(int argc, char **argv)
{
    constexpr int64_t B = 1;
    constexpr int64_t K = 128;
    constexpr int64_t V = 128;
    constexpr int64_t BT = 64;
    constexpr double SCALE = 1.0;
    const int64_t t = argc > 1 ? std::atoll(argv[1]) : 8192;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    const int64_t hv = argc > 3 ? std::atoll(argv[3]) : 96;
    const int64_t hk = argc > 4 ? std::atoll(argv[4]) : hv;
    const int64_t stage = argc > 5 ? std::atoll(argv[5]) : 2;
    if (t <= 0 || repeats <= 0 || hk <= 0 || hv <= 0 || hv % hk != 0 ||
        hv / hk > 4 || stage < 0 || stage > 2) {
        std::cerr << "Invalid T, repeats, HK, HV, or stage\n";
        return 1;
    }

    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "ACL initialization failed\n";
        return 1;
    }

    const std::vector<int64_t> qkShape{B, hk, t, K};
    const std::vector<int64_t> valueShape{B, hv, t, V};
    const std::vector<int64_t> gateShape{B, hv, t};
    const std::vector<int64_t> aShape{B, hv, t, BT};
    TensorHandle q, k, v, g, beta, a, dO, w, u, dv;
    const bool created =
        CreateZeroTensor(qkShape, ACL_BF16, 2, q) &&
        CreateZeroTensor(qkShape, ACL_BF16, 2, k) &&
        CreateZeroTensor(valueShape, ACL_BF16, 2, v) &&
        CreateZeroTensor(gateShape, ACL_BF16, 2, g) &&
        CreateZeroTensor(gateShape, ACL_BF16, 2, beta) &&
        CreateZeroTensor(aShape, ACL_BF16, 2, a) &&
        CreateZeroTensor(valueShape, ACL_BF16, 2, dO) &&
        CreateZeroTensor(valueShape, ACL_BF16, 2, w) &&
        CreateZeroTensor(valueShape, ACL_BF16, 2, u) &&
        CreateZeroTensor(valueShape, ACL_BF16, 2, dv);
    if (!created) {
        std::cerr << "Tensor allocation failed\n";
        return 2;
    }

#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
    uint64_t fusedWorkspaceSize = 0;
#endif
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
    uint64_t recomputeWorkspaceSize = 0;
    uint64_t dvWorkspaceSize = 0;
#endif
    aclOpExecutor *executor = nullptr;
#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
    std::cerr << "prepare fused\n";
    if (aclnnChunkGdnBwdIntraGetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, a.tensor, dO.tensor,
            nullptr, nullptr, SCALE, BT, true, stage, w.tensor, u.tensor, dv.tensor,
            &fusedWorkspaceSize, &executor) != ACL_SUCCESS) {
        std::cerr << "Fused GetWorkspaceSize failed\n";
        return 3;
    }
    void *fusedWorkspace = nullptr;
    if (fusedWorkspaceSize > 0 && aclrtMalloc(&fusedWorkspace, fusedWorkspaceSize,
                                              ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return 4;
    }
    std::cerr << "launch fused\n";
    if (aclnnChunkGdnBwdIntra(fusedWorkspace, fusedWorkspaceSize, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "Fused warmup launch failed\n";
        return 3;
    }
    std::cerr << "fused complete\n";
#endif
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
    if (
        aclnnRecomputeWUFwdGetWorkspaceSize(
            k.tensor, v.tensor, beta.tensor, a.tensor, g.tensor, nullptr, nullptr,
            nullptr, BT, w.tensor, u.tensor, &recomputeWorkspaceSize,
            &executor) != ACL_SUCCESS) {
        std::cerr << "Recompute GetWorkspaceSize failed\n";
        return 3;
    }
    void *recomputeWorkspace = nullptr;
    if (recomputeWorkspaceSize > 0 && aclrtMalloc(&recomputeWorkspace, recomputeWorkspaceSize,
                                                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return 4;
    }
    std::cerr << "launch recompute\n";
    if (aclnnRecomputeWUFwd(recomputeWorkspace, recomputeWorkspaceSize, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclnnChunkBwdDvLocalGetWorkspaceSize(
            q.tensor, k.tensor, dO.tensor, g.tensor, nullptr, nullptr, nullptr,
            nullptr, SCALE, BT, dv.tensor, &dvWorkspaceSize, &executor) != ACL_SUCCESS) {
        std::cerr << "Recompute launch or dv GetWorkspaceSize failed\n";
        return 3;
    }
    std::cerr << "recompute complete\n";
    void *dvWorkspace = nullptr;
    if (dvWorkspaceSize > 0 && aclrtMalloc(&dvWorkspace, dvWorkspaceSize,
                                           ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return 4;
    }
    std::cerr << "launch dv\n";
    if (aclnnChunkBwdDvLocal(dvWorkspace, dvWorkspaceSize, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "Dv warmup launch failed\n";
        return 3;
    }
    std::cerr << "dv complete\n";
#endif

#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
    const auto prepareFused = [&](aclOpExecutor **out) {
        uint64_t size = 0;
        return aclnnChunkGdnBwdIntraGetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, a.tensor, dO.tensor,
            nullptr, nullptr, SCALE, BT, true, stage, w.tensor, u.tensor, dv.tensor,
            &size, out) == ACL_SUCCESS && size == fusedWorkspaceSize;
    };
    const auto launchFused = [&](aclOpExecutor *current) {
        return aclnnChunkGdnBwdIntra(fusedWorkspace, fusedWorkspaceSize,
                                     current, stream) == ACL_SUCCESS;
    };
#endif
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
    const auto prepareRecompute = [&](aclOpExecutor **out) {
        uint64_t size = 0;
        return aclnnRecomputeWUFwdGetWorkspaceSize(
            k.tensor, v.tensor, beta.tensor, a.tensor, g.tensor, nullptr, nullptr,
            nullptr, BT, w.tensor, u.tensor, &size, out) == ACL_SUCCESS &&
            size == recomputeWorkspaceSize;
    };
    const auto launchRecompute = [&](aclOpExecutor *current) {
        return aclnnRecomputeWUFwd(recomputeWorkspace, recomputeWorkspaceSize,
                                   current, stream) == ACL_SUCCESS;
    };
    const auto prepareDv = [&](aclOpExecutor **out) {
        uint64_t size = 0;
        return aclnnChunkBwdDvLocalGetWorkspaceSize(
            q.tensor, k.tensor, dO.tensor, g.tensor, nullptr, nullptr, nullptr,
            nullptr, SCALE, BT, dv.tensor, &size, out) == ACL_SUCCESS &&
            size == dvWorkspaceSize;
    };
    const auto launchDv = [&](aclOpExecutor *current) {
        return aclnnChunkBwdDvLocal(dvWorkspace, dvWorkspaceSize,
                                    current, stream) == ACL_SUCCESS;
    };
#endif
#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
    const float fusedMs = Measure(stream, 3, repeats, prepareFused, launchFused);
#endif
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
    const float recomputeMs = Measure(stream, 3, repeats, prepareRecompute, launchRecompute);
    const float dvMs = Measure(stream, 3, repeats, prepareDv, launchDv);
    const float baselineMs = recomputeMs + dvMs;
#endif
#if defined(CHUNK_GDN_BWD_INTRA_BASELINE_ONLY)
    std::cout << "T=" << t << " HK=" << hk << " HV=" << hv << " stage=" << stage
              << " repeats=" << repeats
              << " recompute_ms=" << recomputeMs << " dv_ms=" << dvMs
              << " baseline_ms=" << baselineMs << '\n';
#elif !defined(CHUNK_GDN_BWD_INTRA_FUSED_ONLY)
    std::cout << "T=" << t << " HK=" << hk << " HV=" << hv << " stage=" << stage
              << " repeats=" << repeats
              << " fused_ms=" << fusedMs
              << " recompute_ms=" << recomputeMs << " dv_ms=" << dvMs
              << " baseline_ms=" << baselineMs << " ratio=" << fusedMs / baselineMs
              << " fused_workspace=" << fusedWorkspaceSize << '\n';
#else
    std::cout << "T=" << t << " HK=" << hk << " HV=" << hv << " stage=" << stage
              << " repeats=" << repeats
              << " fused_ms=" << fusedMs
              << " fused_workspace=" << fusedWorkspaceSize << '\n';
#endif

#ifndef CHUNK_GDN_BWD_INTRA_BASELINE_ONLY
    if (fusedWorkspace != nullptr) {
        aclrtFree(fusedWorkspace);
    }
#endif
#ifndef CHUNK_GDN_BWD_INTRA_FUSED_ONLY
    for (void *workspace : {recomputeWorkspace, dvWorkspace}) {
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
    }
#endif
    for (TensorHandle *handle : {&q, &k, &v, &g, &beta, &a, &dO, &w, &u, &dv}) {
        DestroyTensor(*handle);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
#if defined(CHUNK_GDN_BWD_INTRA_BASELINE_ONLY)
    return baselineMs > 0.0F ? 0 : 5;
#elif !defined(CHUNK_GDN_BWD_INTRA_FUSED_ONLY)
    return (fusedMs > 0.0F && baselineMs > 0.0F) ? 0 : 5;
#else
    return fusedMs > 0.0F ? 0 : 5;
#endif
}
