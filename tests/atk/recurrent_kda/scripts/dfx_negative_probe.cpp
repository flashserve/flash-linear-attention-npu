// Batch 2 负向用例：recurrent_kda / recurrent_gated_delta_rule ACLNN 参数拒绝。
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_recurrent_kda.h"
#include "aclnn_recurrent_gated_delta_rule.h"

// 包内头文件在部分版本未导出 V2 原型，这里显式声明（符号由 libcust_opapi 导出）。
extern "C" aclnnStatus aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLog, const aclTensor *dtBias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

namespace {
int g_fails = 0;

aclTensor *Make(const std::vector<int64_t> &shape, aclDataType dtype)
{
    if (shape.empty()) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }
    int64_t numel = 1;
    for (int64_t d : shape) {
        numel *= d;
    }
    int64_t *dev = nullptr;
    if (aclrtMalloc(reinterpret_cast<void **>(&dev), static_cast<size_t>(numel) * 4,
                    ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        return nullptr;
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), dev);
}

void Check(const char *tag, int code, int expect, const std::string &msg, const char *needle)
{
    const bool hasNeedle = needle == nullptr || msg.find(needle) != std::string::npos;
    const bool ok = code == expect && hasNeedle;
    printf("[%s] ret=%d expect=%d %s\n    msg=%s\n", tag, code, expect, ok ? "PASS" : "FAIL",
           msg.c_str());
    if (!ok) {
        ++g_fails;
    }
}

void RunRecurrentKda(const std::vector<std::vector<int64_t>> &shapes, int expect, const char *needle)
{
    aclTensor *q = Make(shapes[0], ACL_BF16);
    aclTensor *k = Make(shapes[1], ACL_BF16);
    aclTensor *v = Make(shapes[2], ACL_BF16);
    aclTensor *g = Make(shapes[3], ACL_BF16);
    aclTensor *beta = Make(shapes[4], ACL_BF16);
    aclTensor *state = Make({4, 2, 128, 128}, ACL_BF16);
    aclTensor *attn = Make({64, 2, 128}, ACL_BF16);
    aclTensor *finalState = Make({1, 2, 128, 128}, ACL_BF16);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    aclGetRecentErrMsg();
    const aclnnStatus ret = aclnnRecurrentKdaGetWorkspaceSize(
        q, k, v, g, beta, state, nullptr, nullptr, nullptr, nullptr, nullptr,
        "TND", 0.125, false, true, false, false, false, false, false, -5.0, false,
        attn, finalState, &ws, &ex);
    const char *m = aclGetRecentErrMsg();
    Check("recurrent_kda", static_cast<int>(ret), expect, m == nullptr ? "" : m, needle);
    aclDestroyTensor(q); aclDestroyTensor(k); aclDestroyTensor(v);
    aclDestroyTensor(g); aclDestroyTensor(beta); aclDestroyTensor(state);
    aclDestroyTensor(attn); aclDestroyTensor(finalState);
}

void RunRgdn(const std::vector<std::vector<int64_t>> &shapes, int expect, const char *needle)
{
    aclTensor *q = Make(shapes[0], ACL_BF16);
    aclTensor *k = Make(shapes[1], ACL_BF16);
    aclTensor *v = Make(shapes[2], ACL_BF16);
    aclTensor *beta = Make(shapes[3], ACL_BF16);
    aclTensor *state = Make(shapes[4], ACL_BF16);
    aclTensor *cu = Make({2}, ACL_INT32);
    aclTensor *indices = Make({64}, ACL_INT32);
    aclTensor *out = Make({64, 2, 128}, ACL_BF16);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    aclGetRecentErrMsg();
    const aclnnStatus ret = aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize(
        q, k, v, beta, state, cu, indices, nullptr, nullptr, nullptr, nullptr, nullptr,
        0.125f, false, false, false, false, out, &ws, &ex);
    const char *m = aclGetRecentErrMsg();
    Check("rgdn", static_cast<int>(ret), expect, m == nullptr ? "" : m, needle);
    aclDestroyTensor(q); aclDestroyTensor(k); aclDestroyTensor(v);
    aclDestroyTensor(beta); aclDestroyTensor(state); aclDestroyTensor(cu);
    aclDestroyTensor(indices); aclDestroyTensor(out);
}
} // namespace

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    // 缺参：query=nullptr -> 161001 + EZ0004
    RunRecurrentKda({{}, {64, 2, 128}, {64, 2, 128}, {64, 2, 128}, {64, 2}}, 161001, "query");
    RunRgdn({{}, {64, 2, 128}, {64, 2, 128}, {64, 2}, {4, 2, 128, 128}}, 161001, "query");

    // rank 不符：TND 下 query 应为 rank3
    RunRecurrentKda({{64, 128}, {64, 2, 128}, {64, 2, 128}, {64, 2, 128}, {64, 2}}, 161002, "query");
    RunRgdn({{64, 128}, {64, 2, 128}, {64, 2, 128}, {64, 2}, {4, 2, 128, 128}}, 161002, "query");

    printf("FAILS=%d\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
