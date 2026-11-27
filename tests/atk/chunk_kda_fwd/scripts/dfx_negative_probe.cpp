// chunk_kda_fwd ACLNN 参数拒绝负向用例（host 侧，不执行 kernel）。
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_chunk_kda_fwd.h"

namespace {
int g_fails = 0;

struct Shapes {
    std::vector<int64_t> q;
    std::vector<int64_t> k;
    std::vector<int64_t> v;
    std::vector<int64_t> g;
    std::vector<int64_t> beta;
    std::vector<int64_t> attn;
};

Shapes DenseBnsd()
{
    // B=1, HK=HV=2, T=64, K=V=128, BNSD
    return {{1, 2, 64, 128}, {1, 2, 64, 128}, {1, 2, 64, 128},
            {1, 2, 64, 128}, {1, 2, 64}, {1, 2, 64, 128}};
}

aclTensor *Make(const std::vector<int64_t> &shape, aclDataType dtype, aclFormat format)
{
    if (shape.empty()) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }
    int64_t *dev = nullptr;
    const int64_t numel = [&] {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }();
    if (aclrtMalloc(reinterpret_cast<void **>(&dev), static_cast<size_t>(numel) * 2, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        return nullptr;
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format,
                           shape.data(), shape.size(), dev);
}

void DestroyTensor(aclTensor *t)
{
    if (t != nullptr) {
        aclDestroyTensor(t);
    }
}

void Report(const char *name, int expectCode, const char *mustContain)
{
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    Shapes s = DenseBnsd();
    if (std::strcmp(name, "q_rank2") == 0) s.q = {64, 128};
    if (std::strcmp(name, "q_rank5") == 0) s.q = {1, 1, 2, 64, 128};
    if (std::strcmp(name, "v_rank3") == 0) s.v = {2, 64, 128};
    if (std::strcmp(name, "beta_rank4") == 0) s.beta = {1, 2, 64, 1};

    aclTensor *q = std::strcmp(name, "q_null") == 0 ? nullptr : Make(s.q, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *k = Make(s.k, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *v = Make(s.v, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *g = Make(s.g, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *beta = Make(s.beta, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *attn = Make(s.attn, ACL_BF16, ACL_FORMAT_ND);
    // aqkOut 是 REQUIRED 输出；先满足它才能进入 shape/rank 校验分支。
    aclTensor *aqk = Make({1, 2, 64, 64}, ACL_BF16, ACL_FORMAT_ND);

    aclGetRecentErrMsg();
    aclnnStatus ret = aclnnChunkKdaFwdGetWorkspaceSize(
        q, k, v, g, beta,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        "BNSD", 0.125, 64, false, -5.0, false, false,
        attn, nullptr, nullptr, aqk, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        &ws, &ex);
    const char *msg = aclGetRecentErrMsg();
    const std::string text = msg == nullptr ? "" : msg;
    const int code = static_cast<int>(ret);
    bool ok = (code == expectCode);
    if (mustContain != nullptr && text.find(mustContain) == std::string::npos) {
        ok = false;
    }
    printf("[%s] ret=%d expect=%d %s\n    msg=%s\n", name, code, expectCode,
           ok ? "PASS" : "FAIL", text.c_str());
    if (!ok) {
        ++g_fails;
    }
    DestroyTensor(q); DestroyTensor(k); DestroyTensor(v);
    DestroyTensor(g); DestroyTensor(beta); DestroyTensor(attn); DestroyTensor(aqk);
}
} // namespace

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    Report("q_rank2", 161002, "ranks must match layout");
    Report("q_rank5", 161002, "ranks must match layout");
    Report("v_rank3", 161002, "ranks must match layout");
    Report("beta_rank4", 161002, "ranks must match layout");
    Report("q_null", 161001, "q");

    printf("FAILS=%d\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
