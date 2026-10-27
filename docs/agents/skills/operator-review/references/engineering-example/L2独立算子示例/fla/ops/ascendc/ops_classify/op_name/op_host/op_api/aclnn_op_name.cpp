/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/op_api/aclnn_op_name.cpp
 *
 * 注意事项：
 *   1. L2 只做：非空校验 -> 参数/档位校验 -> 按契约处理连续性 -> 调 L0 -> 返回 workspaceSize。
 *      计算、分核、同步都不在这里。
 *   2. 档位由"哪些输出指针非空"推导（ResolveOutputMode），非法组合返回
 *      ACLNN_ERR_PARAM_INVALID 并打印实际组合，不能静默取默认档位。
 *   3. 报错文本要带实际值（布局、chunkSize、输出组合），这是用户唯一能看到的上下文。
 *   4. 每次调用都新建 UniqueExecutor 并把所有权交给调用方：GetWorkspaceSize 只登记任务，
 *      Launch 阶段才真正下发；不要在这里同步执行。
 *   5. 连续性按"逐输入契约"处理，不做整批一刀切（这是 L2 最容易写错的地方）：
 *      - 算子内部按紧凑行主序/固定 stride 寻址的输入（本示例 x/g/a_log）→ 对该输入做一次
 *        `l0op::Contiguous`，并在 docs/api.md 的连续性契约里写明"要求连续"；
 *      - 支持 stride 寻址的输入（recurrent 家族的 state/conv_state 是典型）→ 保持原样或用
 *        `CreateView` 交给算子按 stride 寻址，**不要连续化**：连续化会破坏原地写回语义（写回的是拷贝
 *        而不是调用方张量），并带来 host enqueue 与服务性能回退（Issue #491）；
 *      - 布局改写（layout 物化、reshape/transpose、打包 TND 视图）只在确实需要时做一次明确的
 *        `Contiguous` 或 `ViewCopy`，并把这次拷贝的代价写进 docs/api.md；其余情况一律按 view 传递；
 *      - 上述判定只发生在 L2：适配层（Stable-ABI）不做 dense 拷贝，按 view 原样交出（§5.3）。
 *   6. 可选输出这一层要"校验组合"，不要"解释策略"：
 *      - 上层（Python/legacy 包装层）负责把上层语义翻译成"入参属性 + 哪些可选输出非空"，策略本身不进 L2；
 *      - L2 **必须**校验"入参属性取值 + 可空输出指针（含是否与入参同一 view）"的组合是否合法：
 *        组合必须落在文档化档位内，非法组合返回 `ACLNN_ERR_PARAM_INVALID` 并打印实际组合
 *        （例如 `outputMask=0x%x`、哪些指针为空、相关属性取值）；
 *      - 典型例子：非空输出指针组合映射到 `none/forward/save` 档位（参考
 *        `aclnn_chunk_kda_fwd_prepare.cpp` 的 `GetOutputMask`/`GetOutputMode`/`CheckOutputMode`）；
 *        以及 `inplace_state` 这类属性为 true 时，输出指针必须与状态输入是同一 view
 *        （参考 `aclnn_recurrent_kda.cpp` 对 `inplaceFinalState`/`outputFinalState` 与 `finalState` 的校验）。
 *   9. `CheckParams` 按固定顺序做五类校验，前一步通过后再做下一步，报错才能指到真实原因：
 *      ① 必选入参/输出指针非空 → ② attr 属性合法性（取值域与相互约束）→ ③ 可选输出指针组合合法性
 *      → ④ 数据类型合法性 → ⑤ 输入输出 shape 合法性。
 *   10. 数据类型必须逐张量校验，且**允许列表按平台取**：不同 SoC 的实现支持的 dtype 可能不同
 *      （参考 `aclnn_chunk_gated_delta_rule_fwd.cpp` 用 `GetCurrentPlatformInfo().GetCurNpuArch()`
 *       选择路径、`aclnn_recurrent_kda.cpp` 用 `*_TYPE_SUPPORT_LIST` + `OP_CHECK_DTYPE_NOT_SUPPORT`）；
 *       校验失败返回 `ACLNN_ERR_PARAM_INVALID`，报错要说明"哪张张量、允许哪些 dtype、实际是什么"。
 *   7. Launch 阶段失败返回 ACLNN_ERR_INNER，并给出算子名。
 *   8. 原地路径：对 initialStateRef / finalState 分别做非连续处理（executorPtr->CreateView，注意是
 *      CreateView 而不是 Contiguous），并校验
 *      两者 shape/dtype 一致，否则返回 ACLNN_ERR_PARAM_INVALID；inplaceFinalState=false 时不得写回入参
 *      张量（参考 aclnn_recurrent_kda.cpp 组装 finalStateForKernel 的写法）。
 *   11. **拦截的落点在这里（L2）与 tiling 校验，不在 kernel**：L2 的 CheckParams 覆盖指针/属性组合/
 *      dtype/shape，tiling 覆盖平台差异与规模上限，二者合计覆盖算子 README 的每一条「已知限制」。
 *      host 校验通过后 kernel 只消费合法输入，不再重复校验、不返回错误码；kernel 侧出现校验分支
 *      属于分层缺陷（详见规范 §4.1 第 9 条）。
 */

#include "aclnn_op_name.h"

#include "op_name.h"

#include <cstring>

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct OpNameParams {
    const aclTensor *x = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    double epsilon = 1e-6;
    const aclTensor *yOut = nullptr;
    const aclTensor *stateOut = nullptr;
    const aclTensor *xNormOut = nullptr;
};

// 档位只由输出指针组合推导；与 op_name_output_mask.h 的档位定义一一对应。
int64_t ResolveOutputMode(const OpNameParams &params)
{
    const bool hasState = params.stateOut != nullptr;
    const bool hasXNorm = params.xNormOut != nullptr;
    if (!hasState && !hasXNorm) {
        return optiling::OP_NAME_OUTPUT_MODE_NONE;
    }
    if (hasState && hasXNorm) {
        return optiling::OP_NAME_OUTPUT_MODE_SAVE;
    }
    return -1;  // 只给一个可选输出：非法组合
}

// 平台判定：L2 里需要按 SoC 选择支持范围时，用 GetCurrentPlatformInfo().GetCurNpuArch()
// （参考 aclnn_chunk_gated_delta_rule_fwd.cpp 的 IsAscend950()），不要按编译宏判断。
static bool IsAscend950()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
}

// ④ 数据类型合法性：逐张量校验；允许列表按平台取（不同平台支持的 dtype 可能不同），
// 输出 dtype 与输入的跟随关系也在这一层校验（例如 y 跟随 x、x_norm 固定 FP32）。
// 报错要说明"哪张张量、允许哪些 dtype、实际是什么"；dtype 名称的打印方式按 CANN 提供的能力，
// 没有现成格式化函数时至少要把张量名与允许列表写进报错文本。
aclnnStatus CheckDtype(const OpNameParams &params)
{
    const auto xDtype = params.x->GetDataType();
    CHECK_COND(xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16,
               ACLNN_ERR_PARAM_INVALID, "x 只支持 BF16/FP16，当前 x 不在支持列表内。");

    // g 的允许列表随平台不同：A5 走 BF16/FP32，A2/A3 只支持 BF16（参考真实算子的平台分支写法）。
    const auto gDtype = params.g->GetDataType();
    const bool gSupported = IsAscend950()
        ? (gDtype == DataType::DT_BF16 || gDtype == DataType::DT_FLOAT)
        : (gDtype == DataType::DT_BF16);
    CHECK_COND(gSupported, ACLNN_ERR_PARAM_INVALID,
               "当前平台 g 只支持 %s，传入 dtype 不在支持列表内。",
               IsAscend950() ? "BF16/FP32" : "BF16");

    // 可选输入：非空才校验 dtype（空指针的语义在 ③ 已经处理）。
    if (params.aLogOptional != nullptr) {
        CHECK_COND(params.aLogOptional->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "非空 aLogOptional 只支持 FP32。");
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(params.initialStateOptional->GetDataType() == xDtype,
                   ACLNN_ERR_PARAM_INVALID,
                   "initialStateOptional 的 dtype 必须与 x 一致（state 与主输入同 dtype）。");
    }

    // 输出 dtype 契约：y/state 跟随 x，x_norm 固定 FP32；不一致直接拒绝，不靠 kernel 兜底。
    CHECK_COND(params.yOut->GetDataType() == xDtype, ACLNN_ERR_PARAM_INVALID,
               "yOut 的 dtype 必须与 x 一致。");
    if (params.stateOut != nullptr) {
        CHECK_COND(params.stateOut->GetDataType() == xDtype, ACLNN_ERR_PARAM_INVALID,
                   "非空 stateOut 的 dtype 必须与 x 一致。");
    }
    if (params.xNormOut != nullptr) {
        CHECK_COND(params.xNormOut->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "非空 xNormOut 只支持 FP32。");
    }
    return ACLNN_SUCCESS;
}

// ⑤ 输入输出 shape 合法性：rank、维度对应关系、state 与输入的一致性；与 README「已知限制」逐条对应。
aclnnStatus CheckShape(const OpNameParams &params)
{
    CHECK_COND(params.x->GetViewShape().GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "x 必须是 rank-4（BSND/BNSD），当前 rank 不匹配。");
    CHECK_COND(params.x->GetViewShape().GetDim(3) == DIM_128, ACLNN_ERR_PARAM_INVALID,
               "D 只支持 128，当前 D=%ld。",
               static_cast<long>(params.x->GetViewShape().GetDim(3)));
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(IsContiguous(params.initialStateOptional) ||
                       params.initialStateOptional->GetViewShape().GetDimNum() >= 3,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional 至少是 rank-3。");
    }
    // cu_seqlens 单调性、chunk_indices 成对出现等元数据约束也在这里校验。
    return ACLNN_SUCCESS;
}

// CheckParams 的推荐顺序：① 必选指针非空 → ② attr 合法性 → ③ 可选输出指针组合 → ④ dtype → ⑤ shape。
// 顺序固定下来，报错才会指向真实原因（先报 nullptr 再报 dtype，而不是让 dtype 覆盖掉空指针问题）。
aclnnStatus CheckParams(const OpNameParams &params)
{
    // ① 必选输入/输出：缺一个就返回 PARAM_NULLPTR，并点名是哪个。
    CHECK_COND(params.x != nullptr && params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "x 与 g 不能为 nullptr。");
    CHECK_COND(params.yOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yOut 不能为 nullptr。");
    CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layout 不能为 nullptr。");

    // ② attr 属性合法性：取值域与相互约束。
    CHECK_COND(params.scale > 0.0, ACLNN_ERR_PARAM_INVALID,
               "scale 必须为正数，当前 scale=%f。", params.scale);
    CHECK_COND(params.epsilon > 0.0, ACLNN_ERR_PARAM_INVALID,
               "epsilon 必须为正数，当前 epsilon=%f。", params.epsilon);
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize 只支持 64/128，当前 chunkSize=%ld。",
               static_cast<long>(params.chunkSize));

    // ③ 可选输出指针组合合法性：属性取值 + 指针非空组合必须落在文档化档位内。
    const int64_t outputMode = ResolveOutputMode(params);
    CHECK_COND(outputMode >= 0, ACLNN_ERR_PARAM_INVALID,
               "stateOut/xNormOut 必须同时给出或同时为空，当前 stateOut=%s xNormOut=%s。",
               params.stateOut ? "非空" : "空", params.xNormOut ? "非空" : "空");

    // ④ 数据类型合法性（允许列表按平台取）。
    aclnnStatus dtypeStatus = CheckDtype(params);
    if (dtypeStatus != ACLNN_SUCCESS) {
        return dtypeStatus;
    }

    // ⑤ 输入输出 shape 合法性。
    return CheckShape(params);
}

// 只对"接口契约声明要求连续"的输入做一次显式连续化；哪些输入要求连续见 docs/api.md 的连续性契约，
// 不要在实现里凭感觉扩大范围（整批连续化会同时破坏 state 的原地语义和 host 侧性能）。
aclnnStatus MakeInputsContiguous(OpNameParams &params, aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {&params.x, &params.g, &params.aLogOptional};
    for (const aclTensor **input : inputs) {
        if (*input == nullptr || IsContiguous(*input)) {
            continue;
        }
        *input = l0op::Contiguous(*input, executor);
        CHECK_RET(*input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

// state 类输入保持调用方 stride：非连续时用 CreateView 交给算子按 stride 寻址，
// 不调用 l0op::Contiguous（与 aclnn_recurrent_kda.cpp 的 CreateViewIfNonContiguous 一致）。
aclnnStatus MakeStateViews(OpNameParams &params, aclOpExecutor *executor)
{
    const aclTensor **states[] = {&params.initialStateOptional, &params.stateOut};
    for (const aclTensor **state : states) {
        if (*state == nullptr || IsContiguous(*state)) {
            continue;
        }
        *state = executor->CreateView(*state, (*state)->GetViewShape(), (*state)->GetStorageShape(),
                                      (*state)->GetViewStrides(), (*state)->GetViewOffset());
        CHECK_RET(*state != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnOpNameGetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, const aclTensor *yOut,
    const aclTensor *stateOut, const aclTensor *xNormOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnOpName,
                   DFX_IN(x, g, aLogOptional, initialStateOptional, layout, scale, chunkSize,
                          epsilon),
                   DFX_OUT(yOut, stateOut, xNormOut));
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize 不能为 nullptr。");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor 不能为 nullptr。");

    OpNameParams params{x, g, aLogOptional, initialStateOptional, cuSeqlensOptional,
                             chunkIndicesOptional, layout, scale, chunkSize, epsilon,
                             yOut, stateOut, xNormOut};
    const int64_t outputMode = ResolveOutputMode(params);
    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    status = MakeInputsContiguous(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const auto result = l0op::OpName(
        params.x, params.g, params.aLogOptional, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.layout, params.scale,
        params.chunkSize, params.epsilon, params.yOut, params.stateOut, params.xNormOut,
        outputMode, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnOpName(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnOpName);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "OpName AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
