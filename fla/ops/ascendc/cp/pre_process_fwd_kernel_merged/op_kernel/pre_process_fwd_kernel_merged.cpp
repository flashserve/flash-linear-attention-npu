/*!
 * \file pre_process_fwd_kernel_merged.cpp
 * \brief pre_process_fwd_kernel_merged：薄入口：tiling 注册/解析 + workspace + AIC/AIV 分派

 * ============================ Stage 表（每 chunk 一轮跨核流水）============================
 * | 阶段      | 执行 | 动作                                                      | 结束同步            |
 * | --------- | ---- | --------------------------------------------------------- | ------------------- |
 * | prologue | AIV | h←0、m←I（fp32 常驻 UB；bf16 落 GM 供 AIC） | 并入 kFlagInputs(1) |
 * | Stage c   | AIV  | 载入 W/k/v(+gate) → left/decay；尾块零填充                  | kFlagInputs(1)      |
 * | ①③        | AIC  | vTmp = W_c @ bf16(h)；T1 = W_c @ bf16(m)                    | kFlagHalf1(3)       |
 * | v_new     | AIV  | v_new = dg ⊙ (v − vTmp) → bf16(v_new)                       | kFlagVNew(4)        |
 * | ②         | AIC  | dH = k_c^T @ bf16(v_new)                                    | 与 ④ 合并           |
 * | h 相位 | AIV | h = decay⊙h + dH（就地，bf16(h) 落 GM） | 并入 kFlagInputs(1) |
 * | ④         | AIC  | T2 = left^T @ bf16(T1)                                      | kFlagDH(5)          |
 * | m 相位 | AIV | m = decay⊙m − T2（就地，bf16(m) 落 GM） | 并入 kFlagInputs(1) |
 * | epilogue | AIV | h/m 写 hm 输出（每子核一次跨步 DataCopy） | — |
 *
 * ============================ 布局表 ============================
 * 形状（编译期常量）：BT=64（CV_BT）｜K=V=128（CV_K/CV_V）｜AIV 子核数 PPFM_SUB=2
 *                    状态行块 PPFM_SBRB=64（950）/32（A2 回退）｜staging 段 PPFM_SEGROWS=32
 *
 * UB（每个 AIV 核，见 UB_* 常量与 PPFM_VEC_UB_BYTES）：
 * | 区域                     | 用途                                        | 大小（950，cb_=128） |
 * | ------------------------ | ------------------------------------------- | -------------------- |
 * | UB_ROW0/1/2_BF,F32 | 行缓冲（对角/诊断/单行回读） | 数个 128 元素 |
 * | UB_DG/UB_DECAY(_PREV)/…  | dg、decay、exp 标量槽                        | 128~512 B            |
 * | UB_STATE_F/BF            | 状态区块（非常驻路径用）                    | 16 KiB               |
 * | UB_EXT_F                 | vTmp 落点（L0C→UB，SPLIT_M 两半）            | 32 KiB               |
 * | UB_KBLK/WBLK/VBLK_BF     | staging 段缓冲（按子核本地段寻址）           | 8 KiB x2 各          |
 * | UB_SCR_F/BF              | left / v_new 计算暂存                        | 16 KiB / 8 KiB       |
 * | UB_DH_CV / UB_T2_CV      | dH 槽 / T2 槽（L0C→UB，仅 950）              | 32 KiB 各            |
 * | UB_H_UB / UB_M_UB        | h、m 常驻（每子核 64 行 x cb_）               | 32 KiB 各            |
 *
 * GM（每个工作项一块 workspace，PPFM_CORE_WS_BYTES=768 KiB；偏移见 op_kernel/*_struct.h）：
 * 状态 h/m（fp32 + bf16）｜本 chunk 输入 W/k/left/v（bf16，k/left 双槽按 chunk 奇偶）｜
 * vTmp(fp32)｜bf16(v_new)｜dH/T1/T2(950 走 UB 槽)｜gate(g/gk)｜hm 输出（[K][V+K] 行主）
 *
 * ============================ 同步协议表 ============================
 * 跨核 flag（CrossCoreSetFlag<0x4, …>/WaitFlag；子核各占 id 与 id+PPFM_SUBFLAG_STRIDE）：
 * | id | 名称 | 方向 | 载荷 | 生产者 -> 消费者 |
 * | -- | ------------ | --------- | ------------------------------------------- | -------------------------- |
 * | 1 | kFlagInputs | AIV→AIC | staging(W/k/left/v)+bf16(h/m) 就位 | Stage/h → mm1/mm3 |
 * | 2 | kFlagState | AIV→AIC | prologue 的 h/m 初值（并入 1） | prologue → mm1/mm3 |
 * | 3 | kFlagHalf1 | AIC→AIV | vTmp（mm1 的 C；950 上 mm1 完成即发） | mm1 → v_new |
 * | 4  | kFlagVNew    | AIV → AIC | bf16(v_new)                                  | v_new → mm2                 |
 * | 5 | kFlagDH | AIC→AIV | dH(mm2) 与 T2(mm4)，一次通知 | mm2/mm4 → h/m 相位 |
 * | 8 | kFlagDhFree | AIV→AIC | dH 槽归还信用（CV 单槽） | h 相位 → mm2 写槽 |
 * | 10 | kFlagT2Free | AIV→AIC | T2 槽归还信用（CV 单槽） | m 相位 → mm4 写槽 |
 *
 * 核内事件对（AIV 侧；A2 无事件对时退化为 PipeBarrier）：
 * | EVENT_ID | 事件对           | 用途                                   |
 * | -------- | ---------------- | -------------------------------------- |
 * | 0        | MTE2_V           | 载入完成 -> 向量计算                    |
 * | 1        | V_MTE3           | 向量算完 -> UB→GM 落盘                  |
 * | 2        | MTE3_MTE2        | 上一趟落盘读完 -> 覆盖同一 UB            |
 * | 3        | MTE3_V           | 落盘读完 -> 覆盖向量源                  |
 * | 4        | V_MTE2           | 向量读完 UB -> 下一趟载入覆盖            |
 * | 5        | MTE2_MTE3        | 载入完成 -> 落盘                        |
 * | 6/7      | V_S / S_V        | 向量 <-> 标量（Exp2Scalar 等）           |
 * AIC 侧对 MTE1/M/FIX 用 SetFlag/WaitFlag 成对（CopyGmToL1 -> MMAD -> fixpipe）。
 *
 * 本文件由 kernel 主文件按"机械搬运"拆出（代码与拆分前逐字符相同）；
 * Stage/布局/同步协议详表见 *_common.h 顶部注释与 docs/design.md。
 */

#include "pre_process_fwd_kernel_merged_kernel.h"
#include "pre_process_fwd_kernel_merged_tiling_key.h"

// ---- 入口常量自检（§4.4 ②/§7.7：常量集中处配 static_assert）----
// tiling 结构由 host 写入、kernel 解析，两侧必须同源同尺寸
static_assert(sizeof(GDN::PreProcessFwdKernelMergedTilingData) <= 4096,
              "tiling 结构超过预留容量，请检查 host/kernel 是否同源");
static_assert(sizeof(GDN::PreProcessFwdKernelMergedTilingData) % 8 == 0,
              "tiling 结构需按 8B 对齐（host 侧写入按 int64 序列化）");
// 每个工作项一块 workspace：按 512B 对齐，供 UB<->GM 的 DataCopy 落点使用
static_assert(GDN::PPFM_CORE_WS_BYTES % 512 == 0,
              "每核 workspace 尺寸需按 512B 对齐");
#ifndef TORCH_MODE
// GATE_MODE 由 *_tiling_key.h 的 ASCENDC_TPL_SEL 实例化（1/2/3）；与 host 侧
// SetTilingKey(gateMode+1) 一一对应。kernel 内部仍读 tiling->gateMode，两者由 gate 保证一致。
template <int GATE_MODE>
__global__ __aicore__ void pre_process_fwd_kernel_merged(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR bg, GM_ADDR v,
    GM_ADDR cu_seqlens, GM_ADDR hm, GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(GATE_MODE >= PPFM_TPL_GATE_G && GATE_MODE <= PPFM_TPL_GATE_BG,
                  "非法 GATE_MODE TilingKey");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GDN::PreProcessFwdKernelMergedTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::PreProcessFwdKernelMergedTilingData, tilingData, tiling);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    (void)bg;   // DPLR（USE_BG）不支持：host 侧已拒绝非空 bg，这里只是占位
    GDN::PpFwdCtx ctx;
    ctx.k = k;
    ctx.w = w;
    ctx.u = u;
    ctx.g = g;
    ctx.gk = gk;
    ctx.v = v;
    ctx.cu = cu_seqlens;
    ctx.hm = hm;
    ctx.ws = userWS;
    ctx.tiling = &tilingData;
    if ASCEND_IS_AIC {
        GDN::PpFwdCube cube(ctx);
        cube.Run();
    }
    if ASCEND_IS_AIV {
        GDN::PpFwdVector vec(ctx);
        vec.Run();
    }
}
#endif
