#pragma once

#include "chunk_kda_fwd_32_common.h"

// 唯一的 chunk32 算子类；C/V 仅提供该类在不同执行单元上的成员函数实现。
template <typename T, bool IsVarlen = false, bool HasTail = false>
class Kernel
{
public:
    SyncCV syncCV;
#if defined(__DAV_C310_CUBE__)

public:

	__gm__ void *tmp_gm;


	// For compute use
	__aicore__ inline void Init();

	template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void copy_l1_to_l0a(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0a);

	template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void copy_l1_to_l0b(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0b);

	// A 转置:V2+ifTranspose(轴语义:m_slot=K,n=M;dstStride=M/16=n>>4,对齐 adv_api matmul 官方用法)
	template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void copy_l1_to_l0a_transpose(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0a);

	template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void copy_l1_to_l0b_transpose(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0b);

	static constexpr AscendC::FixpipeConfig KDA_CFG_UB_NZ = {AscendC::CO2Layout::NZ, true};
	static constexpr AscendC::FixpipeConfig KDA_CFG_UB_ND = {AscendC::CO2Layout::ROW_MAJOR, true};

	template<uint64_t m_slot, uint64_t n>
	__aicore__ inline void copy_l0c_to_l1_bf16(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<float> &l0c);

	template<bool nz2nd, uint64_t dst_id, uint64_t m_slot, uint64_t n>
	__aicore__ inline void copy_l0c_to_ub_bf16(const AscendC::LocalTensor<T> &ub, const AscendC::LocalTensor<float> &l0c);

	template<bool nz2nd, uint64_t m_slot, uint64_t n>
	__aicore__ inline void copy_l0c_to_ub_f32(const AscendC::LocalTensor<float> &ub, const AscendC::LocalTensor<float> &l0c);

	template<uint64_t vid, uint64_t len, uint64_t element_size>
	__aicore__ inline void copy_l1_to_ub(const AscendC::LocalTensor<float> &ub, const AscendC::LocalTensor<float> &l1);

	template<uint64_t m_slot, uint64_t k, uint64_t n>
	__aicore__ inline void mmad(const AscendC::LocalTensor<float> &l0c, const AscendC::LocalTensor<T> &l0a, const AscendC::LocalTensor<T> &l0b);

	template<uint64_t m_slot, uint64_t k, uint64_t n>
	__aicore__ inline void mmad_acc(const AscendC::LocalTensor<float> &l0c, const AscendC::LocalTensor<T> &l0a, const AscendC::LocalTensor<T> &l0b);


public:

	// 任务化:与 AIV 侧使用同一任务序列;0 表示单任务
	uint64_t task_count = 0, block_num = 0;
	// varlen:AIC 从 tiling GM 的 cu_seqlens 解析每任务组数,
	// 空序列与 AIV 同步跳过(同一数据,判定确定性一致)。
	uint64_t is_varlen = 0, hv_real = 0, cu_base = 0;

	// Define Buffer Addresses and Variables' Buffer IDs

	// 与 AIV 相同的 fp32 gate 双槽重排镜像:id28 前 8KB 由主/auto kminus
	// 顺序复用;id0 的两个 score buffer 顺序复用 319 的后 8KB。
	const uint32_t addrUB[31] = {319, 32, 49, 66, 68, 77, 86, 86, 87, 88, 104, 120, 136, 137, 139, 140, 156, 156, 165, 166, 170, 187, 204, 236, 268, 269, 286, 303, 335, 352};
	const uint32_t qn1_ub_slot = 2;
	const uint32_t qn_ub_slot = 2;
	const uint32_t kn1_ub_slot = 9;
	const uint32_t kn_ub_slot = 9;
	const uint32_t betaA_ub_slot = 24;
	const uint32_t beta_ub_slot = 24;
	const uint32_t betaB_ub_slot = 12;
	const uint32_t gg1_ub_slot = 10;
	const uint32_t gamma1_pre_ub_slot = 18;
	const uint32_t gamma2_pre_ub_slot = 18;
	const uint32_t gg_ub_slot = 10;
	const uint32_t gamma_pre_ub_slot = 18;
	const uint32_t qd_ub_slot = 1;
	const uint32_t qd2_ub_slot = 4;
	const uint32_t kd_ub_slot = 25;
	const uint32_t kd2_ub_slot = 17;
	const uint32_t ki_ub_slot = 28;
	const uint32_t kr_ub_slot = 22;
	const uint32_t gamma_ub_slot = 13;
	const uint32_t kplus_kminus_ub_slot = 27;
	const uint32_t qplus_kminus_ub_slot = 25;
	const uint32_t l_ub_slot = 22;
	const uint32_t mqk_ub_slot = 19;
	const uint32_t m_ub_slot = 21;
	const uint32_t m1_ub_slot = 28;
	const uint32_t s1_partial_ub_slot = 23;
	const uint32_t s1_ub_slot = 23;
	const uint32_t ss1_ub_slot = 23;
	const uint32_t kds1_ub_slot = 27;
	const uint32_t v_corr_a_ub_slot = 17;
	const uint32_t out1_ub_slot = 11;
	const uint32_t score_delta_a_ub1_slot = 1;
	const uint32_t score_delta_a_ub2_slot = 26;
	const uint32_t score_delta_a_ub3_slot = 15;
	const uint32_t score_delta_a_ub4_slot = 29;
	const uint32_t s2_partial_ub_slot = 23;
	const uint32_t s2_ub_slot = 23;
	const uint32_t ss2_ub_slot = 23;
	const uint32_t kds2_ub_slot = 26;
	const uint32_t v_corr_b_ub_slot = 4;
	const uint32_t out2_ub_slot = 15;
	const uint32_t score_delta_b_ub1_slot = 27;
	const uint32_t score_delta_b_ub2_slot = 22;
	const uint32_t score_delta_b_ub3_slot = 26;
	const uint32_t score_delta_b_ub4_slot = 25;
	const uint32_t qn1_ub_auto_unroll_id_1_slot = 11;
	const uint32_t qn_ub_auto_unroll_id_1_slot = 11;
	const uint32_t kn1_ub_auto_unroll_id_1_slot = 29;
	const uint32_t kn_ub_auto_unroll_id_1_slot = 29;
	const uint32_t beta1_ub_auto_unroll_id_1_slot = 8;
	const uint32_t beta_ub_auto_unroll_id_1_slot = 8;
	const uint32_t beta2_ub_auto_unroll_id_1_slot = 14;
	const uint32_t gg1_ub_auto_unroll_id_1_slot = 26;
	const uint32_t gamma1_pre_ub_auto_unroll_id_1_slot = 7;
	const uint32_t gamma2_pre_ub_auto_unroll_id_1_slot = 7;
	const uint32_t gg_ub_auto_unroll_id_1_slot = 26;
	const uint32_t gamma_pre_ub_auto_unroll_id_1_slot = 7;
	const uint32_t qd_ub_auto_unroll_id_1_slot = 21;
	const uint32_t qd2_ub_auto_unroll_id_1_slot = 5;
	const uint32_t kd_ub_auto_unroll_id_1_slot = 20;
	const uint32_t kd2_ub_auto_unroll_id_1_slot = 4;
	const uint32_t ki_ub_auto_unroll_id_1_slot = 28;
	const uint32_t kr_ub_auto_unroll_id_1_slot = 25;
	const uint32_t gamma_ub_auto_unroll_id_1_slot = 3;
	const uint32_t kplus_kminus_ub_auto_unroll_id_1_slot = 1;
	const uint32_t qplus_kminus_ub_auto_unroll_id_1_slot = 0;
	const uint32_t l_ub_auto_unroll_id_1_slot = 5;
	const uint32_t mqk_ub_auto_unroll_id_1_slot = 20;
	const uint32_t m_ub_auto_unroll_id_1_slot = 19;
	const uint32_t m1_ub_auto_unroll_id_1_slot = 17;
	const uint32_t s1_partial_ub_auto_unroll_id_1_slot = 23;
	const uint32_t s1_ub_auto_unroll_id_1_slot = 23;
	const uint32_t ss1_ub_auto_unroll_id_1_slot = 23;
	const uint32_t kds1_ub_auto_unroll_id_1_slot = 26;
	const uint32_t v_corr_a_ub_auto_unroll_id_1_slot = 27;
	const uint32_t out1_ub_auto_unroll_id_1_slot = 15;
	const uint32_t score_delta_a_ub1_auto_unroll_id_1_slot = 27;
	const uint32_t score_delta_a_ub2_auto_unroll_id_1_slot = 21;
	const uint32_t score_delta_a_ub3_auto_unroll_id_1_slot = 20;
	const uint32_t score_delta_a_ub4_auto_unroll_id_1_slot = 10;
	const uint32_t s2_partial_ub_auto_unroll_id_1_slot = 23;
	const uint32_t s2_ub_auto_unroll_id_1_slot = 23;
	const uint32_t ss2_ub_auto_unroll_id_1_slot = 23;
	const uint32_t kds2_ub_auto_unroll_id_1_slot = 0;
	const uint32_t v_corr_b_ub_auto_unroll_id_1_slot = 25;
	const uint32_t out2_ub_auto_unroll_id_1_slot = 2;
	const uint32_t score_delta_b_ub1_auto_unroll_id_1_slot = 2;
	const uint32_t score_delta_b_ub2_auto_unroll_id_1_slot = 9;
	const uint32_t score_delta_b_ub3_auto_unroll_id_1_slot = 27;
	const uint32_t score_delta_b_ub4_auto_unroll_id_1_slot = 20;

	const uint32_t addrL1[31] = {0, 8, 16, 48, 64, 80, 81, 81, 82, 84, 92, 124, 156, 172, 180, 184, 216, 224, 256, 288, 320, 352, 368, 384, 400, 416, 424, 456, 472, 504};
	const uint32_t beta_slot = 26;
	const uint32_t qPlus_slot = 23;
	const uint32_t qPlusB_slot = 29;
	const uint32_t kPlus_slot = 24;
	const uint32_t kPlusB_slot = 28;
	const uint32_t kMinus_slot = 22;
	const uint32_t kRes_slot = 19;
	const uint32_t gamma_slot = 25;
	const uint32_t maskedQk_slot = 27;
	const uint32_t mm_slot = 14;
	const uint32_t invAuxT_slot = 14;
	const uint32_t m_slot = 14;
	const uint32_t invAux1_slot = 20;
	const uint32_t scoreA_slot = 15;
	const uint32_t vCorrA_slot = 16;
	const uint32_t stateAccA_slot = 18;
	const uint32_t scoreB_slot = 28;
	const uint32_t vCorrB_slot = 20;
	const uint32_t stateAccB_slot = 17;
	const uint32_t beta_auto_unroll_id_1_slot = 10;
	const uint32_t qd_auto_unroll_id_1_slot = 3;
	const uint32_t qd2_auto_unroll_id_1_slot = 13;
	const uint32_t kd_auto_unroll_id_1_slot = 12;
	const uint32_t kd2_auto_unroll_id_1_slot = 9;
	const uint32_t ki_auto_unroll_id_1_slot = 4;
	const uint32_t kr_auto_unroll_id_1_slot = 21;
	const uint32_t gamma_auto_unroll_id_1_slot = 8;
	const uint32_t mqk_auto_unroll_id_1_slot = 16;
	const uint32_t mm_auto_unroll_id_1_slot = 2;
	const uint32_t inv_aux_t_auto_unroll_id_1_slot = 2;
	const uint32_t m_auto_unroll_id_1_slot = 2;
	const uint32_t inv_aux1_auto_unroll_id_1_slot = 7;
	const uint32_t s1_auto_unroll_id_1_slot = 11;
	const uint32_t v_corr_a_auto_unroll_id_1_slot = 0;
	const uint32_t y1_auto_unroll_id_1_slot = 18;
	const uint32_t s2_auto_unroll_id_1_slot = 17;
	const uint32_t v_corr_b_auto_unroll_id_1_slot = 1;
	const uint32_t y2_auto_unroll_id_1_slot = 0;

	const uint32_t addrL0A[9] = {0, 24, 24, 48, 48, 58, 59, 63};
	const uint32_t kplus_kminus_a_slot = 0;
	const uint32_t qplus_kminus_a_slot = 2;
	const uint32_t inv_aux1_a_slot = 5;
	const uint32_t inv_aux2_a_slot = 2;
	const uint32_t kplus_qplus_score_a_a_slot = 0;
	const uint32_t y1_a_slot = 2;
	const uint32_t iter_out_a_a_slot = 2;
	const uint32_t kplus_qplus_score_b_a_slot = 0;
	const uint32_t y2_a_slot = 0;
	const uint32_t iter_out_b_a_slot = 2;
	const uint32_t kplus_kminus_a_auto_unroll_id_1_slot = 0;
	const uint32_t qplus_kminus_a_auto_unroll_id_1_slot = 2;
	const uint32_t inv_aux1_a_auto_unroll_id_1_slot = 7;
	const uint32_t inv_aux2_a_auto_unroll_id_1_slot = 2;
	const uint32_t kplus_qplus_score_a_a_auto_unroll_id_1_slot = 0;
	const uint32_t y1_a_auto_unroll_id_1_slot = 2;
	const uint32_t iter_out_a_a_auto_unroll_id_1_slot = 2;
	const uint32_t kplus_qplus_score_b_a_auto_unroll_id_1_slot = 0;
	const uint32_t y2_a_auto_unroll_id_1_slot = 2;
	const uint32_t iter_out_b_a_auto_unroll_id_1_slot = 0;

	const uint32_t addrL0B[9] = {0, 32, 32, 48, 48, 48, 49, 49};
	const uint32_t kplus_kminus_b_slot = 0;
	const uint32_t qplus_kminus_b_slot = 2;
	const uint32_t inv_aux1_b_slot = 5;
	const uint32_t inv_aux2_b_slot = 2;
	const uint32_t kplus_qplus_score_a_b_slot = 0;
	const uint32_t y1_b_slot = 2;
	const uint32_t iter_out_a_b_slot = 2;
	const uint32_t kplus_qplus_score_b_b_slot = 0;
	const uint32_t y2_b_slot = 0;
	const uint32_t iter_out_b_b_slot = 2;
	const uint32_t kplus_kminus_b_auto_unroll_id_1_slot = 0;
	const uint32_t qplus_kminus_b_auto_unroll_id_1_slot = 2;
	const uint32_t inv_aux1_b_auto_unroll_id_1_slot = 7;
	const uint32_t inv_aux2_b_auto_unroll_id_1_slot = 2;
	const uint32_t kplus_qplus_score_a_b_auto_unroll_id_1_slot = 0;
	const uint32_t y1_b_auto_unroll_id_1_slot = 2;
	const uint32_t iter_out_a_b_auto_unroll_id_1_slot = 2;
	const uint32_t kplus_qplus_score_b_b_auto_unroll_id_1_slot = 0;
	const uint32_t y2_b_auto_unroll_id_1_slot = 2;
	const uint32_t iter_out_b_b_auto_unroll_id_1_slot = 0;

	const uint32_t addrL0C[9] = {0, 16, 32, 48, 64, 80, 96, 112};
	const uint32_t kplus_kminus_c_slot = 5;
	const uint32_t qplus_kminus_c_slot = 2;
	const uint32_t inv_aux1_c_slot = 2;
	const uint32_t inv_aux2_c_slot = 0;
	const uint32_t kds1_c_slot = 5;
	const uint32_t qds1_c_slot = 1;
	const uint32_t y1_c_slot = 2;
	const uint32_t out1_c_slot = 1;
	const uint32_t score_delta_a_c1_slot = 5;
	const uint32_t score_delta_a_c2_slot = 6;
	const uint32_t score_delta_a_c3_slot = 4;
	const uint32_t score_delta_a_c4_slot = 7;
	const uint32_t kds2_c_slot = 3;
	const uint32_t qds2_c_slot = 1;
	const uint32_t y2_c_slot = 3;
	const uint32_t out2_c_slot = 1;
	const uint32_t score_delta_b_c1_slot = 6;
	const uint32_t score_delta_b_c2_slot = 7;
	const uint32_t score_delta_b_c3_slot = 0;
	const uint32_t score_delta_b_c4_slot = 4;
	const uint32_t kplus_kminus_c_auto_unroll_id_1_slot = 4;
	const uint32_t qplus_kminus_c_auto_unroll_id_1_slot = 7;
	const uint32_t inv_aux1_c_auto_unroll_id_1_slot = 7;
	const uint32_t inv_aux2_c_auto_unroll_id_1_slot = 7;
	const uint32_t kds1_c_auto_unroll_id_1_slot = 1;
	const uint32_t qds1_c_auto_unroll_id_1_slot = 0;
	const uint32_t y1_c_auto_unroll_id_1_slot = 4;
	const uint32_t out1_c_auto_unroll_id_1_slot = 0;
	const uint32_t score_delta_a_c1_auto_unroll_id_1_slot = 3;
	const uint32_t score_delta_a_c2_auto_unroll_id_1_slot = 1;
	const uint32_t score_delta_a_c3_auto_unroll_id_1_slot = 6;
	const uint32_t score_delta_a_c4_auto_unroll_id_1_slot = 7;
	const uint32_t kds2_c_auto_unroll_id_1_slot = 7;
	const uint32_t qds2_c_auto_unroll_id_1_slot = 6;
	const uint32_t y2_c_auto_unroll_id_1_slot = 1;
	const uint32_t out2_c_auto_unroll_id_1_slot = 6;
	const uint32_t score_delta_b_c1_auto_unroll_id_1_slot = 5;
	const uint32_t score_delta_b_c2_auto_unroll_id_1_slot = 2;
	const uint32_t score_delta_b_c3_auto_unroll_id_1_slot = 0;
	const uint32_t score_delta_b_c4_auto_unroll_id_1_slot = 7;

	// Define operator functions

	__aicore__ inline void matmul_kplus_kminus(int loop_id, uint32_t kplus_kminus_c_slot, uint32_t kplus_kminus_a_slot, uint32_t kplus_kminus_b_slot);

	__aicore__ inline void matmul_kplus_qplus_score_a(int loop_id, uint32_t kds1_c_slot, uint32_t qds1_c_slot, uint32_t kplus_qplus_score_a_a_slot, uint32_t kplus_qplus_score_a_b_slot);
	__aicore__ inline void matmul_state_acc_a(int loop_id, uint32_t y1_c_slot, uint32_t y1_a_slot, uint32_t y1_b_slot);
	__aicore__ inline void matmul_iter_out_a(int loop_id, uint32_t out1_c_slot, uint32_t score_delta_a_c1_slot, uint32_t score_delta_a_c2_slot, uint32_t score_delta_a_c3_slot, uint32_t score_delta_a_c4_slot, uint32_t iter_out_a_a_slot, uint32_t iter_out_a_b_slot, uint32_t qds1_c_slot);











	__aicore__ inline void copy_l1_to_ub_gamma(int loop_id, uint32_t gamma_ub_slot, uint32_t betaB_ub_slot, uint32_t gamma_slot, uint32_t beta_slot);
	__aicore__ inline void copy_l1_to_l0ab_kplus_kminus(int loop_id, uint32_t kplus_kminus_a_slot, uint32_t kplus_kminus_b_slot, uint32_t kPlus_slot, uint32_t kPlusB_slot, uint32_t kMinus_slot);



	__aicore__ inline void copy_l1_to_l0ab_kplus_qplus_score_a(int loop_id, uint32_t kplus_qplus_score_a_a_slot, uint32_t kplus_qplus_score_a_b_slot, uint32_t kPlus_slot, uint32_t qPlus_slot, uint32_t scoreA_slot);
	__aicore__ inline void copy_l1_to_l0ab_y1(int loop_id, uint32_t y1_a_slot, uint32_t y1_b_slot, uint32_t m_slot, uint32_t vCorrA_slot);
	__aicore__ inline void copy_l1_to_l0ab_iter_out_a(int loop_id, uint32_t iter_out_a_a_slot, uint32_t iter_out_a_b_slot, uint32_t kRes_slot, uint32_t maskedQk_slot, uint32_t stateAccA_slot);
	__aicore__ inline void copy_l1_to_l0ab_kplus_qplus_score_b(int loop_id, uint32_t kplus_qplus_score_b_a_slot, uint32_t kplus_qplus_score_b_b_slot, uint32_t kPlus_slot, uint32_t qPlus_slot, uint32_t scoreB_slot);
	__aicore__ inline void copy_l1_to_l0ab_y2(int loop_id, uint32_t y2_a_slot, uint32_t y2_b_slot, uint32_t m_slot, uint32_t vCorrB_slot);
	__aicore__ inline void copy_l1_to_l0ab_iter_out_b(int loop_id, uint32_t iter_out_b_a_slot, uint32_t iter_out_b_b_slot, uint32_t kRes_slot, uint32_t maskedQk_slot, uint32_t stateAccB_slot);











	__aicore__ inline void copy_l0c_to_ub_kplus_kminus(int loop_id, uint32_t kplus_kminus_ub_slot, uint32_t kplus_kminus_c_slot);

	__aicore__ inline void copy_l0c_to_ub_kds1(int loop_id, uint32_t kds1_ub_slot, uint32_t kds1_c_slot);
	__aicore__ inline void copy_l0c_to_l1_y1(int loop_id, uint32_t stateAccA_slot, uint32_t y1_c_slot);

	__aicore__ inline void copy_score_delta_l0c_to_ub(int loop_id, uint32_t score_delta_a_ub1_slot, uint32_t score_delta_a_c1_slot);




























	template <bool STEADY>
	__aicore__ inline void stage_l0ab_y1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_state_acc_a_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l1_y1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_iter_out_a_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_out_a_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_writeback_score_delta_a_primary(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_kminus_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_kminus_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_kminus_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_kminus_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_qplus_score_b_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kplus_kminus_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kplus_kminus_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_qplus_score_a_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_gamma_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_y2_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_state_acc_a_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l1_y1_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_iter_out_b_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_out_a_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_writeback_score_delta_b_primary(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u0_x2(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_qplus_score_a_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_qplus_score_a_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_y1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_state_acc_a_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l1_y1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_iter_out_a_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_out_a_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_writeback_score_delta_a_unrolled(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_kminus_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_kminus_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_kminus_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_qplus_score_b_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_kminus_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kplus_kminus_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kplus_kminus_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_qplus_score_a_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_gamma_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u1_x2(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_y2_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_state_acc_a_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l1_y1_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_iter_out_b_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_out_a_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_writeback_score_delta_b_unrolled(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u1_x3(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_l0ab_kplus_qplus_score_a_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kplus_qplus_score_a_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_ub_kds1_u0_x3(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_finish_pipeline(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void RunPipelineStages(int iter, int groupCount);

	// Define Kernel Process
	__aicore__ inline void Process(int group_count);

#elif defined(__DAV_C310_VEC__)

public:
	AscendC::GlobalTensor<T> g_gm, q_gm, k_gm, v_gm, b_gm, tmp_gm, o_gm;
	AscendC::GlobalTensor<float> g32_gm, b32_gm;  // fp32 原生组合的 g/beta GM 视图(与 16 位视图同址)
	uint64_t gate_flags = 0;  // tiling gateFlags:bit0 = g fp32,bit1 = beta fp32
	AscendC::GlobalTensor<float> dt_gm;
	// 裸地址桥接(Stage 3)
	template <typename U>
	__aicore__ inline uint32_t KdaAddrV(U *p);

	AscendC::GlobalTensor<float> a_gm;
	AscendC::GlobalTensor<float> istate_gm, fstate_gm;  // state IO(fp32 [H,K,V],按头+半幅寻址)
	uint32_t has_istate = 0, out_fstate = 0;
	float a;
	uint32_t group_count;  // 128-token 组数(= tiling nChunks)
	uint64_t seq_len = 0;   // 每头真实 token 数(尾块支持;0 → group_count*tokens_per_group 无尾块)
	uint32_t has_tail_ = 0; // seq_len % tokens_per_group != 0;无尾块时三处钳制点零额外计算
	uint16_t scale16;
	float scale32;
	float gate_scale;
	float epsilon;   // q/k L2 归一化:rstd = 1/sqrt(sum(x^2) + epsilon)

	uint32_t coreid, subid;

	uint32_t gqkdb_cnt = 0, v_cnt = 0, out_cnt = 0;

	uint32_t arrive_cnt = 0;

	// 任务化:dense batch 使用单启动任务循环。
	// task_count = B*HV(拍平的任务数;0 = legacy 单任务,任务号即 coreid);
	// block_num = 实际 blockDim(任务步进);hv_real = 真实 HV(dt_bias/a_log
	// 按它寻址,所有 batch 共享;0 = 不区分,任务号即头号)。
	uint64_t task_count = 0, block_num = 0, hv_real = 0;
	// varlen 单启动:task = seq*HV + value_head;cu_seqlens 经 tiling
	// 落到 GM(attr 打包,免 H2D);逐任务解析 (token_begin, len)。
	// 任务空头 = len<=0:不进流水,仅在请求时写 final=initial(或 0)。
	uint64_t is_varlen = 0, nheads_qk = 0, total_tokens = 0;  // total_tokens = varlen 的 T_total(beta 头连续行距)
	uint64_t cu_base = 0;          // tiling GM 上 cu_seqlens 数组的裸地址
	uint64_t task_empty = 0;       // BindTask 产出:本任务是否空序列
	uint64_t task_tokens = 0;      // 本任务 token 起点(packed 空间)
	uint32_t qk_group = 1;
	// GM 基址(未按任务偏移),BindTask 每次重绑视图;uint64_t 裸地址
	//(GM_ADDR 带 __gm__ 地址空间限定,不能作类成员)
	uint64_t a_base, g_base, q_base, k_base, v_base, b_base, dt_base, tmp_base, o_base;
	uint64_t istate_base, fstate_base;

	__ubuf__ void *g_ub[2], *q_ub[2], *k_ub[2], *v_ub[4], *b_ub[4], *dt_ub, *tmp_ub[2], *strict_tril_mask, *identity;

	__aicore__ inline void InitState(__ubuf__ void *s_ub);
	// ===== initial_state / final_state 支持:GM fp32 ND ↔ UB bf16 NZ =====
	// s_ub(UB slot 23)每 AIV 持 S 的 k 半幅 64×128 bf16,NZ 16×16 分形,fv-major 连续:
	//   offset(bf16 元素) = fv*1024 + j*256 + k_in*16 + v_in
	//   v = fv*16 + v_in  (fv∈[0,8));  k = subid*64 + j*16 + k_in  (j∈[0,4))
	// 装载(FlashMLA 式):MTE2 一次连续 32KB(该半幅在 GM 天然连续),V 侧 cast +
	// DATA_BLOCK_COPY 节距 64 一次散布 8 块 32B 到 fv=0..7 各分形;写回:cast 回
	// fp32 NZ,MTE3 按 fv 合并成 8 次跨步散回 GM。
	// ND 行(128 fp32 连续)→ bf16 NZ 行:row r = (j=r/16, k_in=r%16):
	// DINTLV 偶奇载 + f2bfZero/f2bfOne + Or 得 128 bf16 线性序(与 CastState* 同习语),
	// 散布基址(元素) = r*16(= j*256 + k_in*16),块 p → fv=p(块距 1024 元素 = 64×32B)。
	__aicore__ inline void LoadStateNdScatter(__ubuf__ void *s_ub, __ubuf__ void *stage_f32);
	// s_ub bf16 NZ → stage fp32 ND:vsldb 块聚集载(与装载侧散布存同族镜像),
	// 行 r 从 r*16 聚 8 块 32B(块距 64×32B=2048B=fv 步长)
	// 得 128 bf16 线性序,cast 后 INTLV 存出 128 fp32 连续行。
	__aicore__ inline void StoreStateGatherCast(__ubuf__ void *stage_f32, __ubuf__ void *s_ub);
	// initial_state 装载:GM fp32 ND [64k,128v](本 AIV 半幅)→ s_ub bf16 NZ。
	// V_MTE2(0/1) 是输入区首个使用者的预支信号量:先消费再归还,借用期间
	// prologue 的装数自然被挡在墙外。
	__aicore__ inline void LoadInitState(__ubuf__ void *s_ub);
	// ===== 尾块(T%128≠0)变换后修正 =====
	// 无效行在装载阶段处理:
	//   raw g 尾行填 -inf → step_gate = gate_scale/(1+exp(a*(-inf+dt))) = -0,
	//   gamma(=Σgg)与 cumsum 天然零贡献,原 FixTailGamma 的"先累计无效贡献、
	//   再事后扣减"随之取消;raw q/k 尾行补零 + epsilon 归一化天然归零。
	// 此处仅剩 beta:beta_eff(fp32 [32],槽 ≥512B,64 lane 整寄存器读写安全):
	// 车道 >= valid 清零(sigmoid(0)=0.5/陈旧值)。
	// Arange+Compare+Select(GDN 运行时掩码习语),Select 是位选不传播 NaN。
	__aicore__ inline void FixTailBetaF32(__ubuf__ void *beta_ub, float valid_cnt);
	// final_state 写回:s_ub bf16 NZ → fp32 NZ 暂存(同借输入区)→ MTE3 散回 GM ND。
	// 只在末组状态更新全部落地后调用(Process 收尾);V 管同序保证读到的是终态。
	__aicore__ inline void StoreFinalState(__ubuf__ void *s_ub);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_gm_single(const AscendC::GlobalTensor<typename AscendC::Std::conditional<element_size == 4, float, T>::type> &mem,
		__ubuf__ void * ub);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void gm_ub_single(__ubuf__ void * ub,
		const AscendC::GlobalTensor<typename AscendC::Std::conditional<element_size == 4, float, T>::type> &mem);

	__aicore__ inline void Init(uint64_t N_, uint64_t scale_, uint64_t gate_scale_, GM_ADDR a_gm_, GM_ADDR g_gm_, GM_ADDR q_gm_,
		GM_ADDR k_gm_, GM_ADDR v_gm_, GM_ADDR b_gm_, GM_ADDR dt_gm_, GM_ADDR tmp_gm_, GM_ADDR o_gm_,
		GM_ADDR istate_gm_ = nullptr, GM_ADDR fstate_gm_ = nullptr,
		uint64_t has_istate_ = 0, uint64_t out_fstate_ = 0, uint64_t seq_len_ = 0,
		uint64_t qk_group_ = 1,
		uint64_t task_count_ = 0, uint64_t hv_real_ = 0, uint64_t block_num_ = 0,
		uint64_t is_varlen_ = 0, uint64_t cu_base_ = 0, uint64_t epsilon_bits_ = 0x3F50624Du,
		uint64_t gate_flags_ = 0);

	// 每任务绑定(Process 任务循环开头调用):按拍平任务号 fh 重绑全部 GM 视图、
	// 重载本头的 a/dt_bias、清零流水计数器。
	// fh = batch*HV + value_head;q/k 头 = fh/qk_group(= batch*H + 组内 qk 头);
	// dt_bias/a_log 只按真实 value head 寻址(所有 batch 共享)。
	// dt 装载走 MTE2 同管顺序:排在本任务首次 g/q/k 装载之前,由同一组
	// varlen:task = seq*HV + value_head。从 tiling GM 上的 cu_seqlens 解析本序列
	// [token_begin, token_end);len<=0 为空任务(task_empty=1,不进流水)。
	// 视图基址 = packed token-major [T_total, H*/HV, *] 按 (token_begin, head) 定位;
	// 行距由装载侧 stride 表达(q/k 行距 nheads_qk*key_dim,g 行距 hv_real*key_dim,
	// v/o 行距 hv_real*value_dim,beta 行距 hv_real)。
	__aicore__ inline void BindTaskVarlen(uint32_t fh);

	// 空序列收尾:不进计算流水。final_state 语义 = initial(或 0):
	//  - 有 initial_state:host(aclnn)已整体预填 fs=istate,kernel 不写;
	// 注:核内"读 istate 再写 fs"在空任务上下文实测不稳定(部分元素残留未写,
	// 因此 istate 直通放在 host 侧一次 ViewCopy。
	__aicore__ inline void WriteEmptyFinalState();
	template<bool varlen>
	__aicore__ inline void BindTask(uint32_t fh);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_single(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_src_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_dst_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1);

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_src_strided_dst_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1);

	template<uint32_t id, bool gate_f32, bool beta_f32, bool varlen, bool tail>
	__aicore__ inline void vec_prepare_g3(__ubuf__ void *gate_sc1_ub, __ubuf__ void *gamma_ub, __ubuf__ void *beta_ub, __ubuf__ void *qq2_ub, __ubuf__ void *kk2_ub);

	template<uint32_t id>
	__aicore__ inline void vec_prepare_qplus_kplus_kminus_kres(__ubuf__ void *qplus_ub, __ubuf__ void *qplusb_ub, __ubuf__ void *kplus_ub, __ubuf__ void *kplusb_ub,
		__ubuf__ void *kminus_ub, __ubuf__ void *kres_ub, __ubuf__ void *gate_sc_ub, __ubuf__ void *gamma_ub);

	__aicore__ inline void vec_tril(__ubuf__ void *masked_qk_ub, __ubuf__ void *l_ub, __ubuf__ void *kplus_kminus_ub, __ubuf__ void *qplus_kminus_ub, __ubuf__ void *beta_ub);

	__aicore__ inline void vec_tril_inv1(__ubuf__ void *m1_ub, __ubuf__ void *l_ub);

	__aicore__ inline void vec_tril_inv2(__ubuf__ void *m_ub, __ubuf__ void *m1_ub, __ubuf__ void *l_ub);

	template<uint16_t unroll_id, uint16_t iter_id, bool varlen, bool tail>
	__aicore__ inline void vec_v_corr(__ubuf__ void *v_corr_ub, __ubuf__ void *kplus_score_ub, __ubuf__ void *beta_ub);

	template<uint16_t part_id, uint64_t iter_id>
	__aicore__ inline void vec_iter_half(__ubuf__ void *s_ub, __ubuf__ void *score_delta_a_ub, __ubuf__ void *score_delta_b_ub, __ubuf__ void *gamma_ub);


public:

	// Define Buffer Addresses and Variables' Buffer IDs

	// addrUB 表项单位 = ub_addr_unit(512B)
	// fp32 gate 双槽重排:地址 0 的 16KB 留给 g_ub[1];主/auto kminus 顺序复用
	// id28 的前 8KB;id0 的两个 score buffer 顺序复用 319 的后 8KB,
	// kplus_score2 的 AIC 写入沿用 qplus_kminus 已有的 MTE3toFIX 释放保护。
	const uint32_t addrUB[31] = {319, 32, 49, 66, 68, 77, 86, 86, 87, 88, 104, 120, 136, 137, 139, 140, 156, 156, 165, 166, 170, 187, 204, 236, 268, 269, 286, 303, 335, 352};
	const uint32_t qhat1_ub = 2;
	const uint32_t qhat_ub = 2;
	const uint32_t khat1_ub = 9;
	const uint32_t khat_ub = 9;
	const uint32_t beta_a_ub = 24;
	const uint32_t beta_ub = 24;
	const uint32_t beta_b_ub = 12;
	const uint32_t gate_sc1_ub = 10;
	const uint32_t gate_pre_a_ub = 18;
	const uint32_t gate_pre_b_ub = 18;
	const uint32_t gate_sc_ub = 10;
	const uint32_t gate_pre_ub = 18;
	const uint32_t qplus_ub = 1;
	const uint32_t qplusb_ub = 4;
	const uint32_t kplus_ub = 25;
	const uint32_t kplusb_ub = 17;
	const uint32_t kminus_ub = 28;
	const uint32_t kres_ub = 22;
	const uint32_t gamma_ub = 13;
	const uint32_t kplus_kminus_ub = 27;
	const uint32_t qplus_kminus_ub = 25;
	const uint32_t l_ub = 22;
	const uint32_t masked_qk_ub = 19;
	const uint32_t m_ub = 21;
	const uint32_t m1_ub = 28;
	const uint32_t s1_partial_ub = 23;
	const uint32_t s1_ub = 23;
	const uint32_t ss1_ub = 23;
	const uint32_t kplus_score1_ub = 27;
	const uint32_t v_corr1_ub = 17;
	const uint32_t out1_ub = 11;
	const uint32_t score_delta_a_ub1 = 1;
	const uint32_t score_delta_a_ub2 = 26;
	const uint32_t score_delta_a_ub3 = 15;
	const uint32_t score_delta_a_ub4 = 29;
	const uint32_t s2_partial_ub = 23;
	const uint32_t s2_ub = 23;
	const uint32_t ss2_ub = 23;
	const uint32_t kplus_score2_ub = 26;
	const uint32_t v_corr2_ub = 4;
	const uint32_t out2_ub = 15;
	const uint32_t score_delta_b_ub1 = 27;
	const uint32_t score_delta_b_ub2 = 22;
	const uint32_t score_delta_b_ub3 = 26;
	const uint32_t score_delta_b_ub4 = 25;
	const uint32_t qhat1_ub_auto_unroll_id_1 = 11;
	const uint32_t qhat_ub_auto_unroll_id_1 = 11;
	const uint32_t khat1_ub_auto_unroll_id_1 = 29;
	const uint32_t khat_ub_auto_unroll_id_1 = 29;
	const uint32_t beta_a_ub_auto_unroll_id_1 = 8;
	const uint32_t beta_ub_auto_unroll_id_1 = 8;
	const uint32_t beta_b_ub_auto_unroll_id_1 = 14;
	const uint32_t gate_sc1_ub_auto_unroll_id_1 = 26;
	const uint32_t gate_pre_a_ub_auto_unroll_id_1 = 7;
	const uint32_t gate_pre_b_ub_auto_unroll_id_1 = 7;
	const uint32_t gate_sc_ub_auto_unroll_id_1 = 26;
	const uint32_t gate_pre_ub_auto_unroll_id_1 = 7;
	const uint32_t qplus_ub_auto_unroll_id_1 = 21;
	const uint32_t qplusb_ub_auto_unroll_id_1 = 5;
	const uint32_t kplus_ub_auto_unroll_id_1 = 20;
	const uint32_t kplusb_ub_auto_unroll_id_1 = 4;
	const uint32_t kminus_ub_auto_unroll_id_1 = 28;
	const uint32_t kres_ub_auto_unroll_id_1 = 25;
	const uint32_t gamma_ub_auto_unroll_id_1 = 3;
	const uint32_t kplus_kminus_ub_auto_unroll_id_1 = 1;
	const uint32_t qplus_kminus_ub_auto_unroll_id_1 = 0;
	const uint32_t l_ub_auto_unroll_id_1 = 5;
	const uint32_t masked_qk_ub_auto_unroll_id_1 = 20;
	const uint32_t m_ub_auto_unroll_id_1 = 19;
	const uint32_t m1_ub_auto_unroll_id_1 = 17;
	const uint32_t s1_partial_ub_auto_unroll_id_1 = 23;
	const uint32_t s1_ub_auto_unroll_id_1 = 23;
	const uint32_t ss1_ub_auto_unroll_id_1 = 23;
	const uint32_t kplus_score1_ub_auto_unroll_id_1 = 26;
	const uint32_t v_corr1_ub_auto_unroll_id_1 = 27;
	const uint32_t out1_ub_auto_unroll_id_1 = 15;
	const uint32_t score_delta_a_ub1_auto_unroll_id_1 = 27;
	const uint32_t score_delta_a_ub2_auto_unroll_id_1 = 21;
	const uint32_t score_delta_a_ub3_auto_unroll_id_1 = 20;
	const uint32_t score_delta_a_ub4_auto_unroll_id_1 = 10;
	const uint32_t s2_partial_ub_auto_unroll_id_1 = 23;
	const uint32_t s2_ub_auto_unroll_id_1 = 23;
	const uint32_t ss2_ub_auto_unroll_id_1 = 23;
	const uint32_t kplus_score2_ub_auto_unroll_id_1 = 0;
	const uint32_t v_corr2_ub_auto_unroll_id_1 = 25;
	const uint32_t out2_ub_auto_unroll_id_1 = 2;
	const uint32_t score_delta_b_ub1_auto_unroll_id_1 = 2;
	const uint32_t score_delta_b_ub2_auto_unroll_id_1 = 9;
	const uint32_t score_delta_b_ub3_auto_unroll_id_1 = 27;
	const uint32_t score_delta_b_ub4_auto_unroll_id_1 = 20;

	// addrL1 表项单位 = l1_addr_unit(1KB)
	const uint32_t addrL1[31] = {0, 8, 16, 48, 64, 80, 81, 81, 82, 84, 92, 124, 156, 172, 180, 184, 216, 224, 256, 288, 320, 352, 368, 384, 400, 416, 424, 456, 472, 504};
	const uint32_t beta = 26;
	const uint32_t qplus = 23;
	const uint32_t qplusb = 29;
	const uint32_t kplus = 24;
	const uint32_t kplusb = 28;
	const uint32_t kminus = 22;
	const uint32_t kres = 19;
	const uint32_t gamma = 25;
	const uint32_t masked_qk = 27;
	const uint32_t mm = 14;
	const uint32_t mmt = 14;
	const uint32_t m = 14;
	const uint32_t mm1 = 20;
	const uint32_t s1 = 15;
	const uint32_t v_corr1 = 16;
	const uint32_t y1 = 18;
	const uint32_t s2 = 28;
	const uint32_t v_corr2 = 20;
	const uint32_t y2 = 17;
	const uint32_t beta_auto_unroll_id_1 = 10;
	const uint32_t qd_auto_unroll_id_1 = 3;
	const uint32_t qd2_auto_unroll_id_1 = 13;
	const uint32_t kd_auto_unroll_id_1 = 12;
	const uint32_t kd2_auto_unroll_id_1 = 9;
	const uint32_t ki_auto_unroll_id_1 = 4;
	const uint32_t kr_auto_unroll_id_1 = 21;
	const uint32_t gamma_auto_unroll_id_1 = 8;
	const uint32_t masked_qk_auto_unroll_id_1 = 16;
	const uint32_t mm_auto_unroll_id_1 = 2;
	const uint32_t mmt_auto_unroll_id_1 = 2;
	const uint32_t m_auto_unroll_id_1 = 2;
	const uint32_t mm1_auto_unroll_id_1 = 7;
	const uint32_t s1_auto_unroll_id_1 = 11;
	const uint32_t v_corr1_auto_unroll_id_1 = 0;
	const uint32_t y1_auto_unroll_id_1 = 18;
	const uint32_t s2_auto_unroll_id_1 = 17;
	const uint32_t v_corr2_auto_unroll_id_1 = 1;
	const uint32_t y2_auto_unroll_id_1 = 0;

	const uint32_t addrL0A[9] = {0, 24, 24, 48, 48, 58, 59, 63};
	const uint32_t kplus_kminus_a = 0;
	const uint32_t qplus_kminus_a = 2;
	const uint32_t mm1_a = 5;
	const uint32_t mm2_a = 2;
	const uint32_t kplus_score_qplus_score1_a = 0;
	const uint32_t y1_a = 2;
	const uint32_t iter_out1_a = 2;
	const uint32_t kplus_score_qplus_score2_a = 0;
	const uint32_t y2_a = 0;
	const uint32_t iter_out2_a = 2;
	const uint32_t kplus_kminus_a_auto_unroll_id_1 = 0;
	const uint32_t qplus_kminus_a_auto_unroll_id_1 = 2;
	const uint32_t mm1_a_auto_unroll_id_1 = 7;
	const uint32_t mm2_a_auto_unroll_id_1 = 2;
	const uint32_t kplus_score_qplus_score1_a_auto_unroll_id_1 = 0;
	const uint32_t y1_a_auto_unroll_id_1 = 2;
	const uint32_t iter_out1_a_auto_unroll_id_1 = 2;
	const uint32_t kplus_score_qplus_score2_a_auto_unroll_id_1 = 0;
	const uint32_t y2_a_auto_unroll_id_1 = 2;
	const uint32_t iter_out2_a_auto_unroll_id_1 = 0;

	const uint32_t addrL0B[9] = {0, 32, 32, 48, 48, 48, 49, 49};
	const uint32_t kplus_kminus_b = 0;
	const uint32_t qplus_kminus_b = 2;
	const uint32_t mm1_b = 5;
	const uint32_t mm2_b = 2;
	const uint32_t kplus_score_qplus_score1_b = 0;
	const uint32_t y1_b = 2;
	const uint32_t iter_out1_b = 2;
	const uint32_t kplus_score_qplus_score2_b = 0;
	const uint32_t y2_b = 0;
	const uint32_t iter_out2_b = 2;
	const uint32_t kplus_kminus_b_auto_unroll_id_1 = 0;
	const uint32_t qplus_kminus_b_auto_unroll_id_1 = 2;
	const uint32_t mm1_b_auto_unroll_id_1 = 7;
	const uint32_t mm2_b_auto_unroll_id_1 = 2;
	const uint32_t kplus_score_qplus_score1_b_auto_unroll_id_1 = 0;
	const uint32_t y1_b_auto_unroll_id_1 = 2;
	const uint32_t iter_out1_b_auto_unroll_id_1 = 2;
	const uint32_t kplus_score_qplus_score2_b_auto_unroll_id_1 = 0;
	const uint32_t y2_b_auto_unroll_id_1 = 2;
	const uint32_t iter_out2_b_auto_unroll_id_1 = 0;

	const uint32_t addrL0C[9] = {0, 16, 32, 48, 64, 80, 96, 112};
	const uint32_t kplus_kminus_c = 5;
	const uint32_t qplus_kminus_c = 2;
	const uint32_t mm1_c = 2;
	const uint32_t mm2_c = 0;
	const uint32_t kplus_score1_c = 5;
	const uint32_t qplus_score1_c = 1;
	const uint32_t y1_c = 2;
	const uint32_t out1_c = 1;
	const uint32_t score_delta_a_c1 = 5;
	const uint32_t score_delta_a_c2 = 6;
	const uint32_t score_delta_a_c3 = 4;
	const uint32_t score_delta_a_c4 = 7;
	const uint32_t kplus_score2_c = 3;
	const uint32_t qplus_score2_c = 1;
	const uint32_t y2_c = 3;
	const uint32_t out2_c = 1;
	const uint32_t score_delta_b_c1 = 6;
	const uint32_t score_delta_b_c2 = 7;
	const uint32_t score_delta_b_c3 = 0;
	const uint32_t score_delta_b_c4 = 4;
	const uint32_t kplus_kminus_c_auto_unroll_id_1 = 4;
	const uint32_t qplus_kminus_c_auto_unroll_id_1 = 7;
	const uint32_t mm1_c_auto_unroll_id_1 = 7;
	const uint32_t mm2_c_auto_unroll_id_1 = 7;
	const uint32_t kplus_score1_c_auto_unroll_id_1 = 1;
	const uint32_t qplus_score1_c_auto_unroll_id_1 = 0;
	const uint32_t y1_c_auto_unroll_id_1 = 4;
	const uint32_t out1_c_auto_unroll_id_1 = 0;
	const uint32_t score_delta_a_c1_auto_unroll_id_1 = 3;
	const uint32_t score_delta_a_c2_auto_unroll_id_1 = 1;
	const uint32_t score_delta_a_c3_auto_unroll_id_1 = 6;
	const uint32_t score_delta_a_c4_auto_unroll_id_1 = 7;
	const uint32_t kplus_score2_c_auto_unroll_id_1 = 7;
	const uint32_t qplus_score2_c_auto_unroll_id_1 = 6;
	const uint32_t y2_c_auto_unroll_id_1 = 1;
	const uint32_t out2_c_auto_unroll_id_1 = 6;
	const uint32_t score_delta_b_c1_auto_unroll_id_1 = 5;
	const uint32_t score_delta_b_c2_auto_unroll_id_1 = 2;
	const uint32_t score_delta_b_c3_auto_unroll_id_1 = 0;
	const uint32_t score_delta_b_c4_auto_unroll_id_1 = 7;

	// Define operator functions

	template<int U>  // 0=主流水槽位,1=auto_unroll 流水第二槽位
	__aicore__ inline void vec_prepare_g1(int loop_id, uint32_t gate_sc1_ub, uint32_t gate_pre_a_ub, uint32_t beta_a_ub, uint32_t qhat1_ub, uint32_t khat1_ub);

	template<int U>  // 0=主流水槽位,1=auto_unroll 流水第二槽位
	__aicore__ inline void vec_prepare_qplus_kplus_kminus_kres(int loop_id, uint32_t qplus_ub, uint32_t kplus_ub, uint32_t kplusb_ub, uint32_t qplusb_ub, uint32_t kminus_ub, uint32_t kres_ub, uint32_t gate_pre_ub, uint32_t qhat_ub, uint32_t khat_ub, uint32_t gate_sc_ub, uint32_t gate_pre_b_ub);
	__aicore__ inline void vec_tril(int loop_id, uint32_t masked_qk_ub, uint32_t l_ub, uint32_t kplus_kminus_ub, uint32_t qplus_kminus_ub, uint32_t beta_b_ub);

	__aicore__ inline void vec_tril_inv(int loop_id, uint32_t m_ub, uint32_t l_ub, uint32_t m1_ub);
	template<int U>  // 流水槽位组
	__aicore__ inline void vec_v_corr1(int loop_id, uint32_t v_corr1_ub, uint32_t kplus_score1_ub, uint32_t beta_b_ub);
	__aicore__ inline void vec_iter_partial1(int loop_id, uint32_t s2_partial_ub, uint32_t ss1_ub, uint32_t score_delta_a_ub1, uint32_t score_delta_a_ub2, uint32_t gamma_ub);
	__aicore__ inline void vec_iter_final1(int loop_id, uint32_t s2_ub, uint32_t s2_partial_ub, uint32_t score_delta_a_ub3, uint32_t score_delta_a_ub4, uint32_t gamma_ub);
	template<int U>  // 流水槽位组
	__aicore__ inline void vec_v_corr2(int loop_id, uint32_t v_corr2_ub, uint32_t kplus_score2_ub, uint32_t beta_b_ub);
	__aicore__ inline void vec_iter_partial2(int loop_id, uint32_t s1_partial_ub, uint32_t ss2_ub, uint32_t score_delta_b_ub1, uint32_t score_delta_b_ub2, uint32_t gamma_ub);
	__aicore__ inline void vec_iter_final2(int loop_id, uint32_t s1_ub_auto_unroll_id_1, uint32_t s1_partial_ub, uint32_t score_delta_b_ub3, uint32_t score_delta_b_ub4, uint32_t gamma_ub);












	__aicore__ inline void ub_to_l1_qplus_kplus(int loop_id, uint32_t qplus, uint32_t kplus, uint32_t kplusb, uint32_t qplusb, uint32_t qplus_ub, uint32_t kplus_ub, uint32_t kplusb_ub, uint32_t qplusb_ub);
	__aicore__ inline void ub_to_l1_kminus_kres(int loop_id, uint32_t kminus, uint32_t kres, uint32_t gamma, uint32_t beta, uint32_t kminus_ub, uint32_t kres_ub, uint32_t gate_pre_ub, uint32_t beta_ub);
	__aicore__ inline void ub_to_l1_mm(int loop_id, uint32_t mm, uint32_t m_ub);
	__aicore__ inline void ub_to_l1_masked_qk(int loop_id, uint32_t masked_qk, uint32_t masked_qk_ub);
	__aicore__ inline void ub_to_l1_s1(int loop_id, uint32_t s1, uint32_t ss1_ub, uint32_t s1_ub);
	__aicore__ inline void ub_to_l1_v_corr1(int loop_id, uint32_t v_corr1, uint32_t v_corr1_ub);
	__aicore__ inline void ub_to_gm_out1(int loop_id, uint32_t out1_ub);















	template <bool STEADY>
	__aicore__ inline void stage_prepare_g1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_qplus_kplus_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kminus_kres_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_final1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_s1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_tril_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr2_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_masked_qk_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_out1_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_tril_inv_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial2_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_final2_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_mm_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_prepare_qplus_kplus_kminus_kres_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_s1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_out1_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_qplus_kplus_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_kminus_kres_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_final1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_s1_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_tril_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_masked_qk_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_out1_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_tril_inv_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial2_u1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_iter_final2_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_mm_u0(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_s1_u0_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_out1_u1_x1(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0_x2(int iter, int group_count);

	template <bool STEADY>
	__aicore__ inline void stage_finish_pipeline(int iter, int group_count);

	// 实际执行集(数值验证 group_count=1..8 + 语句流模拟确认与三段逐条一致)。

	template <bool STEADY>
	__aicore__ inline void RunPipelineStages(int iter, int groupCount);

	// Define Kernel Process
	__aicore__ inline void Process(int dense_group_count);

#endif
};

#if defined(__DAV_C310_CUBE__)
#include "chunk_kda_fwd_32_matmul.h"
#elif defined(__DAV_C310_VEC__)
#include "chunk_kda_fwd_32_vf.h"
#endif

#if defined(__DAV_C310_CUBE__)
extern "C" __global__ __aicore__ void kda32_mix_aic(uint64_t group_count, uint64_t scale, uint64_t gate_scale, __gm__ void *a,
	__gm__ void *q, __gm__ void *k, __gm__ void *v, __gm__ void *g, __gm__ void *beta_slot, __gm__ void *dt,
	__gm__ void *o, __gm__ void *tmp)
{
	Kernel<bfloat16_t> kda;   // 遗留测试入口,保持 bf16
	kda.tmp_gm = tmp;
	kda.Init();
	kda.Process(group_count);
}
#elif defined(__DAV_C310_VEC__)
extern "C" __global__ __aicore__ void kda32_mix_aiv(uint64_t group_count, uint64_t scale, uint64_t gate_scale, __gm__ void *a,
	__gm__ void *q, __gm__ void *k, __gm__ void *v, __gm__ void *g, __gm__ void *beta, __gm__ void *dt,
	__gm__ void *o, __gm__ void *tmp)
{
	Kernel<bfloat16_t, false, true> kda;   // 遗留测试入口,保持 bf16 dense + 尾块支持
	//kda.Init(group_count, scale, q, k, q, v, g, beta, o, tmp);
	kda.Init(group_count, scale, gate_scale, (GM_ADDR)a, (GM_ADDR)g, (GM_ADDR)q, (GM_ADDR)k, (GM_ADDR)v, (GM_ADDR)beta, (GM_ADDR)dt, (GM_ADDR)tmp, (GM_ADDR)o);
	kda.Process(group_count);
}
#endif

namespace KdaForward {
namespace Chunk32 {

__aicore__ inline void Run(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cuSeqlens; (void)chunkIndices; (void)gk; (void)aqk; (void)akk;
    (void)w; (void)u; (void)qg; (void)kg; (void)vNew; (void)h;
    (void)qgScaled; (void)uSeed;
    const __gm__ KdaFd32TilingData *td = reinterpret_cast<const __gm__ KdaFd32TilingData *>(tiling);
#if defined(__DAV_C310_CUBE__)
    (void)q; (void)k; (void)v; (void)g; (void)beta; (void)aLog; (void)dtBias;
    (void)attnOut; (void)initialState; (void)finalState;
    Kernel<bfloat16_t> kda;
    kda.tmp_gm = workspace;
    kda.task_count = td->taskCount;
    kda.block_num = td->blockNum;
    kda.is_varlen = td->varlenMode;
    kda.hv_real = td->hvNumReal;
    kda.cu_base = (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData));
    kda.Init();
    kda.Process((int)td->nChunks);
#elif defined(__DAV_C310_VEC__)
    if (td->varlenMode == 0 && td->seqLen % tokens_per_group == 0) {
        Kernel<bfloat16_t, false, false> kda;
        kda.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else if (td->varlenMode == 0) {
        Kernel<bfloat16_t, false, true> kda;
        kda.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else {
        Kernel<bfloat16_t, true, true> kda;
        kda.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    }
#endif
}

} // namespace Chunk32
} // namespace KdaForward
