
// Implement Init for set/wait flag mechanism in Cube
// Implement Cube-side set/wait_intra_block with the "pseudo-set method"

#include "kernel_operator.h"  // SetFlag/WaitFlag/CrossCoreSetFlag/CrossCoreWaitFlag(AscendC 基础 API)





#ifdef __DAV_C310_CUBE__



// KDA 形状常量(与 aiv 侧一致;数值相同的轴按语义分开命名)
// 槽位表(addrL0A/L0B/L0C/L1/addrUB)的地址单位:表项值 x 单位 = 字节偏移
constexpr uint32_t l0_addr_unit = 1024;  // addrL0A/L0B/L0C 表项单位(字节)

// 本地内存裸地址视图。tileSize 是从 byteOff 到窗口末尾的元素数，
// 与原 uint8_t buffer 先偏移、再 ReinterpretCast 的地址和长度语义一致。
constexpr uint32_t l0a_window_bytes = 64 * 1024;
constexpr uint32_t l0b_window_bytes = 64 * 1024;
constexpr uint32_t l0c_window_bytes = 256 * 1024;
constexpr uint32_t ub_window_bytes = 248 * 1024;

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> L0a(uint32_t byteOff)
{
	return AscendC::LocalTensor<U>(AscendC::TPosition::A2, byteOff, (l0a_window_bytes - byteOff) / sizeof(U));
}

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> L0b(uint32_t byteOff)
{
	return AscendC::LocalTensor<U>(AscendC::TPosition::B2, byteOff, (l0b_window_bytes - byteOff) / sizeof(U));
}

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> L0c(uint32_t byteOff)
{
	return AscendC::LocalTensor<U>(AscendC::TPosition::CO1, byteOff, (l0c_window_bytes - byteOff) / sizeof(U));
}

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> L1b(uint32_t byteOff)
{
	return AscendC::LocalTensor<U>(AscendC::TPosition::A1, byteOff, (l1_window_bytes - byteOff) / sizeof(U));
}

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> Ub(uint32_t byteOff)
{
	return AscendC::LocalTensor<U>(AscendC::TPosition::VECCALC, byteOff, (ub_window_bytes - byteOff) / sizeof(U));
}


// T 使用 bfloat16_t,fixpipe 出数量化位、mmad、L0/L1 视图均跟随该类型。

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::Init()
	{
		// Init Fixpipe
		//SPR.FPC, can be configured with [void set_fpc(uint64_t config)]
		//[7:0], [15:8], [23:16], [31:24], [39:32] - all kinds of quant parameter addresses, not used here.
		//[62:40], reserved. [63], the L0C unit flag clean bit for unit-flag mechanism. Not used here.

		//SPR.QUANT_PRE for fixpipe pre-stage quantization: [void set_quant_pre(uint64_t config)]
		uint64_t quant_pre = 0;
		// only set [31:13] fot M1: use TF32, 1 sign bit, 8 exponent bits, 10 mantissa bits
		quant_pre += ( ( ((uint64_t)0<<(uint64_t)18) + ((uint64_t)127<<(uint64_t)10) + (uint64_t)0) << (uint64_t)13); // if use 1.0 as M1
		//quant_pre += ss;
		set_quant_pre(quant_pre);

		//SPR.RELU_ALPHA, SPR.QUANT_POST, SPR.FIX_CLIP_RELU, obviously not required to config here.

		//SPR.LOOP3_PARA is valid when NZ2ND_EN or NZ2DN_EN or LoopEnhance_EN is enabled. Should be configured here.
		//[15:0], loop3_size (must be larger than 0), set to 1
		uint64_t loop3_para = 1;
		//[31:16], src loop3 stride in unit of C0_SIZE. Since Loop3_size is 1, this is not actually valid here, so set to 0
		//[63:32], dst loop3 stride in unit of element. Since Loop3_size is 1, this is not actually valid here, so set to 0
		set_loop3_para(loop3_para);

		//SPR.LOOP4_PARA is only valid when LoopEnhance mode is enabled, obviously not required to config here.
		//SPR.CHANNEL_PARA is only valid when LoopEnhance mode or NZ2DN mode is enabled. Obviously not required here.
		//SPR.ELT_ANTIQ_PARA, SPR.ELT_SRC_PARA, obviously not reuqired to config here.

		//SPR.CTRL
		//[48]: Saturation Ctrl bit for BF16/FP16 compute in Fixpipe. 0 will turn INF to MAX, and NaN to 0.
		//[49]: FP8 NaN preservation mode. 0 will turn NaN to 0.
		// We set this register in Init().

		// Init Mmad Config Reg
		uint64_t ctrl = get_ctrl();
		//[45]: HiF8 ctrl bit, set to 0;
		ctrl = ctrl & (~((uint64_t)1 << 45));
		//[48]: Saturation Ctrl, Set to 1 to saturate;
		ctrl = ctrl | ((uint64_t)1 << 48);
		//[51]: mmad direction ctrl (M or N), set to 1 for N first
		ctrl = ctrl | ((uint64_t)1 << 51);
		//[49]: FP8 NaN preservation mode. 0 will turn NaN to 0. Either is OK, so not set here.
		set_ctrl(ctrl);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0a(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0a) {
		const AscendC::LocalTensor<T> &dst = l0a;
		const AscendC::LocalTensor<T> &src = l1;
		AscendC::LoadData2DParamsV2 p;
		p.mStartPosition = 0; p.kStartPosition = 0;
		p.mStep = (m_slot / fractal_len);                      // 原 cfg0[39:32],单位 16 元素
		p.kStep = (n / fractal_len);                      // 原 cfg0[47:40],单位 32B
		p.srcStride = ((m_slot + m_stride) / fractal_len);     // 原 cfg1[15:0],单位分形
		p.dstStride = (m_slot / fractal_len);                  // 原 cfg1[31:16]
		p.ifTranspose = false; p.sid = 0;
		AscendC::LoadData(dst, src, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0b(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0b) {
		const AscendC::LocalTensor<T> &dst = l0b;
		const AscendC::LocalTensor<T> &src = l1;
		AscendC::LoadData2DParamsV2 p;
		p.mStartPosition = 0; p.kStartPosition = 0;
		p.mStep = (m_slot / fractal_len);
		p.kStep = (n / fractal_len);
		p.srcStride = ((m_slot + m_stride) / fractal_len);
		p.dstStride = (m_slot / fractal_len);
		p.ifTranspose = false; p.sid = 0;
		AscendC::LoadData(dst, src, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0a_transpose(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0a) {
		const AscendC::LocalTensor<T> &dst = l0a;
		const AscendC::LocalTensor<T> &src = l1;
		AscendC::LoadData2DParamsV2 p;
		p.mStartPosition = 0; p.kStartPosition = 0;
		p.mStep = (m_slot / fractal_len);                      // K/16
		p.kStep = (n / fractal_len);                      // M/16
		p.srcStride = ((m_slot + m_stride) / fractal_len);     // L1 K 方向总长/16
		p.dstStride = (n / fractal_len);                  // M/16(转置轴语义,对齐官方 matmul 用法)
		p.ifTranspose = true; p.sid = 0;
		AscendC::LoadData(dst, src, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t n, uint64_t m_stride>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0b_transpose(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<T> &l0b) {
		const AscendC::LocalTensor<T> &dst = l0b;
		const AscendC::LocalTensor<T> &src = l1;
		AscendC::LoadData2DParamsV2 p;
		p.mStartPosition = 0; p.kStartPosition = 0;
		p.mStep = (m_slot / fractal_len);                      // K/16
		p.kStep = (n / fractal_len);                      // N/16
		p.srcStride = ((m_slot + m_stride) / fractal_len);     // L1 K 方向总长/16
		p.dstStride = (n / fractal_len);                  // N/16
		p.ifTranspose = true; p.sid = 0;
		AscendC::LoadData(dst, src, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t n>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_l1_bf16(const AscendC::LocalTensor<T> &l1, const AscendC::LocalTensor<float> &l0c) {
		AscendC::DataCopyCO12DstParams p;
		p.nSize = n;
		p.mSize = m_slot;
		p.dstStride = m_slot;
		p.srcStride = m_slot;
		p.quantPre = QuantMode_t::F322BF16;
		p.nz2ndEn = false;
		AscendC::DataCopy(l1, l0c, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<bool nz2nd, uint64_t dst_id, uint64_t m_slot, uint64_t n>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_ub_bf16(const AscendC::LocalTensor<T> &ub, const AscendC::LocalTensor<float> &l0c) {
		constexpr AscendC::CO2Layout fmt = nz2nd ? AscendC::CO2Layout::ROW_MAJOR : AscendC::CO2Layout::NZ;
		AscendC::FixpipeParamsArch3510<fmt> p;
		p.nSize = n;
		p.mSize = (m_slot >> 1);
		p.srcStride = m_slot;
		p.dstStride = nz2nd ? n : (m_slot << 3);
		p.quantPre = QuantMode_t::F322BF16;
		p.dualDstCtl = 0;
		p.subBlockId = (dst_id != 0);
		if constexpr (nz2nd) AscendC::Fixpipe<T, float, KDA_CFG_UB_ND>(ub, l0c, p);
		else AscendC::Fixpipe<T, float, KDA_CFG_UB_NZ>(ub, l0c, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<bool nz2nd, uint64_t m_slot, uint64_t n>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_ub_f32(const AscendC::LocalTensor<float> &ub, const AscendC::LocalTensor<float> &l0c) {
		constexpr AscendC::CO2Layout fmt = nz2nd ? AscendC::CO2Layout::ROW_MAJOR : AscendC::CO2Layout::NZ;
		AscendC::FixpipeParamsArch3510<fmt> p;
		p.nSize = n;
		p.mSize = m_slot;
		p.srcStride = m_slot;
		p.dstStride = nz2nd ? n : (m_slot << 3);
		p.quantPre = QuantMode_t::NoQuant;
		p.dualDstCtl = 1;
		p.subBlockId = false;
		if constexpr (nz2nd) AscendC::Fixpipe<float, float, KDA_CFG_UB_ND>(ub, l0c, p);
		else AscendC::Fixpipe<float, float, KDA_CFG_UB_NZ>(ub, l0c, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t vid, uint64_t len, uint64_t element_size>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_ub(const AscendC::LocalTensor<float> &ub, const AscendC::LocalTensor<float> &l1)
	{
		AscendC::DataCopyL1ToUB<float, vid>(ub, l1, len);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t k, uint64_t n>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::mmad(const AscendC::LocalTensor<float> &l0c, const AscendC::LocalTensor<T> &l0a, const AscendC::LocalTensor<T> &l0b)
	{
		const AscendC::LocalTensor<float> &c = l0c;
		const AscendC::LocalTensor<T> &a = l0a;
		const AscendC::LocalTensor<T> &b = l0b;
		AscendC::MmadParams p;
		p.m = m_slot; p.n = n; p.k = k;              // 注意:本模板参数序 <m_slot,k,n>,MmadParams 字段序 m/n/k
		p.unitFlag = 0;
		p.cmatrixSource = false;
		p.cmatrixInitVal = true;                // 原 cfg[63]=1 清C
		p.disableGemv = true;                   // 原 cfg[61]=1
		AscendC::Mmad(c, a, b, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
template<uint64_t m_slot, uint64_t k, uint64_t n>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::mmad_acc(const AscendC::LocalTensor<float> &l0c, const AscendC::LocalTensor<T> &l0a, const AscendC::LocalTensor<T> &l0b)
	{
		const AscendC::LocalTensor<float> &c = l0c;
		const AscendC::LocalTensor<T> &a = l0a;
		const AscendC::LocalTensor<T> &b = l0b;
		AscendC::MmadParams p;
		p.m = m_slot; p.n = n; p.k = k;
		p.unitFlag = 0;
		p.cmatrixSource = false;
		p.cmatrixInitVal = false;               // 原 cfg[63]=0 累加
		p.disableGemv = true;
		AscendC::Mmad(c, a, b, p);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::matmul_kplus_kminus(int loop_id, uint32_t kplus_kminus_c_slot, uint32_t kplus_kminus_a_slot, uint32_t kplus_kminus_b_slot) {
		AscendC::LocalTensor<T> kd_l0a_1 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> kd_l0a_2 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> kd2_l0a_1 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + 2*chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> kd2_l0a_2 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + 2*chunk_size*key_dim*sizeof(T) + chunk_half_rows*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki1_l0b_1 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> ki2_l0b_1 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_half_rows*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki1_l0b_2 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki2_l0b_2 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_half_rows*key_dim*sizeof(T) + chunk_size*key_dim*sizeof(T));

		AscendC::LocalTensor<float> kdki1_l0c_1 = L0c<float>(addrL0C[kplus_kminus_c_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> kdki1_l0c_2 = L0c<float>(addrL0C[kplus_kminus_c_slot]*l0_addr_unit + chunk_size*fractal_len*sizeof(float));
		AscendC::LocalTensor<float> kdki2_l0c_1 = L0c<float>(addrL0C[kplus_kminus_c_slot]*l0_addr_unit + state_k_half*fractal_len*sizeof(float) + fractal_elems*sizeof(float)*1);
		AscendC::LocalTensor<float> kdki2_l0c_2 = L0c<float>(addrL0C[kplus_kminus_c_slot]*l0_addr_unit + state_k_half*fractal_len*sizeof(float) + fractal_elems*sizeof(float)*3);

		this->template mmad<chunk_size,key_dim,chunk_half_rows>(kdki1_l0c_1, kd_l0a_1, ki1_l0b_1);
		this->template mmad<chunk_size,key_dim,chunk_half_rows>(kdki1_l0c_2, kd_l0a_2, ki1_l0b_2);
		this->template mmad<chunk_half_rows,key_dim,chunk_half_rows>(kdki2_l0c_1, kd2_l0a_1, ki2_l0b_1);
		this->template mmad<chunk_half_rows,key_dim,chunk_half_rows>(kdki2_l0c_2, kd2_l0a_2, ki2_l0b_2);

	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::matmul_kplus_qplus_score_a(int loop_id, uint32_t kds1_c_slot, uint32_t qds1_c_slot, uint32_t kplus_qplus_score_a_a_slot, uint32_t kplus_qplus_score_a_b_slot) {
		AscendC::LocalTensor<T> kd_l0a = L0a<T>(addrL0A[kplus_qplus_score_a_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> qd_l0a = L0a<T>(addrL0A[kplus_qplus_score_a_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> s_l0b = L0b<T>(addrL0B[kplus_qplus_score_a_b_slot]*l0_addr_unit);

		AscendC::LocalTensor<float> kds_l0c = L0c<float>(addrL0C[kds1_c_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> qds_l0c = L0c<float>(addrL0C[qds1_c_slot]*l0_addr_unit);


		this->template mmad<chunk_size,key_dim,value_dim>(kds_l0c, kd_l0a, s_l0b);
		this->template mmad<chunk_size,key_dim,value_dim>(qds_l0c, qd_l0a, s_l0b);

		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// arrive_cnt ++;
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::matmul_state_acc_a(int loop_id, uint32_t y1_c_slot, uint32_t y1_a_slot, uint32_t y1_b_slot) {
		AscendC::LocalTensor<T> m_l0a = L0a<T>(addrL0A[y1_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> v_kds_l0b = L0b<T>(addrL0B[y1_b_slot]*l0_addr_unit);

		AscendC::LocalTensor<float> y_l0c = L0c<float>(addrL0C[y1_c_slot]*l0_addr_unit);

		this->template mmad<chunk_size,chunk_size,value_dim>(y_l0c, m_l0a, v_kds_l0b);

		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// arrive_cnt ++;
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::matmul_iter_out_a(int loop_id, uint32_t out1_c_slot, uint32_t score_delta_a_c1_slot, uint32_t score_delta_a_c2_slot, uint32_t score_delta_a_c3_slot, uint32_t score_delta_a_c4_slot, uint32_t iter_out_a_a_slot, uint32_t iter_out_a_b_slot, uint32_t qds1_c_slot) {
		AscendC::LocalTensor<T> kr_l0a = L0a<T>(addrL0A[iter_out_a_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> mqk_l0a = L0a<T>(addrL0A[iter_out_a_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_1 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> y_l0b_2 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_3 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*2);
		AscendC::LocalTensor<T> y_l0b_4 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*3);

		AscendC::LocalTensor<float> s_delta_l0c_1 = L0c<float>(addrL0C[score_delta_a_c1_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> s_delta_l0c_2 = L0c<float>(addrL0C[score_delta_a_c2_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> s_delta_l0c_3 = L0c<float>(addrL0C[score_delta_a_c3_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> s_delta_l0c_4 = L0c<float>(addrL0C[score_delta_a_c4_slot]*l0_addr_unit);

		AscendC::LocalTensor<float> out_l0c_1 = L0c<float>(addrL0C[out1_c_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> out_l0c_2 = L0c<float>(addrL0C[out1_c_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(float));
		AscendC::LocalTensor<float> out_l0c_3 = L0c<float>(addrL0C[out1_c_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(float)*2);
		AscendC::LocalTensor<float> out_l0c_4 = L0c<float>(addrL0C[out1_c_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(float)*3);

		this->template mmad_acc<chunk_size,chunk_size,value_tile>(out_l0c_1, mqk_l0a, y_l0b_1);
		this->template mmad_acc<chunk_size,chunk_size,value_tile>(out_l0c_2, mqk_l0a, y_l0b_2);
		this->template mmad_acc<chunk_size,chunk_size,value_tile>(out_l0c_3, mqk_l0a, y_l0b_3);
		this->template mmad_acc<chunk_size,chunk_size,value_tile>(out_l0c_4, mqk_l0a, y_l0b_4);

		this->template mmad<key_dim,chunk_size,value_tile>(s_delta_l0c_1, kr_l0a, y_l0b_1);
		this->template mmad<key_dim,chunk_size,value_tile>(s_delta_l0c_2, kr_l0a, y_l0b_2);
		this->template mmad<key_dim,chunk_size,value_tile>(s_delta_l0c_3, kr_l0a, y_l0b_3);
		this->template mmad<key_dim,chunk_size,value_tile>(s_delta_l0c_4, kr_l0a, y_l0b_4);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_ub_gamma(int loop_id, uint32_t gamma_ub_slot, uint32_t betaB_ub_slot, uint32_t gamma_slot, uint32_t beta_slot) {
		AscendC::LocalTensor<float> gamma_ub = Ub<float>(addrUB[gamma_ub_slot]*ub_addr_unit);
		AscendC::LocalTensor<float> beta_ub = Ub<float>(addrUB[betaB_ub_slot]*ub_addr_unit);

		AscendC::LocalTensor<float> gamma = L1b<float>(addrL1[gamma_slot]*l1_addr_unit);
		AscendC::LocalTensor<float> beta = L1b<float>(addrL1[beta_slot]*l1_addr_unit);

		this->template copy_l1_to_ub<0, 2*key_dim, 4>(gamma_ub, gamma);
		this->template copy_l1_to_ub<0, 2*chunk_size, 4>(beta_ub, beta);
		this->template copy_l1_to_ub<1, 2*key_dim, 4>(gamma_ub, gamma);
		this->template copy_l1_to_ub<1, 2*chunk_size, 4>(beta_ub, beta);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_kplus_kminus(int loop_id, uint32_t kplus_kminus_a_slot, uint32_t kplus_kminus_b_slot, uint32_t kPlus_slot, uint32_t kPlusB_slot, uint32_t kMinus_slot) {
		AscendC::LocalTensor<T> kd_1 = L1b<T>(addrL1[kPlus_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> kd_2 = L1b<T>(addrL1[kPlus_slot]*l1_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> kd2_1 = L1b<T>(addrL1[kPlusB_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> kd2_2 = L1b<T>(addrL1[kPlusB_slot]*l1_addr_unit + chunk_half_rows*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki1_1 = L1b<T>(addrL1[kMinus_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> ki2_1 = L1b<T>(addrL1[kMinus_slot]*l1_addr_unit + fractal_elems*sizeof(T));
		AscendC::LocalTensor<T> ki1_2 = L1b<T>(addrL1[kMinus_slot]*l1_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki2_2 = L1b<T>(addrL1[kMinus_slot]*l1_addr_unit + fractal_elems*sizeof(T) + chunk_size*key_dim*sizeof(T));

		AscendC::LocalTensor<T> kd_l0a_1 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> kd_l0a_2 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> kd2_l0a_1 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + 2*chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> kd2_l0a_2 = L0a<T>(addrL0A[kplus_kminus_a_slot]*l0_addr_unit + 2*chunk_size*key_dim*sizeof(T) + chunk_half_rows*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki1_l0b_1 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> ki2_l0b_1 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_half_rows*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki1_l0b_2 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> ki2_l0b_2 = L0b<T>(addrL0B[kplus_kminus_b_slot]*l0_addr_unit + chunk_half_rows*key_dim*sizeof(T) + chunk_size*key_dim*sizeof(T));

		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(kd_1, kd_l0a_1);
		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(kd_2, kd_l0a_2);
		this->template copy_l1_to_l0a<chunk_half_rows,key_dim,0>(kd2_1, kd2_l0a_1);
		this->template copy_l1_to_l0a<chunk_half_rows,key_dim,0>(kd2_2, kd2_l0a_2);
		this->template copy_l1_to_l0b<chunk_half_rows,key_dim,chunk_half_rows>(ki1_1, ki1_l0b_1);
		this->template copy_l1_to_l0b<chunk_half_rows,key_dim,chunk_half_rows>(ki2_1, ki2_l0b_1);
		this->template copy_l1_to_l0b<chunk_half_rows,key_dim,chunk_half_rows>(ki1_2, ki1_l0b_2);
		this->template copy_l1_to_l0b<chunk_half_rows,key_dim,chunk_half_rows>(ki2_2, ki2_l0b_2);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_kplus_qplus_score_a(int loop_id, uint32_t kplus_qplus_score_a_a_slot, uint32_t kplus_qplus_score_a_b_slot, uint32_t kPlus_slot, uint32_t qPlus_slot, uint32_t scoreA_slot) {
		AscendC::LocalTensor<T> kd_1 = L1b<T>(addrL1[kPlus_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> qd_1 = L1b<T>(addrL1[qPlus_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> s_1 = L1b<T>(addrL1[scoreA_slot]*l1_addr_unit);

		AscendC::LocalTensor<T> kd_l0a = L0a<T>(addrL0A[kplus_qplus_score_a_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> qd_l0a = L0a<T>(addrL0A[kplus_qplus_score_a_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> s_l0b = L0b<T>(addrL0B[kplus_qplus_score_a_b_slot]*l0_addr_unit);


		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(kd_1, kd_l0a);
		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(qd_1, qd_l0a);
		this->template copy_l1_to_l0b_transpose<key_dim,value_dim,0>(s_1, s_l0b);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_y1(int loop_id, uint32_t y1_a_slot, uint32_t y1_b_slot, uint32_t m_slot, uint32_t vCorrA_slot) {
		AscendC::LocalTensor<T> m1 = L1b<T>(addrL1[m_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> v_corr_a = L1b<T>(addrL1[vCorrA_slot]*l1_addr_unit);

		AscendC::LocalTensor<T> m_l0a = L0a<T>(addrL0A[y1_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> v_kds_l0b = L0b<T>(addrL0B[y1_b_slot]*l0_addr_unit);

		this->template copy_l1_to_l0a<chunk_size,chunk_size,0>(m1, m_l0a);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_dim,0>(v_corr_a, v_kds_l0b);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_iter_out_a(int loop_id, uint32_t iter_out_a_a_slot, uint32_t iter_out_a_b_slot, uint32_t kRes_slot, uint32_t maskedQk_slot, uint32_t stateAccA_slot) {
		AscendC::LocalTensor<T> kr1 = L1b<T>(addrL1[kRes_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> mqk1 = L1b<T>(addrL1[maskedQk_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> y1_1 = L1b<T>(addrL1[stateAccA_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> y1_2 = L1b<T>(addrL1[stateAccA_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y1_3 = L1b<T>(addrL1[stateAccA_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*2);
		AscendC::LocalTensor<T> y1_4 = L1b<T>(addrL1[stateAccA_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*3);

		AscendC::LocalTensor<T> kr_l0a = L0a<T>(addrL0A[iter_out_a_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> mqk_l0a = L0a<T>(addrL0A[iter_out_a_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_1 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> y_l0b_2 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_3 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*2);
		AscendC::LocalTensor<T> y_l0b_4 = L0b<T>(addrL0B[iter_out_a_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*3);

		this->template copy_l1_to_l0a_transpose<chunk_size,key_dim,0>(kr1, kr_l0a);
		this->template copy_l1_to_l0a<chunk_size,chunk_size,0>(mqk1, mqk_l0a);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y1_1, y_l0b_1);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y1_2, y_l0b_2);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y1_3, y_l0b_3);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y1_4, y_l0b_4);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_kplus_qplus_score_b(int loop_id, uint32_t kplus_qplus_score_b_a_slot, uint32_t kplus_qplus_score_b_b_slot, uint32_t kPlus_slot, uint32_t qPlus_slot, uint32_t scoreB_slot) {
		AscendC::LocalTensor<T> kd_2 = L1b<T>(addrL1[kPlus_slot]*l1_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> qd_2 = L1b<T>(addrL1[qPlus_slot]*l1_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> s_2 = L1b<T>(addrL1[scoreB_slot]*l1_addr_unit);

		AscendC::LocalTensor<T> kd_l0a = L0a<T>(addrL0A[kplus_qplus_score_b_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> qd_l0a = L0a<T>(addrL0A[kplus_qplus_score_b_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> s_l0b = L0b<T>(addrL0B[kplus_qplus_score_b_b_slot]*l0_addr_unit);

		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(kd_2, kd_l0a);
		this->template copy_l1_to_l0a<chunk_size,key_dim,0>(qd_2, qd_l0a);
		this->template copy_l1_to_l0b_transpose<key_dim,value_dim,0>(s_2, s_l0b);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_y2(int loop_id, uint32_t y2_a_slot, uint32_t y2_b_slot, uint32_t m_slot, uint32_t vCorrB_slot) {
		AscendC::LocalTensor<T> m2 = L1b<T>(addrL1[m_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> v_corr_b = L1b<T>(addrL1[vCorrB_slot]*l1_addr_unit);

		AscendC::LocalTensor<T> m_l0a = L0a<T>(addrL0A[y2_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> v_kds_l0b = L0b<T>(addrL0B[y2_b_slot]*l0_addr_unit);

		this->template copy_l1_to_l0a<chunk_size,chunk_size,0>(m2, m_l0a);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_dim,0>(v_corr_b, v_kds_l0b);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l1_to_l0ab_iter_out_b(int loop_id, uint32_t iter_out_b_a_slot, uint32_t iter_out_b_b_slot, uint32_t kRes_slot, uint32_t maskedQk_slot, uint32_t stateAccB_slot) {
		AscendC::LocalTensor<T> kr2 = L1b<T>(addrL1[kRes_slot]*l1_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> mqk2 = L1b<T>(addrL1[maskedQk_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y2_1 = L1b<T>(addrL1[stateAccB_slot]*l1_addr_unit);
		AscendC::LocalTensor<T> y2_2 = L1b<T>(addrL1[stateAccB_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y2_3 = L1b<T>(addrL1[stateAccB_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*2);
		AscendC::LocalTensor<T> y2_4 = L1b<T>(addrL1[stateAccB_slot]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*3);

		AscendC::LocalTensor<T> kr_l0a = L0a<T>(addrL0A[iter_out_b_a_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> mqk_l0a = L0a<T>(addrL0A[iter_out_b_a_slot]*l0_addr_unit + chunk_size*key_dim*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_1 = L0b<T>(addrL0B[iter_out_b_b_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> y_l0b_2 = L0b<T>(addrL0B[iter_out_b_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T));
		AscendC::LocalTensor<T> y_l0b_3 = L0b<T>(addrL0B[iter_out_b_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*2);
		AscendC::LocalTensor<T> y_l0b_4 = L0b<T>(addrL0B[iter_out_b_b_slot]*l0_addr_unit + chunk_size*chunk_size*sizeof(T)*3);

		this->template copy_l1_to_l0a_transpose<chunk_size,key_dim,0>(kr2, kr_l0a);
		this->template copy_l1_to_l0a<chunk_size,chunk_size,0>(mqk2, mqk_l0a);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y2_1, y_l0b_1);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y2_2, y_l0b_2);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y2_3, y_l0b_3);
		this->template copy_l1_to_l0b_transpose<chunk_size,value_tile,0>(y2_4, y_l0b_4);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_ub_kplus_kminus(int loop_id, uint32_t kplus_kminus_ub_slot, uint32_t kplus_kminus_c_slot) {
		AscendC::LocalTensor<float> kplus_kminus_l0c = L0c<float>(addrL0C[kplus_kminus_c_slot]*l0_addr_unit);

		AscendC::LocalTensor<float> kplus_kminus_ub = Ub<float>(addrUB[kplus_kminus_ub_slot]*ub_addr_unit);

		this->template copy_l0c_to_ub_f32<false,2*chunk_size,chunk_size>(kplus_kminus_ub, kplus_kminus_l0c);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_ub_kds1(int loop_id, uint32_t kds1_ub_slot, uint32_t kds1_c_slot) {
		AscendC::LocalTensor<float> kds_l0c_1 = L0c<float>(addrL0C[kds1_c_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> kds_l0c_2 = L0c<float>(addrL0C[kds1_c_slot]*l0_addr_unit + fractal_elems*sizeof(float));
		AscendC::LocalTensor<T> kds_ub = Ub<T>(addrUB[kds1_ub_slot]*ub_addr_unit);

		this->template copy_l0c_to_ub_bf16<true,0,chunk_size,value_dim>(kds_ub, kds_l0c_1);
		this->template copy_l0c_to_ub_bf16<true,1,chunk_size,value_dim>(kds_ub, kds_l0c_2);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_l1_y1(int loop_id, uint32_t stateAccA_slot, uint32_t y1_c_slot) {
		AscendC::LocalTensor<float> y1_l0c = L0c<float>(addrL0C[y1_c_slot]*l0_addr_unit);
		AscendC::LocalTensor<T> y1_l1 = L1b<T>(addrL1[stateAccA_slot]*l1_addr_unit);

		this->template copy_l0c_to_l1_bf16<chunk_size,value_dim>(y1_l1, y1_l0c);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::copy_l0c_to_ub_1_iter1(int loop_id, uint32_t score_delta_a_ub1_slot, uint32_t score_delta_a_c1_slot) {
		AscendC::LocalTensor<float> s_delta_l0c_1 = L0c<float>(addrL0C[score_delta_a_c1_slot]*l0_addr_unit);
		AscendC::LocalTensor<float> s_delta_ub_1 = Ub<float>(addrUB[score_delta_a_ub1_slot]*ub_addr_unit);

		this->template copy_l0c_to_ub_f32<false,key_dim,value_tile>(s_delta_ub_1, s_delta_l0c_1);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_y1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(inv_aux2_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(inv_aux2_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(inv_aux2_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(inv_aux2_a_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(vCorrA_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(invAuxT_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(inv_aux2_c_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(m_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(inv_aux2_c_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qd2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kd2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qd_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kd_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(vCorrA_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(y1_a_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(m_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(vCorrA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_y1(iter - 1, y1_a_slot, y1_b_slot, m_slot, vCorrA_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_state_acc_a_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(y1_a_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(m_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(vCorrA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(mqk_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(y1_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(y1_c_slot);
		    matmul_state_acc_a(iter - 1, y1_c_slot, y1_a_slot, y1_b_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l1_y1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(y1_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(y1_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(y1_c_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(stateAccA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l0c_to_l1_y1(iter - 1, stateAccA_slot, y1_c_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_iter_out_a_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(y1_c_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(stateAccA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(maskedQk_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kRes_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(maskedQk_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(iter_out_a_a_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kRes_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(stateAccA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_iter_out_a(iter - 1, iter_out_a_a_slot, iter_out_a_b_slot, kRes_slot, maskedQk_slot, stateAccA_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_iter_out_a_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(kRes_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(iter_out_a_a_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(maskedQk_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(stateAccA_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(iter_out_a_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c2_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c4_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c3_slot);
		    matmul_iter_out_a(iter - 1, out1_c_slot, score_delta_a_c1_slot, score_delta_a_c2_slot, score_delta_a_c3_slot, score_delta_a_c4_slot, iter_out_a_a_slot, iter_out_a_b_slot, qds1_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c2_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c4_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(out1_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(iter_out_a_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c3_slot);
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub1_slot, score_delta_a_c1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub2_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c2_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub2_slot, score_delta_a_c2_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x2(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c2_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c4_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub4_slot, score_delta_a_c4_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x3(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(kds2_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_b_ub4_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c4_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(ki_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(beta_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(gamma_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kr_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kds2_ub_slot);
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c3_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub3_slot, score_delta_a_c3_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_kminus_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c3_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kd2_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(ki_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kd_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(ki_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_kminus_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kd2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_kplus_kminus(iter - 1, kplus_kminus_a_auto_unroll_id_1_slot, kplus_kminus_b_auto_unroll_id_1_slot, kd_auto_unroll_id_1_slot, kd2_auto_unroll_id_1_slot, ki_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(ki_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kd2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_kminus_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kd2_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(out1_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(out1_c_slot);
		    copy_l0c_to_ub_kds1(iter - 1, out1_ub_slot, out1_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_kminus_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out2_ub_slot);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out1_ub_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(out1_c_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(qd2_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(qd_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qplus_kminus_a_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(ki_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_kminus_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kplus_kminus_c_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(qd2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_kplus_kminus(iter - 1, qplus_kminus_a_auto_unroll_id_1_slot, qplus_kminus_b_auto_unroll_id_1_slot, qd_auto_unroll_id_1_slot, qd2_auto_unroll_id_1_slot, ki_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_kminus_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  matmul_kplus_kminus(iter - 1, kplus_kminus_c_auto_unroll_id_1_slot, kplus_kminus_a_auto_unroll_id_1_slot, kplus_kminus_b_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_kminus_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(qd2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(qplus_kminus_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_kminus_a_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kplus_kminus_c_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qd2_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(ki_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(ki_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(scoreB_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(scoreB_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qplus_kminus_c_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(scoreB_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(qplus_kminus_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(qPlus_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_b_a_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kPlus_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  matmul_kplus_kminus(iter - 1, qplus_kminus_c_auto_unroll_id_1_slot, qplus_kminus_a_auto_unroll_id_1_slot, qplus_kminus_b_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_qplus_score_b_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_kplus_qplus_score_b(iter - 1, kplus_qplus_score_b_a_slot, kplus_qplus_score_b_b_slot, kPlus_slot, qPlus_slot, scoreB_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kplus_kminus_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qplus_kminus_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qPlus_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_b_a_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kPlus_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(qplus_kminus_c_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(scoreB_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qPlus_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kPlus_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kPlusB_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(qplus_kminus_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kplus_kminus(iter - 1, qplus_kminus_ub_auto_unroll_id_1_slot, qplus_kminus_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kplus_kminus_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(qplus_kminus_c_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kplus_kminus_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kplus_kminus(iter - 1, kplus_kminus_ub_auto_unroll_id_1_slot, kplus_kminus_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_qplus_score_a_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kplus_kminus_c_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kds2_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qds2_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_b_a_slot);
		    matmul_kplus_qplus_score_a(iter - 1, kds2_c_slot, qds2_c_slot, kplus_qplus_score_b_a_slot, kplus_qplus_score_b_b_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_gamma_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_b_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kds2_c_slot);
		    syncCV.WaitIntraBlock<VtoMTE1_UBRelease>(beta2_ub_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(gamma_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(beta_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoMTE1_UBRelease>(gamma_ub_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(beta_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(gamma_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_ub_gamma(iter - 1, gamma_ub_auto_unroll_id_1_slot, beta2_ub_auto_unroll_id_1_slot, gamma_auto_unroll_id_1_slot, beta_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE1toV_UBReady>(gamma_ub_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(gamma_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE1toV_UBReady>(beta2_ub_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(gamma_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(beta_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(beta_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(kds2_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kds2_c_slot);
		    copy_l0c_to_ub_kds1(iter - 1, kds2_ub_slot, kds2_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_y2_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kds2_c_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kds2_ub_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(betaB_ub_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(mqk_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(vCorrB_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(vCorrB_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(m_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(y2_a_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(vCorrB_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_y2(iter - 1, y2_a_slot, y2_b_slot, m_slot, vCorrB_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_state_acc_a_u0_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(m_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(y2_a_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(mm_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(vCorrB_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(y2_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(y2_c_slot);
		    matmul_state_acc_a(iter - 1, y2_c_slot, y2_a_slot, y2_b_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l1_y1_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(y2_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(y2_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(y2_c_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(stateAccB_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l0c_to_l1_y1(iter - 1, stateAccB_slot, y2_c_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_iter_out_b_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(y2_c_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(stateAccB_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(stateAccB_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(iter_out_b_a_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kRes_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(maskedQk_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_iter_out_b(iter - 1, iter_out_b_a_slot, iter_out_b_b_slot, kRes_slot, maskedQk_slot, stateAccB_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_iter_out_a_u0_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(maskedQk_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(stateAccB_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(maskedQk_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kRes_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kRes_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(iter_out_b_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c2_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(iter_out_b_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c4_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c3_slot);
		    matmul_iter_out_a(iter - 1, out2_c_slot, score_delta_b_c1_slot, score_delta_b_c2_slot, score_delta_b_c3_slot, score_delta_b_c4_slot, iter_out_b_a_slot, iter_out_b_b_slot, qds2_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x4(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c2_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c3_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(iter_out_b_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(out2_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c4_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c2_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub2_slot, score_delta_b_c2_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x5(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c2_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub1_slot, score_delta_b_c1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x6(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c1_slot);
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_b_ub4_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c4_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub4_slot, score_delta_b_c4_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u0_x7(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c4_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c3_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub3_slot, score_delta_b_c3_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u0_x2(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c3_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(out2_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(out2_c_slot);
		    copy_l0c_to_ub_kds1(iter - 1, out2_ub_slot, out2_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_qplus_score_a_u1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(gamma_ub_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kds1_ub_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(out2_c_slot);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out2_ub_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(mm_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s1_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(s1_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_a_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(s1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_kplus_qplus_score_a(iter - 1, kplus_qplus_score_a_a_auto_unroll_id_1_slot, kplus_qplus_score_a_b_auto_unroll_id_1_slot, kd_auto_unroll_id_1_slot, qd_auto_unroll_id_1_slot, s1_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_qplus_score_a_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(out1_ub_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s1_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(s1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_a_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_a_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qds1_c_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kds1_c_auto_unroll_id_1_slot);
		    matmul_kplus_qplus_score_a(iter - 1, kds1_c_auto_unroll_id_1_slot, qds1_c_auto_unroll_id_1_slot, kplus_qplus_score_a_a_auto_unroll_id_1_slot, kplus_qplus_score_a_b_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_a_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kds1_c_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(kds1_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kds1_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kds1(iter - 1, kds1_ub_auto_unroll_id_1_slot, kds1_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_y1_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kds1_ub_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kds1_c_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(mm_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(inv_aux1_a_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(mm_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(mm_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(inv_aux1_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(inv_aux1_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(inv_aux1_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(inv_aux1_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(inv_aux1_c_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(inv_aux1_c_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(inv_aux1_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(inv_aux1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(inv_aux1_c_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(mm_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(inv_aux2_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(inv_aux1_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(inv_aux_t_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(inv_aux1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(inv_aux2_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(inv_aux2_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(inv_aux2_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(inv_aux2_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(inv_aux2_a_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr_a_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(inv_aux_t_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(inv_aux2_c_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(m_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(inv_aux2_c_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kPlusB_slot);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kPlus_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qPlusB_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qPlus_slot);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(v_corr_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(v_corr_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(y1_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(m_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_y1(iter - 1, y1_a_auto_unroll_id_1_slot, y1_b_auto_unroll_id_1_slot, m_auto_unroll_id_1_slot, v_corr_a_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_state_acc_a_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(y1_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(m_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(v_corr_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(y1_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(y1_c_auto_unroll_id_1_slot);
		    matmul_state_acc_a(iter - 1, y1_c_auto_unroll_id_1_slot, y1_a_auto_unroll_id_1_slot, y1_b_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l1_y1_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(y1_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(y1_c_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(y1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(y1_c_auto_unroll_id_1_slot);
		    copy_l0c_to_l1_y1(iter - 1, y1_auto_unroll_id_1_slot, y1_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_iter_out_a_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_FIX>(y1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(y1_c_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kr_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(mqk_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(y1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(iter_out_a_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kr_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(mqk_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_iter_out_a(iter - 1, iter_out_a_a_auto_unroll_id_1_slot, iter_out_a_b_auto_unroll_id_1_slot, kr_auto_unroll_id_1_slot, mqk_auto_unroll_id_1_slot, y1_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_iter_out_a_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(iter_out_a_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(mqk_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kr_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(y1_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c2_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c1_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c4_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(iter_out_a_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c3_auto_unroll_id_1_slot);
		    matmul_iter_out_a(iter - 1, out1_c_auto_unroll_id_1_slot, score_delta_a_c1_auto_unroll_id_1_slot, score_delta_a_c2_auto_unroll_id_1_slot, score_delta_a_c3_auto_unroll_id_1_slot, score_delta_a_c4_auto_unroll_id_1_slot, iter_out_a_a_auto_unroll_id_1_slot, iter_out_a_b_auto_unroll_id_1_slot, qds1_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c1_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c2_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(out1_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c4_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(iter_out_a_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c3_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c1_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub1_auto_unroll_id_1_slot, score_delta_a_c1_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c1_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub2_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c2_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub2_auto_unroll_id_1_slot, score_delta_a_c2_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x2(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c4_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub4_auto_unroll_id_1_slot, score_delta_a_c4_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x3(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(beta_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c4_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kMinus_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(gamma_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kRes_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_a_c3_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_a_ub3_auto_unroll_id_1_slot, score_delta_a_c3_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_kminus_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_a_c3_auto_unroll_id_1_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kPlusB_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kPlus_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(kMinus_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_kminus_a_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(kPlus_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kPlusB_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kMinus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  copy_l1_to_l0ab_kplus_kminus(iter, kplus_kminus_a_slot, kplus_kminus_b_slot, kPlus_slot, kPlusB_slot, kMinus_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(kPlus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_kminus_a_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(scoreB_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(kPlusB_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kMinus_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(out1_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(out1_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kds1(iter - 1, out1_ub_auto_unroll_id_1_slot, out1_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_kminus_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(out1_c_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out1_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub4_auto_unroll_id_1_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(qPlusB_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(qPlus_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(qPlusB_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kplus_kminus_c_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(qPlus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_kminus_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qplus_kminus_a_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(kMinus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  copy_l1_to_l0ab_kplus_kminus(iter, qplus_kminus_a_slot, qplus_kminus_b_slot, qPlus_slot, qPlusB_slot, kMinus_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_kminus_u0(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  matmul_kplus_kminus(iter, kplus_kminus_c_slot, kplus_kminus_a_slot, kplus_kminus_b_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_qplus_score_b_u1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kplus_kminus_c_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qPlus_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qPlusB_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kMinus_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kMinus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qPlusB_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_kminus_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(qplus_kminus_a_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s2_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(s2_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(s2_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_b_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(qplus_kminus_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qplus_kminus_c_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_kplus_qplus_score_b(iter - 1, kplus_qplus_score_b_a_auto_unroll_id_1_slot, kplus_qplus_score_b_b_auto_unroll_id_1_slot, kd_auto_unroll_id_1_slot, qd_auto_unroll_id_1_slot, s2_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_kminus_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  matmul_kplus_kminus(iter, qplus_kminus_c_slot, qplus_kminus_a_slot, qplus_kminus_b_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kplus_kminus_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(kd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(qplus_kminus_c_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qd_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kd_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qplus_kminus_a_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_b_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(s2_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qd_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(qplus_kminus_c_slot);
		    copy_l0c_to_ub_kplus_kminus(iter, qplus_kminus_ub_slot, qplus_kminus_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kplus_kminus_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(qplus_kminus_c_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kplus_kminus_c_slot);
		    copy_l0c_to_ub_kplus_kminus(iter, kplus_kminus_ub_slot, kplus_kminus_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_qplus_score_a_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kplus_kminus_c_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qds2_c_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_b_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kds2_c_auto_unroll_id_1_slot);
		    matmul_kplus_qplus_score_a(iter - 1, kds2_c_auto_unroll_id_1_slot, qds2_c_auto_unroll_id_1_slot, kplus_qplus_score_b_a_auto_unroll_id_1_slot, kplus_qplus_score_b_b_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_gamma_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kds2_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_b_a_auto_unroll_id_1_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<VtoMTE1_UBRelease>(betaB_ub_slot);
		    syncCV.WaitIntraBlock<VtoMTE1_UBRelease>(gamma_ub_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(gamma_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(beta_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(gamma_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(beta_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  copy_l1_to_ub_gamma(iter, gamma_ub_slot, betaB_ub_slot, gamma_slot, beta_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u1_x2(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(gamma_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(beta_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(beta_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(gamma_slot);
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(betaB_ub_slot);
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(gamma_ub_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE3toFIX_UBRelease>(kds2_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kds2_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kds1(iter - 1, kds2_ub_auto_unroll_id_1_slot, kds2_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_y2_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kds2_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kds2_c_auto_unroll_id_1_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<VtoMTE1_UBRelease>(beta2_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(maskedQk_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr_b_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(v_corr_b_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(v_corr_b_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(y2_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(m_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_y2(iter - 1, y2_a_auto_unroll_id_1_slot, y2_b_auto_unroll_id_1_slot, m_auto_unroll_id_1_slot, v_corr_b_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_state_acc_a_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(m_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(y2_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(mm_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr_b_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(v_corr_b_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(y2_c_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(y2_a_auto_unroll_id_1_slot);
		    matmul_state_acc_a(iter - 1, y2_c_auto_unroll_id_1_slot, y2_a_auto_unroll_id_1_slot, y2_b_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l1_y1_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(y2_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(y2_c_auto_unroll_id_1_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(y2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(y2_c_auto_unroll_id_1_slot);
		    copy_l0c_to_l1_y1(iter - 1, y2_auto_unroll_id_1_slot, y2_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_iter_out_b_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_FIX>(y2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(y2_c_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kr_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(iter_out_b_a_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(mqk_auto_unroll_id_1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(y2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  copy_l1_to_l0ab_iter_out_b(iter - 1, iter_out_b_a_auto_unroll_id_1_slot, iter_out_b_b_auto_unroll_id_1_slot, kr_auto_unroll_id_1_slot, mqk_auto_unroll_id_1_slot, y2_auto_unroll_id_1_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_iter_out_a_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE1>(kr_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(y2_auto_unroll_id_1_slot);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(iter_out_b_a_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kr_auto_unroll_id_1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(mqk_auto_unroll_id_1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr_a_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(vCorrA_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(iter_out_b_a_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c2_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c4_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c1_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c3_auto_unroll_id_1_slot);
		    matmul_iter_out_a(iter - 1, out2_c_auto_unroll_id_1_slot, score_delta_b_c1_auto_unroll_id_1_slot, score_delta_b_c2_auto_unroll_id_1_slot, score_delta_b_c3_auto_unroll_id_1_slot, score_delta_b_c4_auto_unroll_id_1_slot, iter_out_b_a_auto_unroll_id_1_slot, iter_out_b_b_auto_unroll_id_1_slot, qds2_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x4(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c2_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(iter_out_b_a_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c4_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c1_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(out2_c_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c3_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c2_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub2_auto_unroll_id_1_slot, score_delta_b_c2_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x5(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c2_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c1_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub1_auto_unroll_id_1_slot, score_delta_b_c1_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x6(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c1_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub4_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c4_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub4_auto_unroll_id_1_slot, score_delta_b_c4_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_1_iter1_u1_x7(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c4_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out2_ub_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(score_delta_b_c3_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_1_iter1(iter - 1, score_delta_b_ub3_auto_unroll_id_1_slot, score_delta_b_c3_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u1_x3(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3_auto_unroll_id_1_slot);
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(score_delta_b_c3_auto_unroll_id_1_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(out2_ub_auto_unroll_id_1_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(out2_c_auto_unroll_id_1_slot);
		    copy_l0c_to_ub_kds1(iter - 1, out2_ub_auto_unroll_id_1_slot, out2_c_auto_unroll_id_1_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_l0ab_kplus_qplus_score_a_u0(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<VtoMTE1_UBRelease>(gamma_ub_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kds1_ub_slot);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(out2_c_auto_unroll_id_1_slot);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out2_ub_auto_unroll_id_1_slot);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(mm_slot);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(scoreA_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out1_ub_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub2_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(scoreA_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(qPlus_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(kPlus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_a_a_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(scoreA_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  copy_l1_to_l0ab_kplus_qplus_score_a(iter, kplus_qplus_score_a_a_slot, kplus_qplus_score_a_b_slot, kPlus_slot, qPlus_slot, scoreA_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_kplus_qplus_score_a_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_a_a_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(qPlus_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(scoreA_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(scoreA_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(kPlus_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kds1_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qds1_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kplus_qplus_score_a_a_slot);
		    matmul_kplus_qplus_score_a(iter, kds1_c_slot, qds1_c_slot, kplus_qplus_score_a_a_slot, kplus_qplus_score_a_b_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_ub_kds1_u0_x3(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kds1_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kplus_qplus_score_a_a_slot);
		    syncCV.WaitIntraBlock<VtoFIX_UBRelease>(kds1_ub_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kds1_c_slot);
		    copy_l0c_to_ub_kds1(iter, kds1_ub_slot, kds1_c_slot);
		}
	}

template <typename T, bool IsVarlen, bool HasTail>
template <bool STEADY>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::stage_tail66(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kds1_c_slot);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kds1_ub_slot);
		    syncCV.WaitIntraBlock<MTE3toMTE1_L1Ready>(mm_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(mm_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(inv_aux1_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(inv_aux1_a_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(mm_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(inv_aux1_a_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(inv_aux1_c_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(inv_aux1_a_slot);
		    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(inv_aux1_c_slot);
		    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(inv_aux1_c_slot);
		}
		AscendC::Mutex::Lock<PIPE_FIX>(invAux1_slot);
		AscendC::Mutex::Unlock<PIPE_FIX>(invAux1_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(inv_aux1_c_slot);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1_slot);
		}
		AscendC::Mutex::Lock<PIPE_MTE1>(invAux1_slot);
		AscendC::Mutex::Lock<PIPE_MTE1>(mm_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(inv_aux2_a_slot);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(vCorrB_slot);
		}
		AscendC::Mutex::Unlock<PIPE_MTE1>(invAux1_slot);
		AscendC::Mutex::Unlock<PIPE_MTE1>(invAuxT_slot);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(inv_aux2_a_slot);
	}

template <typename T, bool IsVarlen, bool HasTail>
	__aicore__ inline void Kernel<T, IsVarlen, HasTail>::Process(int group_count) {

		int iterN = 2;

		// 任务循环:与 AIV 侧使用同一任务序列
		// (task_count=0 → legacy 单任务)。AIC 无 GM 视图,任务间只需重置
		// 同步生命周期;stage 计数全部由 iter 参数驱动。
		const uint32_t task_begin = get_block_idx();
		const uint32_t task_end = task_count ? (uint32_t)task_count : (task_begin + 1);
		const uint32_t task_step = task_count ? (uint32_t)block_num : 1;
		for (uint32_t task = task_begin; task < task_end; task += task_step) {
			// 每任务组数:IsVarlen 从 cu_seqlens 解析(与 AIV 同一数据);空序列跳过
			int gc = group_count;
			if (is_varlen) {
				const uint32_t seq = task / (uint32_t)hv_real;
				const __gm__ int64_t *cu = (const __gm__ int64_t *)cu_base;
				const int64_t len = cu[seq + 1] - cu[seq];
				if (len <= 0) {
					continue;
				}
				gc = (int)((len + tokens_per_group - 1) / tokens_per_group);
			}
			syncCV.Init();

			// 边沿组直接复用 stage;守卫、同步和 buffer 管理均留在 stage 内。
			for (int iter = 0; iter < iterN - 1 && iter < gc; iter ++) {
				stage_l0ab_y1_u0<false>(iter, gc);
				stage_state_acc_a_u0<false>(iter, gc);
				stage_l1_y1_u0<false>(iter, gc);
				stage_l0ab_iter_out_a_u0<false>(iter, gc);
				stage_iter_out_a_u0<false>(iter, gc);
				stage_ub_1_iter1_u0<false>(iter, gc);
				stage_ub_1_iter1_u0_x1<false>(iter, gc);
				stage_ub_1_iter1_u0_x2<false>(iter, gc);
				stage_ub_1_iter1_u0_x3<false>(iter, gc);
				stage_l0ab_kplus_kminus_u1<false>(iter, gc);
				stage_ub_kds1_u0<false>(iter, gc);
				stage_l0ab_kplus_kminus_u1_x1<false>(iter, gc);
				stage_kplus_kminus_u1<false>(iter, gc);
				stage_kplus_kminus_u1_x1<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u0<false>(iter, gc);
				stage_ub_kplus_kminus_u1<false>(iter, gc);
				stage_ub_kplus_kminus_u1_x1<false>(iter, gc);
				stage_kplus_qplus_score_a_u0<false>(iter, gc);
				stage_ub_gamma_u1<false>(iter, gc);
				stage_ub_kds1_u0_x1<false>(iter, gc);
				stage_l0ab_y2_u0<false>(iter, gc);
				stage_state_acc_a_u0_x1<false>(iter, gc);
				stage_l1_y1_u0_x1<false>(iter, gc);
				stage_l0ab_iter_out_b_u0<false>(iter, gc);
				stage_iter_out_a_u0_x1<false>(iter, gc);
				stage_ub_1_iter1_u0_x4<false>(iter, gc);
				stage_ub_1_iter1_u0_x5<false>(iter, gc);
				stage_ub_1_iter1_u0_x6<false>(iter, gc);
				stage_ub_1_iter1_u0_x7<false>(iter, gc);
				stage_ub_kds1_u0_x2<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u1<false>(iter, gc);
				stage_kplus_qplus_score_a_u1<false>(iter, gc);
				stage_ub_kds1_u1<false>(iter, gc);
				stage_l0ab_y1_u1<false>(iter, gc);
				stage_state_acc_a_u1<false>(iter, gc);
				stage_l1_y1_u1<false>(iter, gc);
				stage_l0ab_iter_out_a_u1<false>(iter, gc);
				stage_iter_out_a_u1<false>(iter, gc);
				stage_ub_1_iter1_u1<false>(iter, gc);
				stage_ub_1_iter1_u1_x1<false>(iter, gc);
				stage_ub_1_iter1_u1_x2<false>(iter, gc);
				stage_ub_1_iter1_u1_x3<false>(iter, gc);
				stage_l0ab_kplus_kminus_u0<false>(iter, gc);
				stage_ub_kds1_u1_x1<false>(iter, gc);
				stage_l0ab_kplus_kminus_u0_x1<false>(iter, gc);
				stage_kplus_kminus_u0<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u1<false>(iter, gc);
				stage_kplus_kminus_u0_x1<false>(iter, gc);
				stage_ub_kplus_kminus_u0<false>(iter, gc);
				stage_ub_kplus_kminus_u0_x1<false>(iter, gc);
				stage_kplus_qplus_score_a_u1_x1<false>(iter, gc);
				stage_ub_gamma_u0<false>(iter, gc);
				stage_ub_kds1_u1_x2<false>(iter, gc);
				stage_l0ab_y2_u1<false>(iter, gc);
				stage_state_acc_a_u1_x1<false>(iter, gc);
				stage_l1_y1_u1_x1<false>(iter, gc);
				stage_l0ab_iter_out_b_u1<false>(iter, gc);
				stage_iter_out_a_u1_x1<false>(iter, gc);
				stage_ub_1_iter1_u1_x4<false>(iter, gc);
				stage_ub_1_iter1_u1_x5<false>(iter, gc);
				stage_ub_1_iter1_u1_x6<false>(iter, gc);
				stage_ub_1_iter1_u1_x7<false>(iter, gc);
				stage_ub_kds1_u1_x3<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u0<false>(iter, gc);
				stage_kplus_qplus_score_a_u0_x1<false>(iter, gc);
				stage_ub_kds1_u0_x3<false>(iter, gc);
				stage_tail66<false>(iter, gc);
			}

			for (int iter = iterN - 1; iter < gc; iter ++) {
				stage_l0ab_y1_u0<true>(iter, gc);
				stage_state_acc_a_u0<true>(iter, gc);
				stage_l1_y1_u0<true>(iter, gc);
				stage_l0ab_iter_out_a_u0<true>(iter, gc);
				stage_iter_out_a_u0<true>(iter, gc);
				stage_ub_1_iter1_u0<true>(iter, gc);
				stage_ub_1_iter1_u0_x1<true>(iter, gc);
				stage_ub_1_iter1_u0_x2<true>(iter, gc);
				stage_ub_1_iter1_u0_x3<true>(iter, gc);
				stage_l0ab_kplus_kminus_u1<true>(iter, gc);
				stage_ub_kds1_u0<true>(iter, gc);
				stage_l0ab_kplus_kminus_u1_x1<true>(iter, gc);
				stage_kplus_kminus_u1<true>(iter, gc);
				stage_kplus_kminus_u1_x1<true>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u0<true>(iter, gc);
				stage_ub_kplus_kminus_u1<true>(iter, gc);
				stage_ub_kplus_kminus_u1_x1<true>(iter, gc);
				stage_kplus_qplus_score_a_u0<true>(iter, gc);
				stage_ub_gamma_u1<true>(iter, gc);
				stage_ub_kds1_u0_x1<true>(iter, gc);
				stage_l0ab_y2_u0<true>(iter, gc);
				stage_state_acc_a_u0_x1<true>(iter, gc);
				stage_l1_y1_u0_x1<true>(iter, gc);
				stage_l0ab_iter_out_b_u0<true>(iter, gc);
				stage_iter_out_a_u0_x1<true>(iter, gc);
				stage_ub_1_iter1_u0_x4<true>(iter, gc);
				stage_ub_1_iter1_u0_x5<true>(iter, gc);
				stage_ub_1_iter1_u0_x6<true>(iter, gc);
				stage_ub_1_iter1_u0_x7<true>(iter, gc);
				stage_ub_kds1_u0_x2<true>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u1<true>(iter, gc);
				stage_kplus_qplus_score_a_u1<true>(iter, gc);
				stage_ub_kds1_u1<true>(iter, gc);
				stage_l0ab_y1_u1<true>(iter, gc);
				stage_state_acc_a_u1<true>(iter, gc);
				stage_l1_y1_u1<true>(iter, gc);
				stage_l0ab_iter_out_a_u1<true>(iter, gc);
				stage_iter_out_a_u1<true>(iter, gc);
				stage_ub_1_iter1_u1<true>(iter, gc);
				stage_ub_1_iter1_u1_x1<true>(iter, gc);
				stage_ub_1_iter1_u1_x2<true>(iter, gc);
				stage_ub_1_iter1_u1_x3<true>(iter, gc);
				stage_l0ab_kplus_kminus_u0<true>(iter, gc);
				stage_ub_kds1_u1_x1<true>(iter, gc);
				stage_l0ab_kplus_kminus_u0_x1<true>(iter, gc);
				stage_kplus_kminus_u0<true>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u1<true>(iter, gc);
				stage_kplus_kminus_u0_x1<true>(iter, gc);
				stage_ub_kplus_kminus_u0<true>(iter, gc);
				stage_ub_kplus_kminus_u0_x1<true>(iter, gc);
				stage_kplus_qplus_score_a_u1_x1<true>(iter, gc);
				stage_ub_gamma_u0<true>(iter, gc);
				stage_ub_kds1_u1_x2<true>(iter, gc);
				stage_l0ab_y2_u1<true>(iter, gc);
				stage_state_acc_a_u1_x1<true>(iter, gc);
				stage_l1_y1_u1_x1<true>(iter, gc);
				stage_l0ab_iter_out_b_u1<true>(iter, gc);
				stage_iter_out_a_u1_x1<true>(iter, gc);
				stage_ub_1_iter1_u1_x4<true>(iter, gc);
				stage_ub_1_iter1_u1_x5<true>(iter, gc);
				stage_ub_1_iter1_u1_x6<true>(iter, gc);
				stage_ub_1_iter1_u1_x7<true>(iter, gc);
				stage_ub_kds1_u1_x3<true>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u0<true>(iter, gc);
				stage_kplus_qplus_score_a_u0_x1<true>(iter, gc);
				stage_ub_kds1_u0_x3<true>(iter, gc);
				stage_tail66<true>(iter, gc);
			}
			for (int iter = gc; iter < gc + iterN - 1; iter ++) {
				stage_l0ab_y1_u0<false>(iter, gc);
				stage_state_acc_a_u0<false>(iter, gc);
				stage_l1_y1_u0<false>(iter, gc);
				stage_l0ab_iter_out_a_u0<false>(iter, gc);
				stage_iter_out_a_u0<false>(iter, gc);
				stage_ub_1_iter1_u0<false>(iter, gc);
				stage_ub_1_iter1_u0_x1<false>(iter, gc);
				stage_ub_1_iter1_u0_x2<false>(iter, gc);
				stage_ub_1_iter1_u0_x3<false>(iter, gc);
				stage_l0ab_kplus_kminus_u1<false>(iter, gc);
				stage_ub_kds1_u0<false>(iter, gc);
				stage_l0ab_kplus_kminus_u1_x1<false>(iter, gc);
				stage_kplus_kminus_u1<false>(iter, gc);
				stage_kplus_kminus_u1_x1<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u0<false>(iter, gc);
				stage_ub_kplus_kminus_u1<false>(iter, gc);
				stage_ub_kplus_kminus_u1_x1<false>(iter, gc);
				stage_kplus_qplus_score_a_u0<false>(iter, gc);
				stage_ub_gamma_u1<false>(iter, gc);
				stage_ub_kds1_u0_x1<false>(iter, gc);
				stage_l0ab_y2_u0<false>(iter, gc);
				stage_state_acc_a_u0_x1<false>(iter, gc);
				stage_l1_y1_u0_x1<false>(iter, gc);
				stage_l0ab_iter_out_b_u0<false>(iter, gc);
				stage_iter_out_a_u0_x1<false>(iter, gc);
				stage_ub_1_iter1_u0_x4<false>(iter, gc);
				stage_ub_1_iter1_u0_x5<false>(iter, gc);
				stage_ub_1_iter1_u0_x6<false>(iter, gc);
				stage_ub_1_iter1_u0_x7<false>(iter, gc);
				stage_ub_kds1_u0_x2<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u1<false>(iter, gc);
				stage_kplus_qplus_score_a_u1<false>(iter, gc);
				stage_ub_kds1_u1<false>(iter, gc);
				stage_l0ab_y1_u1<false>(iter, gc);
				stage_state_acc_a_u1<false>(iter, gc);
				stage_l1_y1_u1<false>(iter, gc);
				stage_l0ab_iter_out_a_u1<false>(iter, gc);
				stage_iter_out_a_u1<false>(iter, gc);
				stage_ub_1_iter1_u1<false>(iter, gc);
				stage_ub_1_iter1_u1_x1<false>(iter, gc);
				stage_ub_1_iter1_u1_x2<false>(iter, gc);
				stage_ub_1_iter1_u1_x3<false>(iter, gc);
				stage_l0ab_kplus_kminus_u0<false>(iter, gc);
				stage_ub_kds1_u1_x1<false>(iter, gc);
				stage_l0ab_kplus_kminus_u0_x1<false>(iter, gc);
				stage_kplus_kminus_u0<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_b_u1<false>(iter, gc);
				stage_kplus_kminus_u0_x1<false>(iter, gc);
				stage_ub_kplus_kminus_u0<false>(iter, gc);
				stage_ub_kplus_kminus_u0_x1<false>(iter, gc);
				stage_kplus_qplus_score_a_u1_x1<false>(iter, gc);
				stage_ub_gamma_u0<false>(iter, gc);
				stage_ub_kds1_u1_x2<false>(iter, gc);
				stage_l0ab_y2_u1<false>(iter, gc);
				stage_state_acc_a_u1_x1<false>(iter, gc);
				stage_l1_y1_u1_x1<false>(iter, gc);
				stage_l0ab_iter_out_b_u1<false>(iter, gc);
				stage_iter_out_a_u1_x1<false>(iter, gc);
				stage_ub_1_iter1_u1_x4<false>(iter, gc);
				stage_ub_1_iter1_u1_x5<false>(iter, gc);
				stage_ub_1_iter1_u1_x6<false>(iter, gc);
				stage_ub_1_iter1_u1_x7<false>(iter, gc);
				stage_ub_kds1_u1_x3<false>(iter, gc);
				stage_l0ab_kplus_qplus_score_a_u0<false>(iter, gc);
				stage_kplus_qplus_score_a_u0_x1<false>(iter, gc);
				stage_ub_kds1_u0_x3<false>(iter, gc);
				stage_tail66<false>(iter, gc);
			}




			syncCV.Finalize();
		}  // task 循环

	}




#endif
