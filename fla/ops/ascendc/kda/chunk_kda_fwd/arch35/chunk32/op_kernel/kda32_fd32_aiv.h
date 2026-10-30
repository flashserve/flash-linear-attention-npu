#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif


#ifdef __DAV_C310_VEC__

#include "gen_utils_v.h"

#include "kernel_operator.h"
#include "kernel_tensor.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
	AscendC::MicroAPI::RegLayout::ZERO,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
	AscendC::MicroAPI::RegLayout::ONE,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTwo = {
	AscendC::MicroAPI::RegLayout::TWO,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitThree = {
	AscendC::MicroAPI::RegLayout::THREE,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintZero = {
	AscendC::MicroAPI::RegLayout::ZERO,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintOne = {
	AscendC::MicroAPI::RegLayout::ONE,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintTwo = {
	AscendC::MicroAPI::RegLayout::TWO,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintThree = {
	AscendC::MicroAPI::RegLayout::THREE,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait bf2fZero = {
	AscendC::MicroAPI::RegLayout::ZERO,
		AscendC::MicroAPI::SatMode::UNKNOWN,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait bf2fOne = {
	AscendC::MicroAPI::RegLayout::ONE,
		AscendC::MicroAPI::SatMode::UNKNOWN,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait f2bfZero = {
	AscendC::MicroAPI::RegLayout::ZERO,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait f2bfOne = {
	AscendC::MicroAPI::RegLayout::ONE,
		AscendC::MicroAPI::SatMode::SAT,
		AscendC::MicroAPI::MaskMergeMode::ZEROING,
		AscendC::RoundMode::CAST_RINT,
};


// UB/L1 整片句柄(无 TPipe 的轻量裸地址包装)
// LocalTensor(TPosition, addr, size) 官方裸地址构造,地址表数值不变
// 本 kernel 的 UB/L1 窗口字节数(LocalTensor 长度上限,非硬件总量)
constexpr uint32_t ub_window_bytes = 192 * 1024;
constexpr uint32_t l1_window_bytes = 512 * 1024;

__aicore__ inline AscendC::LocalTensor<uint8_t> Ub(uint32_t byteOff)
{
	return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, byteOff, ub_window_bytes);
}

__aicore__ inline AscendC::LocalTensor<uint8_t> L1b(uint32_t byteOff)
{
	return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, byteOff, l1_window_bytes);
}

// KDA 形状常量(与 aic 侧一致;数值相同的轴按语义分开命名)
constexpr uint32_t key_dim = 128;      // q/k/gate/dt_bias 特征维
constexpr uint32_t value_dim = 128;    // v/o/state 列向特征维(当前 == key_dim,语义不同轴)
constexpr uint32_t chunk_size = 32;    // 一个 chunk 的 token 数
constexpr uint32_t chunk_half_rows = chunk_size / 2;  // 一个 chunk 分给两个 V,每 V 处理的行数
constexpr uint32_t tokens_per_prepare_pair = 2 * chunk_size;  // 两个 V 各备一个 chunk:同 AIV 相邻两次准备的行距
constexpr uint32_t tokens_per_group = 4 * chunk_size;  // 一个 group 的 token 数:4 个 chunk(=128)
// vec_prepare_g3 变换/归一化的布局常量(数值巧合的轴不共用名字):
constexpr uint32_t half_channels = key_dim / 2;      // Cast 偶/奇通道拆分的半宽(数值上恰为 fp32 寄存器 64 lane)
constexpr uint32_t row_pair = 2;                // 行对:变换/修正类循环每次迭代处理的一对相邻行
constexpr uint32_t fp32_lanes = 64;                // 一个 fp32 向量寄存器的 lane 数(= 4 行 x 16 列分形表)
constexpr uint32_t tmp_slot_elems = fp32_lanes;    // tmp_ub 内暂存数组(q_sq/k_sq/gamma1/gamma2)的间隔:一个 fp32 寄存器宽
constexpr uint32_t block_elems_16b = 16;           // 一个 32B 数据块容纳的 16 位元素数(DATA_BLOCK_COPY 的地址粒度)
constexpr uint32_t bits_per_16b_elem = 16;         // 16 位元素的位宽(bf16 位技巧:零扩展后左移成 fp32 位模式)
constexpr uint32_t block_bytes = 32;               // DataCopy 的块粒度字节数
constexpr uint32_t block_words32 = block_bytes / 4;  // 一个 32B 块的 uint32 字数(PACK_B32 跨步粒度)
// state IO(s_ub NZ 分形):
constexpr int32_t fractal_len = 16;   // 有符号:tril_inv 的负跨步/回退指针依赖负值语义               // NZ 分形块边长(16x16)
constexpr int32_t fractal_elems = fractal_len * fractal_len;  // 一个 NZ 分形块的元素数
constexpr uint32_t state_k_half = key_dim / 2;     // 每 AIV 分担的 state k 行数(与 half_channels 同值不同轴)
constexpr uint32_t state_half_elems = state_k_half * value_dim;  // 本 AIV 半幅 state 的 fp32 元素数
constexpr uint32_t fv_block_bytes = state_k_half * fractal_len * 2;  // 一个 fv 分形块字节数(64k x 16v x 2B;2 = 16 位元素字节)
constexpr uint32_t state_fill_unroll = 4;          // InitState 清零每次迭代写的行数
constexpr uint32_t value_tile = value_dim / 4;         // state 更新按 value 维 4 段流水的段宽(与 aic 侧一致)

// UB/L1 槽位表(addrUB/addrL1)的地址单位:表项值 x 单位 = 字节偏移
constexpr uint32_t ub_addr_unit = 512;   // addrUB 表项单位(字节)
constexpr uint32_t l1_addr_unit = 1024;  // addrL1 表项单位(字节)

// T 使用 bf16,g/beta 可为 bf16 或 fp32,内部状态与计算视图跟随对应类型。
// g/beta 可为 bf16 或 fp32,由 gate_flags 选择原生视图;内部 s_ub/masked_qk 等跟随 T,
// a_log/dt_bias/state GM 恒为 fp32,不随 T 变。
template <typename T>
class KDAImpl
{
public:
	AscendC::GlobalTensor<T> g_gm, q_gm, k_gm, v_gm, b_gm, tmp_gm, o_gm;
	AscendC::GlobalTensor<float> g32_gm, b32_gm;  // fp32 原生组合的 g/beta GM 视图(与 16 位视图同址)
	uint64_t gate_flags = 0;  // tiling gateFlags:bit0 = g fp32,bit1 = beta fp32
	AscendC::GlobalTensor<float> dt_gm;
	// 裸地址桥接(Stage 3)
	template <typename U>
	__aicore__ inline uint32_t KdaAddrV(U *p)
	{
		return static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
	}

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

	__aicore__ inline void InitState(__ubuf__ void *s_ub)
	{
		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<uint16_t, MaskPattern::ALL>();
			RegTensor<uint16_t> vreg;
			Duplicate(vreg, (uint16_t)0);
			for(uint16_t i = 0; i < (uint16_t)state_k_half; i += state_fill_unroll) {
				DataCopy((__ubuf__ uint16_t *)s_ub + (i + 0)*value_dim, vreg, preg_all);
				DataCopy((__ubuf__ uint16_t *)s_ub + (i + 1)*value_dim, vreg, preg_all);
				DataCopy((__ubuf__ uint16_t *)s_ub + (i + 2)*value_dim, vreg, preg_all);
				DataCopy((__ubuf__ uint16_t *)s_ub + (i + 3)*value_dim, vreg, preg_all);
			}
		}
	}
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
	__aicore__ inline void LoadStateNdScatter(__ubuf__ void *s_ub, __ubuf__ void *stage_f32)
	{
		__VEC_SCOPE__{
			MaskReg preg_f32 = CreateMask<uint8_t, MaskPattern::ALL>();
			MaskReg preg_bf16 = CreateMask<uint16_t, MaskPattern::ALL>();
			for(uint16_t r = 0; r < (uint16_t)state_k_half; r ++) {
				RegTensor<float> vreg_f0, vreg_f1;
				RegTensor<T> vreg_b;
				DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_f0, vreg_f1, (__ubuf__ float *)stage_f32 + r*value_dim);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_f0, vreg_f0, preg_f32);
				Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_f1, vreg_f1, preg_f32);
				Or((RegTensor<uint16_t> &)vreg_b, (RegTensor<uint16_t> &)vreg_f0, (RegTensor<uint16_t> &)vreg_f1, preg_bf16);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
					(__ubuf__ T *)s_ub + r * fractal_len, vreg_b, fv_block_bytes / block_bytes, preg_bf16);
			}
		}
	}
	// s_ub bf16 NZ → stage fp32 ND:vsldb 块聚集载(与装载侧散布存同族镜像),
	// 行 r 从 r*16 聚 8 块 32B(块距 64×32B=2048B=fv 步长)
	// 得 128 bf16 线性序,cast 后 INTLV 存出 128 fp32 连续行。
	__aicore__ inline void StoreStateGatherCast(__ubuf__ void *stage_f32, __ubuf__ void *s_ub)
	{
		__VEC_SCOPE__{
			MaskReg preg_f32 = CreateMask<uint8_t, MaskPattern::ALL>();
			MaskReg preg_bf16 = CreateMask<uint16_t, MaskPattern::ALL>();
			for(uint16_t r = 0; r < (uint16_t)state_k_half; r ++) {
				RegTensor<T> vreg_b;
				RegTensor<float> vreg_f0, vreg_f1;
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
					vreg_b, (__ubuf__ T *)s_ub + r * fractal_len, fv_block_bytes / block_bytes, preg_bf16);
				Cast<float, T, bf2fZero>(vreg_f0, vreg_b, preg_f32);
				Cast<float, T, bf2fOne >(vreg_f1, vreg_b, preg_f32);
				DataCopy<float, StoreDist::DIST_INTLV_B32>((__ubuf__ float *)stage_f32 + r*value_dim, vreg_f0, vreg_f1, preg_f32);
			}
		}
	}
	// initial_state 装载:GM fp32 ND [64k,128v](本 AIV 半幅)→ s_ub bf16 NZ。
	// V_MTE2(0/1) 是输入区首个使用者的预支信号量:先消费再归还,借用期间
	// prologue 的装数自然被挡在墙外。
	__aicore__ inline void LoadInitState(__ubuf__ void *s_ub)
	{
		// 初始状态的 FP32 ND 暂存区,借用从 g_ub[0] 开始的连续 32KiB 输入区
		// (指针别名,不分配新 UB;范围可覆盖多个输入 buffer,借用期靠下方
		// V_MTE2(0/1) 预支/归还挡住 prologue 装数)。
		__ubuf__ void *istate_ub = g_ub[0];
		AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC, KdaAddrV(istate_ub), state_half_elems);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
		AscendC::DataCopy(stage, istate_gm[0], state_half_elems);   // 一次连续 32KB(ND 行主序)
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
		LoadStateNdScatter(s_ub, (__ubuf__ void *)stage.GetPhyAddr());
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
	}
	// ===== 尾块(T%128≠0)变换后修正 =====
	// 无效行在装载阶段处理:
	//   raw g 尾行填 -inf → step_gate = gate_scale/(1+exp(a*(-inf+dt))) = -0,
	//   gamma(=Σgg)与 cumsum 天然零贡献,原 FixTailGamma 的"先累计无效贡献、
	//   再事后扣减"随之取消;raw q/k 尾行补零 + epsilon 归一化天然归零。
	// 此处仅剩 beta:beta_eff(fp32 [32],槽 ≥512B,64 lane 整寄存器读写安全):
	// 车道 >= valid 清零(sigmoid(0)=0.5/陈旧值)。
	// Arange+Compare+Select(GDN 运行时掩码习语),Select 是位选不传播 NaN。
	__aicore__ inline void FixTailBetaF32(__ubuf__ void *beta_ub, float valid_cnt)
	{
		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
			RegTensor<float> vreg_col, vreg_rv, vreg_beta, vreg_zero;
			MaskReg preg_keep;
			Arange(vreg_col, 0.0f);
			Duplicate(vreg_rv, valid_cnt, preg_all);
			Duplicate(vreg_zero, (float)0.0, preg_all);
			DataCopy(vreg_beta, (__ubuf__ float *)beta_ub);
			Compare<float, CMPMODE::LT>(preg_keep, vreg_col, vreg_rv, preg_all);
			Select(vreg_beta, vreg_beta, vreg_zero, preg_keep);
			DataCopy((__ubuf__ float *)beta_ub, vreg_beta, preg_all);
		}
	}
	// final_state 写回:s_ub bf16 NZ → fp32 NZ 暂存(同借输入区)→ MTE3 散回 GM ND。
	// 只在末组状态更新全部落地后调用(Process 收尾);V 管同序保证读到的是终态。
	__aicore__ inline void StoreFinalState(__ubuf__ void *s_ub)
	{
		AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[0]), state_half_elems);
		StoreStateGatherCast((__ubuf__ void *)stage.GetPhyAddr(), s_ub);
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::DataCopy(fstate_gm[0], stage, state_half_elems);   // 一次连续 32KB(ND 行主序)
		// 多任务执行时必须自排干:stage 借用 g_ub[0],MTE3 读未完成时下一任务
		// 的 MTE2 装载会冲掉它(单任务靠 kernel 退出隐式排干,多任务无此保障)
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_gm_single(const AscendC::GlobalTensor<typename AscendC::Std::conditional<element_size == 4, float, T>::type> &mem,
		__ubuf__ void * ub)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> src(AscendC::TPosition::VECCALC, KdaAddrV(ub), m * n);
		AscendC::DataCopy(mem, src, m * n);   // 原 copy_ubuf_to_gm_align_v2 单 burst 连续
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void gm_ub_single(__ubuf__ void * ub,
		const AscendC::GlobalTensor<typename AscendC::Std::conditional<element_size == 4, float, T>::type> &mem)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> dst(AscendC::TPosition::VECCALC, KdaAddrV(ub), m * n);
		AscendC::DataCopy(dst, mem, m * n);
	}

	__aicore__ inline void Init(uint64_t N_, uint64_t scale_, uint64_t gate_scale_, GM_ADDR a_gm_, GM_ADDR g_gm_, GM_ADDR q_gm_,
		GM_ADDR k_gm_, GM_ADDR v_gm_, GM_ADDR b_gm_, GM_ADDR dt_gm_, GM_ADDR tmp_gm_, GM_ADDR o_gm_,
		GM_ADDR istate_gm_ = nullptr, GM_ADDR fstate_gm_ = nullptr,
		uint64_t has_istate_ = 0, uint64_t out_fstate_ = 0, uint64_t seq_len_ = 0,
		uint64_t qk_group_ = 1,
		uint64_t task_count_ = 0, uint64_t hv_real_ = 0, uint64_t block_num_ = 0,
		uint64_t is_varlen_ = 0, uint64_t cu_base_ = 0, uint64_t epsilon_bits_ = 0x3F50624Du,
		uint64_t gate_flags_ = 0)
	{
		coreid = get_block_idx();
		subid = get_subblockid()&1;
		// GVA(issue #829):一核一 value head,q/k 头按组映射 qk_head = head / (HV/H);
		// qk_group_=1 时退化为 H==HV 原契约。batch 拍平后 fh/qk_group 恰为
		// batch*H + qk 头(HV = H*qk_group 整除),见 BindTask。
		qk_group = (uint32_t)qk_group_;
		task_count = task_count_;
		hv_real = hv_real_;
		block_num = block_num_;
		is_varlen = is_varlen_;
		cu_base = cu_base_;
		nheads_qk = hv_real_ / qk_group_;   // varlen 下 q/k 的行距头数(H);dense 不用

		group_count = N_;
		// 头间 GM 步长用真实 T(尾块时 < group_count*tokens_per_group);legacy 入口不传 seq_len 则无尾块
		seq_len = seq_len_ ? seq_len_ : (uint64_t)N_*tokens_per_group;
		total_tokens = seq_len;   // varlen 下 BindTask 逐任务改写 seq_len,此行距保持不变
		has_tail_ = (seq_len % tokens_per_group != 0) ? 1 : 0;
		scale16 = (uint16_t)scale_;
		gate_scale = *((float *)(&gate_scale_));
		epsilon = *((float *)(&epsilon_bits_));
		gate_flags = gate_flags_;
		// 只存基址;按任务的视图偏移全在 BindTask(Process 每个任务开头调用)
		a_base = (uint64_t)a_gm_; g_base = (uint64_t)g_gm_; q_base = (uint64_t)q_gm_; k_base = (uint64_t)k_gm_;
		v_base = (uint64_t)v_gm_; b_base = (uint64_t)b_gm_; dt_base = (uint64_t)dt_gm_;
		tmp_base = (uint64_t)tmp_gm_; o_base = (uint64_t)o_gm_;
		istate_base = (uint64_t)istate_gm_; fstate_base = (uint64_t)fstate_gm_;
		a_gm.SetGlobalBuffer((__gm__ float *)a_gm_);
		has_istate = (uint32_t)has_istate_;
		out_fstate = (uint32_t)out_fstate_;

		// UB 布局表(字节),以下名称仅用于标识地址:
		constexpr uint32_t qkg_slot_bytes = chunk_size * key_dim * sizeof(T);       // q/g/k 槽:整 chunk 32 行 x 128 列 16 位
		constexpr uint32_t v_slot_bytes = chunk_half_rows * value_dim * sizeof(T);  // v 槽:半 chunk(16 行)x 128 列 16 位
		constexpr uint32_t beta_slot_bytes = 256;            // beta 槽:raw 32x2B 与 beta_eff 32x4B 复用
		constexpr uint32_t dt_slot_bytes = key_dim * sizeof(float);                 // dt_bias 槽:128 个 fp32
		constexpr uint32_t tmp_slot_bytes = 5 * 512;         // tmp 槽:暂存数组区 + 16x16 fp32 表
		constexpr uint32_t tmp_table_off = 3 * 512;          // strict_tril_mask/identity 在 tmp 槽内的偏移
		uint64_t addr = 368*512;   // 本 kernel UB 区起点
		q_ub[0] = (__ubuf__ void *)get_imm(addr); addr += qkg_slot_bytes;
		q_ub[1] = (__ubuf__ void *)get_imm(addr); addr += qkg_slot_bytes + 256;   // 256B 间隙,不并入槽
		g_ub[0] = (__ubuf__ void *)get_imm(addr); addr += qkg_slot_bytes;
		// 原 g_ub[1] 与 g_ub[0] 组成 fp32 raw slot 0(16KB);
		// 第二个独立槽移到 addrUB[0] 腾出的 16KB,bf16 仍只使用各槽前 8KB。
		g_ub[1] = (__ubuf__ void *)get_imm(0); addr += qkg_slot_bytes;
		k_ub[0] = (__ubuf__ void *)get_imm(addr); addr += qkg_slot_bytes;
		k_ub[1] = (__ubuf__ void *)get_imm(addr); addr += qkg_slot_bytes;
		v_ub[0] = (__ubuf__ void *)get_imm(addr); addr += v_slot_bytes;
		v_ub[1] = (__ubuf__ void *)get_imm(addr); addr += v_slot_bytes;
		v_ub[2] = (__ubuf__ void *)get_imm(addr); addr += v_slot_bytes;
		v_ub[3] = (__ubuf__ void *)get_imm(addr); addr += v_slot_bytes;
		b_ub[0] = (__ubuf__ void *)get_imm(addr); addr += beta_slot_bytes;
		b_ub[1] = (__ubuf__ void *)get_imm(addr); addr += beta_slot_bytes;
		b_ub[2] = (__ubuf__ void *)get_imm(addr); addr += beta_slot_bytes;
		b_ub[3] = (__ubuf__ void *)get_imm(addr); addr += beta_slot_bytes;
		dt_ub = (__ubuf__ void *)get_imm(addr); addr += dt_slot_bytes;
		tmp_ub[0] = (__ubuf__ void *)get_imm(addr); addr += tmp_slot_bytes;
		tmp_ub[1] = (__ubuf__ void *)get_imm(addr); addr += tmp_slot_bytes;

		strict_tril_mask = (__ubuf__ void *)((__ubuf__ uint8_t *)tmp_ub[0] + tmp_table_off);
		identity = (__ubuf__ void *)((__ubuf__ uint8_t *)tmp_ub[1] + tmp_table_off);

		for(int i = 0; i < (int)fractal_len; i ++) {
			((__ubuf__ float *)identity)[i*fractal_len + i] = (float)1.0;
			((__ubuf__ float *)strict_tril_mask)[i*fractal_len + i] = (float)0.0;
			for(int j = 0; j < i; j ++) {
				((__ubuf__ float *)strict_tril_mask)[i*fractal_len + j] = (float)1.0;
				((__ubuf__ float *)identity)[i*fractal_len + j] = (float)0.0;
			}
			for(int j = i + 1; j < (int)fractal_len; j ++) {
				((__ubuf__ float *)strict_tril_mask)[i*fractal_len + j] = (float)0.0;
				((__ubuf__ float *)identity)[i*fractal_len + j] = (float)0.0;
			}
		}

		// UB2GM Init
		//SPR.LOOP_SIZE_UBTOOUT
		uint64_t loop_size = 0;
		//[20:0]: loop1_size, set to 1
		loop_size = 1;
		//[42:21]: loop2_size, set to 1
		loop_size += (1 << 21);
		set_loop_size_ubtoout(loop_size);

		//SPR.LOOP1_STRIDE_UBTOOUT, set to 0
		//[39:0]: dst loop1 stride, in unit of Byte
		//[60:40]: src loop1 stride, in unit of byte
		set_loop1_stride_ubtoout(0);

		//SPR.LOOP2_STRIDE_UBTOOUT
		//[39:0]: dst loops stride, in unit of Byte
		//[60:40]: src loop2 stride, in unit of Byte
		set_loop2_stride_ubtoout(0);

		// GM2UB Init
		// SPR.PAD_VAL_OUTTOUB
		set_pad_val_outtoub(0);

		//SPR.LOOP_SIZE_OUTTOUB
		//loop_size = 0;
		//[20:0]: loop1_size, set to 1
		loop_size = 1;
		//[42:21]: loop2_size, set to 1
		loop_size += (1 << 21);
		set_loop_size_outtoub(loop_size);

		//SPR.LOOP1_STRIDE_OUTTOUB, set to 0
		//[39:0]: dst loop1 stride, in unit of Byte
		//[60:40]: src loop1 stride, in unit of byte
		set_loop1_stride_outtoub(0);

		//SPR.LOOP2_STRIDE_OUTTOUB
		//[39:0]: dst loops stride, in unit of Byte
		//[60:40]: src loop2 stride, in unit of Byte
		set_loop2_stride_outtoub(0);

		//set SPR
		set_atomic_none();
		set_pcie_wr_ctrl(0);
		set_pcie_rd_ctrl(0);
	}

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
	__aicore__ inline void BindTaskVarlen(uint32_t fh)
	{
		const uint32_t hv = (uint32_t)hv_real;
		const uint32_t seq = fh / hv;
		const uint32_t head = fh - seq * hv;
		const uint32_t qk_head = head / qk_group;
		const __gm__ int64_t *cu = (const __gm__ int64_t *)cu_base;
		const int64_t tok0 = cu[seq];
		const int64_t tok1 = cu[seq + 1];
		const int64_t len = tok1 - tok0;
		task_empty = (len <= 0) ? 1 : 0;
		task_tokens = (uint64_t)tok0;
		seq_len = task_empty ? 1 : (uint64_t)len;
		group_count = task_empty ? 0 : (uint32_t)((len + tokens_per_group - 1) / tokens_per_group);
		has_tail_ = (seq_len % tokens_per_group != 0) ? 1 : 0;
		const uint64_t qkOffset = is_varlen == 2
			? ((uint64_t)qk_head * total_tokens + (uint64_t)tok0) * key_dim
			: ((uint64_t)tok0 * nheads_qk + qk_head) * key_dim;
		const uint64_t gOffset = is_varlen == 2
			? ((uint64_t)head * total_tokens + (uint64_t)tok0) * key_dim
			: ((uint64_t)tok0 * hv + head) * key_dim;
		const uint64_t vOffset = is_varlen == 2
			? ((uint64_t)head * total_tokens + (uint64_t)tok0) * value_dim
			: ((uint64_t)tok0 * hv + head) * value_dim;
		g_gm.SetGlobalBuffer((__gm__ T *)g_base + gOffset);
		q_gm.SetGlobalBuffer((__gm__ T *)q_base + qkOffset);
		k_gm.SetGlobalBuffer((__gm__ T *)k_base + qkOffset);
		v_gm.SetGlobalBuffer((__gm__ T *)v_base + vOffset);
		// beta 由 host 整调用一次转置为头连续 [HV, T_total](
		// 淘汰标量环):本头行基 = head*T_total + tok0
		b_gm.SetGlobalBuffer((__gm__ T *)b_base + (uint64_t)head*total_tokens + (uint64_t)tok0);
		dt_gm.SetGlobalBuffer((__gm__ float *)dt_base + key_dim*head);
		tmp_gm.SetGlobalBuffer((__gm__ T *)tmp_base + ((uint64_t)tok0*hv + head)*value_dim);
		o_gm.SetGlobalBuffer((__gm__ T *)o_base + ((uint64_t)tok0*hv + head)*value_dim);
		if (has_istate) {
			istate_gm.SetGlobalBuffer((__gm__ float *)istate_base + key_dim*value_dim*fh + (uint64_t)subid*(key_dim/2)*value_dim);
		}
		if (out_fstate) {
			fstate_gm.SetGlobalBuffer((__gm__ float *)fstate_base + key_dim*value_dim*fh + (uint64_t)subid*(key_dim/2)*value_dim);
		}
		a = a_gm.GetValue(head);
		a = -a;
		// exp(A_log) 下溢成 +0(A_log < -87.3)时 a = -0:-inf 尾行填充
		// 会算 0×(-inf)=NaN。钳到最小正规负数即可(-FLT_MIN × -inf = +inf,仍得
		// step_gate=-0),对有效行的扰动 ~1e-38 量级,可忽略。
		if (a == 0.0f) {
			a = -1.17549435e-38f;
		}
		if (gate_flags & 1) {
			g32_gm.SetGlobalBuffer((__gm__ float *)g_base + gOffset);
		}
		if (gate_flags & 2) {
			b32_gm.SetGlobalBuffer((__gm__ float *)b_base + (uint64_t)head*total_tokens + (uint64_t)tok0);
		}
		if (!task_empty) {
			gm_ub_single<1, key_dim, 4>(dt_ub, dt_gm);
		}
		gqkdb_cnt = 0; v_cnt = 0; out_cnt = 0; arrive_cnt = 0;
	}

	// 空序列收尾:不进计算流水。final_state 语义 = initial(或 0):
	//  - 有 initial_state:host(aclnn)已整体预填 fs=istate,kernel 不写;
	// 注:核内"读 istate 再写 fs"在空任务上下文实测不稳定(部分元素残留未写,
	// 因此 istate 直通放在 host 侧一次 ViewCopy。
	__aicore__ inline void WriteEmptyFinalState()
	{
		if (!out_fstate || has_istate) {
			return;
		}
		AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[0]), state_half_elems);
		AscendC::Duplicate<float>(stage, 0.0f, state_half_elems);
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::DataCopy(fstate_gm[0], stage, state_half_elems);
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
	}
	template<bool varlen>
	__aicore__ inline void BindTask(uint32_t fh)
	{
		if constexpr (varlen) {
			BindTaskVarlen(fh);
			return;
		}
		task_empty = 0;
		const uint32_t bias_head = hv_real ? (uint32_t)(fh % hv_real) : fh;
		const uint32_t qk_head = fh / qk_group;
		g_gm.SetGlobalBuffer((__gm__ T *)g_base + seq_len*key_dim*fh);
		q_gm.SetGlobalBuffer((__gm__ T *)q_base + seq_len*key_dim*qk_head);
		k_gm.SetGlobalBuffer((__gm__ T *)k_base + seq_len*key_dim*qk_head);
		v_gm.SetGlobalBuffer((__gm__ T *)v_base + seq_len*value_dim*fh);
		b_gm.SetGlobalBuffer((__gm__ T *)b_base + seq_len*fh);
		dt_gm.SetGlobalBuffer((__gm__ float *)dt_base + key_dim*bias_head);  // 每头 key_dim 个 dt_bias(按通道跳,不是按 token 组)
		tmp_gm.SetGlobalBuffer((__gm__ T *)tmp_base + seq_len*value_dim*fh);
		o_gm.SetGlobalBuffer((__gm__ T *)o_base + seq_len*value_dim*fh);
		// state GM 按任务 + k 半幅寻址:[B*HV,K,V] fp32,本 AIV 取 k∈[subid*64,+64)
		if (has_istate) {
			istate_gm.SetGlobalBuffer((__gm__ float *)istate_base + key_dim*value_dim*fh + (uint64_t)subid*(key_dim/2)*value_dim);
		}
		if (out_fstate) {
			fstate_gm.SetGlobalBuffer((__gm__ float *)fstate_base + key_dim*value_dim*fh + (uint64_t)subid*(key_dim/2)*value_dim);
		}
		a = a_gm.GetValue(bias_head);
		a = -a;
		// 同上:exp(A_log) 下溢成 +0 时钳到最小正规负数,防 -inf 尾行出 NaN。
		if (a == 0.0f) {
			a = -1.17549435e-38f;
		}
		if (gate_flags & 1) {
			g32_gm.SetGlobalBuffer((__gm__ float *)g_base + seq_len*key_dim*fh);
		}
		if (gate_flags & 2) {
			b32_gm.SetGlobalBuffer((__gm__ float *)b_base + seq_len*fh);
		}
		gm_ub_single<1, key_dim, 4>(dt_ub, dt_gm);
		gqkdb_cnt = 0; v_cnt = 0; out_cnt = 0; arrive_cnt = 0;
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_single(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> dst = l1.template ReinterpretCast<EleT>();
		AscendC::LocalTensor<EleT> src = ub.template ReinterpretCast<EleT>();
		AscendC::DataCopy(dst, src, m * n);   // 原 copy_ubuf_to_cbuf 1 burst
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_src_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> dst = l1.template ReinterpretCast<EleT>();
		AscendC::LocalTensor<EleT> src = ub.template ReinterpretCast<EleT>();
		AscendC::DataCopyParams p;   // 原 cfg:bursts=(n*es)>>5, len=m(32B), srcGap=1, dstGap=0
		p.blockCount = (n * element_size) >> 5;
		p.blockLen = m;
		p.srcStride = 1;
		p.dstStride = 0;
		AscendC::DataCopy(dst, src, p);
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_dst_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> dst = l1.template ReinterpretCast<EleT>();
		AscendC::LocalTensor<EleT> src = ub.template ReinterpretCast<EleT>();
		AscendC::DataCopyParams p;   // 原 cfg:bursts=(n*es)>>5, len=m, srcGap=0, dstGap=m
		p.blockCount = (n * element_size) >> 5;
		p.blockLen = m;
		p.srcStride = 0;
		p.dstStride = m;
		AscendC::DataCopy(dst, src, p);
	}

	template<uint64_t m, uint64_t n, uint64_t element_size>
	__aicore__ inline void ub_l1_src_strided_dst_strided_nz(const AscendC::LocalTensor<uint8_t> &ub, const AscendC::LocalTensor<uint8_t> &l1)
	{
		using EleT = typename AscendC::Std::conditional<element_size == 4, float, T>::type;
		AscendC::LocalTensor<EleT> dst = l1.template ReinterpretCast<EleT>();
		AscendC::LocalTensor<EleT> src = ub.template ReinterpretCast<EleT>();
		AscendC::DataCopyParams p;   // 原 cfg:bursts=(n*es)>>5, len=m, srcGap=1, dstGap=m
		p.blockCount = (n * element_size) >> 5;
		p.blockLen = m;
		p.srcStride = 1;
		p.dstStride = m;
		AscendC::DataCopy(dst, src, p);
	}

	template<uint32_t id, bool gate_f32, bool beta_f32, bool varlen, bool tail>
	__aicore__ inline void vec_prepare_g3(__ubuf__ void *gate_sc1_ub, __ubuf__ void *gamma_ub, __ubuf__ void *beta_ub, __ubuf__ void *qq2_ub, __ubuf__ void *kk2_ub)
	{
		// if(gqkdb_cnt == 0) {
		// 	for(int i = 0; i < 2; i ++) {
		// 		wait_flag(PIPE_V, PIPE_MTE2, i);

		// 		gm_ub_single<32, 128, 2>(g_ub[id], g_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 		gm_ub_single<32, 128, 2>(q_ub[id], q_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 		gm_ub_single<32, 128, 2>(k_ub[id], k_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 		gm_ub_single<1, 32, 2>(b_ub[id], b_gm[1*64*gqkdb_cnt + 1*32*subid]);
		// 		gm_ub_single<1, 32, 4>(dt_ub[id], dt_gm[1*64*gqkdb_cnt + 1*32*subid]);

		// 		gqkdb_cnt ++;

		// 		set_flag(PIPE_MTE2, PIPE_V, i);
		// 	}
		// }

		// 尾块:本 chunk 的头内行基 = tokens_per_prepare_pair*gqkdb_cnt + chunk_size*subid(计数器推出,与流水提前量无关)
		const int64_t row_base = tokens_per_prepare_pair*(int64_t)gqkdb_cnt + chunk_size*(int64_t)subid;
		uint32_t valid_rows = chunk_size;
		if (tail && has_tail_) {
			const int64_t rows_left = (int64_t)seq_len - row_base;
			valid_rows = rows_left <= 0 ? 0 : (rows_left >= chunk_size ? chunk_size : (uint32_t)rows_left);
		}
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(id);

		if constexpr (varlen) {
			// token-major packed 布局:g/q/k 行宽按各自 dtype 取 32B 整数倍,行距 = 头数*行宽;
			// 尾块只装有效行(无效行由下方 raw 填充兜底);beta 走 DataCopyPad。
			if (valid_rows > 0) {
				AscendC::LocalTensor<T> g_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
				AscendC::LocalTensor<float> g32_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
				AscendC::LocalTensor<T> q_t(AscendC::TPosition::VECCALC, KdaAddrV(q_ub[id]), chunk_size*key_dim);
				AscendC::LocalTensor<T> k_t(AscendC::TPosition::VECCALC, KdaAddrV(k_ub[id]), chunk_size*key_dim);
				AscendC::DataCopyParams sp;
				sp.blockCount = (uint16_t)valid_rows;
				sp.dstStride = 0;
				if constexpr (gate_f32) {
					sp.blockLen = key_dim*sizeof(float)/block_bytes;
					sp.srcStride = is_varlen == 2 ? 0 : (uint32_t)(hv_real - 1)*key_dim*sizeof(float)/block_bytes;
					AscendC::DataCopy(g32_t, g32_gm[row_base*(is_varlen == 2 ? key_dim : hv_real*key_dim)], sp);
				} else {
					sp.blockLen = key_dim*sizeof(T)/block_bytes;
					sp.srcStride = is_varlen == 2 ? 0 : (uint32_t)(hv_real - 1)*key_dim*sizeof(T)/block_bytes;
					AscendC::DataCopy(g_t, g_gm[row_base*(is_varlen == 2 ? key_dim : hv_real*key_dim)], sp);
				}
				sp.blockLen = key_dim*sizeof(T)/block_bytes;
				sp.srcStride = is_varlen == 2 ? 0 : (uint32_t)(nheads_qk - 1)*key_dim*sizeof(T)/block_bytes;
				AscendC::DataCopy(q_t, q_gm[row_base*(is_varlen == 2 ? key_dim : nheads_qk*key_dim)], sp);
				AscendC::DataCopy(k_t, k_gm[row_base*(is_varlen == 2 ? key_dim : nheads_qk*key_dim)], sp);
				// beta 由 host 转置为头连续 [HV,T_total];任意字节对齐走 DataCopyPad
				AscendC::DataCopyParams bp;
				bp.blockCount = 1; bp.srcStride = 0; bp.dstStride = 0;
				AscendC::DataCopyPadParams pad{false, 0, 0, 0};
				if constexpr (beta_f32) {
					AscendC::LocalTensor<float> b32_t(AscendC::TPosition::VECCALC, KdaAddrV(b_ub[id]), chunk_size);
					bp.blockLen = (uint16_t)(valid_rows*sizeof(float));
					AscendC::DataCopyPad(b32_t, b32_gm[row_base], bp, pad);
				} else {
					AscendC::LocalTensor<T> b_t(AscendC::TPosition::VECCALC, KdaAddrV(b_ub[id]), chunk_size);
					bp.blockLen = (uint16_t)(valid_rows*sizeof(T));
					AscendC::DataCopyPad(b_t, b_gm[row_base], bp, pad);
				}
			}
		} else if (valid_rows == chunk_size) {
			if constexpr (gate_f32) {
				gm_ub_single<chunk_size, key_dim, 4>(g_ub[id], g32_gm[row_base*key_dim]);
			} else {
				gm_ub_single<chunk_size, key_dim, 2>(g_ub[id], g_gm[row_base*key_dim]);
			}
			gm_ub_single<chunk_size, key_dim, 2>(q_ub[id], q_gm[row_base*key_dim]);
			gm_ub_single<chunk_size, key_dim, 2>(k_ub[id], k_gm[row_base*key_dim]);
			if constexpr (beta_f32) {
				gm_ub_single<1, chunk_size, 4>(b_ub[id], b32_gm[row_base]);
			} else {
				gm_ub_single<1, chunk_size, 2>(b_ub[id], b_gm[row_base]);
			}
		} else if (valid_rows > 0) {
			// 尾块:g 按 dtype 钳到 valid_rows;q/k 仍为 16 位;beta 任意字节长走 DataCopyPad
			AscendC::LocalTensor<T> g_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
			AscendC::LocalTensor<float> g32_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
			AscendC::LocalTensor<T> q_t(AscendC::TPosition::VECCALC, KdaAddrV(q_ub[id]), chunk_size*key_dim);
			AscendC::LocalTensor<T> k_t(AscendC::TPosition::VECCALC, KdaAddrV(k_ub[id]), chunk_size*key_dim);
			if constexpr (gate_f32) {
				AscendC::DataCopy(g32_t, g32_gm[row_base*key_dim], valid_rows*key_dim);
			} else {
				AscendC::DataCopy(g_t, g_gm[row_base*key_dim], valid_rows*key_dim);
			}
			AscendC::DataCopy(q_t, q_gm[row_base*key_dim], valid_rows*key_dim);
			AscendC::DataCopy(k_t, k_gm[row_base*key_dim], valid_rows*key_dim);
			AscendC::DataCopyParams bp;
			bp.blockCount = 1; bp.srcStride = 0; bp.dstStride = 0;
			AscendC::DataCopyPadParams pad{false, 0, 0, 0};
			if constexpr (beta_f32) {
				AscendC::LocalTensor<float> b32_t(AscendC::TPosition::VECCALC, KdaAddrV(b_ub[id]), chunk_size);
				bp.blockLen = (uint16_t)(valid_rows*sizeof(float));
				AscendC::DataCopyPad(b32_t, b32_gm[row_base], bp, pad);
			} else {
				AscendC::LocalTensor<T> b_t(AscendC::TPosition::VECCALC, KdaAddrV(b_ub[id]), chunk_size);
				bp.blockLen = (uint16_t)(valid_rows*sizeof(T));
				AscendC::DataCopyPad(b_t, b_gm[row_base], bp, pad);
			}
		}
		// valid_rows == 0:整块无效,跳过装载(下方 raw 清零兜底)

		gqkdb_cnt ++;

		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(id);

		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(id);

		if (valid_rows < chunk_size) {
			// raw 尾行填充(行粒度 256B 对齐;V 管,与随后变换同管顺序)。
			// g 填 -inf:safe gate 后 step_gate
			// = gate_scale/(1+exp(a*(-inf+dt))) = -0(gate_scale = lowerBound,
			// 契约 [-5,0) 恒负),无效行对 gamma/cumsum 天然零贡献——
			// (bf16 0xFF80 / fp32 0xFF800000),避免依赖 host 侧 inf 转换实现;
			// 变换段 Cast 到 fp32 保持 -inf 不变。
			// q/k 在装载阶段清零:
			// 归一化带 epsilon,0/sqrt(0+eps) = +0,无效行经归一化后自然保持 0,
			// 此处的补零是唯一清零位置;cube 侧(qplus/kplus/kminus)消费到的零与
			// 后续计算读取到的尾行均为零。
			if constexpr (gate_f32) {
				float ninf;
				*reinterpret_cast<uint32_t *>(&ninf) = 0xFF800000u;
				AscendC::LocalTensor<float> g32_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
				AscendC::Duplicate<float>(g32_t[valid_rows*key_dim], ninf, (chunk_size-valid_rows)*key_dim);
			} else {
				const uint16_t ninf_bits = 0xFF80u;
				T ninf;
				*reinterpret_cast<uint16_t *>(&ninf) = ninf_bits;
				AscendC::LocalTensor<T> g_t(AscendC::TPosition::VECCALC, KdaAddrV(g_ub[id]), chunk_size*key_dim);
				AscendC::Duplicate<T>(g_t[valid_rows*key_dim], ninf, (chunk_size-valid_rows)*key_dim);
			}
			AscendC::LocalTensor<T> q_t(AscendC::TPosition::VECCALC, KdaAddrV(q_ub[id]), chunk_size*key_dim);
			AscendC::LocalTensor<T> k_t(AscendC::TPosition::VECCALC, KdaAddrV(k_ub[id]), chunk_size*key_dim);
			AscendC::Duplicate<T>(q_t[valid_rows*key_dim], (T)0, (chunk_size-valid_rows)*key_dim);
			AscendC::Duplicate<T>(k_t[valid_rows*key_dim], (T)0, (chunk_size-valid_rows)*key_dim);
		}

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(id == 0 && subid == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// }
		// pipe_barrier(PIPE_ALL);

		__ubuf__ float *q_sq_sum = (__ubuf__ float *)tmp_ub[id];
		__ubuf__ float *k_sq_sum = (__ubuf__ float *)tmp_ub[id] + tmp_slot_elems;
		__ubuf__ float *q_sq_sum2 = (__ubuf__ float *)tmp_ub[id];
		__ubuf__ float *k_sq_sum2 = (__ubuf__ float *)tmp_ub[id] + tmp_slot_elems;
		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<uint16_t, MaskPattern::ALL>();

			{
				RegTensor<float> vreg_scale, vreg_gamma1, vreg_gamma2, vreg_dt1, vreg_dt2;

				UnalignReg ureg_q, ureg_k;

				Duplicate(vreg_scale, gate_scale);

				Duplicate(vreg_gamma1, (float)0.0);
				Duplicate(vreg_gamma2, (float)0.0);

				DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_dt1, vreg_dt2, (__ubuf__ float *)dt_ub);

				uint16_t offset_id = 0;

				for(uint16_t i = 0; i < (uint16_t)(chunk_half_rows / row_pair); i ++) {
					RegTensor<T> vreg_g1, vreg_g2, vreg_q1, vreg_q2, vreg_k1, vreg_k2;
					RegTensor<float> vreg_gg1, vreg_gg2, vreg_gg3, vreg_gg4,
						vreg_qq1, vreg_qq2, vreg_qq3, vreg_qq4, vreg_kk1, vreg_kk2, vreg_kk3, vreg_kk4;
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_g1, vreg_g2, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					if constexpr (gate_f32) {
						DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_gg1, vreg_gg2, (__ubuf__ float *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
						DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_gg3, vreg_gg4, (__ubuf__ float *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					} else {
						DataCopy(vreg_g1, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
						DataCopy(vreg_g2, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					}
					// DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_dt1, (__ubuf__ float *)dt_ub[id] + i*2 + offset_id*16*4);
					// DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_dt2, (__ubuf__ float *)dt_ub[id] + i*2 + offset_id*16*4 + 1);
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_q1, vreg_q2, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_k1, vreg_k2, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					DataCopy(vreg_q1, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
					DataCopy(vreg_q2, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					DataCopy(vreg_k1, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
					DataCopy(vreg_k2, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);

					if constexpr (!gate_f32) {
						Cast<float, T, f2bfZero>(vreg_gg1, vreg_g1, preg_all);
						Cast<float, T, f2bfOne>(vreg_gg2, vreg_g1, preg_all);
						Cast<float, T, f2bfZero>(vreg_gg3, vreg_g2, preg_all);
						Cast<float, T, f2bfOne>(vreg_gg4, vreg_g2, preg_all);
					}

					Cast<float, T, f2bfZero>(vreg_qq1, vreg_q1, preg_all);
					Cast<float, T, f2bfOne>(vreg_qq2, vreg_q1, preg_all);
					Cast<float, T, f2bfZero>(vreg_qq3, vreg_q2, preg_all);
					Cast<float, T, f2bfOne>(vreg_qq4, vreg_q2, preg_all);

					DataCopy((__ubuf__ float*)q_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_qq1, preg_all);
					DataCopy((__ubuf__ float*)q_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_qq3, preg_all);
					DataCopy((__ubuf__ float*)qq2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_qq2, preg_all);
					DataCopy((__ubuf__ float*)qq2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_qq4, preg_all);

					Cast<float, T, f2bfZero>(vreg_kk1, vreg_k1, preg_all);
					Cast<float, T, f2bfOne>(vreg_kk2, vreg_k1, preg_all);
					Cast<float, T, f2bfZero>(vreg_kk3, vreg_k2, preg_all);
					Cast<float, T, f2bfOne>(vreg_kk4, vreg_k2, preg_all);

					DataCopy((__ubuf__ float*)k_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_kk1, preg_all);
					DataCopy((__ubuf__ float*)k_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_kk3, preg_all);
					DataCopy((__ubuf__ float*)kk2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_kk2, preg_all);
					DataCopy((__ubuf__ float*)kk2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_kk4, preg_all);

					Add(vreg_gg1, vreg_gg1, vreg_dt1, preg_all);
					Add(vreg_gg2, vreg_gg2, vreg_dt2, preg_all);
					Add(vreg_gg3, vreg_gg3, vreg_dt1, preg_all);
					Add(vreg_gg4, vreg_gg4, vreg_dt2, preg_all);

					Mul(vreg_qq1, vreg_qq1, vreg_qq1, preg_all);
					Mul(vreg_qq2, vreg_qq2, vreg_qq2, preg_all);
					Mul(vreg_qq3, vreg_qq3, vreg_qq3, preg_all);
					Mul(vreg_qq4, vreg_qq4, vreg_qq4, preg_all);

					Muls(vreg_gg1, vreg_gg1, a, preg_all);
					Muls(vreg_gg2, vreg_gg2, a, preg_all);
					Muls(vreg_gg3, vreg_gg3, a, preg_all);
					Muls(vreg_gg4, vreg_gg4, a, preg_all);

					Exp(vreg_gg1, vreg_gg1, preg_all);
					Exp(vreg_gg2, vreg_gg2, preg_all);
					Exp(vreg_gg3, vreg_gg3, preg_all);
					Exp(vreg_gg4, vreg_gg4, preg_all);

					Mul(vreg_kk1, vreg_kk1, vreg_kk1, preg_all);
					Mul(vreg_kk2, vreg_kk2, vreg_kk2, preg_all);
					Mul(vreg_kk3, vreg_kk3, vreg_kk3, preg_all);
					Mul(vreg_kk4, vreg_kk4, vreg_kk4, preg_all);

					Adds(vreg_gg1, vreg_gg1, (float)1.0, preg_all);
					Adds(vreg_gg2, vreg_gg2, (float)1.0, preg_all);
					Adds(vreg_gg3, vreg_gg3, (float)1.0, preg_all);
					Adds(vreg_gg4, vreg_gg4, (float)1.0, preg_all);

					Div(vreg_gg1, vreg_scale, vreg_gg1, preg_all);
					Div(vreg_gg2, vreg_scale, vreg_gg2, preg_all);
					Div(vreg_gg3, vreg_scale, vreg_gg3, preg_all);
					Div(vreg_gg4, vreg_scale, vreg_gg4, preg_all);

					Add(vreg_qq1, vreg_qq1, vreg_qq2, preg_all);
					ReduceSum(vreg_qq1, vreg_qq1, preg_all);
					Add(vreg_qq3, vreg_qq3, vreg_qq4, preg_all);
					ReduceSum(vreg_qq3, vreg_qq3, preg_all);

					DataCopy((__ubuf__ float *)g_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_gg1, preg_all);
					DataCopy((__ubuf__ float *)g_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_gg3, preg_all);
					DataCopy((__ubuf__ float *)gate_sc1_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_gg2, preg_all);
					DataCopy((__ubuf__ float *)gate_sc1_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_gg4, preg_all);

					Add(vreg_kk1, vreg_kk1, vreg_kk2, preg_all);
					ReduceSum(vreg_kk1, vreg_kk1, preg_all);
					Add(vreg_kk3, vreg_kk3, vreg_kk4, preg_all);
					ReduceSum(vreg_kk3, vreg_kk3, preg_all);

					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)q_sq_sum, vreg_qq1, ureg_q, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)k_sq_sum, vreg_kk1, ureg_k, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)q_sq_sum, vreg_qq3, ureg_q, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)k_sq_sum, vreg_kk3, ureg_k, 1);

					Add(vreg_gg1, vreg_gg1, vreg_gg3, preg_all);
					Add(vreg_gg2, vreg_gg2, vreg_gg4, preg_all);
					Add(vreg_gamma1, vreg_gamma1, vreg_gg1, preg_all);
					Add(vreg_gamma2, vreg_gamma2, vreg_gg2, preg_all);

				}

				DataCopy((__ubuf__ float *)tmp_ub[id] + 2*tmp_slot_elems, vreg_gamma1, preg_all);
				DataCopy((__ubuf__ float *)tmp_ub[id] + 3*tmp_slot_elems, vreg_gamma2, preg_all);

				// vstas(ureg_q, (__ubuf__ float *&)q_sq_sum, 0, POST_UPDATE);
				// vstas(ureg_k, (__ubuf__ float *&)k_sq_sum, 0, POST_UPDATE);

				offset_id = 1;

				for(uint16_t i = 0; i < (uint16_t)(chunk_half_rows / row_pair); i ++) {
					RegTensor<T> vreg_g1, vreg_g2, vreg_q1, vreg_q2, vreg_k1, vreg_k2;
					RegTensor<float> vreg_gg1, vreg_gg2, vreg_gg3, vreg_gg4,
						vreg_qq1, vreg_qq2, vreg_qq3, vreg_qq4, vreg_kk1, vreg_kk2, vreg_kk3, vreg_kk4;
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_g1, vreg_g2, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					if constexpr (gate_f32) {
						DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_gg1, vreg_gg2, (__ubuf__ float *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
						DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_gg3, vreg_gg4, (__ubuf__ float *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					} else {
						DataCopy(vreg_g1, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
						DataCopy(vreg_g2, (__ubuf__ T *)g_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					}
					// DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_dt1, (__ubuf__ float *)dt_ub[id] + i*2 + offset_id*16);
					// DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_dt2, (__ubuf__ float *)dt_ub[id] + i*2 + offset_id*16 + 1);
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_q1, vreg_q2, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					// DataCopy<T, LoadDist::DIST_DINTLV_B16>(vreg_k1, vreg_k2, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim)*2);
					DataCopy(vreg_q1, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
					DataCopy(vreg_q2, (__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);
					DataCopy(vreg_k1, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim));
					DataCopy(vreg_k2, (__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + offset_id*(chunk_half_rows*key_dim) + key_dim);

					if constexpr (!gate_f32) {
						Cast<float, T, f2bfZero>(vreg_gg1, vreg_g1, preg_all);
						Cast<float, T, f2bfOne>(vreg_gg2, vreg_g1, preg_all);
						Cast<float, T, f2bfZero>(vreg_gg3, vreg_g2, preg_all);
						Cast<float, T, f2bfOne>(vreg_gg4, vreg_g2, preg_all);
					}

					Cast<float, T, f2bfZero>(vreg_qq1, vreg_q1, preg_all);
					Cast<float, T, f2bfOne>(vreg_qq2, vreg_q1, preg_all);
					Cast<float, T, f2bfZero>(vreg_qq3, vreg_q2, preg_all);
					Cast<float, T, f2bfOne>(vreg_qq4, vreg_q2, preg_all);

					DataCopy((__ubuf__ float*)q_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_qq1, preg_all);
					DataCopy((__ubuf__ float*)q_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_qq3, preg_all);
					DataCopy((__ubuf__ float*)qq2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_qq2, preg_all);
					DataCopy((__ubuf__ float*)qq2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_qq4, preg_all);

					Cast<float, T, f2bfZero>(vreg_kk1, vreg_k1, preg_all);
					Cast<float, T, f2bfOne>(vreg_kk2, vreg_k1, preg_all);
					Cast<float, T, f2bfZero>(vreg_kk3, vreg_k2, preg_all);
					Cast<float, T, f2bfOne>(vreg_kk4, vreg_k2, preg_all);

					DataCopy((__ubuf__ float*)k_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_kk1, preg_all);
					DataCopy((__ubuf__ float*)k_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_kk3, preg_all);
					DataCopy((__ubuf__ float*)kk2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_kk2, preg_all);
					DataCopy((__ubuf__ float*)kk2_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_kk4, preg_all);

					Add(vreg_gg1, vreg_gg1, vreg_dt1, preg_all);
					Add(vreg_gg2, vreg_gg2, vreg_dt2, preg_all);
					Add(vreg_gg3, vreg_gg3, vreg_dt1, preg_all);
					Add(vreg_gg4, vreg_gg4, vreg_dt2, preg_all);

					Mul(vreg_qq1, vreg_qq1, vreg_qq1, preg_all);
					Mul(vreg_qq2, vreg_qq2, vreg_qq2, preg_all);
					Mul(vreg_qq3, vreg_qq3, vreg_qq3, preg_all);
					Mul(vreg_qq4, vreg_qq4, vreg_qq4, preg_all);

					Muls(vreg_gg1, vreg_gg1, a, preg_all);
					Muls(vreg_gg2, vreg_gg2, a, preg_all);
					Muls(vreg_gg3, vreg_gg3, a, preg_all);
					Muls(vreg_gg4, vreg_gg4, a, preg_all);

					Exp(vreg_gg1, vreg_gg1, preg_all);
					Exp(vreg_gg2, vreg_gg2, preg_all);
					Exp(vreg_gg3, vreg_gg3, preg_all);
					Exp(vreg_gg4, vreg_gg4, preg_all);

					Mul(vreg_kk1, vreg_kk1, vreg_kk1, preg_all);
					Mul(vreg_kk2, vreg_kk2, vreg_kk2, preg_all);
					Mul(vreg_kk3, vreg_kk3, vreg_kk3, preg_all);
					Mul(vreg_kk4, vreg_kk4, vreg_kk4, preg_all);

					Adds(vreg_gg1, vreg_gg1, (float)1.0, preg_all);
					Adds(vreg_gg2, vreg_gg2, (float)1.0, preg_all);
					Adds(vreg_gg3, vreg_gg3, (float)1.0, preg_all);
					Adds(vreg_gg4, vreg_gg4, (float)1.0, preg_all);

					Div(vreg_gg1, vreg_scale, vreg_gg1, preg_all);
					Div(vreg_gg2, vreg_scale, vreg_gg2, preg_all);
					Div(vreg_gg3, vreg_scale, vreg_gg3, preg_all);
					Div(vreg_gg4, vreg_scale, vreg_gg4, preg_all);

					Add(vreg_qq1, vreg_qq1, vreg_qq2, preg_all);
					ReduceSum(vreg_qq1, vreg_qq1, preg_all);
					Add(vreg_qq3, vreg_qq3, vreg_qq4, preg_all);
					ReduceSum(vreg_qq3, vreg_qq3, preg_all);

					DataCopy((__ubuf__ float *)g_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_gg1, preg_all);
					DataCopy((__ubuf__ float *)g_ub[id] + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_gg3, preg_all);
					DataCopy((__ubuf__ float *)gate_sc1_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels), vreg_gg2, preg_all);
					DataCopy((__ubuf__ float *)gate_sc1_ub + i*(row_pair*half_channels) + offset_id*(chunk_half_rows*half_channels) + half_channels, vreg_gg4, preg_all);

					Add(vreg_kk1, vreg_kk1, vreg_kk2, preg_all);
					ReduceSum(vreg_kk1, vreg_kk1, preg_all);
					Add(vreg_kk3, vreg_kk3, vreg_kk4, preg_all);
					ReduceSum(vreg_kk3, vreg_kk3, preg_all);

					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)q_sq_sum, vreg_qq1, ureg_q, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)k_sq_sum, vreg_kk1, ureg_k, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)q_sq_sum, vreg_qq3, ureg_q, 1);
					DataCopyUnAlign<float, PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)k_sq_sum, vreg_kk3, ureg_k, 1);

					Add(vreg_gg1, vreg_gg1, vreg_gg3, preg_all);
					Add(vreg_gg2, vreg_gg2, vreg_gg4, preg_all);
					Add(vreg_gamma1, vreg_gamma1, vreg_gg1, preg_all);
					Add(vreg_gamma2, vreg_gamma2, vreg_gg2, preg_all);

				}

				DataCopy((__ubuf__ float *)gamma_ub, vreg_gamma1, preg_all);
				DataCopy((__ubuf__ float *)gamma_ub + half_channels, vreg_gamma2, preg_all);

				vstas(ureg_q, (__ubuf__ float *&)q_sq_sum, 0, POST_UPDATE);
				vstas(ureg_k, (__ubuf__ float *&)k_sq_sum, 0, POST_UPDATE);

				{
					RegTensor<float> vreg_qsum, vreg_ksum, vreg_bb, vreg_ones, vreg_zeros;
					Duplicate(vreg_ones, (float)1.0);
					Duplicate(vreg_zeros, (float)0.0);
					if constexpr (beta_f32) {
						DataCopy(vreg_bb, (__ubuf__ float *)b_ub[id]);
					} else {
						// bf16 位技巧:bf16 = fp32 高 16 位,零扩展装载后 <<16 即得 fp32 位模式
						DataCopy<uint16_t, LoadDist::DIST_US_B16>((RegTensor<uint16_t> &)vreg_bb, (__ubuf__ uint16_t *)b_ub[id]);
						ShiftLefts((RegTensor<uint32_t> &)vreg_bb, (RegTensor<uint32_t> &)vreg_bb, (int16_t)bits_per_16b_elem, preg_all);
					}
					FusedExpSub(vreg_bb, vreg_zeros, vreg_bb, preg_all);
					Adds(vreg_bb, vreg_bb, (float)1.0, preg_all);
					Div(vreg_bb, vreg_ones, vreg_bb, preg_all);

					LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
					DataCopy((__ubuf__ float *)beta_ub, vreg_bb, preg_all);
					DataCopy(vreg_qsum, (__ubuf__ float *)q_sq_sum2);
					DataCopy(vreg_ksum, (__ubuf__ float *)k_sq_sum2);
					// rstd = 1/sqrt(sum + epsilon);
					// 两个 pass 的平方和已先行汇入 tmp_ub,此处是唯一取 rstd 的单点
					Adds(vreg_qsum, vreg_qsum, epsilon, preg_all);
					Adds(vreg_ksum, vreg_ksum, epsilon, preg_all);
					Sqrt(vreg_qsum, vreg_qsum, preg_all);
					Sqrt(vreg_ksum, vreg_ksum, preg_all);
					Div(vreg_qsum, vreg_ones, vreg_qsum, preg_all);
					Div(vreg_ksum, vreg_ones, vreg_ksum, preg_all);
					DataCopy((__ubuf__ float *)q_sq_sum2, vreg_qsum, preg_all);
					DataCopy((__ubuf__ float *)k_sq_sum2, vreg_ksum, preg_all);
					LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
				}

				for(uint16_t i = 0; i < (uint16_t)(chunk_size / row_pair); i ++) {
					RegTensor<float> vreg_kk1, vreg_kk2, vreg_kk3, vreg_kk4, vreg_qq1, vreg_qq2, vreg_qq3, vreg_qq4;
					RegTensor<float> vreg_kn1, vreg_kn2, vreg_qn1, vreg_qn2;
					//DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_qq1, vreg_qq2, (__ubuf__ float *)q_ub[id] + i*(row_pair*half_channels));
					DataCopy(vreg_qq1, (__ubuf__ float *)q_ub[id] + i*(row_pair*half_channels));
					DataCopy(vreg_qq3, (__ubuf__ float *)q_ub[id] + i*(row_pair*half_channels) + half_channels);
					DataCopy(vreg_qq2, (__ubuf__ float *)qq2_ub + i*(row_pair*half_channels));
					DataCopy(vreg_qq4, (__ubuf__ float *)qq2_ub + i*(row_pair*half_channels) + half_channels);
					//DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_qq3, vreg_qq4, (__ubuf__ float *)qq2_ub + i*(row_pair*half_channels));
					//DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_kk1, vreg_kk2, (__ubuf__ float *)k_ub[id] + i*(row_pair*half_channels));
					DataCopy(vreg_kk1, (__ubuf__ float *)k_ub[id] + i*(row_pair*half_channels));
					DataCopy(vreg_kk3, (__ubuf__ float *)k_ub[id] + i*(row_pair*half_channels) + half_channels);
					DataCopy(vreg_kk2, (__ubuf__ float *)kk2_ub + i*(row_pair*half_channels));
					DataCopy(vreg_kk4, (__ubuf__ float *)kk2_ub + i*(row_pair*half_channels) + half_channels);
					//DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_kk3, vreg_kk4, (__ubuf__ float *)kk2_ub + i*(row_pair*half_channels));

					DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_qn1, (__ubuf__ float *)q_sq_sum2 + i*row_pair);
					DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_qn2, (__ubuf__ float *)q_sq_sum2 + i*row_pair + 1);
					DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_kn1, (__ubuf__ float *)k_sq_sum2 + i*row_pair);
					DataCopy<float, LoadDist::DIST_BRC_B32>(vreg_kn2, (__ubuf__ float *)k_sq_sum2 + i*row_pair + 1);

					Mul(vreg_qq1, vreg_qq1, vreg_qn1, preg_all);
					Mul(vreg_qq2, vreg_qq2, vreg_qn1, preg_all);
					Mul(vreg_qq3, vreg_qq3, vreg_qn2, preg_all);
					Mul(vreg_qq4, vreg_qq4, vreg_qn2, preg_all);

					Mul(vreg_kk1, vreg_kk1, vreg_kn1, preg_all);
					Mul(vreg_kk2, vreg_kk2, vreg_kn1, preg_all);
					Mul(vreg_kk3, vreg_kk3, vreg_kn2, preg_all);
					Mul(vreg_kk4, vreg_kk4, vreg_kn2, preg_all);

					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qq1, vreg_qq1, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_qq2, vreg_qq2, preg_all);
					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qq3, vreg_qq3, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_qq4, vreg_qq4, preg_all);

					Or((RegTensor<uint16_t>&)vreg_qq1, (RegTensor<uint16_t>&)vreg_qq1, (RegTensor<uint16_t>&)vreg_qq2, preg_all);
					Or((RegTensor<uint16_t>&)vreg_qq3, (RegTensor<uint16_t>&)vreg_qq3, (RegTensor<uint16_t>&)vreg_qq4, preg_all);

					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_kk1, vreg_kk1, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_kk2, vreg_kk2, preg_all);
					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_kk3, vreg_kk3, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_kk4, vreg_kk4, preg_all);

					Or((RegTensor<uint16_t>&)vreg_kk1, (RegTensor<uint16_t>&)vreg_kk1, (RegTensor<uint16_t>&)vreg_kk2, preg_all);
					Or((RegTensor<uint16_t>&)vreg_kk3, (RegTensor<uint16_t>&)vreg_kk3, (RegTensor<uint16_t>&)vreg_kk4, preg_all);

					DataCopy((__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim), (RegTensor<T> &)vreg_qq1, preg_all);
					DataCopy((__ubuf__ T *)q_ub[id] + i*(row_pair*key_dim) + key_dim, (RegTensor<T> &)vreg_qq3, preg_all);
					DataCopy((__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim), (RegTensor<T> &)vreg_kk1, preg_all);
					DataCopy((__ubuf__ T *)k_ub[id] + i*(row_pair*key_dim) + key_dim, (RegTensor<T> &)vreg_kk3, preg_all);
				}
			}
		}
		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(id == 0 && subid == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<32,64,4>(tmp_gm, gate_sc1_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// }
		// pipe_barrier(PIPE_ALL);

		if (valid_rows < chunk_size) {
			// 变换后修正只剩 beta:beta_eff 无效车道清零(sigmoid(0)=0.5/陈旧值,
			// Select 位选杀 NaN)。gamma 无需修正:raw g 尾行已填 -inf,无效行
			// step_gate 恒 -0;q/k 无需二次清零:装载期补零 + epsilon
			// 归一化天然归零。
			FixTailBetaF32(beta_ub, (float)valid_rows);
		}
	}

	template<uint32_t id>
	__aicore__ inline void vec_prepare_qplus_kplus_kminus_kres(__ubuf__ void *qplus_ub, __ubuf__ void *qplusb_ub, __ubuf__ void *kplus_ub, __ubuf__ void *kplusb_ub,
		__ubuf__ void *kminus_ub, __ubuf__ void *kres_ub, __ubuf__ void *gate_sc_ub, __ubuf__ void *gamma_ub)
	{
		uint32_t scale_bits = (uint32_t)scale16 << bits_per_16b_elem;
		float scale = *((float *)&scale_bits);

		__VEC_SCOPE__{
			RegTensor<float> vreg_scale;
			RegTensor<float> vreg_gs1, vreg_gs2, vreg_gss3, vreg_gss4, vreg_ngs1_exp, vreg_ngs2_exp, vreg_ngss3_exp, vreg_ngss4_exp;
			RegTensor<float> vreg_gamma_exp1, vreg_gamma_exp2, vreg_gamma_exp3, vreg_zero, vreg_gs3, vreg_gs4;
			RegTensor<float> vreg_gamma_exp1_f32_low, vreg_gamma_exp1_f32_high;

			MaskReg preg_all = CreateMask<uint16_t, MaskPattern::ALL>();

			Duplicate(vreg_zero, (float)0.0);
			Duplicate(vreg_scale, scale);
			Duplicate(vreg_gs1, (float)0.0);
			Duplicate(vreg_gs2, (float)0.0);
			Duplicate(vreg_gss3, (float)0.0);
			Duplicate(vreg_gss4, (float)0.0);

			{
				DataCopy(vreg_gs3, (__ubuf__ float *)tmp_ub[id] + 2*tmp_slot_elems);
				DataCopy(vreg_gs4, (__ubuf__ float *)tmp_ub[id] + 3*tmp_slot_elems);
				Exp(vreg_gamma_exp1, vreg_gs3, preg_all);
				Exp(vreg_gamma_exp2, vreg_gs4, preg_all);
				Adds(vreg_gamma_exp1_f32_low, vreg_gamma_exp1, (float)0.0, preg_all);
				Adds(vreg_gamma_exp1_f32_high, vreg_gamma_exp2, (float)0.0, preg_all);

				DataCopy(vreg_gamma_exp2, (__ubuf__ float *)gamma_ub);
				DataCopy(vreg_gamma_exp3, (__ubuf__ float *)gamma_ub + half_channels);

				Exp(vreg_ngss3_exp, vreg_gamma_exp2, preg_all);
				Exp(vreg_ngss4_exp, vreg_gamma_exp3, preg_all);
				Interleave((RegTensor<uint32_t> &)vreg_ngss3_exp, (RegTensor<uint32_t> &)vreg_ngss4_exp, (RegTensor<uint32_t> &)vreg_ngss3_exp, (RegTensor<uint32_t> &)vreg_ngss4_exp);

				DataCopy((__ubuf__ float *)gamma_ub, vreg_ngss3_exp, preg_all);
				DataCopy((__ubuf__ float *)gamma_ub + half_channels, vreg_ngss4_exp, preg_all);

				FusedExpSub(vreg_gamma_exp2, vreg_gamma_exp2, vreg_gs3, preg_all);
				FusedExpSub(vreg_gamma_exp3, vreg_gamma_exp3, vreg_gs4, preg_all);

			}

			for(uint16_t i = 0; i < (uint16_t)chunk_half_rows; i ++) {
				RegTensor<T> vreg_kk1, vreg_kk2, vreg_qq1, vreg_qq2;
				RegTensor<float> vreg_gg1, vreg_gg2, vreg_gg3, vreg_gg4;
				RegTensor<float> vreg_gs1_exp, vreg_gs2_exp, vreg_gss3_exp, vreg_gss4_exp;
				RegTensor<float> vreg_kk1_f32_low, vreg_kk1_f32_high;
				RegTensor<float> vreg_kk2_f32_low, vreg_kk2_f32_high;
				RegTensor<float> vreg_qq1_f32_low, vreg_qq1_f32_high;
				RegTensor<float> vreg_qq2_f32_low, vreg_qq2_f32_high;
				DataCopy(vreg_kk1, (__ubuf__ T *)k_ub[id] + i*key_dim);
				DataCopy(vreg_kk2, (__ubuf__ T *)k_ub[id] + i*key_dim + chunk_half_rows*key_dim);

				Cast<float, T, bf2fZero>(vreg_kk1_f32_low, vreg_kk1, preg_all);
				Cast<float, T, bf2fOne>(vreg_kk1_f32_high, vreg_kk1, preg_all);
				Cast<float, T, bf2fZero>(vreg_kk2_f32_low, vreg_kk2, preg_all);
				Cast<float, T, bf2fOne>(vreg_kk2_f32_high, vreg_kk2, preg_all);

				DataCopy(vreg_gg1, (__ubuf__ float *)g_ub[id] + i*half_channels);
				DataCopy(vreg_gg2, (__ubuf__ float *)gate_sc_ub + i*half_channels);
				DataCopy(vreg_gg3, (__ubuf__ float *)g_ub[id] + i*half_channels + chunk_half_rows*half_channels);
				DataCopy(vreg_gg4, (__ubuf__ float *)gate_sc_ub + i*half_channels + chunk_half_rows*half_channels);

				Add(vreg_gs1, vreg_gg1, vreg_gs1, preg_all);
				Add(vreg_gs2, vreg_gg2, vreg_gs2, preg_all);
				// Add(vreg_gs3, vreg_gg3, vreg_gs3, preg_all);
				// Add(vreg_gs4, vreg_gg4, vreg_gs4, preg_all);
				Add(vreg_gss3, vreg_gg3, vreg_gss3, preg_all);
				Add(vreg_gss4, vreg_gg4, vreg_gss4, preg_all);

				FusedExpSub(vreg_ngs1_exp, vreg_zero, vreg_gs1, preg_all);
				FusedExpSub(vreg_ngs2_exp, vreg_zero, vreg_gs2, preg_all);
				FusedExpSub(vreg_ngss3_exp, vreg_zero, vreg_gss3, preg_all);
				FusedExpSub(vreg_ngss4_exp, vreg_zero, vreg_gss4, preg_all);
				Exp(vreg_gs1_exp, vreg_gs1, preg_all);
				Exp(vreg_gs2_exp, vreg_gs2, preg_all);
				//Exp(vreg_gs3, vreg_gs3, preg_all);
				//Exp(vreg_gs4, vreg_gs4, preg_all);
				Exp(vreg_gss3_exp, vreg_gss3, preg_all);
				Exp(vreg_gss4_exp, vreg_gss4, preg_all);

				// kMinus/kRes 与 qPlus/kPlus 的全部中间计算均保持 FP32。
				Mul(vreg_gg1, vreg_kk1_f32_low, vreg_ngs1_exp, preg_all);
				Mul(vreg_gg3, vreg_kk1_f32_high, vreg_ngs2_exp, preg_all);
				Mul(vreg_gg2, vreg_kk2_f32_low, vreg_ngss3_exp, preg_all);
				Mul(vreg_gg4, vreg_kk2_f32_high, vreg_ngss4_exp, preg_all);

				RegTensor<float> vreg_kres1_f32_low, vreg_kres1_f32_high;
				RegTensor<float> vreg_kres2_f32_low, vreg_kres2_f32_high;
				Mul(vreg_kres1_f32_low, vreg_gg1, vreg_gamma_exp1_f32_low, preg_all);
				Mul(vreg_kres1_f32_high, vreg_gg3, vreg_gamma_exp1_f32_high, preg_all);
				Mul(vreg_kres2_f32_low, vreg_gg2, vreg_gamma_exp2, preg_all);
				Mul(vreg_kres2_f32_high, vreg_gg4, vreg_gamma_exp3, preg_all);
				Mul(vreg_kres1_f32_low, vreg_kres1_f32_low, vreg_gamma_exp2, preg_all);
				Mul(vreg_kres1_f32_high, vreg_kres1_f32_high, vreg_gamma_exp3, preg_all);

				RegTensor<T> vreg_pack;
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg3, vreg_gg3, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg1, vreg_gg1, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg1, (RegTensor<uint16_t> &)vreg_gg3, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kminus_ub + i*block_elems_16b, vreg_pack, 33, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg4, vreg_gg4, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg2, vreg_gg2, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg2, (RegTensor<uint16_t> &)vreg_gg4, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kminus_ub + i*block_elems_16b + chunk_half_rows*block_elems_16b, vreg_pack, 33, preg_all);

				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_kres1_f32_high, vreg_kres1_f32_high, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_kres1_f32_low, vreg_kres1_f32_low, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_kres1_f32_low, (RegTensor<uint16_t> &)vreg_kres1_f32_high, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kres_ub + i*block_elems_16b, vreg_pack, 33, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_kres2_f32_high, vreg_kres2_f32_high, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_kres2_f32_low, vreg_kres2_f32_low, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_kres2_f32_low, (RegTensor<uint16_t> &)vreg_kres2_f32_high, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kres_ub + i*block_elems_16b + chunk_half_rows*block_elems_16b, vreg_pack, 33, preg_all);

				// 前半 kPlus。
				Mul(vreg_gg1, vreg_kk1_f32_low, vreg_gs1_exp, preg_all);
				Mul(vreg_gg3, vreg_kk1_f32_high, vreg_gs2_exp, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg1, vreg_gg1, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg3, vreg_gg3, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg1, (RegTensor<uint16_t> &)vreg_gg3, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kplus_ub + i*block_elems_16b, vreg_pack, 33, preg_all);

				// 后半先保留 kPlusB，再乘 anchor 得到 kPlus。
				Mul(vreg_gg1, vreg_kk2_f32_low, vreg_gss3_exp, preg_all);
				Mul(vreg_gg3, vreg_kk2_f32_high, vreg_gss4_exp, preg_all);
				Mul(vreg_gg2, vreg_gg1, vreg_gamma_exp1_f32_low, preg_all);
				Mul(vreg_gg4, vreg_gg3, vreg_gamma_exp1_f32_high, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg1, vreg_gg1, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg3, vreg_gg3, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg1, (RegTensor<uint16_t> &)vreg_gg3, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kplusb_ub + i*block_elems_16b, vreg_pack, 17, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg2, vreg_gg2, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg4, vreg_gg4, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg2, (RegTensor<uint16_t> &)vreg_gg4, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)kplus_ub + i*block_elems_16b + chunk_half_rows*block_elems_16b, vreg_pack, 33, preg_all);

				// k 全部写回后再处理 q，缩短 FP32 寄存器跨 Stage 的生命周期。
				DataCopy(vreg_qq1, (__ubuf__ T *)q_ub[id] + i*key_dim);
				DataCopy(vreg_qq2, (__ubuf__ T *)q_ub[id] + i*key_dim + chunk_half_rows*key_dim);
				Cast<float, T, bf2fZero>(vreg_qq1_f32_low, vreg_qq1, preg_all);
				Cast<float, T, bf2fOne>(vreg_qq1_f32_high, vreg_qq1, preg_all);
				Cast<float, T, bf2fZero>(vreg_qq2_f32_low, vreg_qq2, preg_all);
				Cast<float, T, bf2fOne>(vreg_qq2_f32_high, vreg_qq2, preg_all);
				Mul(vreg_qq1_f32_low, vreg_qq1_f32_low, vreg_scale, preg_all);
				Mul(vreg_qq1_f32_high, vreg_qq1_f32_high, vreg_scale, preg_all);
				Mul(vreg_qq2_f32_low, vreg_qq2_f32_low, vreg_scale, preg_all);
				Mul(vreg_qq2_f32_high, vreg_qq2_f32_high, vreg_scale, preg_all);

				Mul(vreg_gg1, vreg_qq1_f32_low, vreg_gs1_exp, preg_all);
				Mul(vreg_gg3, vreg_qq1_f32_high, vreg_gs2_exp, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg1, vreg_gg1, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg3, vreg_gg3, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg1, (RegTensor<uint16_t> &)vreg_gg3, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)qplus_ub + i*block_elems_16b, vreg_pack, 33, preg_all);

				Mul(vreg_gg1, vreg_qq2_f32_low, vreg_gss3_exp, preg_all);
				Mul(vreg_gg3, vreg_qq2_f32_high, vreg_gss4_exp, preg_all);
				Mul(vreg_gg2, vreg_gg1, vreg_gamma_exp1_f32_low, preg_all);
				Mul(vreg_gg4, vreg_gg3, vreg_gamma_exp1_f32_high, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg1, vreg_gg1, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg3, vreg_gg3, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg1, (RegTensor<uint16_t> &)vreg_gg3, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)qplusb_ub + i*block_elems_16b, vreg_pack, 17, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_gg2, vreg_gg2, preg_all);
				Cast<T, float, f2bfOne>((RegTensor<T> &)vreg_gg4, vreg_gg4, preg_all);
				Or((RegTensor<uint16_t> &)vreg_pack, (RegTensor<uint16_t> &)vreg_gg2, (RegTensor<uint16_t> &)vreg_gg4, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)qplus_ub + i*block_elems_16b + chunk_half_rows*block_elems_16b, vreg_pack, 33, preg_all);

			}
		}

		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(id);



		// if(gqkdb_cnt < group_count*2) {
		// 	wait_flag(PIPE_V, PIPE_MTE2, id);

		// 	gm_ub_single<32, 128, 2>(g_ub[id], g_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 	gm_ub_single<32, 128, 2>(q_ub[id], q_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 	gm_ub_single<32, 128, 2>(k_ub[id], k_gm[64*128*gqkdb_cnt + 32*128*subid]);
		// 	gm_ub_single<1, 32, 2>(b_ub[id], b_gm[1*64*gqkdb_cnt + 1*32*subid]);
		// 	gm_ub_single<1, 32, 4>(dt_ub[id], dt_gm[1*64*gqkdb_cnt + 1*32*subid]);

		// 	gqkdb_cnt ++;

		// 	set_flag(PIPE_MTE2, PIPE_V, id);
		// }

	}

	__aicore__ inline void vec_tril(__ubuf__ void *masked_qk_ub, __ubuf__ void *l_ub, __ubuf__ void *kplus_kminus_ub, __ubuf__ void *qplus_kminus_ub, __ubuf__ void *beta_ub)
	{
		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(subid == 1 && arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<64,16,4>(tmp_gm, kplus_kminus_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);

		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<uint16_t, MaskPattern::ALL>();

			int16_t i = 0;
			{
				RegTensor<T> vreg_qk34;
				// RegTensor<float> vreg_kk3, vreg_kk4;
				Duplicate((RegTensor<uint16_t> &)vreg_qk34, (uint16_t)0);
				// Duplicate(vreg_kk3, (float)0.0);
				// Duplicate(vreg_kk4, (float)0.0);
				for(uint16_t j = 0; j < 2; j ++) {
					RegTensor<float> vreg_tril_mask1, vreg_tril_mask2;
					RegTensor<float> vreg_identity1, vreg_identity2;
					RegTensor<float> vreg_beta1, vreg_beta2;
					RegTensor<float> vreg_qk1, vreg_qk2, vreg_kk1, vreg_kk2;

					DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_beta1, (__ubuf__ float *)beta_ub + i*chunk_half_rows + j*(chunk_half_rows/2) + subid*chunk_size);
					Interleave((RegTensor<float> &)vreg_beta1, (RegTensor<float> &)vreg_beta2, (RegTensor<float> &)vreg_beta1, (RegTensor<float> &)vreg_beta1);

					DataCopy(vreg_kk1, (__ubuf__ float *)kplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len));
					DataCopy(vreg_kk2, (__ubuf__ float *)kplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len);

					DataCopy(vreg_qk1, (__ubuf__ float *)qplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len));
					DataCopy(vreg_qk2, (__ubuf__ float *)qplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len);

					Mul(vreg_kk1, vreg_kk1, vreg_beta1, preg_all);
					Mul(vreg_kk2, vreg_kk2, vreg_beta2, preg_all);

					DataCopy(vreg_tril_mask1, (__ubuf__ float *)strict_tril_mask + j*(8*fractal_len));
					DataCopy(vreg_tril_mask2, (__ubuf__ float *)strict_tril_mask + j*(8*fractal_len) + 4*fractal_len);

					Mul(vreg_kk1, vreg_kk1, vreg_tril_mask1, preg_all);
					Mul(vreg_kk2, vreg_kk2, vreg_tril_mask2, preg_all);

					DataCopy(vreg_identity1, (__ubuf__ float *)identity + j*(8*fractal_len));
					DataCopy(vreg_identity2, (__ubuf__ float *)identity + j*(8*fractal_len) + 4*fractal_len);

					Add(vreg_tril_mask1, vreg_tril_mask1, vreg_identity1, preg_all);
					Add(vreg_tril_mask2, vreg_tril_mask2, vreg_identity2, preg_all);

					// Add(vreg_kk1, vreg_kk1, vreg_identity1, preg_all);
					// Add(vreg_kk2, vreg_kk2, vreg_identity2, preg_all);

					Mul(vreg_qk1, vreg_qk1, vreg_tril_mask1, preg_all);
					Mul(vreg_qk2, vreg_qk2, vreg_tril_mask2, preg_all);

					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk1, vreg_qk1, preg_all);
					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk2, vreg_qk2, preg_all);

					DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + j*(8*fractal_len), vreg_kk1, preg_all);
					DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len, vreg_kk2, preg_all);
					// DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + 2*fractal_elems + j*(8*fractal_len), vreg_kk3, preg_all);
					// DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + 2*fractal_elems + j*(8*fractal_len) + 4*fractal_len, vreg_kk4, preg_all);
					DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*(3*fractal_elems) + j*(8*fractal_len)), (RegTensor<uint32_t> &)vreg_qk1, preg_all);
					DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len), (RegTensor<uint32_t> &)vreg_qk2, preg_all);
					DataCopy((__ubuf__ T *)masked_qk_ub + 2*fractal_elems + j*(8*fractal_len), vreg_qk34, preg_all);
				}
			}

			i = 1;
			for(uint16_t j = 0; j < 2; j ++) {
				RegTensor<float> vreg_tril_mask1, vreg_tril_mask2;
				RegTensor<float> vreg_identity1, vreg_identity2;
				RegTensor<float> vreg_beta1, vreg_beta2;
				RegTensor<float> vreg_qk1, vreg_qk2, vreg_qk3, vreg_qk4, vreg_kk1, vreg_kk2, vreg_kkk1, vreg_kkk2;

				DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_beta1, (__ubuf__ float *)beta_ub + i*chunk_half_rows + j*(chunk_half_rows/2) + subid*chunk_size);
				Interleave((RegTensor<float> &)vreg_beta1, (RegTensor<float> &)vreg_beta2, (RegTensor<float> &)vreg_beta1, (RegTensor<float> &)vreg_beta1);

				DataCopy(vreg_kk1, (__ubuf__ float *)kplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len));
				DataCopy(vreg_kk2, (__ubuf__ float *)kplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len);

				DataCopy(vreg_kkk1, (__ubuf__ float *)kplus_kminus_ub + i*16*16 + j*(8*fractal_len));
				DataCopy(vreg_kkk2, (__ubuf__ float *)kplus_kminus_ub + i*16*16 + j*(8*fractal_len) + 4*fractal_len);

				DataCopy(vreg_qk1, (__ubuf__ float *)qplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len));
				DataCopy(vreg_qk2, (__ubuf__ float *)qplus_kminus_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len);

				DataCopy(vreg_qk3, (__ubuf__ float *)qplus_kminus_ub + i*16*16*1 + j*(8*fractal_len));
				DataCopy(vreg_qk4, (__ubuf__ float *)qplus_kminus_ub + i*16*16*1 + j*(8*fractal_len) + 4*fractal_len);

				//Muls(vreg_kkk1, vreg_kkk1, (float)-1.0, preg_all);
				//Muls(vreg_kkk2, vreg_kkk2, (float)-1.0, preg_all);

				Mul(vreg_kk1, vreg_kk1, vreg_beta1, preg_all);
				Mul(vreg_kk2, vreg_kk2, vreg_beta2, preg_all);
				Mul(vreg_kkk1, vreg_kkk1, vreg_beta1, preg_all);
				Mul(vreg_kkk2, vreg_kkk2, vreg_beta2, preg_all);

				DataCopy(vreg_tril_mask1, (__ubuf__ float *)strict_tril_mask + j*(8*fractal_len));
				DataCopy(vreg_tril_mask2, (__ubuf__ float *)strict_tril_mask + j*(8*fractal_len) + 4*fractal_len);

				Mul(vreg_kk1, vreg_kk1, vreg_tril_mask1, preg_all);
				Mul(vreg_kk2, vreg_kk2, vreg_tril_mask2, preg_all);

				DataCopy(vreg_identity1, (__ubuf__ float *)identity + j*(8*fractal_len));
				DataCopy(vreg_identity2, (__ubuf__ float *)identity + j*(8*fractal_len) + 4*fractal_len);

				Add(vreg_tril_mask1, vreg_tril_mask1, vreg_identity1, preg_all);
				Add(vreg_tril_mask2, vreg_tril_mask2, vreg_identity2, preg_all);

				// Add(vreg_kk1, vreg_kk1, vreg_identity1, preg_all);
				// Add(vreg_kk2, vreg_kk2, vreg_identity2, preg_all);

				Mul(vreg_qk1, vreg_qk1, vreg_tril_mask1, preg_all);
				Mul(vreg_qk2, vreg_qk2, vreg_tril_mask2, preg_all);

				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk1, vreg_qk1, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk2, vreg_qk2, preg_all);

				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk3, vreg_qk3, preg_all);
				Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_qk4, vreg_qk4, preg_all);

				DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + j*(8*fractal_len), vreg_kk1, preg_all);
				DataCopy((__ubuf__ float *)l_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len, vreg_kk2, preg_all);
				DataCopy((__ubuf__ float *)l_ub + i*16*16 + j*(8*fractal_len), vreg_kkk1, preg_all);
				DataCopy((__ubuf__ float *)l_ub + i*16*16 + j*(8*fractal_len) + 4*fractal_len, vreg_kkk2, preg_all);
				DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*(3*fractal_elems) + j*(8*fractal_len)), (RegTensor<uint32_t> &)vreg_qk1, preg_all);
				DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*(3*fractal_elems) + j*(8*fractal_len) + 4*fractal_len), (RegTensor<uint32_t> &)vreg_qk2, preg_all);
				DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*16*16*1 + j*(8*fractal_len)), (RegTensor<uint32_t> &)vreg_qk3, preg_all);
				DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)((__ubuf__ T *)masked_qk_ub + i*16*16*1 + j*(8*fractal_len) + 4*fractal_len), (RegTensor<uint32_t> &)vreg_qk4, preg_all);
			}
		}

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(subid == 0 && arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<64,16,4>(tmp_gm, qplus_kminus_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(subid == 0 && arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<64,16,2>(tmp_gm, masked_qk_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);
	}

	__aicore__ inline void vec_tril_inv1(__ubuf__ void *m1_ub, __ubuf__ void *l_ub)
	{
		__ubuf__ float *l_ub1 = (__ubuf__ float *)l_ub + 1*fractal_len;
		__ubuf__ float *l_ub2 = (__ubuf__ float *)l_ub + 2*fractal_len;
		__ubuf__ float *l_ub3 = (__ubuf__ float *)l_ub + 3*fractal_len;
		__ubuf__ float *l_ub4 = (__ubuf__ float *)l_ub + 4*fractal_len;
		__ubuf__ float *id_ub1 = (__ubuf__ float *)identity + 0*fractal_len;
		__ubuf__ float *id_ub2 = (__ubuf__ float *)identity + 1*fractal_len;
		__ubuf__ float *id_ub3 = (__ubuf__ float *)identity + 2*fractal_len;
		__ubuf__ float *id_ub4 = (__ubuf__ float *)identity + 3*fractal_len;
		__ubuf__ uint32_t *m_ub1 = (__ubuf__ uint32_t *)m1_ub;
		// __ubuf__ float *m_ub2 = (__ubuf__ float *)m_ub + 1*fractal_len;
		// __ubuf__ float *m_ub3 = (__ubuf__ float *)m_ub + 2*fractal_len;
		// __ubuf__ float *m_ub4 = (__ubuf__ float *)m_ub + 3*fractal_len;
		__VEC_SCOPE__{
			//MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();
			//MaskReg preg_val = CreateMask<uint32_t, MaskPattern::Q>();
			AscendC::Reg::MaskReg preg_val = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::Q>();
			AscendC::Reg::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8, vreg9, vreg10, vreg11, vreg12, vreg13, vreg14, vreg15, vreg16;
			AscendC::Reg::RegTensor<float> vreg_s1, vreg_s2, vreg_s3, vreg_s4;

			// line 1
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg2, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg3, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg4, id_ub4, fp32_lanes);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg5, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg6, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg7, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg8, id_ub4, fp32_lanes);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg9, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg10, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg11, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg12, id_ub4, fp32_lanes);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg13, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg14, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg15, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg16, id_ub4, fp32_lanes);

			// // line 2
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg2, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg3, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg4, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg6, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			//vcvt((vector_bf16 &)vreg1, vreg1, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg1, fractal_len, preg_val);

			// // line 3

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg3, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg4, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg7, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			//vcvt((vector_bf16 &)vreg2, vreg2, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg2, fractal_len, preg_val);

			// // line 4

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg4, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg8, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			//vcvt((vector_bf16 &)vreg3, vreg3, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg3, fractal_len, preg_val);

			// // line 5

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg5, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg9, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			//vcvt((vector_bf16 &)vreg4, vreg4, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg4, fractal_len, preg_val);

			// // line 6

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg5, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg6, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg5, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			//vcvt((vector_bf16 &)vreg5, vreg5, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg5, fractal_len, preg_val);

			// // line 7

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg6, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg7, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg6, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			//vcvt((vector_bf16 &)vreg6, vreg6, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg6, fractal_len, preg_val);

			// // line 8

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg7, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg8, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg7, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			//vcvt((vector_bf16 &)vreg7, vreg7, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg7, fractal_len, preg_val);


			// // line 9

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg8, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg9, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg8, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			//vcvt((vector_bf16 &)vreg8, vreg8, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg8, fractal_len, preg_val);

			// // line 10

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg9, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg9, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			//vcvt((vector_bf16 &)vreg9, vreg9, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg9, fractal_len, preg_val);

			// // line 11

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg10, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg10, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			//vcvt((vector_bf16 &)vreg10, vreg10, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg10, fractal_len, preg_val);

			// // line 12

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg11, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg11, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			//vcvt((vector_bf16 &)vreg11, vreg11, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg11, fractal_len, preg_val);

			// // line 13

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg12, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			//vcvt((vector_bf16 &)vreg12, vreg12, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg12, fractal_len, preg_val);

			// // line 14

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg13, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			//vcvt((vector_bf16 &)vreg13, vreg13, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg13, fractal_len, preg_val);

			// // line 15

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg14, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			//vcvt((vector_bf16 &)vreg14, vreg14, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg14, fractal_len, preg_val);

			// // line 16

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg15, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			//vcvt((vector_bf16 &)vreg15, vreg15, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg15, fractal_len, preg_val);

			//vcvt((vector_bf16 &)vreg16, vreg16, preg_val, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_NORM_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg16, fractal_len, preg_val);
		}

	}

	__aicore__ inline void vec_tril_inv2(__ubuf__ void *m_ub, __ubuf__ void *m1_ub, __ubuf__ void *l_ub)
	{
		__ubuf__ float *l1_ub1 = (__ubuf__ float *)l_ub + fractal_elems + 0*fractal_len;
		__ubuf__ float *l1_ub2 = (__ubuf__ float *)l_ub + fractal_elems + 1*fractal_len;
		__ubuf__ float *l1_ub3 = (__ubuf__ float *)l_ub + fractal_elems + 2*fractal_len;
		__ubuf__ float *l1_ub4 = (__ubuf__ float *)l_ub + fractal_elems + 3*fractal_len;
		__ubuf__ float *l_ub1  = (__ubuf__ float *)l_ub + 3*fractal_elems + 1*fractal_len;
		__ubuf__ float *l_ub2  = (__ubuf__ float *)l_ub + 3*fractal_elems + 2*fractal_len;
		__ubuf__ float *l_ub3  = (__ubuf__ float *)l_ub + 3*fractal_elems + 3*fractal_len;
		__ubuf__ float *l_ub4  = (__ubuf__ float *)l_ub + 3*fractal_elems + 4*fractal_len;
		__ubuf__ float *id_ub1 = (__ubuf__ float *)identity + 0*fractal_len - fractal_len;
		__ubuf__ float *id_ub2 = (__ubuf__ float *)identity + 1*fractal_len - fractal_len;
		__ubuf__ float *id_ub3 = (__ubuf__ float *)identity + 2*fractal_len - fractal_len;
		__ubuf__ float *id_ub4 = (__ubuf__ float *)identity + 3*fractal_len - fractal_len;
		__ubuf__ uint32_t *m_ub0 = (__ubuf__ uint32_t *)m_ub;
		__ubuf__ uint32_t *m_ub1 = (__ubuf__ uint32_t *)m_ub + fractal_elems/2;
		__ubuf__ uint32_t *m_ub2 = (__ubuf__ uint32_t *)m_ub + 3*(fractal_elems/2) - fractal_len/2;
		// __ubuf__ float *m_ub2 = (__ubuf__ float *)m_ub + 1*fractal_len;
		// __ubuf__ float *m_ub3 = (__ubuf__ float *)m_ub + 2*fractal_len;
		// __ubuf__ float *m_ub4 = (__ubuf__ float *)m_ub + 3*fractal_len;
		__VEC_SCOPE__{
			//MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();
			//MaskReg preg_val = CreateMask<uint32_t, MaskPattern::Q>();
			AscendC::Reg::MaskReg preg_val = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::H>();
			AscendC::Reg::MaskReg preg_vq = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::Q>();
			AscendC::Reg::MaskReg preg_vb = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::VL32>();
			AscendC::Reg::MaskReg preg_vq2;
			pxor(preg_vq2, preg_vq, preg_val, preg_val);

			AscendC::Reg::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8, vreg9, vreg10, vreg11, vreg12, vreg13, vreg14, vreg15, vreg16;
			AscendC::Reg::RegTensor<float> vreg_s1, vreg_s2, vreg_s3, vreg_s4;
			AscendC::Reg::RegTensor<float> vreg_zero;

			vdup((AscendC::Reg::RegTensor<uint8_t> &)vreg_zero, (uint8_t)0, preg_val, MODE_ZEROING);

			// line 1
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg2, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg3, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg4, id_ub4, fp32_lanes);
			AscendC::Reg::Select(vreg1, vreg_zero, vreg1, preg_vq);
			AscendC::Reg::Select(vreg2, vreg_zero, vreg2, preg_vq);
			AscendC::Reg::Select(vreg3, vreg_zero, vreg3, preg_vq);
			AscendC::Reg::Select(vreg4, vreg_zero, vreg4, preg_vq);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg5, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg6, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg7, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg8, id_ub4, fp32_lanes);
			AscendC::Reg::Select(vreg5, vreg_zero, vreg5, preg_vq);
			AscendC::Reg::Select(vreg6, vreg_zero, vreg6, preg_vq);
			AscendC::Reg::Select(vreg7, vreg_zero, vreg7, preg_vq);
			AscendC::Reg::Select(vreg8, vreg_zero, vreg8, preg_vq);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg9, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg10, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg11, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg12, id_ub4, fp32_lanes);
			AscendC::Reg::Select(vreg9, vreg_zero, vreg9, preg_vq);
			AscendC::Reg::Select(vreg10, vreg_zero, vreg10, preg_vq);
			AscendC::Reg::Select(vreg11, vreg_zero, vreg11, preg_vq);
			AscendC::Reg::Select(vreg12, vreg_zero, vreg12, preg_vq);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg13, id_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg14, id_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg15, id_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg16, id_ub4, fp32_lanes);

			AscendC::Reg::Select(vreg13, vreg_zero, vreg13, preg_vq);
			AscendC::Reg::Select(vreg14, vreg_zero, vreg14, preg_vq);
			AscendC::Reg::Select(vreg15, vreg_zero, vreg15, preg_vq);
			AscendC::Reg::Select(vreg16, vreg_zero, vreg16, preg_vq);

			for(uint16_t i = 0; i < (uint16_t)fractal_len; i ++) {
				AscendC::Reg::RegTensor<float> vreg0;
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_NORM>(vreg0, (__ubuf__ float *&)m1_ub, fractal_len);
				AscendC::Reg::Select(vreg0, vreg0, vreg_zero, preg_vq);

				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l1_ub1, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l1_ub2, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l1_ub3, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l1_ub4, fp32_lanes);

				AscendC::Reg::Mul(vreg_s1, vreg0, vreg_s1, preg_val);
				AscendC::Reg::Mul(vreg_s2, vreg0, vreg_s2, preg_val);
				AscendC::Reg::Mul(vreg_s3, vreg0, vreg_s3, preg_val);
				AscendC::Reg::Mul(vreg_s4, vreg0, vreg_s4, preg_val);

				AscendC::Reg::Sub(vreg1, vreg1, vreg_s1, preg_val);
				AscendC::Reg::Sub(vreg2, vreg2, vreg_s2, preg_val);
				AscendC::Reg::Sub(vreg3, vreg3, vreg_s3, preg_val);
				AscendC::Reg::Sub(vreg4, vreg4, vreg_s4, preg_val);

				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l1_ub1, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l1_ub2, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l1_ub3, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l1_ub4, fp32_lanes);

				AscendC::Reg::Mul(vreg_s1, vreg0, vreg_s1, preg_val);
				AscendC::Reg::Mul(vreg_s2, vreg0, vreg_s2, preg_val);
				AscendC::Reg::Mul(vreg_s3, vreg0, vreg_s3, preg_val);
				AscendC::Reg::Mul(vreg_s4, vreg0, vreg_s4, preg_val);

				AscendC::Reg::Sub(vreg5, vreg5, vreg_s1, preg_val);
				AscendC::Reg::Sub(vreg6, vreg6, vreg_s2, preg_val);
				AscendC::Reg::Sub(vreg7, vreg7, vreg_s3, preg_val);
				AscendC::Reg::Sub(vreg8, vreg8, vreg_s4, preg_val);

				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l1_ub1, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l1_ub2, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l1_ub3, fp32_lanes);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l1_ub4, fp32_lanes);

				AscendC::Reg::Mul(vreg_s1, vreg0, vreg_s1, preg_val);
				AscendC::Reg::Mul(vreg_s2, vreg0, vreg_s2, preg_val);
				AscendC::Reg::Mul(vreg_s3, vreg0, vreg_s3, preg_val);
				AscendC::Reg::Mul(vreg_s4, vreg0, vreg_s4, preg_val);

				AscendC::Reg::Sub(vreg9, vreg9, vreg_s1, preg_val);
				AscendC::Reg::Sub(vreg10, vreg10, vreg_s2, preg_val);
				AscendC::Reg::Sub(vreg11, vreg11, vreg_s3, preg_val);
				AscendC::Reg::Sub(vreg12, vreg12, vreg_s4, preg_val);

				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l1_ub1, (-12)*16 + 1);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l1_ub2, (-12)*16 + 1);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l1_ub3, (-12)*16 + 1);
				AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l1_ub4, (-12)*16 + 1);

				AscendC::Reg::Mul(vreg_s1, vreg0, vreg_s1, preg_val);
				AscendC::Reg::Mul(vreg_s2, vreg0, vreg_s2, preg_val);
				AscendC::Reg::Mul(vreg_s3, vreg0, vreg_s3, preg_val);
				AscendC::Reg::Mul(vreg_s4, vreg0, vreg_s4, preg_val);

				AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
				AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
				AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
				AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

				AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg0, vreg0, preg_vq);
				vsts((AscendC::Reg::RegTensor<uint32_t> &)vreg_zero, m_ub0, fractal_elems, NORM_B32, preg_vb);
				AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub0, (AscendC::Reg::RegTensor<uint32_t> &)vreg0, block_words32, preg_vq);
			}

			// // line 2
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg2, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg3, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg4, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg6, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg1, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg1, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg1, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg1, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg1, vreg1, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg1, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg1, block_words32, preg_vq2);

			// // line 3

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg3, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg4, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg7, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg2, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg2, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg2, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg2, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg2, vreg2, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg2, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg2, block_words32, preg_vq2);

			// // line 4

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg4, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg5, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg8, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg3, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg3, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg3, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-12+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg3, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg3, vreg3, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg3, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg3, block_words32, preg_vq2);

			// // line 5

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg5, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg6, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg9, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg4, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg4, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg4, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg4, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg4, vreg4, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg4, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg4, block_words32, preg_vq2);

			// // line 6

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg5, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg6, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg7, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg5, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg5, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg5, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg5, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg5, vreg5, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg5, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg5, block_words32, preg_vq2);

			// // line 7

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg6, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg7, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg8, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg6, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg6, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg6, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg6, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg6, vreg6, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg6, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg6, block_words32, preg_vq2);

			// // line 8

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg7, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg8, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg9, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg7, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg7, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg7, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-8+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg7, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg7, vreg7, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg7, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg7, block_words32, preg_vq2);


			// // line 9

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, fp32_lanes);

			AscendC::Reg::Mul(vreg_s1, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg8, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg9, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg10, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg8, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg8, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg8, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg8, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg8, vreg8, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg8, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg8, block_words32, preg_vq2);

			// // line 10

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg9, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg9, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg10, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg11, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg9, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg9, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg9, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg9, vreg9, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg9, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg9, block_words32, preg_vq2);

			// // line 11

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg10, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg10, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg10, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg11, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg12, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg10, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg10, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg10, vreg10, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg10, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg10, block_words32, preg_vq2);

			// // line 12

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, fp32_lanes);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg11, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg11, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg11, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg11, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg12, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg13, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s4, preg_val);

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (-4+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg11, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg11, vreg11, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg11, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg11, block_words32, preg_vq2);

			// // line 13

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s4, l_ub4, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg12, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg12, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg12, vreg_s3, preg_val);
			AscendC::Reg::Mul(vreg_s4, vreg12, vreg_s4, preg_val);

			AscendC::Reg::Sub(vreg13, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg14, vreg14, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s3, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s4, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg12, vreg12, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg12, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg12, block_words32, preg_vq2);

			// // line 14

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s3, l_ub3, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg13, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg13, vreg_s2, preg_val);
			AscendC::Reg::Mul(vreg_s3, vreg13, vreg_s3, preg_val);

			AscendC::Reg::Sub(vreg14, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg15, vreg15, vreg_s2, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s3, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg13, vreg13, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg13, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg13, block_words32, preg_vq2);

			// // line 15

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);
			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s2, l_ub2, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg14, vreg_s1, preg_val);
			AscendC::Reg::Mul(vreg_s2, vreg14, vreg_s2, preg_val);

			AscendC::Reg::Sub(vreg15, vreg15, vreg_s1, preg_val);
			AscendC::Reg::Sub(vreg16, vreg16, vreg_s2, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg14, vreg14, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg14, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg14, block_words32, preg_vq2);

			// // line 16

			AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_s1, l_ub1, (0+1)*fractal_len + 1);

			AscendC::Reg::Mul(vreg_s1, vreg15, vreg_s1, preg_val);

			AscendC::Reg::Sub(vreg16, vreg16, vreg_s1, preg_val);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg15, vreg15, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg15, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg15, block_words32, preg_vq2);

			AscendC::Reg::Cast<T, float, castTraitRintZero>((AscendC::Reg::RegTensor<T> &)vreg16, vreg16, preg_val);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub1, (AscendC::Reg::RegTensor<uint32_t> &)vreg16, block_words32, preg_vq);
			AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::StoreDist::DIST_PACK_B32>(m_ub2, (AscendC::Reg::RegTensor<uint32_t> &)vreg16, block_words32, preg_vq2);
		}

	}

	template<uint16_t unroll_id, uint16_t iter_id, bool varlen, bool tail>
	__aicore__ inline void vec_v_corr(__ubuf__ void *v_corr_ub, __ubuf__ void *kplus_score_ub, __ubuf__ void *beta_ub)
	{
		uint16_t vid = unroll_id*2 + iter_id;

		// if(v_cnt == 0) {
		// 	for(int i = 0; i < 4; i ++){
		// 		wait_flag(PIPE_V, PIPE_MTE2, 2 + i);

		// 		gm_ub_single<16, 128, 2>(v_ub[vid], k_gm[32*128*v_cnt + 1*fractal_len28*subid]);
		// 		v_cnt ++;

		// 		set_flag(PIPE_MTE2, PIPE_V, 2 + i);
		// 	}
		// }
		// 尾块:本半 chunk 的头内行基 = chunk_size*v_cnt + chunk_half_rows*subid
		const int64_t row_base = chunk_size*(int64_t)v_cnt + chunk_half_rows*(int64_t)subid;
		uint32_t valid_rows = chunk_half_rows;
		if (tail && has_tail_) {
			const int64_t rows_left = (int64_t)seq_len - row_base;
			valid_rows = rows_left <= 0 ? 0 : (rows_left >= chunk_half_rows ? chunk_half_rows : (uint32_t)rows_left);
		}
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vid + 2);

		if constexpr (varlen) {
			if (valid_rows > 0) {
				AscendC::LocalTensor<T> v_t(AscendC::TPosition::VECCALC, KdaAddrV(v_ub[vid]), chunk_half_rows*value_dim);
				AscendC::DataCopyParams sp;
				sp.blockCount = (uint16_t)valid_rows; sp.blockLen = value_dim*sizeof(T)/block_bytes;
				sp.dstStride = 0;
				sp.srcStride = is_varlen == 2 ? 0 : (uint32_t)(hv_real - 1)*value_dim*sizeof(T)/block_bytes;
				AscendC::DataCopy(v_t, v_gm[row_base*(is_varlen == 2 ? value_dim : hv_real*value_dim)], sp);
			}
		} else if (valid_rows == chunk_half_rows) {
			gm_ub_single<chunk_half_rows, value_dim, 2>(v_ub[vid], v_gm[row_base*value_dim]);
		} else if (valid_rows > 0) {
			AscendC::LocalTensor<T> v_t(AscendC::TPosition::VECCALC, KdaAddrV(v_ub[vid]), chunk_half_rows*value_dim);
			AscendC::DataCopy(v_t, v_gm[row_base*value_dim], valid_rows*value_dim);
		}
		v_cnt ++;

		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vid + 2);

		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vid + 2);

		if (valid_rows < chunk_half_rows) {
			// raw 尾行清零:v_corr=(v-kplus_score)·beta_eff,陈旧/未初始化数据会经 0×NaN 污染
			AscendC::LocalTensor<T> v_t(AscendC::TPosition::VECCALC, KdaAddrV(v_ub[vid]), chunk_half_rows*value_dim);
			AscendC::Duplicate<T>(v_t[valid_rows*value_dim], (T)0, (chunk_half_rows-valid_rows)*value_dim);
		}

		// // debug
		// pipe_barrier(PIPE_ALL);
		// //for(volatile int i = 0; i < 1000*1000*100; i ++);
		// if(subid == 1 && arrive_cnt == 1) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<16,128,2>(tmp_gm, v_ub[vid]);
		// 	ub_gm_single<16,128,2>(tmp_gm, kplus_score_ub);
		// }
		// arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);

		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<uint8_t, MaskPattern::ALL>();
			if constexpr (iter_id == 0) {
				RegTensor<float> vreg_b32;
				RegTensor<T> vreg_b16;
				DataCopy(vreg_b32, (__ubuf__ float *)beta_ub);
				Cast<T, float, f2bfZero>(vreg_b16, vreg_b32, preg_all);
				//DataCopy((__ubuf__ T *)beta_ub, vreg_b16, preg_all);
				DataCopy<uint32_t, StoreDist::DIST_PACK_B32>((__ubuf__ uint32_t *)beta_ub, (RegTensor<uint32_t> &)vreg_b16, preg_all);
				LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
			}
			for(uint16_t i = 0; i < (uint16_t)(chunk_half_rows / row_pair); i ++) {
				RegTensor<T> vreg_beta1, vreg_beta2, vreg_v1, vreg_v2, vreg_kplus_score1, vreg_kplus_score2, vreg_v_corr1, vreg_v_corr2;
				DataCopy(vreg_kplus_score1, (__ubuf__ T *)kplus_score_ub + i*(row_pair*value_dim));
				DataCopy(vreg_kplus_score2, (__ubuf__ T *)kplus_score_ub + i*(row_pair*value_dim) + value_dim);
				DataCopy(vreg_v1, (__ubuf__ T *)v_ub[vid] + i*(row_pair*value_dim));
				DataCopy(vreg_v2, (__ubuf__ T *)v_ub[vid] + i*(row_pair*value_dim) + value_dim);
				Sub(vreg_v_corr1, vreg_v1, vreg_kplus_score1, preg_all);
				Sub(vreg_v_corr2, vreg_v2, vreg_kplus_score2, preg_all);
				DataCopy<T, LoadDist::DIST_BRC_B16>(vreg_beta1, (__ubuf__ T *)beta_ub + i*row_pair + iter_id*chunk_size + subid*chunk_half_rows);
				DataCopy<T, LoadDist::DIST_BRC_B16>(vreg_beta2, (__ubuf__ T *)beta_ub + i*row_pair + 1 + iter_id*chunk_size + subid*chunk_half_rows);
				Mul(vreg_v_corr1, vreg_v_corr1, vreg_beta1, preg_all);
				Mul(vreg_v_corr2, vreg_v_corr2, vreg_beta2, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)v_corr_ub + i*(row_pair*block_elems_16b), (RegTensor<T> &)vreg_v_corr1, 17, preg_all);
				DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ T *)v_corr_ub + i*(row_pair*block_elems_16b) + block_elems_16b, (RegTensor<T> &)vreg_v_corr2, 17, preg_all);
			}
		}
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vid + 2);

		// // debug
		// pipe_barrier(PIPE_ALL);
		// //for(volatile int i = 0; i < 1000*1000*100; i ++);
		// if(subid == 1 && arrive_cnt == 1) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	ub_gm_single<16,128,2>(tmp_gm, v_ub[vid]);
		// 	//ub_gm_single<16,128,2>(tmp_gm, kplus_score_ub);
		// }
		// arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);

		// // debug
		// pipe_barrier(PIPE_ALL);
		// //for(volatile int i = 0; i < 1000*1000*100; i ++);
		// if(subid == 0 && arrive_cnt == 1) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<17,128,2>(tmp_gm, v_corr_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// }
		// arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);

		// if(v_cnt < group_count*4) {
		// 	wait_flag(PIPE_V, PIPE_MTE2, vid + 2);

		// 	gm_ub_single<16, 128, 2>(v_ub[vid], k_gm[32*128*v_cnt + 1*fractal_len28*subid]);
		// 	v_cnt ++;

		// 	set_flag(PIPE_MTE2, PIPE_V, vid + 2);
		// }
	}

	template<uint16_t part_id, uint64_t iter_id>
	__aicore__ inline void vec_iter_half(__ubuf__ void *s_ub, __ubuf__ void *score_delta_a_ub, __ubuf__ void *score_delta_b_ub, __ubuf__ void *gamma_ub)
	{
		// // debug
		// pipe_barrier(PIPE_ALL);
		// //for(volatile int i = 0; i < 1000*1000*100; i ++);
		// if(subid == 1 && arrive_cnt == 2) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	ub_gm_single<64,32,4>(tmp_gm, score_delta_a_ub);
		// 	ub_gm_single<64,32,4>(tmp_gm[64*32], score_delta_b_ub);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// }
		// arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);

		__VEC_SCOPE__{
			MaskReg preg_all = CreateMask<uint8_t, MaskPattern::ALL>();
			for(uint16_t i = 0; i < (uint16_t)((value_dim/2) / value_tile); i ++) {
				for(uint16_t j = 0; j < (uint16_t)((value_dim/2) / fractal_len); j ++) {
					RegTensor<float> vreg_g1, vreg_g2;
					RegTensor<T> vreg_s1, vreg_s2;
					RegTensor<float> vreg_s11, vreg_s12, vreg_s21, vreg_s22;
					RegTensor<float> vreg_s_delta11, vreg_s_delta12, vreg_s_delta21, vreg_s_delta22;
					DataCopy(vreg_s1, (__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + i*(state_k_half*fractal_len) + j*fractal_elems);
					DataCopy(vreg_s2, (__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len);
					Cast<float, T, bf2fZero>(vreg_s11, vreg_s1, preg_all);
					Cast<float, T, bf2fOne >(vreg_s12, vreg_s1, preg_all);
					Cast<float, T, bf2fZero>(vreg_s21, vreg_s2, preg_all);
					Cast<float, T, bf2fOne >(vreg_s22, vreg_s2, preg_all);

					DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_g1, (__ubuf__ float *)gamma_ub + j*fractal_len + subid*half_channels + iter_id*key_dim);
					DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_g2, (__ubuf__ float *)gamma_ub + j*fractal_len + fractal_len/2 + subid*half_channels + iter_id*key_dim);

					Mul(vreg_s11, vreg_s11, vreg_g1, preg_all);
					Mul(vreg_s12, vreg_s12, vreg_g1, preg_all);
					Mul(vreg_s21, vreg_s21, vreg_g2, preg_all);
					Mul(vreg_s22, vreg_s22, vreg_g2, preg_all);

					DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_s_delta11, vreg_s_delta12, (__ubuf__ float *)score_delta_a_ub + i*(state_k_half*fractal_len) + j*fractal_elems);
					DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_s_delta21, vreg_s_delta22, (__ubuf__ float *)score_delta_a_ub + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len);

					Add(vreg_s11, vreg_s11, vreg_s_delta11, preg_all);
					Add(vreg_s12, vreg_s12, vreg_s_delta12, preg_all);
					Add(vreg_s21, vreg_s21, vreg_s_delta21, preg_all);
					Add(vreg_s22, vreg_s22, vreg_s_delta22, preg_all);

					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_s11, vreg_s11, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_s12, vreg_s12, preg_all);
					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_s21, vreg_s21, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_s22, vreg_s22, preg_all);

					Or((RegTensor<uint16_t> &)vreg_s1, (RegTensor<uint16_t> &)vreg_s11, (RegTensor<uint16_t> &)vreg_s12, preg_all);
					Or((RegTensor<uint16_t> &)vreg_s2, (RegTensor<uint16_t> &)vreg_s21, (RegTensor<uint16_t> &)vreg_s22, preg_all);

					DataCopy((__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + i*(state_k_half*fractal_len) + j*fractal_elems, vreg_s1, preg_all);
					DataCopy((__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len, vreg_s2, preg_all);
				}
			}
			for(uint16_t i = 0; i < (uint16_t)((value_dim/2) / value_tile); i ++) {
				for(uint16_t j = 0; j < (uint16_t)((value_dim/2) / fractal_len); j ++) {
					RegTensor<float> vreg_g1, vreg_g2;
					RegTensor<T> vreg_s1, vreg_s2;
					RegTensor<float> vreg_s11, vreg_s12, vreg_s21, vreg_s22;
					RegTensor<float> vreg_s_delta11, vreg_s_delta12, vreg_s_delta21, vreg_s_delta22;
					DataCopy(vreg_s1, (__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + state_k_half*value_tile + i*(state_k_half*fractal_len) + j*fractal_elems);
					DataCopy(vreg_s2, (__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + state_k_half*value_tile + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len);
					Cast<float, T, bf2fZero>(vreg_s11, vreg_s1, preg_all);
					Cast<float, T, bf2fOne >(vreg_s12, vreg_s1, preg_all);
					Cast<float, T, bf2fZero>(vreg_s21, vreg_s2, preg_all);
					Cast<float, T, bf2fOne >(vreg_s22, vreg_s2, preg_all);

					DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_g1, (__ubuf__ float *)gamma_ub + j*fractal_len + subid*half_channels + iter_id*key_dim);
					DataCopy<float, LoadDist::DIST_E2B_B32>(vreg_g2, (__ubuf__ float *)gamma_ub + j*fractal_len + fractal_len/2 + subid*half_channels + iter_id*key_dim);

					Mul(vreg_s11, vreg_s11, vreg_g1, preg_all);
					Mul(vreg_s12, vreg_s12, vreg_g1, preg_all);
					Mul(vreg_s21, vreg_s21, vreg_g2, preg_all);
					Mul(vreg_s22, vreg_s22, vreg_g2, preg_all);

					DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_s_delta11, vreg_s_delta12, (__ubuf__ float *)score_delta_b_ub + i*(state_k_half*fractal_len) + j*fractal_elems);
					DataCopy<float, LoadDist::DIST_DINTLV_B32>(vreg_s_delta21, vreg_s_delta22, (__ubuf__ float *)score_delta_b_ub + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len);

					Add(vreg_s11, vreg_s11, vreg_s_delta11, preg_all);
					Add(vreg_s12, vreg_s12, vreg_s_delta12, preg_all);
					Add(vreg_s21, vreg_s21, vreg_s_delta21, preg_all);
					Add(vreg_s22, vreg_s22, vreg_s_delta22, preg_all);

					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_s11, vreg_s11, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_s12, vreg_s12, preg_all);
					Cast<T, float, f2bfZero>((RegTensor<T> &)vreg_s21, vreg_s21, preg_all);
					Cast<T, float, f2bfOne >((RegTensor<T> &)vreg_s22, vreg_s22, preg_all);

					Or((RegTensor<uint16_t> &)vreg_s1, (RegTensor<uint16_t> &)vreg_s11, (RegTensor<uint16_t> &)vreg_s12, preg_all);
					Or((RegTensor<uint16_t> &)vreg_s2, (RegTensor<uint16_t> &)vreg_s21, (RegTensor<uint16_t> &)vreg_s22, preg_all);

					DataCopy((__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + state_k_half*value_tile + i*(state_k_half*fractal_len) + j*fractal_elems, vreg_s1, preg_all);
					DataCopy((__ubuf__ T *)s_ub + part_id*(state_k_half*(value_dim/2))  + state_k_half*value_tile + i*(state_k_half*fractal_len) + j*fractal_elems + 8*fractal_len, vreg_s2, preg_all);
				}
			}
		}
		// // debug
		// pipe_barrier(PIPE_ALL);
		// //for(volatile int i = 0; i < 1000*1000*100; i ++);
		// if(subid == 0 && arrive_cnt == 1) {
		// 	ub_gm_single<64,128,2>(tmp_gm, s_ub);
		// }
		// arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);
	}
};


// tail=false 仅用于 dense 无尾块实例,裁掉三处热 I/O 的尾块钳制;varlen 必须为 true。
template <typename T, bool varlen, bool tail>
class Kernel
{
public:
	SyncCV syncCV;
	KDAImpl<T> kdaImp;

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
	__aicore__ inline void vec_prepare_g1(int loop_id, uint32_t gate_sc1_ub, uint32_t gate_pre_a_ub, uint32_t beta_a_ub, uint32_t qhat1_ub, uint32_t khat1_ub) {
		AscendC::LocalTensor<uint8_t> gate_sc1_ub_addr = Ub(addrUB[gate_sc1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gate_pre_a_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> beta_ub_addr = Ub(addrUB[beta_a_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> qq2_ub_addr = Ub(addrUB[qhat1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kk2_ub_addr = Ub(addrUB[khat1_ub]*ub_addr_unit);

		if (kdaImp.gate_flags & 1) {
			kdaImp.template vec_prepare_g3<U, true, true, varlen, tail>((__ubuf__ void *)gate_sc1_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr(), (__ubuf__ void *)beta_ub_addr.GetPhyAddr(), (__ubuf__ void *)qq2_ub_addr.GetPhyAddr(), (__ubuf__ void *)kk2_ub_addr.GetPhyAddr());
		} else {
			kdaImp.template vec_prepare_g3<U, false, false, varlen, tail>((__ubuf__ void *)gate_sc1_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr(), (__ubuf__ void *)beta_ub_addr.GetPhyAddr(), (__ubuf__ void *)qq2_ub_addr.GetPhyAddr(), (__ubuf__ void *)kk2_ub_addr.GetPhyAddr());
		}
	}

	template<int U>  // 0=主流水槽位,1=auto_unroll 流水第二槽位
	__aicore__ inline void vec_prepare_qplus_kplus_kminus_kres(int loop_id, uint32_t qplus_ub, uint32_t kplus_ub, uint32_t kplusb_ub, uint32_t qplusb_ub, uint32_t kminus_ub, uint32_t kres_ub, uint32_t gate_pre_ub, uint32_t qhat_ub, uint32_t khat_ub, uint32_t gate_sc_ub, uint32_t gate_pre_b_ub) {
		AscendC::LocalTensor<uint8_t> qplus_ub_addr = Ub(addrUB[qplus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> qplusb_ub_addr = Ub(addrUB[qplusb_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplus_ub_addr = Ub(addrUB[kplus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplusb_ub_addr = Ub(addrUB[kplusb_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kminus_ub_addr = Ub(addrUB[kminus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kres_ub_addr = Ub(addrUB[kres_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> gate_sc_ub_addr = Ub(addrUB[gate_sc_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gate_pre_ub]*ub_addr_unit);

		kdaImp.template vec_prepare_qplus_kplus_kminus_kres<U>((__ubuf__ void *)qplus_ub_addr.GetPhyAddr(), (__ubuf__ void *)qplusb_ub_addr.GetPhyAddr(), (__ubuf__ void *)kplus_ub_addr.GetPhyAddr(), (__ubuf__ void *)kplusb_ub_addr.GetPhyAddr(), (__ubuf__ void *)kminus_ub_addr.GetPhyAddr(), (__ubuf__ void *)kres_ub_addr.GetPhyAddr(), (__ubuf__ void *)gate_sc_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr());
	}
	__aicore__ inline void vec_tril(int loop_id, uint32_t masked_qk_ub, uint32_t l_ub, uint32_t kplus_kminus_ub, uint32_t qplus_kminus_ub, uint32_t beta_b_ub) {
		AscendC::LocalTensor<uint8_t> masked_qk_ub_addr = Ub(addrUB[masked_qk_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> l_ub_addr = Ub(addrUB[l_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplus_kminus_ub_addr = Ub(addrUB[kplus_kminus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> qplus_kminus_ub_addr = Ub(addrUB[qplus_kminus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> beta_ub_addr = Ub(addrUB[beta_b_ub]*ub_addr_unit);

		kdaImp.vec_tril((__ubuf__ void *)masked_qk_ub_addr.GetPhyAddr(), (__ubuf__ void *)l_ub_addr.GetPhyAddr(), (__ubuf__ void *)kplus_kminus_ub_addr.GetPhyAddr(), (__ubuf__ void *)qplus_kminus_ub_addr.GetPhyAddr(), (__ubuf__ void *)beta_ub_addr.GetPhyAddr());
	}

	__aicore__ inline void vec_tril_inv(int loop_id, uint32_t m_ub, uint32_t l_ub, uint32_t m1_ub) {
		AscendC::LocalTensor<uint8_t> m_ub_addr = Ub(addrUB[m_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> l_ub_addr = Ub(addrUB[l_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> m1_ub_addr = Ub(addrUB[m1_ub]*ub_addr_unit);


		// debug
		// pipe_barrier(PIPE_ALL);
		// if(kdaImp.subid == 1 && kdaImp.arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	kdaImp.template ub_gm_single<64,16,4>(kdaImp.tmp_gm, l_ub_addr);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	kdaImp.arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);

		kdaImp.vec_tril_inv1((__ubuf__ void *)m1_ub_addr.GetPhyAddr(), (__ubuf__ void *)l_ub_addr.GetPhyAddr());

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(kdaImp.subid == 0 && kdaImp.arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	kdaImp.template ub_gm_single<64,16,4>(kdaImp.tmp_gm, l_ub_addr);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	kdaImp.arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(kdaImp.subid == 0 && kdaImp.arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	kdaImp.template ub_gm_single<64,16,2>(kdaImp.tmp_gm, m_ub_addr);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	kdaImp.arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);

		//kdaImp.vec_tril_cast((__ubuf__ void *)m_ub_addr.GetPhyAddr(), (__ubuf__ void *)l_ub_addr.GetPhyAddr());

		kdaImp.vec_tril_inv2((__ubuf__ void *)m_ub_addr.GetPhyAddr(), (__ubuf__ void *)m1_ub_addr.GetPhyAddr(), (__ubuf__ void *)l_ub_addr.GetPhyAddr());

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(kdaImp.subid == 0 && kdaImp.arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	kdaImp.template ub_gm_single<64,16,2>(kdaImp.tmp_gm, m_ub_addr);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// }
		// kdaImp.arrive_cnt ++;
		// pipe_barrier(PIPE_ALL);
	}
	template<int U>  // 流水槽位组
	__aicore__ inline void vec_v_corr1(int loop_id, uint32_t v_corr1_ub, uint32_t kplus_score1_ub, uint32_t beta_b_ub) {
		AscendC::LocalTensor<uint8_t> beta_ub_addr = Ub(addrUB[beta_b_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplus_score_ub_addr = Ub(addrUB[kplus_score1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> v_corr_ub_addr = Ub(addrUB[v_corr1_ub]*ub_addr_unit);

		kdaImp.template vec_v_corr<U,0,varlen,tail>((__ubuf__ void *)v_corr_ub_addr.GetPhyAddr(), (__ubuf__ void *)kplus_score_ub_addr.GetPhyAddr(), (__ubuf__ void *)beta_ub_addr.GetPhyAddr());
	}
	__aicore__ inline void vec_iter_partial1(int loop_id, uint32_t s2_partial_ub, uint32_t ss1_ub, uint32_t score_delta_a_ub1, uint32_t score_delta_a_ub2, uint32_t gamma_ub) {
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gamma_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> s_ub_addr = Ub(addrUB[ss1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_a_ub_addr = Ub(addrUB[score_delta_a_ub1]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_b_ub_addr = Ub(addrUB[score_delta_a_ub2]*ub_addr_unit);

		kdaImp.template vec_iter_half<0,0>((__ubuf__ void *)s_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_a_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_b_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr());
	}
	__aicore__ inline void vec_iter_final1(int loop_id, uint32_t s2_ub, uint32_t s2_partial_ub, uint32_t score_delta_a_ub3, uint32_t score_delta_a_ub4, uint32_t gamma_ub) {
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gamma_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> s_ub_addr = Ub(addrUB[s2_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_a_ub_addr = Ub(addrUB[score_delta_a_ub3]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_b_ub_addr = Ub(addrUB[score_delta_a_ub4]*ub_addr_unit);

		kdaImp.template vec_iter_half<1,0>((__ubuf__ void *)s_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_a_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_b_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr());
	}
	template<int U>  // 流水槽位组
	__aicore__ inline void vec_v_corr2(int loop_id, uint32_t v_corr2_ub, uint32_t kplus_score2_ub, uint32_t beta_b_ub) {
		AscendC::LocalTensor<uint8_t> beta_ub_addr = Ub(addrUB[beta_b_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplus_score_ub_addr = Ub(addrUB[kplus_score2_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> v_corr_ub_addr = Ub(addrUB[v_corr2_ub]*ub_addr_unit);

		kdaImp.template vec_v_corr<U,1,varlen,tail>((__ubuf__ void *)v_corr_ub_addr.GetPhyAddr(), (__ubuf__ void *)kplus_score_ub_addr.GetPhyAddr(), (__ubuf__ void *)beta_ub_addr.GetPhyAddr());
	}
	__aicore__ inline void vec_iter_partial2(int loop_id, uint32_t s1_partial_ub, uint32_t ss2_ub, uint32_t score_delta_b_ub1, uint32_t score_delta_b_ub2, uint32_t gamma_ub) {
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gamma_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> s_ub_addr = Ub(addrUB[ss2_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_a_ub_addr = Ub(addrUB[score_delta_b_ub1]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_b_ub_addr = Ub(addrUB[score_delta_b_ub2]*ub_addr_unit);

		kdaImp.template vec_iter_half<0,1>((__ubuf__ void *)s_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_a_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_b_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr());
	}
	__aicore__ inline void vec_iter_final2(int loop_id, uint32_t s1_ub_auto_unroll_id_1, uint32_t s1_partial_ub, uint32_t score_delta_b_ub3, uint32_t score_delta_b_ub4, uint32_t gamma_ub) {
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gamma_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> s_ub_addr = Ub(addrUB[s1_partial_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_a_ub_addr = Ub(addrUB[score_delta_b_ub3]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> score_delta_b_ub_addr = Ub(addrUB[score_delta_b_ub4]*ub_addr_unit);

		kdaImp.template vec_iter_half<1,1>((__ubuf__ void *)s_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_a_ub_addr.GetPhyAddr(), (__ubuf__ void *)score_delta_b_ub_addr.GetPhyAddr(), (__ubuf__ void *)gamma_ub_addr.GetPhyAddr());
	}












	__aicore__ inline void ub_to_l1_qplus_kplus(int loop_id, uint32_t qplus, uint32_t kplus, uint32_t kplusb, uint32_t qplusb, uint32_t qplus_ub, uint32_t kplus_ub, uint32_t kplusb_ub, uint32_t qplusb_ub) {
		AscendC::LocalTensor<uint8_t> qplus_ub_addr = Ub(addrUB[qplus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> qplusb_ub_addr = Ub(addrUB[qplusb_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplus_ub_addr = Ub(addrUB[kplus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kplusb_ub_addr = Ub(addrUB[kplusb_ub]*ub_addr_unit);

		AscendC::LocalTensor<uint8_t> qd_addr = L1b(addrL1[qplus]*l1_addr_unit + chunk_size*key_dim*sizeof(T)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> qd2_addr = L1b(addrL1[qplusb]*l1_addr_unit + chunk_half_rows*key_dim*sizeof(T)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> kd_addr = L1b(addrL1[kplus]*l1_addr_unit + chunk_size*key_dim*sizeof(T)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> kd2_addr = L1b(addrL1[kplusb]*l1_addr_unit + chunk_half_rows*key_dim*sizeof(T)*kdaImp.subid);

		kdaImp.template ub_l1_src_strided_nz<chunk_size, key_dim, sizeof(T)>(qplus_ub_addr, qd_addr);
		kdaImp.template ub_l1_src_strided_nz<chunk_half_rows, key_dim, sizeof(T)>(qplusb_ub_addr, qd2_addr);
		kdaImp.template ub_l1_src_strided_nz<chunk_size, key_dim, sizeof(T)>(kplus_ub_addr, kd_addr);
		kdaImp.template ub_l1_src_strided_nz<chunk_half_rows, key_dim, sizeof(T)>(kplusb_ub_addr, kd2_addr);
	}
	__aicore__ inline void ub_to_l1_kminus_kres(int loop_id, uint32_t kminus, uint32_t kres, uint32_t gamma, uint32_t beta, uint32_t kminus_ub, uint32_t kres_ub, uint32_t gate_pre_ub, uint32_t beta_ub) {
		AscendC::LocalTensor<uint8_t> kminus_ub_addr = Ub(addrUB[kminus_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> kres_ub_addr = Ub(addrUB[kres_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> gamma_ub_addr = Ub(addrUB[gate_pre_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> beta_ub_addr = Ub(addrUB[beta_ub]*ub_addr_unit);

		AscendC::LocalTensor<uint8_t> ki_addr = L1b(addrL1[kminus]*l1_addr_unit + chunk_size*key_dim*sizeof(T)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> kr_addr = L1b(addrL1[kres]*l1_addr_unit + chunk_size*key_dim*sizeof(T)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> gamma_addr = L1b(addrL1[gamma]*l1_addr_unit + key_dim*sizeof(float)*kdaImp.subid);
		AscendC::LocalTensor<uint8_t> beta_addr = L1b(addrL1[beta]*l1_addr_unit + chunk_size*sizeof(float)*kdaImp.subid);

		kdaImp.template ub_l1_src_strided_nz<chunk_size, key_dim, sizeof(T)>(kminus_ub_addr, ki_addr);
		kdaImp.template ub_l1_src_strided_nz<chunk_size, key_dim, sizeof(T)>(kres_ub_addr, kr_addr);
		kdaImp.template ub_l1_single<1, key_dim, sizeof(float)>(gamma_ub_addr, gamma_addr);
		kdaImp.template ub_l1_single<1, chunk_size, sizeof(float)>(beta_ub_addr, beta_addr);
	}
	__aicore__ inline void ub_to_l1_mm(int loop_id, uint32_t mm, uint32_t m_ub) {
		AscendC::LocalTensor<uint8_t> m_ub_addr = Ub(addrUB[m_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> mm_addr = L1b(addrL1[mm]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*kdaImp.subid);

		kdaImp.template ub_l1_single<chunk_size, chunk_size, sizeof(T)>(m_ub_addr, mm_addr);
	}
	__aicore__ inline void ub_to_l1_masked_qk(int loop_id, uint32_t masked_qk, uint32_t masked_qk_ub) {
		AscendC::LocalTensor<uint8_t> masked_qk_ub_addr = Ub(addrUB[masked_qk_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> masked_qk_addr = L1b(addrL1[masked_qk]*l1_addr_unit + chunk_size*chunk_size*sizeof(T)*kdaImp.subid);

		kdaImp.template ub_l1_single<chunk_size, chunk_size, sizeof(T)>(masked_qk_ub_addr, masked_qk_addr);
	}
	__aicore__ inline void ub_to_l1_s1(int loop_id, uint32_t s1, uint32_t ss1_ub, uint32_t s1_ub) {
		AscendC::LocalTensor<uint8_t> s1_ub_addr = Ub(addrUB[s1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> s1_addr = L1b(addrL1[s1]*l1_addr_unit + state_k_half*fractal_len*sizeof(T)*kdaImp.subid);

		// // debug
		// pipe_barrier(PIPE_ALL);
		// if(kdaImp.subid == 0 && kdaImp.arrive_cnt == 0) {
		// 	// for(int i = 0; i < 32*128; i ++) ((__ubuf__ uint16_t *)(g_ub[id]))[i] = 0;
		// 	//ub_gm_single<32,128,2>(tmp_gm, k_ub[id]);
		// 	kdaImp.template ub_gm_single<64,128,2>(kdaImp.tmp_gm, s1_ub_addr);
		// 	//ub_gm_single<1,32,4>(tmp_gm, dt_ub[id]);
		// 	kdaImp.arrive_cnt ++;
		// }
		// pipe_barrier(PIPE_ALL);
		kdaImp.template ub_l1_dst_strided_nz<state_k_half, value_dim, sizeof(T)>(s1_ub_addr, s1_addr);
	}
	__aicore__ inline void ub_to_l1_v_corr1(int loop_id, uint32_t v_corr1, uint32_t v_corr1_ub) {
		AscendC::LocalTensor<uint8_t> v_corr1_ub_addr = Ub(addrUB[v_corr1_ub]*ub_addr_unit);
		AscendC::LocalTensor<uint8_t> v_corr1_addr = L1b(addrL1[v_corr1]*l1_addr_unit + chunk_half_rows*fractal_len*sizeof(T)*kdaImp.subid);

		kdaImp.template ub_l1_src_strided_dst_strided_nz<chunk_half_rows, key_dim, sizeof(T)>(v_corr1_ub_addr, v_corr1_addr);
	}
	__aicore__ inline void ub_to_gm_out1(int loop_id, uint32_t out1_ub) {
		AscendC::LocalTensor<uint8_t> out_ub_addr = Ub(addrUB[out1_ub]*ub_addr_unit);

		// 尾块:本半 chunk 的头内行基 = chunk_size*out_cnt + chunk_half_rows*subid;只写有效行,valid_rows<=0 整块跳过
		// (本函数无旗语,跳过不影响同步配平)
		const int64_t row_base = chunk_size*(int64_t)kdaImp.out_cnt + chunk_half_rows*(int64_t)kdaImp.subid;
		int64_t valid_rows = chunk_half_rows;
		if (tail && kdaImp.has_tail_) {
			valid_rows = (int64_t)kdaImp.seq_len - row_base;
		}
		if constexpr (varlen) {
			// valid_rows 可能超过 16(尾块第二组):UB 槽只有 16 行,先钳到 chunk_half_rows
			const int64_t wr = valid_rows >= chunk_half_rows ? chunk_half_rows : valid_rows;
			if (wr > 0) {
				AscendC::LocalTensor<T> src(AscendC::TPosition::VECCALC, addrUB[out1_ub]*ub_addr_unit, chunk_half_rows*value_dim);
				AscendC::DataCopyParams dp;
				dp.blockCount = (uint16_t)wr; dp.blockLen = value_dim*sizeof(T)/block_bytes;
				dp.srcStride = 0;
				dp.dstStride = (uint32_t)(kdaImp.hv_real - 1)*value_dim*sizeof(T)/block_bytes;
				AscendC::DataCopy(kdaImp.o_gm[row_base*kdaImp.hv_real*value_dim], src, dp);
			}
		} else if (valid_rows >= chunk_half_rows) {
			kdaImp.template ub_gm_single<chunk_half_rows, value_dim, 2>(kdaImp.o_gm[row_base*value_dim], (__ubuf__ void *)out_ub_addr.GetPhyAddr());
		} else if (valid_rows > 0) {
			AscendC::LocalTensor<T> src(AscendC::TPosition::VECCALC, addrUB[out1_ub]*ub_addr_unit, chunk_half_rows*value_dim);
			AscendC::DataCopy(kdaImp.o_gm[row_base*value_dim], src, (uint32_t)valid_rows*value_dim);
		}
		kdaImp.out_cnt ++;
	}















	template <bool STEADY>
	__aicore__ inline void stage_prepare_g1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(v_corr1);
		AscendC::Mutex::Lock<PIPE_V>(qhat1_ub);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc1_ub);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_a_ub);
		AscendC::Mutex::Lock<PIPE_V>(beta_a_ub);
		AscendC::Mutex::Lock<PIPE_V>(khat1_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(v_corr1_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_prepare_g1<0>(iter - 0, gate_sc1_ub, gate_pre_a_ub, beta_a_ub, qhat1_ub, khat1_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_v_corr1(iter - 1, v_corr1, v_corr1_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_qplus_kplus_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_a_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(v_corr1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(qhat1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(khat1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(beta_a_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kd2_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(qd2_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(qd_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kd_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_MTE3>(qplus_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(kplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(qplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(kplus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_qplus_kplus(iter - 1, qd_auto_unroll_id_1, kd_auto_unroll_id_1, kd2_auto_unroll_id_1, qd2_auto_unroll_id_1, qplus_ub_auto_unroll_id_1, kplus_ub_auto_unroll_id_1, kplusb_ub_auto_unroll_id_1, qplusb_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_kminus_kres_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qd2_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub2_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kd2_auto_unroll_id_1);
		}
		AscendC::Mutex::Unlock<PIPE_MTE3>(kplusb_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qd_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kplus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(qplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(qplus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kd_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(masked_qk_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(ki_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(gamma_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(beta_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kr_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_MTE3>(gate_pre_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc1_ub);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_a_ub);
		AscendC::Mutex::Lock<PIPE_V>(qhat1_ub);
		AscendC::Mutex::Lock<PIPE_V>(khat1_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(beta_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(beta_a_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(kres_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(kminus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_kminus_kres(iter - 1, ki_auto_unroll_id_1, kr_auto_unroll_id_1, gamma_auto_unroll_id_1, beta_auto_unroll_id_1, kminus_ub_auto_unroll_id_1, kres_ub_auto_unroll_id_1, gate_pre_ub_auto_unroll_id_1, beta_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(kplus_score2_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(khat_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(gate_pre_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_b_ub4);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kres_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(beta_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(ki_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(beta_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(gamma_auto_unroll_id_1);
		}
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc_ub);
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_b_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kr_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kminus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4);
		AscendC::Mutex::Unlock<PIPE_V>(qhat_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1);
		    syncCV.WaitIntraBlock<MTE1toV_UBReady>(gamma_ub);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub2);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub1);
		AscendC::Mutex::Lock<PIPE_V>(ss1_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_partial1(iter - 1, s2_partial_ub, ss1_ub, score_delta_a_ub1, score_delta_a_ub2, gamma_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_final1_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub2);
		AscendC::Mutex::Unlock<PIPE_V>(s2_partial_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_score2_ub);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kd2_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub4);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub);
		AscendC::Mutex::Lock<PIPE_V>(s2_partial_ub);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub3);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_final1(iter - 1, s2_ub, s2_partial_ub, score_delta_a_ub3, score_delta_a_ub4, gamma_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_s1_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out2_ub);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out1_ub);
		}
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub3);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub4);
		AscendC::Mutex::Unlock<PIPE_V>(s2_ub);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qd2_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(ki_auto_unroll_id_1);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(s2);
		AscendC::Mutex::Lock<PIPE_MTE3>(s2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_s1(iter - 1, s2, ss2_ub, s2_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_tril_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(ss2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s2);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qplus);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kplus);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kplusb);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(gamma_ub_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(beta_b_ub_auto_unroll_id_1);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(gamma_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(beta_auto_unroll_id_1);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toV_UBReady>(beta_b_ub_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(l_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(qplus_kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(masked_qk_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplus_kminus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_tril(iter - 1, masked_qk_ub_auto_unroll_id_1, l_ub_auto_unroll_id_1, kplus_kminus_ub_auto_unroll_id_1, qplus_kminus_ub_auto_unroll_id_1, beta_b_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr2_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(l_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(masked_qk_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qplus_kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_kminus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_score2_ub);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_score2_ub);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(masked_qk_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub);
		AscendC::Mutex::Lock<PIPE_V>(kplus_score2_ub);
		AscendC::Mutex::Lock<PIPE_V>(v_corr2_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(masked_qk_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_v_corr2<0>(iter - 1, v_corr2_ub, kplus_score2_ub, beta_b_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_masked_qk_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_masked_qk(iter - 1, masked_qk_auto_unroll_id_1, masked_qk_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(beta_b_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(masked_qk_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3_auto_unroll_id_1);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3);
		}
		AscendC::Mutex::Unlock<PIPE_V>(v_corr2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(masked_qk_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_score2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(v_corr2);
		AscendC::Mutex::Lock<PIPE_V>(l_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(m1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(v_corr2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_v_corr1(iter - 1, v_corr2, v_corr2_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_out1_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(m1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr2);
		AscendC::Mutex::Unlock<PIPE_MTE3>(v_corr2_ub);
		AscendC::Mutex::Unlock<PIPE_V>(l_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<FIXtoMTE3_UBReady>(out1_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(out1_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_gm_out1(iter - 1, out1_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_tril_inv_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(out1_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(mm);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(masked_qk);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s2_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kres);
		AscendC::Mutex::Lock<PIPE_V>(m1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(m_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(l_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_tril_inv(iter - 1, m_ub_auto_unroll_id_1, l_ub_auto_unroll_id_1, m1_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial2_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(m1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(m_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(l_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub2);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub1);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub);
		AscendC::Mutex::Lock<PIPE_V>(ss2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_partial2(iter - 1, s1_partial_ub, ss2_ub, score_delta_b_ub1, score_delta_b_ub2, gamma_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_final2_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub2);
		AscendC::Mutex::Unlock<PIPE_V>(s1_partial_ub);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub1);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub3);
		AscendC::Mutex::Lock<PIPE_V>(s1_partial_ub);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub4);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_final2(iter - 1, s1_ub_auto_unroll_id_1, s1_partial_ub, score_delta_b_ub3, score_delta_b_ub4, gamma_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_mm_u1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(gamma_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_score1_ub_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out2_ub);
		}
		AscendC::Mutex::Unlock<PIPE_V>(s1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub3);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub4);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(mm_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(m_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_mm(iter - 1, mm_auto_unroll_id_1, m_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_prepare_qplus_kplus_kminus_kres_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(m_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(mm_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(s1_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(qplus_ub);
		AscendC::Mutex::Lock<PIPE_V>(qplusb_ub);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_b_ub);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc_ub);
		AscendC::Mutex::Lock<PIPE_V>(kplus_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(s1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplusb_ub);
		AscendC::Mutex::Lock<PIPE_V>(qhat_ub);
		AscendC::Mutex::Lock<PIPE_V>(kminus_ub);
		AscendC::Mutex::Lock<PIPE_V>(khat_ub);
		AscendC::Mutex::Lock<PIPE_V>(kres_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_prepare_qplus_kplus_kminus_kres<0>(iter - 0, qplus_ub, kplus_ub, kplusb_ub, qplusb_ub, kminus_ub, kres_ub, gate_pre_ub, qhat_ub, khat_ub, gate_sc_ub, gate_pre_b_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_s1_u1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_s1(iter - 1, s1_auto_unroll_id_1, ss1_ub_auto_unroll_id_1, s1_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_out1_u0_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_ub);
		AscendC::Mutex::Unlock<PIPE_V>(qplus_ub);
		AscendC::Mutex::Unlock<PIPE_V>(kres_ub);
		AscendC::Mutex::Unlock<PIPE_V>(kminus_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s1_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qplusb_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(ss1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_ub);
		AscendC::Mutex::Unlock<PIPE_V>(kplusb_ub);
		AscendC::Mutex::Unlock<PIPE_V>(qhat_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(khat_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoMTE3_UBReady>(out2_ub);
		}
		AscendC::Mutex::Lock<PIPE_MTE3>(out2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_gm_out1(iter - 1, out2_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(out2_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(out1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s1_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_score1_ub_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_score1_ub_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplus_score1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(v_corr1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_v_corr1<1>(iter - 1, v_corr1_ub_auto_unroll_id_1, kplus_score1_ub_auto_unroll_id_1, beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_score1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(v_corr1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(v_corr1_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(khat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(qhat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(v_corr1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_a_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(beta_a_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_prepare_g1<1>(iter - 0, gate_sc1_ub_auto_unroll_id_1, gate_pre_a_ub_auto_unroll_id_1, beta_a_ub_auto_unroll_id_1, qhat1_ub_auto_unroll_id_1, khat1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_v_corr1(iter - 1, v_corr1_auto_unroll_id_1, v_corr1_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_qplus_kplus_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_a_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qhat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(khat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_a_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(v_corr1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr1_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kplusb);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kplus);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(qplusb);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(qplus);
		}
		AscendC::Mutex::Lock<PIPE_MTE3>(kplus_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(qplusb_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(qplus_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(kplusb_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  ub_to_l1_qplus_kplus(iter - 0, qplus, kplus, kplusb, qplusb, qplus_ub, kplus_ub, kplusb_ub, qplusb_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_kminus_kres_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(kplusb_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kplus_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kplusb);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub);
		}
		AscendC::Mutex::Unlock<PIPE_MTE3>(qplus_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kplus);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qplusb);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(qplus);
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub1);
		}
		AscendC::Mutex::Unlock<PIPE_MTE3>(qplusb_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2_auto_unroll_id_1);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(gamma);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(beta);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kminus);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(kres);
		}
		AscendC::Mutex::Lock<PIPE_V>(beta_a_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(beta_ub);
		AscendC::Mutex::Lock<PIPE_V>(khat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_a_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(qhat1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(gate_pre_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(kminus_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(kres_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  ub_to_l1_kminus_kres(iter - 0, kminus, kres, gamma, beta, kminus_ub, kres_ub, gate_pre_ub, beta_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial1_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(beta_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(gate_pre_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kminus_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(kres_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(khat_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(qplus_kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qhat_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(beta);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kminus);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(gamma);
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(kres);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<MTE1toV_UBReady>(gamma_ub_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub1_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub2_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub1_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub2_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(ss1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_partial1(iter - 1, s2_partial_ub_auto_unroll_id_1, ss1_ub_auto_unroll_id_1, score_delta_a_ub1_auto_unroll_id_1, score_delta_a_ub2_auto_unroll_id_1, gamma_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_final1_u1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_kminus_ub);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub2_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub1_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(s2_partial_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s2);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub3_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_a_ub4_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(s2_partial_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub4_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_a_ub3_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_final1(iter - 1, s2_ub_auto_unroll_id_1, s2_partial_ub_auto_unroll_id_1, score_delta_a_ub3_auto_unroll_id_1, score_delta_a_ub4_auto_unroll_id_1, gamma_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_s1_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub4_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_a_ub3_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out1_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(s2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub4_auto_unroll_id_1);
		if (STEADY || (iter < group_count && (group_count > 1 || iter > 0))) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kminus);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qplusb);
		}
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(s2_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(s2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_s1(iter - 1, s2_auto_unroll_id_1, ss2_ub_auto_unroll_id_1, s2_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_tril_u0(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s2_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(ss2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(qd_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kd_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub);
		}
		if (STEADY || (iter < group_count && (group_count > 1 || iter > 0))) {
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(beta);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(gamma);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(beta_b_ub);
		    syncCV.SetIntraBlock<MTE1toV_UBReady>(gamma_ub);
		    syncCV.WaitIntraBlock<MTE1toV_UBReady>(beta_b_ub);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_kminus_ub);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(qplus_kminus_ub);
		}
		AscendC::Mutex::Lock<PIPE_V>(kplus_kminus_ub);
		AscendC::Mutex::Lock<PIPE_V>(masked_qk_ub);
		AscendC::Mutex::Lock<PIPE_V>(qplus_kminus_ub);
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub);
		AscendC::Mutex::Lock<PIPE_V>(l_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_tril(iter - 0, masked_qk_ub, l_ub, kplus_kminus_ub, qplus_kminus_ub, beta_b_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_masked_qk_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub);
		AscendC::Mutex::Unlock<PIPE_V>(qplus_kminus_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_score2_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(l_ub);
		AscendC::Mutex::Unlock<PIPE_V>(masked_qk_ub);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_kminus_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub3_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_score2_ub_auto_unroll_id_1);
		}
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(masked_qk);
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(v_corr2_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplus_score2_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(masked_qk_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  ub_to_l1_masked_qk(iter - 0, masked_qk, masked_qk_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_v_corr2<1>(iter - 1, v_corr2_ub_auto_unroll_id_1, kplus_score2_ub_auto_unroll_id_1, beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(masked_qk_ub);
		AscendC::Mutex::Unlock<PIPE_V>(v_corr2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_score2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(masked_qk);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(v_corr2_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(l_ub);
		AscendC::Mutex::Lock<PIPE_V>(m1_ub);
		AscendC::Mutex::Lock<PIPE_MTE3>(v_corr2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_l1_v_corr1(iter - 1, v_corr2_auto_unroll_id_1, v_corr2_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_out1_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(l_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(v_corr2_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(m1_ub);
		AscendC::Mutex::Unlock<PIPE_MTE3>(v_corr2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<FIXtoMTE3_UBReady>(out1_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(out1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_gm_out1(iter - 1, out1_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_tril_inv_u0(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(out1_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toFIX_UBRelease>(score_delta_a_ub3);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(mm_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr2_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(kr_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr1_auto_unroll_id_1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr1);
		}
		AscendC::Mutex::Lock<PIPE_V>(m1_ub);
		AscendC::Mutex::Lock<PIPE_V>(l_ub);
		AscendC::Mutex::Lock<PIPE_V>(m_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_tril_inv(iter - 0, m_ub, l_ub, m1_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_partial2_u1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(m1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(l_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub2);
		AscendC::Mutex::Unlock<PIPE_V>(m_ub);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub1_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub2_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub2_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(ss2_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub1_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_partial2(iter - 1, s1_partial_ub_auto_unroll_id_1, ss2_ub_auto_unroll_id_1, score_delta_b_ub1_auto_unroll_id_1, score_delta_b_ub2_auto_unroll_id_1, gamma_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_iter_final2_u1_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub2_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub1_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(s1_partial_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out2_ub_auto_unroll_id_1);
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub3_auto_unroll_id_1);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(score_delta_b_ub4_auto_unroll_id_1);
		}
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub3_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(score_delta_b_ub4_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(s1_partial_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  vec_iter_final2(iter - 1, s1_ub, s1_partial_ub_auto_unroll_id_1, score_delta_b_ub3_auto_unroll_id_1, score_delta_b_ub4_auto_unroll_id_1, gamma_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_mm_u0(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoMTE1_UBRelease>(gamma_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub3_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(score_delta_b_ub4_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<VtoFIX_UBRelease>(kplus_score1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(s1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(gamma_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.SetIntraBlock<FIXtoMTE3_UBReady>(out2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(mm);
		AscendC::Mutex::Lock<PIPE_MTE3>(m_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  ub_to_l1_mm(iter - 0, mm, m_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_s1_u0_x1(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(m_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(mm);
		    syncCV.WaitIntraBlock<MTE1toMTE3_L1Release>(s1);
		}
		AscendC::Mutex::Lock<PIPE_V>(qhat_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(s1_ub);
		AscendC::Mutex::Lock<PIPE_V>(qplus_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kres_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_sc_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(gate_pre_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(qplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplus_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(kplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_V>(khat_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  ub_to_l1_s1(iter - 0, s1, ss1_ub, s1_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_out1_u1_x1(int iter, int group_count) {
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_prepare_qplus_kplus_kminus_kres<1>(iter - 0, qplus_ub_auto_unroll_id_1, kplus_ub_auto_unroll_id_1, kplusb_ub_auto_unroll_id_1, qplusb_ub_auto_unroll_id_1, kminus_ub_auto_unroll_id_1, kres_ub_auto_unroll_id_1, gate_pre_ub_auto_unroll_id_1, qhat_ub_auto_unroll_id_1, khat_ub_auto_unroll_id_1, gate_sc_ub_auto_unroll_id_1, gate_pre_b_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qplus_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  syncCV.SetIntraBlock<MTE3toMTE1_L1Ready>(s1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_pre_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(gate_sc_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_MTE3>(ss1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(kres_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(kplusb_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(khat_ub_auto_unroll_id_1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(out1_ub);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub2);
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_a_ub4);
		}
		AscendC::Mutex::Unlock<PIPE_V>(kminus_ub_auto_unroll_id_1);
		AscendC::Mutex::Unlock<PIPE_V>(qhat_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  syncCV.WaitIntraBlock<FIXtoMTE3_UBReady>(out2_ub_auto_unroll_id_1);
		AscendC::Mutex::Lock<PIPE_MTE3>(out2_ub_auto_unroll_id_1);
		if (STEADY || (iter - 1 >= 0 && iter - 1 < group_count))  ub_to_gm_out1(iter - 1, out2_ub_auto_unroll_id_1);
	}

	template <bool STEADY>
	__aicore__ inline void stage_v_corr1_u0_x2(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_MTE3>(out2_ub_auto_unroll_id_1);
		if (STEADY || (iter < group_count && (group_count > 1 || iter > 0)))  syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(s1);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<FIXtoV_UBReady>(kplus_score1_ub);
		    syncCV.WaitIntraBlock<FIXtoV_UBReady>(kplus_score1_ub);
		}
		AscendC::Mutex::Lock<PIPE_V>(beta_b_ub);
		AscendC::Mutex::Lock<PIPE_V>(kplus_score1_ub);
		AscendC::Mutex::Lock<PIPE_V>(v_corr1_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count))  vec_v_corr1<0>(iter - 0, v_corr1_ub, kplus_score1_ub, beta_b_ub);
	}

	template <bool STEADY>
	__aicore__ inline void stage_tail36(int iter, int group_count) {
		AscendC::Mutex::Unlock<PIPE_V>(kplus_score1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(v_corr1_ub);
		AscendC::Mutex::Unlock<PIPE_V>(beta_b_ub);
		if (STEADY || (iter - 0 >= 0 && iter - 0 < group_count)) {
		    syncCV.SetIntraBlock<VtoFIX_UBRelease>(score_delta_b_ub1);
		    syncCV.SetIntraBlock<MTE1toMTE3_L1Release>(v_corr2);
		}
	}

	// 实际执行集(数值验证 group_count=1..8 + 语句流模拟确认与三段逐条一致)。
	__aicore__ inline void ProcessEdge(int iter, int group_count) {
		stage_prepare_g1_u0<false>(iter, group_count);
		stage_v_corr1_u0<false>(iter, group_count);
		stage_qplus_kplus_u1<false>(iter, group_count);
		stage_kminus_kres_u1<false>(iter, group_count);
		stage_iter_partial1_u0<false>(iter, group_count);
		stage_iter_final1_u0<false>(iter, group_count);
		stage_s1_u0<false>(iter, group_count);
		stage_tril_u1<false>(iter, group_count);
		stage_v_corr2_u0<false>(iter, group_count);
		stage_masked_qk_u1<false>(iter, group_count);
		stage_v_corr1_u0_x1<false>(iter, group_count);
		stage_out1_u0<false>(iter, group_count);
		stage_tril_inv_u1<false>(iter, group_count);
		stage_iter_partial2_u0<false>(iter, group_count);
		stage_iter_final2_u1<false>(iter, group_count);
		stage_mm_u1<false>(iter, group_count);
		stage_prepare_qplus_kplus_kminus_kres_u0<false>(iter, group_count);
		stage_s1_u1<false>(iter, group_count);
		stage_out1_u0_x1<false>(iter, group_count);
		stage_v_corr1_u1<false>(iter, group_count);
		stage_qplus_kplus_u0<false>(iter, group_count);
		stage_kminus_kres_u0<false>(iter, group_count);
		stage_iter_partial1_u1<false>(iter, group_count);
		stage_iter_final1_u1<false>(iter, group_count);
		stage_s1_u1_x1<false>(iter, group_count);
		stage_tril_u0<false>(iter, group_count);
		stage_masked_qk_u0<false>(iter, group_count);
		stage_v_corr1_u1_x1<false>(iter, group_count);
		stage_out1_u1<false>(iter, group_count);
		stage_tril_inv_u0<false>(iter, group_count);
		stage_iter_partial2_u1<false>(iter, group_count);
		stage_iter_final2_u1_x1<false>(iter, group_count);
		stage_mm_u0<false>(iter, group_count);
		stage_s1_u0_x1<false>(iter, group_count);
		stage_out1_u1_x1<false>(iter, group_count);
		stage_v_corr1_u0_x2<false>(iter, group_count);
		stage_tail36<false>(iter, group_count);
	}

	// Define Kernel Process
	__aicore__ inline void Process(int dense_group_count) {

		int iterN = 2;

		// 任务循环:task_count=0 表示单任务
		// fh = batch*HV + value_head。AIC 侧 Process 走同一任务序列(配平前提)。
		const uint32_t task_begin = get_block_idx();
		const uint32_t task_end = kdaImp.task_count ? (uint32_t)kdaImp.task_count : (task_begin + 1);
		const uint32_t task_step = kdaImp.task_count ? (uint32_t)kdaImp.block_num : 1;
		for (uint32_t task = task_begin; task < task_end; task += task_step) {
			kdaImp.template BindTask<varlen>(task);
			// varlen 空序列:不进流水,仅在请求时写 final=initial(或 0);
			// AIC 侧以同一 cu_seqlens 解析跳过同一任务,跨核旗语不挂账。
			if constexpr (varlen) {
				if (kdaImp.task_empty) {
					kdaImp.WriteEmptyFinalState();
					continue;
				}
			}
			// 每任务组数:varlen 逐任务解析(BindTask 写入),dense 恒为入口值
			const int group_count = varlen ? (int)kdaImp.group_count : dense_group_count;

			syncCV.Init();

			AscendC::Mutex::Lock<PIPE_V>(s1_ub);
			if (kdaImp.has_istate) {
				kdaImp.LoadInitState((__ubuf__ void *)Ub(addrUB[s1_ub]*ub_addr_unit).GetPhyAddr());
			} else {
				kdaImp.InitState((__ubuf__ void *)Ub(addrUB[s1_ub]*ub_addr_unit).GetPhyAddr());
			}
			AscendC::Mutex::Unlock<PIPE_V>(s1_ub);

			ProcessEdge(0, group_count);

			for (int iter = iterN - 1; iter < group_count; iter ++) {
				stage_prepare_g1_u0<true>(iter, group_count);
				stage_v_corr1_u0<true>(iter, group_count);
				stage_qplus_kplus_u1<true>(iter, group_count);
				stage_kminus_kres_u1<true>(iter, group_count);
				stage_iter_partial1_u0<true>(iter, group_count);
				stage_iter_final1_u0<true>(iter, group_count);
				stage_s1_u0<true>(iter, group_count);
				stage_tril_u1<true>(iter, group_count);
				stage_v_corr2_u0<true>(iter, group_count);
				stage_masked_qk_u1<true>(iter, group_count);
				stage_v_corr1_u0_x1<true>(iter, group_count);
				stage_out1_u0<true>(iter, group_count);
				stage_tril_inv_u1<true>(iter, group_count);
				stage_iter_partial2_u0<true>(iter, group_count);
				stage_iter_final2_u1<true>(iter, group_count);
				stage_mm_u1<true>(iter, group_count);
				stage_prepare_qplus_kplus_kminus_kres_u0<true>(iter, group_count);
				stage_s1_u1<true>(iter, group_count);
				stage_out1_u0_x1<true>(iter, group_count);
				stage_v_corr1_u1<true>(iter, group_count);
				stage_qplus_kplus_u0<true>(iter, group_count);
				stage_kminus_kres_u0<true>(iter, group_count);
				stage_iter_partial1_u1<true>(iter, group_count);
				stage_iter_final1_u1<true>(iter, group_count);
				stage_s1_u1_x1<true>(iter, group_count);
				stage_tril_u0<true>(iter, group_count);
				stage_masked_qk_u0<true>(iter, group_count);
				stage_v_corr1_u1_x1<true>(iter, group_count);
				stage_out1_u1<true>(iter, group_count);
				stage_tril_inv_u0<true>(iter, group_count);
				stage_iter_partial2_u1<true>(iter, group_count);
				stage_iter_final2_u1_x1<true>(iter, group_count);
				stage_mm_u0<true>(iter, group_count);
				stage_s1_u0_x1<true>(iter, group_count);
				stage_out1_u1_x1<true>(iter, group_count);
				stage_v_corr1_u0_x2<true>(iter, group_count);
				stage_tail36<true>(iter, group_count);
			}
			ProcessEdge(group_count, group_count);




			syncCV.Finalize();

			// final_state:末组状态已全部落回 slot 23(epilogue 的 vec_iter_final 收尾),
			// V 管同序读 s_ub,无需额外同步;MTE3 写出由 V_MTE3(0) 自保。
			if (kdaImp.out_fstate) {
				kdaImp.StoreFinalState((__ubuf__ void *)Ub(addrUB[s1_ub]*ub_addr_unit).GetPhyAddr());
			}
		}  // task 循环

	}
};


extern "C" __global__ __aicore__ void kda32_mix_aiv(uint64_t group_count, uint64_t scale, uint64_t gate_scale, __gm__ void *a,
	__gm__ void *q, __gm__ void *k, __gm__ void *v, __gm__ void *g, __gm__ void *beta, __gm__ void *dt,
	__gm__ void *o, __gm__ void *tmp)
{
	Kernel<bfloat16_t, false, true> kda;   // 遗留测试入口,保持 bf16 dense + 尾块支持
	//kda.Init(group_count, scale, q, k, q, v, g, beta, o, tmp);
	kda.kdaImp.Init(group_count, scale, gate_scale, (GM_ADDR)a, (GM_ADDR)g, (GM_ADDR)q, (GM_ADDR)k, (GM_ADDR)v, (GM_ADDR)beta, (GM_ADDR)dt, (GM_ADDR)tmp, (GM_ADDR)o);
	kda.Process(group_count);
}


#endif
