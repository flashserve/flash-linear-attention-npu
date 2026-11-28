#pragma once
#if defined(__DAV_CUBE__)
namespace OriginalGramA {
inline __aicore__ void inject_set_fmatrix(uint64_t value, uint64_t& last_value) {
  if (value != last_value) {
    last_value = value;
    set_fmatrix(value);
  }
}

inline __aicore__ void Compute(__gm__ float* __restrict__ tensor_a, __gm__ float* __restrict__ tensor_b, __gm__ float* __restrict__ tensor_c_gm, __gm__ uint8_t* __restrict__ tiling_data) {
set_ctrl(get_ctrl() & 0xffff3fffffffffff);
set_l1_3d_size(0);
set_padding(0);
set_mask_norm();
set_atomic_none();
uint64_t last_set_fmatrix_value = 0;
  for (int32_t i = 0; i < (int64_t)3; ++i) {
    dc_preload(((__gm__ uint64_t *)((__gm__ uint8_t *)tiling_data + (((int64_t)i) * (int64_t)64))), (int16_t)0);
  }
  int32_t k_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data)));
  int32_t m_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 4)));
  int32_t n_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 8)));
  int32_t k = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 12)));
  int32_t m = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 16)));
  int32_t n = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 20)));
  int32_t batch = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 24)));
  int32_t batch_a1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 28)));
  int32_t batch_a2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 32)));
  int32_t batch_a3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 36)));
  int32_t batch_a4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 40)));
  int32_t batch_b1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 44)));
  int32_t batch_b2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 48)));
  int32_t batch_b3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 52)));
  int32_t batch_b4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 56)));
  int32_t batch_c1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 60)));
  int32_t batch_c2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 64)));
  int32_t batch_c3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 68)));
  int32_t batch_c4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 72)));
  int32_t batch_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 76)));
  int32_t m_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 80)));
  int32_t n_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 84)));
  int32_t batch_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 88)));
  int32_t n_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 92)));
  int32_t m_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 96)));
  int32_t k_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 100)));
  int32_t m_al1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 104)));
  int32_t n_bl1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 108)));
  int32_t cub_n1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 112)));
  int32_t m_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 116)));
  int32_t k_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 120)));
  int32_t n_ub_l0_time = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 124)));
  int32_t kal0_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 128)));
  int32_t kbl0_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 132)));
  int32_t kal1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 136)));
  int32_t kbl1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 140)));
  int32_t kal1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 144)));
  int32_t kbl1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 148)));
  int32_t kl1_times = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 152)));
  int32_t batch_l1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 156)));
  int32_t batch_ub_l0_time = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 160)));
  int32_t batch_cub = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 164)));
  int32_t out_branch_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 168)));
  int32_t bias_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 172)));
  int32_t hf32_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 176)));
  int32_t datatype_bf16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 180)));
  int32_t al1_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 184)));
  int32_t bl1_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 188)));
  int32_t l0c_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 192)));
  int32_t l2_cache_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 196)));
  int32_t close_k_shift = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 200)));
  int64_t host_var = min(((((((((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)) * (int64_t)8) - (int64_t)1) >> (int64_t)4) + (int64_t)1), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4));
  int8_t host_var1 = ((int8_t)al1_db);
  int64_t host_var2 = ((((int64_t)(m_al1 * m_l0)) * (((((int64_t)(kal0_factor * k_l0)) * (int64_t)8) + (int64_t)15) >> (int64_t)4)) * (int64_t)8192);
  int64_t host_var3 = ((((int64_t)(m_al1 * m_l0)) * (((((int64_t)(kal0_factor * k_l0)) * (int64_t)8) + (int64_t)15) >> (int64_t)4)) * (int64_t)256);
  int64_t host_var4 = ((((int64_t)((n_bl1 * n_ub_l0_time) * cub_n1)) * (((((int64_t)(kal0_factor * k_l0)) * (int64_t)8) + (int64_t)15) >> (int64_t)4)) * (int64_t)256);
  int8_t host_var5 = ((int8_t)bl1_db);
  int64_t host_var6 = ((((int64_t)((m_al1 * m_l0) * ((((kal0_factor * k_l0) * 8) + 15) >> 4))) * (int64_t)8192) + (((int64_t)(((n_bl1 * n_ub_l0_time) * cub_n1) * ((((kal0_factor * k_l0) * 8) + 15) >> 4))) * (int64_t)8192));
  int64_t host_var7 = (((int64_t)((m_al1 * m_l0) * ((((kal0_factor * k_l0) * 8) + 15) >> 4))) * (int64_t)8192);
  int64_t host_var8 = ((int64_t)((m_dim * n_dim) * batch_dim));
  int64_t host_var9 = (((int64_t)((cub_n1 * m_l0) * n_ub_l0_time)) * (int64_t)256);
  int64_t host_var10 = ((((int64_t)m_l0) * ((((int64_t)k_l0) + (int64_t)1) >> (int64_t)1)) * (int64_t)256);
  int64_t host_var11 = ((((int64_t)(n_ub_l0_time * cub_n1)) * ((((int64_t)k_l0) + (int64_t)1) >> (int64_t)1)) * (int64_t)256);
  int64_t host_var12 = ((int64_t)(m_dim * n_dim));
  int64_t host_var13 = ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1);
  int64_t host_var14 = ((((int64_t)m_ori) + (int64_t)7) >> (int64_t)3);
  int64_t host_var15 = ((((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1) * ((int64_t)m_l0)) * (int64_t)2);
  int64_t host_var16 = ((int64_t)(n_ub_l0_time * cub_n1));
  int64_t host_var17 = ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1);
  int64_t host_var18 = ((((int64_t)n_ori) + (int64_t)7) >> (int64_t)3);
  int64_t host_var19 = ((((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((int64_t)(n_ub_l0_time * cub_n1))) * (int64_t)2);
  int64_t host_var20 = ((int64_t)((batch_c2 * batch_c3) * batch_c4));
  int64_t host_var21 = ((int64_t)(((batch_a2 * batch_a3) * batch_a4) * (batch_a1 / batch_c1)));
  int64_t host_var22 = ((int64_t)(batch_c3 * batch_c4));
  int64_t host_var23 = ((int64_t)((batch_a3 * batch_a4) * (batch_a2 / batch_c2)));
  int64_t host_var24 = ((int64_t)(batch_a4 * (batch_a3 / batch_c3)));
  int64_t host_var25 = ((int64_t)(batch_a4 / batch_c4));
  int64_t host_var26 = ((int64_t)(((batch_b2 * batch_b3) * batch_b4) * (batch_b1 / batch_c1)));
  int64_t host_var27 = ((int64_t)((batch_b3 * batch_b4) * (batch_b2 / batch_c2)));
  int64_t host_var28 = ((int64_t)(batch_b4 * (batch_b3 / batch_c3)));
  int64_t host_var29 = ((int64_t)(batch_b4 / batch_c4));
  int64_t host_var30 = ((int64_t)(int32_t)(min(k_ori, ((int32_t)(min(((((((((((k_ori + 7) >> 3) - 1) / k_l0) + 1) * k_l0) * 8) - 1) >> 4) + 1), ((k_ori + 15) >> 4))) * 16))));
  int64_t host_var31 = ((int64_t)m_ori);
  int64_t host_var32 = ((int64_t)(int32_t)(min(((((((((((k_ori + 7) >> 3) - 1) / k_l0) + 1) * k_l0) * 8) - 1) >> 4) + 1), ((k_ori + 15) >> 4))));
  int64_t host_var33 = min(((int64_t)k_ori), (min(((((((((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)) * (int64_t)8) - (int64_t)1) >> (int64_t)4) + (int64_t)1), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)16));
  int64_t host_var34 = (((int64_t)m_l0) * (int64_t)16);
  int64_t host_var35 = (min(((((((((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)) * (int64_t)8) - (int64_t)1) >> (int64_t)4) + (int64_t)1), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)128);
  int64_t host_var36 = ((((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1) * ((int64_t)m_l0)) * (int64_t)16);
  int64_t host_var37 = (((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1) * ((int64_t)m_l0));
  int64_t host_var38 = (min(((((((((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)) * (int64_t)8) - (int64_t)1) >> (int64_t)4) + (int64_t)1), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)16);
  int64_t host_var39 = ((int64_t)n_ori);
  int64_t host_var40 = (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16);
  int64_t host_var41 = ((((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((int64_t)(n_ub_l0_time * cub_n1))) * (int64_t)16);
  int64_t host_var42 = (((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((int64_t)(n_ub_l0_time * cub_n1)));
  int64_t host_var43 = ((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor));
  int64_t host_var44 = ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1));
  int64_t host_var45 = (((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1));
  int64_t host_var46 = (((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1);
  int64_t host_var47 = ((int64_t)(((batch_c1 * batch_c2) * batch_c3) * batch_c4));
  int64_t host_var48 = ((((int64_t)n_ori) + (int64_t)15) >> (int64_t)4);
  int64_t host_var49 = ((((int64_t)m_ori) + (int64_t)15) >> (int64_t)4);
  int64_t host_var50 = ((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3);
  int64_t host_var51 = ((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0));
  int64_t host_var52 = (((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1);
  int64_t host_var53 = ((int64_t)(int32_t)(max(m_dim, n_dim)));
  int64_t host_var54 = ((int64_t)(m % (m_al1 * m_l0)));
  int64_t host_var55 = ((int64_t)(m_al1 * m_l0));
  int64_t host_var56 = ((int64_t)(n % ((n_bl1 * n_ub_l0_time) * cub_n1)));
  int64_t host_var57 = ((int64_t)((n_bl1 * n_ub_l0_time) * cub_n1));
  uint64_t host_var58 = ((uint64_t)((int32_t)(min(((((((((((k_ori + 7) >> 3) - 1) / k_l0) + 1) * k_l0) * 8) - 1) >> 4) + 1), ((k_ori + 15) >> 4))) * 16));
  uint64_t host_var59 = ((uint64_t)hf32_flag);
  int64_t host_var60 = (((((int64_t)batch_l1_factor) * ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1))) * (((((((int64_t)k_ori) + (int64_t)7) >> (int64_t)3) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1)) - (int64_t)2);
  int64_t host_var61 = (((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (int64_t)1);
  int64_t host_var62 = (((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) + (int64_t)1);
  int64_t host_var63 = ((((int64_t)batch_l1_factor) * ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1))) - (int64_t)2);
  int64_t host_var64 = (((int64_t)(batch_l1_factor - al1_db)) - (int64_t)1);
  int64_t host_var65 = (((int64_t)(batch_l1_factor - bl1_db)) - (int64_t)1);
  int64_t host_var66 = ((((int64_t)(n_ub_l0_time * cub_n1)) * ((((int64_t)k_l0) + (int64_t)1) >> (int64_t)1)) * (int64_t)1024);
  int64_t host_var67 = ((((int64_t)m_l0) * ((((int64_t)k_l0) + (int64_t)1) >> (int64_t)1)) * (int64_t)1024);
  int64_t host_var68 = (((int64_t)((cub_n1 * m_l0) * n_ub_l0_time)) * (int64_t)1024);
  int64_t host_var69 = (((int64_t)((m_al1 * m_l0) * ((((kal0_factor * k_l0) * 8) + 15) >> 4))) * (int64_t)1024);
  int64_t hoist_addr_tensor_a_fractal_db_pong = (host_var1 ? host_var2 : 0);
__cbuf__   float* tensor_a_fractal_db_pong = (__cbuf__  float *)(((int32_t)((hoist_addr_tensor_a_fractal_db_pong + (int64_t)7) >> (int64_t)3)));
__cbuf__   float* tensor_b_fractal_db_ping = (__cbuf__  float *)(((int32_t)(host_var69 + ((hoist_addr_tensor_a_fractal_db_pong + (int64_t)7) >> (int64_t)3))));
  int64_t hoist_addr_tensor_b_fractal_db_pong = (host_var5 ? (host_var6 + hoist_addr_tensor_a_fractal_db_pong) : (host_var7 + hoist_addr_tensor_a_fractal_db_pong));
__cbuf__   float* tensor_b_fractal_db_pong = (__cbuf__  float *)(((int32_t)((hoist_addr_tensor_b_fractal_db_pong + (int64_t)7) >> (int64_t)3)));
// "aicore arch: Ascend910B3"
  if (((int64_t)((int32_t)block_idx)) < host_var8) {
    if (((int32_t)block_idx) < ((m_dim * n_dim) * batch_dim)) {
__cbuf__       float* tensor_a_fractal_db_ping = (__cbuf__  float *)get_imm(0);
__cc__       float* tensor_mmad_db_ping = (__cc__  float *)get_imm(0);
__cc__       float* tensor_mmad_db_pong = (__cc__  float *)(host_var68);
__ca__       float* tensor_a_zz_db_ping = (__ca__  float *)get_imm(0);
__ca__       float* tensor_a_zz_db_pong = (__ca__  float *)(host_var67);
__cb__       float* tensor_b_zn_db_ping = (__cb__  float *)get_imm(0);
__cb__       float* tensor_b_zn_db_pong = (__cb__  float *)(host_var66);
      int64_t i0_outer_outer = ((((int64_t)((int32_t)block_idx)) / host_var12) % ((int64_t)batch_dim));
      int64_t i2_outer_outer = ((int64_t)((((int32_t)block_idx) / m_dim) % n_dim));
      int64_t i1_outer_outer = ((int64_t)(((int32_t)block_idx) % m_dim));
      int64_t tensor_a_fractal_orig_bound_min_1 = (i1_outer_outer * host_var15);
      int64_t i1_ext = max(min(host_var15, (host_var14 - tensor_a_fractal_orig_bound_min_1)), (int64_t)0);
      int64_t tensor_b_fractal_orig_bound_min_1 = (i2_outer_outer * host_var19);
      int64_t i1_ext1 = max(min(host_var19, (host_var18 - tensor_b_fractal_orig_bound_min_1)), (int64_t)0);
      for (int64_t i0_outer_inner_outer_db = 0; i0_outer_inner_outer_db < ((int64_t)batch_l1_factor); ++i0_outer_inner_outer_db) {
        int8_t i0_outer_inner_outer_fold_db_var = ((i0_outer_inner_outer_db & 1) == 0);
        int64_t tensor_a_fractal_orig_bound_min_0 = (((((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var20) * host_var21) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var22) % ((int64_t)batch_a2)) * host_var23)) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / ((int64_t)batch_c4)) % ((int64_t)batch_a3)) * host_var24)) + ((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) % ((int64_t)batch_a4)) * host_var25));
        int64_t tensor_b_fractal_orig_bound_min_0 = (((((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var20) * host_var26) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var22) % ((int64_t)batch_b2)) * host_var27)) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / ((int64_t)batch_c4)) % ((int64_t)batch_b3)) * host_var28)) + ((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) % ((int64_t)batch_b4)) * host_var29));
        if (((int64_t)al1_db) < i0_outer_inner_outer_db) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)al1_db)) & 1) == 0) ? 0 : 1));
        }
        if ((tensor_a_fractal_orig_bound_min_0 < ((int64_t)batch)) && (((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) < ((int64_t)batch))) {
          if ((int64_t)0 < i1_ext) {
            if (m_ori < 8) {
              if ((host_var30 <= (int64_t)16384) && ((int64_t)1 <= (host_var31 - ((i1_outer_outer * ((int64_t)m_l0)) * (int64_t)16)))) {
                if ((int64_t)511 < host_var32) {
                  for (int64_t nd_num = 0; nd_num < i1_ext; ++nd_num) {
                    copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + (((host_var * max((tensor_a_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)128) + ((host_var * nd_num) * (int64_t)128)), ((__gm__ float *)tensor_a + ((((i1_outer_outer * ((int64_t)m_l0)) * (int64_t)16) + (nd_num * (int64_t)8)) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, (((int64_t)m_ori) - (i1_outer_outer * host_var34)), 1, m_ori, 1, 1, 1);
                  }
                } else {
                  copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_a + (((i1_outer_outer * ((int64_t)m_l0)) * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, i1_ext, host_var33, (((int64_t)m_ori) - (i1_outer_outer * host_var34)), 8, m_ori, 1, 1, host_var35);
                }
              }
            } else if ((min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)) == (int64_t)8) && (min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)) == ((int64_t)m_ori))) {
              copy_gm_to_cbuf((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_a + (((i1_outer_outer * host_var37) * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, ((((host_var33 * min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8))) * (int64_t)4) + (int64_t)31) >> (int64_t)5), 0, 0, PAD_NONE);
            } else if ((((host_var30 <= (int64_t)16384) && ((int64_t)1 <= min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)))) && (min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)) <= (int64_t)65535)) && (host_var32 <= (int64_t)1024)) {
              if ((int64_t)65535 < host_var31) {
                for (int64_t n_value = 0; n_value < host_var33; ++n_value) {
                  copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + (((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)128) + (n_value * (int64_t)8)), ((__gm__ float *)tensor_a + ((((i1_outer_outer * host_var37) * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori)))) + (((int64_t)m_ori) * n_value))), 0, 1, 1, min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)), 1, min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)), host_var38, 1, 1);
                }
              } else {
                copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_a + (((i1_outer_outer * host_var37) * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, min((((int64_t)m_ori) - (i1_outer_outer * host_var36)), (i1_ext * (int64_t)8)), 1, m_ori, host_var38, 1, 1);
              }
            }
          }
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        if (((int64_t)bl1_db) < i0_outer_inner_outer_db) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)bl1_db)) & 1) == 0) ? 2 : 3));
        }
        if ((tensor_b_fractal_orig_bound_min_0 < ((int64_t)batch)) && (((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) < ((int64_t)batch))) {
          if ((int64_t)0 < i1_ext1) {
            if (n_ori < 8) {
              if ((host_var30 <= (int64_t)16384) && ((int64_t)1 <= (host_var39 - ((i2_outer_outer * host_var16) * (int64_t)16)))) {
                if ((int64_t)511 < host_var32) {
                  for (int64_t nd_num1 = 0; nd_num1 < i1_ext1; ++nd_num1) {
                    copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + (((host_var * max((tensor_b_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)128) + ((host_var * nd_num1) * (int64_t)128)), ((__gm__ float *)tensor_b + ((((i2_outer_outer * host_var16) * (int64_t)16) + (nd_num1 * (int64_t)8)) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, (((int64_t)n_ori) - (i2_outer_outer * host_var40)), 1, n_ori, 1, 1, 1);
                  }
                } else {
                  copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_b + (((i2_outer_outer * host_var16) * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, i1_ext1, host_var33, (((int64_t)n_ori) - (i2_outer_outer * host_var40)), 8, n_ori, 1, 1, host_var35);
                }
              }
            } else if ((min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)) == (int64_t)8) && (min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)) == ((int64_t)n_ori))) {
              copy_gm_to_cbuf((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_b + (((i2_outer_outer * host_var42) * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, ((((host_var33 * min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8))) * (int64_t)4) + (int64_t)31) >> (int64_t)5), 0, 0, PAD_NONE);
            } else if ((((host_var30 <= (int64_t)16384) && ((int64_t)1 <= min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)))) && (min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)) <= (int64_t)65535)) && (host_var32 <= (int64_t)1024)) {
              if ((int64_t)65535 < host_var39) {
                for (int64_t n_value1 = 0; n_value1 < host_var33; ++n_value1) {
                  copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + (((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)128) + (n_value1 * (int64_t)8)), ((__gm__ float *)tensor_b + ((((i2_outer_outer * host_var42) * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori)))) + (((int64_t)n_ori) * n_value1))), 0, 1, 1, min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)), 1, min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)), host_var38, 1, 1);
                }
              } else {
                copy_gm_to_cbuf_multi_nd2nz_b32s((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)128), ((__gm__ float *)tensor_b + (((i2_outer_outer * host_var42) * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, min((((int64_t)n_ori) - (i2_outer_outer * host_var41)), (i1_ext1 * (int64_t)8)), 1, n_ori, host_var38, 1, 1);
              }
            }
          }
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        for (int64_t i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db = 0; i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db < host_var44; ++i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) {
          int64_t div_var2 = (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db / host_var45);
          int8_t i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var = ((((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0);
          int64_t i2_outer_inner = (((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var45)) / host_var13) + (((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var45)) % host_var13) >> (int64_t)63));
          int64_t i1_outer_inner = ((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var45)) - (i2_outer_inner * host_var13));
          int64_t tensor_mmad_orig_bound_min_0 = ((div_var2 + (i0_outer_inner_outer_db * host_var43)) + (i0_outer_outer * host_var46));
          int64_t tensor_mmad_extent_1 = (min((host_var16 * ((i2_outer_inner + (i2_outer_outer * host_var17)) + (int64_t)1)), host_var48) - min((host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17))), host_var48));
          int64_t tensor_mmad_orig_bound_min_1 = (host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17)));
          int64_t i2_ext = max(min(host_var16, (host_var48 - tensor_mmad_orig_bound_min_1)), (int64_t)0);
          int64_t n1_inner_ext = max(min(host_var16, (host_var48 - (host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17))))), (int64_t)0);
          int64_t i2_inner_ext = max(min(host_var40, (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16))), (int64_t)0);
          int64_t tensor_mmad_extent_2 = (min((((int64_t)m_l0) * ((i1_outer_inner + (i1_outer_outer * host_var13)) + (int64_t)1)), host_var49) - min((((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13))), host_var49));
          int64_t tensor_mmad_orig_bound_min_2 = (((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13)));
          int64_t i1_ext2 = max(min(((int64_t)m_l0), (host_var49 - tensor_mmad_orig_bound_min_2)), (int64_t)0);
          int64_t m1_inner_ext = max(min(((int64_t)m_l0), (host_var49 - (((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13))))), (int64_t)0);
          int64_t i1_inner_ext = max(min(host_var34, (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16))), (int64_t)0);
          if ((int64_t)1 < ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) {
            wait_flag(PIPE_FIX, PIPE_M, (((((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0) ? 0 : 1));
          }
          for (int64_t k1_outer_db = 0; k1_outer_db < host_var52; ++k1_outer_db) {
            int64_t bind_block = (((n_dim <= m_dim) || ((int64_t)24 < host_var8)) ? i1_outer_outer : i2_outer_outer);
            int64_t bind_block_ext = (((int64_t)24 < host_var8) ? (int64_t)8 : host_var53);
            int8_t overlap_cond = ((((((int64_t)24 < host_var8) ? (int64_t)8 : host_var53) == ((int64_t)m_dim)) || ((int64_t)24 < (((int64_t)24 < host_var8) ? (int64_t)8 : host_var53))) ? (((host_var54 != (int64_t)0) && (((int64_t)m) < ((i1_outer_outer + (int64_t)2) * host_var55))) || (1 <= close_k_shift)) : (((host_var56 != (int64_t)0) && (((int64_t)n) < ((i2_outer_outer + (int64_t)2) * host_var57))) || (1 <= close_k_shift)));
            int64_t k1_outer_db_shift = (((((int64_t)4 <= bind_block_ext) && ((bind_block_ext / (int64_t)2) <= host_var52)) && !overlap_cond) ? (((((bind_block % (int64_t)2) == (int64_t)0) ? k1_outer_db : (host_var51 - k1_outer_db)) + ((host_var52 / (bind_block_ext / (int64_t)2)) * ((bind_block / (int64_t)2) % (bind_block_ext / (int64_t)2)))) % host_var52) : k1_outer_db);
            int8_t k1_outer_fold_db_var = ((((host_var52 * ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0);
            int64_t tensor_a_zz_extent_2 = min(((int64_t)k_l0), (host_var50 - (k1_outer_db_shift * ((int64_t)k_l0))));
            int64_t tensor_a_zz_orig_bound_min_2 = (k1_outer_db_shift * ((int64_t)k_l0));
            int64_t i2_ext1 = max(min(((int64_t)k_l0), (host_var50 - tensor_a_zz_orig_bound_min_2)), (int64_t)0);
            if ((int64_t)1 < ((host_var52 * ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db)) {
              wait_flag(PIPE_M, PIPE_MTE1, (((((host_var52 * ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0) ? 0 : 1));
            }
            if (tensor_a_fractal_orig_bound_min_0 < ((int64_t)batch)) {
              if (((int64_t)0 < i1_ext2) && ((int64_t)0 < i2_ext1)) {
                inject_set_fmatrix((host_var58 | (uint64_t)65536ULL), last_set_fmatrix_value);
                img2colv2_cbuf_to_ca((k1_outer_fold_db_var ? ((__ca__ float *)tensor_a_zz_db_ping) : ((__ca__ float *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * max((tensor_mmad_orig_bound_min_2 - host_var49), (int64_t)0)) * (int64_t)128), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_a_fractal_db_ping) : ((__cbuf__ float *)tensor_a_fractal_db_pong)) + (((host_var * ((tensor_mmad_orig_bound_min_2 * (int64_t)2) - min(tensor_a_fractal_orig_bound_min_1, host_var14))) * (int64_t)128) + (tensor_a_zz_orig_bound_min_2 * (int64_t)64)), (i1_ext2 * (int64_t)16), (i2_ext1 * (int64_t)8), 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, (i1_ext2 * (int64_t)16));
              }
            }
            if ((k1_outer_db == (int64_t)0) && (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db == (int64_t)0)) {
              wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            }
            if (tensor_b_fractal_orig_bound_min_0 < ((int64_t)batch)) {
              if (((int64_t)0 < i2_ext1) && ((int64_t)0 < i2_ext)) {
                set_fmatrix_b((host_var58 | (uint64_t)65536ULL));
                img2colv2_cbuf_to_cb((k1_outer_fold_db_var ? ((__cb__ float *)tensor_b_zn_db_ping) : ((__cb__ float *)tensor_b_zn_db_pong)) + (max((tensor_mmad_orig_bound_min_1 - host_var48), (int64_t)0) * (int64_t)128), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ float *)tensor_b_fractal_db_ping) : ((__cbuf__ float *)tensor_b_fractal_db_pong)) + (((host_var * ((tensor_mmad_orig_bound_min_1 * (int64_t)2) - min(tensor_b_fractal_orig_bound_min_1, host_var18))) * (int64_t)128) + (tensor_a_zz_orig_bound_min_2 * (int64_t)64)), (i2_ext * (int64_t)16), (i2_ext1 * (int64_t)8), 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, (i2_ext * (int64_t)16));
              }
            }
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (((div_var2 + (i0_outer_inner_outer_db * host_var43)) + (i0_outer_outer * host_var46)) < host_var47) {
              if ((((int64_t)0 < n1_inner_ext) && ((int64_t)0 < m1_inner_ext)) && ((int64_t)0 < i2_ext1)) {
                set_ctrl((get_ctrl() | ((host_var59 << (uint64_t)46ULL) | (uint64_t)0ULL)));
                if (((int64_t)0 < ((((tensor_a_zz_orig_bound_min_2 * (int64_t)8) - ((int64_t)k_ori)) < (int64_t)0) ? (((int64_t)k_ori) - (tensor_a_zz_orig_bound_min_2 * (int64_t)8)) : ((tensor_a_zz_orig_bound_min_2 * (int64_t)8) - ((int64_t)k_ori)))) && ((int64_t)0 < ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori))))) {
                  mad((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + ((((tensor_mmad_extent_2 * (((max((tensor_mmad_orig_bound_min_0 - host_var47), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var48))) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var49) * (int64_t)256)), (k1_outer_fold_db_var ? ((__ca__ float *)tensor_a_zz_db_ping) : ((__ca__ float *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * (((((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var21) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_a2)) * host_var23)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_a3)) * host_var24)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_a4)) * host_var25)) - min(tensor_a_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_mmad_extent_2) + tensor_mmad_orig_bound_min_2) - min(tensor_mmad_orig_bound_min_2, host_var49))) * (int64_t)128), (k1_outer_fold_db_var ? ((__cb__ float *)tensor_b_zn_db_ping) : ((__cb__ float *)tensor_b_zn_db_pong)) + ((((tensor_mmad_extent_1 * (((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var26) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_b2)) * host_var27)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_b3)) * host_var28)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_b4)) * host_var29)) - min(tensor_b_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_a_zz_extent_2)) * (int64_t)128) + (tensor_mmad_orig_bound_min_1 * (int64_t)128)) - (min(tensor_mmad_orig_bound_min_1, host_var48) * (int64_t)128)), max(min((m1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)) < (int64_t)0) ? (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)))), (int64_t)2), min((i2_ext1 * (int64_t)8), ((((tensor_a_zz_orig_bound_min_2 * (int64_t)8) - ((int64_t)k_ori)) < (int64_t)0) ? (((int64_t)k_ori) - (tensor_a_zz_orig_bound_min_2 * (int64_t)8)) : ((tensor_a_zz_orig_bound_min_2 * (int64_t)8) - ((int64_t)k_ori)))), min((n1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)))), (uint8_t)0ULL, 1, 0, (k1_outer_db == (int64_t)0));
                  if ((max(min((m1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)) < (int64_t)0) ? (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)))), (int64_t)2) * min((n1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori))))) < (int64_t)2560) {
                    pipe_barrier(PIPE_M);
                  }
                }
                set_ctrl((get_ctrl() & (~((host_var59 << (uint64_t)46ULL) | (uint64_t)0ULL))));
              }
            }
            if (((host_var52 * ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) < host_var60) {
              set_flag(PIPE_M, PIPE_MTE1, (((((host_var52 * ((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0) ? 0 : 1));
            }
          }
          set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          if ((((((div_var2 + (i0_outer_inner_outer_db * host_var43)) + (i0_outer_outer * host_var46)) < ((int64_t)batch)) && (tensor_mmad_orig_bound_min_0 < ((int64_t)batch))) && ((i1_outer_inner + (i1_outer_outer * host_var13)) < host_var61)) && ((i2_outer_inner + (i2_outer_outer * host_var17)) < host_var62)) {
            if (((int64_t)0 < i1_inner_ext) && ((int64_t)0 < i2_inner_ext)) {
              if (i2_inner_ext <= (int64_t)8192) {
                set_nd_para((uint64_t)4295032833ULL);
                pipe_barrier(PIPE_FIX);
                copy_matrix_cc_to_gm(((__gm__ float *)tensor_c_gm + ((tensor_mmad_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * ((tensor_mmad_orig_bound_min_2 * (int64_t)16) + (tensor_mmad_orig_bound_min_0 * ((int64_t)m_ori)))))), (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + (((((((max((tensor_mmad_orig_bound_min_0 - host_var47), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var48)) * tensor_mmad_extent_2) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var49) * (int64_t)256)), (uint8_t)0ULL, i2_inner_ext, i1_inner_ext, n_ori, (tensor_mmad_extent_2 * (int64_t)16), 0, NoQuant, 0, 0, 1);
              }
            }
          }
          if (((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) < host_var63) {
            set_flag(PIPE_FIX, PIPE_M, (((((i0_outer_inner_outer_db * host_var44) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0) ? 0 : 1));
          }
        }
        if (i0_outer_inner_outer_db < host_var64) {
          set_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)al1_db)) & 1) == 0) ? 0 : 1));
        }
        if (i0_outer_inner_outer_db < host_var65) {
          set_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)bl1_db)) & 1) == 0) ? 2 : 3));
        }
      }
    }
  }
  pipe_barrier(PIPE_ALL);
  pipe_barrier(PIPE_ALL);

  uint64_t origin_fpc = get_fpc();
  uint64_t final_fpc = origin_fpc | 0x8000000000000000;
  set_fpc(final_fpc);
}
inline __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR tile){Compute((__gm__ float*)a,(__gm__ float*)b,(__gm__ float*)c,tile);}
}
namespace OriginalGramB {
inline __aicore__ void Compute(__gm__ half* __restrict__ tensor_a, __gm__ half* __restrict__ tensor_b, __gm__ half* __restrict__ tensor_c_gm, __gm__ uint8_t* __restrict__ tiling_data) {
set_ctrl(get_ctrl() & 0xffff3fffffffffff);
set_padding(0);
set_mask_norm();
set_atomic_none();
  for (int32_t i = 0; i < (int64_t)3; ++i) {
    dc_preload(((__gm__ uint64_t *)((__gm__ uint8_t *)tiling_data + (((int64_t)i) * (int64_t)64))), (int16_t)0);
  }
  int32_t k_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data)));
  int32_t m_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 4)));
  int32_t n_ori = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 8)));
  int32_t k = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 12)));
  int32_t m = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 16)));
  int32_t n = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 20)));
  int32_t batch = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 24)));
  int32_t batch_a1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 28)));
  int32_t batch_a2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 32)));
  int32_t batch_a3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 36)));
  int32_t batch_a4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 40)));
  int32_t batch_b1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 44)));
  int32_t batch_b2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 48)));
  int32_t batch_b3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 52)));
  int32_t batch_b4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 56)));
  int32_t batch_c1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 60)));
  int32_t batch_c2 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 64)));
  int32_t batch_c3 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 68)));
  int32_t batch_c4 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 72)));
  int32_t batch_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 76)));
  int32_t m_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 80)));
  int32_t n_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 84)));
  int32_t batch_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 88)));
  int32_t n_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 92)));
  int32_t m_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 96)));
  int32_t k_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 100)));
  int32_t m_al1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 104)));
  int32_t n_bl1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 108)));
  int32_t cub_n1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 112)));
  int32_t m_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 116)));
  int32_t k_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 120)));
  int32_t n_ub_l0_time = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 124)));
  int32_t kal0_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 128)));
  int32_t kbl0_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 132)));
  int32_t kal1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 136)));
  int32_t kbl1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 140)));
  int32_t kal1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 144)));
  int32_t kbl1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 148)));
  int32_t kl1_times = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 152)));
  int32_t batch_l1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 156)));
  int32_t batch_ub_l0_time = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 160)));
  int32_t batch_cub = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 164)));
  int32_t out_branch_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 168)));
  int32_t bias_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 172)));
  int32_t hf32_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 176)));
  int32_t datatype_bf16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 180)));
  int32_t al1_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 184)));
  int32_t bl1_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 188)));
  int32_t l0c_db = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 192)));
  int32_t l2_cache_flag = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 196)));
  int32_t close_k_shift = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 200)));
  int64_t host_var = min(((((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4));
  int8_t host_var1 = ((int8_t)al1_db);
  int64_t host_var2 = (((int64_t)((kal0_factor * k_l0) * (m_al1 * m_l0))) * (int64_t)4096);
  int64_t host_var3 = (((int64_t)((kal0_factor * k_l0) * (m_al1 * m_l0))) * (int64_t)256);
  int64_t host_var4 = (((int64_t)((kal0_factor * k_l0) * ((n_bl1 * n_ub_l0_time) * cub_n1))) * (int64_t)256);
  int8_t host_var5 = ((int8_t)bl1_db);
  int64_t host_var6 = ((((int64_t)((kal0_factor * k_l0) * (m_al1 * m_l0))) * (int64_t)4096) + (((int64_t)((kal0_factor * k_l0) * ((n_bl1 * n_ub_l0_time) * cub_n1))) * (int64_t)4096));
  int64_t host_var7 = (((int64_t)((kal0_factor * k_l0) * (m_al1 * m_l0))) * (int64_t)4096);
  int64_t host_var8 = ((int64_t)((m_dim * n_dim) * batch_dim));
  int64_t host_var9 = (((int64_t)((cub_n1 * m_l0) * n_ub_l0_time)) * (int64_t)256);
  int64_t host_var10 = (((int64_t)(m_l0 * k_l0)) * (int64_t)256);
  int64_t host_var11 = (((int64_t)(k_l0 * (n_ub_l0_time * cub_n1))) * (int64_t)256);
  int64_t host_var12 = ((int64_t)(m_dim * n_dim));
  int64_t host_var13 = ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1);
  int64_t host_var14 = ((((int64_t)m_ori) + (int64_t)15) >> (int64_t)4);
  int64_t host_var15 = (((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1) * ((int64_t)m_l0));
  int64_t host_var16 = ((int64_t)(n_ub_l0_time * cub_n1));
  int64_t host_var17 = ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1);
  int64_t host_var18 = ((((int64_t)n_ori) + (int64_t)15) >> (int64_t)4);
  int64_t host_var19 = (((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((int64_t)(n_ub_l0_time * cub_n1)));
  int64_t host_var20 = ((int64_t)((batch_c2 * batch_c3) * batch_c4));
  int64_t host_var21 = ((int64_t)(((batch_a2 * batch_a3) * batch_a4) * (batch_a1 / batch_c1)));
  int64_t host_var22 = ((int64_t)(batch_c3 * batch_c4));
  int64_t host_var23 = ((int64_t)((batch_a3 * batch_a4) * (batch_a2 / batch_c2)));
  int64_t host_var24 = ((int64_t)(batch_a4 * (batch_a3 / batch_c3)));
  int64_t host_var25 = ((int64_t)(batch_a4 / batch_c4));
  int64_t host_var26 = ((int64_t)(((batch_b2 * batch_b3) * batch_b4) * (batch_b1 / batch_c1)));
  int64_t host_var27 = ((int64_t)((batch_b3 * batch_b4) * (batch_b2 / batch_c2)));
  int64_t host_var28 = ((int64_t)(batch_b4 * (batch_b3 / batch_c3)));
  int64_t host_var29 = ((int64_t)(batch_b4 / batch_c4));
  int64_t host_var30 = ((int64_t)(int32_t)(min(k_ori, ((int32_t)(min(((((((k_ori + 15) >> 4) - 1) / k_l0) + 1) * k_l0), ((k_ori + 15) >> 4))) * 16))));
  int64_t host_var31 = ((int64_t)m_ori);
  int64_t host_var32 = ((int64_t)(int32_t)(min(((((((k_ori + 15) >> 4) - 1) / k_l0) + 1) * k_l0), ((k_ori + 15) >> 4))));
  int64_t host_var33 = min(((int64_t)k_ori), (min(((((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)16));
  int64_t host_var34 = (min(((((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)256);
  int64_t host_var35 = (min(((((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1) * ((int64_t)k_l0)), ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4)) * (int64_t)16);
  int64_t host_var36 = ((int64_t)n_ori);
  int64_t host_var37 = ((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor));
  int64_t host_var38 = ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1));
  int64_t host_var39 = (((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1));
  int64_t host_var40 = (((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1);
  int64_t host_var41 = ((int64_t)(((batch_c1 * batch_c2) * batch_c3) * batch_c4));
  int64_t host_var42 = (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16);
  int64_t host_var43 = (((int64_t)m_l0) * (int64_t)16);
  int64_t host_var44 = ((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4);
  int64_t host_var45 = ((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0));
  int64_t host_var46 = (((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1);
  int64_t host_var47 = ((int64_t)(int32_t)(max(m_dim, n_dim)));
  int64_t host_var48 = ((int64_t)(m % (m_al1 * m_l0)));
  int64_t host_var49 = ((int64_t)(m_al1 * m_l0));
  int64_t host_var50 = ((int64_t)(n % ((n_bl1 * n_ub_l0_time) * cub_n1)));
  int64_t host_var51 = ((int64_t)((n_bl1 * n_ub_l0_time) * cub_n1));
  int64_t host_var52 = (((((int64_t)batch_l1_factor) * ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1))) * (((((((int64_t)k_ori) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1)) - (int64_t)2);
  int64_t host_var53 = (((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (int64_t)1);
  int64_t host_var54 = (((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) + (int64_t)1);
  int64_t host_var55 = ((((int64_t)batch_l1_factor) * ((((((((int64_t)batch) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) / ((int64_t)batch_l1_factor)) * ((((((int64_t)n_ori) - (int64_t)1) / (((int64_t)(n_ub_l0_time * cub_n1)) * (int64_t)16)) / ((int64_t)n_dim)) + (int64_t)1)) * ((((((int64_t)m_ori) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_dim)) + (int64_t)1))) - (int64_t)2);
  int64_t host_var56 = (((int64_t)(batch_l1_factor - al1_db)) - (int64_t)1);
  int64_t host_var57 = (((int64_t)(batch_l1_factor - bl1_db)) - (int64_t)1);
  int64_t host_var58 = (((int64_t)(k_l0 * (n_ub_l0_time * cub_n1))) * (int64_t)512);
  int64_t host_var59 = (((int64_t)(m_l0 * k_l0)) * (int64_t)512);
  int64_t host_var60 = (((int64_t)((cub_n1 * m_l0) * n_ub_l0_time)) * (int64_t)1024);
  int64_t host_var61 = (((int64_t)((kal0_factor * k_l0) * (m_al1 * m_l0))) * (int64_t)512);
  int64_t hoist_addr_tensor_a_fractal_db_pong = (host_var1 ? host_var2 : 0);
__cbuf__   half* tensor_a_fractal_db_pong = (__cbuf__  half *)(((int32_t)((hoist_addr_tensor_a_fractal_db_pong + (int64_t)7) >> (int64_t)3)));
__cbuf__   half* tensor_b_fractal_db_ping = (__cbuf__  half *)(((int32_t)(host_var61 + ((hoist_addr_tensor_a_fractal_db_pong + (int64_t)7) >> (int64_t)3))));
  int64_t hoist_addr_tensor_b_fractal_db_pong = (host_var5 ? (host_var6 + hoist_addr_tensor_a_fractal_db_pong) : (host_var7 + hoist_addr_tensor_a_fractal_db_pong));
__cbuf__   half* tensor_b_fractal_db_pong = (__cbuf__  half *)(((int32_t)((hoist_addr_tensor_b_fractal_db_pong + (int64_t)7) >> (int64_t)3)));
// "aicore arch: Ascend910B3"
  if (((int64_t)((int32_t)block_idx)) < host_var8) {
    if (((int32_t)block_idx) < ((m_dim * n_dim) * batch_dim)) {
__cbuf__       half* tensor_a_fractal_db_ping = (__cbuf__  half *)get_imm(0);
__cc__       float* tensor_mmad_db_ping = (__cc__  float *)get_imm(0);
__cc__       float* tensor_mmad_db_pong = (__cc__  float *)(host_var60);
__ca__       half* tensor_a_zz_db_ping = (__ca__  half *)get_imm(0);
__ca__       half* tensor_a_zz_db_pong = (__ca__  half *)(host_var59);
__cb__       half* tensor_b_zn_db_ping = (__cb__  half *)get_imm(0);
__cb__       half* tensor_b_zn_db_pong = (__cb__  half *)(host_var58);
      int64_t i0_outer_outer = ((((int64_t)((int32_t)block_idx)) / host_var12) % ((int64_t)batch_dim));
      int64_t i2_outer_outer = ((int64_t)((((int32_t)block_idx) / m_dim) % n_dim));
      int64_t i1_outer_outer = ((int64_t)(((int32_t)block_idx) % m_dim));
      int64_t tensor_a_fractal_orig_bound_min_1 = (i1_outer_outer * host_var15);
      int64_t i1_ext = max(min(host_var15, (host_var14 - tensor_a_fractal_orig_bound_min_1)), (int64_t)0);
      int64_t tensor_b_fractal_orig_bound_min_1 = (i2_outer_outer * host_var19);
      int64_t i1_ext1 = max(min(host_var19, (host_var18 - tensor_b_fractal_orig_bound_min_1)), (int64_t)0);
      for (int64_t i0_outer_inner_outer_db = 0; i0_outer_inner_outer_db < ((int64_t)batch_l1_factor); ++i0_outer_inner_outer_db) {
        int8_t i0_outer_inner_outer_fold_db_var = ((i0_outer_inner_outer_db & 1) == 0);
        int64_t tensor_a_fractal_orig_bound_min_0 = (((((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var20) * host_var21) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var22) % ((int64_t)batch_a2)) * host_var23)) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / ((int64_t)batch_c4)) % ((int64_t)batch_a3)) * host_var24)) + ((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) % ((int64_t)batch_a4)) * host_var25));
        int64_t tensor_b_fractal_orig_bound_min_0 = (((((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var20) * host_var26) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / host_var22) % ((int64_t)batch_b2)) * host_var27)) + (((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) / ((int64_t)batch_c4)) % ((int64_t)batch_b3)) * host_var28)) + ((((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) % ((int64_t)batch_b4)) * host_var29));
        if (((int64_t)al1_db) < i0_outer_inner_outer_db) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)al1_db)) & 1) == 0) ? 0 : 1));
        }
        if ((tensor_a_fractal_orig_bound_min_0 < ((int64_t)batch)) && (((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) < ((int64_t)batch))) {
          if ((int64_t)0 < i1_ext) {
            if (m_ori < 16) {
              if ((host_var30 <= (int64_t)16384) && ((int64_t)1 <= (host_var31 - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)))) {
                if ((int64_t)255 < host_var32) {
                  for (int64_t nd_num = 0; nd_num < i1_ext; ++nd_num) {
                    copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + (((host_var * max((tensor_a_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)256) + ((host_var * nd_num) * (int64_t)256)), ((__gm__ half *)tensor_a + (((tensor_a_fractal_orig_bound_min_1 * (int64_t)16) + (nd_num * (int64_t)16)) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, (((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), 1, m_ori, 1, 1, 1);
                  }
                } else {
                  copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_a + ((tensor_a_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, i1_ext, host_var33, (((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), 16, m_ori, 1, 1, host_var34);
                }
              }
            } else if ((min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)) == (int64_t)16) && (min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)) == ((int64_t)m_ori))) {
              copy_gm_to_cbuf((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_a + ((tensor_a_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, ((((host_var33 * min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16))) * (int64_t)2) + (int64_t)31) >> (int64_t)5), 0, 0, PAD_NONE);
            } else if ((((host_var30 <= (int64_t)16384) && ((int64_t)1 <= min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)))) && (min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)) <= (int64_t)65535)) && (host_var32 <= (int64_t)1024)) {
              if ((int64_t)65535 < host_var31) {
                for (int64_t n_value = 0; n_value < host_var33; ++n_value) {
                  copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + (((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)256) + (n_value * (int64_t)16)), ((__gm__ half *)tensor_a + (((tensor_a_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori)))) + (((int64_t)m_ori) * n_value))), 0, 1, 1, min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)), 1, min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)), host_var35, 1, 1);
                }
              } else {
                copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + ((host_var * max((tensor_a_fractal_orig_bound_min_1 - host_var14), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_a + ((tensor_a_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)m_ori) * (tensor_a_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, min((((int64_t)m_ori) - (tensor_a_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext * (int64_t)16)), 1, m_ori, host_var35, 1, 1);
              }
            }
          }
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        if (((int64_t)bl1_db) < i0_outer_inner_outer_db) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)bl1_db)) & 1) == 0) ? 2 : 3));
        }
        if ((tensor_b_fractal_orig_bound_min_0 < ((int64_t)batch)) && (((i0_outer_outer * ((int64_t)batch_single_core)) + i0_outer_inner_outer_db) < ((int64_t)batch))) {
          if ((int64_t)0 < i1_ext1) {
            if (n_ori < 16) {
              if ((host_var30 <= (int64_t)16384) && ((int64_t)1 <= (host_var36 - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)))) {
                if ((int64_t)255 < host_var32) {
                  for (int64_t nd_num1 = 0; nd_num1 < i1_ext1; ++nd_num1) {
                    copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + (((host_var * max((tensor_b_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)256) + ((host_var * nd_num1) * (int64_t)256)), ((__gm__ half *)tensor_b + (((tensor_b_fractal_orig_bound_min_1 * (int64_t)16) + (nd_num1 * (int64_t)16)) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, (((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), 1, n_ori, 1, 1, 1);
                  }
                } else {
                  copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - (int64_t)1), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_b + ((tensor_b_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, i1_ext1, host_var33, (((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), 16, n_ori, 1, 1, host_var34);
                }
              }
            } else if ((min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)) == (int64_t)16) && (min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)) == ((int64_t)n_ori))) {
              copy_gm_to_cbuf((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_b + ((tensor_b_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, ((((host_var33 * min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16))) * (int64_t)2) + (int64_t)31) >> (int64_t)5), 0, 0, PAD_NONE);
            } else if ((((host_var30 <= (int64_t)16384) && ((int64_t)1 <= min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)))) && (min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)) <= (int64_t)65535)) && (host_var32 <= (int64_t)1024)) {
              if ((int64_t)65535 < host_var36) {
                for (int64_t n_value1 = 0; n_value1 < host_var33; ++n_value1) {
                  copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + (((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)256) + (n_value1 * (int64_t)16)), ((__gm__ half *)tensor_b + (((tensor_b_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori)))) + (((int64_t)n_ori) * n_value1))), 0, 1, 1, min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)), 1, min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)), host_var35, 1, 1);
                }
              } else {
                copy_gm_to_cbuf_multi_nd2nz_b16((i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + ((host_var * max((tensor_b_fractal_orig_bound_min_1 - host_var18), (int64_t)0)) * (int64_t)256), ((__gm__ half *)tensor_b + ((tensor_b_fractal_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * (tensor_b_fractal_orig_bound_min_0 * ((int64_t)k_ori))))), 0, 1, host_var33, min((((int64_t)n_ori) - (tensor_b_fractal_orig_bound_min_1 * (int64_t)16)), (i1_ext1 * (int64_t)16)), 1, n_ori, host_var35, 1, 1);
              }
            }
          }
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        for (int64_t i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db = 0; i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db < host_var38; ++i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) {
          int64_t div_var2 = (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db / host_var39);
          int8_t i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var = ((((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0);
          int64_t i2_outer_inner = (((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var39)) / host_var13) + (((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var39)) % host_var13) >> (int64_t)63));
          int64_t i1_outer_inner = ((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db - (div_var2 * host_var39)) - (i2_outer_inner * host_var13));
          int64_t tensor_mmad_orig_bound_min_0 = ((div_var2 + (i0_outer_inner_outer_db * host_var37)) + (i0_outer_outer * host_var40));
          int64_t tensor_mmad_extent_1 = (min((host_var16 * ((i2_outer_inner + (i2_outer_outer * host_var17)) + (int64_t)1)), host_var18) - min((host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17))), host_var18));
          int64_t tensor_mmad_orig_bound_min_1 = (host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17)));
          int64_t n1_ext = max(min(host_var16, (host_var18 - tensor_mmad_orig_bound_min_1)), (int64_t)0);
          int64_t n1_inner_ext = max(min(host_var16, (host_var18 - (host_var16 * (i2_outer_inner + (i2_outer_outer * host_var17))))), (int64_t)0);
          int64_t i2_inner_ext = max(min(host_var42, (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16))), (int64_t)0);
          int64_t tensor_mmad_extent_2 = (min((((int64_t)m_l0) * ((i1_outer_inner + (i1_outer_outer * host_var13)) + (int64_t)1)), host_var14) - min((((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13))), host_var14));
          int64_t tensor_mmad_orig_bound_min_2 = (((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13)));
          int64_t m1_ext = max(min(((int64_t)m_l0), (host_var14 - tensor_mmad_orig_bound_min_2)), (int64_t)0);
          int64_t m1_inner_ext = max(min(((int64_t)m_l0), (host_var14 - (((int64_t)m_l0) * (i1_outer_inner + (i1_outer_outer * host_var13))))), (int64_t)0);
          int64_t i1_inner_ext = max(min(host_var43, (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16))), (int64_t)0);
          if ((int64_t)1 < ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) {
            wait_flag(PIPE_FIX, PIPE_M, (((((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0) ? 0 : 1));
          }
          for (int64_t k1_outer_db = 0; k1_outer_db < host_var46; ++k1_outer_db) {
            int64_t bind_block = (((n_dim <= m_dim) || ((int64_t)24 < host_var8)) ? i1_outer_outer : i2_outer_outer);
            int64_t bind_block_ext = (((int64_t)24 < host_var8) ? (int64_t)8 : host_var47);
            int8_t overlap_cond = ((((((int64_t)24 < host_var8) ? (int64_t)8 : host_var47) == ((int64_t)m_dim)) || ((int64_t)24 < (((int64_t)24 < host_var8) ? (int64_t)8 : host_var47))) ? (((host_var48 != (int64_t)0) && (((int64_t)m) < ((i1_outer_outer + (int64_t)2) * host_var49))) || (1 <= close_k_shift)) : (((host_var50 != (int64_t)0) && (((int64_t)n) < ((i2_outer_outer + (int64_t)2) * host_var51))) || (1 <= close_k_shift)));
            int64_t k1_outer_db_shift = (((((int64_t)4 <= bind_block_ext) && ((bind_block_ext / (int64_t)2) <= host_var46)) && !overlap_cond) ? (((((bind_block % (int64_t)2) == (int64_t)0) ? k1_outer_db : (host_var45 - k1_outer_db)) + ((host_var46 / (bind_block_ext / (int64_t)2)) * ((bind_block / (int64_t)2) % (bind_block_ext / (int64_t)2)))) % host_var46) : k1_outer_db);
            int8_t k1_outer_fold_db_var = ((((host_var46 * ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0);
            int64_t tensor_a_zz_extent_2 = min(((int64_t)k_l0), (host_var44 - (k1_outer_db_shift * ((int64_t)k_l0))));
            int64_t tensor_a_zz_orig_bound_min_2 = (k1_outer_db_shift * ((int64_t)k_l0));
            int64_t k1_ext = max(min(((int64_t)k_l0), (host_var44 - tensor_a_zz_orig_bound_min_2)), (int64_t)0);
            if ((int64_t)1 < ((host_var46 * ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db)) {
              wait_flag(PIPE_M, PIPE_MTE1, (((((host_var46 * ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0) ? 0 : 1));
            }
            if (tensor_a_fractal_orig_bound_min_0 < ((int64_t)batch)) {
              if (((int64_t)0 < m1_ext) && ((int64_t)0 < k1_ext)) {
                pipe_barrier(PIPE_MTE1);
                if (((tensor_a_zz_extent_2 * (int64_t)256) == (k1_ext * (int64_t)256)) && (host_var34 == (k1_ext * (int64_t)256))) {
                  load_cbuf_to_ca((k1_outer_fold_db_var ? ((__ca__ half *)tensor_a_zz_db_ping) : ((__ca__ half *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * max((tensor_mmad_orig_bound_min_2 - host_var14), (int64_t)0)) * (int64_t)256), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + (((host_var * (tensor_mmad_orig_bound_min_2 - min(tensor_a_fractal_orig_bound_min_1, host_var14))) * (int64_t)256) + (tensor_a_zz_orig_bound_min_2 * (int64_t)256)), 0, (m1_ext * k1_ext), 1, 0, 0, 1, inc);
                } else {
                  for (int64_t copy_part_0 = 0; copy_part_0 < m1_ext; ++copy_part_0) {
                    load_cbuf_to_ca((k1_outer_fold_db_var ? ((__ca__ half *)tensor_a_zz_db_ping) : ((__ca__ half *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * (max((tensor_mmad_orig_bound_min_2 - host_var14), (int64_t)0) + copy_part_0)) * (int64_t)256), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_a_fractal_db_ping) : ((__cbuf__ half *)tensor_a_fractal_db_pong)) + ((((host_var * (tensor_mmad_orig_bound_min_2 - min(tensor_a_fractal_orig_bound_min_1, host_var14))) * (int64_t)256) + (tensor_a_zz_orig_bound_min_2 * (int64_t)256)) + ((host_var * copy_part_0) * (int64_t)256)), 0, k1_ext, 1, 0, 0, 1, inc);
                  }
                }
              }
            }
            if ((k1_outer_db == (int64_t)0) && (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db == (int64_t)0)) {
              wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            }
            if (tensor_b_fractal_orig_bound_min_0 < ((int64_t)batch)) {
              if (((int64_t)0 < k1_ext) && ((int64_t)0 < n1_ext)) {
                pipe_barrier(PIPE_MTE1);
                if (((tensor_mmad_extent_1 * (int64_t)256) == (n1_ext * (int64_t)256)) && (((host_var * n1_ext) * (int64_t)256) == (int64_t)256)) {
                  load_cbuf_to_cb((k1_outer_fold_db_var ? ((__cb__ half *)tensor_b_zn_db_ping) : ((__cb__ half *)tensor_b_zn_db_pong)) + (max((tensor_mmad_orig_bound_min_1 - host_var18), (int64_t)0) * (int64_t)256), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + (((host_var * (tensor_mmad_orig_bound_min_1 - min(tensor_b_fractal_orig_bound_min_1, host_var18))) * (int64_t)256) + (tensor_a_zz_orig_bound_min_2 * (int64_t)256)), 0, (k1_ext * n1_ext), host_var, 0, 0, 1, inc);
                } else {
                  for (int64_t copy_part_01 = 0; copy_part_01 < k1_ext; ++copy_part_01) {
                    load_cbuf_to_cb((k1_outer_fold_db_var ? ((__cb__ half *)tensor_b_zn_db_ping) : ((__cb__ half *)tensor_b_zn_db_pong)) + ((max((tensor_mmad_orig_bound_min_1 - host_var18), (int64_t)0) * (int64_t)256) + ((tensor_mmad_extent_1 * copy_part_01) * (int64_t)256)), (i0_outer_inner_outer_fold_db_var ? ((__cbuf__ half *)tensor_b_fractal_db_ping) : ((__cbuf__ half *)tensor_b_fractal_db_pong)) + ((((host_var * (tensor_mmad_orig_bound_min_1 - min(tensor_b_fractal_orig_bound_min_1, host_var18))) * (int64_t)256) + (tensor_a_zz_orig_bound_min_2 * (int64_t)256)) + (copy_part_01 * (int64_t)256)), 0, n1_ext, host_var, 0, 0, 1, inc);
                  }
                }
              }
            }
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (((div_var2 + (i0_outer_inner_outer_db * host_var37)) + (i0_outer_outer * host_var40)) < host_var41) {
              if ((((int64_t)0 < n1_inner_ext) && ((int64_t)0 < m1_inner_ext)) && ((int64_t)0 < k1_ext)) {
                if (((int64_t)0 < ((((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)) < (int64_t)0) ? (((int64_t)k_ori) - (tensor_a_zz_orig_bound_min_2 * (int64_t)16)) : ((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)))) && ((int64_t)0 < ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori))))) {
                  if (datatype_bf16 == 1) {
                    mad((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + ((((tensor_mmad_extent_2 * (((max((tensor_mmad_orig_bound_min_0 - host_var41), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var18))) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var14) * (int64_t)256)), (k1_outer_fold_db_var ? ((__ca__ bfloat16_t *)tensor_a_zz_db_ping) : ((__ca__ bfloat16_t *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * (((((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var21) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_a2)) * host_var23)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_a3)) * host_var24)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_a4)) * host_var25)) - min(tensor_a_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_mmad_extent_2) + tensor_mmad_orig_bound_min_2) - min(tensor_mmad_orig_bound_min_2, host_var14))) * (int64_t)256), (k1_outer_fold_db_var ? ((__cb__ bfloat16_t *)tensor_b_zn_db_ping) : ((__cb__ bfloat16_t *)tensor_b_zn_db_pong)) + ((((tensor_mmad_extent_1 * (((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var26) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_b2)) * host_var27)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_b3)) * host_var28)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_b4)) * host_var29)) - min(tensor_b_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_a_zz_extent_2)) * (int64_t)256) + (tensor_mmad_orig_bound_min_1 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_1, host_var18) * (int64_t)256)), max(min((m1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)) < (int64_t)0) ? (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)))), (int64_t)2), min((k1_ext * (int64_t)16), ((((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)) < (int64_t)0) ? (((int64_t)k_ori) - (tensor_a_zz_orig_bound_min_2 * (int64_t)16)) : ((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)))), min((n1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)))), (uint8_t)0ULL, 0, 0, (k1_outer_db == (int64_t)0));
                  } else {
                    mad((i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + ((((tensor_mmad_extent_2 * (((max((tensor_mmad_orig_bound_min_0 - host_var41), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var18))) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var14) * (int64_t)256)), (k1_outer_fold_db_var ? ((__ca__ half *)tensor_a_zz_db_ping) : ((__ca__ half *)tensor_a_zz_db_pong)) + ((tensor_a_zz_extent_2 * (((((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var21) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_a2)) * host_var23)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_a3)) * host_var24)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_a4)) * host_var25)) - min(tensor_a_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_mmad_extent_2) + tensor_mmad_orig_bound_min_2) - min(tensor_mmad_orig_bound_min_2, host_var14))) * (int64_t)256), (k1_outer_fold_db_var ? ((__cb__ half *)tensor_b_zn_db_ping) : ((__cb__ half *)tensor_b_zn_db_pong)) + ((((tensor_mmad_extent_1 * (((((((tensor_mmad_orig_bound_min_0 / host_var20) * host_var26) + (((tensor_mmad_orig_bound_min_0 / host_var22) % ((int64_t)batch_b2)) * host_var27)) + (((tensor_mmad_orig_bound_min_0 / ((int64_t)batch_c4)) % ((int64_t)batch_b3)) * host_var28)) + ((tensor_mmad_orig_bound_min_0 % ((int64_t)batch_b4)) * host_var29)) - min(tensor_b_fractal_orig_bound_min_0, ((int64_t)batch))) * tensor_a_zz_extent_2)) * (int64_t)256) + (tensor_mmad_orig_bound_min_1 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_1, host_var18) * (int64_t)256)), max(min((m1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)) < (int64_t)0) ? (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)))), (int64_t)2), min((k1_ext * (int64_t)16), ((((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)) < (int64_t)0) ? (((int64_t)k_ori) - (tensor_a_zz_orig_bound_min_2 * (int64_t)16)) : ((tensor_a_zz_orig_bound_min_2 * (int64_t)16) - ((int64_t)k_ori)))), min((n1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)))), (uint8_t)0ULL, 0, 0, (k1_outer_db == (int64_t)0));
                  }
                  if ((max(min((m1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)) < (int64_t)0) ? (((int64_t)m_ori) - (tensor_mmad_orig_bound_min_2 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_2 * (int64_t)16) - ((int64_t)m_ori)))), (int64_t)2) * min((n1_inner_ext * (int64_t)16), ((((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori)) < (int64_t)0) ? (((int64_t)n_ori) - (tensor_mmad_orig_bound_min_1 * (int64_t)16)) : ((tensor_mmad_orig_bound_min_1 * (int64_t)16) - ((int64_t)n_ori))))) < (int64_t)2560) {
                    pipe_barrier(PIPE_M);
                  }
                }
              }
            }
            if (((host_var46 * ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) < host_var52) {
              set_flag(PIPE_M, PIPE_MTE1, (((((host_var46 * ((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db)) + k1_outer_db) & 1) == 0) ? 0 : 1));
            }
          }
          set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          if ((((((div_var2 + (i0_outer_inner_outer_db * host_var37)) + (i0_outer_outer * host_var40)) < ((int64_t)batch)) && (tensor_mmad_orig_bound_min_0 < ((int64_t)batch))) && ((i1_outer_inner + (i1_outer_outer * host_var13)) < host_var53)) && ((i2_outer_inner + (i2_outer_outer * host_var17)) < host_var54)) {
            if (((int64_t)0 < i1_inner_ext) && ((int64_t)0 < i2_inner_ext)) {
              if (datatype_bf16 == 1) {
                if (i2_inner_ext <= (int64_t)8192) {
                  set_nd_para((uint64_t)4295032833ULL);
                  pipe_barrier(PIPE_FIX);
                  copy_matrix_cc_to_gm(((__gm__ bfloat16_t *)tensor_c_gm + ((tensor_mmad_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * ((tensor_mmad_orig_bound_min_2 * (int64_t)16) + (tensor_mmad_orig_bound_min_0 * ((int64_t)m_ori)))))), (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + (((((((max((tensor_mmad_orig_bound_min_0 - host_var41), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var18)) * tensor_mmad_extent_2) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var14) * (int64_t)256)), (uint8_t)0ULL, i2_inner_ext, i1_inner_ext, n_ori, (tensor_mmad_extent_2 * (int64_t)16), 0, F322BF16, 0, 0, 1);
                }
              } else if (i2_inner_ext <= (int64_t)8192) {
                set_nd_para((uint64_t)4295032833ULL);
                pipe_barrier(PIPE_FIX);
                copy_matrix_cc_to_gm(((__gm__ half *)tensor_c_gm + ((tensor_mmad_orig_bound_min_1 * (int64_t)16) + (((int64_t)n_ori) * ((tensor_mmad_orig_bound_min_2 * (int64_t)16) + (tensor_mmad_orig_bound_min_0 * ((int64_t)m_ori)))))), (i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__fold_db_var ? ((__cc__ float *)tensor_mmad_db_ping) : ((__cc__ float *)tensor_mmad_db_pong)) + (((((((max((tensor_mmad_orig_bound_min_0 - host_var41), (int64_t)0) * tensor_mmad_extent_1) + tensor_mmad_orig_bound_min_1) - min(tensor_mmad_orig_bound_min_1, host_var18)) * tensor_mmad_extent_2) * (int64_t)256) + (tensor_mmad_orig_bound_min_2 * (int64_t)256)) - (min(tensor_mmad_orig_bound_min_2, host_var14) * (int64_t)256)), (uint8_t)0ULL, i2_inner_ext, i1_inner_ext, n_ori, (tensor_mmad_extent_2 * (int64_t)16), 0, F322F16, 0, 0, 1);
              }
            }
          }
          if (((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) < host_var55) {
            set_flag(PIPE_FIX, PIPE_M, (((((i0_outer_inner_outer_db * host_var38) + i0_outer_inner_inner_i2_outer_inner_i1_outer_inner__db) & 1) == 0) ? 0 : 1));
          }
        }
        if (i0_outer_inner_outer_db < host_var56) {
          set_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)al1_db)) & 1) == 0) ? 0 : 1));
        }
        if (i0_outer_inner_outer_db < host_var57) {
          set_flag(PIPE_MTE1, PIPE_MTE2, ((((i0_outer_inner_outer_db * ((int64_t)bl1_db)) & 1) == 0) ? 2 : 3));
        }
      }
    }
  }
  pipe_barrier(PIPE_ALL);
  pipe_barrier(PIPE_ALL);

  uint64_t origin_fpc = get_fpc();
  uint64_t final_fpc = origin_fpc | 0x8000000000000000;
  set_fpc(final_fpc);
}
inline __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR tile){Compute((__gm__ half*)a,(__gm__ half*)b,(__gm__ half*)c,tile);}
}
#endif
