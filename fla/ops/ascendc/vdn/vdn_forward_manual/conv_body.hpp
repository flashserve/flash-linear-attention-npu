#pragma once
#if defined(__DAV_CUBE__)
namespace OriginalConv {
inline __aicore__ void inject_set_fmatrix(uint64_t value, uint64_t& last_value) {
  if (value != last_value) {
    last_value = value;
    set_fmatrix(value);
  }
}

inline __aicore__ void Spatial2097177(__gm__ bfloat16_t* __restrict__ Fmap, __gm__ bfloat16_t* __restrict__ Filter, __gm__ bfloat16_t* __restrict__ remove_pad_cc, __gm__ uint8_t* __restrict__ tiling_data, int32_t batch_n) {
set_ctrl(get_ctrl() & 0xffff3fffffffffff);
set_l1_3d_size(0);
set_padding(0);
set_mask_norm();
set_atomic_none();
uint64_t last_set_fmatrix_value = 0;
  for (int32_t i = 0; i < (int64_t)3; ++i) {
    dc_preload(((__gm__ uint64_t *)((__gm__ uint8_t *)tiling_data + (((int64_t)i) * (int64_t)64))), (int16_t)0);
  }
  int32_t c_in = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 4)));
  int32_t fmap_h = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 8)));
  int32_t fmap_w = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 12)));
  int32_t c_out = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 16)));
  int32_t k_h = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 20)));
  int32_t k_w = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 24)));
  int32_t group_opt = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 28)));
  int32_t cin1_opt = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 32)));
  int32_t cout1_opt = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 36)));
  int32_t ho = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 40)));
  int32_t wo = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 44)));
  int32_t dilation_h = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 48)));
  int32_t dilation_w = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 52)));
  int32_t stride_h = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 56)));
  int32_t stride_w = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 60)));
  int32_t pad_top = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 64)));
  int32_t pad_bottom = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 68)));
  int32_t pad_left = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 72)));
  int32_t pad_right = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 76)));
  int32_t hf32_mode = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 80)));
  int32_t batch_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 84)));
  int32_t n_single_core = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 88)));
  int32_t batch_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 92)));
  int32_t n_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 96)));
  int32_t m_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 100)));
  int32_t group_dim = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 104)));
  int32_t k_aub = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 108)));
  int32_t m_aub = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 112)));
  int32_t cub_n1 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 116)));
  int32_t n_ub_l0c_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 120)));
  int32_t m_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 124)));
  int32_t k_l0 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 128)));
  int32_t m_al1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 132)));
  int32_t n_bl1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 136)));
  int32_t kal1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 140)));
  int32_t kbl1_16 = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 144)));
  int32_t kal1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 148)));
  int32_t kbl1_factor = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 152)));
  int32_t group = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 156)));
  int32_t enlarge = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 160)));
  int32_t al1_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 164)));
  int32_t bl1_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 168)));
  int32_t al0_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 172)));
  int32_t bl0_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 176)));
  int32_t cl0_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 180)));
  int32_t aub_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 184)));
  int32_t bub_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 188)));
  int32_t cub_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 192)));
  int32_t ubg_pbuffer = (*((__gm__ int32_t *)((__gm__ uint8_t *)tiling_data + 196)));
  int64_t host_var = ((((int64_t)c_out) + (int64_t)15) >> (int64_t)4);
  int64_t host_var1 = ((int64_t)(ho * wo));
  int64_t host_var2 = (((int64_t)((n_ub_l0c_factor * cub_n1) * ((((cout1_opt - 1) / (n_ub_l0c_factor * cub_n1)) / n_dim) + 1))) % (int64_t)255);
  int64_t host_var3 = min(((((((((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + ((((int64_t)kal1_factor) - (int64_t)1) * ((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + (int64_t)1))) + (int64_t)1) * ((int64_t)k_l0)) - (int64_t)1) / ((int64_t)k_w)) / ((int64_t)k_h)) + (int64_t)1), ((int64_t)cin1_opt));
  int64_t host_var4 = max(min((((((((int64_t)wo) - (int64_t)1) * ((int64_t)stride_w)) + ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) + (int64_t)1) - ((int64_t)pad_left)), ((int64_t)fmap_w)), (int64_t)0);
  int64_t host_var5 = ((((((int64_t)((fmap_w + pad_left) + pad_right)) - ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) - (int64_t)1) / ((int64_t)stride_w)) + (int64_t)1);
  int64_t host_var6 = (((int64_t)m_l0) * (int64_t)16);
  int64_t host_var7 = ((((int64_t)kal1_16) / ((((((int64_t)k_h) - (int64_t)1) * ((int64_t)dilation_h)) + (int64_t)1) * (((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w)) + (int64_t)1))) * (int64_t)16);
  int8_t host_var8 = ((int8_t)bl0_pbuffer);
  int64_t host_var9 = ((((((((int64_t)(((cin1_opt * k_h) * k_w) * cout1_opt)) * (int64_t)256) / ((int64_t)n_dim)) * (int64_t)16) + (int64_t)4095) >> (int64_t)12) * (int64_t)4096);
  int64_t host_var10 = ((((int64_t)(((cin1_opt * k_h) * k_w) * cout1_opt)) * (int64_t)256) / ((int64_t)n_dim));
  int8_t host_var11 = ((int8_t)al1_pbuffer);
  int64_t host_var12 = ((((int64_t)k_h) - (int64_t)1) * ((int64_t)dilation_h));
  int64_t host_var13 = ((((int64_t)(m_l0 * m_al1_factor)) * (int64_t)16) / ((int64_t)wo));
  int64_t host_var14 = ((((int64_t)(m_l0 * m_al1_factor)) * (int64_t)16) % ((int64_t)wo));
  int64_t host_var15 = ((((int64_t)(m_l0 * m_al1_factor)) * (int64_t)32) % ((int64_t)wo));
  int64_t host_var16 = ((((int64_t)fmap_w) * (((int64_t)kal1_16) / ((((((int64_t)k_h) - (int64_t)1) * ((int64_t)dilation_h)) + (int64_t)1) * (((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w)) + (int64_t)1)))) * (int64_t)256);
  int64_t host_var17 = ((((int64_t)fmap_w) * (((int64_t)kal1_16) / ((((((int64_t)k_h) - (int64_t)1) * ((int64_t)dilation_h)) + (int64_t)1) * (((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w)) + (int64_t)1)))) * (int64_t)16);
  int8_t host_var18 = ((int8_t)cl0_pbuffer);
  int64_t host_var19 = (((int64_t)((n_ub_l0c_factor * cub_n1) * m_l0)) * (int64_t)8192);
  int64_t host_var20 = (((int64_t)((n_ub_l0c_factor * cub_n1) * m_l0)) * (int64_t)256);
  int8_t host_var21 = ((int8_t)al0_pbuffer);
  int64_t host_var22 = (((int64_t)(m_l0 * k_l0)) * (int64_t)4096);
  int64_t host_var23 = (((int64_t)(m_l0 * k_l0)) * (int64_t)256);
  int64_t host_var24 = (((int64_t)((m_dim * n_dim) * group_dim)) * ((((int64_t)batch_n) + ((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim))) / (((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1)));
  int64_t host_var25 = ((int64_t)((m_dim * n_dim) * group_dim));
  int64_t host_var26 = ((((int64_t)batch_n) + ((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim))) / (((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1));
  int64_t host_var27 = ((int64_t)(m_dim * n_dim));
  int64_t host_var28 = (((int64_t)(n_ub_l0c_factor * cub_n1)) * ((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1));
  int64_t host_var29 = ((((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) / ((int64_t)group_dim)) + (int64_t)1);
  int64_t host_var30 = ((int64_t)((cin1_opt * k_h) * k_w));
  int64_t host_var31 = (((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim));
  int64_t host_var32 = ((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1);
  int64_t host_var33 = ((int64_t)(n_ub_l0c_factor * cub_n1));
  int64_t host_var34 = ((int64_t)(((group_opt * cin1_opt) * k_h) * k_w));
  int64_t host_var35 = (((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1) * ((int64_t)(n_ub_l0c_factor * cub_n1)));
  int64_t host_var36 = ((((int64_t)((n_ub_l0c_factor * cub_n1) * ((((cout1_opt - 1) / (n_ub_l0c_factor * cub_n1)) / n_dim) + 1))) / (int64_t)255) * (int64_t)65280);
  int64_t host_var37 = ((((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1));
  int64_t host_var38 = (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1);
  int64_t host_var39 = (((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1);
  int64_t host_var40 = max(((((int64_t)batch_n) - ((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim))) - (int64_t)1), (int64_t)0);
  int64_t host_var41 = (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1));
  int64_t host_var42 = (((((int64_t)k_h) - (int64_t)1) * ((int64_t)dilation_h)) + (int64_t)1);
  int64_t host_var43 = (((((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1)) * ((int64_t)m_l0)) * (int64_t)16);
  int64_t host_var44 = ((int64_t)(fmap_h + pad_top));
  int64_t host_var45 = ((((((((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + ((((int64_t)kal1_factor) - (int64_t)1) * ((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + (int64_t)1))) + (int64_t)1) * ((int64_t)k_l0)) - (int64_t)1) / ((int64_t)k_w)) / ((int64_t)k_h)) + (int64_t)1);
  int64_t host_var46 = ((((int64_t)c_in) + (int64_t)15) >> (int64_t)4);
  int64_t host_var47 = (max(min((((((((int64_t)wo) - (int64_t)1) * ((int64_t)stride_w)) + ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) + (int64_t)1) - ((int64_t)pad_left)), ((int64_t)fmap_w)), (int64_t)0) * (int64_t)16);
  int64_t host_var48 = (((int64_t)(fmap_h * fmap_w)) * (int64_t)16);
  int64_t host_var49 = (((int64_t)fmap_w) * (int64_t)16);
  uint64_t host_var50 = ((uint64_t)(int32_t)(max((int32_t)(min((((((wo - 1) * stride_w) + ((k_w - 1) * dilation_w)) + 1) - pad_left), fmap_w)), 0)));
  uint64_t host_var51 = ((uint64_t)(int32_t)(min((int32_t)(min((0 - (0 - pad_left)), ((((wo - 1) * stride_w) + ((k_w - 1) * dilation_w)) + 1))), pad_left)));
  int64_t host_var52 = (((((int64_t)0 - ((int64_t)pad_left)) + ((((int64_t)wo) - (int64_t)1) * ((int64_t)stride_w))) + ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) + (int64_t)1);
  int64_t host_var53 = ((((((int64_t)0 - ((int64_t)pad_left)) + ((((int64_t)wo) - (int64_t)1) * ((int64_t)stride_w))) + ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) + (int64_t)1) - ((int64_t)fmap_w));
  int64_t host_var54 = ((((int64_t)(int32_t)(max((int32_t)(min((((((wo - 1) * stride_w) + ((k_w - 1) * dilation_w)) + 1) - pad_left), fmap_w)), 0))) / (int64_t)65535) * (int64_t)1048560);
  int64_t host_var55 = (max(min((((((((int64_t)wo) - (int64_t)1) * ((int64_t)stride_w)) + ((((int64_t)k_w) - (int64_t)1) * ((int64_t)dilation_w))) + (int64_t)1) - ((int64_t)pad_left)), ((int64_t)fmap_w)), (int64_t)0) % (int64_t)65535);
  int64_t host_var56 = (((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1)));
  int64_t host_var57 = ((((int64_t)(ho * wo)) + (int64_t)15) >> (int64_t)4);
  int64_t host_var58 = (((int64_t)kal1_factor) * ((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + (int64_t)1));
  int64_t host_var59 = ((((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) / ((int64_t)kal1_factor)) + (int64_t)1);
  int64_t host_var60 = ((((((int64_t)c_in) + (int64_t)15) >> (int64_t)4) * ((int64_t)k_w)) * ((int64_t)k_h));
  int64_t host_var61 = ((int64_t)((cin1_opt * k_w) * k_h));
  int64_t host_var62 = (((((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1)) * (((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1))));
  int64_t host_var63 = ((((int64_t)0 - ((int64_t)pad_left)) + ((int64_t)pad_left)) / ((int64_t)stride_w));
  int64_t host_var64 = (((((int64_t)((cin1_opt * k_h) * k_w)) - (int64_t)1) / ((int64_t)k_l0)) + (int64_t)1);
  int64_t host_var65 = ((((((int64_t)c_in) + (int64_t)15) >> (int64_t)4) * ((int64_t)k_h)) * ((int64_t)k_w));
  int64_t host_var66 = (((((((int64_t)c_in) + (int64_t)15) >> (int64_t)4) * ((int64_t)k_h)) * ((int64_t)k_w)) * (int64_t)16);
  int64_t host_var67 = (((int64_t)((cin1_opt * k_h) * k_w)) * (int64_t)16);
  int64_t host_var68 = (((int64_t)(n_ub_l0c_factor * cub_n1)) * (int64_t)256);
  int64_t host_var69 = ((((((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) / ((int64_t)group_dim)) + (int64_t)1) * ((((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1))) * (((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1))));
  int64_t host_var70 = (((int64_t)al0_pbuffer) + (int64_t)1);
  int64_t host_var71 = (((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) + (int64_t)1);
  int64_t host_var72 = (((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) + (int64_t)1);
  int64_t host_var73 = (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (int64_t)1);
  int64_t host_var74 = ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1);
  int64_t host_var75 = ((((((((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) / ((int64_t)group_dim)) + (int64_t)1) * ((((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1))) * (((((((int64_t)cout1_opt) - (int64_t)1) / ((int64_t)(n_ub_l0c_factor * cub_n1))) / ((int64_t)n_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) + (((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor))) + (int64_t)1) / ((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) + (int64_t)1)))) - ((int64_t)cl0_pbuffer)) - (int64_t)1);
  int64_t host_var76 = (((((((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) / ((int64_t)group_dim)) + (int64_t)1) * ((((((int64_t)batch_n) - (int64_t)1) / ((int64_t)batch_dim)) + (int64_t)1) * (((((((int64_t)(ho * wo)) - (int64_t)1) / (((int64_t)m_l0) * (int64_t)16)) / ((int64_t)m_al1_factor)) / ((int64_t)m_dim)) + (int64_t)1))) - ((int64_t)al1_pbuffer)) - (int64_t)1);
  int64_t host_var77 = ((((((((int64_t)c_out) + (int64_t)15) >> (int64_t)4) - (int64_t)1) / ((int64_t)cout1_opt)) / ((int64_t)group_dim)) - ((int64_t)bl0_pbuffer));
  int64_t hoist_addr_bl0_db_pong = (host_var8 ? host_var9 : 0);
__cb__   bfloat16_t* bl0_db_pong = (__cb__  bfloat16_t *)(((int32_t)((hoist_addr_bl0_db_pong + (int64_t)7) >> (int64_t)3)));
  int64_t hoist_addr_fmap_l1_db_pong = (host_var11 ? (min(((host_var12 + (((host_var13 + ((int64_t)((host_var14 == (int64_t)0) ? 0 : ((host_var15 == (int64_t)0) ? 1 : 2)))) - (int64_t)1) * ((int64_t)stride_h))) + (int64_t)1), ((int64_t)fmap_h)) * host_var16) : 0);
__cbuf__   bfloat16_t* fmap_l1_db_pong = (__cbuf__  bfloat16_t *)(((int32_t)((hoist_addr_fmap_l1_db_pong + (int64_t)7) >> (int64_t)3)));
  int64_t hoist_addr_mad1_db_pong = (host_var18 ? host_var19 : 0);
__cc__   float* mad1_db_pong = (__cc__  float *)(((int32_t)((hoist_addr_mad1_db_pong + (int64_t)7) >> (int64_t)3)));
  int64_t hoist_addr_im2col_fractal_v2_db_pong = (host_var21 ? host_var22 : 0);
__ca__   bfloat16_t* im2col_fractal_v2_db_pong = (__ca__  bfloat16_t *)(((int32_t)((hoist_addr_im2col_fractal_v2_db_pong + (int64_t)7) >> (int64_t)3)));
// "aicore arch: Ascend910B3"
  if (((int64_t)((int32_t)block_idx)) < host_var24) {
    if (((int32_t)block_idx) < (((m_dim * n_dim) * group_dim) * ((batch_n + ((batch_n - 1) / batch_dim)) / (((batch_n - 1) / batch_dim) + 1)))) {
__cb__       bfloat16_t* bl0_db_ping = (__cb__  bfloat16_t *)get_imm(0);
__cbuf__       bfloat16_t* fmap_l1_db_ping = (__cbuf__  bfloat16_t *)get_imm(0);
__cc__       float* mad1_db_ping = (__cc__  float *)get_imm(0);
__ca__       bfloat16_t* im2col_fractal_v2_db_ping = (__ca__  bfloat16_t *)get_imm(0);
      int64_t n_idx_outer = ((((int64_t)((int32_t)block_idx)) / host_var25) % host_var26);
      int64_t co1_idx_outer_outer = ((((int64_t)((int32_t)block_idx)) / host_var27) % ((int64_t)group_dim));
      int64_t co1_idx_inner_outer_outer_outer = ((int64_t)((((int32_t)block_idx) / m_dim) % n_dim));
      int64_t howo_idx_outer_outer_outer_outer = ((int64_t)(((int32_t)block_idx) % m_dim));
      int64_t bl0_extent_1 = (min((host_var28 * (co1_idx_inner_outer_outer_outer + (int64_t)1)), ((int64_t)cout1_opt)) - min((co1_idx_inner_outer_outer_outer * host_var28), ((int64_t)cout1_opt)));
      int64_t bl0_orig_bound_min_1 = (co1_idx_inner_outer_outer_outer * host_var28);
      for (int64_t co1_idx_outer_inner_outer_db = 0; co1_idx_outer_inner_outer_db < host_var29; ++co1_idx_outer_inner_outer_db) {
        int8_t co1_idx_outer_inner_outer_fold_db_var = ((co1_idx_outer_inner_outer_db & 1) == 0);
        int64_t bl0_orig_bound_min_0 = (((((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) * host_var30);
        int64_t fmap_l1_orig_bound_min_0 = ((((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29));
        if (((int64_t)bl0_pbuffer) < co1_idx_outer_inner_outer_db) {
          wait_flag(PIPE_M, PIPE_MTE2, ((((co1_idx_outer_inner_outer_db * ((int64_t)bl0_pbuffer)) & 1) == 0) ? 0 : 1));
        }
        if ((int64_t)0 < (host_var30 * ((((((((((host_var31 + (co1_idx_inner_outer_outer_outer * host_var32)) + (int64_t)1) * host_var33) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))) - (int64_t)1) / ((int64_t)cout1_opt)) + (int64_t)1) - ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)))) {
          if ((int64_t)0 < host_var2) {
            int64_t div_var0 = ((((((host_var31 + (co1_idx_inner_outer_outer_outer * host_var32)) + (int64_t)1) * host_var33) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))) - (int64_t)1) / ((int64_t)cout1_opt));
            for (int64_t copy_part_0 = 0; copy_part_0 < (host_var30 * ((((div_var0 + (int64_t)1) - ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29))); ++copy_part_0) {
              load_gm_to_cb((co1_idx_outer_inner_outer_fold_db_var ? ((__cb__ bfloat16_t *)bl0_db_ping) : ((__cb__ bfloat16_t *)bl0_db_pong)) + ((((host_var36 + ((bl0_extent_1 * max((bl0_orig_bound_min_0 - host_var34), (int64_t)0)) * (int64_t)256)) + (bl0_orig_bound_min_1 * (int64_t)256)) + ((bl0_extent_1 * copy_part_0) * (int64_t)256)) - (min(bl0_orig_bound_min_1, ((int64_t)cout1_opt)) * (int64_t)256)), ((__gm__ bfloat16_t *)Filter + (((host_var36 + ((((int64_t)cout1_opt) * bl0_orig_bound_min_0) * (int64_t)256)) + (bl0_orig_bound_min_1 * (int64_t)256)) + ((((int64_t)cout1_opt) * copy_part_0) * (int64_t)256))), 0, host_var2, 1, 0, 0, inc);
            }
          }
        }
        set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
        for (int64_t n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db = 0; n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db < host_var37; ++n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db) {
          int64_t div_var51 = (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db / host_var38);
          int8_t n_idx_inner_outer_outer_howo_idx_outer_outer_inner__fold_db_var = ((((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db) & 1) == 0);
          int64_t howo_idx_outer_outer_inner = (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db - (div_var51 * host_var38));
          int64_t fmap_l1_extent_1 = (min(((min((n_idx_outer * host_var39), host_var40) + div_var51) + (int64_t)1), ((int64_t)batch_n)) - min((min((n_idx_outer * host_var39), host_var40) + div_var51), ((int64_t)batch_n)));
          int64_t fmap_l1_orig_bound_min_1 = (min((n_idx_outer * host_var39), host_var40) + div_var51);
          int64_t fmap_l1_extent_3 = (max(min(((((((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) / ((int64_t)wo)) + (((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)) + host_var42) - ((int64_t)pad_top)), ((int64_t)fmap_h)), (int64_t)0) - min(max(((((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)) - ((int64_t)pad_top)), (int64_t)0), ((int64_t)fmap_h)));
          int64_t fmap_l1_orig_bound_min_3 = ((((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)) - ((int64_t)pad_top));
          int64_t hi_idx_ext = max(((min((((((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) / ((int64_t)wo)) + (((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)) + host_var42), host_var44) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))) - max((((int64_t)pad_top) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))), (int64_t)0)), (int64_t)0);
          if (((int64_t)al1_pbuffer) < ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, ((((((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db) * ((int64_t)al1_pbuffer)) & 1) == 0) ? 0 : 1));
          }
          int64_t div_var12 = ((((((host_var31 + (co1_idx_inner_outer_outer_outer * host_var32)) + (int64_t)1) * host_var33) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))) - (int64_t)1) / ((int64_t)cout1_opt));
          for (int64_t group_idx = 0; group_idx < (((((div_var12 + (((host_var31 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt))) + (int64_t)1) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) - (((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt)) * (int64_t)2)); ++group_idx) {
            int64_t ci1_idx_ext = max(min(host_var45, min(((int64_t)cin1_opt), (host_var46 - ((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) * ((int64_t)cin1_opt))))), (int64_t)0);
            int64_t loop_part_0 = ((ci1_idx_ext == max(ci1_idx_ext, hi_idx_ext)) ? ((hi_idx_ext == max(ci1_idx_ext, hi_idx_ext)) ? ci1_idx_ext : ((hi_idx_ext == max(ci1_idx_ext, hi_idx_ext)) ? ci1_idx_ext : hi_idx_ext)) : ci1_idx_ext);
            int64_t loop_part_1 = max(ci1_idx_ext, hi_idx_ext);
            int64_t oper_0_stride_0 = ((ci1_idx_ext == loop_part_1) ? ((hi_idx_ext == loop_part_1) ? (fmap_l1_extent_3 * host_var47) : ((hi_idx_ext == loop_part_1) ? (fmap_l1_extent_3 * host_var47) : host_var47)) : (fmap_l1_extent_3 * host_var47));
            int64_t oper_0_stride_1 = ((hi_idx_ext == loop_part_1) ? host_var47 : ((ci1_idx_ext == loop_part_1) ? (fmap_l1_extent_3 * host_var47) : host_var47));
            int64_t oper_1_stride_0 = ((ci1_idx_ext == loop_part_1) ? ((hi_idx_ext == loop_part_1) ? host_var48 : ((hi_idx_ext == loop_part_1) ? host_var48 : host_var49)) : host_var48);
            int64_t oper_1_stride_1 = ((hi_idx_ext == loop_part_1) ? host_var49 : ((ci1_idx_ext == loop_part_1) ? host_var48 : host_var49));
            if (((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) < ((int64_t)group_opt)) && (fmap_l1_orig_bound_min_1 < ((int64_t)batch_n))) {
              if ((((int64_t)0 < ci1_idx_ext) && ((int64_t)0 < hi_idx_ext)) && ((int64_t)0 < host_var4)) {
                if ((int64_t)0 < fmap_l1_extent_3) {
                  inject_set_fmatrix((((((host_var50 | (((uint64_t)fmap_l1_extent_3) << (uint64_t)16ULL)) | (host_var51 << (uint64_t)32ULL)) | (((uint64_t)min(((((int64_t)fmap_w) < host_var52) ? host_var53 : (int64_t)0), ((int64_t)pad_right))) << (uint64_t)40ULL)) | (((uint64_t)min(((fmap_l1_orig_bound_min_3 < (int64_t)0) ? min(((int64_t)0 - fmap_l1_orig_bound_min_3), ((((((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) / ((int64_t)wo)) + (((((host_var41 * ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) + (int64_t)1)) * host_var6) - (int64_t)1) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)) + host_var42) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h)))) : (int64_t)0), ((int64_t)pad_top))) << (uint64_t)48ULL)) | (uint64_t)18374686479671623680), last_set_fmatrix_value);
                }
                pipe_barrier(PIPE_MTE2);
                if ((((host_var4 <= (int64_t)65535) && (loop_part_1 <= (int64_t)4095)) && (oper_1_stride_1 < (int64_t)1048560)) && (((int64_t)-15 <= (oper_0_stride_1 - host_var47)) && ((int64_t)-15 <= (oper_1_stride_1 - host_var47)))) {
                  for (int64_t copy_part_01 = 0; copy_part_01 < loop_part_0; ++copy_part_01) {
                    copy_gm_to_cbuf((n_idx_inner_outer_outer_howo_idx_outer_outer_inner__fold_db_var ? ((__cbuf__ bfloat16_t *)fmap_l1_db_ping) : ((__cbuf__ bfloat16_t *)fmap_l1_db_pong)) + (((host_var4 * ((((max((((int64_t)pad_top) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))), (int64_t)0) + (fmap_l1_extent_3 * ((((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) - min(fmap_l1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1) * host_var3))) + (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))) - ((int64_t)pad_top)) - min(max(fmap_l1_orig_bound_min_3, (int64_t)0), ((int64_t)fmap_h)))) * (int64_t)16) + (oper_0_stride_0 * copy_part_01)), ((__gm__ bfloat16_t *)Fmap + (((((int64_t)fmap_w) * (((max((((int64_t)pad_top) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))), (int64_t)0) + (((int64_t)fmap_h) * ((fmap_l1_orig_bound_min_1 * host_var46) + ((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) * ((int64_t)cin1_opt))))) + (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))) - ((int64_t)pad_top))) * (int64_t)16) + (oper_1_stride_0 * copy_part_01))), (uint8_t)0ULL, loop_part_1, host_var4, ((oper_1_stride_1 - host_var47) / (int64_t)16), ((oper_0_stride_1 - host_var47) / (int64_t)16), PAD_NONE);
                  }
                } else {
                  for (int64_t copy_part_02 = 0; copy_part_02 < loop_part_0; ++copy_part_02) {
                    for (int64_t copy_part_1 = 0; copy_part_1 < loop_part_1; ++copy_part_1) {
                      copy_gm_to_cbuf((n_idx_inner_outer_outer_howo_idx_outer_outer_inner__fold_db_var ? ((__cbuf__ bfloat16_t *)fmap_l1_db_ping) : ((__cbuf__ bfloat16_t *)fmap_l1_db_pong)) + (((host_var54 + ((host_var4 * ((((max((((int64_t)pad_top) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))), (int64_t)0) + (fmap_l1_extent_3 * ((((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) - min(fmap_l1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1) * host_var3))) + (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))) - ((int64_t)pad_top)) - min(max(fmap_l1_orig_bound_min_3, (int64_t)0), ((int64_t)fmap_h)))) * (int64_t)16)) + (oper_0_stride_0 * copy_part_02)) + (oper_0_stride_1 * copy_part_1)), ((__gm__ bfloat16_t *)Fmap + (((host_var54 + ((((int64_t)fmap_w) * (((max((((int64_t)pad_top) - (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))), (int64_t)0) + (((int64_t)fmap_h) * ((fmap_l1_orig_bound_min_1 * host_var46) + ((((group_idx + ((co1_idx_inner_outer_outer_outer * host_var35) / ((int64_t)cout1_opt))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) * ((int64_t)cin1_opt))))) + (((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) / ((int64_t)wo)) + ((((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var43) % ((int64_t)wo)) >> (int64_t)63)) * ((int64_t)stride_h))) - ((int64_t)pad_top))) * (int64_t)16)) + (oper_1_stride_0 * copy_part_02)) + (oper_1_stride_1 * copy_part_1))), 0, 1, host_var55, 0, 0, PAD_NONE);
                    }
                  }
                }
              }
            }
          }
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          for (int64_t co1_idx_inner_outer_inner_howo_idx_outer_inner__db = 0; co1_idx_inner_outer_inner_howo_idx_outer_inner__db < host_var56; ++co1_idx_inner_outer_inner_howo_idx_outer_inner__db) {
            int64_t div_var42 = (co1_idx_inner_outer_inner_howo_idx_outer_inner__db / host_var41);
            int8_t co1_idx_inner_outer_inner_howo_idx_outer_inner__fold_db_var = ((((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db) & 1) == 0);
            int64_t howo_idx_outer_inner = (co1_idx_inner_outer_inner_howo_idx_outer_inner__db - (div_var42 * host_var41));
            int64_t mad1_extent_2 = (min((((host_var28 * co1_idx_inner_outer_outer_outer) + (div_var42 * host_var33)) + host_var33), ((int64_t)cout1_opt)) - min(((host_var28 * co1_idx_inner_outer_outer_outer) + (div_var42 * host_var33)), ((int64_t)cout1_opt)));
            int64_t mad1_orig_bound_min_0 = ((((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29));
            int64_t mad1_orig_bound_min_2 = ((host_var28 * co1_idx_inner_outer_outer_outer) + (div_var42 * host_var33));
            int64_t co1_idx_inner_ext = max(min(host_var33, ((((int64_t)cout1_opt) - (host_var28 * co1_idx_inner_outer_outer_outer)) - (div_var42 * host_var33))), (int64_t)0);
            int64_t mad1_extent_3 = ((min((((int64_t)m_l0) * ((howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41)) + (int64_t)1)), host_var57) * (int64_t)16) - (min((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))), host_var57) * (int64_t)16));
            int64_t im2col_fractal_v2_orig_bound_min_2 = (((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41)));
            int64_t howo_idx_inner_ext = (max(min(((int64_t)m_l0), (host_var57 - (((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))))), (int64_t)0) * (int64_t)16);
            int64_t howo_idx_inner_inner_ext = max(min(host_var6, (host_var1 - ((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))) * (int64_t)16))), (int64_t)0);
            if (((int64_t)cl0_pbuffer) < ((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) {
              wait_flag(PIPE_FIX, PIPE_M, ((((((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db) * ((int64_t)cl0_pbuffer)) & 1) == 0) ? 0 : 1));
            }
            int64_t div_var32 = (((((host_var33 * ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) + (int64_t)1)) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))) - (int64_t)1) / ((int64_t)cout1_opt)) + (((((host_var33 * ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) + (int64_t)1)) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))) - (int64_t)1) % ((int64_t)cout1_opt)) >> (int64_t)63));
            for (int64_t group_idx1 = 0; group_idx1 < ((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)); ++group_idx1) {
              int64_t im2col_fractal_v2_orig_bound_min_0 = (((group_idx1 + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29));
              for (int64_t cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db = 0; cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db < host_var58; ++cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db) {
                int64_t div_var29 = (cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db / host_var59);
                int8_t cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__fold_db_var = ((((host_var58 * ((((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) * ((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) + group_idx1)) + cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db) & 1) == 0);
                int64_t cin_1_kh_kw_outer_inner = (cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db - (div_var29 * host_var59));
                int64_t im2col_fractal_v2_extent_3 = (min((((int64_t)k_l0) * ((cin_1_kh_kw_outer_inner + (div_var29 * host_var59)) + (int64_t)1)), host_var30) - min(((cin_1_kh_kw_outer_inner + (div_var29 * host_var59)) * ((int64_t)k_l0)), host_var30));
                int64_t im2col_fractal_v2_orig_bound_min_3 = ((cin_1_kh_kw_outer_inner + (div_var29 * host_var59)) * ((int64_t)k_l0));
                int64_t k1_idx_inner_ext = max(min(((int64_t)k_l0), ((host_var60 - ((((group_idx1 + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) * host_var61)) - im2col_fractal_v2_orig_bound_min_3)), (int64_t)0);
                int64_t lower_steps_k = (k1_idx_inner_ext * (int64_t)16);
                uint64_t lower_repeat_expr = ((((uint64_t)(((int64_t)m_l0) * k1_idx_inner_ext)) << (uint64_t)16ULL) + (uint64_t)1ULL);
                int64_t cin_1_kh_kw_inner_ext = max(min(((int64_t)k_l0), (min(host_var30, (host_var60 - (im2col_fractal_v2_orig_bound_min_0 * host_var61))) - ((cin_1_kh_kw_outer_inner + (div_var29 * host_var59)) * ((int64_t)k_l0)))), (int64_t)0);
                if (((int64_t)al0_pbuffer) < ((host_var58 * ((((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) * (((co1_idx_outer_inner_outer_db * host_var62) + (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db * host_var56)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) + group_idx1)) + cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db)) {
                  wait_flag(PIPE_M, PIPE_MTE1, ((((((host_var58 * ((((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) * (((co1_idx_outer_inner_outer_db * host_var62) + (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db * host_var56)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) + group_idx1)) + cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db) * ((int64_t)al0_pbuffer)) & 1) == 0) ? 0 : 1));
                }
                if ((int64_t)0 < k1_idx_inner_ext) {
                  pipe_barrier(PIPE_MTE1);
                  if (((int64_t)0 < fmap_l1_extent_3) && ((int64_t)0 < host_var4)) {
                    pipe_barrier(PIPE_MTE1);
                    img2colv2_cbuf_to_ca((cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__fold_db_var ? ((__ca__ half *)im2col_fractal_v2_db_ping) : ((__ca__ half *)im2col_fractal_v2_db_pong)) + ((((im2col_fractal_v2_extent_3 * (((int64_t)m_l0) * (((min((n_idx_outer * host_var39), host_var40) + (max((im2col_fractal_v2_orig_bound_min_0 - ((int64_t)group_opt)), (int64_t)0) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n))))) * (int64_t)256) + (im2col_fractal_v2_orig_bound_min_3 * (int64_t)256)) - (min(im2col_fractal_v2_orig_bound_min_3, host_var30) * (int64_t)256)), (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__fold_db_var ? ((__cbuf__ half *)fmap_l1_db_ping) : ((__cbuf__ half *)fmap_l1_db_pong)), lower_steps_k, host_var6, (min(im2col_fractal_v2_orig_bound_min_3, host_var30) * (int64_t)16), ((((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))) * (int64_t)16) - (((fmap_l1_orig_bound_min_3 + ((int64_t)pad_top)) / ((int64_t)stride_h)) * host_var5)) - host_var63), stride_w, stride_h, (k_w & 255), (k_h & 255), dilation_w, dilation_h, ((k_w >> 8) & 255), ((k_h >> 8) & 255), 0, 0, host_var7);
                  } else {
                    create_ca_matrix((cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__fold_db_var ? ((__ca__ bfloat16_t *)im2col_fractal_v2_db_ping) : ((__ca__ bfloat16_t *)im2col_fractal_v2_db_pong)) + ((((im2col_fractal_v2_extent_3 * (((int64_t)m_l0) * (((min((n_idx_outer * host_var39), host_var40) + (max((im2col_fractal_v2_orig_bound_min_0 - ((int64_t)group_opt)), (int64_t)0) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n))))) * (int64_t)256) + (im2col_fractal_v2_orig_bound_min_3 * (int64_t)256)) - (min(im2col_fractal_v2_orig_bound_min_3, host_var30) * (int64_t)256)), lower_repeat_expr, (uint32_t)0);
                  }
                }
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                if ((((cin_1_kh_kw_outer_inner + (div_var29 * host_var59)) < host_var64) && (im2col_fractal_v2_orig_bound_min_0 < ((int64_t)group_opt))) && ((min((n_idx_outer * host_var39), host_var40) + div_var51) < ((int64_t)batch_n))) {
                  if ((((int64_t)0 < co1_idx_inner_ext) && ((int64_t)0 < howo_idx_inner_ext)) && ((int64_t)0 < cin_1_kh_kw_inner_ext)) {
                    if ((int64_t)0 < (((((im2col_fractal_v2_orig_bound_min_0 * host_var30) + im2col_fractal_v2_orig_bound_min_3) - host_var65) < (int64_t)0) ? ((host_var66 - (im2col_fractal_v2_orig_bound_min_0 * host_var67)) - (im2col_fractal_v2_orig_bound_min_3 * (int64_t)16)) : (((im2col_fractal_v2_orig_bound_min_0 * host_var67) + (im2col_fractal_v2_orig_bound_min_3 * (int64_t)16)) - host_var66))) {
                      mad((co1_idx_inner_outer_inner_howo_idx_outer_inner__fold_db_var ? ((__cc__ float *)mad1_db_ping) : ((__cc__ float *)mad1_db_pong)) + (((im2col_fractal_v2_orig_bound_min_2 * (int64_t)256) + ((mad1_extent_3 * ((((mad1_extent_2 * (((min((n_idx_outer * host_var39), host_var40) + ((im2col_fractal_v2_orig_bound_min_0 - min(mad1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n)))) + (host_var28 * co1_idx_inner_outer_outer_outer)) + (div_var42 * host_var33)) - min(mad1_orig_bound_min_2, ((int64_t)cout1_opt)))) * (int64_t)16)) - (min(im2col_fractal_v2_orig_bound_min_2, host_var57) * (int64_t)256)), (cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__fold_db_var ? ((__ca__ bfloat16_t *)im2col_fractal_v2_db_ping) : ((__ca__ bfloat16_t *)im2col_fractal_v2_db_pong)) + (((((max((fmap_l1_orig_bound_min_1 - ((int64_t)batch_n)), (int64_t)0) * ((int64_t)m_l0)) * im2col_fractal_v2_extent_3) * (int64_t)256) + (im2col_fractal_v2_orig_bound_min_3 * (int64_t)256)) - (min(im2col_fractal_v2_orig_bound_min_3, host_var30) * (int64_t)256)), (co1_idx_outer_inner_outer_fold_db_var ? ((__cb__ bfloat16_t *)bl0_db_ping) : ((__cb__ bfloat16_t *)bl0_db_pong)) + (((((bl0_extent_1 * (((im2col_fractal_v2_orig_bound_min_0 * host_var30) + im2col_fractal_v2_orig_bound_min_3) - min((fmap_l1_orig_bound_min_0 * host_var30), host_var34))) * (int64_t)256) + ((host_var28 * co1_idx_inner_outer_outer_outer) * (int64_t)256)) + (div_var42 * host_var68)) - (min(bl0_orig_bound_min_1, ((int64_t)cout1_opt)) * (int64_t)256)), howo_idx_inner_ext, min((cin_1_kh_kw_inner_ext * (int64_t)16), (((((im2col_fractal_v2_orig_bound_min_0 * host_var30) + im2col_fractal_v2_orig_bound_min_3) - host_var65) < (int64_t)0) ? ((host_var66 - (im2col_fractal_v2_orig_bound_min_0 * host_var67)) - (im2col_fractal_v2_orig_bound_min_3 * (int64_t)16)) : (((im2col_fractal_v2_orig_bound_min_0 * host_var67) + (im2col_fractal_v2_orig_bound_min_3 * (int64_t)16)) - host_var66))), (co1_idx_inner_ext * (int64_t)16), (uint8_t)0ULL, 0, 0, ((div_var29 == (int64_t)0) && (cin_1_kh_kw_outer_inner == (int64_t)0)));
                      if ((co1_idx_inner_ext * howo_idx_inner_ext) < (int64_t)160) {
                        pipe_barrier(PIPE_M);
                      }
                    }
                  }
                }
                if (((host_var58 * ((((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) * (((co1_idx_outer_inner_outer_db * host_var62) + (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db * host_var56)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) + group_idx1)) + cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db) < (((host_var69 * ((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29))) * host_var58) - host_var70)) {
                  set_flag(PIPE_M, PIPE_MTE1, ((((((host_var58 * ((((((div_var32 + (int64_t)1) - ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) / ((int64_t)cout1_opt)) + ((((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33) % ((int64_t)cout1_opt)) >> (int64_t)63))) - co1_idx_outer_inner_outer_db) - (co1_idx_outer_outer * host_var29)) * (((co1_idx_outer_inner_outer_db * host_var62) + (n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db * host_var56)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db)) + group_idx1)) + cin_1_kh_kw_outer_outer_inner_cin_1_kh_kw_outer_inner__db) * ((int64_t)al0_pbuffer)) & 1) == 0) ? 0 : 1));
                }
              }
            }
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            if ((((((min((n_idx_outer * host_var39), host_var40) + div_var51) < ((int64_t)batch_n)) && ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) < host_var71)) && ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) < host_var72)) && ((howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41)) < host_var73)) && ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) < host_var74)) {
              for (int64_t co1_idx_inner_inner_outer = 0; co1_idx_inner_inner_outer < ((int64_t)n_ub_l0c_factor); ++co1_idx_inner_inner_outer) {
                int64_t co1_idx_inner_inner_inner_ext = max(min(((int64_t)cub_n1), min((((host_var - (co1_idx_inner_inner_outer * ((int64_t)cub_n1))) - ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) - ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt))), ((((int64_t)cout1_opt) - (co1_idx_inner_inner_outer * ((int64_t)cub_n1))) - ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)))), (int64_t)0);
                if (((int64_t)0 < co1_idx_inner_inner_inner_ext) && ((int64_t)0 < howo_idx_inner_inner_ext)) {
                  pipe_barrier(PIPE_FIX);
                  copy_matrix_cc_to_gm(((__gm__ bfloat16_t *)remove_pad_cc + (((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))) * (int64_t)256) + ((host_var1 * ((((fmap_l1_orig_bound_min_1 * host_var) + (co1_idx_inner_inner_outer * ((int64_t)cub_n1))) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) + ((co1_idx_outer_inner_outer_db + (co1_idx_outer_outer * host_var29)) * ((int64_t)cout1_opt)))) * (int64_t)16))), (co1_idx_inner_outer_inner_howo_idx_outer_inner__fold_db_var ? ((__cc__ float *)mad1_db_ping) : ((__cc__ float *)mad1_db_pong)) + ((((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))) * (int64_t)256) + ((((((((min((n_idx_outer * host_var39), host_var40) + ((((((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) / ((int64_t)cout1_opt)) + ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) >> (int64_t)63)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) - min(mad1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n))) * mad1_extent_2) + ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) + (((int64_t)cout1_opt) & ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) >> (int64_t)63)))) - min(mad1_orig_bound_min_2, ((int64_t)cout1_opt))) * mad1_extent_3) * (int64_t)16)) - (min((((int64_t)m_l0) * (howo_idx_outer_inner + ((howo_idx_outer_outer_inner + (howo_idx_outer_outer_outer_outer * host_var38)) * host_var41))), host_var57) * (int64_t)256)), (uint8_t)0ULL, (co1_idx_inner_inner_inner_ext * (int64_t)16), howo_idx_inner_inner_ext, host_var1, ((((((((min((n_idx_outer * host_var39), host_var40) + (((((((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) + (int64_t)1) / ((int64_t)cout1_opt)) + (((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) + (int64_t)1) % ((int64_t)cout1_opt)) >> (int64_t)63)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) - min(mad1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n))) * mad1_extent_2) + (((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) + (int64_t)1) % ((int64_t)cout1_opt)) + (((int64_t)cout1_opt) & (((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) + (int64_t)1) % ((int64_t)cout1_opt)) >> (int64_t)63)))) - ((((min((n_idx_outer * host_var39), host_var40) + ((((((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) / ((int64_t)cout1_opt)) + ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) >> (int64_t)63)) + co1_idx_outer_inner_outer_db) + (co1_idx_outer_outer * host_var29)) - min(mad1_orig_bound_min_0, ((int64_t)group_opt))) * fmap_l1_extent_1)) + div_var51) - min(fmap_l1_orig_bound_min_1, ((int64_t)batch_n))) * mad1_extent_2)) - ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) + (((int64_t)cout1_opt) & ((((co1_idx_inner_inner_outer * ((int64_t)cub_n1)) + ((div_var42 + (co1_idx_inner_outer_outer_outer * host_var32)) * host_var33)) % ((int64_t)cout1_opt)) >> (int64_t)63)))) * mad1_extent_3), 0, F322BF16, 0, 0, 0);
                }
              }
            }
            if (((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db) < host_var75) {
              set_flag(PIPE_FIX, PIPE_M, ((((((host_var56 * ((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db)) + co1_idx_inner_outer_inner_howo_idx_outer_inner__db) * ((int64_t)cl0_pbuffer)) & 1) == 0) ? 0 : 1));
            }
          }
          if (((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db) < host_var76) {
            set_flag(PIPE_MTE1, PIPE_MTE2, ((((((co1_idx_outer_inner_outer_db * host_var37) + n_idx_inner_outer_outer_howo_idx_outer_outer_inner__db) * ((int64_t)al1_pbuffer)) & 1) == 0) ? 0 : 1));
          }
        }
        if (co1_idx_outer_inner_outer_db < host_var77) {
          set_flag(PIPE_M, PIPE_MTE2, ((((co1_idx_outer_inner_outer_db * ((int64_t)bl0_pbuffer)) & 1) == 0) ? 0 : 1));
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
}
#endif
