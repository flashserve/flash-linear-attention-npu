# arch35 (Ascend950 / A5)

A5 currently compiles from the **unified** kernel source
(`../chunk_kda_fwd_intra_sub_chunk.cpp` + `*_common/_cube/_vector.h`): the
`CATLASS_ARCH` switch in `chunk_kda_fwd_intra_sub_chunk_common.h` selects
`Catlass::Arch::Ascend950` when `__CCE_AICORE__ == 310`, so no A5-specific
kernel body is required for the compile gate (DESIGN §7.4: A5 = 注册 + 编译通过).

A5 precision tuning (dedicated arch35 implementation) is deferred; see DESIGN §0/§7.4.
