; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: alwaysinline
define private void @broadcast_scalar_float_to_1d(float %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5) #0 {
  %7 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %1, 0
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %7, ptr addrspace(6) %2, 1
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, i64 %3, 2
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %4, 3, 0
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %5, 4, 0
  %12 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, ptr %12, align 8
  call void @_mlir_ciface_broadcast_scalar_float_to_1d(float %0, ptr %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_float_to_1d(float, ptr) #0

; Function Attrs: alwaysinline
define private void @broadcast_scalar_int32_t_to_1d(i32 %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5) #0 {
  %7 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %1, 0
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %7, ptr addrspace(6) %2, 1
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, i64 %3, 2
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %4, 3, 0
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %5, 4, 0
  %12 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, ptr %12, align 8
  call void @_mlir_ciface_broadcast_scalar_int32_t_to_1d(i32 %0, ptr %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_int32_t_to_1d(i32, ptr) #0

; Function Attrs: alwaysinline
define private void @vcmp_eq_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr %21, align 8
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %6, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %7, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %11, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %12, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_vcmp_eq_1d_float(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcmp_eq_1d_float(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vnot_1d_bool(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr %21, align 8
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %6, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %7, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %11, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %12, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_vnot_1d_bool(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vnot_1d_bool(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @broadcast_scalar_bfloat16_t_to_1d(bfloat %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5) #0 {
  %7 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %1, 0
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %7, ptr addrspace(6) %2, 1
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, i64 %3, 2
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %4, 3, 0
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %5, 4, 0
  %12 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, ptr %12, align 8
  call void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_1d(bfloat %0, ptr %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_1d(bfloat, ptr) #0

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_2d_bfloat16_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14, bfloat %15, i64 %16) #0 {
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %0, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %18, ptr addrspace(1) %1, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %19, i64 %2, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %20, i64 %3, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %21, i64 %5, 4, 0
  %23 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %22, i64 %4, 3, 1
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %23, i64 %6, 4, 1
  %25 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, ptr %25, align 8
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, ptr addrspace(6) %8, 1
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %9, 2
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %10, 3, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %12, 4, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %11, 3, 1
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %13, 4, 1
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_2d_bfloat16_t(ptr %25, ptr %33, i32 %14, bfloat %15, i64 %16)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_2d_bfloat16_t(ptr, ptr, i32, bfloat, i64) #0

; Function Attrs: alwaysinline
define private void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(6) %6, 1
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_vcast_bfloat16_t_to_float_1d_with_mode(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcast_bfloat16_t_to_float_1d_with_mode(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @vmuls_vs_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vmuls_vs_1d_float(ptr %22, float %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmuls_vs_1d_float(ptr, float, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadds_vs_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vadds_vs_1d_float(ptr %22, float %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadds_vs_1d_float(ptr, float, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vexp_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr %21, align 8
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %6, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %7, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %11, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %12, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_vexp_1d_float(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vexp_1d_float(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vdiv_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vdiv_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vdiv_1d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadds_vs_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i32 %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vadds_vs_1d_int32_t(ptr %22, i32 %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadds_vs_1d_int32_t(ptr, i32, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vshrs_vs_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i32 %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(6) %7, 1
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %8, 2
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %9, 3, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %10, 4, 0
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_vshrs_vs_1d_int32_t(ptr %17, i32 %5, ptr %23)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vshrs_vs_1d_int32_t(ptr, i32, ptr) #0

; Function Attrs: alwaysinline
define private void @vand_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vand_1d_int32_t(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vand_1d_int32_t(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadd_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vadd_1d_int32_t(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadd_1d_int32_t(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vmul_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vmul_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmul_1d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @enablevc_reduce_sum_ar_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, float %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %5, 4, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %4, 3, 1
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %6, 4, 1
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr addrspace(6) %8, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr %36, align 8
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr addrspace(6) %15, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %16, 2
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %17, 3, 0
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %18, 4, 0
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, ptr %42, align 8
  call void @_mlir_ciface_enablevc_reduce_sum_ar_float(ptr %28, ptr %36, ptr %42, float %19)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_enablevc_reduce_sum_ar_float(ptr, ptr, ptr, float) #0

; Function Attrs: alwaysinline
define private void @vsqrt_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr %21, align 8
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %6, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %7, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %11, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %12, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_vsqrt_1d_float(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vsqrt_1d_float(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vmaxs_vs_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vmaxs_vs_1d_float(ptr %22, float %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmaxs_vs_1d_float(ptr, float, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vsel_sv_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15, ptr addrspace(6) %16, ptr addrspace(6) %17, i64 %18, i64 %19, i64 %20) #0 {
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %1, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %4, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %7, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %10, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, ptr addrspace(6) %12, 1
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 2
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 3, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, i64 %15, 4, 0
  %39 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, ptr %39, align 8
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %16, 0
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, ptr addrspace(6) %17, 1
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 2
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 3, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, i64 %20, 4, 0
  %45 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %44, ptr %45, align 8
  call void @_mlir_ciface_vsel_sv_1d_float(ptr %27, float %5, ptr %33, ptr %39, ptr %45)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vsel_sv_1d_float(ptr, float, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vsel_vv_1d_bool_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19, ptr addrspace(6) %20, ptr addrspace(6) %21, i64 %22, i64 %23, i64 %24) #0 {
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr addrspace(6) %1, 1
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, i64 %2, 2
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %3, 3, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %4, 4, 0
  %31 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, ptr %31, align 8
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr addrspace(6) %6, 1
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, i64 %7, 2
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %8, 3, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %9, 4, 0
  %37 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, ptr %37, align 8
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, ptr addrspace(6) %11, 1
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %12, 2
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %13, 3, 0
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %14, 4, 0
  %43 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, ptr %43, align 8
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %44, ptr addrspace(6) %16, 1
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %45, i64 %17, 2
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, i64 %18, 3, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %47, i64 %19, 4, 0
  %49 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %48, ptr %49, align 8
  %50 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %20, 0
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %50, ptr addrspace(6) %21, 1
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, i64 %22, 2
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 3, 0
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 4, 0
  %55 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, ptr %55, align 8
  call void @_mlir_ciface_vsel_vv_1d_bool_float(ptr %31, ptr %37, ptr %43, ptr %49, ptr %55)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vsel_vv_1d_bool_float(ptr, ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vdiv_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr addrspace(6) %1, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %5, 4, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %6, 4, 1
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(6) %8, 1
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %36, i64 %9, 2
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %37, i64 %10, 3, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %38, i64 %12, 4, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, i64 %11, 3, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 4, 1
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %14, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, ptr addrspace(6) %15, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %16, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, i64 %17, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %46, i64 %19, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, i64 %18, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %48, i64 %20, 4, 1
  %50 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %49, ptr %50, align 8
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %21, 0
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, ptr addrspace(6) %22, 1
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 2
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 3, 0
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, i64 %25, 4, 0
  %56 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, ptr %56, align 8
  call void @_mlir_ciface_vdiv_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vdiv_2d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(6) %6, 1
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_vcast_float_to_bfloat16_t_1d_with_mode(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcast_float_to_bfloat16_t_1d_with_mode(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_bfloat16_t(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_bfloat16_t(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, bfloat %11, i64 %12) #0 {
  %14 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %14, ptr addrspace(1) %1, 1
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, i64 %2, 2
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %3, 3, 0
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %4, 4, 0
  %19 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr %19, align 8
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr addrspace(6) %6, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %7, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %8, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %9, 4, 0
  %25 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, ptr %25, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_bfloat16_t(ptr %19, ptr %25, i32 %10, bfloat %11, i64 %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_bfloat16_t(ptr, ptr, i32, bfloat, i64) #0

; Function Attrs: alwaysinline
define private void @vmul_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr addrspace(6) %1, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %5, 4, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %6, 4, 1
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(6) %8, 1
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %36, i64 %9, 2
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %37, i64 %10, 3, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %38, i64 %12, 4, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, i64 %11, 3, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 4, 1
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %14, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, ptr addrspace(6) %15, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %16, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, i64 %17, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %46, i64 %19, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, i64 %18, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %48, i64 %20, 4, 1
  %50 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %49, ptr %50, align 8
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %21, 0
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, ptr addrspace(6) %22, 1
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 2
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 3, 0
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, i64 %25, 4, 0
  %56 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, ptr %56, align 8
  call void @_mlir_ciface_vmul_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmul_2d_float(ptr, ptr, ptr, ptr) #0

define dso_local void @JyKV.vector(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i64 %virtual_id) #1 {
  %15 = alloca i64, i64 1, align 8
  store i64 0, ptr %15, align 4
  %16 = mul i32 %11, %12
  %17 = mul i32 %16, %13
  %18 = add i64 %virtual_id, 0
  %19 = trunc i64 %18 to i32
  %20 = mul i32 %8, %9
  %21 = add i32 %10, 63
  %22 = sdiv i32 %21, 64
  %23 = mul i32 %20, %22
  %24 = mul i32 %9, %22
  call void @broadcast_scalar_float_to_1d(float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  %25 = sext i32 %9 to i64
  %26 = mul i64 %25, 128
  %27 = sext i32 %10 to i64
  call void @broadcast_scalar_int32_t_to_1d(i32 1, ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), i64 0, i64 8192, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 -65536, ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), i64 0, i64 8192, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 1, ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 -65536, ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcmp_eq_1d_float(ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vnot_1d_bool(ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  br label %28

28:                                               ; preds = %155, %14
  %29 = phi i32 [ %156, %155 ], [ %19, %14 ]
  %30 = icmp slt i32 %29, %17
  br i1 %30, label %31, label %157

31:                                               ; preds = %28
  %32 = call i64 @llvm.hivm.GET.CTRL()
  %33 = call i64 @llvm.hivm.SBITSET0(i64 %32, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %33)
  %34 = srem i32 %29, %11
  br label %35

35:                                               ; preds = %150, %31
  %36 = phi i32 [ %154, %150 ], [ %34, %31 ]
  %37 = icmp slt i32 %36, %23
  br i1 %37, label %38, label %155

38:                                               ; preds = %35
  %39 = load i64, ptr %15, align 4
  %40 = urem i64 %39, 2
  %41 = icmp eq i64 %40, 1
  %42 = select i1 %41, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %43 = select i1 %41, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %44 = urem i64 %39, 2
  %45 = icmp eq i64 %44, 1
  %46 = select i1 %45, ptr addrspace(6) inttoptr (i64 65664 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6))
  %47 = select i1 %45, ptr addrspace(6) inttoptr (i64 65664 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6))
  %48 = urem i64 %39, 2
  %49 = icmp eq i64 %48, 1
  %50 = select i1 %49, ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), ptr addrspace(6) null
  %51 = select i1 %49, ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), ptr addrspace(6) null
  %52 = urem i64 %39, 2
  %53 = icmp eq i64 %52, 1
  %54 = select i1 %53, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %55 = select i1 %53, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %56 = urem i64 %39, 2
  %57 = icmp eq i64 %56, 1
  %58 = select i1 %57, ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), ptr addrspace(6) null
  %59 = select i1 %57, ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), ptr addrspace(6) null
  %60 = urem i64 %39, 2
  %61 = icmp eq i64 %60, 1
  %62 = select i1 %61, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %63 = select i1 %61, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %64 = urem i64 %39, 2
  %65 = icmp eq i64 %64, 1
  %66 = select i1 %65, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %67 = select i1 %65, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %68 = urem i64 %39, 2
  %69 = icmp eq i64 %68, 1
  %70 = select i1 %69, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %71 = select i1 %69, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %72 = urem i64 %39, 2
  %73 = icmp eq i64 %72, 1
  %74 = select i1 %73, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %75 = select i1 %73, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %76 = urem i64 %39, 2
  %77 = icmp eq i64 %76, 1
  %78 = select i1 %77, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %79 = select i1 %77, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %80 = urem i64 %39, 2
  %81 = icmp eq i64 %80, 1
  %82 = select i1 %81, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %83 = select i1 %81, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %84 = urem i64 %39, 2
  %85 = icmp eq i64 %84, 1
  %86 = select i1 %85, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %87 = select i1 %85, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6))
  %88 = urem i64 %39, 2
  %89 = icmp eq i64 %88, 1
  %90 = select i1 %89, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %91 = select i1 %89, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %92 = urem i64 %39, 2
  %93 = icmp eq i64 %92, 1
  %94 = select i1 %93, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %95 = select i1 %93, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %96 = urem i64 %39, 2
  %97 = icmp eq i64 %96, 1
  %98 = select i1 %97, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %99 = select i1 %97, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %100 = urem i64 %39, 2
  %101 = icmp eq i64 %100, 1
  %102 = select i1 %101, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %103 = select i1 %101, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) null
  %104 = urem i64 %39, 2
  %105 = trunc i64 %104 to i1
  %106 = xor i1 %105, true
  %107 = zext i1 %106 to i64
  %108 = sdiv i32 %36, %24
  %109 = sdiv i32 %36, %22
  %110 = srem i32 %109, %9
  %111 = srem i32 %36, %22
  %112 = mul i32 %111, 64
  %113 = mul i32 %108, %10
  %114 = mul i32 %108, %9
  %115 = add i32 %114, %110
  %116 = mul i32 %115, %10
  %117 = sext i32 %113 to i64
  %118 = sext i32 %112 to i64
  %119 = sext i32 %110 to i64
  %120 = mul i64 %119, 128
  %121 = add i64 %117, %118
  %122 = mul i64 %121, %25
  %123 = mul i64 %122, 128
  %124 = add i64 %120, %123
  %125 = icmp sgt i64 %27, %118
  %126 = select i1 %125, i64 %27, i64 %118
  %127 = add i64 %118, 64
  %128 = icmp slt i64 %127, %126
  %129 = select i1 %128, i64 %127, i64 %126
  %130 = mul i64 %118, -1
  %131 = add i64 %129, %130
  %132 = icmp slt i64 %131, 64
  br i1 %132, label %133, label %134

133:                                              ; preds = %38
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %58, ptr addrspace(6) %59, i64 0, i64 8192, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %134

134:                                              ; preds = %133, %38
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 4, i64 %107)
  call void @load_gm_to_ubuf_2d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %124, i64 %131, i64 128, i64 %26, i64 1, ptr addrspace(6) %58, ptr addrspace(6) %59, i64 0, i64 %131, i64 128, i64 128, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %58, ptr addrspace(6) %59, i64 0, i64 8192, i64 1, ptr addrspace(6) %62, ptr addrspace(6) %63, i64 0, i64 8192, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %62, ptr addrspace(6) %63, i64 0, i64 8192, i64 1, float -1.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, float 0.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, float 1.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_1d_float(ptr addrspace(6) %62, ptr addrspace(6) %63, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) %66, ptr addrspace(6) %67, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) %66, ptr addrspace(6) %67, i64 0, i64 8192, i64 1, i32 32767, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) %66, ptr addrspace(6) %67, i64 0, i64 8192, i64 1, i32 16, ptr addrspace(6) %70, ptr addrspace(6) %71, i64 0, i64 8192, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) %70, ptr addrspace(6) %71, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) %74, ptr addrspace(6) %75, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) %74, ptr addrspace(6) %75, i64 0, i64 8192, i64 1, ptr addrspace(6) %78, ptr addrspace(6) %79, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) %78, ptr addrspace(6) %79, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) %82, ptr addrspace(6) %83, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_1d_float(ptr addrspace(6) %82, ptr addrspace(6) %83, i64 0, i64 8192, i64 1, ptr addrspace(6) %82, ptr addrspace(6) %83, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_sum_ar_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float 0.000000e+00)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsqrt_1d_float(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i32 32767, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcmp_eq_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vnot_1d_bool(ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vmaxs_vs_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 288 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 288 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsel_sv_1d_float(ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 288 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 288 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 800 to ptr addrspace(6)), i64 0, i64 16, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsel_vv_1d_bool_float(ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98304 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), i64 0, i64 8, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_2d_float(ptr addrspace(6) %82, ptr addrspace(6) %83, i64 0, i64 64, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, i64 1, i64 1, ptr addrspace(6) %86, ptr addrspace(6) %87, i64 0, i64 64, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), i64 0, i64 512, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  %135 = sext i32 %116 to i64
  %136 = mul i64 %135, 128
  %137 = mul i64 %118, 128
  %138 = add i64 %136, %137
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %86, ptr addrspace(6) %87, i64 0, i64 8192, i64 1, ptr addrspace(6) %54, ptr addrspace(6) %55, i64 0, i64 8192, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %139 = mul i64 %131, 128
  %140 = mul i64 %131, 128
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %54, ptr addrspace(6) %55, i64 0, i64 %139, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %138, i64 %140, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  br i1 %132, label %141, label %142

141:                                              ; preds = %134
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %50, ptr addrspace(6) %51, i64 0, i64 8192, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  br label %142

142:                                              ; preds = %141, %134
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @load_gm_to_ubuf_2d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %124, i64 %131, i64 128, i64 %26, i64 1, ptr addrspace(6) %50, ptr addrspace(6) %51, i64 0, i64 %131, i64 128, i64 128, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %50, ptr addrspace(6) %51, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, float -1.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, float 0.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, float 1.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_1d_float(ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, i32 32767, ptr addrspace(6) %90, ptr addrspace(6) %91, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98560 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) %90, ptr addrspace(6) %91, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 65536 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) %94, ptr addrspace(6) %95, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) %94, ptr addrspace(6) %95, i64 0, i64 8192, i64 1, ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 131328 to ptr addrspace(6)), i64 0, i64 8192, i64 1, ptr addrspace(6) %98, ptr addrspace(6) %99, i64 0, i64 8192, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %frame_rows = mul i32 %9, %10
  %143 = mul i32 %108, %frame_rows
  %144 = mul i32 %110, 405
  %145 = sext i32 %143 to i64
  %146 = sext i32 %144 to i64
  %147 = add i64 %118, %145
  %148 = add i64 %147, %146
  br i1 %132, label %149, label %150

149:                                              ; preds = %142
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %46, ptr addrspace(6) %47, i64 0, i64 64, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  br label %150

150:                                              ; preds = %149, %142
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %148, i64 %131, i64 1, ptr addrspace(6) %46, ptr addrspace(6) %47, i64 0, i64 %131, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %46, ptr addrspace(6) %47, i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i32 32767, ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164096 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164608 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164352 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) %98, ptr addrspace(6) %99, i64 0, i64 64, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 164864 to ptr addrspace(6)), i64 0, i64 64, i64 1, i64 1, i64 1, ptr addrspace(6) %102, ptr addrspace(6) %103, i64 0, i64 64, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 165120 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 165120 to ptr addrspace(6)), i64 0, i64 512, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %102, ptr addrspace(6) %103, i64 0, i64 8192, i64 1, ptr addrspace(6) %42, ptr addrspace(6) %43, i64 0, i64 8192, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  %151 = mul i64 %131, 128
  %152 = mul i64 %131, 128
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %42, ptr addrspace(6) %43, i64 0, i64 %151, i64 1, ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %138, i64 %152, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 4, i64 %107)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  %153 = add i64 %39, 1
  store i64 %153, ptr %15, align 4
  %154 = add nsw i32 %36, %11
  br label %35

155:                                              ; preds = %35
  %156 = add nsw i32 %29, %11
  br label %28

157:                                              ; preds = %28
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #3

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

attributes #0 = { alwaysinline }
attributes #1 = { alwaysinline }
attributes #2 = { nounwind readnone  }
attributes #3 = { nounwind  inaccessiblememonly }
attributes #4 = { nounwind }

