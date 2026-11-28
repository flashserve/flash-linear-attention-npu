; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: alwaysinline
define private void @arange_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) #0 {
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, ptr addrspace(6) %1, 1
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, i64 %4, 4, 0
  %13 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr %13, align 8
  call void @_mlir_ciface_arange_1d_int32_t(ptr %13, i64 %5, i64 %6)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_arange_1d_int32_t(ptr, i64, i64) #0

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
define private void @load_gm_to_ubuf_2d_float(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14, float %15, i64 %16) #0 {
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
  call void @_mlir_ciface_load_gm_to_ubuf_2d_float(ptr %25, ptr %33, i32 %14, float %15, i64 %16)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_2d_float(ptr, ptr, i32, float, i64) #0

; Function Attrs: alwaysinline
define private void @broadcast_last_axis_align_2d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18) #0 {
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, ptr addrspace(6) %1, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %2, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %3, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %5, 4, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %4, 3, 1
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %6, 4, 1
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, ptr addrspace(6) %8, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %9, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %10, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %12, 4, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %11, 3, 1
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %13, 4, 1
  %35 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, ptr %35, align 8
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, ptr addrspace(6) %15, 1
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, i64 %16, 2
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %17, 3, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %18, 4, 0
  %41 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, ptr %41, align 8
  call void @_mlir_ciface_broadcast_last_axis_align_2d_int32_t(ptr %27, ptr %35, ptr %41)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_last_axis_align_2d_int32_t(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @broadcast_first_axis_align_2d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #0 {
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr addrspace(6) %1, 1
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, ptr addrspace(6) %8, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %9, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %10, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %12, 4, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %11, 3, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %13, 4, 1
  %30 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr %30, align 8
  call void @_mlir_ciface_broadcast_first_axis_align_2d_int32_t(ptr %22, ptr %30)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_first_axis_align_2d_int32_t(ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vcmp_eq_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
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
  call void @_mlir_ciface_vcmp_eq_1d_int32_t(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcmp_eq_1d_int32_t(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vcast_bool_to_float_1d_with_temp(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, i32 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %6, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %7, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %8, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %9, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %11, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %12, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %13, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %14, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vcast_bool_to_float_1d_with_temp(ptr %22, ptr %28, ptr %34, i32 %15)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcast_bool_to_float_1d_with_temp(ptr, ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @transpose_2d_with_last_axis_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #0 {
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr addrspace(6) %1, 1
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, ptr addrspace(6) %8, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %9, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %10, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %12, 4, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %11, 3, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %13, 4, 1
  %30 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr %30, align 8
  call void @_mlir_ciface_transpose_2d_with_last_axis_float(ptr %22, ptr %30)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_transpose_2d_with_last_axis_float(ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadd_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
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
  call void @_mlir_ciface_vadd_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadd_1d_float(ptr, ptr, ptr, ptr) #0

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
define private void @store_ubuf_to_gm_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(1) %7, ptr addrspace(1) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %5, 4, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %4, 3, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %6, 4, 1
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, ptr %23, align 8
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %7, 0
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, ptr addrspace(1) %8, 1
  %26 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %27, i64 %12, 4, 0
  %29 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %28, i64 %11, 3, 1
  %30 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %29, i64 %13, 4, 1
  %31 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %30, ptr %31, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_2d_float(ptr %23, ptr %31, i32 %14)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_2d_float(ptr, ptr, i32) #0

define dso_local void @JyInit.vector(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, i32 %6, i32 %7, i32 %8, i32 %9, i64 %virtual_id) #1 {
  %11 = alloca i64, i64 1, align 8
  store i64 0, ptr %11, align 4
  %12 = mul i32 %7, %8
  %13 = mul i32 %12, %9
  %14 = add i64 %virtual_id, 0
  %15 = trunc i64 %14 to i32
  call void @arange_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, i64 0, i64 1)
  %16 = mul i32 %6, 4
  call void @llvm.hivm.BARRIER(i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  br label %17

17:                                               ; preds = %91, %10
  %18 = phi i32 [ %92, %91 ], [ %15, %10 ]
  %19 = icmp slt i32 %18, %13
  br i1 %19, label %20, label %93

20:                                               ; preds = %17
  %21 = call i64 @llvm.hivm.GET.CTRL()
  %22 = call i64 @llvm.hivm.SBITSET0(i64 %21, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %22)
  %23 = srem i32 %18, %7
  br label %24

24:                                               ; preds = %27, %20
  %25 = phi i32 [ %90, %27 ], [ %23, %20 ]
  %26 = icmp slt i32 %25, %16
  br i1 %26, label %27, label %91

27:                                               ; preds = %24
  %28 = load i64, ptr %11, align 4
  %29 = urem i64 %28, 2
  %30 = icmp eq i64 %29, 1
  %31 = select i1 %30, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %32 = select i1 %30, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %33 = urem i64 %28, 2
  %34 = icmp eq i64 %33, 1
  %35 = select i1 %34, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %36 = select i1 %34, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %37 = urem i64 %28, 2
  %38 = icmp eq i64 %37, 1
  %39 = select i1 %38, ptr addrspace(6) inttoptr (i64 133984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 17152 to ptr addrspace(6))
  %40 = select i1 %38, ptr addrspace(6) inttoptr (i64 133984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 17152 to ptr addrspace(6))
  %41 = urem i64 %28, 2
  %42 = icmp eq i64 %41, 1
  %43 = select i1 %42, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %44 = select i1 %42, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %45 = urem i64 %28, 2
  %46 = icmp eq i64 %45, 1
  %47 = select i1 %46, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %48 = select i1 %46, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %49 = urem i64 %28, 2
  %50 = icmp eq i64 %49, 1
  %51 = select i1 %50, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %52 = select i1 %50, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %53 = urem i64 %28, 2
  %54 = icmp eq i64 %53, 1
  %55 = select i1 %54, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %56 = select i1 %54, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %57 = urem i64 %28, 2
  %58 = icmp eq i64 %57, 1
  %59 = select i1 %58, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %60 = select i1 %58, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %61 = urem i64 %28, 2
  %62 = icmp eq i64 %61, 1
  %63 = select i1 %62, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %64 = select i1 %62, ptr addrspace(6) inttoptr (i64 150368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 68352 to ptr addrspace(6))
  %65 = urem i64 %28, 2
  %66 = icmp eq i64 %65, 1
  %67 = select i1 %66, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %68 = select i1 %66, ptr addrspace(6) inttoptr (i64 117600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 768 to ptr addrspace(6))
  %69 = urem i64 %28, 2
  %70 = trunc i64 %69 to i1
  %71 = xor i1 %70, true
  %72 = zext i1 %71 to i64
  %73 = sdiv i32 %25, 4
  %74 = sext i32 %73 to i64
  %75 = mul i64 %74, 16384
  %76 = sdiv i32 %25, 2
  %77 = srem i32 %76, 2
  %78 = mul i32 %77, 64
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, i32 %78, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %79 = srem i32 %25, 2
  %80 = mul i32 %79, 64
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 64, i64 1, i32 %80, ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), i64 0, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %81 = sext i32 %78 to i64
  %82 = sext i32 %80 to i64
  %83 = add i64 %82, %75
  %84 = mul i64 %81, 128
  %85 = add i64 %83, %84
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 4, i64 %72)
  call void @load_gm_to_ubuf_2d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %85, i64 64, i64 64, i64 128, i64 1, ptr addrspace(6) %43, ptr addrspace(6) %44, i64 0, i64 64, i64 64, i64 64, i64 1, i32 0, float 0.000000e+00, i64 0)
  %86 = add i64 %81, %75
  %87 = mul i64 %82, 128
  %88 = add i64 %86, %87
  call void @load_gm_to_ubuf_2d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %88, i64 64, i64 64, i64 128, i64 1, ptr addrspace(6) %39, ptr addrspace(6) %40, i64 0, i64 64, i64 64, i64 64, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @broadcast_last_axis_align_2d_int32_t(ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 64, i64 1, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), i64 0, i64 64, i64 64, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 49920 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 49920 to ptr addrspace(6)), i64 0, i64 512, i64 1)
  call void @broadcast_first_axis_align_2d_int32_t(ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), i64 0, i64 1, i64 64, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 51968 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 51968 to ptr addrspace(6)), i64 0, i64 64, i64 64, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcmp_eq_1d_int32_t(ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 51968 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 51968 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %72)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bool_to_float_1d_with_temp(ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %47, ptr addrspace(6) %48, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 84736 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84736 to ptr addrspace(6)), i64 0, i64 24, i64 1, i32 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @transpose_2d_with_last_axis_float(ptr addrspace(6) %39, ptr addrspace(6) %40, i64 0, i64 64, i64 64, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 84832 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84832 to ptr addrspace(6)), i64 0, i64 64, i64 64, i64 64, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) %43, ptr addrspace(6) %44, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 84832 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84832 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %51, ptr addrspace(6) %52, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %51, ptr addrspace(6) %52, i64 0, i64 4096, i64 1, float 5.000000e-01, ptr addrspace(6) %55, ptr addrspace(6) %56, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) %55, ptr addrspace(6) %56, i64 0, i64 4096, i64 1, ptr addrspace(6) %47, ptr addrspace(6) %48, i64 0, i64 4096, i64 1, ptr addrspace(6) %59, ptr addrspace(6) %60, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %47, ptr addrspace(6) %48, i64 0, i64 4096, i64 1, float 0x3F8420B520000000, ptr addrspace(6) %63, ptr addrspace(6) %64, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %59, ptr addrspace(6) %60, i64 0, i64 4096, i64 1, float 0xBEF95207C0000000, ptr addrspace(6) inttoptr (i64 101216 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 101216 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) %63, ptr addrspace(6) %64, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 101216 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 101216 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %35, ptr addrspace(6) %36, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %59, ptr addrspace(6) %60, i64 0, i64 4096, i64 1, float -1.000000e+00, ptr addrspace(6) %67, ptr addrspace(6) %68, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) %67, ptr addrspace(6) %68, i64 0, i64 4096, i64 1, float 0.000000e+00, ptr addrspace(6) %31, ptr addrspace(6) %32, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  ; Complete the Newton system sign change before the tile leaves UB.
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) %31, ptr addrspace(6) %32, i64 0, i64 4096, i64 1, float -1.000000e+00, ptr addrspace(6) %31, ptr addrspace(6) %32, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.BARRIER(i64 5)
  call void @store_ubuf_to_gm_2d_float(ptr addrspace(6) %31, ptr addrspace(6) %32, i64 0, i64 64, i64 64, i64 64, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %85, i64 64, i64 64, i64 128, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 4, i64 %72)
  call void @store_ubuf_to_gm_2d_float(ptr addrspace(6) %35, ptr addrspace(6) %36, i64 0, i64 64, i64 64, i64 64, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %85, i64 64, i64 64, i64 128, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %72)
  %89 = add i64 %28, 1
  store i64 %89, ptr %11, align 4
  %90 = add nsw i32 %25, %7
  br label %24

91:                                               ; preds = %24
  %92 = add nsw i32 %18, %7
  br label %17

93:                                               ; preds = %17
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
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

