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

define dso_local void @JyKey.vector(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i64 %virtual_id) #1 {
  %16 = alloca i64, i64 1, align 8
  store i64 0, ptr %16, align 4
  %17 = mul i32 %12, %13
  %18 = mul i32 %17, %14
  %19 = add i64 %virtual_id, 0
  %20 = trunc i64 %19 to i32
  %21 = add i32 %8, 31
  %22 = sdiv i32 %21, 32
  %23 = mul i32 %11, %7
  %24 = mul i32 %23, %22
  %25 = mul i32 %7, %22
  %26 = mul i32 %7, 128
  %27 = mul i32 %7, 256
  %28 = mul i32 %7, 384
  %29 = mul i32 %7, 512
  call void @broadcast_scalar_float_to_1d(float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  %30 = sext i32 %8 to i64
  %31 = sext i32 %26 to i64
  %32 = sext i32 %27 to i64
  %33 = sext i32 %28 to i64
  %34 = sext i32 %29 to i64
  call void @broadcast_scalar_int32_t_to_1d(i32 1, ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 -65536, ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 -65536, ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcmp_eq_1d_float(ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vnot_1d_bool(ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 4)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 5)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 6)
  br label %35

35:                                               ; preds = %334, %15
  %36 = phi i32 [ %335, %334 ], [ %20, %15 ]
  %37 = icmp slt i32 %36, %18
  br i1 %37, label %38, label %336

38:                                               ; preds = %35
  %39 = call i64 @llvm.hivm.GET.CTRL()
  %40 = call i64 @llvm.hivm.SBITSET0(i64 %39, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %40)
  %41 = srem i32 %36, %12
  br label %42

42:                                               ; preds = %319, %38
  %43 = phi i32 [ %333, %319 ], [ %41, %38 ]
  %44 = icmp slt i32 %43, %24
  br i1 %44, label %45, label %334

45:                                               ; preds = %42
  %46 = load i64, ptr %16, align 4
  %47 = urem i64 %46, 2
  %48 = icmp eq i64 %47, 1
  %49 = select i1 %48, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %50 = select i1 %48, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %51 = urem i64 %46, 2
  %52 = icmp eq i64 %51, 1
  %53 = select i1 %52, ptr addrspace(6) inttoptr (i64 84224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 83968 to ptr addrspace(6))
  %54 = select i1 %52, ptr addrspace(6) inttoptr (i64 84224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 83968 to ptr addrspace(6))
  %55 = urem i64 %46, 2
  %56 = icmp eq i64 %55, 1
  %57 = select i1 %56, ptr addrspace(6) inttoptr (i64 75776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67584 to ptr addrspace(6))
  %58 = select i1 %56, ptr addrspace(6) inttoptr (i64 75776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67584 to ptr addrspace(6))
  %59 = urem i64 %46, 2
  %60 = icmp eq i64 %59, 1
  %61 = select i1 %60, ptr addrspace(6) inttoptr (i64 67328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67072 to ptr addrspace(6))
  %62 = select i1 %60, ptr addrspace(6) inttoptr (i64 67328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67072 to ptr addrspace(6))
  %63 = urem i64 %46, 2
  %64 = icmp eq i64 %63, 1
  %65 = select i1 %64, ptr addrspace(6) inttoptr (i64 58880 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50688 to ptr addrspace(6))
  %66 = select i1 %64, ptr addrspace(6) inttoptr (i64 58880 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50688 to ptr addrspace(6))
  %67 = urem i64 %46, 2
  %68 = icmp eq i64 %67, 1
  %69 = select i1 %68, ptr addrspace(6) inttoptr (i64 50432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50176 to ptr addrspace(6))
  %70 = select i1 %68, ptr addrspace(6) inttoptr (i64 50432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50176 to ptr addrspace(6))
  %71 = urem i64 %46, 2
  %72 = icmp eq i64 %71, 1
  %73 = select i1 %72, ptr addrspace(6) inttoptr (i64 41984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6))
  %74 = select i1 %72, ptr addrspace(6) inttoptr (i64 41984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6))
  %75 = urem i64 %46, 2
  %76 = icmp eq i64 %75, 1
  %77 = select i1 %76, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33280 to ptr addrspace(6))
  %78 = select i1 %76, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33280 to ptr addrspace(6))
  %79 = urem i64 %46, 2
  %80 = icmp eq i64 %79, 1
  %81 = select i1 %80, ptr addrspace(6) inttoptr (i64 25088 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16896 to ptr addrspace(6))
  %82 = select i1 %80, ptr addrspace(6) inttoptr (i64 25088 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16896 to ptr addrspace(6))
  %83 = urem i64 %46, 2
  %84 = icmp eq i64 %83, 1
  %85 = select i1 %84, ptr addrspace(6) inttoptr (i64 16640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6))
  %86 = select i1 %84, ptr addrspace(6) inttoptr (i64 16640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6))
  %87 = urem i64 %46, 2
  %88 = icmp eq i64 %87, 1
  %89 = select i1 %88, ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) null
  %90 = select i1 %88, ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) null
  %91 = urem i64 %46, 2
  %92 = icmp eq i64 %91, 1
  %93 = select i1 %92, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %94 = select i1 %92, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %95 = urem i64 %46, 2
  %96 = icmp eq i64 %95, 1
  %97 = select i1 %96, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %98 = select i1 %96, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %99 = urem i64 %46, 2
  %100 = icmp eq i64 %99, 1
  %101 = select i1 %100, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %102 = select i1 %100, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %103 = urem i64 %46, 2
  %104 = icmp eq i64 %103, 1
  %105 = select i1 %104, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %106 = select i1 %104, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %107 = urem i64 %46, 2
  %108 = icmp eq i64 %107, 1
  %109 = select i1 %108, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %110 = select i1 %108, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %111 = urem i64 %46, 2
  %112 = icmp eq i64 %111, 1
  %113 = select i1 %112, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %114 = select i1 %112, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %115 = urem i64 %46, 2
  %116 = icmp eq i64 %115, 1
  %117 = select i1 %116, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %118 = select i1 %116, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %119 = urem i64 %46, 2
  %120 = icmp eq i64 %119, 1
  %121 = select i1 %120, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %122 = select i1 %120, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %123 = urem i64 %46, 2
  %124 = icmp eq i64 %123, 1
  %125 = select i1 %124, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %126 = select i1 %124, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %127 = urem i64 %46, 2
  %128 = icmp eq i64 %127, 1
  %129 = select i1 %128, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %130 = select i1 %128, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %131 = urem i64 %46, 2
  %132 = icmp eq i64 %131, 1
  %133 = select i1 %132, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %134 = select i1 %132, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %135 = urem i64 %46, 2
  %136 = icmp eq i64 %135, 1
  %137 = select i1 %136, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %138 = select i1 %136, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %139 = urem i64 %46, 2
  %140 = icmp eq i64 %139, 1
  %141 = select i1 %140, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %142 = select i1 %140, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %143 = urem i64 %46, 2
  %144 = icmp eq i64 %143, 1
  %145 = select i1 %144, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %146 = select i1 %144, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %147 = urem i64 %46, 2
  %148 = icmp eq i64 %147, 1
  %149 = select i1 %148, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %150 = select i1 %148, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %151 = urem i64 %46, 2
  %152 = icmp eq i64 %151, 1
  %153 = select i1 %152, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %154 = select i1 %152, ptr addrspace(6) inttoptr (i64 100864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %155 = urem i64 %46, 2
  %156 = trunc i64 %155 to i1
  %157 = select i1 %156, i64 5, i64 6
  %158 = trunc i64 %155 to i1
  %159 = select i1 %158, i64 3, i64 4
  %160 = trunc i64 %155 to i1
  %161 = select i1 %160, i64 1, i64 2
  %162 = trunc i64 %155 to i1
  %163 = xor i1 %162, true
  %164 = zext i1 %163 to i64
  %165 = sdiv i32 %43, %25
  %166 = sdiv i32 %43, %22
  %167 = srem i32 %166, %7
  %168 = srem i32 %43, %22
  %169 = mul i32 %168, 32
  %170 = mul i32 %167, 128
  %171 = add i32 %165, %9
  %172 = sub i32 %171, 2
  %173 = mul i32 %172, %7
  %174 = add i32 %173, %167
  %175 = mul i32 %174, %8
  %176 = sext i32 %175 to i64
  %177 = sext i32 %169 to i64
  %178 = mul i64 %176, 128
  %179 = mul i64 %177, 128
  %180 = add i64 %178, %179
  %181 = icmp sge i32 %172, 0
  %182 = icmp slt i32 %172, %10
  %183 = icmp sgt i64 %30, %177
  %184 = select i1 %183, i64 %30, i64 %177
  %185 = add i64 %177, 32
  %186 = icmp slt i64 %185, %184
  %187 = select i1 %186, i64 %185, i64 %184
  %188 = mul i64 %177, -1
  %189 = add i64 %187, %188
  %190 = zext i1 %181 to i64
  %191 = zext i1 %182 to i64
  %192 = mul i64 %191, 32
  %193 = icmp sgt i64 %192, 0
  %194 = select i1 %193, i64 %192, i64 0
  %195 = mul i64 %190, 32
  %196 = icmp sgt i64 %195, 0
  %197 = select i1 %196, i64 %195, i64 0
  %198 = mul i64 %177, -1
  %199 = add i64 %187, %198
  %200 = icmp sgt i64 %199, 0
  %201 = select i1 %200, i64 %199, i64 0
  %202 = icmp slt i64 %201, %197
  %203 = select i1 %202, i64 %201, i64 %197
  %204 = icmp slt i64 %203, %194
  %205 = select i1 %204, i64 %203, i64 %194
  %206 = icmp slt i64 %205, 32
  br i1 %206, label %207, label %208

207:                                              ; preds = %45
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %89, ptr addrspace(6) %90, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %208

208:                                              ; preds = %207, %45
  %209 = mul i64 %205, 128
  %210 = mul i64 %205, 128
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %161)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %180, i64 %209, i64 1, ptr addrspace(6) %89, ptr addrspace(6) %90, i64 0, i64 %210, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %89, ptr addrspace(6) %90, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %211 = sext i32 %170 to i64
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %211, i64 128, i64 1, ptr addrspace(6) %85, ptr addrspace(6) %86, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %85, ptr addrspace(6) %86, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %212 = add i32 %171, -1
  %213 = mul i32 %212, %7
  %214 = add i32 %213, %167
  %215 = mul i32 %214, %8
  %216 = sext i32 %215 to i64
  %217 = mul i64 %216, 128
  %218 = mul i64 %177, 128
  %219 = add i64 %217, %218
  %220 = icmp sge i32 %212, 0
  %221 = icmp slt i32 %212, %10
  %222 = zext i1 %220 to i64
  %223 = zext i1 %221 to i64
  %224 = mul i64 %223, 32
  %225 = icmp sgt i64 %224, 0
  %226 = select i1 %225, i64 %224, i64 0
  %227 = mul i64 %222, 32
  %228 = icmp sgt i64 %227, 0
  %229 = select i1 %228, i64 %227, i64 0
  %230 = icmp slt i64 %201, %229
  %231 = select i1 %230, i64 %201, i64 %229
  %232 = icmp slt i64 %231, %226
  %233 = select i1 %232, i64 %231, i64 %226
  %234 = icmp slt i64 %233, 32
  br i1 %234, label %235, label %236

235:                                              ; preds = %208
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %81, ptr addrspace(6) %82, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %236

236:                                              ; preds = %235, %208
  %237 = mul i64 %233, 128
  %238 = mul i64 %233, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %219, i64 %237, i64 1, ptr addrspace(6) %81, ptr addrspace(6) %82, i64 0, i64 %238, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %81, ptr addrspace(6) %82, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %239 = add i64 %31, %211
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %239, i64 128, i64 1, ptr addrspace(6) %77, ptr addrspace(6) %78, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %77, ptr addrspace(6) %78, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %164)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %93, ptr addrspace(6) %94, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %93, ptr addrspace(6) %94, i64 0, i64 4096, i64 1, ptr addrspace(6) %97, ptr addrspace(6) %98, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %97, ptr addrspace(6) %98, i64 0, i64 4096, i64 1, ptr addrspace(6) %101, ptr addrspace(6) %102, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %101, ptr addrspace(6) %102, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %240 = mul i32 %171, %7
  %241 = add i32 %240, %167
  %242 = mul i32 %241, %8
  %243 = sext i32 %242 to i64
  %244 = mul i64 %243, 128
  %245 = mul i64 %177, 128
  %246 = add i64 %244, %245
  %247 = icmp sge i32 %171, 0
  %248 = icmp slt i32 %171, %10
  %249 = zext i1 %247 to i64
  %250 = zext i1 %248 to i64
  %251 = mul i64 %250, 32
  %252 = icmp sgt i64 %251, 0
  %253 = select i1 %252, i64 %251, i64 0
  %254 = mul i64 %249, 32
  %255 = icmp sgt i64 %254, 0
  %256 = select i1 %255, i64 %254, i64 0
  %257 = icmp slt i64 %201, %256
  %258 = select i1 %257, i64 %201, i64 %256
  %259 = icmp slt i64 %258, %253
  %260 = select i1 %259, i64 %258, i64 %253
  %261 = icmp slt i64 %260, 32
  br i1 %261, label %262, label %263

262:                                              ; preds = %236
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %73, ptr addrspace(6) %74, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %263

263:                                              ; preds = %262, %236
  %264 = mul i64 %260, 128
  %265 = mul i64 %260, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %246, i64 %264, i64 1, ptr addrspace(6) %73, ptr addrspace(6) %74, i64 0, i64 %265, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %73, ptr addrspace(6) %74, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %266 = add i64 %32, %211
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %266, i64 128, i64 1, ptr addrspace(6) %69, ptr addrspace(6) %70, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %69, ptr addrspace(6) %70, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %105, ptr addrspace(6) %106, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %105, ptr addrspace(6) %106, i64 0, i64 4096, i64 1, ptr addrspace(6) %109, ptr addrspace(6) %110, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %109, ptr addrspace(6) %110, i64 0, i64 4096, i64 1, ptr addrspace(6) %113, ptr addrspace(6) %114, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %113, ptr addrspace(6) %114, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %267 = add i32 %171, 1
  %268 = mul i32 %267, %7
  %269 = add i32 %268, %167
  %270 = mul i32 %269, %8
  %271 = sext i32 %270 to i64
  %272 = mul i64 %271, 128
  %273 = mul i64 %177, 128
  %274 = add i64 %272, %273
  %275 = icmp sge i32 %267, 0
  %276 = icmp slt i32 %267, %10
  %277 = zext i1 %275 to i64
  %278 = zext i1 %276 to i64
  %279 = mul i64 %278, 32
  %280 = icmp sgt i64 %279, 0
  %281 = select i1 %280, i64 %279, i64 0
  %282 = mul i64 %277, 32
  %283 = icmp sgt i64 %282, 0
  %284 = select i1 %283, i64 %282, i64 0
  %285 = icmp slt i64 %201, %284
  %286 = select i1 %285, i64 %201, i64 %284
  %287 = icmp slt i64 %286, %281
  %288 = select i1 %287, i64 %286, i64 %281
  %289 = icmp slt i64 %288, 32
  br i1 %289, label %290, label %291

290:                                              ; preds = %263
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %291

291:                                              ; preds = %290, %263
  %292 = mul i64 %288, 128
  %293 = mul i64 %288, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %274, i64 %292, i64 1, ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 %293, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %294 = add i64 %33, %211
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %294, i64 128, i64 1, ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %161)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %117, ptr addrspace(6) %118, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %117, ptr addrspace(6) %118, i64 0, i64 4096, i64 1, ptr addrspace(6) %121, ptr addrspace(6) %122, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %121, ptr addrspace(6) %122, i64 0, i64 4096, i64 1, ptr addrspace(6) %125, ptr addrspace(6) %126, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %125, ptr addrspace(6) %126, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %295 = add i32 %171, 2
  %296 = mul i32 %295, %7
  %297 = add i32 %296, %167
  %298 = mul i32 %297, %8
  %299 = sext i32 %298 to i64
  %300 = mul i64 %299, 128
  %301 = mul i64 %177, 128
  %302 = add i64 %300, %301
  %303 = icmp sge i32 %295, 0
  %304 = icmp slt i32 %295, %10
  %305 = zext i1 %303 to i64
  %306 = zext i1 %304 to i64
  %307 = mul i64 %306, 32
  %308 = icmp sgt i64 %307, 0
  %309 = select i1 %308, i64 %307, i64 0
  %310 = mul i64 %305, 32
  %311 = icmp sgt i64 %310, 0
  %312 = select i1 %311, i64 %310, i64 0
  %313 = icmp slt i64 %201, %312
  %314 = select i1 %313, i64 %201, i64 %312
  %315 = icmp slt i64 %314, %309
  %316 = select i1 %315, i64 %314, i64 %309
  %317 = icmp slt i64 %316, 32
  br i1 %317, label %318, label %319

318:                                              ; preds = %291
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %57, ptr addrspace(6) %58, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %319

319:                                              ; preds = %318, %291
  %320 = mul i64 %316, 128
  %321 = mul i64 %316, 128
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %159)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %302, i64 %320, i64 1, ptr addrspace(6) %57, ptr addrspace(6) %58, i64 0, i64 %321, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %57, ptr addrspace(6) %58, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %159)
  %322 = add i64 %34, %211
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %157)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %322, i64 128, i64 1, ptr addrspace(6) %53, ptr addrspace(6) %54, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %53, ptr addrspace(6) %54, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %157)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 183168 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %129, ptr addrspace(6) %130, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %129, ptr addrspace(6) %130, i64 0, i64 4096, i64 1, ptr addrspace(6) %133, ptr addrspace(6) %134, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %133, ptr addrspace(6) %134, i64 0, i64 4096, i64 1, ptr addrspace(6) %137, ptr addrspace(6) %138, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %137, ptr addrspace(6) %138, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float -1.000000e+00, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float 0.000000e+00, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float 1.000000e+00, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 32767, ptr addrspace(6) %141, ptr addrspace(6) %142, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) %141, ptr addrspace(6) %142, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %145, ptr addrspace(6) %146, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) %145, ptr addrspace(6) %146, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_1d_float(ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 4096, i64 1, ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_sum_ar_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 166912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166912 to ptr addrspace(6)), i64 0, i64 2048, i64 1, float 0.000000e+00)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsqrt_1d_float(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 32767, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166784 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcmp_eq_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vnot_1d_bool(ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vmaxs_vs_1d_float(ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 150560 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150560 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsel_sv_1d_float(ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1, float 0x3EB0C00000000000, ptr addrspace(6) inttoptr (i64 150560 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150560 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150688 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150688 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150816 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150816 to ptr addrspace(6)), i64 0, i64 16, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsel_vv_1d_bool_float(ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117248 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150400 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150688 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150688 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150656 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150656 to ptr addrspace(6)), i64 0, i64 8, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_2d_float(ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150528 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 1, i64 1, ptr addrspace(6) %153, ptr addrspace(6) %154, i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150656 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150656 to ptr addrspace(6)), i64 0, i64 256, i64 1)
  %323 = mul i32 %165, %7
  %324 = add i32 %323, %167
  %325 = mul i32 %324, %8
  %326 = sext i32 %325 to i64
  %327 = mul i64 %326, 128
  %328 = mul i64 %177, 128
  %329 = add i64 %327, %328
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %153, ptr addrspace(6) %154, i64 0, i64 4096, i64 1, ptr addrspace(6) %49, ptr addrspace(6) %50, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %330 = mul i64 %189, 128
  %331 = mul i64 %189, 128
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.BARRIER(i64 5)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %49, ptr addrspace(6) %50, i64 0, i64 %330, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %329, i64 %331, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %164)
  %332 = add i64 %46, 1
  store i64 %332, ptr %16, align 4
  %333 = add nsw i32 %43, 40
  br label %42

334:                                              ; preds = %42
  %335 = add nsw i32 %36, 40
  br label %35

336:                                              ; preds = %35
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 4)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 5)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 6)
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

