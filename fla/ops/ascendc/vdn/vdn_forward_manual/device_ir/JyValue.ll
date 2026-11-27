; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

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

define dso_local void @JyValue.vector(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i64 %virtual_id) #1 {
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
  %30 = sext i32 %8 to i64
  %31 = sext i32 %26 to i64
  %32 = sext i32 %27 to i64
  %33 = sext i32 %28 to i64
  %34 = sext i32 %29 to i64
  call void @broadcast_scalar_int32_t_to_1d(i32 1, ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @broadcast_scalar_int32_t_to_1d(i32 -65536, ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
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

42:                                               ; preds = %326, %38
  %43 = phi i32 [ %333, %326 ], [ %41, %38 ]
  %44 = icmp slt i32 %43, %24
  br i1 %44, label %45, label %334

45:                                               ; preds = %42
  %46 = load i64, ptr %16, align 4
  %47 = urem i64 %46, 2
  %48 = icmp eq i64 %47, 1
  %49 = select i1 %48, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %50 = select i1 %48, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %51 = urem i64 %46, 2
  %52 = icmp eq i64 %51, 1
  %53 = select i1 %52, ptr addrspace(6) inttoptr (i64 84544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %54 = select i1 %52, ptr addrspace(6) inttoptr (i64 84544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84480 to ptr addrspace(6))
  %55 = urem i64 %46, 2
  %56 = icmp eq i64 %55, 1
  %57 = select i1 %56, ptr addrspace(6) inttoptr (i64 84224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 83968 to ptr addrspace(6))
  %58 = select i1 %56, ptr addrspace(6) inttoptr (i64 84224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 83968 to ptr addrspace(6))
  %59 = urem i64 %46, 2
  %60 = icmp eq i64 %59, 1
  %61 = select i1 %60, ptr addrspace(6) inttoptr (i64 75776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67584 to ptr addrspace(6))
  %62 = select i1 %60, ptr addrspace(6) inttoptr (i64 75776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67584 to ptr addrspace(6))
  %63 = urem i64 %46, 2
  %64 = icmp eq i64 %63, 1
  %65 = select i1 %64, ptr addrspace(6) inttoptr (i64 67328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67072 to ptr addrspace(6))
  %66 = select i1 %64, ptr addrspace(6) inttoptr (i64 67328 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 67072 to ptr addrspace(6))
  %67 = urem i64 %46, 2
  %68 = icmp eq i64 %67, 1
  %69 = select i1 %68, ptr addrspace(6) inttoptr (i64 58880 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50688 to ptr addrspace(6))
  %70 = select i1 %68, ptr addrspace(6) inttoptr (i64 58880 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50688 to ptr addrspace(6))
  %71 = urem i64 %46, 2
  %72 = icmp eq i64 %71, 1
  %73 = select i1 %72, ptr addrspace(6) inttoptr (i64 50432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50176 to ptr addrspace(6))
  %74 = select i1 %72, ptr addrspace(6) inttoptr (i64 50432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 50176 to ptr addrspace(6))
  %75 = urem i64 %46, 2
  %76 = icmp eq i64 %75, 1
  %77 = select i1 %76, ptr addrspace(6) inttoptr (i64 41984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6))
  %78 = select i1 %76, ptr addrspace(6) inttoptr (i64 41984 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6))
  %79 = urem i64 %46, 2
  %80 = icmp eq i64 %79, 1
  %81 = select i1 %80, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33280 to ptr addrspace(6))
  %82 = select i1 %80, ptr addrspace(6) inttoptr (i64 33536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33280 to ptr addrspace(6))
  %83 = urem i64 %46, 2
  %84 = icmp eq i64 %83, 1
  %85 = select i1 %84, ptr addrspace(6) inttoptr (i64 25088 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16896 to ptr addrspace(6))
  %86 = select i1 %84, ptr addrspace(6) inttoptr (i64 25088 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16896 to ptr addrspace(6))
  %87 = urem i64 %46, 2
  %88 = icmp eq i64 %87, 1
  %89 = select i1 %88, ptr addrspace(6) inttoptr (i64 16640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6))
  %90 = select i1 %88, ptr addrspace(6) inttoptr (i64 16640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6))
  %91 = urem i64 %46, 2
  %92 = icmp eq i64 %91, 1
  %93 = select i1 %92, ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) null
  %94 = select i1 %92, ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) null
  %95 = urem i64 %46, 2
  %96 = icmp eq i64 %95, 1
  %97 = select i1 %96, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %98 = select i1 %96, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %99 = urem i64 %46, 2
  %100 = icmp eq i64 %99, 1
  %101 = select i1 %100, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %102 = select i1 %100, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %103 = urem i64 %46, 2
  %104 = icmp eq i64 %103, 1
  %105 = select i1 %104, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %106 = select i1 %104, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %107 = urem i64 %46, 2
  %108 = icmp eq i64 %107, 1
  %109 = select i1 %108, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %110 = select i1 %108, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %111 = urem i64 %46, 2
  %112 = icmp eq i64 %111, 1
  %113 = select i1 %112, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %114 = select i1 %112, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %115 = urem i64 %46, 2
  %116 = icmp eq i64 %115, 1
  %117 = select i1 %116, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %118 = select i1 %116, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %119 = urem i64 %46, 2
  %120 = icmp eq i64 %119, 1
  %121 = select i1 %120, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %122 = select i1 %120, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %123 = urem i64 %46, 2
  %124 = icmp eq i64 %123, 1
  %125 = select i1 %124, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %126 = select i1 %124, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %127 = urem i64 %46, 2
  %128 = icmp eq i64 %127, 1
  %129 = select i1 %128, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %130 = select i1 %128, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %131 = urem i64 %46, 2
  %132 = icmp eq i64 %131, 1
  %133 = select i1 %132, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %134 = select i1 %132, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %135 = urem i64 %46, 2
  %136 = icmp eq i64 %135, 1
  %137 = select i1 %136, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %138 = select i1 %136, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %139 = urem i64 %46, 2
  %140 = icmp eq i64 %139, 1
  %141 = select i1 %140, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %142 = select i1 %140, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %143 = urem i64 %46, 2
  %144 = icmp eq i64 %143, 1
  %145 = select i1 %144, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %146 = select i1 %144, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %147 = urem i64 %46, 2
  %148 = icmp eq i64 %147, 1
  %149 = select i1 %148, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %150 = select i1 %148, ptr addrspace(6) inttoptr (i64 100992 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 84608 to ptr addrspace(6))
  %151 = urem i64 %46, 2
  %152 = trunc i64 %151 to i1
  %153 = select i1 %152, i64 5, i64 6
  %154 = trunc i64 %151 to i1
  %155 = select i1 %154, i64 3, i64 4
  %156 = trunc i64 %151 to i1
  %157 = select i1 %156, i64 1, i64 2
  %158 = trunc i64 %151 to i1
  %159 = xor i1 %158, true
  %160 = zext i1 %159 to i64
  %161 = sdiv i32 %43, %25
  %162 = sdiv i32 %43, %22
  %163 = srem i32 %162, %7
  %164 = srem i32 %43, %22
  %165 = mul i32 %164, 32
  %166 = mul i32 %163, 128
  %167 = add i32 %161, %9
  %168 = sub i32 %167, 2
  %169 = mul i32 %168, %7
  %170 = add i32 %169, %163
  %171 = mul i32 %170, %8
  %172 = sext i32 %171 to i64
  %173 = sext i32 %165 to i64
  %174 = mul i64 %172, 128
  %175 = mul i64 %173, 128
  %176 = add i64 %174, %175
  %177 = icmp sge i32 %168, 0
  %178 = icmp slt i32 %168, %10
  %179 = icmp sgt i64 %30, %173
  %180 = select i1 %179, i64 %30, i64 %173
  %181 = add i64 %173, 32
  %182 = icmp slt i64 %181, %180
  %183 = select i1 %182, i64 %181, i64 %180
  %184 = mul i64 %173, -1
  %185 = add i64 %183, %184
  %186 = zext i1 %177 to i64
  %187 = zext i1 %178 to i64
  %188 = mul i64 %187, 32
  %189 = icmp sgt i64 %188, 0
  %190 = select i1 %189, i64 %188, i64 0
  %191 = mul i64 %186, 32
  %192 = icmp sgt i64 %191, 0
  %193 = select i1 %192, i64 %191, i64 0
  %194 = mul i64 %173, -1
  %195 = add i64 %183, %194
  %196 = icmp sgt i64 %195, 0
  %197 = select i1 %196, i64 %195, i64 0
  %198 = icmp slt i64 %197, %193
  %199 = select i1 %198, i64 %197, i64 %193
  %200 = icmp slt i64 %199, %190
  %201 = select i1 %200, i64 %199, i64 %190
  %202 = icmp slt i64 %201, 32
  br i1 %202, label %203, label %204

203:                                              ; preds = %45
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %93, ptr addrspace(6) %94, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %204

204:                                              ; preds = %203, %45
  %205 = mul i64 %201, 128
  %206 = mul i64 %201, 128
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %157)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %176, i64 %205, i64 1, ptr addrspace(6) %93, ptr addrspace(6) %94, i64 0, i64 %206, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %93, ptr addrspace(6) %94, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %207 = sext i32 %166 to i64
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %207, i64 128, i64 1, ptr addrspace(6) %89, ptr addrspace(6) %90, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %89, ptr addrspace(6) %90, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %208 = add i32 %167, -1
  %209 = mul i32 %208, %7
  %210 = add i32 %209, %163
  %211 = mul i32 %210, %8
  %212 = sext i32 %211 to i64
  %213 = mul i64 %212, 128
  %214 = mul i64 %173, 128
  %215 = add i64 %213, %214
  %216 = icmp sge i32 %208, 0
  %217 = icmp slt i32 %208, %10
  %218 = zext i1 %216 to i64
  %219 = zext i1 %217 to i64
  %220 = mul i64 %219, 32
  %221 = icmp sgt i64 %220, 0
  %222 = select i1 %221, i64 %220, i64 0
  %223 = mul i64 %218, 32
  %224 = icmp sgt i64 %223, 0
  %225 = select i1 %224, i64 %223, i64 0
  %226 = icmp slt i64 %197, %225
  %227 = select i1 %226, i64 %197, i64 %225
  %228 = icmp slt i64 %227, %222
  %229 = select i1 %228, i64 %227, i64 %222
  %230 = icmp slt i64 %229, 32
  br i1 %230, label %231, label %232

231:                                              ; preds = %204
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %85, ptr addrspace(6) %86, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %232

232:                                              ; preds = %231, %204
  %233 = mul i64 %229, 128
  %234 = mul i64 %229, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %215, i64 %233, i64 1, ptr addrspace(6) %85, ptr addrspace(6) %86, i64 0, i64 %234, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %85, ptr addrspace(6) %86, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %235 = add i64 %31, %207
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %235, i64 128, i64 1, ptr addrspace(6) %81, ptr addrspace(6) %82, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %81, ptr addrspace(6) %82, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %160)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %97, ptr addrspace(6) %98, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %97, ptr addrspace(6) %98, i64 0, i64 4096, i64 1, ptr addrspace(6) %101, ptr addrspace(6) %102, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %101, ptr addrspace(6) %102, i64 0, i64 4096, i64 1, ptr addrspace(6) %105, ptr addrspace(6) %106, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %105, ptr addrspace(6) %106, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %236 = mul i32 %167, %7
  %237 = add i32 %236, %163
  %238 = mul i32 %237, %8
  %239 = sext i32 %238 to i64
  %240 = mul i64 %239, 128
  %241 = mul i64 %173, 128
  %242 = add i64 %240, %241
  %243 = icmp sge i32 %167, 0
  %244 = icmp slt i32 %167, %10
  %245 = zext i1 %243 to i64
  %246 = zext i1 %244 to i64
  %247 = mul i64 %246, 32
  %248 = icmp sgt i64 %247, 0
  %249 = select i1 %248, i64 %247, i64 0
  %250 = mul i64 %245, 32
  %251 = icmp sgt i64 %250, 0
  %252 = select i1 %251, i64 %250, i64 0
  %253 = icmp slt i64 %197, %252
  %254 = select i1 %253, i64 %197, i64 %252
  %255 = icmp slt i64 %254, %249
  %256 = select i1 %255, i64 %254, i64 %249
  %257 = icmp slt i64 %256, 32
  br i1 %257, label %258, label %259

258:                                              ; preds = %232
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %77, ptr addrspace(6) %78, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %259

259:                                              ; preds = %258, %232
  %260 = mul i64 %256, 128
  %261 = mul i64 %256, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %242, i64 %260, i64 1, ptr addrspace(6) %77, ptr addrspace(6) %78, i64 0, i64 %261, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %77, ptr addrspace(6) %78, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %262 = add i64 %32, %207
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %262, i64 128, i64 1, ptr addrspace(6) %73, ptr addrspace(6) %74, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %73, ptr addrspace(6) %74, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %109, ptr addrspace(6) %110, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %109, ptr addrspace(6) %110, i64 0, i64 4096, i64 1, ptr addrspace(6) %113, ptr addrspace(6) %114, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %113, ptr addrspace(6) %114, i64 0, i64 4096, i64 1, ptr addrspace(6) %117, ptr addrspace(6) %118, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %117, ptr addrspace(6) %118, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %263 = add i32 %167, 1
  %264 = mul i32 %263, %7
  %265 = add i32 %264, %163
  %266 = mul i32 %265, %8
  %267 = sext i32 %266 to i64
  %268 = mul i64 %267, 128
  %269 = mul i64 %173, 128
  %270 = add i64 %268, %269
  %271 = icmp sge i32 %263, 0
  %272 = icmp slt i32 %263, %10
  %273 = zext i1 %271 to i64
  %274 = zext i1 %272 to i64
  %275 = mul i64 %274, 32
  %276 = icmp sgt i64 %275, 0
  %277 = select i1 %276, i64 %275, i64 0
  %278 = mul i64 %273, 32
  %279 = icmp sgt i64 %278, 0
  %280 = select i1 %279, i64 %278, i64 0
  %281 = icmp slt i64 %197, %280
  %282 = select i1 %281, i64 %197, i64 %280
  %283 = icmp slt i64 %282, %277
  %284 = select i1 %283, i64 %282, i64 %277
  %285 = icmp slt i64 %284, 32
  br i1 %285, label %286, label %287

286:                                              ; preds = %259
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %69, ptr addrspace(6) %70, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %287

287:                                              ; preds = %286, %259
  %288 = mul i64 %284, 128
  %289 = mul i64 %284, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %270, i64 %288, i64 1, ptr addrspace(6) %69, ptr addrspace(6) %70, i64 0, i64 %289, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %69, ptr addrspace(6) %70, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %290 = add i64 %33, %207
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %290, i64 128, i64 1, ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %121, ptr addrspace(6) %122, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %121, ptr addrspace(6) %122, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  %291 = add i32 %167, 2
  %292 = mul i32 %291, %7
  %293 = add i32 %292, %163
  %294 = mul i32 %293, %8
  %295 = sext i32 %294 to i64
  %296 = mul i64 %295, 128
  %297 = mul i64 %173, 128
  %298 = add i64 %296, %297
  %299 = icmp sge i32 %291, 0
  %300 = icmp slt i32 %291, %10
  %301 = zext i1 %299 to i64
  %302 = zext i1 %300 to i64
  %303 = mul i64 %302, 32
  %304 = icmp sgt i64 %303, 0
  %305 = select i1 %304, i64 %303, i64 0
  %306 = mul i64 %301, 32
  %307 = icmp sgt i64 %306, 0
  %308 = select i1 %307, i64 %306, i64 0
  %309 = icmp slt i64 %197, %308
  %310 = select i1 %309, i64 %197, i64 %308
  %311 = icmp slt i64 %310, %305
  %312 = select i1 %311, i64 %310, i64 %305
  %313 = icmp slt i64 %312, 32
  br i1 %313, label %314, label %315

314:                                              ; preds = %287
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 4096, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %315

315:                                              ; preds = %314, %287
  %316 = mul i64 %312, 128
  %317 = mul i64 %312, 128
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %298, i64 %316, i64 1, ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 %317, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %157)
  %318 = add i64 %34, %207
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %155)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %318, i64 128, i64 1, ptr addrspace(6) %57, ptr addrspace(6) %58, i64 0, i64 128, i64 1, i32 0, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %57, ptr addrspace(6) %58, i64 0, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 128, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %155)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 182912 to ptr addrspace(6)), i64 0, i64 1, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %125, ptr addrspace(6) %126, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %125, ptr addrspace(6) %126, i64 0, i64 4096, i64 1, ptr addrspace(6) %129, ptr addrspace(6) %130, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %129, ptr addrspace(6) %130, i64 0, i64 4096, i64 1, ptr addrspace(6) %133, ptr addrspace(6) %134, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %133, ptr addrspace(6) %134, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmuls_vs_1d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float -1.000000e+00, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float 0.000000e+00, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, float 1.000000e+00, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_1d_float(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 32767, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %319 = mul i32 %161, %7
  %320 = add i32 %319, %163
  %321 = mul i32 %320, %8
  %322 = sext i32 %321 to i64
  %323 = add i64 %322, %173
  %324 = icmp slt i64 %185, 32
  br i1 %324, label %325, label %326

325:                                              ; preds = %315
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) %53, ptr addrspace(6) %54, i64 0, i64 32, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %326

326:                                              ; preds = %325, %315
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %153)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %323, i64 %185, i64 1, ptr addrspace(6) %53, ptr addrspace(6) %54, i64 0, i64 %185, i64 1, i32 2, bfloat 0xR0000, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vcast_bfloat16_t_to_float_1d_with_mode(ptr addrspace(6) %53, ptr addrspace(6) %54, i64 0, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %153)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 32767, ptr addrspace(6) %137, ptr addrspace(6) %138, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vshrs_vs_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, i32 16, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 117376 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_int32_t(ptr addrspace(6) %137, ptr addrspace(6) %138, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 166528 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %141, ptr addrspace(6) %142, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vand_1d_int32_t(ptr addrspace(6) %141, ptr addrspace(6) %142, i64 0, i64 4096, i64 1, ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 133760 to ptr addrspace(6)), i64 0, i64 4096, i64 1, ptr addrspace(6) %145, ptr addrspace(6) %146, i64 0, i64 4096, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_2d_float(ptr addrspace(6) %145, ptr addrspace(6) %146, i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150144 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 1, i64 1, ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 32, i64 128, i64 128, i64 1, ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 150272 to ptr addrspace(6)), i64 0, i64 256, i64 1)
  %327 = mul i64 %322, 128
  %328 = mul i64 %173, 128
  %329 = add i64 %327, %328
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %149, ptr addrspace(6) %150, i64 0, i64 4096, i64 1, ptr addrspace(6) %49, ptr addrspace(6) %50, i64 0, i64 4096, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %330 = mul i64 %185, 128
  %331 = mul i64 %185, 128
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.BARRIER(i64 5)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %49, ptr addrspace(6) %50, i64 0, i64 %330, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %329, i64 %331, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %160)
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

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #4

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #3

attributes #0 = { alwaysinline }
attributes #1 = { alwaysinline }
attributes #2 = { nounwind readnone  }
attributes #3 = { nounwind }
attributes #4 = { nounwind  inaccessiblememonly }

