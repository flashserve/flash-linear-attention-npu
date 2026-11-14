; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: alwaysinline
define private void @mmad_half_half_float(ptr addrspace(3) %0, ptr addrspace(3) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(4) %5, ptr addrspace(4) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(5) %10, ptr addrspace(5) %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i1 %18, i8 %19) #0 {
  %21 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(3) %0, 0
  %22 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(3) %1, 1
  %23 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(4) %5, 0
  %28 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(4) %6, 1
  %29 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(5) %10, 0
  %34 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(5) %11, 1
  %35 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  call void @_mlir_ciface_mmad_half_half_float(ptr %26, ptr %32, ptr %38, i64 %15, i64 %16, i64 %17, i1 %18, i8 %19)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mmad_half_half_float(ptr, ptr, ptr, i64, i64, i64, i1, i8) #0

; Function Attrs: alwaysinline
define private void @copy_l0c_to_gm_RowMajor_float_float(ptr addrspace(5) %0, ptr addrspace(5) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i8 %36) #0 {
  %38 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(5) %0, 0
  %39 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %38, ptr addrspace(5) %1, 1
  %40 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %39, i64 %2, 2
  %41 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %40, i64 %3, 3, 0
  %42 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %41, i64 %4, 4, 0
  %43 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [1 x i64], [1 x i64] } %42, ptr %43, align 8
  %44 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %5, 0
  %45 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %44, ptr addrspace(1) %6, 1
  %46 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %45, i64 %7, 2
  %47 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %46, i64 %8, 3, 0
  %48 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %47, i64 %10, 4, 0
  %49 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %48, i64 %9, 3, 1
  %50 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %49, i64 %11, 4, 1
  %51 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %50, ptr %51, align 8
  call void @_mlir_ciface_copy_l0c_to_gm_RowMajor_float_float(ptr %43, ptr %51, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i8 %36)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_l0c_to_gm_RowMajor_float_float(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i8) #0

; Function Attrs: alwaysinline
define private void @copy_l1_zN_to_l0b_nZ_half(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(4) %5, ptr addrspace(4) %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33) #0 {
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(2) %0, 0
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %35, ptr addrspace(2) %1, 1
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %36, i64 %2, 2
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %37, i64 %3, 3, 0
  %39 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %38, i64 %4, 4, 0
  %40 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %39, ptr %40, align 8
  %41 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(4) %5, 0
  %42 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %41, ptr addrspace(4) %6, 1
  %43 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %42, i64 %7, 2
  %44 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %43, i64 %8, 3, 0
  %45 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %44, i64 %9, 4, 0
  %46 = alloca { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(4), ptr addrspace(4), i64, [1 x i64], [1 x i64] } %45, ptr %46, align 8
  call void @_mlir_ciface_copy_l1_zN_to_l0b_nZ_half(ptr %40, ptr %46, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_l1_zN_to_l0b_nZ_half(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #0

; Function Attrs: alwaysinline
define private void @copy_l1_zN_to_l0a_zN_half(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(3) %5, ptr addrspace(3) %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33) #0 {
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(2) %0, 0
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %35, ptr addrspace(2) %1, 1
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %36, i64 %2, 2
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %37, i64 %3, 3, 0
  %39 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %38, i64 %4, 4, 0
  %40 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %39, ptr %40, align 8
  %41 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(3) %5, 0
  %42 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %41, ptr addrspace(3) %6, 1
  %43 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %42, i64 %7, 2
  %44 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %43, i64 %8, 3, 0
  %45 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %44, i64 %9, 4, 0
  %46 = alloca { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(3), ptr addrspace(3), i64, [1 x i64], [1 x i64] } %45, ptr %46, align 8
  call void @_mlir_ciface_copy_l1_zN_to_l0a_zN_half(ptr %40, ptr %46, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_l1_zN_to_l0a_zN_half(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #0

; Function Attrs: alwaysinline
define private void @copy_gm_RowMajor_to_l1_zN_half(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(2) %7, ptr addrspace(2) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35) #0 {
  %37 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %0, 0
  %38 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %37, ptr addrspace(1) %1, 1
  %39 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %38, i64 %2, 2
  %40 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %39, i64 %3, 3, 0
  %41 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %40, i64 %5, 4, 0
  %42 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %41, i64 %4, 3, 1
  %43 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %42, i64 %6, 4, 1
  %44 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %43, ptr %44, align 8
  %45 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(2) %7, 0
  %46 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %45, ptr addrspace(2) %8, 1
  %47 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %46, i64 %9, 2
  %48 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %47, i64 %10, 3, 0
  %49 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %48, i64 %11, 4, 0
  %50 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [1 x i64], [1 x i64] } %49, ptr %50, align 8
  call void @_mlir_ciface_copy_gm_RowMajor_to_l1_zN_half(ptr %44, ptr %50, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_gm_RowMajor_to_l1_zN_half(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #0

define dso_local void @matmul_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) #1 {
  %4 = call i64 @llvm.hivm.GET.CTRL()
  %5 = call i64 @llvm.hivm.SBITSET0(i64 %4, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %5)
  %6 = call i64 @llvm.hivm.GET.CTRL()
  %7 = call i64 @llvm.hivm.SBITSET1(i64 %6, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %7)
  %8 = call i64 @llvm.hivm.GET.CTRL()
  %9 = call i64 @llvm.hivm.SBITSET0(i64 %8, i64 46)
  call void @llvm.hivm.SET.CTRL(i64 %9)
  %10 = call i64 @llvm.hivm.GET.CTRL()
  %11 = call i64 @llvm.hivm.SBITSET0(i64 %10, i64 47)
  call void @llvm.hivm.SET.CTRL(i64 %11)
  %12 = call i64 @llvm.hivm.GET.CTRL()
  %13 = call i64 @llvm.hivm.SBITSET0(i64 %12, i64 51)
  call void @llvm.hivm.SET.CTRL(i64 %13)
  call void @copy_gm_RowMajor_to_l1_zN_half(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 16, i64 16, i64 16, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 256, i64 1, i64 16, i64 16, i64 1, i64 1, i64 16, i64 1, i64 1, i64 1, i64 0, i64 0, i64 16, i64 16, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16)
  call void @copy_gm_RowMajor_to_l1_zN_half(ptr addrspace(1) %1, ptr addrspace(1) %1, i64 0, i64 16, i64 16, i64 16, i64 1, ptr addrspace(2) inttoptr (i64 512 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 512 to ptr addrspace(2)), i64 0, i64 256, i64 1, i64 16, i64 16, i64 1, i64 1, i64 16, i64 1, i64 1, i64 1, i64 0, i64 0, i64 16, i64 16, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 3, i64 0)
  call void @copy_l1_zN_to_l0a_zN_half(ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 256, i64 1, ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 256, i64 1, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16)
  call void @copy_l1_zN_to_l0b_nZ_half(ptr addrspace(2) inttoptr (i64 512 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 512 to ptr addrspace(2)), i64 0, i64 256, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 0, i64 256, i64 1, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16, i64 16, i64 1, i64 16, i64 1, i64 1, i64 256, i64 16, i64 256, i64 0, i64 0, i64 16, i64 16)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 2, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 2, i64 0)
  call void @mmad_half_half_float(ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 256, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 0, i64 256, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 256, i64 1, i64 16, i64 16, i64 16, i1 true, i8 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @copy_l0c_to_gm_RowMajor_float_float(ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 256, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %2, i64 0, i64 16, i64 16, i64 16, i64 1, i64 16, i64 1, i64 16, i64 1, i64 16, i64 256, i64 1, i64 256, i64 0, i64 0, i64 16, i64 16, i64 16, i64 16, i64 1, i64 1, i64 16, i64 1, i64 1, i64 1, i64 0, i64 0, i64 16, i64 16, i8 0)
  %14 = call i64 @llvm.hivm.GET.CTRL()
  %15 = call i64 @llvm.hivm.SBITSET0(i64 %14, i64 51)
  call void @llvm.hivm.SET.CTRL(i64 %15)
  %16 = call i64 @llvm.hivm.GET.CTRL()
  %17 = call i64 @llvm.hivm.SBITSET0(i64 %16, i64 46)
  call void @llvm.hivm.SET.CTRL(i64 %17)
  %18 = call i64 @llvm.hivm.GET.CTRL()
  %19 = call i64 @llvm.hivm.SBITSET0(i64 %18, i64 47)
  call void @llvm.hivm.SET.CTRL(i64 %19)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #2

; Function Attrs: nounwind readnone
declare i64 @llvm.hivm.SBITSET0(i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #2

; Function Attrs: nounwind readnone
declare i64 @llvm.hivm.SBITSET1(i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #2

attributes #0 = { alwaysinline }
attributes #1 = { "cube-only-kernel" "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { nounwind  inaccessiblememonly }
attributes #3 = { nounwind readnone  }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @matmul_kernel, !"kernel", i32 1}
