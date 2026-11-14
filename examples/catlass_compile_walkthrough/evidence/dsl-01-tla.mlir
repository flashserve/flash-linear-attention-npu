"builtin.module"() ({
  "tla.func"() <{function_type = (!tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "matmul_kernel"}> ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, %arg2: !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = "tla.flag"() <{dst_pipe = #tla.pipe<mte1>, id = -1 : i64, name = "input_ready", src_pipe = #tla.pipe<mte2>}> : () -> !tla.flag
    %1 = "tla.flag"() <{dst_pipe = #tla.pipe<cube>, id = -1 : i64, name = "operands_ready", src_pipe = #tla.pipe<mte1>}> : () -> !tla.flag
    %2 = "tla.flag"() <{dst_pipe = #tla.pipe<fix>, id = -1 : i64, name = "result_ready", src_pipe = #tla.pipe<cube>}> : () -> !tla.flag
    %3 = "tla.alloc_ptr"() <{size_bytes = 512 : i64}> : () -> !tla.ptr<f16, l1, 512>
    %4 = "tla.alloc_ptr"() <{size_bytes = 512 : i64}> : () -> !tla.ptr<f16, l1, 512>
    %5 = "tla.alloc_ptr"() <{size_bytes = 512 : i64}> : () -> !tla.ptr<f16, l0a, 512>
    %6 = "tla.alloc_ptr"() <{size_bytes = 512 : i64}> : () -> !tla.ptr<f16, l0b, 512>
    %7 = "tla.alloc_ptr"() <{size_bytes = 1024 : i64}> : () -> !tla.ptr<f32, l0c, 512>
    "tla.cube"() ({
      %8 = "tla.make_tensor_like"(%3, %arg0) <{layoutTag = #tla.layout_tag<zN>}> : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
      %9 = "tla.make_tensor_like"(%4, %arg1) <{layoutTag = #tla.layout_tag<zN>}> : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
      "tla.copy"(%8, %arg0) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> ()
      "tla.copy"(%9, %arg1) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> ()
      "tla.set_flag"(%0) : (!tla.flag) -> ()
      "tla.wait_flag"(%0) : (!tla.flag) -> ()
      %10 = "tla.make_tensor_like"(%5, %8) <{layoutTag = #tla.layout_tag<zN>}> : (!tla.ptr<f16, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>
      %11 = "tla.make_tensor_like"(%6, %9) <{layoutTag = #tla.layout_tag<nZ>}> : (!tla.ptr<f16, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>
      %12 = "tla.make_tensor_like"(%7, %arg2) <{layoutTag = #tla.layout_tag<L0Clayout>}> : (!tla.ptr<f32, l0c, 512>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>
      "tla.copy"(%10, %8) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>) -> ()
      "tla.copy"(%11, %9) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>) -> ()
      "tla.set_flag"(%1) : (!tla.flag) -> ()
      "tla.wait_flag"(%1) : (!tla.flag) -> ()
      %13 = "arith.constant"() <{value = true}> : () -> i1
      %14 = "arith.constant"() <{value = 0 : i64}> : () -> i64
      "tla.mmad"(%12, %10, %11, %13, %14) <{compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>}> : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, i1, i64) -> ()
      "tla.set_flag"(%2) : (!tla.flag) -> ()
      "tla.wait_flag"(%2) : (!tla.flag) -> ()
      %15 = "tla.CopyL0C2DstParams"() <{l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 0, 0, 0>, quant_mode = #tla.quant_mode<NO_QUANT>, relu_enable = false, unit_flag = 0 : i64}> : () -> !tla.copy_l0c2dst_params
      "tla.copy"(%arg2, %12, %15) : (!tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
