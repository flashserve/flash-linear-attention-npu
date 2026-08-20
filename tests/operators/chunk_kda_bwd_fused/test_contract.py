from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
A_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd_a"
C_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd_c"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _all_source(root: Path) -> str:
    return "\n".join(
        _read(path) for path in root.rglob("*")
        if path.suffix in {".h", ".cpp"}
    )


def test_each_fused_kernel_has_one_host_launch_and_one_device_entry():
    for name, root in (("a", A_ROOT), ("c", C_ROOT)):
        host = _read(root / f"op_host/op_api/chunk_kda_bwd_{name}.cpp")
        device = _read(root / f"op_kernel/chunk_kda_bwd_{name}.cpp")
        assert host.count("ADD_TO_LAUNCHER_LIST_AICORE") == 2  # call + error text
        assert host.count("const auto ret = ADD_TO_LAUNCHER_LIST_AICORE") == 1
        assert device.count('extern "C" __global__ __aicore__ void') == 1


def test_no_block_mmad_in_a_or_c():
    assert "BlockMmad" not in _all_source(A_ROOT)
    assert "BlockMmad" not in _all_source(C_ROOT)


def test_kernel_c_is_one_deep_fused_device_chain():
    entry = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c.cpp")
    wy_aic = entry.index("ChunkKdaBwdCCubeProcess")
    intra_aic = entry.index("ChunkKdaBwdCIntraCubeProcess")
    wy_aiv = entry.index("ChunkKdaBwdCVectorProcess")
    intra_aiv = entry.index("ChunkKdaBwdCIntraVectorProcess")
    gate_aiv = entry.index("ChunkKdaBwdCGateProcess")
    assert wy_aic < intra_aic < wy_aiv < intra_aiv < gate_aiv


def test_kernel_c_owner_grid_and_value_dimension_contract():
    common = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_common.h")
    tiling = _read(C_ROOT / "op_host/chunk_kda_bwd_c_tiling_processor.h")
    entry = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c.cpp")
    assert "kWyFusedHeadsPerWindow = 2" in common
    assert "KDA_C_SLOT_COUNT = 4" in tiling
    assert "headWindows % blockDim_" in tiling
    assert "tiling_.valueDim != 128" in tiling
    assert "tiling_.valueDim != 256" in tiling
    assert "tiling_.valueDim == 256 ? 4U : 0U" in tiling
    for key in range(1, 9):
        assert f"TILING_KEY_IS({key})" in entry


def test_kernel_c_preserves_inplace_dv_scan_until_its_last_read():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_vector.h")
    cube = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_cube.h")
    s2_ready = vector.index("CrossCoreWaitFlag(kS2Ready)")
    s3_ready = vector.index("CrossCoreWaitFlag(kS3aReady)", s2_ready)
    dv_write = vector.index("FinishBaseStage", s3_ready)
    assert s2_ready < s3_ready < dv_write
    assert "(slot + tiling_.zaOutputOffset) / sizeof(DataT)" in cube
    assert "PersistTza" not in _all_source(C_ROOT / "op_kernel")


def test_saved_state_layout_matches_forward_export_and_kernel_b():
    a_common = _read(A_ROOT / "op_kernel/chunk_kda_bwd_a_common.h")
    c_common = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_common.h")
    a_tiling = _read(A_ROOT / "op_host/chunk_kda_bwd_a_tiling_processor.h")
    c_tiling = _read(C_ROOT / "op_host/chunk_kda_bwd_c_tiling_processor.h")
    assert "taskIdx) * tiling.headNum + head" in a_common
    assert "chunkIdx) * tiling.headNum + headIdx" in c_common
    assert "[B,chunkNum,H,K,V]" in a_tiling
    assert "[B,chunkNum,H,K,V]" in c_tiling


def test_varlen_metadata_stays_inside_the_single_device_launch():
    for root, prefix in ((A_ROOT, "KdaBwdA"), (C_ROOT, "Wy")):
        common = _all_source(root / "op_kernel")
        assert "cuSeqlens" in common
        assert "chunkIndices" in common
        assert "GetValue" in common
    for name, root in (("a", A_ROOT), ("c", C_ROOT)):
        host = _read(root / f"op_host/op_api/chunk_kda_bwd_{name}.cpp")
        assert "ADD_TO_LAUNCHER_LIST_AICORE" in host
        assert "cuSeqlens" in host
        assert "chunkIndices" in host


def test_kernel_a_daqk_fixpipe_writes_directly_to_final_gm():
    cube = _read(A_ROOT / "op_kernel/chunk_kda_bwd_a_cube.h")
    assert "reinterpret_cast<__gm__ float *>(dAqk_) + outOffset" in cube
    assert "RunDAqk(task, head);" in cube


def test_kernel_a_is_pure_aic_and_leaves_q0_inside_kernel_b():
    entry = _read(A_ROOT / "op_kernel/chunk_kda_bwd_a.cpp")
    op_def = _read(A_ROOT / "op_host/chunk_kda_bwd_a_def.cpp")
    assert "KERNEL_TYPE_AIC_ONLY" in entry
    assert "ASCEND_IS_AIV" not in entry
    assert 'Input("qg")' not in op_def
    assert 'Output("Q0")' not in op_def


def test_three_kernel_wrapper_preserves_launch_order_and_pr291_dv_contract():
    wrapper = _read(A_ROOT / "op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    launch_a = wrapper.index("l0op::ChunkKdaBwdA(")
    launch_b = wrapper.index("l0op::ChunkGatedDeltaRuleBwdDhu(")
    launch_c = wrapper.index("l0op::ChunkKdaBwdC(")
    assert launch_a < launch_b < launch_c
    b_call = wrapper[launch_b:launch_c]
    assert "dOB, dv0B" in b_call
    assert "dh, nullptr, dvScanB" in b_call
    assert "h, dh, dvScan" in wrapper[launch_c:]
    assert "const aclTensor *dv0" in wrapper
    assert "const aclTensor *dvScan" in wrapper


def test_three_kernel_wrapper_adapts_pr291_varlen_with_views_only():
    wrapper = _read(A_ROOT / "op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    assert "l0op::Reshape" in wrapper
    assert "qgB = AsRank4" in wrapper
    assert "dvScanB = AsRank4" in wrapper
    assert "never splits cu_seqlens" in wrapper
    assert "for (int64_t sequence" not in wrapper


def test_kernel_c_accepts_pr291_head_major_dh_without_transpose():
    common = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_common.h")
    tiling = _read(C_ROOT / "op_host/chunk_kda_bwd_c_tiling_processor.h")
    assert "tiling.dhHeadMajor != 0" in common
    assert "CheckDenseDhHeadMajor" in tiling
    assert "CheckVarlenDhHeadMajor" in tiling
    assert "Transpose" not in common


def test_kernel_c_lower_mmad_pads_fp32_k():
    cube = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_intra_cube.h")
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_intra_vector.h")
    assert "const uint32_t lowerK = (prefix + 15U) & ~15U" in cube
    assert "const uint32_t lowerK = (prefix + 15U) & ~15U" in vector
    assert "kProcessRowBlock, prefix, lowerK" in vector
    assert "static_cast<uint64_t>(rowBase + row) * lowerK" in vector
    assert "rows * cols" in vector.split(
        "Cube reduces over lowerK=align16(prefix)", 1)[1]


def test_kernel_c_a5_fp16_conversion_preserves_column_order():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_intra_vector.h")
    convert = vector.split("inline void ConvertToFp32", 1)[1].split(
        "template <typename T>", 1)[0]
    assert "IsSameType<SrcT, half>::value" in convert
    assert "AscendC::Cast(dst, src" in convert
    assert "KdaRegbaseCastBf16ToFp32" in convert


def test_kernel_c_varlen_upper_mmad_keeps_full_physical_m_tile():
    cube = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_intra_cube.h")
    assert "RunUpper(upper, slotBase, future, processRowBlock)" in cube
    assert "processRowBlock, tiling_.keyDim, reduction" in cube


def test_kernel_c_intra_waits_for_both_vector_subblocks():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_intra_vector.h")
    prepare = vector.split("PrepareHead(task", 1)[1].split("FinishHead(task", 1)[0]
    assert "CrossCoreBarrier<0x1, PIPE_MTE3>()" in prepare
    assert prepare.index("CrossCoreBarrier<0x1, PIPE_MTE3>()") < prepare.index(
        "CrossCoreSetFlag<0x2, PIPE_MTE3>")


def test_kernel_c_a5_gate_partitions_head_window_across_subblocks():
    gate = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_gate.h")
    assert gate.count("const uint32_t laneBegin = AscendC::GetSubBlockIdx();") == 2
    assert gate.count("laneBegin < headCount ? laneBegin + 1U : laneBegin") == 2


def test_kernel_c_a5_gate_uses_single_pass_register_reverse_scan():
    gate = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_gate.h")
    assert "KdaBwdCGateReverseScanA5" in gate
    scan = gate.split("KdaBwdCGateReverseScanA5(", 1)[1]
    scan = scan.split("template <bool SAFE_GATE", 1)[0]
    assert "for (uint32_t row = rows; row > 0; --row)" in scan
    assert "Add(accumulator, accumulator, value, mask)" in scan


def test_kernel_c_a5_fuses_safe_gate_backward_in_registers():
    gate = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_gate.h")
    assert "KdaBwdCSafeGateBackwardA5" in gate
    fused = gate.split("KdaBwdCSafeGateBackwardA5(", 1)[1]
    fused = fused.split("template <bool SAFE_GATE", 1)[0]
    for operation in (
        "Exp(", "Div(", "Mul(", "Add(dbReg", "ReduceSum(dABlock",
        "Add(upstreamAcc", "DataCopy(dg", "DataCopy(dbAcc",
        "StoreDist::DIST_FIRST_ELEMENT_B32",
    ):
        assert operation in fused
    a5_path = gate.split("if constexpr (SAFE_GATE)", 1)[1]
    assert "KdaBwdCSafeGateBackwardA5<true>" in a5_path
    assert "KdaBwdCSafeGateBackwardA5<false>" in a5_path
    assert "if (!applyRaw || !SAFE_GATE)" in gate


def test_kernel_c_wy_preserves_vector_raw_dependencies():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_vector.h")
    dq_stage = vector.split("if (stage == 0)", 1)[1].split(
        "else if (stage == 1)", 1)[0]
    assert "Mul(z, x, y" in dq_stage
    assert dq_stage.index("Mul(z, x, y") < dq_stage.index(
        "PipeBarrier<PIPE_V>()") < dq_stage.index("Muls(z, z")
    gate_stage = vector.split("// gate_qk + gate_w", 1)[1].split(
        "// tmp still contains", 1)[0]
    assert "Mul(acc, e, acc" in gate_stage
    assert gate_stage.index("Mul(acc, e, acc") < gate_stage.index(
        "PipeBarrier<PIPE_V>()") < gate_stage.index("Sub(qk, qk, acc")


def test_kernel_c_a5_reuses_resident_gate_for_ke_and_dq():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_vector.h")
    assert "BuildKE<true>" in vector
    build_ke = vector.split("__aicore__ inline void BuildKE(", 1)[1]
    build_ke = build_ke.split(
        "__aicore__ inline void FinishBaseStage(", 1)[0]
    assert "e still contains exp2(gk)" in build_ke
    assert "dqRawGm_" in build_ke
    assert "dqGm_" in build_ke


def test_kernel_c_a5_reduces_state_gate_dot_products_in_registers():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_vector.h")
    assert "KdaBwdCRowDotAccA5" in vector
    helper = vector.split("KdaBwdCRowDotAccA5(", 1)[1]
    helper = helper.split("template <typename DataT", 1)[0]
    for operation in (
        "Mul(product", "Add(acc", "ReduceSum(sum",
        "StoreDist::DIST_FIRST_ELEMENT_B32",
    ):
        assert operation in helper
    state_gate = vector.split("__aicore__ inline void PrepareStateGate", 1)[1]
    state_gate = state_gate.split("__aicore__ inline void AddPreparedStateGate", 1)[0]
    assert "KdaBwdCRowDotAccA5(" in state_gate
    # Definition + state-gate h*dh. Token db is now fused into the dedicated
    # dkgb product/subtract helper below instead of calling this helper.
    assert vector.count("KdaBwdCRowDotAccA5(") == 2


def test_kernel_c_a5_fuses_dkgb_product_and_db_reduction():
    vector = _read(C_ROOT / "op_kernel/chunk_kda_bwd_c_vector.h")
    assert "KdaBwdCMulRowDotSubA5" in vector
    helper = vector.split("KdaBwdCMulRowDotSubA5(", 1)[1]
    helper = helper.split("template <typename DataT", 1)[0]
    for operation in (
        "Mul(product", "DataCopy(productDst", "ReduceSum(sum",
        "Sub(current", "StoreDist::DIST_FIRST_ELEMENT_B32",
    ):
        assert operation in helper
    assert vector.count("KdaBwdCMulRowDotSubA5(") == 2


def test_source_files_have_balanced_braces():
    # This deliberately simple guard catches accidental truncation while
    # porting the large fused headers before a CANN compiler is available.
    for root in (A_ROOT, C_ROOT):
        for path in root.rglob("*"):
            if path.suffix in {".h", ".cpp"}:
                source = _read(path)
                assert source.count("{") == source.count("}"), path
