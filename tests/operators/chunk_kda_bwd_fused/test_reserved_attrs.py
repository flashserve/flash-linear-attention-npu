from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd"


def _read(relative: str) -> str:
    return (OP_ROOT / relative).read_text(encoding="utf-8")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def test_reserved_attrs_default_to_current_implementation():
    op_def = _read("op_host/chunk_kda_bwd_def.cpp")
    assert 'Attr("disable_recompute").AttrType(OPTIONAL).Bool(true)' in op_def
    assert 'Attr("use_exp2").AttrType(OPTIONAL).Bool(true)' in op_def
    assert 'Attr("state_v_first").AttrType(OPTIONAL).Bool(false)' in op_def
    assert 'Attr("defer_gate_post").AttrType(OPTIONAL).Bool(false)' in op_def


def test_aclnn_and_l0_keep_the_same_reserved_attr_order():
    aclnn_header = _read("op_host/op_api/aclnn_chunk_kda_bwd.h")
    l0_header = _read("op_host/op_api/chunk_kda_bwd.h")
    l0_source = _read("op_host/op_api/chunk_kda_bwd.cpp")
    signature = "bool disableRecompute, bool useExp2, bool stateVFirst"
    assert signature in _normalize_whitespace(aclnn_header)
    assert signature in _normalize_whitespace(l0_header)
    assert (
        "OP_ATTR(scale, chunkSize, safeGate, false, lowerBound, "
        "disableRecompute, useExp2, stateVFirst, useGateInKernel)"
    ) in _normalize_whitespace(l0_source)


def test_false_reserved_modes_fail_before_launch():
    aclnn = _read("op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    tiling = _read("op_host/chunk_kda_bwd_tiling.cpp")
    for attr, message in (
        ("disableRecompute", "disable_recompute=false is reserved but not supported"),
        ("useExp2", "use_exp2=false is reserved but not supported"),
    ):
        assert f"CHECK_COND({attr}" in aclnn
        assert message in aclnn
        assert f"!*{attr}" in tiling
        assert message in tiling

    assert "CHECK_COND(!stateVFirst" in aclnn
    assert "state_v_first=true is reserved but not supported" in aclnn
    assert "*stateVFirst" in tiling
    assert "state_v_first=true is reserved but not supported" in tiling


def test_forward_saved_intermediates_are_conditionally_optional():
    op_def = _read("op_host/chunk_kda_bwd_def.cpp")
    aclnn = _read("op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    for name in ("w", "qg", "kg", "v_new", "h"):
        assert f'Input("{name}").ParamType(OPTIONAL)' in op_def
    assert (
        "w, qg, kg, v_new and h are required when "
        "disable_recompute=true"
    ) in aclnn


def test_gate_modes_have_explicit_contracts():
    aclnn = _read("op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    l0 = _read("op_host/op_api/chunk_kda_bwd.cpp")
    # Rejecting the true branch outright would use the condition as the
    # complete CHECK_COND predicate. Input-contract checks may still begin
    # with `!useGateInKernel || ...` and are required.
    assert "CHECK_COND(!useGateInKernel, ACLNN_ERR" not in aclnn
    assert "OP_TYPE_REGISTER(KdaGateBwdPost)" in l0
    assert "OP_ATTR(scale, chunkSize, safeGate, false, lowerBound" in l0
    assert "if (useGatePost)" in l0
    assert "KdaGateBwdPost" in l0
    assert "raw_g, a_log and dA are required for raw-gate backward" in aclnn
    assert (
        "raw_g, a_log, dt_bias, dA and dbias require "
        "use_gate_in_kernel=true"
    ) in aclnn
    assert "dt_bias is required when dbias output is requested" in aclnn
    assert "dbias output is required when dt_bias is present" in aclnn
