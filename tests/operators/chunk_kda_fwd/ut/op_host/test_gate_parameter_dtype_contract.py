"""Static dtype contract for ChunkKdaFwd raw-gate parameters."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"


def _initializer_body(source: str, kind: str, name: str) -> str:
    marker = f"const std::initializer_list<ge::{kind}> {name} = {{"
    return source.split(marker, 1)[1].split("};", 1)[0]


def test_op_def_registers_full_gate_parameter_dtype_cartesian_product():
    source = (OP_ROOT / "op_host/chunk_kda_fwd_def.cpp").read_text(
        encoding="utf-8"
    )
    for name in (
        "dataTypes",
        "gateTypes",
        "betaTypes",
        "aLogTypes",
        "dtBiasTypes",
        "stateTypes",
    ):
        assert _initializer_body(source, "DataType", name).count("ge::DT_") == 32

    a_log = _initializer_body(source, "DataType", "aLogTypes")
    dt_bias = _initializer_body(source, "DataType", "dtBiasTypes")
    assert a_log.count("ge::DT_FLOAT") == a_log.count("ge::DT_BF16") == 16
    assert dt_bias.count("ge::DT_FLOAT") == dt_bias.count("ge::DT_BF16") == 16
    assert 'Input("initial_state").ParamType(OPTIONAL).DataType(stateTypes)' in source
    assert 'Output("final_state").ParamType(OPTIONAL).DataType(stateTypes)' in source
    assert 'Output("gk").ParamType(OPTIONAL).DataType(stateTypes)' in source


def test_host_wrappers_accept_only_fp32_or_bf16_gate_parameters():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    ctypes_wrapper = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    legacy = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")

    assert aclnn.count(
        "DataType::DT_FLOAT || aLogType == DataType::DT_BF16"
    ) == 1
    assert aclnn.count(
        "DataType::DT_FLOAT || dtBiasType == DataType::DT_BF16"
    ) == 1
    assert "gate_param_dtypes = {torch.float32, torch.bfloat16}" in ctypes_wrapper
    assert legacy.count("A_log->scalar_type() == at::kBFloat16") == 1
    assert legacy.count("dt_bias->scalar_type() == at::kBFloat16") >= 1


def test_kernel_runtime_dispatches_optional_gate_dtypes_only_in_gate_stage():
    entry = (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.h").read_text(
        encoding="utf-8"
    )
    prepare = (
        OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_prepare.h"
    ).read_text(encoding="utf-8")
    gate = (
        ROOT
        / "fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
    ).read_text(encoding="utf-8")

    assert "DispatchGateMode" in entry
    assert "DTYPE_G" in entry
    assert '"g dtype must be FP32 or BF16"' in entry
    assert "DTYPE_A_LOG" not in entry
    assert "DTYPE_DT_BIAS" not in entry
    assert "KDA_PARAM_DTYPE_BF16 = 1" in entry
    assert "tiling.aLogDataType == KDA_PARAM_DTYPE_BF16" in entry
    assert "tiling.dtBiasDataType == KDA_PARAM_DTYPE_BF16" in entry
    assert "aLogDataType" in tiling and "dtBiasDataType" in tiling
    assert "RunGateCumsum<false, false, G_T, float, float>" in entry
    for gate_types in (
        "G_T, float, float",
        "G_T, float, bfloat16_t",
        "G_T, bfloat16_t, float",
        "G_T, bfloat16_t, bfloat16_t",
    ):
        assert f"RunGateCumsum<true, SAFE_GATE, {gate_types}>" in entry
    gate_dispatch = entry.split(
        "__aicore__ inline void DispatchGateMode", 1
    )[1].split(
        "template <bool SAFE_GATE, typename T, typename BETA_T", 1
    )[0]
    assert gate_dispatch.index("if (tiling.computeGateInPrepare)") < gate_dispatch.index(
        "tiling.aLogDataType"
    )
    assert gate_dispatch.index("if (!tiling.useGateInKernel)") < gate_dispatch.index(
        "tiling.aLogDataType"
    )
    assert "RunPrepareStage<SAFE_GATE, T, float, BETA_T, float, float" in entry
    assert "GlobalTensor<A_LOG_T> aLog_" in prepare
    assert "GlobalTensor<DT_BIAS_T> dtBias_" in prepare
    assert "LoadAsFloatRow(aLog_" in prepare
    assert "LoadAsFloatRow(dtBias_" in prepare
    assert "ChunkKdaFwdPrepareKernel<USE_GATE_IN_KERNEL" not in prepare
    assert "RunGateCumsum<" in gate_dispatch
    assert "RunPrepareStage<" not in gate_dispatch
    assert "LoadGateParamAsFloat" in gate

    gate_init = gate.split("__aicore__ inline void Init(", 1)[1].split(
        "__aicore__ inline void Process()", 1
    )[0]
    assert gate_init.index("if constexpr (USE_GATE_IN_KERNEL)") < gate_init.index(
        "aLog_.SetGlobalBuffer"
    )
    assert gate_init.index("if constexpr (USE_GATE_IN_KERNEL)") < gate_init.index(
        "dtBias_.SetGlobalBuffer"
    )
    prepare_gate = gate.split("__aicore__ inline void PrepareGate(", 1)[1].split(
        "__aicore__ inline void ApplyGate(", 1
    )[0]
    assert "if constexpr (USE_GATE_IN_KERNEL)" in prepare_gate
    assert prepare_gate.index("if constexpr (USE_GATE_IN_KERNEL)") < prepare_gate.index(
        "ReadGateParam(aLog_"
    )
    assert prepare_gate.index("if constexpr (USE_GATE_IN_KERNEL)") < prepare_gate.index(
        "LoadGateParamAsFloat(bias, dtBias_"
    )


def test_adapter_does_not_create_hidden_fp32_gate_parameter_copies():
    adapter = (
        ROOT / "torch_custom/fla_npu/fla_npu/adapters/triton_ascend_kda.py"
    ).read_text(encoding="utf-8")
    assert "_gate_parameter_for_ascendc" not in adapter
    assert "A_log=A_log" in adapter
    assert "dt_bias=dt_bias" in adapter
