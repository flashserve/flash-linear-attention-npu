"""Static and host-only contract tests for the Triton-Ascend KDA adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
ADAPTER_PATH = (
    ROOT
    / "torch_custom/fla_npu/fla_npu/adapters/triton_ascend_kda.py"
)
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd.json"


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "fla_npu_triton_ascend_kda_adapter",
        ADAPTER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compatible_original(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    transpose_state_layout=False,
):
    del (
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        disable_recompute,
        return_intermediate_states,
        transpose_state_layout,
    )


def _original_l2norm(x, eps=1e-6, output_dtype=None):
    return x, (eps, output_dtype)


def _optimized_l2norm(x, eps=1e-6, output_dtype=None):
    return x, (eps, output_dtype)


def _adapter_target_modules(adapter):
    modules = {
        name: types.SimpleNamespace(chunk_kda_fwd=_compatible_original)
        for name in adapter._TARGET_MODULES
    }
    modules[adapter._L2NORM_TARGET_MODULE].l2norm_fwd = _original_l2norm
    return modules


def test_adapter_patches_and_restores_both_bound_forward_symbols(monkeypatch):
    adapter = _load_adapter()
    modules = _adapter_target_modules(adapter)
    adapter._load_ascendc_ops = lambda: None
    adapter._load_optimized_l2norm_fwd = lambda: _optimized_l2norm
    adapter._install_triton_extra_ascend_compat = lambda: False
    monkeypatch.setattr(adapter.importlib, "import_module", modules.__getitem__)

    assert adapter.install_triton_ascend_kda_adapter() is True
    assert adapter.install_triton_ascend_kda_adapter() is False
    assert adapter.is_triton_ascend_kda_adapter_installed()
    assert all(
        module.chunk_kda_fwd is adapter.triton_ascend_chunk_kda_fwd
        for module in modules.values()
    )
    assert (
        modules[adapter._L2NORM_TARGET_MODULE].l2norm_fwd
        is _optimized_l2norm
    )

    assert adapter.remove_triton_ascend_kda_adapter() is True
    assert adapter.remove_triton_ascend_kda_adapter() is False
    assert not adapter.is_triton_ascend_kda_adapter_installed()
    assert all(
        module.chunk_kda_fwd is _compatible_original
        for module in modules.values()
    )
    assert (
        modules[adapter._L2NORM_TARGET_MODULE].l2norm_fwd is _original_l2norm
    )


def test_adapter_signature_failure_does_not_leave_partial_install_state(monkeypatch):
    adapter = _load_adapter()

    def incompatible_original(q):
        del q

    modules = {
        adapter._TARGET_MODULES[0]: types.SimpleNamespace(
            chunk_kda_fwd=_compatible_original
        ),
        adapter._TARGET_MODULES[1]: types.SimpleNamespace(
            chunk_kda_fwd=incompatible_original
        ),
    }
    modules[adapter._L2NORM_TARGET_MODULE].l2norm_fwd = _original_l2norm
    adapter._load_ascendc_ops = lambda: None
    adapter._load_optimized_l2norm_fwd = lambda: _optimized_l2norm
    adapter._install_triton_extra_ascend_compat = lambda: False
    monkeypatch.setattr(adapter.importlib, "import_module", modules.__getitem__)

    try:
        adapter.install_triton_ascend_kda_adapter()
    except RuntimeError as exc:
        assert "missing parameters" in str(exc)
    else:
        raise AssertionError("incompatible upstream signature must be rejected")

    assert not adapter.is_triton_ascend_kda_adapter_installed()
    assert modules[adapter._TARGET_MODULES[0]].chunk_kda_fwd is _compatible_original
    assert modules[adapter._TARGET_MODULES[1]].chunk_kda_fwd is incompatible_original


def test_adapter_registers_packaged_opp_before_importing_triton(monkeypatch):
    adapter = _load_adapter()
    events = []
    modules = _adapter_target_modules(adapter)

    def load_ascendc_ops():
        events.append("ascendc")
        return None

    def import_module(name):
        events.append(name)
        return modules[name]

    adapter._load_ascendc_ops = load_ascendc_ops
    adapter._load_optimized_l2norm_fwd = lambda: events.append(
        "optimized_l2norm"
    ) or _optimized_l2norm
    adapter._install_triton_extra_ascend_compat = lambda: events.append(
        "triton_compat"
    )
    monkeypatch.setattr(adapter.importlib, "import_module", import_module)

    assert adapter.install_triton_ascend_kda_adapter() is True
    assert events == [
        "ascendc",
        "triton_compat",
        *adapter._TARGET_MODULES,
        "optimized_l2norm",
    ]


def test_adapter_bridges_pinned_upstream_extra_ascend_eager_import(monkeypatch):
    adapter = _load_adapter()
    extra = types.SimpleNamespace()
    cann = types.SimpleNamespace()
    libdevice = types.SimpleNamespace()
    modules = {
        "triton.language.extra": extra,
        "triton.language.extra.cann": cann,
        "triton.language.extra.cann.libdevice": libdevice,
    }

    def fake_import(name):
        if name == "triton.language.extra.ascend.libdevice":
            raise ModuleNotFoundError(
                "No module named 'triton.language.extra.ascend'",
                name="triton.language.extra.ascend",
            )
        return modules[name]

    monkeypatch.setattr(adapter.importlib, "import_module", fake_import)
    sys.modules.pop("triton.language.extra.ascend", None)
    sys.modules.pop("triton.language.extra.ascend.libdevice", None)
    try:
        assert adapter._install_triton_extra_ascend_compat() is True
        assert extra.ascend is cann
        assert sys.modules["triton.language.extra.ascend"] is cann
        assert (
            sys.modules["triton.language.extra.ascend.libdevice"] is libdevice
        )
    finally:
        sys.modules.pop("triton.language.extra.ascend", None)
        sys.modules.pop("triton.language.extra.ascend.libdevice", None)


def test_adapter_does_not_patch_reverse_cumsum_or_backward_modules():
    adapter = _load_adapter()
    assert adapter._TARGET_MODULES == (
        "triton_ascend_kernels.attention.fla.kda.chunk_fwd",
        "triton_ascend_kernels.attention.fla.kda.chunk",
    )
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "cumsum_kda.chunk_local_cumsum" not in source
    assert "chunk_bwd.chunk_kda_bwd" not in source
    assert adapter._L2NORM_TARGET_MODULE == adapter._TARGET_MODULES[1]


def test_adapter_forwards_gate_parameter_dtypes_without_conversion():
    adapter = _load_adapter()
    bf16 = object()
    fp32 = object()
    calls = {}

    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype
            self.shape = (1, 1, 1, 1)

        def dim(self):
            return 4

        def float(self):
            raise AssertionError("gate parameters must not be promoted in the adapter")

    def fake_chunk_kda_fwd(*args, **kwargs):
        calls.update(kwargs)
        return (None,) * 12

    q = FakeTensor(fp32)
    a_log = FakeTensor(bf16)
    a_log_fp32 = FakeTensor(fp32)
    dt_bias_bf16 = FakeTensor(bf16)
    dt_bias = FakeTensor(fp32)
    observed = []
    adapter._load_ascendc_ops = lambda: fake_chunk_kda_fwd
    for a_log_input, dt_bias_input in (
        (a_log, dt_bias),
        (a_log_fp32, dt_bias_bf16),
        (a_log, dt_bias_bf16),
    ):
        calls.clear()
        adapter.triton_ascend_chunk_kda_fwd(
            q, q, q, q, q, 1.0, None, False,
            use_gate_in_kernel=True,
            A_log=a_log_input,
            dt_bias=dt_bias_input,
        )
        observed.append((calls["A_log"], calls["dt_bias"]))

    assert observed == [
        (a_log, dt_bias),
        (a_log_fp32, dt_bias_bf16),
        (a_log, dt_bias_bf16),
    ]


def test_model_backward_h96_case_is_pinned_in_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["coverage_requirements"]["model_adapter_case_ids"] == [
        "chunk_kda_model_bwd_h96"
    ]
    case = next(
        item
        for item in manifest["cases"]
        if item["id"] == "chunk_kda_model_bwd_h96"
    )
    assert case["reference"].endswith(
        "4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8"
    )
    assert case["shape"] == {
        "B": 1,
        "H_k": 96,
        "H_v": 96,
        "T": 18432,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "N_c": 288,
    }
    assert case["expect"]["backward_input_contract"]["dAqk"] == [
        [1, 18432, 96, 64],
        "float32",
    ]
    assert case["expect"]["backward_input_contract"]["dAkk"] == [
        [1, 18432, 96, 64],
        "float32",
    ]
