"""Static op_host contract for chunk_kda_fwd; device execution lives in accuracy/routes."""

from pathlib import Path

from tests.operators.chunk_kda_fwd.common.case_matrix import manifest


ROOT = Path(__file__).resolve().parents[5]


def test_host_contract_has_platform_and_negative_matrix():
    data = manifest()
    assert set(data["capability"]["soc"]) >= {"ascend910b", "ascend910_93", "ascend950"}
    negatives = [case for case in data["cases"] if "negative" in case["tags"]]
    assert negatives
    for case in negatives:
        assert case["expect"]["return_code"] != "ACLNN_SUCCESS"
        assert case["expect"].get("message_contains")
        assert "aclnn" in case["run_on"] or case["expect"]["return_code"] == "RuntimeError"


def test_route_case_uses_one_shape_definition():
    data = manifest()
    route_cases = [case for case in data["cases"] if "route" in case["tags"]]
    assert route_cases
    assert any({"ascendc", "aclnn", "direct_launch"} <= set(case["run_on"]) for case in route_cases)


def test_rank4_varlen_h_uses_flattened_public_abi_across_entrypoints():
    op_api = (
        ROOT
        / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp"
    ).read_text(encoding="utf-8")
    plugin = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")

    assert "info.isRank3 || params.cuSeqlensOptional != nullptr" in op_api
    assert "hExport = AsRank4(hExport, hExportShape5, executorPtr);" in op_api
    assert "(is_rank3 || cu_seqlens.has_value())" in plugin


def test_sequence_count_limit_applies_only_to_packed_varlen():
    op_api = (
        ROOT
        / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp"
    ).read_text(encoding="utf-8")
    plugin = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")
    ctypes = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    api = (
        ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/docs/api.md"
    ).read_text(encoding="utf-8")

    assert (
        "params.cuSeqlensOptional == nullptr || "
        "info.seqNum <= MAX_KDA_VARLEN_SEQUENCES"
    ) in op_api
    assert "!cu_seqlens.has_value() || seq_num <= 1024" in plugin
    assert 'TORCH_CHECK(seq_num <= 1024' not in plugin
    assert "if cu is not None:" in ctypes
    assert "if len(cu) - 1 > 1024:" in ctypes
    assert "该 1024 上限不适用于" in api


def test_a5_h96_model_performance_cases_keep_full_preprocess_contract():
    data = manifest()
    cases = {case["id"]: case for case in data["cases"]}
    for suffix, tokens, chunks in (("t8k", 8192, 128), ("t16k", 16384, 256)):
        case = cases[f"chunk_kda_fwd_h96_{suffix}_model_performance"]
        assert case["soc"] == ["ascend950"]
        assert case["layout"] == "BSND"
        assert case["shape"] == {
            "B": 1,
            "H_k": 96,
            "H_v": 96,
            "T": tokens,
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "N_c": chunks,
        }
        assert case["dtype"]["q_k_v"] == "bfloat16"
        assert case["dtype"]["beta"] == "bfloat16"
        assert case["attrs"]["use_gate_in_kernel"] is True
        assert case["attrs"]["use_qk_l2norm_in_kernel"] is True
        assert case["attrs"]["use_beta_sigmoid_in_kernel"] is True
        assert case["attrs"]["safe_gate"] is True
        assert case["attrs"]["lower_bound"] == -5.0


def test_a5_one_click_entry_builds_and_runs_the_acceptance_matrix():
    shell = (ROOT / "scripts/validate_kda_a5.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/validate_kda_a5.py").read_text(encoding="utf-8")
    probe = (
        ROOT / "tests/operators/chunk_kda_fwd/st/probe_a5_tail.py"
    ).read_text(encoding="utf-8")

    assert 'DEFAULT_REF="refs/pull/264/head"' in shell
    assert 'export FLA_NPU_OPS="$ops"' in shell
    assert "check_packaged_wheel_api.py" in shell
    assert 'cache_root="$work_root/cache"' in shell
    assert "Reusing cached wheel" in shell
    assert "kda_a5_diagnostics.tar.gz" in shell
    assert "tail_sync" in runner
    assert "bf16_gate_params" in runner
    assert "h96_t8k_t16k" in runner
    assert "profile_h96_t8k" in runner
    assert "profile_h96_t16k" in runner
    assert "returncode == 124" in runner
    assert "extract_probe_records" in runner
    assert "extract_probe_progress" in runner
    assert "format_probe_progress" in runner
    assert "summary.txt" in runner
    assert runner.index('Case("h96_t8k_t16k"') < runner.index(
        'Case("bf16_gate_params"'
    )
    assert "changed_outputs=" in runner
    assert "last_progress=" in runner
    assert '"--bf16-gate-params"' in probe
    assert "value.detach().cpu().contiguous()" in probe
    assert '"deterministic_by_output"' in probe
    assert '"repeat_summaries"' in probe
    assert "collect remaining tail diagnostics" in probe
    assert '"launch_begin"' in probe
    assert '"launch_returned"' in probe
    assert '"synchronize_done"' in probe
    assert "FLA_NPU_KDA_ADAPTER_DEBUG_SYNC" in probe

    adapter = (
        ROOT
        / "torch_custom/fla_npu/fla_npu/adapters/triton_ascend_kda.py"
    ).read_text(encoding="utf-8")
    assert "def _debug_synchronize" in adapter
    assert '_debug_synchronize("adapter_core")' in adapter
    assert '_debug_synchronize("adapter_layout_exports")' in adapter
