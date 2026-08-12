from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "scripts" / "benchmark_kda_b200_triton.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "benchmark_kda_b200_triton", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_b200_matrix_matches_the_eight_dense_atk_cases():
    runner = load_runner()
    assert [case.case_id for case in runner.CASES] == list(range(250, 258))
    assert [case.total_tokens for case in runner.CASES] == [
        1024,
        1024,
        8192,
        8192,
        16384,
        16384,
        65536,
        65536,
    ]
    assert [case.recompute_enabled for case in runner.CASES] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [case.chunk_count for case in runner.CASES] == [
        16,
        16,
        128,
        128,
        256,
        256,
        1024,
        1024,
    ]
    assert runner.select_cases("250,257") == [runner.CASES[0], runner.CASES[-1]]


def test_default_run_count_is_ten(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH)])
    args = runner.parse_args()
    assert args.warmup == 5
    assert args.runs == 10


def test_summary_uses_the_arithmetic_mean_of_all_measurements():
    runner = load_runner()
    timings = [float(value) for value in range(1, 11)]
    summary = runner.summarize_timings(
        runner.CASES[0], timings, peak_tflops=2250.0
    )
    assert summary["mean_us"] == 5.5
    assert summary["median_us"] == 5.5
    assert summary["min_us"] == 1.0
    assert summary["max_us"] == 10.0
    assert summary["a5_over_b200"] == runner.CASES[0].a5_us / 5.5


def test_invoke_passes_the_fixed_dense_fla_arguments():
    runner = load_runner()
    captured = {}

    def operation(**kwargs):
        captured.update(kwargs)
        return (None,) * 12

    inputs = {
        "q": object(),
        "k": object(),
        "v": object(),
        "g": object(),
        "beta": object(),
        "A_log": object(),
        "dt_bias": object(),
    }
    case = runner.CASES[1]
    runner.invoke(operation, inputs, case)

    assert captured["initial_state"] is None
    assert captured["output_final_state"] is False
    assert captured["cu_seqlens"] is None
    assert captured["chunk_indices"] is None
    assert captured["chunk_size"] == 64
    assert captured["safe_gate"] is True
    assert captured["lower_bound"] == -5.0
    assert captured["use_gate_in_kernel"] is True
    assert captured["disable_recompute"] is True
    assert captured["return_intermediate_states"] is False
    assert captured["state_v_first"] is True


def test_runner_uses_installed_fla_without_install_or_clone_commands():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'metadata.distribution("flash-linear-attention")' in source
    assert 'import_module("fla.ops.kda.chunk_fwd")' in source
    assert 'os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"' in source
    assert "subprocess" not in source
    assert "pip install" not in source
    assert "git clone" not in source
