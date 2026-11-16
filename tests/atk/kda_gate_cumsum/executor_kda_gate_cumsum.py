"""Explicit ATK inputs; NPU dependencies load only on the NPU path."""
import importlib.util
from pathlib import Path
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_CPU_PATH = Path(__file__).resolve().parent / "scripts" / "kda_gate_cumsum_cpu.py"
_CPU_SPEC = importlib.util.spec_from_file_location("atk_kda_gate_cumsum_cpu", _CPU_PATH)
_CPU_MODULE = importlib.util.module_from_spec(_CPU_SPEC)
_CPU_SPEC.loader.exec_module(_CPU_MODULE)


def build_inputs(kwargs):
    """Keep ATK floating samples; construct only value-dependent metadata."""
    values = dict(kwargs)
    g = values["g"]
    tokens = int(g.shape[-2])
    varlen = bool(values.pop("varlen"))
    values.pop("use_rank3", None)
    with_bias = bool(values.pop("with_bias"))
    values["cu_seqlens"] = [0, tokens // 2, tokens] if varlen else None
    if not bool(values["use_gate_in_kernel"]):
        values["a_log"] = None
        values["dt_bias"] = None
    elif not with_bias:
        values["dt_bias"] = None
    return values


def run_cpu(values):
    return _CPU_MODULE.kda_gate_cumsum_cpu(**values)


def run_npu(values):
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc
    values = dict(values)
    a_log = values.pop("a_log")
    return ascendc.npu_kda_gate_cumsum(A_log=a_log, **values)


@register("executor_kda_gate_cumsum")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        values = build_inputs(input_data.kwargs)
        if self.device == "cpu":
            return run_cpu(values)
        if self.device == "npu":
            return run_npu(values)
        raise RuntimeError(f"Unsupported device: {self.device}")
