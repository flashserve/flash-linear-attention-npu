"""Phase-A shape scaffold; CPU zero outputs deliberately fail numerical accuracy."""
import importlib.util
from pathlib import Path
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_path = Path(__file__).resolve().parent / "scripts" / "chunk_kda_bwd_cpu.py"
_spec = importlib.util.spec_from_file_location("atk_chunk_kda_bwd_cpu", _path)
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def build_inputs(kwargs):
    values = dict(kwargs)
    packed = bool(values.pop("is_varlen"))
    with_bias = bool(values.pop("with_bias"))
    q = values["q"]
    tokens = int(q.shape[-2])
    chunk_size = int(values["chunk_size"])
    if packed:
        cu = [0, tokens // 2, tokens]
        chunks = []
        for seq, (start, end) in enumerate(zip(cu, cu[1:])):
            for chunk in range((end - start + chunk_size - 1) // chunk_size):
                chunks.extend((seq, chunk))
        values["cu_seqlens"], values["chunk_indices"] = cu, chunks
    else:
        values["cu_seqlens"], values["chunk_indices"] = None, None
    if not with_bias:
        values["dt_bias"] = None
    # The selected original path fixes the head-major public layouts.
    values["input_layout"] = "NTD" if packed else "BNSD"
    values["output_layout"] = values["input_layout"]
    return values


def run_cpu(values):
    return _cpu.chunk_kda_bwd_cpu(**values)


def run_npu(values):
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc
    return ascendc.npu_chunk_kda_bwd(**values)


@register("executor_chunk_kda_bwd")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        values = build_inputs(input_data.kwargs)
        if self.device == "cpu":
            return run_cpu(values)
        if self.device == "npu":
            return run_npu(values)
        raise RuntimeError(f"Unsupported device: {self.device}")
