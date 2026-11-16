"""ATK shape scaffolding for a private L0 launcher with no standalone ABI."""
import importlib.util
from pathlib import Path
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_CPU_PATH = Path(__file__).resolve().parent / "scripts" / "kda_gate_bwd_post_varlen_cpu.py"
_CPU_SPEC = importlib.util.spec_from_file_location("atk_kda_gate_bwd_post_varlen_cpu", _CPU_PATH)
_CPU_MODULE = importlib.util.module_from_spec(_CPU_SPEC)
_CPU_SPEC.loader.exec_module(_CPU_MODULE)


def build_inputs(kwargs):
    values = dict(kwargs)
    tokens = int(values["raw_g"].shape[-2])
    if True:
        cuts = [0, tokens // 2, tokens]
        chunks = [(seq, chunk) for seq, (start, end) in enumerate(zip(cuts, cuts[1:]))
                  for chunk in range((end - start + 63) // 64)]
        # Value-dependent host tensors match the private tiling schema.
        values["cu_seqlens"] = torch.tensor(cuts, dtype=torch.int64)
        values["chunk_indices"] = torch.tensor(chunks, dtype=torch.int64)
    else:
        values["cu_seqlens"] = None
        values["chunk_indices"] = None
    return values


def run_cpu(values):
    return _CPU_MODULE.kda_gate_bwd_post_varlen_cpu(**values)


def run_npu(values):
    raise NotImplementedError(
        "kda_gate_bwd_post_varlen is a private L0 launcher without a standalone ACLNN/torch entry; "
        "provide a test-only launcher before direct NPU execution. "
        "The CPU output is a shape-only zero placeholder and cannot validate accuracy.")


@register("executor_kda_gate_bwd_post_varlen")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        values = build_inputs(input_data.kwargs)
        if self.device == "cpu":
            return run_cpu(values)
        if self.device == "npu":
            return run_npu(values)
        raise RuntimeError(f"Unsupported device: {self.device}")
