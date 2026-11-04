# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

import importlib.util
from pathlib import Path
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_CPU_PATH = Path(__file__).resolve().parent / "scripts" / "chunk_kda_bwd_prepare_cpu.py"
_spec = importlib.util.spec_from_file_location("chunk_kda_bwd_prepare_cpu", _CPU_PATH)
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def canonical_metadata(t):
    # Same split as the constraint generator: one empty and two positive/tail sequences.
    cu = [0, 0, t // 2, t]
    indices = [entry for seq, (begin, end) in enumerate(zip(cu, cu[1:]))
               for chunk in range((end - begin + 63) // 64) for entry in (seq, chunk)]
    return cu, indices


def prepare_metadata(kwargs, t):
    variable = kwargs.pop("is_varlen")
    explicit = kwargs.pop("explicit_chunk_indices")
    cu, indices = canonical_metadata(t) if variable else (None, None)
    kwargs["cu_seqlens"] = cu
    kwargs["chunk_indices"] = indices if explicit else None

@register("executor_chunk_kda_bwd_prepare")
class FunctionApi(BaseApi):
    def init_by_input_data(self, input_data):
        kwargs = input_data.kwargs
        layout = kwargs["layout"]
        shape = kwargs["aqk"].shape
        t = shape[1] if layout in ("BSND", "NTD") else shape[-2] if layout == "BNSD" else shape[0]
        prepare_metadata(kwargs, t)

    def __call__(self, input_data, with_output=False):
        if self.device == "cpu":
            return _cpu.chunk_kda_bwd_prepare_cpu(**input_data.kwargs)
        if self.device == "npu":
            from fla_npu.ops import ascendc
            return ascendc.npu_chunk_kda_bwd_prepare(**input_data.kwargs)
        raise RuntimeError(f"Unsupported device: {self.device}")
