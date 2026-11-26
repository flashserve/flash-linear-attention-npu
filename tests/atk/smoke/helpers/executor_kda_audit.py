"""ATK adapters using the pinned work_tool CPU equations and FLA public NPU APIs."""

import os
import importlib
import json
import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

ROOT = Path(os.environ["FLA_ATK_REFERENCE"])
sys.path.insert(0, str(ROOT / "common"))


def load_reference(folder, module):
    sys.path.insert(0, str(ROOT / folder))
    return importlib.import_module(module)


def finalize_device(module, inputs, spec, cu, chunks, device):
    from fla_npu.ops.ascendc import npu_chunk_kda_bwd_finalize

    dev = {}
    for name, value in inputs.items():
        if name not in ("a_log", "dt_bias"):
            if spec["lengths"] is not None and spec["rank"] == 4:
                value = value.unsqueeze(0)
            elif spec["lengths"] is None and spec["rank"] == 3:
                value = value.squeeze(0)
        dev[name] = value.contiguous().to(device)
    dev["d_aqk"] = dev.pop("dAqk")
    indices = (
        [n for item in chunks for n in item[:2]]
        if cu and spec["metadata"] != "auto"
        else None
    )
    return npu_chunk_kda_bwd_finalize(
        **dev,
        scale=spec["scale"],
        lower_bound=spec["lower_bound"],
        chunk_size=64,
        safe_gate=True,
        use_gate_in_kernel=True,
        use_exp2=True,
        state_v_first=False,
        output_layout=spec["output_layout"],
        cu_seqlens=cu or None,
        chunk_indices=indices,
        use_beta_sigmoid_in_kernel=spec["beta_mode"] != 0,
        allow_neg_eigval=spec["beta_mode"] == 2,
    )


@register("executor_kda_audit")
class KdaAudit(BaseApi):
    def __call__(self, input_data, with_output=False):
        field = self.task_result.case_config.get_input_data_config(name="case_spec")
        s = json.loads(field.range_values)
        device = next(
            x.device for x in input_data.kwargs.values() if torch.is_tensor(x)
        )
        cpu = self.device == "cpu"
        high = cpu and bool(self.task_result.is_benchmark_task)
        if "beta_mode" in s and "output_layout" in s:
            m = load_reference(
                "chunk_kda_bwd_finalize", "executor_chunk_kda_bwd_finalize"
            )
            inputs, cu, chunks = m.build_inputs(s, s["seed"])
            out = (
                m.reference(inputs, s, chunks, high)
                if cpu
                else finalize_device(m, inputs, s, cu, chunks, device)
            )
        elif s.get("op") == "chunk_kda_bwd_prepare":
            m = load_reference(
                "chunk_kda_bwd_prepare_a3", "executor_chunk_kda_bwd_prepare_a3"
            )
            out = m.run_cpu(s, high_precision=high) if cpu else m.run_npu(s, input_data)
        elif s.get("op") == "chunk_kda_bwd":
            m = load_reference("chunk_kda_bwd", "executor_chunk_kda_bwd_A2")
            inputs = m.build_inputs(s)
            if cpu:
                out = m.cpu_reference(inputs, s, high_precision=high)
                if not high:
                    out = m._cast_cpu_outputs_to_abi(out, s)
            else:
                from fla_npu.ops.ascendc import chunk_kda_bwd

                out = chunk_kda_bwd(**m.npu_inputs(inputs, s, device))
            out = m._finite_outputs(out, probe_device=not cpu)
        else:
            raise ValueError(f"Unknown KDA profile {s}")
        if not out or any(not torch.isfinite(t).all().item() for t in out):
            raise ValueError("Missing or non-finite KDA result")
        return tuple(out)
