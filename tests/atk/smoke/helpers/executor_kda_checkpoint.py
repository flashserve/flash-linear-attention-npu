"""Use a forward-consistent BF16 triangular checkpoint for standalone backward recompute."""

import json
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import executor_chunk_kda_bwd_recompute as base


def checkpoint_inputs(spec, device, high):
    s = base.build_inputs(spec, torch.device("cpu"), False)
    for name in ("q", "k"):
        s[name] = torch.nn.functional.normalize(s[name].float(), dim=-1).to(
            torch.bfloat16
        )
    raw = s["g"].double()
    if s["use_gate"]:
        if "dt_bias" in s:
            raw = raw + s["dt_bias"].double()[None, :, None]
        raw = s["lower_bound"] * torch.sigmoid(
            raw * torch.exp(s["A_log"].double())[None, :, None, None]
        )
    a = torch.zeros_like(s["A"])
    keys = base._expand_hk_to_hv(s["k"].double(), raw.shape[1])
    for start, end in base._input_chunks(s):
        gate = raw[:, :, start:end].cumsum(2)
        kk = keys[:, :, start:end]
        n = end - start
        eye = torch.eye(n, dtype=torch.float64)
        diff = gate.unsqueeze(3) - gate.unsqueeze(2)
        lower = eye.new_ones(n, n).tril(-1).bool()
        exp = torch.exp(diff.masked_fill(~lower[None, None, :, :, None], 0))
        gram = (kk.unsqueeze(3) * kk.unsqueeze(2) * exp).sum(-1)
        lhs = (gram * s["beta"][:, :, start:end, None].double()).masked_fill(~lower, 0)
        inverse = torch.linalg.solve_triangular(
            lhs + eye, eye.expand_as(lhs), upper=False, unitriangular=True
        )
        a[:, :, start:end, :n] = inverse.to(a.dtype)
    s["A"] = a
    return {
        k: (
            v.to(torch.float64 if high else v.dtype).to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in s.items()
    }


@register("executor_kda_checkpoint")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        r = self.task_result
        s = json.loads(
            r.case_config.get_input_data_config(name="case_spec").range_values
        )
        cpu = self.device == "cpu"
        x = checkpoint_inputs(
            s,
            torch.device("cpu") if cpu else base._marker_device(input_data),
            cpu and bool(r.is_benchmark_task),
        )
        if cpu:
            return base._finite_outputs(base._fused_ref(x))
        from fla_npu.ops.ascendc import npu_chunk_kda_bwd_recompute

        return base._finite_outputs(
            npu_chunk_kda_bwd_recompute(
                x["q"],
                x["k"],
                x["v"],
                x["g"],
                x["beta"],
                x["A"],
                x["chunk_size"],
                A_log=x["A_log"] if x["use_gate"] else None,
                dt_bias=x.get("dt_bias"),
                cu_seqlens=x.get("cu_seqlens"),
                chunk_indices=x.get("chunk_indices"),
                use_gate_in_kernel=x["use_gate"],
                use_exp2=x["use_exp2"],
                lower_bound=x["lower_bound"],
                input_layout=s.get("input_layout", "BNSD"),
            )
        )
