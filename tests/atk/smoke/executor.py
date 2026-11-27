"""Dispatch the NPU DUT and both CPU references using each suite's tested output ABI."""

import importlib.util, json, sys, os
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

suite = json.loads(os.environ["FLA_ATK_SUITE"])
op = suite["op"]
reference_root = Path(__file__).resolve().parent / "cases"
operator_dir = (reference_root if suite.get("reference_source") != "repository"
                else reference_root.parents[1]) / op
helpers = Path(__file__).resolve().parent / "helpers"
sys.path.insert(0, str(helpers))


def load(p, name):
    module_spec = importlib.util.spec_from_file_location(name, p)
    reference = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = reference
    module_spec.loader.exec_module(reference)
    return reference


sys.path[:0] = [
    str(operator_dir),
    str(reference_root / "common"),
    str(reference_root),
]
reference_file = operator_dir / suite["reference"]
reference = load(reference_file, "verified_smoke_reference")
executor_base = getattr(reference, "StateApi", None) or getattr(
    reference, "FunctionApi", BaseApi
)
kda_backward_profile = op in (
    "chunk_kda_bwd",
    "chunk_kda_bwd_finalize",
    "chunk_kda_bwd_prepare",
)
if kda_backward_profile:
    executor_base = BaseApi
if op == "chunk_kda_fwd":
    executor_base = reference.PublicForwardApi
if op == "chunk_kda_fwd_fused":
    executor_base = reference.ChunkKdaFwdApi


def tensors(x):
    if torch.is_tensor(x):
        return (x,)
    return tuple(t for t in x if torch.is_tensor(t))


@register("fla_smoke")
class Smoke(executor_base):
    def __call__(self, input_data, with_output=False):
        cpu = self.device == "cpu"
        high = cpu and bool(self.task_result.is_benchmark_task)
        case = self.task_result.case_config
        if kda_backward_profile:
            from executor_kda_audit import run_kda_profile

            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            out = run_kda_profile(op, reference, spec, input_data, cpu, high)
        elif op == "recurrent_gated_delta_rule":
            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            out = (
                reference.run_cpu(spec, high)
                if cpu
                else reference.run_npu(spec, input_data)
            )
        elif op == "chunk_kda_fwd_finalize":
            x = reference._prepare(input_data.kwargs)
            reference._clear_aqk_upper_(x)
            out = (
                reference.run_cpu(x, high_precision=high)
                if cpu
                else reference.run_npu(x)
            )
        else:
            out = super().__call__(input_data, True)
        result = tensors(out)
        if not result or any(not torch.isfinite(t).all().item() for t in result):
            raise RuntimeError("nonfinite or empty output")
        if high and suite.get("reference_source") == "repository" and suite["output"] == "native":
            if any(t.is_floating_point() and t.dtype != torch.float64 for t in result):
                raise RuntimeError(f"{op}: the FP64 benchmark returned a reduced-precision tensor")
        role = "high" if high else "low" if cpu else "dut"
        if any(t.device.type != ("cpu" if cpu else "npu") for t in result):
            raise RuntimeError("wrong device")
        if not cpu:
            torch.npu.synchronize()
        with open(os.environ["FLA_ATK_TRACE"] + "." + role, "a") as file:
            file.write(
                json.dumps(
                    [
                        case.id,
                        case.name,
                        [[list(t.shape), str(t.dtype)] for t in result],
                    ]
                )
                + "\n"
            )
        if suite["output"] == "native":
            return result
        return tuple(t.double() if high else t.float() for t in result)
