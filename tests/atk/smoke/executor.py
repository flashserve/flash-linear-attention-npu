"""Dispatch the NPU DUT and both CPU references using each suite's tested output ABI."""

import importlib.util, json, sys, os
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

suite = json.loads(os.environ["FLA_ATK_SUITE"])
op = suite["op"]
group = suite["group"]
reference_root = Path(__file__).resolve().parent / "cases"
operator_dir = reference_root / op
repo_root = Path(__file__).resolve().parents[3]
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
uses_case_spec = op in (
    "chunk_fwd_h",
    "chunk_gdn_bwd_intra",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_gated_delta_rule_bwd_finalize",
    "pre_process_bwd_kernel_merged",
) or (op == "causal_conv1d_bwd" and group.endswith("functional-defined"))
if op == "chunk_kda_bwd_recompute":
    uses_case_spec = True
if uses_case_spec:
    executor_base = BaseApi
    module = repo_root / suite["plugin"]
    sys.path[:0] = [str(module.parent), str(repo_root / "tests/atk" / op), str(helpers)]
    input_executor = load(module, "historical_dut")
if op in ("merge_fwd_bwd_kernel", "chunk_gated_delta_rule_bwd"):
    sys.path.insert(0, str(repo_root / "tests/atk" / op))
    input_executor = load(repo_root / suite["plugin"], "historical_direct")
    executor_base = BaseApi


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
        if uses_case_spec:
            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
        if kda_backward_profile:
            from executor_kda_audit import run_kda_profile

            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            out = run_kda_profile(op, reference, spec, input_data, cpu, high)
        elif op == "chunk_gated_delta_rule_bwd":
            if cpu:
                spec = json.loads(
                    case.get_input_data_config(name="case_spec").range_values
                )
                inputs = input_executor.build_inputs(spec, torch.device("cpu"))
                options = dict(spec)
                for key in (
                    "state_v_first",
                    "use_qk_l2norm",
                    "use_beta_sigmoid",
                    "use_exp2",
                ):
                    options.setdefault(key, False)
                options.setdefault("chunk_size", 64)
                bounds = inputs["cu_seqlens"]
                inputs.setdefault(
                    "seqlens",
                    None
                    if bounds is None
                    else [right - left for left, right in zip(bounds, bounds[1:])],
                )
                values = list(reference.run_cpu(options, inputs, high))
                if options.get("layout", "BNSD") in ("BNSD", "NTD"):
                    values[:3] = [v.transpose(1, 2).contiguous() for v in values[:3]]
                out = input_executor._finite_tuple(values)
            else:
                spec = json.loads(
                    case.get_input_data_config(name="case_spec").range_values
                )
                out = input_executor.run_npu(spec, input_data)
            if not high:
                out = tuple(t.float() for t in tensors(out))
        elif op == "recurrent_gated_delta_rule":
            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            out = (
                reference.run_cpu(spec, high)
                if cpu
                else reference.run_npu(spec, input_data)
            )
        elif op == "chunk_kda_bwd_recompute":
            if cpu:
                x = (
                    input_executor.checkpoint_inputs(spec, torch.device("cpu"), high)
                    if hasattr(input_executor, "checkpoint_inputs")
                    else input_executor.build_inputs(spec, torch.device("cpu"), high)
                )
                # The reference accepts one sequence. Apply it to each independent packed
                # sequence before restoring the wheel's public tuple of optional outputs.
                bounds = x.get("cu_seqlens")
                if bounds is None:
                    bounds = [0, x["q"].shape[2]]
                parts = []
                for left, right in zip(bounds, bounds[1:]):
                    subset = {
                        k: (
                            v[:, :, left:right]
                            if k in ("q", "k", "v", "g", "beta", "A")
                            else v
                        )
                        for k, v in x.items()
                    }
                    parts.append(reference._fused_ref(subset))
                out = tuple(torch.cat(items, dim=2) for items in zip(*parts))
                if not x["use_gate"]:
                    out = out[1:]
            elif hasattr(input_executor, "checkpoint_inputs"):
                out = input_executor.FunctionApi.__call__(self, input_data, True)
            else:
                out = input_executor.run_npu(spec, input_data)
        elif op == "chunk_gated_delta_rule_fwd_h" and group.endswith("generated"):
            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            if cpu:
                h, v = reference.run_cpu(spec, high)
                out = (h.transpose(1, 2).contiguous(), v)
            else:
                out = reference.run_npu(spec, input_data)
        elif op == "chunk_fwd_h":
            x = input_executor.build_inputs(
                spec,
                torch.device("cpu")
                if cpu
                else next(
                    t.device for t in input_data.kwargs.values() if torch.is_tensor(t)
                ),
            )
            if cpu:
                out = reference._reference(
                    x,
                    output_final_state=spec["output_final_state"],
                    use_exp2=spec["use_exp2"],
                    state_v_first=spec["state_v_first"],
                    high_precision=high,
                )
            else:
                out = input_executor.run_npu(spec, x)
        elif op == "chunk_gdn_bwd_intra":
            if cpu:
                x = input_executor.build_inputs(
                    spec, torch.device("cpu"), high_precision=False
                )
                out = reference.run_cpu(spec, x, high)
            else:
                out = input_executor.run_npu(spec, input_data)
        elif op == "chunk_gated_delta_rule_bwd_dhu":
            if cpu:
                x = input_executor.build_inputs(
                    spec, torch.device("cpu"), high_precision=False
                )
                out = reference.chunk_gated_delta_rule_bwd_dhu_cpu(
                    x["q"],
                    x["k"],
                    x["w"],
                    x["do"],
                    x["dv"],
                    g=x["g"],
                    gK=x["gK"],
                    h0=x["h0"],
                    dht=x["dht"],
                    cu_seqlens=x["cu_seqlens"],
                    chunk_indices=x["chunk_indices"],
                    scale=x["scale"],
                    chunk_size=x["chunk_size"],
                    state_v_first=spec.get("state_v_first", False),
                    nt_first=True,
                    use_exp2=spec.get("use_exp2", False)
                    if group.endswith("gate-base")
                    else spec.get("use_gk", False),
                    golden_mode="fp64" if high else "npu",
                )
            else:
                out = input_executor.run_npu(spec, input_data)
        elif op == "chunk_kda_fwd_finalize":
            x = reference._prepare(input_data.kwargs)
            reference._clear_aqk_upper_(x)
            out = (
                reference.run_cpu(x, high_precision=high)
                if cpu
                else reference.run_npu(x)
            )
        elif (
            op == "chunk_gated_delta_rule_bwd_finalize"
            or op == "pre_process_bwd_kernel_merged"
            or (op == "causal_conv1d_bwd" and group.endswith("functional-defined"))
        ):
            from bridges import bridge

            out = bridge(
                op, reference, input_executor, spec, input_data, cpu, high, self
            )
        elif op == "merge_fwd_bwd_kernel":
            v = input_data.kwargs
            if cpu:
                t = reference._cpu().merge_reference(
                    v["ag_hm"].to(torch.float64 if high else torch.float32),
                    int(v["rank"]),
                    int(v["pre_or_post_num_ranks"]),
                    bool(v["forward"]),
                    bool(v["state_v_first"]),
                )
                out = t if high else t.to(v["h"].dtype)
            else:
                out = input_executor.run_npu(v)
        elif op == "causal_conv1d" and group.endswith("prior-gaps"):
            spec = json.loads(case.get_input_data_config(name="case_spec").range_values)
            device = (
                torch.device("cpu") if cpu else reference._marker_device(input_data)
            )
            x = reference.build_inputs(spec, device, high_precision=high)
            if not spec["with_bias"]:
                x["bias"] = None
            if not spec["with_conv_states"]:
                x["conv_states"] = None
            if cpu:
                out = reference._causal_conv1d_ref(x["x"], x["weight"], x["bias"])
            else:
                from fla_npu.ops import ascendc

                out = ascendc.causal_conv1d_fn(
                    x["x"],
                    x["weight"],
                    x["bias"],
                    x["conv_states"],
                    x.get("query_start_loc"),
                    activation=None,
                )
        else:
            out = super().__call__(input_data, True)
        result = tensors(out)
        if not result or any(not torch.isfinite(t).all().item() for t in result):
            raise RuntimeError("nonfinite or empty output")
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
