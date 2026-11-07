"""Compare public sequence-major gradients with the head-major control path."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_executor():
    path = Path(__file__).resolve().parents[1] / "executor_chunk_gated_delta_rule_bwd.py"
    spec = importlib.util.spec_from_file_location("gdn_layout_executor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def supplemental_specs():
    specs = []
    for tokens in (1, 63, 64, 65):
        for exp2 in (False, True):
            specs.append(dict(B=2, HK=2, HV=6, T=tokens, use_exp2=exp2,
                              with_state=True, with_dht=True,
                              use_qk_l2norm=True, use_beta_sigmoid=True))
    specs += [
        dict(B=1, HK=2, HV=6, T=129, seqlens=[0, 1, 0, 63, 65, 0],
             with_state=True, with_dht=True, use_exp2=exp2,
             state_v_first=vfirst, g_dtype="bf16", beta_dtype="fp32")
        for exp2 in (False, True) for vfirst in (False, True)
    ]
    specs += [dict(B=1, HK=4, HV=4, T=4, use_exp2=True),
              dict(B=2, HK=2, HV=6, T=65, use_exp2=True,
                   noncontiguous=True, trace_values=True)]
    for index, spec in enumerate(specs):
        spec.update(name=f"layout_supplement_{index}", seed=20261107 + index)
    return specs


def launch(executor, inputs, spec, sequence_major):
    from fla_npu.ops import ascendc

    def public(name, seq):
        value = executor._public_layout(inputs[name], seq)
        if value is not None and spec.get("noncontiguous"):
            value = torch.stack((value, value), dim=-1)[..., 0]
        return value

    state, dht = inputs["initial_state"], inputs["dht"]
    if spec.get("state_v_first"):
        state = state.transpose(-1, -2).contiguous() if state is not None else None
        dht = dht.transpose(-1, -2).contiguous() if dht is not None else None
    packed = inputs["cu_seqlens"] is not None
    layout = ("TND" if sequence_major else "NTD") if packed else (
        "BSND" if sequence_major else "BNSD")
    return ascendc.npu_chunk_gated_delta_rule_bwd(
        public("q", sequence_major), public("k", sequence_major),
        public("v", sequence_major), public("g", True), public("beta", True),
        inputs["A"], public("d_o", True), inputs["scale"],
        chunk_size=64, layout=layout, initial_state=state, dht=dht,
        q_rstd=inputs["q_rstd"], k_rstd=inputs["k_rstd"],
        beta_raw=public("beta_raw", True), a_log=inputs["a_log"],
        dt_bias=inputs["dt_bias"], use_exp2=spec.get("use_exp2", True),
        use_gate_in_kernel=False,
        use_qk_l2norm_in_kernel=spec.get("use_qk_l2norm", False),
        use_beta_sigmoid_in_kernel=spec.get("use_beta_sigmoid", False),
        state_v_first=spec.get("state_v_first", False),
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"],
        return_intermediate_states=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--emit-supplemental-cases", type=Path)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--write-control-dir", type=Path,
                          help="Freeze inputs and head-major outputs using the baseline package")
    controls.add_argument("--read-control-dir", type=Path,
                          help="Compare against frozen inputs and baseline head-major outputs")
    parser.add_argument("--baseline-empty-dh0-fix", action="store_true",
                        help="Check the documented empty dh0=dht contract where the old package returned zero")
    args = parser.parse_args()
    if args.baseline_empty_dh0_fix and not args.read_control_dir:
        parser.error("--baseline-empty-dh0-fix requires --read-control-dir")
    case_path = Path(__file__).resolve().parents[1] / "atk_chunk_gated_delta_rule_bwd.json"
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    if args.emit_supplemental_cases:
        supplements = []
        for index, profile in enumerate(supplemental_specs()):
            case = deepcopy(cases[0])
            item = next(item for item in case["inputs"] if item["name"] == "case_spec")
            spec = json.loads(item["range_values"])
            spec.update(profile, case_id=index)
            spec["layout"] = "TND" if "seqlens" in profile else "BSND"
            # Noncontiguous storage and trace values are exercised by the byte
            # comparison below; ATK retains the original random input ranges.
            spec.pop("noncontiguous", None)
            spec.pop("trace_values", None)
            item["range_values"] = json.dumps(spec, separators=(",", ":"))
            case.update(id=index, name=f"chunk_gated_delta_rule_bwd_supplement_{index}",
                        default_seed=spec["seed"])
            case.pop("uuid", None)
            supplements.append(case)
        args.emit_supplemental_cases.write_text(json.dumps(supplements), encoding="utf-8")
        print(f"wrote {len(supplements)} supplemental ATK cases")
        return
    result_dir = args.write_control_dir or args.save_dir
    if result_dir is None:
        parser.error("--save-dir or --write-control-dir is required for device comparisons")
    torch.npu.set_device(args.device)
    executor = load_executor()
    specs = [json.loads(next(item["range_values"] for item in case["inputs"]
                             if item["name"] == "case_spec")) for case in cases]
    specs += supplemental_specs()
    result_dir.mkdir(parents=True, exist_ok=False)
    failures = []
    for case_id, spec in enumerate(specs):
        try:
            if args.read_control_dir:
                frozen = torch.load(args.read_control_dir / f"case{case_id}.pt", weights_only=False)
                assert frozen["spec"] == spec
                inputs = {name: value.to(f"npu:{args.device}") if isinstance(value, torch.Tensor)
                          else value for name, value in frozen["inputs"].items()}
                control = frozen["outputs"]
            else:
                inputs = executor.build_inputs(spec, torch.device(f"npu:{args.device}"))
                if spec.get("trace_values"):
                    for offset, name in enumerate(("q", "k", "v")):
                        value = inputs[name]
                        traced = ((torch.arange(value.numel(), dtype=torch.int64) // 128 +
                                   offset * 7) % 31 - 15).float() / 128
                        inputs[name] = traced.reshape(value.shape).to(value.device, value.dtype)
                control = launch(executor, inputs, spec, False)
            if args.write_control_dir:
                frozen_inputs = {name: value.cpu() if isinstance(value, torch.Tensor) else value
                                 for name, value in inputs.items()}
                frozen_outputs = tuple(value.cpu() if value is not None else None for value in control)
                assert all(torch.isfinite(value).all().item() for value in frozen_outputs if value is not None)
                torch.save(dict(spec=spec, inputs=frozen_inputs, outputs=frozen_outputs),
                           result_dir / f"case{case_id}.pt")
                print(case_id, spec["name"], "FROZEN", flush=True)
                continue
            actual = launch(executor, inputs, spec, True)
            torch.npu.synchronize()
            mismatched = []
            for index, (expected, observed) in enumerate(zip(control, actual)):
                if expected is None:
                    if observed is not None:
                        mismatched.append(index)
                    continue
                expected = expected.cpu()
                if index == 5 and args.baseline_empty_dh0_fix and inputs["seqlens"] is not None:
                    empty = [i for i, length in enumerate(inputs["seqlens"]) if length == 0]
                    if empty:
                        expected = expected.clone()
                        terminal = inputs["dht"]
                        if terminal is None:
                            expected[empty] = 0
                        else:
                            terminal = terminal.cpu()
                            if spec.get("state_v_first"):
                                terminal = terminal.transpose(-1, -2)
                            expected[empty] = terminal[empty]
                        print(case_id, "empty dh0 checked against dht contract", empty, flush=True)
                observed_contiguous = observed.is_contiguous()
                observed = observed.cpu()
                if index < 3:
                    expected = expected.transpose(1, 2).contiguous()
                matches = (expected.shape == observed.shape and
                           expected.dtype == observed.dtype and observed_contiguous and
                           torch.isfinite(observed).all().item() and
                           torch.equal(expected.view(torch.uint8), observed.view(torch.uint8)))
                if not matches:
                    mismatched.append(index)
                    torch.save(expected, result_dir / f"case{case_id}_out{index}_head.pt")
                    torch.save(observed, result_dir / f"case{case_id}_out{index}_seq.pt")
            if mismatched:
                failures.append(dict(case_id=case_id, outputs=mismatched, spec=spec))
            print(case_id, spec["name"], "FAIL" if mismatched else "PASS", mismatched, flush=True)
        except Exception as exc:
            failures.append(dict(case_id=case_id, error=str(exc), spec=spec))
            print(case_id, spec["name"], "ERROR", str(exc), flush=True)
    (result_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"layout equivalence: total={len(specs)} pass={len(specs)-len(failures)} fail={len(failures)}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
