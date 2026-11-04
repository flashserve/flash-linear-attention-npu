"""Generate only the two issue #795 model shapes for each of the four ATK entries.

Keep existing seeds, input generators and mixed_tolerance_bm standards.
Run: python tests/atk/chunk_gated_delta_rule_bwd/scripts/gen_gva_typical.py --output-dir <dir>
"""
import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

ATK = Path(__file__).resolve().parents[2]
OPS = ("chunk_gdn_bwd_intra", "chunk_gated_delta_rule_bwd_dhu",
       "chunk_gated_delta_rule_bwd_finalize", "chunk_gated_delta_rule_bwd")


def generate(op):
    folder = ATK / op
    template = json.loads((folder / f"atk_{op}.json").read_text())[0]
    base = json.loads(next(x["range_values"] for x in template["inputs"]
                           if x["name"] == "case_spec"))
    module = None
    if op == "chunk_gated_delta_rule_bwd_finalize":
        module_name = "gen_chunk_gated_delta_rule_bwd_finalize"
        if module_name not in sys.modules:
            module_spec = importlib.util.spec_from_file_location(module_name, folder / f"gen_{op}.py")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        module = sys.modules[module_name]
    cases = []
    for index, tokens in enumerate((11264, 16384)):
        spec = deepcopy(base)
        name = f"issue795_g8_{'fixed' if index == 0 else 'varlen16x1024'}"
        spec.update(B=1, HK=16, HV=128, T=tokens, K=128, V=128,
                    chunk_size=64, dtype="bf16", scalar_dtype="fp32",
                    g_dtype="fp32", beta_dtype="fp32", layout="BNSD",
                    name=name, case_key=name, case_id=index,
                    seed=template["default_seed"], soc="ascend950",
                    varlen=bool(index), use_exp2=True,
                    use_qk_l2norm=False, use_beta_sigmoid=False,
                    state_v_first=False, with_state=False, with_h0=False,
                    with_dht=False, use_gk=False)
        spec.pop("cu_seqlens", None)
        spec.pop("seqlens", None)
        if index:
            spec["cu_seqlens"] = list(range(0, tokens + 1, 1024))
            spec["seqlens"] = [1024] * 16
        if op == "chunk_gated_delta_rule_bwd_finalize":
            case = module._case_payload(index, spec)
        else:
            case = deepcopy(template)
            case.update(id=index, name=f"{op}_{name}", default_seed=spec["seed"])
            for item in case["inputs"]:
                if item["name"] == "case_spec":
                    item["range_values"] = json.dumps(spec, separators=(",", ":"))
                elif item["name"] in spec:
                    item["range_values"] = spec[item["name"]]
        # ATK excludes non_param attributes from executor kwargs. Keep the full
        # spec as a string so explicit varlen boundaries and auxiliary dtypes survive.
        for item in case["inputs"]:
            if item["name"] == "case_spec":
                item["dtype"] = "string"
        cases.append(case)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for op in OPS:
        target = args.output_dir / f"atk_{op}_gva_typical.json"
        target.write_text(json.dumps(generate(op), ensure_ascii=False, indent=2) + "\n")
        print(target)
