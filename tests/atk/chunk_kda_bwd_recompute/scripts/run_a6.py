#!/usr/bin/env python3
"""Emit 200 A6 double-benchmark cases and run ATK in the caller's environment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess


OP_DIR = Path(__file__).resolve().parents[1]
CASE_COUNT = 200


def profiles():
    cases = []
    # All four dtype keys, on both the full VF and the split tail VF.
    for gate in ("bf16", "fp32"):
        for beta in ("bf16", "fp32"):
            for tokens in (64, 65):
                cases.append(dict(name=f"{gate}_{beta}_t{tokens}", g_dtype=gate,
                                  beta_dtype=beta, T=tokens))
    for tokens in (1, 15, 16, 17, 63, 127, 129):
        cases.append(dict(name=f"tail_{tokens}", T=tokens))
    for heads in (2, 3, 4, 5):
        cases.append(dict(name=f"heads_{heads}", HV=heads, T=129))
    cases.extend([
        dict(name="gqa_batch", B=2, HK=2, HV=4, T=65),
        # Many more works than physical cores, for local slot wraparound.
        dict(name="slot_wrap", HV=3, T=8193),
        dict(name="varlen", HK=1, HV=3, T=130, seq_lengths=[1, 0, 64, 65]),
        dict(name="varlen_tail", HK=2, HV=4, T=127, seq_lengths=[63, 64]),
        dict(name="no_bias", has_dt_bias=False, T=65),
        dict(name="natural_exp", use_exp2=False, g_dtype="fp32", T=65),
        dict(name="raw_gate_exp2", use_gate=False, g_dtype="fp32", T=65),
        dict(name="raw_gate_exp", use_gate=False, use_exp2=False, T=65),
        dict(name="lower_bound", lower_bound=-1.0, T=65),
    ])
    return cases


def write_cases(path):
    template = json.loads((OP_DIR / "atk_chunk_kda_bwd_recompute.json").read_text())[0]
    spec_item = next(item for item in template["inputs"] if item["name"] == "case_spec")
    baseline = json.loads(spec_item["range_values"])
    scene_profiles = profiles()
    cases = []
    for index in range(CASE_COUNT):
        profile = scene_profiles[index % len(scene_profiles)]
        case = deepcopy(template)
        spec = dict(baseline, B=1, HK=1, HV=1, T=64, g_dtype="bf16", beta_dtype="bf16",
                    use_gate=True, use_exp2=True, has_dt_bias=True, lower_bound=-5.0)
        spec.update(profile, soc="ascend960dt", case_id=index, seed=20260817 + index)
        case.update(id=index, default_seed=spec["seed"],
                    name=f"recompute_a6_{profile['name']}_{index:03d}")
        for item in case["inputs"]:
            if item["name"] == "case_spec":
                item["range_values"] = json.dumps(spec, separators=(",", ":"))
            elif item["name"] in spec:
                item["range_values"] = spec[item["name"]]
        cases.append(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2) + "\n")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OP_DIR / "atk_output" / "a6")
    parser.add_argument("--device", default="0")
    parser.add_argument("--task", choices=("accuracy", "determinism"), default="accuracy")
    parser.add_argument("--emit-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    case_path = output / "atk_chunk_kda_bwd_recompute_a6.json"
    count = write_cases(case_path)
    print(f"Generated {count} cases: {case_path}", flush=True)
    if args.emit_only:
        return 0
    atk = shutil.which("atk")
    if atk is None:
        parser.error("atk is not installed in the active environment")
    cmd = [atk, "node", "--name", "npu_dut", "--backend", "npu", "--devices", args.device,
           "--output_path", str(output / args.task)]
    if args.task == "accuracy":
        cmd += ["node", "--name", "cpu_golden", "--backend", "cpu",
                "--output_path", str(output / args.task)]
    task = "accuracy" if args.task == "accuracy" else "accuracy_dc"
    cmd += ["task", "-c", str(case_path), "--task", task,
            "-p", str(OP_DIR / "executor_chunk_kda_bwd_recompute.py")]
    if args.task == "accuracy":
        cmd += ["--bm_device", "cpu", "--gm_init_flag", "-to", "14400"]
    else:
        cmd += ["--dc_loop_nums", "50", "-to", "3600"]
    return subprocess.call(cmd, cwd=OP_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
