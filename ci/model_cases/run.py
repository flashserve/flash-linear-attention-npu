"""Run both model suites on the current CI runtime and save each verdict as it finishes."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from daily_plan import daily_plan
from verdict import passed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke_cache import ResultCache

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_bytes())


def plan(day):
    if not any((HERE / name).exists() for name in ("kda_cases.json", "gdn_cases.json")):
        return []
    kda = daily_plan(read("kda_cases.json"), day)
    gdn = read("gdn_cases.json")
    cases = [dict(c, suite="gdn", id=c["name"], dimension=gdn["dimension"], chunk_size=gdn["chunk_size"],
                  tensor_seed=gdn["seed"]) for c in gdn["cases"]]
    cases.extend(dict(c, suite="kda", cu_seqlens=kda["cu_seqlens"].get(c.get("group"))) for c in kda["cases"])
    if len(cases) != 44 or len({c["id"] for c in cases}) != 44:
        raise ValueError("Expected 5 GDN and 39 distinct KDA cases")
    return cases


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def execute(case, args):
    import torch
    import torch_npu  # Register the NPU backend.

    torch.set_num_threads(2)
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(device)
    if ("910B" if args.platform == "a2" else "950") not in name:
        raise RuntimeError(f"Expected {args.platform}, found {name}")
    result = dict(case, hardware=name, passed=False)
    with torch.no_grad():
        if case["suite"] == "kda":
            import kda
            result["passed"] = kda.run_case(case, case["cu_seqlens"], device, result)
        else:
            import gdn
            torch.manual_seed(case["tensor_seed"])
            if case["entry"] == "prefill":
                result["passed"] = gdn.run_prefill(case, case["dimension"], case["chunk_size"], device,
                                                   args.platform, result)
            else:
                result["passed"] = gdn.run_decode(case, case["dimension"], device, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--platform", required=True, choices=("a2", "a5"))
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--case", help="Run one model case by its catalog ID")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cases = plan(args.date)
    if args.case:
        case = next((c for c in cases if c["id"] == args.case), None)
        if case is None:
            parser.error(f"Unknown case {args.case}")
        start = time.monotonic()
        try:
            result = execute(case, args)
        except Exception:
            result = dict(case, passed=False, error=traceback.format_exc())
        result["seconds"] = round(time.monotonic() - start, 2)
        save(args.report_dir / (args.case + ".json"), result)
        print(args.case, result["passed"], result.get("error", ""), flush=True)
        return 0 if result["passed"] else 1

    cache = ResultCache()
    hits = 0
    report = dict(platform=args.platform, date=args.date.isoformat(), head_sha=os.getenv("CI_ACCURACY_HEAD_SHA"),
                  run_id=os.getenv("CI_ACCURACY_RUN_ID"), attempt=os.getenv("CI_ACCURACY_RUN_ATTEMPT"), cases={})
    path = args.report_dir / "summary.json"
    if not cases:
        report.update(cache_hits=0, executed=0)
        save(path, report)
        print("[CI][MODELS] No model cases registered", flush=True)
        return 0
    save(path, report)
    for case in cases:
        ident = case["id"]
        detail = args.report_dir / (ident + ".json")
        key = cache.key("model", [args.platform, case])
        entry = cache.get(key)
        if entry and passed(case, entry["value"], args.platform):
            print(f"[smoke-cache] HIT model {ident} from {entry['origin']}", flush=True)
            save(detail, dict(entry["value"], cache={"hit": True, "origin": entry["origin"], "key": key}))
            hits += 1
            report["cases"][ident] = True
            save(path, report)
            continue
        print(f"[smoke-cache] RUN model {ident}", flush=True)
        detail.unlink(missing_ok=True)
        # Isolate large cases so CPU and NPU allocations are released even after an error.
        run = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--case", ident], check=False)
        if not detail.is_file():
            save(detail, dict(case, passed=False, error=f"Case process exited {run.returncode} without a report"))
        row = json.loads(detail.read_bytes())
        ok = run.returncode == 0 and passed(case, row, args.platform)
        if ok:
            cache.put(key, row)
        save(detail, dict(row, cache={"hit": False, "key": key if cache.root else None}))
        report["cases"][ident] = ok
        save(path, report)
    report["cache_hits"] = hits
    report["executed"] = len(cases) - hits
    save(path, report)
    print(f"[CI][MODELS] {sum(report['cases'].values())}/{len(cases)} passed, "
          f"{hits} reused, {len(cases)-hits} executed", flush=True)
    return 0 if len(report["cases"]) == len(cases) and all(report["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
