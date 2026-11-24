"""Manual real-wheel/run incremental install, execution and recovery on A2."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import isolation

ROOT = Path(__file__).resolve().parents[2]


def installed_base(env):
    # Locate the manager's default base without importing the experimental runtime.
    code = ("import importlib.util,json\nfrom pathlib import Path\n"
            "spec=importlib.util.find_spec('fla_npu_experimental_manage')\n"
            "if spec is None or spec.origin is None:\n"
            "    raise SystemExit('Install an experimental wheel in the current Python environment first')\n"
            "print(json.dumps(str(Path(spec.origin).resolve().parent/'fla_npu_experimental')))\n")
    package = Path(json.loads(subprocess.check_output(
        [sys.executable, "-c", code], env=env, text=True)))
    if not (package / "__init__.py").is_file():
        raise RuntimeError(f"Installed experimental base is missing: {package}")
    return package


def validate(wheel, run, output, device=0):
    env = isolation.environment({"mode": "stable", "scenario": "stable"}, {})
    package = installed_base(env)
    output.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    env = isolation.environment({"mode": "experimental", "scenario": "experimental"},
                                {"experimental_package": str(package)})
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONPATH=os.pathsep.join(filter(None, (str(package.parent), env.get("PYTHONPATH")))))
    before = isolation.snapshot([package])
    results = []

    def manage(*args):
        result = subprocess.run([python, "-m", "fla_npu_experimental_manage",
                                 "--package-dir", str(package), *map(str, args)],
                                env=env, check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def exercise(name, adapter, opp, disabled=False):
        current = dict(env, FLA_A2_TRACE_DIR=str(output / (name + "-trace")))
        if disabled:
            current["FLA_NPU_OVERLAY_VENDOR"] = ""
        script = ('source "$1" && exec "$2" "$3" --mode quick --device "$4" --output "$5"')
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(["bash", "-c", script, "incremental",
                str(package / "opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash"),
                python, str(ROOT / "experimental/_tests/accuracy.py"), str(device),
                str(output / (name + ".json"))], env=current, check=True,
                stdout=log, stderr=subprocess.STDOUT, timeout=600)
        report = json.loads((output / (name + ".json")).read_text())
        if not report.get("complete") or report.get("check_pass") != 3 or not report.get("all_pass"):
            raise RuntimeError(f"Incomplete incremental accuracy: {name}")
        traces = list((output / (name + "-trace")).glob("*.json"))
        if len(traces) != 1:
            raise RuntimeError(f"Missing incremental execution trace: {name}")
        trace = json.loads(traces[0].read_text())
        if (trace["wrapper_module"] == "_fla_npu_experimental_layer") != adapter:
            raise RuntimeError(f"Wrong Python adapter selected: {name}")
        if ("/wheel_layers/" in trace["custom_opapi"]) != opp:
            raise RuntimeError(f"Wrong OPP selected: {name}")
        results.append({"name": name, "status": "PASS", "trace": trace})

    # Each exercise is a new process. No process observes an installation in flight.
    try:
        result = manage("--source", wheel)
        if result["state"] != "active":
            raise RuntimeError(f"Wheel layer did not activate: {result}")
        exercise("wheel", adapter=True, opp=True)
        result = manage("--source", run)
        if result["state"] != "active":
            raise RuntimeError(f"Run component did not activate: {result}")
        exercise("wheel-plus-run", adapter=True, opp=True)
        exercise("disabled", adapter=False, opp=False, disabled=True)
        manage("--remove-overlay")
        exercise("restored", adapter=False, opp=False)
        result = manage("--source", run)
        if result["state"] != "active":
            raise RuntimeError(f"OPP-only layer did not activate: {result}")
        exercise("run-only", adapter=False, opp=True)
    finally:
        manage("--remove-overlay")
    after = isolation.snapshot([package])
    key = str(package.absolute())
    for path, record in before[key].items():
        if after[key].get(path) != record:
            raise RuntimeError(f"Incremental installation changed a base file: {path}")
    report = {"complete": True, "base_files_unchanged": True, "results": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    validate(args.wheel.resolve(), args.run.resolve(), args.output.resolve(), args.device)
