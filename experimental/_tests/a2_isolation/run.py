"""Run real-wheel isolation cases on A2; report missing prerequisites as BLOCKED."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MUTATIONS = {"manifest_changed", "corrupt_host", "missing_abi", "wrong_launcher",
             "wrong_stamp", "wrong_selection", "escaping_link", "changed_python"}
RESET = {"ASCEND_CUSTOM_OPP_PATH", "FLA_NPU_DISABLE_PTH", "FLA_NPU_EXPERIMENTAL_OPP_PATH",
         "FLA_NPU_EXPERIMENTAL_OP_API_LIB", "FLA_NPU_OPP_PATH", "FLA_NPU_OP_API_LIB",
         "FLA_NPU_OP_API_LIBS", "FLA_NPU_OVERLAY_VENDOR", "FLA_NPU_STABLE_LIB", "PYTHONPATH"}


def snapshot(roots):
    result = {}
    for root in sorted(set(map(str, roots))):
        path = Path(root).absolute()
        entries = {}
        for p in [path, *sorted(path.rglob("*"))]:
            if "__pycache__" in p.relative_to(path).parts or p.suffix == ".pyc":
                continue
            key = p.relative_to(path).as_posix()
            if p.is_symlink():
                entries[key] = {"symlink": os.readlink(p)}
            elif p.is_file():
                entries[key] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            elif p.is_dir():
                entries[key] = {"directory": True}
            else:
                entries[key] = {"missing_or_special": True}
        result[str(path)] = entries
    return result


def environment(case, profile):
    env = {k: v for k, v in os.environ.items() if k not in RESET}
    # Keep only toolkit-owned Python dependencies (acl/tbe), never source-tree packages.
    cann = os.environ.get("ASCEND_HOME_PATH")
    pythonpath = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and cann
                  and Path(p).resolve().is_relative_to(Path(cann).resolve())]
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # Native-first cases must load exactly the requested library, not torch's NPU autoload.
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONNOUSERSITE="1", FLA_NPU_DISABLE_PTH="1")
    stable = Path(profile["stable_package"]) / "opp/vendors/fla_npu_transformer" if profile.get("stable_package") else None
    if case["mode"] == "experimental":
        opp = Path(profile["experimental_package"]) / "opp"
        vendor = opp / "vendors/fla_npu_experimental_transformer"
        env.update(FLA_NPU_EXPERIMENTAL_OPP_PATH=str(opp), ASCEND_CUSTOM_OPP_PATH=str(vendor))
    if case.get("pth") == "enabled":
        env.pop("FLA_NPU_DISABLE_PTH")
    if case["scenario"] == "mixed_path":
        parts = [str(stable), str(vendor)] if case.get("stable_first") else [str(vendor), str(stable)]
        env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(parts)
    if case["scenario"] == "stable_env_ignored":
        env.update(FLA_NPU_OP_API_LIB=str(stable / "op_api/lib/libcust_opapi.so"),
                   FLA_NPU_OP_API_LIBS=str(stable / "op_api/lib/libcust_opapi.so"),
                   FLA_NPU_OPP_PATH=str(stable.parent.parent))
    if profile.get("mutation_root"):
        env["PYTHONPATH"] = os.pathsep.join([str(Path(profile["experimental_package"]).parent), *pythonpath])
    if profile.get("overlay_vendor"):
        env["FLA_NPU_OVERLAY_VENDOR"] = Path(profile["overlay_vendor"]).name
    return env


def evaluate(case, result, returncode):
    if returncode != 0 or result.get("stage") != "exercise":
        return "FAIL"
    error = result.get("error")
    pattern = case.get("error")
    if pattern:
        accepted = {"RuntimeError", "ImportError", "FileNotFoundError", "ValueError"}
        return "PASS" if error and error.get("type") in accepted and re.search(pattern, error.get("message", "")) else "FAIL"
    return "FAIL" if error else "PASS"


def run_case(case, config, output):
    case_dir = output / case["id"]
    case_dir.mkdir()
    profile = dict(config["profiles"].get(case["profile"], {}))
    if not profile:
        return {"id": case["id"], "status": "BLOCKED", "reason": f"Missing profile: {case['profile']}"}
    required = {"python"} | set(case.get("requires", []))
    if case["mode"] == "experimental":
        required.add("experimental_package")
    missing = [key for key in required if not profile.get(key) or not Path(profile[key]).is_absolute()
               or not Path(profile[key]).exists()]
    if missing:
        return {"id": case["id"], "status": "BLOCKED", "reason": f"Missing/invalid absolute paths: {missing}"}
    if case["scenario"] in MUTATIONS:
        scratch = case_dir / "mutation"
        scratch.mkdir()
        copied = scratch / "fla_npu_experimental"
        shutil.copytree(profile["experimental_package"], copied, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        profile.update(experimental_package=str(copied), mutation_root=str(scratch))
    request = case_dir / "request.json"
    result_file = case_dir / "observation.json"
    request.write_text(json.dumps({**case, "profile": profile}, indent=2), encoding="utf-8")
    command = [profile["python"], str(HERE / "probe.py"), str(request), str(result_file)]
    try:
        done = subprocess.run(command, env=environment(case, profile), capture_output=True, text=True,
                              timeout=int(config.get("timeout", 120)))
        (case_dir / "stdout.log").write_text(done.stdout, encoding="utf-8")
        (case_dir / "stderr.log").write_text(done.stderr, encoding="utf-8")
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        return {"id": case["id"], "status": evaluate(case, result, done.returncode),
                "returncode": done.returncode, "observation": result}
    except subprocess.TimeoutExpired as exc:
        for name, data in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            (case_dir / (name + ".log")).write_bytes(data if isinstance(data, bytes) else (data or "").encode())
        return {"id": case["id"], "status": "FAIL", "reason": "Process timed out; a timeout is not expected rejection"}
    except (OSError, ValueError) as exc:
        return {"id": case["id"], "status": "FAIL", "reason": str(exc)}


def preflight(config, output):
    profile = config["profiles"]["combined"]
    command = [profile["python"], "-c", "import json,torch,torch_npu; "
               "print(json.dumps({'device':torch.npu.get_device_name(" + str(int(config["device"])) + ")}))"]
    done = subprocess.run(command, env=environment({"mode": "stable", "scenario": "stable"}, profile),
                          capture_output=True, text=True, timeout=60)
    (output / "device-check.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode or not re.search(r'"device"\s*:\s*"[^"\n]*910B', done.stdout, re.I):
        raise RuntimeError("A2 device identity was not confirmed; see device-check.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protected = list(config.get("protected_roots", []))
    protected += [p["stable_package"] for p in config["profiles"].values() if p.get("stable_package")]
    # Resolve once: a later replacement of the root symlink must not change the baseline target.
    protected = [str(Path(p).resolve()) for p in protected]
    before = snapshot(protected)
    (output / "protected-before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    results = []
    try:
        invalid = [p for p in protected if not Path(p).is_dir()]
        if invalid:
            raise RuntimeError(f"Protected directories do not exist: {invalid}")
        preflight(config, output)
        for case in cases:
            result = run_case(case, config, output)
            results.append(result)
            print(result["id"], result["status"], flush=True)
    except Exception as exc:
        results += [{"id": case["id"], "status": "BLOCKED", "reason": str(exc)}
                    for case in cases[len(results):]]
    finally:
        after = snapshot(protected)
        (output / "protected-after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
        unchanged = before == after
        counts = {state: sum(r["status"] == state for r in results) for state in ("PASS", "FAIL", "BLOCKED")}
        report = {"suite": "A2 load isolation (no operator execution)", "planned": len(cases),
                  "counts": counts, "protected_unchanged": unchanged, "results": results,
                  "complete": len(results) == len(cases) and counts["PASS"] == len(cases) and unchanged}
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["complete"] else (1 if counts["FAIL"] or not unchanged else 2)


if __name__ == "__main__":
    sys.exit(main())
