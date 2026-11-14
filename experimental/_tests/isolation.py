"""Run real-wheel isolation cases on A2; report missing prerequisites as BLOCKED."""
from __future__ import annotations

import ctypes
import importlib
import importlib.util
import types
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CASES = [
    {'id': 'L01', 'scenario': 'experimental'},
    {'id': 'L07', 'scenario': 'missing_disable', 'pth': 'enabled', 'error': 'FLA_NPU_DISABLE_PTH'},
    {'id': 'L09', 'scenario': 'mixed_path', 'stable_first': True, 'error': 'only the selected experimental'},
    {'id': 'L10', 'scenario': 'mixed_path', 'stable_first': False, 'error': 'only the selected experimental'},
    {'id': 'L11', 'scenario': 'stable_env_ignored'},
    {'id': 'L12', 'scenario': 'stable_then_experimental', 'error': 'separate Python processes'},
    {'id': 'L13', 'scenario': 'experimental_then_stable', 'error': 'cannot be imported after'},
    {'id': 'L14', 'scenario': 'torch_npu_first', 'error': 'before torch_npu/acl'},
    {'id': 'L15', 'scenario': 'native_launcher_first', 'error': 'Stable launcher is already loaded'},
    {'id': 'L16', 'scenario': 'native_opapi_first', 'error': 'different custom OPP|CANN is already loaded'},
    {'id': 'L18', 'scenario': 'environment_changed', 'error': 'environment changed'},
    {'id': 'L20', 'scenario': 'corrupt_host', 'error': 'ABI manifest'},
    {'id': 'L21', 'scenario': 'missing_abi', 'error': 'Missing .*abi_manifest'},
    {'id': 'L22', 'scenario': 'wrong_launcher', 'error': 'launcher hash mismatch'},
    {'id': 'L23', 'scenario': 'wrong_stamp', 'error': 'source fingerprint mismatch'},
    {'id': 'L25', 'scenario': 'unselected_operator', 'error': 'not selected'},
    {'id': 'L27', 'scenario': 'escaping_link', 'error': 'escapes the experimental OPP root'},
]

MUTATIONS = {"corrupt_host", "missing_abi", "wrong_launcher", "wrong_stamp", "escaping_link"}
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
    profile = dict(config)
    required = {"python", "experimental_package", "stable_package", "external_experimental_opp"}
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
    command = [profile["python"], str(Path(__file__).resolve()), "--probe", str(request), str(result_file)]
    try:
        done = subprocess.run(command, env=environment({"mode": "experimental", **case}, profile), capture_output=True, text=True,
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
    profile = config
    command = [profile["python"], "-c", "import json,torch,torch_npu; "
               "print(json.dumps({'device':torch.npu.get_device_name(" + str(int(config["device"])) + ")}))"]
    done = subprocess.run(command, env=environment({"mode": "stable", "scenario": "stable"}, profile),
                          capture_output=True, text=True, timeout=60)
    (output / "device-check.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode or not re.search(r'"device"\s*:\s*"[^"\n]*910B', done.stdout, re.I):
        raise RuntimeError("A2 device identity was not confirmed; see device-check.log")


def mapped_libraries():
    result = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)"))
            if path.name.startswith(("libcust_", "libfla_npu_", "libopapi", "libascendcl")):
                result.add(str(path.resolve()))
    return sorted(result)


def observe():
    torch = sys.modules.get("torch")
    names = sorted(n for n in torch._C._dispatch_get_all_op_names()
                   if n.startswith(("fla_npu_stable::", "fla_npu_experimental_stable::"))) if torch else []
    return {"libraries": mapped_libraries(), "dispatcher": names,
            "modules": sorted(n for n in sys.modules if n in {"fla_npu", "fla_npu_experimental", "torch_npu"}),
            "custom_opp_path": os.getenv("ASCEND_CUSTOM_OPP_PATH", ""),
            "npu_initialized": bool(torch and hasattr(torch, "npu") and torch.npu.is_initialized())}


def checked_import(name, expected):
    module = importlib.import_module(name)
    actual = Path(module.__file__).resolve().parent
    if actual != Path(expected).resolve():
        raise AssertionError(f"Imported the wrong {name}: {actual} != {expected}")
    return module


def load_experimental(profile):
    checked_import("fla_npu_experimental", profile["experimental_package"])
    from fla_npu_experimental import _stable
    _stable.load()
    state = observe()
    declared = json.loads((Path(profile["experimental_package"]) / "_build.json").read_text())["operators"]
    expected = {"fla_npu_experimental_stable::npu_" + op["name"].replace("-", "_").replace(".", "_")
                for op in declared}
    if set(state["dispatcher"]) != expected or "fla_npu" in state["modules"]:
        raise AssertionError(f"Unexpected experimental registration: {state}")
    own = Path(profile["experimental_package"]) / "opp/vendors/fla_npu_experimental_transformer/op_api/lib/libcust_opapi.so"
    foreign = [p for p in state["libraries"] if Path(p).name == "libcust_opapi.so" and Path(p) != own.resolve()]
    if foreign or state["npu_initialized"]:
        raise AssertionError(f"Foreign library or unexpected NPU initialization: {state}")


def exercise(scenario, profile, result):
    exp = Path(profile["experimental_package"]) if profile.get("experimental_package") else None
    stable = Path(profile["stable_package"]) if profile.get("stable_package") else None
    vendor = exp / "opp/vendors/fla_npu_experimental_transformer" if exp else None
    stable_vendor = stable / "opp/vendors/fla_npu_transformer" if stable else None
    if scenario in MUTATIONS:
        root = Path(profile["mutation_root"]).resolve(strict=True)
        if exp is None or not exp.resolve().is_relative_to(root) or exp.resolve() == root:
            raise ValueError("Mutation requires a runner-owned package copy")
        if any(p.is_symlink() and not p.resolve().is_relative_to(exp.resolve()) for p in exp.rglob("*")):
            raise ValueError("Mutation copy contains an external symlink")
    target = lambda: load_experimental(profile)
    if scenario == "stable_then_experimental":
        # Simulate an already imported formal module; no formal wheel is required.
        sys.modules["fla_npu"] = types.ModuleType("fla_npu")
    elif scenario == "experimental_then_stable":
        load_experimental(profile)
        target = lambda: checked_import("fla_npu", stable)
    elif scenario == "torch_npu_first":
        importlib.import_module("torch_npu")
    elif scenario == "native_launcher_first":
        ctypes.CDLL(str(stable / "libfla_npu_stable.so"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        if not any(Path(p).name == "libfla_npu_stable.so" for p in mapped_libraries()):
            raise AssertionError("Native stable launcher was not actually loaded")
    elif scenario == "native_opapi_first":
        ctypes.CDLL(str(stable_vendor / "op_api/lib/libcust_opapi.so"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    elif scenario == "environment_changed":
        load_experimental(profile)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(stable_vendor)
        from fla_npu_experimental import _loader
        target = _loader.validate_active_environment
    elif scenario == "unselected_operator":
        load_experimental(profile)
        from fla_npu_experimental import _stable
        absent = "npu_unselected_fixture"
        if absent in _stable._OPS:
            raise AssertionError("The rejection fixture must not be registered")
        target = lambda: _stable.invoke(absent, None, ())
    elif scenario == "corrupt_host":
        path = vendor / "op_api/lib/libcust_opapi.so"
        path.write_bytes(path.read_bytes() + b"invalid")
    elif scenario == "missing_abi":
        (vendor / "abi_manifest.json").unlink()
    elif scenario == "wrong_launcher":
        path = exp / "libfla_npu_experimental_stable.so"
        path.write_bytes((stable / "libfla_npu_stable.so").read_bytes())
    elif scenario == "wrong_stamp":
        path = exp / "_build.json"
        data = json.loads(path.read_text())
        data["source_hash"] = "0" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
    elif scenario == "escaping_link":
        path = vendor / "op_api/lib/libcust_opapi.so"
        path.unlink()
        path.symlink_to(stable_vendor / "op_api/lib/libcust_opapi.so")
    elif scenario not in {"experimental", "missing_disable", "mixed_path", "stable_env_ignored"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    result["before_exercise"] = observe()
    result["stage"] = "exercise"
    target()


def probe_main(request_file, result_file):
    request = json.loads(Path(request_file).read_text())
    result = {"case_id": request["id"], "stage": "setup", "operator_executed": False}
    try:
        if importlib.util.find_spec("fla_npu") is not None:
            raise AssertionError("The experimental-only environment contains a formal package")
        exercise(request["scenario"], request["profile"], result)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["after"] = observe()
    Path(result_file).write_text(json.dumps(result, indent=2), encoding="utf-8")


def validate(config, output):
    output.mkdir(parents=True, exist_ok=False)
    protected = [config["stable_package"]]
    before = snapshot(protected)
    results = []
    try:
        preflight(config, output)
        for case in CASES:
            result = run_case(case, config, output)
            results.append(result)
            print(result["id"], result["status"], flush=True)
    except Exception as exc:
        results += [{"id": case["id"], "status": "BLOCKED", "reason": str(exc)}
                    for case in CASES[len(results):]]
    unchanged = before == snapshot(protected)
    counts = {state: sum(r["status"] == state for r in results) for state in ("PASS", "FAIL", "BLOCKED")}
    report = {"suite": "Experimental load isolation (no operator execution)", "planned": len(CASES),
              "counts": counts, "protected_unchanged": unchanged, "results": results,
              "complete": len(results) == len(CASES) and counts["PASS"] == len(CASES) and unchanged}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--probe":
        raise SystemExit("Use experimental/_tests/validate.py to run this suite")
    probe_main(*sys.argv[2:])
