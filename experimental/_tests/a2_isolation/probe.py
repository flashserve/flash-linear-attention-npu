"""One fresh process per load-order experiment. Never execute an NPU operator."""
from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

MUTATIONS = {"manifest_changed", "corrupt_host", "missing_abi", "wrong_launcher",
             "wrong_stamp", "wrong_selection", "escaping_link", "changed_python"}


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


def load_stable(profile):
    checked_import("fla_npu", profile["stable_package"])
    from fla_npu.ops.ascendc import _stable
    _stable._op("npu_chunk_fwd_o")
    state = observe()
    if "fla_npu_experimental" in state["modules"] or any(
            n.startswith("fla_npu_experimental_stable::") for n in state["dispatcher"]):
        raise AssertionError(f"Experimental registration leaked into stable process: {state}")
    if "fla_npu_stable::npu_chunk_fwd_o" not in state["dispatcher"]:
        raise AssertionError("Stable control operator is missing")
    allowed = (Path(profile["stable_package"]) / "opp/vendors").resolve()
    if any(Path(p).name == "libcust_opapi.so" and not Path(p).is_relative_to(allowed)
           for p in state["libraries"]):
        raise AssertionError(f"Stable process loaded an external opapi: {state}")
    if profile.get("overlay_vendor"):
        overlay_lib = str((Path(profile["overlay_vendor"]) / "op_api/lib/libcust_opapi.so").resolve())
        ordered = os.getenv("FLA_NPU_OP_API_LIBS", "").split(os.pathsep)
        if not ordered or ordered[0] != overlay_lib or overlay_lib not in state["libraries"]:
            raise AssertionError("The configured overlay was not actually loaded first")


def experimental_environment(profile):
    opp = Path(profile["experimental_package"]) / "opp"
    os.environ.update(FLA_NPU_DISABLE_PTH="1", FLA_NPU_EXPERIMENTAL_OPP_PATH=str(opp),
                      ASCEND_CUSTOM_OPP_PATH=str(opp / "vendors/fla_npu_experimental_transformer"))


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
    if scenario in {"stable", "overlay"}:
        target = lambda: load_stable(profile)
    elif scenario == "late_disable_pth":
        configured = {Path(p).resolve() for p in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p}
        if stable_vendor.resolve() not in configured:
            raise AssertionError("The real stable .pth did not run; this case was not exercised")
        os.environ["FLA_NPU_DISABLE_PTH"] = "1"
    elif scenario == "stable_then_experimental":
        load_stable(profile)
        experimental_environment(profile)
    elif scenario == "experimental_then_stable":
        load_experimental(profile)
        target = lambda: checked_import("fla_npu", stable)
    elif scenario == "torch_npu_first":
        importlib.import_module("torch_npu")
    elif scenario == "native_launcher_first":
        import torch
        torch.ops.load_library(str(stable / "libfla_npu_stable.so"))
        if not any(Path(p).name == "libfla_npu_stable.so" for p in mapped_libraries()):
            raise AssertionError("Native stable launcher was not actually loaded")
    elif scenario == "native_opapi_first":
        ctypes.CDLL(str(stable_vendor / "op_api/lib/libcust_opapi.so"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    elif scenario == "native_launcher_after":
        load_experimental(profile)
        import torch
        torch.ops.load_library(str(stable / "libfla_npu_stable.so"))
        from fla_npu_experimental import _loader
        target = _loader.validate_active_environment
    elif scenario in {"environment_changed", "manifest_changed"}:
        load_experimental(profile)
        if scenario == "environment_changed":
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(stable_vendor)
        else:
            path = vendor / "experimental-build-manifest.json"
            path.write_text(path.read_text() + "\n ", encoding="utf-8")
        from fla_npu_experimental import _loader
        target = _loader.validate_active_environment
    elif scenario == "external_opp":
        os.environ["FLA_NPU_EXPERIMENTAL_OPP_PATH"] = profile["external_experimental_opp"]
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
    elif scenario in {"wrong_stamp", "wrong_selection"}:
        path = exp / "_build.json"
        data = json.loads(path.read_text())
        data["source_hash" if scenario == "wrong_stamp" else "operators"] = "0" * 64 if scenario == "wrong_stamp" else []
        path.write_text(json.dumps(data), encoding="utf-8")
    elif scenario == "escaping_link":
        path = vendor / "op_api/lib/libcust_opapi.so"
        path.unlink()
        path.symlink_to(stable_vendor / "op_api/lib/libcust_opapi.so")
    elif scenario == "changed_python":
        path = exp / "_contract.py"
        path.write_text(path.read_text() + "\n# changed after build\n", encoding="utf-8")
    elif scenario not in {"experimental", "missing_disable", "mixed_path", "stable_env_ignored"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    result["before_exercise"] = observe()
    result["stage"] = "exercise"
    target()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    result = {"case_id": request["id"], "stage": "setup", "operator_executed": False}
    try:
        if request.get("pth") == "enabled" and request["mode"] == "stable":
            expected = (Path(request["profile"]["stable_package"]) / "opp/vendors/fla_npu_transformer").resolve()
            actual = {Path(p).resolve() for p in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p}
            if expected not in actual:
                raise AssertionError("The real stable .pth did not run before package import")
        if request.get("absent") and importlib.util.find_spec(request["absent"]) is not None:
            raise AssertionError(f"Isolated profile unexpectedly contains {request['absent']}")
        exercise(request["scenario"], request["profile"], result)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["after"] = observe()
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
