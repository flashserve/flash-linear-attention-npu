"""Bind public operators to an isolated wheel's Python and Stable adapter."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def install_exports(namespace):
    import fla_npu

    plan = getattr(fla_npu, "_WHEEL_LAYER_PLAN", None)
    if not plan or not plan["exports"]:
        return
    layer = Path(plan["layer"])
    name = "_fla_npu_layer_" + layer.name
    if name not in sys.modules:
        # Do not execute the incoming wheel's root import/bootstrap. The base
        # controller owns CANN initialization and resolves the active stack.
        spec = importlib.util.spec_from_file_location(
            name, layer / "fla_npu/__init__.py",
            submodule_search_locations=[str(layer / "fla_npu")])
        package = importlib.util.module_from_spec(spec)
        package._WHEEL_LAYER_CONTEXT = {
            "namespace": "fla_npu_layer_" + layer.name,
            "libraries": plan["bindings"]["overlay"],
        }
        sys.modules[name] = package
    provider = importlib.import_module(name + ".ops.ascendc")
    provider.BACKENDS = namespace.setdefault("BACKENDS", {})
    for op in plan["exports"]:
        if op not in provider._ASCENDC_OPS:
            raise RuntimeError(f"Wheel declares {op} but has no public wrapper")
        alias = op.removeprefix("npu_")
        namespace[op] = getattr(provider, op)
        namespace[alias] = getattr(provider, alias)
        for field in ("BACKWARD_OPS", "MUTATED_ARGUMENTS", "MUTATION_FLAGS", "MUTATION_PREDICATES"):
            for key in (op, alias):
                namespace[field].pop(key, None)
                if key in getattr(provider, field):
                    namespace[field][key] = getattr(provider, field)[key]
        for key in (op, alias):
            if key not in namespace["__all__"]:
                namespace["__all__"].append(key)
    namespace["_ASCENDC_OPS"] = tuple(dict.fromkeys((*namespace["_ASCENDC_OPS"], *plan["exports"])))
    selected = set(plan["exports"])
    namespace["_TORCH_NPU_COMPAT_OPS"] = tuple(dict.fromkeys(
        [op for op in namespace.get("_TORCH_NPU_COMPAT_OPS", ()) if op not in selected]
        + [op for op in getattr(provider, "_TORCH_NPU_COMPAT_OPS", ()) if op in selected]))
