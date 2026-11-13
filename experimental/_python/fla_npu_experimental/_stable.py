"""Explicit experimental Stable launcher; no ctypes descriptor execution."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from . import _loader

_LIB = None
_OPS = None


def _namespace():
    import sys
    return getattr(sys.modules[__package__], "_NAMESPACE", "fla_npu_experimental_stable")


def _package_contract(directory, libraries):
    build = json.loads((directory / "_build.json").read_text(encoding="utf-8"))
    from ._op_abi import validated_contracts
    from ._layers import manager
    manifest = manager.manifest(directory)
    active_ops = set(manifest["exports"])
    plan = _loader._PLAN
    if plan and plan.get("layer") and directory.resolve() != (Path(plan["layer"]) / manager.PACKAGE).resolve():
        # Covered exports use the overlay's adapter and contract, not the base's.
        active_ops.difference_update(plan["exports"])
    vendors = [Path(p).parents[2] for p in libraries]
    contracts = [validated_contracts(vendor, require_manifest=True) for vendor in vendors]
    for name in sorted(active_ops):
        op = manifest["operators"][name]
        for symbol, expected in op["requires"].items():
            actual = next((c[symbol] for c in contracts if symbol in c), None)
            if actual is None or tuple(actual) != tuple(expected):
                raise RuntimeError(f"Experimental ACLNN contract mismatch: {symbol}")
    if manifest["source_hash"] != build["source_hash"]:
        raise RuntimeError("Experimental layer source fingerprint mismatch")
    library = _loader._inside(directory / "libfla_npu_experimental_stable.so", directory, "launcher")
    if hashlib.sha256(library.read_bytes()).hexdigest() != build["launcher_sha256"]:
        raise RuntimeError("Experimental launcher hash mismatch")
    return build, library, active_ops


def load():
    global _LIB, _OPS
    _loader.validate_active_environment()
    if _LIB is not None:
        return
    directory = Path(__file__).resolve().parent
    build, library, expected = _package_contract(directory, _loader._LIBRARIES)
    import torch
    import torch_npu
    # This library registers only its experimental namespace; it is not a CPython extension.
    torch.ops.load_library(str(library))
    probe = ctypes.CDLL(str(library))
    stamp = probe.fla_npu_experimental_source_hash
    stamp.restype = ctypes.c_char_p
    if stamp().decode() != build["source_hash"]:
        raise RuntimeError("Experimental launcher source fingerprint mismatch")
    register = probe.fla_npu_experimental_register
    register.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    register.restype = ctypes.c_char_p
    error = register(_namespace().encode(), os.pathsep.join(_loader._LIBRARIES).encode(),
                     ",".join(sorted(expected)).encode())
    if error:
        raise RuntimeError("Experimental registration failed: " + error.decode("utf-8", "replace"))
    for name in expected:
        if not hasattr(getattr(torch.ops, _namespace()), name):
            raise RuntimeError(f"Experimental launcher is missing {name}")
    _OPS = expected
    _LIB = probe


def invoke(name, reference, arguments):
    load()
    if name not in _OPS:
        raise RuntimeError(f"Operator {name} was not selected in this experimental build")
    import torch
    import torch_npu
    if reference.device.type != "npu":
        raise RuntimeError("Experimental reference tensor must be on NPU")
    for value in arguments:
        if not isinstance(value, torch.Tensor):
            continue
        if value.device != reference.device:
            raise RuntimeError("All experimental data tensor arguments must be on the same NPU device")
    with torch.npu.device(reference.device):
        # Match main's queue capability: only a queued launch can use NoWait.
        probe = _LIB.fla_npu_experimental_queue_enqueue_available
        probe.restype = ctypes.c_int32
        accessor = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None) if probe() else None
        accessor = accessor or getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
        if accessor is None:
            raise RuntimeError("torch_npu does not expose a safe current raw stream accessor")
        stream = int(accessor(reference.device.index))
        return getattr(getattr(torch.ops, _namespace()), name)(*arguments, stream)
