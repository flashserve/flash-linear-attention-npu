"""Explicit experimental Stable launcher; no ctypes descriptor execution."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from . import _loader

_LIB = None
_OPS = None


def _package_contract(directory, vendor):
    build = json.loads((directory / "_build.json").read_text(encoding="utf-8"))
    actual_python = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.py")}
    if actual_python != build["python_sha256"]:
        raise RuntimeError("Experimental Python source differs from the built launcher snapshot")
    from ._op_abi import validated_contracts
    validated_contracts(vendor, require_manifest=True)
    manifest = json.loads((vendor / _loader.MANIFEST).read_text(encoding="utf-8"))
    if manifest["operators"] != build["operators"]:
        raise RuntimeError("Experimental launcher and OPP operator snapshots differ")
    library = _loader._inside(directory / "libfla_npu_experimental_stable.so", directory, "launcher")
    if hashlib.sha256(library.read_bytes()).hexdigest() != build["launcher_sha256"]:
        raise RuntimeError("Experimental launcher hash mismatch")
    return build, library


def load():
    global _LIB, _OPS
    _loader.validate_active_environment()
    if _LIB is not None:
        return
    directory = Path(__file__).resolve().parent
    root, vendor, custom = _loader._CONFIGURATION
    build, library = _package_contract(directory, vendor)
    os.environ["FLA_NPU_EXPERIMENTAL_OP_API_LIB"] = str(custom)
    import torch
    import torch_npu
    # This library registers only its experimental namespace; it is not a CPython extension.
    torch.ops.load_library(str(library))
    probe = ctypes.CDLL(str(library))
    stamp = probe.fla_npu_experimental_source_hash
    stamp.restype = ctypes.c_char_p
    if stamp().decode() != build["source_hash"]:
        raise RuntimeError("Experimental launcher source fingerprint mismatch")
    expected = {"npu_" + op["name"].replace("-", "_").replace(".", "_") for op in build["operators"]}
    for name in expected:
        if not hasattr(torch.ops.fla_npu_experimental_stable, name):
            raise RuntimeError(f"Experimental launcher is missing {name}")
    _OPS = expected
    _LIB = probe


def invoke(name, reference, arguments):
    load()
    if name not in _OPS:
        raise RuntimeError(f"Operator {name} was not selected in this experimental build")
    if os.environ.get("FLA_NPU_EXPERIMENTAL_OP_API_LIB") != str(_loader._CONFIGURATION[2]):
        raise RuntimeError("Experimental host library selection changed; restart the process")
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
        return getattr(torch.ops.fla_npu_experimental_stable, name)(*arguments, stream)
