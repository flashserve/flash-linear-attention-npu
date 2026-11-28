"""Python wrappers and Stable dispatcher for experimental operators."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from . import _loader
from ._contract import validate_chunk_len

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


def _shape(tensor):
    return tuple(tensor.shape)

def _optional_bool(value, default):
    return default if value is None else bool(value)

def ensure_npu_tensor(tensor, name):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu":
        raise RuntimeError(f"{name} must be an NPU tensor")

def fused_recurrent_rwkv8(
    q,
    w,
    k,
    v,
    z,
    b,
    *,
    scale=1.0,
    initial_state=None,
    reverse=False,
    output_chunk_state=False,
    output_sa=False,
    chunk_len=16,
):
    """RWKV-v8（WKV7）fused recurrent 前向，逐 token 递推。

    布局为 BHTC：q/w/k/z/b 是 (B,H,T,K)，v 是 (B,H,T,V)（K 与 V 独立，
    K==V 为特例）。与 fla 上游的 (B,T,H,C) 口径不同，对接时需
    transpose(1,2)——transpose 只改视图，建议随后 .contiguous()，否则
    aclnn 层会对每个非连续输入静默补一次 GM 拷贝。initial_state 朝向
    (B,H,K,V)（= 内核 Sᵀ 原样，与 s 快照一致），fp32，缺省零初态。

    返回 (o, s, sa)：o 为 (B,H,T,V)、dtype 跟随 q；s 为 chunk 快照
    (B,H,T//chunk_len,K,V) fp32（output_chunk_state=True 时产出，
    否则为 None）；sa 为每 token 更新前的 state@z，(B,H,T,V) fp32
    （output_sa=True 时产出，否则为 None）。
    """
    _loader.validate_active_environment()
    import torch

    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    tensors = {"q": q, "w": w, "k": k, "v": v, "z": z, "b": b}
    for name, tensor in tensors.items():
        ensure_npu_tensor(tensor, name)
    if initial_state is not None:
        ensure_npu_tensor(initial_state, "initial_state")
    shapes = {name: _shape(tensor) for name, tensor in tensors.items()}
    for name, tensor in tensors.items():
        if len(shapes[name]) != 4:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} must be rank-4 (B,H,T,C).")
        if tensor.dtype not in allowed_dtypes:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} dtype must be float16/bfloat16/float32.")
        if tensor.dtype != q.dtype:
            raise RuntimeError("npu_fused_recurrent_rwkv8: q/w/k/v/z/b must share the same dtype.")
    q_shape = shapes["q"]
    for name in ("w", "k", "z", "b"):
        if shapes[name] != q_shape:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} shape must equal q shape (B,H,T,K).")
    batch, heads, seqlen, k_dim = q_shape
    v_shape = shapes["v"]
    if v_shape[:3] != (batch, heads, seqlen):
        raise RuntimeError("npu_fused_recurrent_rwkv8: v must be (B,H,T,V) with the same B/H/T as q.")
    v_dim = v_shape[3]
    if k_dim <= 0 or v_dim <= 0 or k_dim % 8 or k_dim > 128 or v_dim % 8 or v_dim > 128:
        raise RuntimeError("npu_fused_recurrent_rwkv8: K/V must be multiples of 8 and <= 128.")

    reverse = _optional_bool(reverse, False)
    output_chunk_state = _optional_bool(output_chunk_state, False)
    output_sa = _optional_bool(output_sa, False)
    chunk_len = validate_chunk_len(chunk_len)

    if initial_state is not None:
        if _shape(initial_state) != (batch, heads, k_dim, v_dim):
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be (B,H,K,V).")
        if initial_state.dtype != torch.float32:
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be float32.")

    return invoke("npu_fused_recurrent_rwkv8", q,
                  (q, w, k, v, z, b, initial_state, float(scale), reverse,
                   output_chunk_state, output_sa, chunk_len))
