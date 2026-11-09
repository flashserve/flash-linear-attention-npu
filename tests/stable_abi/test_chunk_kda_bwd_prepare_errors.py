#!/usr/bin/env python3
"""Check ACLNN return codes, structured Error Messages and public propagation.

The raw two-phase API is used only to exercise errors hidden by Python shape
inference (including bad output buffers). Rejected calls must not produce an
executor or launch a kernel.
"""
import argparse
import ctypes
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atk" / "common"))
import aclnn_error_probe as _raw_abi


def has_error_code(message, code):
    # CANN 9.0.0 only has the legacy attribute/dtype entries in its error catalog.
    legacy = {"EZ0008": "EZ0002", "EZ0019": "EZ0007", "EZ0024": "EZ0002"}
    return code in message or (code in legacy and legacy[code] in message)


def fixture():
    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="npu")
    return dict(aqk=empty((1, 2, 3, 64)), v_new=empty((1, 2, 3, 128)),
                d_o=empty((1, 2, 3, 128)), h=empty((1, 1, 2, 128, 128)),
                d_aqk=empty((1, 2, 3, 64), torch.float32), dv=empty((1, 2, 3, 128)),
                dq_raw=empty((1, 2, 3, 128), torch.float32), cu=None, indices=None,
                layout="BNSD", scale=0.125, chunk_size=64)


def check_raw(name, updates, code, parameter, *, null_output_pointer=None, status=161002):
    case = fixture()
    case.update(updates)
    runtime = _raw_abi.runtime()
    context = _raw_abi._CallContext(runtime, case["v_new"].device)
    api = runtime.symbol("aclnnChunkKdaBwdPrepareGetWorkspaceSize")
    api.argtypes = ([ctypes.c_void_p] * 6 +
                    [ctypes.c_char_p, ctypes.c_double, ctypes.c_int64, ctypes.c_bool] +
                    [ctypes.c_void_p] * 3 +
                    [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    api.restype = ctypes.c_int
    workspace, executor = ctypes.c_uint64(0), ctypes.c_void_p()
    try:
        def tensor(key):
            value = case[key]
            return context.tensor(value, key, acl_format_override=2,
                storage_shape_override=tuple(value.shape)) if value is not None else None
        args = [tensor(key) for key in ("aqk", "v_new", "d_o", "h")]
        args += [context.int_array(case["cu"]), context.int_array(case["indices"]),
                 case["layout"].encode() if case["layout"] is not None else None,
                 case["scale"], case["chunk_size"], False]
        args += [tensor(key) for key in ("d_aqk", "dv", "dq_raw")]
        args += [None if null_output_pointer == "workspaceSize" else ctypes.byref(workspace),
                 None if null_output_pointer == "executor" else ctypes.byref(executor)]
        runtime._recent_error_message()  # discard earlier errors before this call
        result = api(*args)
        message = runtime._recent_error_message()
        assert result == status, (name, result, status, message)
        assert has_error_code(message, code) and parameter in message and "aclnnChunkKdaBwdPrepare" in message, (name, message)
        assert not executor.value and workspace.value == 0, (name, workspace.value, executor.value)
        print(f"PASS {name}: aclnnStatus={result}; {message.strip()}", flush=True)
    finally:
        context.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="npu")
    cases = [
        ("fp16_removed", {"aqk": empty((1, 2, 3, 64), torch.float16)}, "EZ0019", "aqk"),
        ("state_shape", {"h": empty((1, 1, 2, 64, 128))}, "EZ0008", "h"),
        ("state_rank", {"h": empty((1, 2, 128, 128))}, "EZ0008", "h"),
        ("input_dtype", {"aqk": empty((1, 2, 3, 64), torch.float32)}, "EZ0019", "aqk"),
        ("output_dtype", {"dq_raw": empty((1, 2, 3, 128))}, "EZ0019", "dq_raw"),
        ("token_rank", {"aqk": empty((2, 3, 64))}, "EZ0008", "aqk"),
        ("token_shape", {"v_new": empty((1, 3, 3, 128))}, "EZ0008", "v_new"),
        ("output_layout", {"dv": empty((1, 3, 2, 128))}, "EZ0008", "dv"),
        ("zero_tokens", {"aqk": empty((1, 2, 0, 64))}, "EZ0008", "aqk"),
        ("aqk_width", {"aqk": empty((1, 2, 3, 32))}, "EZ0024", "aqk.D"),
        ("chunk_size", {"chunk_size": 32}, "EZ0024", "chunk_size"),
        ("scale_nan", {"scale": float("nan")}, "EZ0024", "scale"),
        ("scale_inf", {"scale": float("inf")}, "EZ0024", "scale"),
        ("scale_fp32_overflow", {"scale": 1e100}, "EZ0024", "scale"),
        ("layout", {"layout": "BAD"}, "EZ0024", "layout"),
        ("metadata_pair", {"indices": [0, 0]}, "EZ0024", "chunk_indices"),
        ("variable_batch", {
            "aqk": empty((2, 2, 3, 64)), "v_new": empty((2, 2, 3, 128)),
            "d_o": empty((2, 2, 3, 128)), "cu": [0, 3], "indices": [0, 0]},
         "EZ0024", "B"),
        ("layout_rank_mismatch", {"layout": "NTD"}, "EZ0008", "aqk"),
        ("state_chunks", {"h": empty((1, 2, 2, 128, 128))}, "EZ0024", "h.chunks"),
        ("strided_input", {"d_o": empty((1, 2, 3, 256))[..., ::2]}, "EZ0024", "d_o"),
    ]
    for case in cases:
        check_raw(*case)
    packed = dict(aqk=empty((2, 3, 64)), v_new=empty((2, 3, 128)),
                  d_o=empty((2, 3, 128)), h=empty((1, 2, 128, 128)),
                  d_aqk=empty((2, 3, 64), torch.float32), dv=empty((2, 3, 128)),
                  dq_raw=empty((2, 3, 128), torch.float32), layout="NTD", cu=[0, 3], indices=[0, 0])
    metadata = [("cu_size", {"cu": [0]}, "EZ0024", "cu_seqlens.size"),
                ("cu_endpoint", {"cu": [1, 3]}, "EZ0024", "cu_seqlens"),
                ("cu_order", {"cu": [0, 2, 1, 3]}, "EZ0024", "cu_seqlens"),
                ("index_size", {"indices": [0]}, "EZ0024", "chunk_indices.size"),
                ("index_order", {"indices": [0, 1]}, "EZ0024", "chunk_indices"),
                ("packed_chunks", {"h": empty((2, 2, 128, 128))}, "EZ0024", "h.total_chunks")]
    for name, updates, code, parameter in metadata:
        check_raw(name, {**packed, **updates}, code, parameter)
    for key in ("aqk", "layout"):
        check_raw("missing_" + key, {key: None}, "EZ0004", key, status=161001)
    for key in ("workspaceSize", "executor"):
        check_raw("missing_" + key, {}, "EZ0004", key, null_output_pointer=key, status=161001)
    case = fixture()
    case["h"] = empty((1, 1, 2, 64, 128))
    try:
        ascendc.chunk_kda_bwd_prepare(case["aqk"], case["v_new"], case["d_o"], case["h"],
                                    scale=0.125, layout="BNSD")
    except RuntimeError as error:
        message = str(error)
        assert "161002" in message and has_error_code(message, "EZ0008") and "h" in message, message
        assert "128" in message and "64" in message, message
        print("PASS public Stable-ABI exception contains aclnnStatus, structured code and shape details", flush=True)
    else:
        raise AssertionError("invalid state shape was accepted")
    assert ascendc.BACKENDS["npu_chunk_kda_bwd_prepare"] == "stable", ascendc.BACKENDS
    assert not ascendc.FALLBACKS.get("npu_chunk_kda_bwd_prepare", 0), ascendc.FALLBACKS
    print(f"PASS {len(cases) + len(metadata) + 4} raw ACLNN errors and public Stable-ABI propagation")


if __name__ == "__main__":
    main()
