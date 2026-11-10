#!/usr/bin/env python3
"""Check ChunkKdaFwd raw ACLNN return codes and structured error details."""

import argparse
import ctypes

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import _runtime


def _empty(shape, dtype=torch.bfloat16, device="npu"):
    return torch.empty(shape, dtype=dtype, device=device)


def fixture(k_dim=128, v_dim=128, seqlen=4):
    chunk_size = 64
    total_chunks = (seqlen + chunk_size - 1) // chunk_size
    return {
        "q": _empty((1, seqlen, 1, k_dim)),
        "k": _empty((1, seqlen, 1, k_dim)),
        "v": _empty((1, seqlen, 1, v_dim)),
        "g": _empty((1, seqlen, 1, k_dim), torch.float32),
        "beta": _empty((1, seqlen, 1), torch.float32),
        "a_log": None,
        "dt_bias": None,
        "initial_state": None,
        "cu_seqlens": None,
        "chunk_indices": None,
        "layout": b"BSND",
        "scale": k_dim ** -0.5,
        "chunk_size": chunk_size,
        "safe_gate": False,
        "lower_bound": -5.0,
        "use_gate_in_kernel": False,
        "state_v_first": False,
        "attn": _empty((1, seqlen, 1, v_dim)),
        "final_state": _empty((1, 1, k_dim, v_dim), torch.float32),
        "gk": _empty((1, 1, seqlen, k_dim), torch.float32),
        "aqk": _empty((1, 1, seqlen, chunk_size)),
        "akk": _empty((1, 1, seqlen, chunk_size)),
        "w": _empty((1, 1, seqlen, k_dim)),
        "u": _empty((1, 1, seqlen, v_dim)),
        "qg": _empty((1, 1, seqlen, k_dim)),
        "kg": _empty((1, 1, seqlen, k_dim)),
        "v_new": _empty((1, 1, seqlen, v_dim)),
        "h": _empty((1, total_chunks, 1, k_dim, v_dim)),
    }


def check_raw(name, updates, code, parameter, *, status=161002, null_output_pointer=None):
    case = fixture()
    case.update(updates)
    runtime = _runtime.runtime()
    context = _runtime._CallContext(runtime, case["attn"].device)
    api = runtime.symbol("aclnnChunkKdaFwdGetWorkspaceSize")
    api.argtypes = (
        [ctypes.c_void_p] * 10
        + [ctypes.c_char_p, ctypes.c_double, ctypes.c_int64, ctypes.c_bool, ctypes.c_double,
           ctypes.c_bool, ctypes.c_bool]
        + [ctypes.c_void_p] * 11
        + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    )
    api.restype = ctypes.c_int
    workspace, executor = ctypes.c_uint64(0), ctypes.c_void_p()

    def tensor(key):
        value = case[key]
        if value is None:
            return None
        return context.tensor(
            value,
            key,
            acl_format_override=2,
            storage_shape_override=tuple(value.shape),
        )

    try:
        args = [
            tensor("q"), tensor("k"), tensor("v"), tensor("g"), tensor("beta"),
            tensor("a_log"), tensor("dt_bias"), tensor("initial_state"),
            context.int_array(case["cu_seqlens"]),
            context.int_array(case["chunk_indices"]),
            case["layout"],
            float(case["scale"]),
            int(case["chunk_size"]),
            bool(case["safe_gate"]),
            float(case["lower_bound"]),
            bool(case["use_gate_in_kernel"]),
            bool(case["state_v_first"]),
            tensor("attn"), tensor("final_state"), tensor("gk"), tensor("aqk"),
            tensor("akk"), tensor("w"), tensor("u"), tensor("qg"), tensor("kg"),
            tensor("v_new"), tensor("h"),
            None if null_output_pointer == "workspaceSize" else ctypes.byref(workspace),
            None if null_output_pointer == "executor" else ctypes.byref(executor),
        ]
        runtime._recent_error_message()
        result = api(*args)
        message = runtime._recent_error_message()
        assert result == status, (name, result, status, message)
        assert code in message and parameter in message and "aclnnChunkKdaFwd" in message, (
            name,
            code,
            parameter,
            message,
        )
        assert not executor.value and workspace.value == 0, (name, workspace.value, executor.value)
        print(f"PASS {name}: aclnnStatus={result}; {message.strip()}", flush=True)
    finally:
        context.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    empty_case = fixture(seqlen=0)
    cases = [
        ("chunk_size", {"chunk_size": 128}, "EZ0024", "chunkSize"),
        ("layout", {"layout": b"BAD"}, "EZ0024", "layout"),
        ("q_dtype", {"q": _empty((1, 4, 1, 128), torch.float32)}, "EZ0019", "q"),
        ("empty_tokens", empty_case, "EZ0024", "q.shape[T]"),
        ("output_shape", {"attn": _empty((1, 5, 1, 128))}, "EZ0008", "attnOut"),
        ("cu_size", {"cu_seqlens": [0]}, "EZ0025", "cuSeqlensOptional"),
        ("cu_endpoint", {"cu_seqlens": [0, 3]}, "EZ0024", "cuSeqlensOptional[last]"),
    ]
    for case in cases:
        check_raw(*case)

    mixed_kv = fixture(k_dim=32, v_dim=32)
    check_raw("mixed_kv", mixed_kv, "EZ0027", "q.shape[K],v.shape[V]")
    check_raw("missing_q", {"q": None}, "EZ0004", "q", status=161001)
    for key in ("workspaceSize", "executor"):
        check_raw(
            "missing_" + key,
            {},
            "EZ0004",
            key,
            status=161001,
            null_output_pointer=key,
        )
    print(f"PASS {len(cases) + 4} ChunkKdaFwd raw ACLNN error cases")


if __name__ == "__main__":
    main()
