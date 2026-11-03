#!/usr/bin/env python3
"""Assert rejected ACLNN Prepare calls report a specific registered message.

Uses the test-only raw descriptor probe to cover output buffers and null output
pointers that the public Stable dispatcher cannot express. Never launches a
rejected call or treats an arbitrary exception as a passing negative test.
"""
import argparse
import ctypes
from pathlib import Path
import re
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atk" / "common"))
from aclnn_error_probe import runtime, _CallContext  # noqa: E402


def empty(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="npu")


def fixture():
    result = {name: empty((1, 2, 3, 128)) for name in ("q", "k", "v")}
    result.update(g=empty((1, 2, 3, 128), torch.float32), beta=empty((1, 2, 3)),
                  a_log=None, dt_bias=None, cu_seqlens=None, chunk_indices=None,
                  layout="BNSD", scale=0.125, chunk_size=64, epsilon=1e-6,
                  norm=False, gate=False, beta_sigmoid=False, allow_neg=False,
                  safe=False, lower_bound=-5.0, exp2=True)
    for name in ("wOut", "uOut", "qgOut", "kgOut", "qgScaledOut", "qHatOut", "kHatOut"):
        result[name] = empty((1, 2, 3, 128))
    for name in ("aqkOut", "akkOut"):
        result[name] = empty((1, 2, 3, 64))
    result["gkOut"] = empty((1, 2, 3, 128), torch.float32)
    for name in ("qRstdOut", "kRstdOut", "betaEffOut"):
        result[name] = empty((1, 2, 3), torch.float32)
    return result


CATALOG = "native"


def reject(name, updates, code, parameter, *, status=161002, null_pointer=None, detail=None):
    case = fixture()
    case.update(updates)
    api = runtime()
    context = _CallContext(api)
    function = api.symbol("aclnnChunkKdaFwdPrepareGetWorkspaceSize")
    function.argtypes = ([ctypes.c_void_p] * 9 +
                         [ctypes.c_char_p, ctypes.c_double, ctypes.c_int64, ctypes.c_double] +
                         [ctypes.c_bool] * 5 + [ctypes.c_double, ctypes.c_bool] +
                         [ctypes.c_void_p] * 13 +
                         [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    function.restype = ctypes.c_int
    workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
    try:
        def tensor(key):
            value = case[key]
            return None if value is None else context.tensor(
                value, key, acl_format_override=2, storage_shape_override=tuple(value.shape))
        args = [tensor(key) for key in ("q", "k", "v", "g", "beta", "a_log", "dt_bias")]
        args += [context.int_array(case[key]) for key in ("cu_seqlens", "chunk_indices")]
        args += [None if case["layout"] is None else case["layout"].encode(),
                 case["scale"], case["chunk_size"], case["epsilon"],
                 case["norm"], case["gate"], case["beta_sigmoid"], case["allow_neg"], case["safe"],
                 case["lower_bound"], case["exp2"]]
        args += [tensor(key) for key in (
            "gkOut", "aqkOut", "akkOut", "wOut", "uOut", "qgOut", "kgOut", "qgScaledOut",
            "qHatOut", "kHatOut", "qRstdOut", "kRstdOut", "betaEffOut")]
        args += [None if null_pointer == "workspaceSize" else ctypes.byref(workspace),
                 None if null_pointer == "executor" else ctypes.byref(executor)]
        api._recent_error_message()
        observed = function(*args)
        message = api._recent_error_message()
        if CATALOG == "legacy":
            legacy = {"EZ0017": "EZ0006", "EZ0018": "EZ0006", "EZ0019": "EZ0007",
                      "EZ0021": "EZ0007", "EZ0025": "EZ0003"}
            code = "EZ0004" if status == 161001 else legacy.get(code, "EZ0002")
        assert observed == status, (name, observed, status, message)
        assert code in message and parameter in message and "aclnnChunkKdaFwdPrepare" in message, (name, message)
        entity = re.escape("aclnnChunkKdaFwdPrepare")
        patterns = (
            rf"(?:Parameters?|Attribute|Input parameter) (.*?) of {entity}",
            rf"OP\[{entity}\] attr\[: (.*?)\],",
            rf"OP\[{entity}\] (.*?) has incorrect (?:dtype|format)",
            rf"OP\[{entity}\] get \[(.*?)\] failed\.",
        )
        field = next((match for pattern in patterns
                      if (match := re.search(pattern, message)) is not None), None)
        assert field is not None, (name, message)
        reported = field.group(1)
        assert (reported == parameter or parameter in reported.split(",") or
                reported.startswith(parameter + " ")), (name, parameter, reported)
        assert "EZ9999" not in message and "EZ1001" not in message, (name, message)
        if detail is not None:
            assert detail in message, (name, detail, message)
        assert not workspace.value and not executor.value, (name, workspace.value, executor.value)
        print(f"PASS {name}: status={observed}; {message.strip()}", flush=True)
    finally:
        context.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--catalog", choices=("native", "legacy"), default="native")
    arguments = parser.parse_args()
    global CATALOG
    CATALOG = arguments.catalog
    torch.npu.set_device(arguments.device)
    cases = [
        ("layout", {"layout": "bnsd"}, "EZ0017", "layout"),
        ("layout_rank", {"layout": "NTD"}, "EZ0013", "q,v,g,beta"),
        ("q_k_shape", {"k": empty((1, 2, 4, 128))}, "EZ0010", "q,k"),
        ("value_shape", {"v": empty((1, 2, 4, 128))}, "EZ0010", "v,g,beta"),
        ("gate_layout", {"g": empty((1, 3, 2, 128), torch.float32)}, "EZ0010", "v,g,beta"),
        ("beta_rank", {"beta": empty((1, 2, 3, 1))}, "EZ0013", "q,v,g,beta"),
        ("q_dtype", {"q": empty((1, 2, 3, 128), torch.float16)}, "EZ0019", "q"),
        ("g_dtype", {"g": empty((1, 2, 3, 128), torch.float16)}, "EZ0019", "g"),
        ("beta_dtype", {"beta": empty((1, 2, 3), torch.float16)}, "EZ0019", "beta"),
        ("output_dtype", {"gkOut": empty((1, 2, 3, 128))}, "EZ0019", "gkOut"),
        ("output_shape", {"gkOut": empty((1, 3, 2, 128), torch.float32)}, "EZ0008", "gkOut"),
        ("output_stride", {"wOut": empty((1, 2, 3, 256))[..., ::2]}, "EZ0037", "wOut"),
        ("output_mask", {"akkOut": None}, "EZ0027", "output pointers"),
        ("chunk_size", {"chunk_size": 32}, "EZ0024", "chunk_size"),
        ("scale_nan", {"scale": float("nan")}, "EZ0026", "scale"),
        ("scale_overflow", {"scale": 1e100}, "EZ0026", "scale"),
        ("epsilon_zero", {"epsilon": 0}, "EZ0026", "epsilon"),
        ("epsilon_underflow", {"epsilon": 1e-100}, "EZ0026", "epsilon"),
        ("epsilon_negative", {"epsilon": -1}, "EZ0026", "epsilon"),
        ("lower_bound_nan", {"lower_bound": float("nan")}, "EZ0026", "lower_bound"),
        ("beta_mode", {"allow_neg": True}, "EZ0027", "allow_neg_eigval"),
        ("safe_without_gate", {"safe": True}, "EZ0027", "use_gate_in_kernel"),
        ("log_without_gate", {"a_log": empty((2,), torch.float32)}, "EZ0027", "use_gate_in_kernel"),
        ("gate_missing_log", {"gate": True}, "EZ0037", "a_log"),
        ("log_shape", {"gate": True, "a_log": empty((3,), torch.float32)}, "EZ0008", "a_log"),
        ("bias_shape", {"gate": True, "a_log": empty((2,), torch.float32), "dt_bias": empty((2, 128), torch.float32)}, "EZ0008", "dt_bias"),
        ("safe_bound", {"gate": True, "safe": True, "a_log": empty((2,), torch.float32), "lower_bound": -6}, "EZ0026", "lower_bound"),
        ("indices_without_cu", {"chunk_indices": [0, 0]}, "EZ0037", "chunk_indices"),
        ("cu_size", {"cu_seqlens": [0]}, "EZ0025", "cu_seqlens"),
        ("cu_first", {"cu_seqlens": [1, 3]}, "EZ0024", "cu_seqlens[0]"),
        ("cu_last", {"cu_seqlens": [0, 2]}, "EZ0024", "cu_seqlens[-1]"),
        ("cu_order", {"cu_seqlens": [0, 2, 1, 3]}, "EZ0027", "cu_seqlens"),
        ("indices_size", {"cu_seqlens": [0, 3], "chunk_indices": [0]}, "EZ0025", "chunk_indices"),
        ("indices_order", {"cu_seqlens": [0, 3], "chunk_indices": [0, 1]}, "EZ0027", "chunk_indices pair"),
        ("gva", {"q": empty((1, 3, 3, 128)), "k": empty((1, 3, 3, 128))}, "EZ0027", "HK,HV"),
    ]
    for case in cases:
        reject(*case, status=161001 if case[0] == "gate_missing_log" else 161002)
    for name in ("q", "k", "v", "g", "beta", "layout", "gkOut", "aqkOut", "wOut", "uOut", "kgOut", "qgScaledOut"):
        reject("missing_" + name, {name: None}, "EZ0037", name, status=161001)
    for name in ("workspaceSize", "executor"):
        reject("missing_" + name, {}, "EZ0037", name, status=161001, null_pointer=name)
    print(f"PASS {len(cases) + 14} rejected ACLNN calls with exact return status and registered Error Message")


if __name__ == "__main__":
    main()
