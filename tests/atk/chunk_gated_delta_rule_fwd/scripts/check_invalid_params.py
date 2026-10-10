#!/usr/bin/env python3
"""直接验证 ACLNN host 的返回码和标准参数错误，非法调用不启动 kernel。"""

from __future__ import annotations

import argparse
import ctypes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import _aclnn_ctypes, _runtime

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    def tensor(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    base = dict(q=tensor((1, 2, 65, 128)), k=tensor((1, 2, 65, 128)),
                v=tensor((1, 4, 65, 128)), g=tensor((1, 65, 4), torch.float32),
                beta=tensor((1, 65, 4), torch.float32), oOut=tensor((1, 65, 4, 128)),
                initialStateOptional=None, finalStateOutOptional=None,
                qHatOutOptional=None, kHatOutOptional=None, qRstdOutOptional=None,
                kRstdOutOptional=None, betaEffOutOptional=None, gCumsumOutOptional=None,
                aOutOptional=None, hOutOptional=None,
                cuSeqlensOptional=None, chunkIndicesOptional=None,
                layout="BNSD", scale=128 ** -0.5, chunkSize=64, stateVFirst=False)
    # name, override, expected status, predefined error, offending parameter
    cases = [
        ("required_q", {"q": None}, 161001, "EZ0004", "q"),
        ("rank_k", {"k": tensor((1, 128))}, 161002, "EZ0011", "k"),
        ("shape_k", {"k": tensor((1, 2, 64, 128))}, 161002, "EZ0008", "k"),
        ("shape_g", {"g": tensor((1, 64, 4), torch.float32)}, 161002, "EZ0008", "g"),
        ("dtype_q", {"q": tensor((1, 2, 65, 128), torch.float32)}, 161002, "EZ0007", "q"),
        ("dtype_k", {"k": tensor((1, 2, 65, 128), torch.float16)}, 161002, "EZ0007", "k"),
        ("output_dtype", {"oOut": tensor((1, 65, 4, 128), torch.float32)}, 161002, "EZ0019", "oOut"),
        ("layout", {"layout": "invalid"}, 161002, "EZ0002", "layout"),
        ("chunk_zero", {"chunkSize": 0}, 161002, "EZ0002", "chunkSize"),
        ("chunk_negative", {"chunkSize": -1}, 161002, "EZ0002", "chunkSize"),
        ("chunk_unsupported", {"chunkSize": 65}, 161002, "EZ0002", "chunkSize"),
        ("scale_nan", {"scale": float("nan")}, 161002, "EZ0002", "scale"),
        ("scale_inf", {"scale": float("inf")}, 161002, "EZ0002", "scale"),
        ("scale_fp32_overflow", {"scale": 1e300}, 161002, "EZ0002", "scale"),
        ("missing_indices", {"cuSeqlensOptional": (0, 65)}, 161002, "EZ0037", "cuSeqlensOptional"),
        ("missing_cu", {"chunkIndicesOptional": (0, 0)}, 161002, "EZ0037", "chunkIndicesOptional"),
        ("short_cu", {"cuSeqlensOptional": (0,), "chunkIndicesOptional": (0, 0)},
         161002, "EZ0025", "cuSeqlensOptional"),
        ("cu_endpoint", {"cuSeqlensOptional": (0, 64), "chunkIndicesOptional": (0, 0)},
         161002, "EZ0037", "cuSeqlensOptional"),
        ("cu_decreasing", {"cuSeqlensOptional": (0, 64, 1, 65), "chunkIndicesOptional": (0, 0)},
         161002, "EZ0037", "cuSeqlensOptional"),
        ("odd_indices", {"cuSeqlensOptional": (0, 65), "chunkIndicesOptional": (0,)},
         161002, "EZ0025", "chunkIndicesOptional"),
        ("index_order", {"cuSeqlensOptional": (0, 65), "chunkIndicesOptional": (0, 1, 0, 0)},
         161002, "EZ0037", "chunkIndicesOptional"),
        ("initial_shape", {"initialStateOptional": tensor((2, 4, 128, 128), torch.float32)},
         161002, "EZ0008", "initialStateOptional"),
        ("final_shape", {"finalStateOutOptional": tensor((2, 4, 128, 128), torch.float32)},
         161002, "EZ0008", "finalStateOutOptional"),
        ("state_v_first", {"v": tensor((1, 4, 65, 256)), "oOut": tensor((1, 65, 4, 256)),
                           "stateVFirst": True, "initialStateOptional": tensor((1, 4, 128, 256))},
         161002, "EZ0008", "initialStateOptional"),
        ("output_shape", {"oOut": tensor((1, 64, 4, 128))}, 161002, "EZ0008", "oOut"),
        ("rstd_shape", {"qRstdOutOptional": tensor((1, 65, 2), torch.float32)},
         161002, "EZ0008", "qRstdOutOptional"),
    ]
    runtime = _runtime.runtime()
    get_workspace = runtime.symbol("aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize")
    get_workspace.argtypes = _aclnn_ctypes._GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleFwd"]
    get_workspace.restype = ctypes.c_int
    inputs = ("q", "k", "v", "g", "beta", "aLogOptional", "dtBiasOptional", "initialStateOptional")
    outputs = ("oOut", "finalStateOutOptional", "qHatOutOptional", "kHatOutOptional",
               "qRstdOutOptional", "kRstdOutOptional", "betaEffOutOptional", "gCumsumOutOptional",
               "aOutOptional", "hOutOptional")
    for name, override, status, error_code, parameter in cases:
        values = {**base, **override}
        context = _runtime._CallContext(runtime, device)
        try:
            arguments = [context.tensor(values.get(key), key) for key in inputs]
            arguments += [context.int_array(values[key]) for key in ("cuSeqlensOptional", "chunkIndicesOptional")]
            layout = ctypes.create_string_buffer(values["layout"].encode("utf-8"))
            arguments += [ctypes.cast(layout, ctypes.c_char_p), ctypes.c_double(values["scale"]),
                          ctypes.c_int64(values["chunkSize"]), ctypes.c_bool(False), ctypes.c_bool(False),
                          ctypes.c_bool(False), ctypes.c_bool(values["stateVFirst"])]
            arguments += [context.tensor(values[key], key) for key in outputs]
            workspace = ctypes.c_uint64()
            executor = ctypes.c_void_p()
            # Drain an older error before checking the current call's report.
            runtime._recent_error_message()
            result = get_workspace(*arguments, ctypes.byref(workspace), ctypes.byref(executor))
            message = runtime._recent_error_message()
            passed = (result == status and error_code in message and parameter in message
                      and not executor.value)
            print(f"{'PASS' if passed else 'FAIL'} {name}: {result} (expected {status} {error_code})", flush=True)
            print(message if message else "<no ACL error message>", flush=True)
            if result != status or error_code not in message or parameter not in message:
                raise AssertionError(f"{name}: expected {status}/{error_code}/{parameter}, got {result}: {message}")
            if executor.value:
                raise AssertionError(f"{name}: invalid call returned an executor")
        finally:
            context.destroy()
    print(f"PASS {len(cases)} direct ACLNN negative cases", flush=True)


if __name__ == "__main__":
    main()
