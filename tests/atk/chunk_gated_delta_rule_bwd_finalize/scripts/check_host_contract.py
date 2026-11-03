"""真实 ACLNN GetWorkspaceSize 返回码/消息回归；从不调用 kernel launch。"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def cases():
    # id, T, cu, flat pairs, NT, expected code, required message fragments
    invalid = 161002  # ACLNN_ERR_PARAM_INVALID；由当前 CANN op_errno.h 定义。
    return [
        ("bad_cu_first", 64, [1, 64], [0, 0], 1, invalid, ["cuSeqlensOptional[0]", "1", "0"]),
        ("bad_cu_last", 64, [0, 63], [0, 0], 1, invalid, ["cuSeqlensOptional[last]", "63", "T=64"]),
        ("bad_cu_decrease", 64, [0, 40, 32, 64], [0, 0, 2, 0], 2, invalid, ["cuSeqlensOptional", "40", "32", "begin"]),
        ("bad_pair_seq", 64, [0, 64], [1, 0], 1, invalid, ["chunkIndicesOptional", ".seq", "1", "0 <= seq < 1"]),
        ("bad_pair_local", 64, [0, 64], [0, 2], 1, invalid, ["chunkIndicesOptional", ".localChunk", "2", "0 <= localChunk < 1"]),
        ("missing_chunk", 128, [0, 128], [0, 0], 1, invalid, ["chunkIndicesOptional", "complete chunk count=2"]),
        ("duplicate_chunk", 128, [0, 128], [0, 0, 0, 0], 2, invalid, ["chunkIndicesOptional", "exactly once"]),
        ("empty_sequence_has_chunk", 65, [0, 0, 65], [0, 0, 1, 1], 2, invalid, ["chunkIndicesOptional", ".localChunk", "0 <= localChunk < 0"]),
        ("odd_pair_size", 64, [0, 64], [0, 0, 0], 1, invalid, ["chunkIndicesOptional.size", "positive even size"]),
        ("valid_empty_sequence", 65, [0, 0, 65], [1, 0, 1, 1], 2, 0, []),
        ("valid_reordered_chunks", 65, [0, 0, 65], [1, 1, 1, 0], 2, 0, []),
        ("valid_dense", 64, None, None, 1, 0, []),
        ("bad_K", 64, None, None, 1, invalid, ["q.K", "64", "128"], {"K": 64}),
        ("bad_V", 64, None, None, 1, invalid, ["v.V", "64", "128", "256"], {"V": 64}),
        ("bad_head_ratio", 64, None, None, 1, invalid, ["v.HV/q.HK", "3", "2"], {"HK": 2, "HV": 3}),
        ("bad_state_NT", 64, None, None, 2, invalid, ["h.shape", "2", "1", "128"]),
        ("bad_q_rstd_shape", 64, None, None, 1, invalid, ["qRstdOptional.shape", "63", "64"], {"norm": True}),
        ("bad_beta_raw_shape", 64, None, None, 1, invalid, ["betaRawOptional.shape", "63", "64"], {"sigmoid": True}),
    ]


def run_case(runtime, torch, device, item):
    from fla_npu.ops.ascendc._runtime import _CallContext
    identifier, tokens, cu, pairs, nt, expected, fragments = item[:7]
    options = item[7] if len(item) > 7 else {}
    b, hk, hv, kd, vd = 1, options.get("HK", 1), options.get("HV", 1), options.get("K", 128), options.get("V", 128)
    bf16, fp32 = torch.bfloat16, torch.float32
    shapes = {"q": (b, hk, tokens, kd), "k": (b, hk, tokens, kd),
              **{n: (b, hv, tokens, vd) for n in ("v", "v_new", "do", "du")},
              "g": (b, hv, tokens), "beta": (b, hv, tokens),
              "h": (b, nt, hv, kd, vd), "dh": (b, nt, hv, kd, vd),
              "a": (b, hv, tokens, 64)}
    values = {name: torch.zeros(shape, dtype=fp32 if name in ("g", "beta") else bf16,
                                device=device) for name, shape in shapes.items()}
    optional = [None, None, None]
    if options.get("norm"):
        optional[:2] = [torch.ones((b, hk, tokens - 1), dtype=fp32, device=device),
                        torch.ones((b, hk, tokens), dtype=fp32, device=device)]
    if options.get("sigmoid"):
        optional[2] = torch.zeros((b, hv, tokens - 1), dtype=fp32, device=device)
    outputs = [torch.empty_like(values[name]) for name in ("q", "k", "v", "beta", "g")]
    context = _CallContext(runtime, torch.device(device))
    try:
        descriptor = lambda value, name: ctypes.c_void_p() if value is None else context.tensor(
            value, name, acl_format_override=2, storage_shape_override=tuple(value.shape))
        args = [descriptor(values[name], name) for name in
                ("q", "k", "v", "v_new", "do", "du", "g", "beta", "h", "dh", "a")]
        args += [descriptor(value, name) for name, value in zip(("q_rstd", "k_rstd", "beta_raw"), optional)]
        args += [context.int_array(cu), context.int_array(pairs), ctypes.c_double(128 ** -0.5),
                 ctypes.c_int64(64), ctypes.c_bool(options.get("norm", False)),
                 ctypes.c_bool(options.get("sigmoid", False)), *[ctypes.c_bool(False)] * 3]
        args += [descriptor(value, name) for name, value in zip(("dq", "dk", "dv", "dbeta", "dg"), outputs)]
        workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
        fn = runtime.symbol("aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize")
        fn.argtypes = ([ctypes.c_void_p] * 16 + [ctypes.c_double, ctypes.c_int64] + [ctypes.c_bool] * 5
                       + [ctypes.c_void_p] * 5 + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
        fn.restype = ctypes.c_int
        # 不调用 public wrapper 或 runtime.call：它们可能触发 launch/先行 Python 校验。
        runtime._recent_error_message()  # 清除本线程先前消息（若 CANN 版本提供）。
        code = fn(*args, ctypes.byref(workspace), ctypes.byref(executor))
        message = runtime._recent_error_message()
        matched = int(code) == expected and all(fragment in message for fragment in fragments)
        if expected != 0:
            matched = matched and not executor.value
        return dict(id=identifier, actual_code=int(code), expected_code=expected, message=message,
                    required_fragments=fragments, passed=matched, launched=False,
                    workspace_bytes=workspace.value, executor_created=bool(executor.value))
    finally:
        context.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print(json.dumps(cases(), ensure_ascii=False, indent=2))
        return 0
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    import torch_npu  # noqa: F401
    import fla_npu
    from fla_npu.ops.ascendc._runtime import runtime
    from arch22_identity import _arch22_runtime_identity
    torch.npu.set_device(a.device)
    rt = runtime()
    pre_call_identity = _arch22_runtime_identity()
    records = [run_case(rt, torch, f"npu:{a.device}", item) for item in cases()]
    identity = _arch22_runtime_identity(require_tiling=True, refresh=True)
    maps = identity["proc_maps"]
    libraries = sorted({line.split()[-1] for line in maps.splitlines() if "cust_opapi" in line})
    result = dict(schema="finalize-arch22-host-contract/v1", verdict="pass" if all(x["passed"] for x in records) else "fail",
                  launched=False, fla_npu_file=str(Path(fla_npu.__file__).resolve()),
                  source_commit_requested=os.getenv("FINALIZE_SOURCE_COMMIT"),
                  binding_sha256_requested=os.getenv("FINALIZE_BINDING_SHA256"),
                  measured_identity=identity["measured_identity"],
                  pre_call_identity=pre_call_identity["measured_identity"],
                  libraries=[dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in libraries],
                  proc_maps=maps, cases=records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(verdict=result["verdict"], passed=sum(x["passed"] for x in records), total=len(records))))
    return 0 if result["verdict"] == "pass" else 99


if __name__ == "__main__":
    raise SystemExit(main())
