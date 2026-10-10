#!/usr/bin/env python3
"""通过公开 AscendC 入口验证 GDN forward 异常，输出完整异常和 ACL 错误信息。"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import traceback
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "stable", "ctypes"), default="auto",
                        help="auto 使用公开入口默认后端；其它选项要求实际使用指定后端")
    args = parser.parse_args()
    if args.backend != "auto":
        os.environ["FLA_NPU_STABLE_ABI"] = args.backend

    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    acl = ctypes.CDLL("libascendcl.so")
    recent_error = acl.aclGetRecentErrMsg
    recent_error.argtypes = []
    recent_error.restype = ctypes.c_char_p

    def acl_error():
        message = recent_error()
        return message.decode("utf-8", errors="replace") if message else ""

    def tensor(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    base = dict(q=tensor((1, 2, 65, 128)), k=tensor((1, 2, 65, 128)),
                v=tensor((1, 4, 65, 128)), g=tensor((1, 65, 4), torch.float32),
                beta=tensor((1, 65, 4), torch.float32), scale=128 ** -0.5,
                layout="BNSD", chunk_size=64, disable_recompute=False)
    # name, public arguments, host status/EZ/parameter, allowed adapter error pattern.
    # Adapter patterns describe current behavior; they do not imply host validation.
    missing_tensor = r"NoneType|Tensor.*None|None.*Tensor|aoti_torch_get_data_ptr\(handle, &tensor_data\) API call failed"
    cases = [
        ("required_q", {"q": None}, 161001, "EZ0004", "q", missing_tensor),
        ("required_k", {"k": None}, 161001, "EZ0004", "k", missing_tensor),
        ("required_v", {"v": None}, 161001, "EZ0004", "v", missing_tensor),
        ("required_g", {"g": None}, 161001, "EZ0004", "g", missing_tensor),
        ("required_beta", {"beta": None}, 161001, "EZ0004", "beta", missing_tensor),
        ("rank_q", {"q": tensor((1, 2, 128))}, 161002, "EZ0011", "q", r"rank 4|size_of dim"),
        ("rank_k", {"k": tensor((1, 128))}, 161002, "EZ0011", "k", None),
        ("rank_v", {"v": tensor((1, 4, 128))}, 161002, "EZ0011", "v", r"rank 4|size_of dim"),
        ("rank_g", {"g": tensor((65, 4), torch.float32)}, 161002, "EZ0011", "g", None),
        ("rank_beta", {"beta": tensor((65, 4), torch.float32)}, 161002, "EZ0011", "beta", None),
        ("shape_k", {"k": tensor((1, 2, 64, 128))}, 161002, "EZ0008", "k", None),
        ("shape_v", {"v": tensor((1, 4, 64, 128))}, 161002, "EZ0008", "v", None),
        ("shape_g", {"g": tensor((1, 64, 4), torch.float32)}, 161002, "EZ0008", "g", None),
        ("shape_beta", {"beta": tensor((1, 64, 4), torch.float32)}, 161002, "EZ0008", "beta", None),
        ("dtype_q", {"q": tensor((1, 2, 65, 128), torch.float32)}, 161002, "EZ0007", "q", None),
        ("dtype_k", {"k": tensor((1, 2, 65, 128), torch.float16)}, 161002, "EZ0007", "k", None),
        ("dtype_v", {"v": tensor((1, 4, 65, 128), torch.float16)}, 161002, "EZ0007", "v", None),
        ("dtype_g", {"g": tensor((1, 65, 4), torch.int32)}, 161002, "EZ0007", "g", None),
        ("dtype_beta", {"beta": tensor((1, 65, 4), torch.int32)}, 161002, "EZ0007", "beta", None),
        ("layout", {"layout": "invalid"}, 161002, "EZ0002", "layout", r"layout|unknown layout"),
        ("chunk_zero", {"chunk_size": 0}, 161002, "EZ0002", "chunkSize", r"positive int64"),
        ("chunk_negative", {"chunk_size": -1}, 161002, "EZ0002", "chunkSize", r"positive int64"),
        ("chunk_unsupported", {"chunk_size": 65}, 161002, "EZ0002", "chunkSize", None),
        ("scale_nan", {"scale": float("nan")}, 161002, "EZ0002", "scale", None),
        ("scale_inf", {"scale": float("inf")}, 161002, "EZ0002", "scale", None),
        ("scale_fp32_overflow", {"scale": 1e300}, 161002, "EZ0002", "scale", None),
        ("missing_indices", {"cu_seqlens": (0, 65)}, 161002, "EZ0037", "cuSeqlensOptional", None),
        ("missing_cu", {"chunk_indices": (0, 0)}, 161002, "EZ0037", "cuSeqlensOptional", None),
        ("short_cu", {"cu_seqlens": (0,), "chunk_indices": (0, 0)},
         161002, "EZ0025", "cuSeqlensOptional", None),
        ("cu_endpoint", {"cu_seqlens": (0, 64), "chunk_indices": (0, 0)},
         161002, "EZ0037", "cuSeqlensOptional", None),
        ("cu_decreasing", {"cu_seqlens": (0, 64, 1, 65), "chunk_indices": (0, 0)},
         161002, "EZ0037", "cuSeqlensOptional", None),
        ("odd_indices", {"cu_seqlens": (0, 65), "chunk_indices": (0,)},
         161002, "EZ0025", "chunkIndicesOptional", None),
        ("index_order", {"cu_seqlens": (0, 65), "chunk_indices": (0, 1, 0, 0)},
         161002, "EZ0037", "chunkIndicesOptional", None),
        ("initial_shape", {"initial_state": tensor((2, 4, 128, 128), torch.float32)},
         161002, "EZ0008", "initialStateOptional", None),
        ("state_v_first", {"v": tensor((1, 4, 65, 256)), "state_v_first": True,
                           "initial_state": tensor((1, 4, 128, 256))},
         161002, "EZ0008", "initialStateOptional", None),
        ("reserved_gate", {"use_gate_in_kernel": True}, None, None, None, r"use_gate_in_kernel.*not supported"),
        ("reserved_a_log", {"a_log": tensor((4,), torch.float32)}, None, None, None, r"a_log and dt_bias.*must be None"),
        ("reserved_dt_bias", {"dt_bias": tensor((4,), torch.float32)}, None, None, None, r"a_log and dt_bias.*must be None"),
    ]
    counts = Counter()
    for name, override, expected_status, expected_ez, parameter, adapter_pattern in cases:
        acl_error()  # Drain previous reports before the public call.
        print(f"\nCASE {name}", flush=True)
        try:
            ascendc.npu_chunk_gated_delta_rule_fwd(**{**base, **override})
        except Exception as exc:
            message = str(exc)
            report = acl_error()
            full_message = message + "\n" + report
            status_match = re.search(
                r"aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize(?: failed with aclnnStatus=| failed: )(\d+)",
                message)
            if status_match:
                origin = "ACLNN host"
                passed = (int(status_match.group(1)) == expected_status
                          and expected_ez is not None and expected_ez in full_message
                          and parameter in full_message)
            else:
                origin = "adapter/dispatcher"
                passed = (adapter_pattern is not None
                          and re.search(adapter_pattern, message, re.DOTALL) is not None)
            print(f"{'PASS' if passed else 'FAIL'} {name}: {origin}", flush=True)
            print(traceback.format_exc(), end="", flush=True)
            if report:
                print("ACL error report:", flush=True)
                print(report, flush=True)
            counts[origin if passed else "failed"] += 1
        else:
            counts["failed"] += 1
            print(f"FAIL {name}: invalid public call unexpectedly succeeded", flush=True)

    backend = ascendc.BACKENDS.get("npu_chunk_gated_delta_rule_fwd", "unknown")
    if args.backend != "auto" and backend != args.backend:
        counts["failed"] += 1
        print(f"FAIL backend: requested {args.backend}, used {backend}", flush=True)
    print(f"SUMMARY cases={len(cases)} backend={backend} host={counts['ACLNN host']} "
          f"adapter={counts['adapter/dispatcher']} failed={counts['failed']}", flush=True)
    raise SystemExit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
