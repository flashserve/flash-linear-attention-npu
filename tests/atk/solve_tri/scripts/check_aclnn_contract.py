"""A2 ACLNN parameter rejection checks shared by the CLI and ATK executor.

The descriptors reference live tensors. Invalid calls stop in GetWorkspaceSize;
no malformed argument is passed to the second-stage API. The separate positive
case checks legal chunk permutations through the public Stable dispatcher.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import re
import sys


OP_DIR = Path(__file__).resolve().parents[1]
PARAM_INVALID = 161002


def _case(name, updates, parameter, actual, expected, packed=False):
    return dict(name=name, updates=updates, parameter=parameter, actual=actual,
                expected=expected, packed=packed, expected_return_code=PARAM_INVALID,
                value_diagnostic=parameter in ("x.dtype", "xOut.dtype", "x.rank", "xOut.shape",
                                               "x.chunkSize", "layout"))


# Each case changes one constraint; expected fragments distinguish its branch.
CASES = [
    _case("fp32_input", {"dtype": "float32"}, "x.dtype",
          [r"(?<![a-z])(?:dt_)?float(?![a-z0-9])"], ["float16", "bf16"]),
    _case("output_dtype", {"output_dtype": "float16"}, "xOut.dtype", ["float16"], ["b(?:float)?16"]),
    _case("rank_two", {"shape": [64, 64]}, "x.rank", [r"(?<![a-z0-9])2(?![a-z0-9])"], ["3or4"]),
    _case("output_shape", {"output_shape": [1, 63, 1, 64]}, "xOut.shape",
          [r"\[1,63,1,64\]"], [r"\[1,64,1,64\]"]),
    _case("output_stride", {"strided_output": True}, "xOut", ["non-contiguous"], ["contiguous"]),
    _case("chunk_48", {"shape": [1, 64, 1, 48]}, "x.chunkSize", ["48"], ["16", "32", "64", "128"]),
    _case("unknown_layout", {"layout": "bad_layout"}, "layout", ["bad_layout"],
          ["bsnd,bnsd,bhtd,tnd,orntd"]),
    _case("layout_rank", {"layout": "tnd"}, "x.rank,layout", ["actualrank=4,layout=tnd"], ["expectedrank3"]),
    _case("zero_batch", {"shape": [0, 64, 1, 64]}, "x.shape",
          [r"actualshape=\[0,64,1,64\]"], ["expectedpositivebatch,head,andtokendimensions"]),
    _case("zero_tokens_dense", {"shape": [1, 0, 1, 64]}, "x.shape",
          [r"actualshape=\[1,0,1,64\]"], ["expectedpositivebatch,head,andtokendimensions"]),
    _case("zero_heads_dense", {"shape": [1, 64, 0, 64]}, "x.shape",
          [r"actualshape=\[1,64,0,64\]"], ["expectedpositivebatch,head,andtokendimensions"]),
    _case("zero_tokens_tnd", {"shape": [0, 1, 64], "cu": [0, 0]}, "x.shape",
          [r"actualshape=\[0,1,64\]"], ["expectedpositivebatch,head,andtokendimensions"], True),
    _case("zero_tokens_ntd", {"shape": [1, 0, 64], "layout": "ntd", "cu": [0, 0]}, "x.shape",
          [r"actualshape=\[1,0,64\]"], ["expectedpositivebatch,head,andtokendimensions"], True),
    _case("zero_heads_tnd", {"shape": [65, 0, 64]}, "x.shape",
          [r"actualshape=\[65,0,64\]"], ["expectedpositivebatch,head,andtokendimensions"], True),
    _case("missing_indices", {"indices": None}, "cu_seqlens,chunk_indices",
          ["actualcu_seqlens=present,chunk_indices=null"], ["expectedbothpresentorbothnull"], True),
    _case("missing_cu", {"cu": None}, "cu_seqlens,chunk_indices",
          ["actualcu_seqlens=null,chunk_indices=present"], ["expectedbothpresentorbothnull"], True),
    _case("missing_both", {"cu": None, "indices": None}, "cu_seqlens,chunk_indices",
          ["actual=bothnull"], ["expectedbothpresentforlayouttnd"], True),
    _case("dense_metadata", {"cu": [0, 64], "indices": [0, 0]}, "cu_seqlens,chunk_indices",
          ["actual=present"], ["expectedbothnullforlayoutbsnd"]),
    _case("cu_size", {"cu": [0]}, "cu_seqlens", ["actualsize=1"], ["expectedatleast2"], True),
    _case("cu_first", {"cu": [1, 65]}, "cu_seqlens", ["actualfirst=1,last=65"],
          ["expectedfirst=0,last=65"], True),
    _case("cu_last", {"cu": [0, 64]}, "cu_seqlens", ["actualfirst=0,last=64"],
          ["expectedfirst=0,last=65"], True),
    _case("cu_decreasing", {"cu": [0, 50, 30, 65]}, "cu_seqlens",
          [r"sequence1actual=\[50,30\]"], ["expected0<=begin<=end<=65"], True),
    _case("cu_negative", {"cu": [0, -1, 65]}, "cu_seqlens",
          [r"sequence0actual=\[0,-1\]"], ["expected0<=begin<=end<=65"], True),
    _case("cu_out_of_range", {"cu": [0, 66, 65]}, "cu_seqlens",
          [r"sequence0actual=\[0,66\]"], ["expected0<=begin<=end<=65"], True),
    _case("indices_odd", {"indices": [0, 0, 0]}, "chunk_indices",
          ["actualelementcount=3"], [r"expectedonepairperchunk\(2pairs\)"], True),
    _case("indices_missing", {"indices": [0, 0]}, "chunk_indices",
          ["actualelementcount=2"], [r"expectedonepairperchunk\(2pairs\)"], True),
    _case("indices_duplicate", {"indices": [0, 0, 0, 0]}, "chunk_indices",
          [r"sortedpair1actual=\[0,0\]"], [r"expected\[0,1\]"], True),
    _case("indices_sequence_out_of_range", {"indices": [0, 0, 1, 0]}, "chunk_indices",
          [r"sortedpair1actual=\[1,0\]"], [r"expected\[0,1\]"], True),
    _case("indices_chunk_out_of_range", {"indices": [0, 0, 0, 2]}, "chunk_indices",
          [r"sortedpair1actual=\[0,2\]"], [r"expected\[0,1\]"], True),
    _case("indices_negative_sequence", {"indices": [-1, 0, 0, 1]}, "chunk_indices",
          [r"sortedpair0actual=\[-1,0\]"], [r"expected\[0,0\]"], True),
    _case("indices_negative_chunk", {"indices": [0, -1, 0, 1]}, "chunk_indices",
          [r"sortedpair0actual=\[0,-1\]"], [r"expected\[0,0\]"], True),
]
CASE_BY_NAME = {case["name"]: case for case in CASES}
POSITIVE_CASE = "valid_empty_and_reordered"


def validate_diagnostic(case, status, message, workspace, executor):
    if status != PARAM_INVALID:
        raise AssertionError(f"{case['name']}: expected {PARAM_INVALID}, got {status}: {message}")
    if workspace or executor:
        raise AssertionError(f"{case['name']}: rejected call returned an executor or workspace")
    compact = re.sub(r"\s+", "", message).lower()
    if "aclnnsolvetri" not in compact or not re.search(r"\bez\d{4}\b", message, re.I):
        raise AssertionError(f"{case['name']}: missing operator or structured error code: {message}")
    if case["parameter"].lower() not in compact:
        raise AssertionError(f"{case['name']}: missing parameter name: {message}")
    fields = {"actual": compact, "expected": compact}
    if case["value_diagnostic"]:
        value_message = re.search(r"incorrect\s+(?:value|dtype|data\s*type|shape\s+dim|shape|format)\s+(.*?)\.\s*It\s+should\s+be\s+([^\n]+)",
                                  message, re.I)
        if not value_message:
            raise AssertionError(f"{case['name']}: missing structured actual/expected fields: {message}")
        fields = dict(zip(("actual", "expected"),
                          (re.sub(r"\s+", "", item).lower() for item in value_message.groups())))
    for kind in ("actual", "expected"):
        for pattern in case[kind]:
            if not re.search(pattern, fields[kind], re.I):
                raise AssertionError(f"{case['name']}: missing {kind} pattern {pattern!r}: {message}")


def check_rejection(name, device, expected_return_code=PARAM_INVALID):
    import torch

    sys.path.insert(0, str(OP_DIR.parent / "common"))
    from aclnn_error_probe import _CallContext, runtime

    if expected_return_code != PARAM_INVALID:
        raise AssertionError("negative manifest must require ACLNN_ERR_PARAM_INVALID=161002")
    case = CASE_BY_NAME[name]
    fixture = dict(shape=[65, 1, 64] if case["packed"] else [1, 64, 1, 64],
                   dtype="bfloat16", cu=[0, 65] if case["packed"] else None,
                   indices=[0, 0, 0, 1] if case["packed"] else None,
                   layout="tnd" if case["packed"] else "bsnd")
    fixture.update(case["updates"])
    def make_tensor(shape, dtype):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel() == 0:
            # Keep a real device address for descriptors with a zero-size view.
            storage = torch.empty(1, dtype=dtype, device=device)
            tensor = storage.as_strided(shape, tensor.stride())
        return tensor

    x = make_tensor(fixture["shape"], getattr(torch, fixture["dtype"]))
    output_shape = fixture.get("output_shape", fixture["shape"])
    output_dtype = getattr(torch, fixture.get("output_dtype", fixture["dtype"]))
    if fixture.get("strided_output"):
        storage_shape = list(output_shape)
        storage_shape[-1] *= 2
        out = torch.empty(storage_shape, dtype=output_dtype, device=device)[..., ::2]
    else:
        out = make_tensor(output_shape, output_dtype)
    api = runtime()
    context = _CallContext(api, device)
    workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
    try:
        args = [context.tensor(x, "x", acl_format_override=2),
                context.int_array(fixture["cu"]), context.int_array(fixture["indices"]),
                fixture["layout"].encode(), context.tensor(out, "xOut", acl_format_override=2)]
        function = api.symbol("aclnnSolveTriGetWorkspaceSize")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                             ctypes.POINTER(ctypes.c_void_p)]
        function.restype = ctypes.c_int
        api._recent_error_message()
        status = function(*args, ctypes.byref(workspace), ctypes.byref(executor))
        message = api._recent_error_message()
        validate_diagnostic(case, status, message, workspace.value, executor.value)
        print(f"PASS {name}: aclnnStatus={status}; {message.strip()}", flush=True)
        return status
    finally:
        context.destroy()


def check_execution(device):
    import torch
    from fla_npu.ops.ascendc import npu_solve_tri

    cu = [0, 0, 65, 65, 129, 129]
    metadata = [(cu, [1, 0, 1, 1, 3, 0]),
                (cu, [3, 0, 1, 1, 1, 0]),
                ([0, 65, 129], [0, 0, 0, 1, 1, 0])]
    for layout in ("tnd", "ntd"):
        generator = torch.Generator(device="cpu").manual_seed(20261109)
        x = torch.zeros((129, 2, 64), dtype=torch.bfloat16)
        for begin, end in ((0, 64), (64, 65), (65, 129)):
            size = end - begin
            block = torch.tril(torch.randn((2, size, size), generator=generator) * 0.05, diagonal=-1)
            x[begin:end, :, :size] = block.transpose(0, 1).bfloat16()
        if layout == "ntd":
            x = x.transpose(0, 1).contiguous()
        x = x.to(device)
        results = []
        for offsets, indices in metadata:
            result = npu_solve_tri(x, cu_seqlens=offsets, chunk_indices=indices, layout=layout)
            result = result.detach().cpu().float()
            data = result if layout == "tnd" else result.transpose(0, 1)
            data[64:65, :, 1:] = 0  # Tail padding is outside the operator contract.
            if not torch.isfinite(result).all():
                raise AssertionError(f"{layout}: non-finite result in valid region")
            results.append(result)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)
        print(f"PASS {layout}: canonical, reordered and compacted metadata agree", flush=True)
    return 0


def atk_cases():
    result = []
    for index, name in enumerate([case["name"] for case in CASES] + [POSITIVE_CASE]):
        result.append({
            "id": index, "default_seed": 20261109, "name": "solve_tri_contract_" + name,
            "version": "v2.1", "api": "pytorch", "api_type": "executor_solve_tri_a2_contract",
            "expected_error_msg": None, "backward": False,
            "standard": {"acc": "single_bm", "perf": "not_key"},
            "inputs": [
                {"name": "contract_case", "type": "attr", "required": True, "dtype": "string",
                 "shape": None, "range_values": name},
                {"name": "expected_return_code", "type": "attr", "required": True, "dtype": "int",
                 "shape": None, "range_values": 0 if name == POSITIVE_CASE else PARAM_INVALID},
            ],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--generate-json", action="store_true")
    parser.add_argument("--check-json", action="store_true")
    args = parser.parse_args()
    path = OP_DIR / "atk_solve_tri_a2_negative.json"
    if args.generate_json:
        path.write_text(json.dumps(atk_cases(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(CASES)} negative and one positive contract case to {path}")
        return
    if args.check_json:
        if json.loads(path.read_text(encoding="utf-8")) != atk_cases():
            raise AssertionError("ATK contract JSON does not match the checked-in case definitions")
        print(f"PASS contract manifest: {len(CASES)} negative and one positive case")
        return
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    for case in CASES:
        check_rejection(case["name"], device)
    check_execution(device)


if __name__ == "__main__":
    main()
