"""通过真实 SDK 的完整 Bwd GetWorkspaceSize 检查参数返回码和错误文本。

--negative-only 仅覆盖 CheckParams 早退，可配合 RTLD_LAZY 加载实际 Bwd
Host TU 共享库。完整模式还覆盖合法输入构图，需要完整 Bwd 依赖。所有模式
均通过公开 ACLNN 入口调用，不执行 kernel launch。
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys


INVALID = 161002  # ACLNN_ERR_PARAM_INVALID，CANN op_errno.h。
NULLPTR = 161001  # ACLNN_ERR_PARAM_NULLPTR，CANN op_errno.h。


@dataclass(frozen=True)
class Case:
    name: str
    expected: int
    fragments: tuple[str, ...] = ()
    tokens: int = 64
    q_rstd: bool = False
    k_rstd: bool = False
    norm: bool = False
    cu: tuple[int, ...] | None = None
    pairs: tuple[int, ...] | None = None


def cases():
    return [
        Case("q_only_norm_on", INVALID, ("qRstd,kRstd", "kRstd=nullptr", "both be present"),
             q_rstd=True, norm=True),
        Case("k_only_norm_on", INVALID, ("qRstd,kRstd", "qRstd=nullptr", "both be present"),
             k_rstd=True, norm=True),
        Case("q_only_norm_off", INVALID, ("qRstd,kRstd", "kRstd=nullptr", "both be present"),
             q_rstd=True),
        Case("k_only_norm_off", INVALID, ("qRstd,kRstd", "qRstd=nullptr", "both be present"),
             k_rstd=True),
        Case("both_absent_norm_on", NULLPTR, ("qRstd,kRstd",), norm=True),
        Case("both_present_norm_off", INVALID, ("qRstd,kRstd", "require Q/K L2Norm backward"),
             q_rstd=True, k_rstd=True),
        Case("empty_pairs", INVALID, ("chunk_indices", "0", "at least 2"),
             cu=(0, 64), pairs=()),
        Case("first_pair_truncated", INVALID, ("chunk_indices", "1", "at least 2"),
             cu=(0, 64), pairs=(0,)),
        Case("second_pair_missing", INVALID, ("chunk_indices", "2", "at least 4"),
             tokens=65, cu=(0, 65), pairs=(0, 0)),
        Case("second_pair_truncated", INVALID, ("chunk_indices", "3", "at least 4"),
             tokens=65, cu=(0, 65), pairs=(0, 0, 0)),
        Case("noncanonical_pair", INVALID, ("chunk_indices[offset]", "canonical sequence-major order"),
             cu=(0, 64), pairs=(0, 1)),
        Case("both_absent_norm_off", 0),
        Case("both_present_norm_on", 0, q_rstd=True, k_rstd=True, norm=True),
        Case("valid_single_pair", 0, cu=(0, 64), pairs=(0, 0)),
        Case("valid_tail_pair", 0, tokens=65, cu=(0, 65), pairs=(0, 0, 0, 1)),
        Case("valid_empty_sequence", 0, cu=(0, 0, 64), pairs=(1, 0)),
    ]


def run_case(runtime, function, torch, device, case):
    from host_probe import HostDescriptors

    shapes = {
        "q": (1, 1, case.tokens, 128),
        "k": (1, 1, case.tokens, 128),
        "v": (1, 1, case.tokens, 128),
        "g": (1, case.tokens, 1),
        "beta": (1, case.tokens, 1),
        "a": (1, 1, case.tokens, 64),
        "do": (1, case.tokens, 1, 128),
    }
    values = {
        name: torch.empty(shape, dtype=torch.float32 if name in ("g", "beta") else torch.bfloat16,
                          device=device)
        for name, shape in shapes.items()
    }
    outputs = [torch.empty_like(values[name]) for name in ("q", "k", "v", "beta", "g")]
    context = HostDescriptors(runtime)
    null = ctypes.c_void_p()
    try:
        args = [context.tensor(values[name], name) for name in ("q", "k", "v", "g", "beta", "a", "do")]
        args += [null, null]  # initial_state、dht。
        for present, name in ((case.q_rstd, "q_rstd"), (case.k_rstd, "k_rstd")):
            value = torch.empty((1, 1, case.tokens), dtype=torch.float32, device=device) if present else None
            args.append(context.tensor(value, name) if present else null)
        args += [null, null, null]  # beta_raw、a_log、dt_bias。
        args += [context.int_array(case.cu), context.int_array(case.pairs)]
        args += [b"BNSD", 128 ** -0.5, 64, False, False, case.norm, False, False]
        args += [context.tensor(value, name) for value, name in
                 zip(outputs, ("dq", "dk", "dv", "dbeta", "dg"))]
        args += [null, null, null]  # dh0、d_a_log、d_dt_bias。
        workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
        runtime._recent_error_message()
        code = int(function(*args, ctypes.byref(workspace), ctypes.byref(executor)))
        message = runtime._recent_error_message()
        passed = code == case.expected and all(fragment in message for fragment in case.fragments)
        passed = passed and bool(executor.value) == (case.expected == 0)
        return dict(case=asdict(case), actual_code=code, message=message, passed=passed,
                    workspace_bytes=workspace.value, executor_created=bool(executor.value), launched=False)
    finally:
        context.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="显式加载实际 Bwd Host TU .so，采用 RTLD_LAZY")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative-only", action="store_true", help="仅执行参数校验早退用例")
    parser.add_argument("--case", action="append", choices=[case.name for case in cases()],
                        help="只执行指定用例，可多次传入")
    parser.add_argument("--list", action="store_true", help="列出选中的用例，无需 SDK")
    options = parser.parse_args()
    selected = [case for case in cases()
                if (not options.negative_only or case.expected != 0)
                and (not options.case or case.name in options.case)]
    if not selected:
        parser.error("没有选中的用例")
    if options.list:
        print(json.dumps([asdict(case) for case in selected], ensure_ascii=False, indent=2))
        return 0
    if options.output is None:
        parser.error("执行 SDK 测试需要 --output")
    if options.output.exists():
        raise FileExistsError(options.output)
    if options.library is not None and not options.library.is_file():
        parser.error(f"共享库不存在 {options.library}")

    import torch
    import torch_npu
    import fla_npu

    # 复用仓内实际 ACL descriptor/symbol helper，避免复制 SDK 描述符实现。
    probe_directory = Path(__file__).resolve().parents[2] / "chunk_gated_delta_rule_bwd_finalize" / "scripts"
    sys.path.insert(0, str(probe_directory))
    from host_probe import HostProbe

    torch.npu.set_device(options.device)
    runtime = HostProbe()
    library_path = options.library.resolve() if options.library is not None else None
    if library_path is not None:
        library = ctypes.CDLL(str(library_path), mode=os.RTLD_LAZY | os.RTLD_LOCAL)
        function = library.aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize
    else:
        function = runtime.symbol("aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize")
    function.argtypes = ([ctypes.c_void_p] * 16 + [ctypes.c_char_p, ctypes.c_double, ctypes.c_int64]
                         + [ctypes.c_bool] * 5 + [ctypes.c_void_p] * 8
                         + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    function.restype = ctypes.c_int
    records = []
    for case in selected:
        print(json.dumps(dict(running=case.name)), flush=True)
        records.append(run_case(runtime, function, torch, f"npu:{options.device}", case))
    maps_path = Path("/proc/self/maps")
    result = dict(
        schema="gdn-bwd-host-contract/v1",
        verdict="pass" if all(record["passed"] for record in records) else "fail",
        launched=False,
        negative_only=options.negative_only,
        device=options.device,
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
        fla_npu_file=str(Path(fla_npu.__file__).resolve()),
        library_requested=str(library_path) if library_path is not None else None,
        library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest() if library_path else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host_probe_sha256=hashlib.sha256((probe_directory / "host_probe.py").read_bytes()).hexdigest(),
        proc_maps=maps_path.read_text(encoding="utf-8") if maps_path.exists() else None,
        cases=records,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(verdict=result["verdict"], passed=sum(record["passed"] for record in records),
                         total=len(records))))
    return 0 if result["verdict"] == "pass" else 99


if __name__ == "__main__":
    raise SystemExit(main())
