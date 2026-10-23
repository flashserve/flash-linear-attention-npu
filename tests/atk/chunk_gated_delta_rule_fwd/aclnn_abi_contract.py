#!/usr/bin/env python3
"""校验 chunk_gated_delta_rule_fwd 的稳定 ACLNN ABI 与 Stable adapter 映射。"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h"
)
IMPLEMENTATION = HEADER.with_suffix(".cpp")
STABLE = ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py"
PREPARE_API = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/"
    "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_prepare.cpp"
)
PREPARE_TILING = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/"
    "op_host/chunk_gated_delta_rule_fwd_prepare_tiling.cpp"
)
PREPARE_KERNEL = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/"
    "op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h"
)
FWD_O_TILING = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/"
    "op_host/chunk_fwd_o_tiling_processor.h"
)
FWD_O_KERNEL = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp"
)
SYMBOL = "aclnnChunkGatedDeltaRuleFwd"

EXPECTED_PARAMETERS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "aLogOptional",
    "dtBiasOptional",
    "initialStateOptional",
    "cuSeqlensOptional",
    "chunkIndicesOptional",
    "layout",
    "scale",
    "chunkSize",
    "useExp2",
    "useQkL2norm",
    "allowNegEigval",
    "stateVFirst",
    "oOut",
    "finalStateOutOptional",
    "qHatOutOptional",
    "kHatOutOptional",
    "qRstdOutOptional",
    "kRstdOutOptional",
    "betaEffOutOptional",
    "gCumsumOutOptional",
    "aOutOptional",
    "hOutOptional",
    "workspaceSize",
    "executor",
)

EXPECTED_TYPES = (
    *("const aclTensor *",) * 8, *("const aclIntArray *",) * 2,
    "const char *", "double", "int64_t", * ("bool",) * 4,
    *("const aclTensor *",) * 10, "uint64_t *", "aclOpExecutor **",
)


def _parameter_names() -> tuple[str, ...]:
    text = HEADER.read_text(encoding="utf-8")
    match = re.search(
        rf"ACLNN_API\s+aclnnStatus\s+{SYMBOL}GetWorkspaceSize\s*\((.*?)\);",
        text,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"未找到 {SYMBOL}GetWorkspaceSize 的 ACLNN_API 声明")
    names = []
    for declaration in match.group(1).split(","):
        name = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration.strip())
        if name is None:
            raise RuntimeError(f"无法解析 ACLNN 参数：{declaration!r}")
        names.append(name.group(1))
    return tuple(names)


def _check_stable_adapter():
    path = ROOT / "torch_custom/fla_npu/tools/op_abi_validate.py"
    spec = importlib.util.spec_from_file_location("gdn_abi_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = module.adapter_calls(ROOT / "torch_custom/fla_npu/csrc/src")
    if SYMBOL not in calls:
        raise SystemExit(f"missing Stable adapter for {SYMBOL}")
    headers = module.parse_headers(HEADER.parent, {SYMBOL})
    problems = module.compare(headers[SYMBOL], calls[SYMBOL][1])
    if problems:
        raise SystemExit(f"Stable adapter ABI mismatch: {problems}")
    text = HEADER.read_text(encoding="utf-8")
    body = re.search(rf"{SYMBOL}GetWorkspaceSize\s*\((.*?)\);", text, re.S).group(1)
    actual = tuple(re.sub(r"\s+", " ", re.sub(r"[A-Za-z_]\w*\s*$", "", declaration).strip())
                   for declaration in body.split(","))
    canonical = lambda value: re.sub(r"\s*\*\s*", "*", value)
    if tuple(map(canonical, actual)) != tuple(map(canonical, EXPECTED_TYPES)):
        raise SystemExit(f"ACLNN full parameter types changed: {actual}")


def main() -> None:
    parameters = _parameter_names()
    _check_stable_adapter()
    if parameters != EXPECTED_PARAMETERS:
        raise SystemExit(
            "ACLNN 参数合同不一致：\n"
            f"expected={EXPECTED_PARAMETERS}\nactual={parameters}"
        )
    implementation = IMPLEMENTATION.read_text(encoding="utf-8")
    required_patterns = {
        "final-state selector": r"const\s+bool\s+outputFinalState\s*=\s*"
        r"params\.finalStateOutOptional\s*!=\s*nullptr\s*;",
        "BNSD legacy layout": r'std::strcmp\(params\.layout,\s*"BNSD"\)\s*==\s*0',
        "NTD legacy layout": r'std::strcmp\(params\.layout,\s*"NTD"\)\s*==\s*0',
        "prepare path A5 guard": r"if\s*\(UsePreparePath\(params\)\)\s*\{\s*"
        r"CHECK_COND\(IsAscend950\(\),\s*ACLNN_ERR_PARAM_INVALID",
        "direct BSND output": r"ChunkFwdO\(.*?params\.useExp2,\s*params\.stateVFirst,\s*"
        r'"BSND",\s*params\.oOut,\s*executorPtr\)',
        "gCumsum scratch": r"gCumsumCompute\s*=\s*executorPtr->AllocTensor",
        "A scratch": r"aCompute\s*=\s*executorPtr->AllocTensor",
    }
    missing = [
        name
        for name, pattern in required_patterns.items()
        if re.search(pattern, implementation, re.DOTALL) is None
    ]
    if missing:
        raise SystemExit(f"ACLNN 默认路径映射缺失：{missing}")

    prepare_api = PREPARE_API.read_text(encoding="utf-8")
    prepare_tiling = PREPARE_TILING.read_text(encoding="utf-8")
    prepare_kernel = PREPARE_KERNEL.read_text(encoding="utf-8")
    forbidden_exp2_guards = {
        "prepare ACLNN": (prepare_api, r"CHECK_COND\(params\.useExp2"),
        "prepare tiling": (prepare_tiling, r"if\s*\(\s*!useExp2\s*\)"),
        "prepare Torch wrapper": (STABLE.read_text(encoding="utf-8"),
                                  r"use_exp2 currently only supports True"),
    }
    rejected = [
        name for name, (text, pattern) in forbidden_exp2_guards.items()
        if re.search(pattern, text) is not None
    ]
    if rejected:
        raise SystemExit(f"useExp2=false 仍被拒绝：{rejected}")
    if re.search(
        r"float\s+scale\s*=\s*\(useExp2\s*!=\s*0\)\s*\?\s*kGdnRcpLn2\s*:\s*1\.0f",
        prepare_kernel,
    ) is None:
        raise SystemExit("prepare kernel 缺少 useExp2 对应的累计缩放选择")
    for helper in (r"GateLowerLVF", r"ScaleRowsBetaExp2gVF<InDtype,"):
        prefix = helper + (r"\s*" if "<" in helper else r"<")
        pattern = r"if\s*\(useExp2\s*!=\s*0\)\s*\{\s*" + prefix + r"true>.*?}\s*else\s*\{\s*" + prefix + r"false>"
        if re.search(pattern, prepare_kernel, re.S) is None:
            raise SystemExit(f"prepare kernel 缺少 useExp2 分支选择: {helper}")
    if implementation.count("params.useExp2") < 4:
        raise SystemExit("大融合新路径未完整透传 useExp2")

    fwd_o_tiling = FWD_O_TILING.read_text(encoding="utf-8")
    fwd_o_kernel = FWD_O_KERNEL.read_text(encoding="utf-8")
    if "processor.GetOutputTokenFirst()" not in (ROOT / (
        "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_host/chunk_fwd_o_tiling.cpp"
    )).read_text(encoding="utf-8"):
        raise SystemExit("chunk_fwd_o TilingKey 未按输出的 token-first 分类")
    if re.search(r"if constexpr\s*\(OutputTokenFirst\).*?"
                 r"ChunkFwdOA5DispatchByGateType<UseExp2>",
                 fwd_o_kernel, re.DOTALL) is None:
        raise SystemExit("chunk_fwd_o 未按输出的 token-first 分类选择优化实现")
    fwd_o_tiling_key = (ROOT / (
        "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o_tiling_key.h"
    )).read_text(encoding="utf-8")
    if len(re.findall(r"CHUNK_FWD_O_TPL_SEL_ENTRY\([01], [01]\)", fwd_o_tiling_key)) != 3:
        raise SystemExit("chunk_fwd_o 应只注册三个有效的 useExp2/token-first TilingKey 组合")

    print(
        json.dumps(
            {
                "schema": "chunk-gated-delta-rule-fwd-aclnn-abi/v1",
                "result": "passed",
                "symbol": SYMBOL,
                "parameter_count": len(parameters),
                "final_state_selector": "finalStateOutOptional != nullptr",
                "legacy_layouts": ["BNSD", "NTD"],
                "prepare_path_use_exp2": [False, True],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
