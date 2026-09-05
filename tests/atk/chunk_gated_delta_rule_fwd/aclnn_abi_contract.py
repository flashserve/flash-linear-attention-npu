#!/usr/bin/env python3
"""校验 chunk_gated_delta_rule_fwd 的稳定 ACLNN ABI 与 ctypes 映射。"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h"
)
IMPLEMENTATION = HEADER.with_suffix(".cpp")
CTYPES = ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
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

EXPECTED_CTYPES = (
    *("ctypes.c_void_p",) * 10,
    "ctypes.c_char_p",
    "ctypes.c_double",
    "ctypes.c_int64",
    "ctypes.c_bool",
    "ctypes.c_bool",
    "ctypes.c_bool",
    *("ctypes.c_void_p",) * 10,
    "ctypes.POINTER(ctypes.c_uint64)",
    "ctypes.POINTER(ctypes.c_void_p)",
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


def _expression_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call) and _expression_name(node.func) == "ctypes.POINTER":
        if len(node.args) != 1:
            raise RuntimeError("ctypes.POINTER 参数数量异常")
        return f"ctypes.POINTER({_expression_name(node.args[0])})"
    raise RuntimeError(f"无法解析 ctypes 类型表达式：{ast.dump(node)}")


def _ctypes_signature() -> tuple[str, ...]:
    tree = ast.parse(CTYPES.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_GET_WORKSPACE_ARGTYPES"
                   for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == SYMBOL:
                if not isinstance(value, ast.List):
                    raise RuntimeError(f"{SYMBOL} ctypes 签名不是列表")
                return tuple(_expression_name(item) for item in value.elts)
    raise RuntimeError(f"未找到 {SYMBOL} 的 ctypes 签名")


def main() -> None:
    parameters = _parameter_names()
    ctypes_signature = _ctypes_signature()
    if parameters != EXPECTED_PARAMETERS:
        raise SystemExit(
            "ACLNN 参数合同不一致：\n"
            f"expected={EXPECTED_PARAMETERS}\nactual={parameters}"
        )
    if ctypes_signature != EXPECTED_CTYPES:
        raise SystemExit(
            "ctypes 参数合同不一致：\n"
            f"expected={EXPECTED_CTYPES}\nactual={ctypes_signature}"
        )

    implementation = IMPLEMENTATION.read_text(encoding="utf-8")
    required_patterns = {
        "final-state selector": r"const\s+bool\s+outputFinalState\s*=\s*"
        r"params\.finalStateOutOptional\s*!=\s*nullptr\s*;",
        "BNSD default path": r'std::strcmp\(params\.layout,\s*"BNSD"\)\s*==\s*0',
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

    print(
        json.dumps(
            {
                "schema": "chunk-gated-delta-rule-fwd-aclnn-abi/v1",
                "result": "passed",
                "symbol": SYMBOL,
                "parameter_count": len(parameters),
                "final_state_selector": "finalStateOutOptional != nullptr",
                "supported_layout": "BNSD",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
