#!/usr/bin/env python3
"""Validate the checked-in chunk_kda_fwd ATK manifests without an NPU."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import yaml


HERE = Path(__file__).resolve()
OP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
GEN_PATH = OP_DIR / "gen_chunk_kda_fwd.py"
EXECUTOR_PATH = OP_DIR / "executor_chunk_kda_fwd.py"
STRESS_PATH = OP_DIR / "stress_npu_determinism.py"
PREPARE_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h"
)
KDA_KERNEL_ROOT = (
    REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel"
)
KDA_ROOT = REPO_ROOT / "fla/ops/ascendc/kda"
KDA_COMMON_PATH = KDA_KERNEL_ROOT / "chunk_kda_fwd_common.h"
KDA_ENTRY_PATH = KDA_KERNEL_ROOT / "chunk_kda_fwd.cpp"
KDA_ARCH35_FINALIZE_PATH = (
    KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_finalize.h"
)
KDA_OP_API_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp"
)
KDA_ACLNN_HEADER_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.h"
)
KDA_L0_OP_API_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/chunk_kda_fwd.cpp"
)
KDA_L0_HEADER_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/chunk_kda_fwd.h"
)
KDA_OP_DEF_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_def.cpp"
)
KDA_TILING_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp"
)
KDA_TILING_HEADER_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.h"
)
FWD_H_ROOT = KDA_KERNEL_ROOT / "fwd_h"
FWD_H_SCHEDULER_PATHS = (
    FWD_H_ROOT / "gemm/block/block_scheduler_kda_fwd_h.hpp",
    FWD_H_ROOT / "arch35/gemm/block/block_scheduler_kda_fwd_h.hpp",
)
FWD_H_KERNEL_PATHS = (
    FWD_H_ROOT / "gemm/kernel/kda_fwd_h_kernel.hpp",
    FWD_H_ROOT / "arch35/gemm/kernel/kda_fwd_h_kernel.hpp",
)
KDA_KERNEL_UTILS_ROOT = (
    REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/kernel_utils"
)
KDA_MMAD_MULTI_PATH = (
    KDA_KERNEL_UTILS_ROOT / "block/block_mmad_pingpong_tla_multi.hpp"
)
KDA_MMAD_PATHS = (
    KDA_KERNEL_UTILS_ROOT / "block/block_mmad_pingpong_tla.hpp",
    KDA_MMAD_MULTI_PATH,
    KDA_KERNEL_UTILS_ROOT / "block/block_mmad_pingpong_tla_preloadA_l1B.hpp",
)
KDA_PRIVATE_UTILITY_PATHS = (
    *KDA_MMAD_PATHS,
    KDA_KERNEL_UTILS_ROOT / "tile/copy_l0c_to_ub.hpp",
    KDA_KERNEL_UTILS_ROOT / "vector/regbase.hpp",
)
KDA_PRIVATE_FWD_H_PATHS = (
    FWD_H_ROOT / "chunk_kda_fwd_h_struct.h",
    FWD_H_ROOT / "epilogue/kda_fwd_h_epilogue_policies.hpp",
    FWD_H_ROOT / "epilogue/block/block_epilogue_kda_fwdh_update.hpp",
    FWD_H_ROOT / "epilogue/block/block_epilogue_kda_fwdh_vnew.hpp",
    FWD_H_ROOT / "gemm/block/block_scheduler_kda_fwd_h.hpp",
    FWD_H_ROOT / "gemm/kernel/kda_fwd_h_kernel.hpp",
    FWD_H_ROOT / "arch35/epilogue/kda_fwd_h_epilogue_policies.hpp",
    FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_regbase.hpp",
    FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_update.hpp",
    FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_vnew.hpp",
    FWD_H_ROOT / "arch35/gemm/block/block_scheduler_kda_fwd_h.hpp",
    FWD_H_ROOT / "arch35/gemm/kernel/kda_fwd_h_kernel.hpp",
)
POST_WU_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_post_wu.h"
)
VNEW_EPILOGUE_PATH = (
    FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_vnew.hpp"
)
FWD_H_UPDATE_EPILOGUE_PATH = (
    FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_update.hpp"
)
SOCS = {"ascend910b", "ascend910_93", "ascend950"}
ACCURACY_STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key"}
GATE_VARIANTS = {
    (False, False, False),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
}
OUTPUT_POLICIES = {(False, False), (False, True), (True, False), (True, True)}
SHAPE_VARIANTS = {
    (64, 128, 128),
    (128, 128, 128),
    (64, 128, 256),
    (64, 16, 128),
    (64, 256, 128),
}
ENUMERATED_YAML_INPUTS = {
    "soc", "batch", "head", "value_head", "total_tokens", "key_dim", "value_dim",
    "chunk_size", "layout", "scale", "q_dtype", "g_dtype", "beta_dtype",
    "initial_state", "output_final_state", "varlen", "explicit_chunk_indices",
    "safe_gate", "lower_bound", "use_gate_in_kernel", "dt_bias",
    "disable_recompute", "return_intermediate_states", "state_v_first",
    "negative_case", "tiling_key", "expected_tiling_key", "execution_mode",
    "coverage_only", "runtime_status", "data_profile", "data_scale", "gate_scale",
    "qk_scale", "v_scale", "beta_scale", "beta_bias", "a_log_scale",
    "dt_bias_scale", "dt_bias_mean", "beta_low", "beta_high", "state_scale", "profile",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("chunk_kda_fwd_generator", GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_stress_driver(expected_count: int) -> None:
    spec = importlib.util.spec_from_file_location("chunk_kda_fwd_stress", STRESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {STRESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module._load_specs(OP_DIR / "atk_chunk_kda_fwd_mss.json")
    selected_ids = [int(item["case_id"]) for item in selected]
    if selected_ids != list(range(expected_count)):
        raise ValueError(
            f"determinism driver selected MSS IDs {selected_ids}, "
            f"expected 0--{expected_count - 1}"
        )
    if {str(item.get("route", "")) for item in selected} != {"ascendc"}:
        raise ValueError("determinism driver must select only the public ascendc route")


def _read(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a JSON list")
    return value


def _spec(payload: dict) -> dict:
    for item in payload.get("inputs", []):
        if item.get("name") == "case_spec":
            value = json.loads(item["range_values"])
            if not isinstance(value, dict):
                raise ValueError("case_spec must decode to an object")
            return value
    raise ValueError(f"payload {payload.get('id')} has no case_spec")


def _input_values(payload: dict) -> dict[str, object]:
    return {
        str(item["name"]): item.get("range_values")
        for item in payload.get("inputs", [])
        if isinstance(item, dict) and "name" in item
    }


def _check_manifest(
    path: Path,
    expected_count: int,
    expected_keys: set[int],
    *,
    expected_soc: str = "all",
    expected_platforms: set[str] = SOCS,
) -> list[dict]:
    cases = _read(path)
    if len(cases) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} cases, got {len(cases)}")
    if [int(case["id"]) for case in cases] != list(range(expected_count)):
        raise ValueError(f"{path.name}: ids must be contiguous from zero")
    specs = []
    for case in cases:
        spec = _spec(case)
        specs.append(spec)
        values = _input_values(case)
        if int(case["id"]) != int(spec["case_id"]):
            raise ValueError(f"{path.name}: id/case_spec mismatch")
        aliases = {
            "batch": "B",
            "head": "H",
            "value_head": "HV",
            "total_tokens": "T",
            "key_dim": "K",
            "value_dim": "V",
        }
        for input_name, spec_name in aliases.items():
            if values.get(input_name) != spec[spec_name]:
                raise ValueError(
                    f"{path.name}: {input_name} does not match case_spec.{spec_name}"
                )
        key = int(spec["tiling_key"])
        expected = 2 if (int(spec["chunk_size"]), int(spec["K"]), int(spec["V"])) == (64, 128, 128) else 1
        if key != expected or int(spec["expected_tiling_key"]) != expected:
            raise ValueError(f"{path.name}: stale tiling key in {spec['case_key']}")
        if key not in expected_keys:
            raise ValueError(f"{path.name}: unexpected key {key}")
        if spec.get("soc") != expected_soc:
            raise ValueError(
                f"{path.name}: expected soc={expected_soc}, got {spec.get('soc')}"
            )
        if set(spec.get("target_platforms", [])) != expected_platforms:
            raise ValueError(
                f"{path.name}: target platform matrix drifted"
            )
        if bool(spec.get("coverage_only")):
            raise ValueError(f"{path.name}: runtime manifests cannot be coverage-only")
    return specs


def _check_source_evidence() -> None:
    tiling = (REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp").read_text(encoding="utf-8")
    kernel = (REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    if "SetTilingKey(useChunk64K128V128Template ? 2 : 1)" not in tiling:
        raise ValueError("host tiling key predicate is missing")
    for key in (1, 2):
        if f"TILING_KEY_IS({key})" not in kernel:
            raise ValueError(f"kernel source has no TILING_KEY_IS({key}) dispatch")

    _check_cpp_arch_branch_selector()
    finalize = _cpp_source_for_aicore(
        KDA_ARCH35_FINALIZE_PATH.read_text(encoding="utf-8"), 310
    )
    output_cube = _normalize_source(
        _source_braced_block(
            finalize, "__aicore__ inline void ComputeOutputCube("
        )
    )
    staged_dispatch = (
        "if constexpr (!USE_FP32_V_NEW) { "
        "if (BT_ == 64 && K_ == 128 && V_ == 128 && curT == BT_) { "
        "ComputeOutputCubeStagedArch35(b, hv, chunkIdx, start, curT); return; } }"
    )
    if staged_dispatch not in output_cube:
        raise ValueError(
            "A5 staged output MMAD must remain restricted to low-precision "
            "BT64/K128/V128 operands"
        )

    dense_predicate = (
        "!isVarLen_ && T_ % BT_ == 0 && BT_ == 64 && K_ == 128 && V_ == 128"
    )
    process_out_aic = _normalize_source(
        _source_braced_block(finalize, "__aicore__ inline void ProcessOutAic()")
    )
    process_out_aiv = _normalize_source(
        _source_braced_block(finalize, "__aicore__ inline void ProcessOutAiv()")
    )
    aic_dense_dispatch = (
        "if constexpr (!USE_FP32_V_NEW) { "
        f"if ({dense_predicate}) {{ ProcessOutAicPipelinedArch35(); return; }} }}"
    )
    aiv_dense_dispatch = (
        "if constexpr (IsSameType<T, bfloat16_t>::value) { "
        f"if ({dense_predicate}) {{ return; }} }}"
    )
    if aic_dense_dispatch not in process_out_aic:
        raise ValueError("A5 dense staged AIC dispatch contract changed")
    if aiv_dense_dispatch not in process_out_aiv:
        raise ValueError("A5 dense staged AIV ownership contract changed")

    op_api = KDA_OP_API_PATH.read_text(encoding="utf-8")
    for token in (
        "const bool useDenseA5FastPath =",
        "IsAscend950() && info.totalChunks > 1 && !useDenseA5FastPath",
        "for (int64_t stage = KDA_STAGE_GATE_PREPARE; stage < KDA_STAGE_COUNT;",
        "result = launchStage(stage);",
        "result = launchStage(KDA_STAGE_FULL);",
    ):
        if token not in op_api:
            raise ValueError(f"A5 full/staged launch contract is missing {token}")
    _check_stage_dependency_contract()
    _check_direct_update_ub_contract()

    executor_source = EXECUTOR_PATH.read_text(encoding="utf-8")
    _check_uniform_executor_contract(executor_source)
    _check_kernel_launch_trace_contract(executor_source)
    _check_public_fp16_reference_contract(executor_source)


def _source_section(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return " ".join(source[start:end].split())


def _source_braced_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    state = "code"
    index = opening
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 1
        elif state in {"string", "character"}:
            delimiter = '"' if state == "string" else "'"
            if char == "\\":
                index += 1
            elif char == delimiter:
                state = "code"
        elif char == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif char == '"':
            state = "string"
        elif char == "'":
            state = "character"
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
        index += 1
    raise ValueError(f"unbalanced C++ block after {marker!r}")


def _normalize_source(source: str) -> str:
    return " ".join(source.split())


def _check_direct_update_ub_contract() -> None:
    source = _cpp_source_for_aicore(
        FWD_H_UPDATE_EPILOGUE_PATH.read_text(encoding="utf-8"), 310
    )
    normalized_source = _normalize_source(source)
    constructor = _normalize_source(
        _source_braced_block(
            source, "BlockEpilogue(Arch::Resource<ArchTag> &resource)"
        )
    )
    class_contracts = (
        "static constexpr uint32_t ROW_TILE = 16;",
        "static constexpr uint32_t MAX_UPDATE_V_DIM = 256;",
        "static constexpr uint32_t DIRECT_C2_SLOT_BYTES = 32 * 1024;",
        "static constexpr uint32_t VNEW_GATE_REGION_OFFSET = 160 * 1024;",
        "static constexpr uint32_t MAX_UPDATE_TILE_BYTES = ROW_TILE * "
        "MAX_UPDATE_V_DIM * sizeof(float);",
    )
    for contract in class_contracts:
        if normalized_source.count(contract) != 1:
            raise ValueError(
                f"A5 FwdH UB class contract is missing or duplicated: {contract}"
            )

    constructor_contracts = (
        "static_assert( PING_BUF_0_OFFSET + DIRECT_C2_SLOT_BYTES <= "
        "PING_BUF_3_OFFSET && PING_BUF_3_OFFSET + MAX_UPDATE_TILE_BYTES <= "
        "PONG_BUF_0_OFFSET, \"ping update scratch overlaps the direct C2 "
        "payload\");",
        "static_assert( PONG_BUF_0_OFFSET + DIRECT_C2_SLOT_BYTES <= "
        "PONG_BUF_3_OFFSET && PONG_BUF_3_OFFSET + MAX_UPDATE_TILE_BYTES <= "
        "VNEW_GATE_REGION_OFFSET, \"pong update scratch overlaps the direct "
        "C2 payload\");",
        "hUpdateUbTensor_ping = resource.ubBuf.template "
        "GetBufferByByte<float>(PING_BUF_0_OFFSET);",
        "hUbTensor_ping = resource.ubBuf.template "
        "GetBufferByByte<HElementOutput>(PING_BUF_3_OFFSET);",
        "finalOutputUbTensor_ping = resource.ubBuf.template "
        "GetBufferByByte<FinalStateElement>(PING_BUF_3_OFFSET);",
        "hUpdateUbTensor_pong = resource.ubBuf.template "
        "GetBufferByByte<float>(PONG_BUF_0_OFFSET);",
        "hUbTensor_pong = resource.ubBuf.template "
        "GetBufferByByte<HElementOutput>(PONG_BUF_3_OFFSET);",
        "finalOutputUbTensor_pong = resource.ubBuf.template "
        "GetBufferByByte<FinalStateElement>(PONG_BUF_3_OFFSET);",
    )
    for contract in constructor_contracts:
        if contract not in constructor:
            raise ValueError(
                f"A5 FwdH direct-C2/update UB separation is missing {contract}"
            )

    vnew_constructor = _normalize_source(
        _source_braced_block(
            _cpp_source_for_aicore(
                VNEW_EPILOGUE_PATH.read_text(encoding="utf-8"), 310
            ),
            "BlockEpilogue(Arch::Resource<ArchTag> &resource)",
        )
    )
    if "constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;" not in vnew_constructor:
        raise ValueError("A5 V1 gate region no longer starts at the asserted 160 KiB boundary")

    op_api = _normalize_source(KDA_OP_API_PATH.read_text(encoding="utf-8"))
    if op_api.count("constexpr int64_t MAX_KDA_V_DIM = 256;") != 1:
        raise ValueError("public V dimension limit drifted from the A5 update UB contract")
    if "info.vDim <= MAX_KDA_V_DIM" not in op_api:
        raise ValueError("public V dimension validation bypasses MAX_KDA_V_DIM")

    fwd_h = _normalize_source(
        _cpp_source_for_aicore(
            FWD_H_KERNEL_PATHS[1].read_text(encoding="utf-8"), 310
        )
    )
    direct_predicate = (
        "useDirectFp32Ub = !FP32_C2 && !FP32_H && "
        "std::is_same<ElementVWork, float>::value && !HI_LO_C2 && "
        "!isVariedLen && chunkSize <= 64 && seqlen % chunkSize == 0 && "
        "kHeadDim == 128 && vHeadDim == 128 && denseTaskCount >= "
        "AscendC::GetBlockNum();"
    )
    if fwd_h.count(direct_predicate) != 2:
        raise ValueError("A5 direct-C2 eligibility predicate drifted")


def _check_public_fp16_reference_contract(source: str) -> None:
    module = ast.parse(source)
    parents = {
        id(child): node
        for node in ast.walk(module)
        for child in ast.iter_child_nodes(node)
    }
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    required = {
        "_project_public_output",
        "_reference_impl",
        "_reference_model_parallel",
    }
    missing = required.difference(functions)
    if missing:
        raise ValueError(
            f"executor is missing FP16 public reference functions: {sorted(missing)}"
        )

    def expression(source_text: str) -> ast.expr:
        return ast.parse(source_text, mode="eval").body

    def statement(source_text: str) -> ast.stmt:
        return ast.parse(source_text).body[0]

    def same_ast(actual: ast.AST, expected: ast.AST) -> bool:
        return ast.dump(actual, include_attributes=False) == ast.dump(
            expected, include_attributes=False
        )

    helper = functions["_project_public_output"]
    if [argument.arg for argument in helper.args.args] != ["tensor", "spec", "name"]:
        raise ValueError("FP16 public projection arguments changed")
    if len(helper.body) != 4:
        raise ValueError("FP16 public projection must contain exactly four statements")

    dtype_guard, finite_guard, limit_assignment, clamp_return = helper.body
    dtype_condition = expression('str(spec["q_dtype"]) != "fp16"')
    if not (
        isinstance(dtype_guard, ast.If)
        and not dtype_guard.orelse
        and len(dtype_guard.body) == 1
        and same_ast(dtype_guard.body[0], statement("return tensor"))
        and isinstance(dtype_guard.test, ast.BoolOp)
        and isinstance(dtype_guard.test.op, ast.Or)
        and len(dtype_guard.test.values) == 2
    ):
        raise ValueError("FP16 public projection dtype guard changed")
    dtype_matches = [
        value for value in dtype_guard.test.values if same_ast(value, dtype_condition)
    ]
    output_guards = [
        value for value in dtype_guard.test.values if not same_ast(value, dtype_condition)
    ]
    if len(dtype_matches) != 1 or len(output_guards) != 1:
        raise ValueError("FP16 public projection dtype guard changed")
    output_guard = output_guards[0]
    if not (
        isinstance(output_guard, ast.Compare)
        and isinstance(output_guard.left, ast.Name)
        and output_guard.left.id == "name"
        and len(output_guard.ops) == 1
        and isinstance(output_guard.ops[0], ast.NotIn)
        and len(output_guard.comparators) == 1
        and isinstance(output_guard.comparators[0], ast.Set)
        and {
            element.value
            for element in output_guard.comparators[0].elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        == {"attn_out", "v_new", "h"}
        | {"aqk", "akk", "w", "u", "qg", "kg"}
        and len(output_guard.comparators[0].elts) == 9
    ):
        raise ValueError("FP16 public projection output allowlist changed")
    if not (
        isinstance(finite_guard, ast.If)
        and not finite_guard.orelse
        and len(finite_guard.body) == 1
        and same_ast(
            finite_guard.test,
            expression("not torch.isfinite(tensor).all().item()"),
        )
        and same_ast(
            finite_guard.body[0],
            statement(
                'raise RuntimeError(f"{name} raw CPU reference contains NaN or Inf")'
            ),
        )
    ):
        raise ValueError("FP16 public projection finite-value guard changed")
    if not same_ast(
        limit_assignment,
        statement("limit = torch.finfo(torch.float16).max"),
    ):
        raise ValueError("FP16 public projection limit changed")
    if not same_ast(
        clamp_return,
        statement("return tensor.clamp(min=-limit, max=limit)"),
    ):
        raise ValueError("FP16 public projection clamp changed")

    expected_assignments = {
        "_reference_impl": {
            "o": ("out_block.permute(1, 0, 2)", "attn_out"),
            "aqk": ("qk", "aqk"),
            "akk": ("inverse", "akk"),
            "w_out": ("w_block", "w"),
            "u_out": ("u_block", "u"),
            "qg_out": ("qg_block", "qg"),
            "kg_out": ("kg_block", "kg"),
            "v_new_out": ("v_new_block", "v_new"),
            "h_out": ("previous", "h"),
        },
        "_reference_model_parallel": {
            "o": ("out_block", "attn_out"),
            "aqk": ("qk", "aqk"),
            "akk": ("inverse", "akk"),
            "w_out": ("w_block", "w"),
            "u_out": ("u_block", "u"),
            "qg_out": ("qg_block", "qg"),
            "kg_out": ("kg_block", "kg"),
            "v_new_out": ("v_new_block", "v_new"),
            "h_out": ("previous", "h"),
        },
    }
    expected_parents = {
        "_reference_impl": {
            "o": (ast.For, None),
            "aqk": (ast.For, None),
            "akk": (ast.For, None),
            "w_out": (ast.If, "export_full"),
            "u_out": (ast.If, "export_full"),
            "qg_out": (ast.If, "export_full"),
            "kg_out": (ast.If, "export_full"),
            "v_new_out": (ast.If, "export_full"),
            "h_out": (ast.If, "export_h"),
        },
        "_reference_model_parallel": {
            "o": (ast.For, None),
            "aqk": (ast.For, None),
            "akk": (ast.For, None),
            "w_out": (ast.For, None),
            "u_out": (ast.For, None),
            "qg_out": (ast.For, None),
            "kg_out": (ast.For, None),
            "v_new_out": (ast.For, None),
            "h_out": (ast.For, None),
        },
    }
    allowed_calls: set[int] = set()
    for function_name, expected_outputs in expected_assignments.items():
        found_outputs: set[str] = set()
        allowed_output_writes: set[int] = set()
        for assignment in (
            node
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Assign)
        ):
            if len(assignment.targets) != 1:
                continue
            target = assignment.targets[0]
            if not isinstance(target, ast.Subscript):
                continue
            while isinstance(target, ast.Subscript):
                target = target.value
            if not isinstance(target, ast.Name) or target.id not in expected_outputs:
                continue
            conversion = assignment.value
            if not (
                isinstance(conversion, ast.Call)
                and isinstance(conversion.func, ast.Attribute)
                and conversion.func.attr == "to"
                and len(conversion.args) == 1
                and isinstance(conversion.args[0], ast.Name)
                and conversion.args[0].id == "output_dtype"
                and not conversion.keywords
                and isinstance(conversion.func.value, ast.Call)
            ):
                raise ValueError(
                    f"{function_name} writes {target.id} without public projection"
                )
            projection = conversion.func.value
            if not (
                isinstance(projection.func, ast.Name)
                and projection.func.id == "_project_public_output"
                and len(projection.args) == 3
                and not projection.keywords
            ):
                raise ValueError(
                    f"{function_name} writes {target.id} without public projection"
                )
            expected_expression, expected_name = expected_outputs[target.id]
            if not (
                same_ast(projection.args[0], expression(expected_expression))
                and isinstance(projection.args[1], ast.Name)
                and projection.args[1].id == "spec"
                and isinstance(projection.args[2], ast.Constant)
                and projection.args[2].value == expected_name
            ):
                raise ValueError(
                    f"{function_name} projects the wrong value for {target.id}"
                )
            parent_type, parent_condition = expected_parents[function_name][target.id]
            parent = parents[id(assignment)]
            if not isinstance(parent, parent_type) or (
                parent_condition is not None
                and not (
                    isinstance(parent, ast.If)
                    and isinstance(parent.test, ast.Name)
                    and parent.test.id == parent_condition
                )
            ):
                raise ValueError(
                    f"{function_name} writes {target.id} under the wrong control flow"
                )
            if target.id in found_outputs:
                raise ValueError(
                    f"{function_name} projects {target.id} more than once"
                )
            found_outputs.add(target.id)
            allowed_calls.add(id(projection))
            allowed_output_writes.add(id(assignment.targets[0]))
        missing_outputs = set(expected_outputs).difference(found_outputs)
        if missing_outputs:
            raise ValueError(
                f"{function_name} does not project public outputs: "
                f"{sorted(missing_outputs)}"
            )
        all_output_writes: set[int] = set()
        for node in ast.walk(functions[function_name]):
            if not isinstance(node, ast.Subscript) or not isinstance(
                node.ctx, ast.Store
            ):
                continue
            root = node.value
            while isinstance(root, ast.Subscript):
                root = root.value
            if isinstance(root, ast.Name) and root.id in expected_outputs:
                all_output_writes.add(id(node))
        if all_output_writes != allowed_output_writes:
            raise ValueError(
                f"{function_name} has an unvalidated public output write"
            )

    all_projection_calls = {
        id(node)
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_project_public_output"
    }
    if all_projection_calls != allowed_calls:
        raise ValueError(
            "FP16 public projection must only occur at the 18 public output writes"
        )
    all_projection_name_loads = {
        id(node)
        for node in ast.walk(module)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "_project_public_output"
    }
    allowed_projection_name_loads = {
        id(node.func)
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and id(node) in allowed_calls
    }
    if all_projection_name_loads != allowed_projection_name_loads:
        raise ValueError(
            "FP16 public projection cannot be aliased or used outside public writes"
        )


def _strip_c_like_comments(source: str) -> str:
    result: list[str] = []
    state = "code"
    index = 0
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char == "\n":
                result.append(char)
                state = "code"
        elif state == "block_comment":
            if char == "\n":
                result.append(char)
            elif char == "*" and following == "/":
                state = "code"
                index += 1
        elif state in {"string", "character"}:
            result.append(char)
            if char == "\\" and following:
                result.append(following)
                index += 1
            elif char == ('"' if state == "string" else "'"):
                state = "code"
        elif char == "/" and following == "/":
            result.append(" ")
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            result.append(" ")
            state = "block_comment"
            index += 1
        else:
            result.append(char)
            if char == '"':
                state = "string"
            elif char == "'":
                state = "character"
        index += 1
    return "".join(result)


_CPP_UNKNOWN = object()


def _eval_cpp_ast(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return node.value
    if isinstance(node, ast.Name):
        return _CPP_UNKNOWN
    if isinstance(node, ast.UnaryOp):
        value = _eval_cpp_ast(node.operand)
        if value is _CPP_UNKNOWN:
            return _CPP_UNKNOWN
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Invert):
            return ~value
        return _CPP_UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [_eval_cpp_ast(value) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is not _CPP_UNKNOWN and not bool(value) for value in values):
                return False
            return _CPP_UNKNOWN if _CPP_UNKNOWN in values else True
        if isinstance(node.op, ast.Or):
            if any(value is not _CPP_UNKNOWN and bool(value) for value in values):
                return True
            return _CPP_UNKNOWN if _CPP_UNKNOWN in values else False
        return _CPP_UNKNOWN
    if isinstance(node, ast.Compare):
        left = _eval_cpp_ast(node.left)
        comparators = [_eval_cpp_ast(value) for value in node.comparators]
        if left is _CPP_UNKNOWN or _CPP_UNKNOWN in comparators:
            return _CPP_UNKNOWN
        values = [left, *comparators]
        for index, operator in enumerate(node.ops):
            lhs, rhs = values[index:index + 2]
            if isinstance(operator, ast.Eq):
                matched = lhs == rhs
            elif isinstance(operator, ast.NotEq):
                matched = lhs != rhs
            elif isinstance(operator, ast.Lt):
                matched = lhs < rhs
            elif isinstance(operator, ast.LtE):
                matched = lhs <= rhs
            elif isinstance(operator, ast.Gt):
                matched = lhs > rhs
            elif isinstance(operator, ast.GtE):
                matched = lhs >= rhs
            else:
                return _CPP_UNKNOWN
            if not matched:
                return False
        return True
    if isinstance(node, ast.BinOp):
        left = _eval_cpp_ast(node.left)
        right = _eval_cpp_ast(node.right)
        if left is _CPP_UNKNOWN or right is _CPP_UNKNOWN:
            return _CPP_UNKNOWN
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.BitAnd: lambda: left & right,
            ast.BitOr: lambda: left | right,
            ast.BitXor: lambda: left ^ right,
        }
        operation = operations.get(type(node.op))
        return _CPP_UNKNOWN if operation is None else operation()
    return _CPP_UNKNOWN


def _cpp_condition_for_aicore(expression: str, aicore: int) -> bool | None:
    expression = re.sub(
        r"\bdefined\s*\(\s*([A-Za-z_]\w*)\s*\)",
        lambda match: "1" if match.group(1) == "__CCE_AICORE__" else f"defined_{match.group(1)}",
        expression,
    )
    expression = re.sub(
        r"\bdefined\s+([A-Za-z_]\w*)",
        lambda match: "1" if match.group(1) == "__CCE_AICORE__" else f"defined_{match.group(1)}",
        expression,
    )
    expression = re.sub(r"\b__CCE_AICORE__\b", str(aicore), expression)
    expression = re.sub(r"\b(0[xX][0-9A-Fa-f]+|\d+)[uUlL]+\b", r"\1", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression).strip()
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    value = _eval_cpp_ast(parsed.body)
    return None if value is _CPP_UNKNOWN else bool(value)


def _cpp_source_for_aicore(source: str, aicore: int) -> str:
    """Select branches decidable for a target AiCore and retain unknown alternatives."""
    source = _strip_c_like_comments(source)
    lines = source.splitlines(keepends=True)
    output: list[str] = []
    stack: list[dict[str, bool]] = []
    current_keep = True
    directive = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
    index = 0
    while index < len(lines):
        line = lines[index]
        match = directive.match(line)
        if match is None:
            output.append(line if current_keep else "\n")
            index += 1
            continue

        physical_lines = [line]
        logical_line = line.rstrip("\r\n")
        while logical_line.rstrip().endswith("\\"):
            logical_line = logical_line.rstrip()[:-1]
            index += 1
            if index >= len(lines):
                raise ValueError("unterminated preprocessor line continuation")
            physical_lines.append(lines[index])
            logical_line += " " + lines[index].strip()
        logical_match = directive.match(logical_line)
        if logical_match is None:
            raise ValueError("invalid continued preprocessor directive")
        kind, expression = logical_match.groups()
        if kind in {"if", "ifdef", "ifndef"}:
            if kind == "ifdef":
                expression = f"defined({expression.strip()})"
            elif kind == "ifndef":
                expression = f"!defined({expression.strip()})"
            condition = _cpp_condition_for_aicore(expression, aicore)
            can_be_true = condition is not False
            can_be_false = condition is not True
            parent_keep = current_keep
            stack.append(
                {
                    "parent_keep": parent_keep,
                    "remaining_possible": parent_keep and can_be_false,
                    "seen_else": False,
                }
            )
            current_keep = parent_keep and can_be_true
        elif kind == "elif":
            if not stack or stack[-1]["seen_else"]:
                raise ValueError("orphan #elif in C++ source")
            condition = _cpp_condition_for_aicore(expression, aicore)
            can_be_true = condition is not False
            can_be_false = condition is not True
            remaining = stack[-1]["remaining_possible"]
            current_keep = remaining and can_be_true
            stack[-1]["remaining_possible"] = remaining and can_be_false
        elif kind == "else":
            if not stack or stack[-1]["seen_else"]:
                raise ValueError("orphan #else in C++ source")
            current_keep = stack[-1]["remaining_possible"]
            stack[-1]["remaining_possible"] = False
            stack[-1]["seen_else"] = True
        else:
            if not stack:
                raise ValueError("orphan #endif in C++ source")
            current_keep = stack.pop()["parent_keep"]
        output.extend("\n" for _ in physical_lines)
        index += 1
    if stack:
        raise ValueError("unterminated preprocessor branch in C++ source")
    return "".join(output)


def _check_cpp_arch_branch_selector() -> None:
    source = """\
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
A5_ONLY
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ != 310
NON_A5_ONLY
#else
UNREACHABLE
#endif
#if UNKNOWN_FEATURE
UNKNOWN_ON
#else
UNKNOWN_OFF
#endif
"""
    a5 = _cpp_source_for_aicore(source, 310)
    non_a5 = _cpp_source_for_aicore(source, 200)
    a5_tokens = set(a5.split())
    non_a5_tokens = set(non_a5.split())
    if "A5_ONLY" not in a5_tokens or {
        "NON_A5_ONLY", "UNREACHABLE"
    } & a5_tokens:
        raise ValueError("A5 preprocessor branch selector self-check failed")
    if "NON_A5_ONLY" not in non_a5_tokens or {
        "A5_ONLY", "UNREACHABLE"
    } & non_a5_tokens:
        raise ValueError("non-A5 preprocessor branch selector self-check failed")
    if not all(
        token in a5_tokens and token in non_a5_tokens
        for token in ("UNKNOWN_ON", "UNKNOWN_OFF")
    ):
        raise ValueError("unknown preprocessor branches must be retained conservatively")


def _cpp_function_parameter_names(source: str, marker: str) -> tuple[str, ...]:
    start = source.index(marker)
    opening = source.index("(", start + len(marker))
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stack: list[str] = []
    parameters: list[str] = []
    parameter_start = opening + 1
    closing = -1
    for index in range(opening + 1, len(source)):
        char = source[index]
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if char == ")" and not stack:
                parameters.append(source[parameter_start:index])
                closing = index
                break
            if not stack or stack.pop() != char:
                raise ValueError(f"unbalanced C++ parameter list for {marker}")
        elif char == "," and not stack:
            parameters.append(source[parameter_start:index])
            parameter_start = index + 1
    if closing < 0:
        raise ValueError(f"unterminated C++ parameter list for {marker}")

    names = []
    for parameter in parameters:
        parameter = parameter.strip()
        if not parameter or parameter == "void":
            continue
        match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", parameter)
        if match is None:
            raise ValueError(f"cannot parse C++ parameter name from {parameter!r}")
        names.append(match.group(1))
    return tuple(names)


def _check_hidden_fp32_state_tiling_contract() -> None:
    tiling = _cpp_source_for_aicore(
        KDA_TILING_PATH.read_text(encoding="utf-8"), 310
    )
    normalized_tiling = _normalize_source(tiling)
    index_contracts = {
        "INPUT_STAGE_V_NEW_FP32_IDX": 20,
        "INPUT_STAGE_H_FP32_IDX": 21,
        "INPUT_STAGE_AKK_FP32_IDX": 22,
        "INPUT_STAGE_W_FP32_IDX": 23,
        "OUTPUT_V_NEW_FP32_IDX": 13,
        "OUTPUT_H_FP32_IDX": 14,
        "OUTPUT_AKK_FP32_IDX": 15,
        "OUTPUT_W_FP32_IDX": 16,
    }
    for name, index in index_contracts.items():
        if re.search(
            rf"constexpr\s+size_t\s+{name}\s*=\s*{index}\s*;", tiling
        ) is None:
            raise ValueError(f"{name} must remain ABI index {index}")

    tiling_header = _normalize_source(
        _cpp_source_for_aicore(
            KDA_TILING_HEADER_PATH.read_text(encoding="utf-8"), 310
        )
    )
    for field in (
        "TILING_DATA_FIELD_DEF(bool, storeVNewFp32);",
        "TILING_DATA_FIELD_DEF(bool, storeHFp32);",
        "TILING_DATA_FIELD_DEF(bool, storeAkkFp32);",
        "TILING_DATA_FIELD_DEF(bool, storeWFp32);",
        "TILING_DATA_FIELD_DEF(int64_t, vNewFp32StorageOffset);",
        "TILING_DATA_FIELD_DEF(int64_t, hFp32StorageOffset);",
        "TILING_DATA_FIELD_DEF(int64_t, stateOperandFp32Offset);",
    ):
        if field not in tiling_header:
            raise ValueError(f"ChunkKdaFwd tiling data is missing {field}")
    for contract in (
        "const bool storeVNewFp32 = HasOutput(context, OUTPUT_V_NEW_FP32_IDX);",
        "const bool storeHFp32 = HasOutput(context, OUTPUT_H_FP32_IDX);",
        "const bool storeAkkFp32 = HasOutput(context, OUTPUT_AKK_FP32_IDX);",
        "const bool storeWFp32 = HasOutput(context, OUTPUT_W_FP32_IDX);",
        "tiling.set_storeVNewFp32(storeVNewFp32);",
        "tiling.set_storeHFp32(storeHFp32);",
        "tiling.set_storeAkkFp32(storeAkkFp32);",
        "tiling.set_storeWFp32(storeWFp32);",
        "tiling.set_hFp32StorageOffset(hFp32StorageOffset);",
        "tiling.set_stateOperandFp32Offset(stateOperandFp32Offset);",
    ):
        if contract not in normalized_tiling:
            raise ValueError(f"hidden FP32 state contract is missing {contract}")

    shape_matcher = _normalize_source(
        _source_braced_block(tiling, "bool MatchesVNewFp32Shape(")
    )
    expected_shape_paths = (
        "if (storageShape.GetDimNum() == 4) { return "
        "storageShape.GetDim(0) == info.batch && "
        "storageShape.GetDim(1) == info.vHeads && "
        "storageShape.GetDim(2) == info.seqlen && "
        "storageShape.GetDim(3) == info.vDim; }",
        "return info.rank == 3 && storageShape.GetDimNum() == 3 && "
        "storageShape.GetDim(0) == info.vHeads && "
        "storageShape.GetDim(1) == info.seqlen && "
        "storageShape.GetDim(2) == info.vDim;",
    )
    for contract in expected_shape_paths:
        if contract not in shape_matcher:
            raise ValueError(
                f"hidden v_new_fp32 rank3/rank4 shape contract is missing {contract}"
            )

    w_shape_matcher = _normalize_source(
        _source_braced_block(tiling, "bool MatchesWFp32Shape(")
    )
    expected_w_shape_paths = (
        "if (storageShape.GetDimNum() == 4) { return "
        "storageShape.GetDim(0) == info.batch && "
        "storageShape.GetDim(1) == info.vHeads && "
        "storageShape.GetDim(2) == info.seqlen && "
        "storageShape.GetDim(3) == info.kDim; }",
        "return info.rank == 3 && storageShape.GetDimNum() == 3 && "
        "storageShape.GetDim(0) == info.vHeads && "
        "storageShape.GetDim(1) == info.seqlen && "
        "storageShape.GetDim(2) == info.kDim;",
    )
    for contract in expected_w_shape_paths:
        if contract not in w_shape_matcher:
            raise ValueError(
                f"hidden w_fp32 rank3/rank4 shape contract is missing {contract}"
            )

    akk_shape_matcher = _normalize_source(
        _source_braced_block(tiling, "bool MatchesAkkFp32Shape(")
    )
    expected_akk_shape_paths = (
        "if (storageShape.GetDimNum() == 4) { return "
        "storageShape.GetDim(0) == info.batch && "
        "storageShape.GetDim(1) == info.vHeads && "
        "storageShape.GetDim(2) == info.seqlen && "
        "storageShape.GetDim(3) == chunkSize; }",
        "return info.rank == 3 && storageShape.GetDimNum() == 3 && "
        "storageShape.GetDim(0) == info.vHeads && "
        "storageShape.GetDim(1) == info.seqlen && "
        "storageShape.GetDim(2) == chunkSize;",
    )
    for contract in expected_akk_shape_paths:
        if contract not in akk_shape_matcher:
            raise ValueError(
                f"hidden akk_fp32 rank3/rank4 shape contract is missing {contract}"
            )

    h_shape_matcher = _normalize_source(
        _source_braced_block(tiling, "bool MatchesHFp32Shape(")
    )
    expected_h_shape_paths = (
        "if (storageShape.GetDimNum() == 5) { return "
        "storageShape.GetDim(0) == info.batch && "
        "storageShape.GetDim(1) == info.vHeads && "
        "storageShape.GetDim(2) == totalChunks && "
        "storageShape.GetDim(3) == info.kDim && "
        "storageShape.GetDim(4) == info.vDim; }",
        "return info.rank == 3 && storageShape.GetDimNum() == 4 && "
        "storageShape.GetDim(0) == info.vHeads && "
        "storageShape.GetDim(1) == totalChunks && "
        "storageShape.GetDim(2) == info.kDim && "
        "storageShape.GetDim(3) == info.vDim;",
    )
    for contract in expected_h_shape_paths:
        if contract not in h_shape_matcher:
            raise ValueError(
                f"hidden h_fp32 canonical rank4/rank5 shape contract is missing {contract}"
            )

    finalize_condition = (
        "if (isAscend950 && qDesc->GetDataType() == ge::DT_FLOAT16 && "
        "stage == KDA_STAGE_FINALIZE) {"
    )
    if finalize_condition not in normalized_tiling:
        raise ValueError("hidden input validation must target A5 fp16 Finalize")
    finalize_input = _normalize_source(
        _source_braced_block(tiling, "stage == KDA_STAGE_FINALIZE) {")
    )
    for contract in (
        "context->GetOptionalInputDesc(INPUT_STAGE_V_NEW_FP32_IDX)",
        "context->GetOptionalInputShape(INPUT_STAGE_V_NEW_FP32_IDX)",
        "stageVNewFp32Desc == nullptr || stageVNewFp32Shape == nullptr || "
        "stageVNewFp32Desc->GetDataType() != ge::DT_FLOAT",
        "MatchesVNewFp32Shape(storageShape, shape)",
        "context->GetOptionalInputDesc(INPUT_STAGE_H_FP32_IDX)",
        "context->GetOptionalInputShape(INPUT_STAGE_H_FP32_IDX)",
        "stageHFp32Desc->GetDataType() != ge::DT_FLOAT",
        "MatchesHFp32Shape(stageHFp32Shape->GetStorageShape(), shape, totalChunks)",
    ):
        if contract not in finalize_input:
            raise ValueError(
                f"A5 fp16 Finalize hidden-input validation is missing {contract}"
            )

    fp32_wu_condition = (
        "const bool useFp32Wu = isAscend950 && "
        "qDesc->GetDataType() == ge::DT_FLOAT16;"
    )
    if fp32_wu_condition not in normalized_tiling:
        raise ValueError("hidden FP32 Akk/W/U carriers must target A5 fp16")

    post_wu_input = _normalize_source(
        _source_braced_block(tiling, "stage == KDA_STAGE_POST_WU) {")
    )
    for contract in (
        "context->GetOptionalInputDesc(INPUT_STAGE_AKK_FP32_IDX)",
        "context->GetOptionalInputShape(INPUT_STAGE_AKK_FP32_IDX)",
        "desc == nullptr || inputShape == nullptr || "
        "desc->GetDataType() != ge::DT_FLOAT",
        "MatchesAkkFp32Shape(inputShape->GetStorageShape(), shape, chunkSize)",
    ):
        if contract not in post_wu_input:
            raise ValueError(
                f"A5 fp16 Post-WU raw Akk validation is missing {contract}"
            )

    fwd_h_input = _normalize_source(
        _source_braced_block(tiling, "stage == KDA_STAGE_FWD_H) {")
    )
    for contract in (
        "context->GetOptionalInputDesc(INPUT_STAGE_W_FP32_IDX)",
        "context->GetOptionalInputShape(INPUT_STAGE_W_FP32_IDX)",
        "context->GetOptionalInputDesc(INPUT_STAGE_V_NEW_FP32_IDX)",
        "context->GetOptionalInputShape(INPUT_STAGE_V_NEW_FP32_IDX)",
        "wDesc == nullptr || wShape == nullptr || "
        "wDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesWFp32Shape(wShape->GetStorageShape(), shape)",
        "uDesc == nullptr || uShape == nullptr || "
        "uDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesVNewFp32Shape(uShape->GetStorageShape(), shape)",
    ):
        if contract not in fwd_h_input:
            raise ValueError(
                f"A5 fp16 FwdH raw W/U validation is missing {contract}"
            )

    output_condition = (
        "if (isAscend950 && qDesc->GetDataType() == ge::DT_FLOAT16 && "
        "storeVNewFp32) {"
    )
    if output_condition not in normalized_tiling:
        raise ValueError("hidden output validation must target A5 fp16 storage")
    hidden_output = _normalize_source(
        _source_braced_block(tiling, "storeVNewFp32) {")
    )
    for contract in (
        "context->GetIrOutputInstanceInfo(OUTPUT_V_NEW_FP32_IDX)",
        "outputDesc == nullptr || outputShape == nullptr || "
        "outputDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesVNewFp32Shape(outputShape->GetStorageShape(), shape)",
    ):
        if contract not in hidden_output:
            raise ValueError(
                f"A5 fp16 hidden-output validation is missing {contract}"
            )

    hidden_akk_output = _normalize_source(
        _source_braced_block(tiling, "storeAkkFp32) {")
    )
    for contract in (
        "context->GetIrOutputInstanceInfo(OUTPUT_AKK_FP32_IDX)",
        "outputDesc == nullptr || outputShape == nullptr || "
        "outputDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesAkkFp32Shape(outputShape->GetStorageShape(), shape, chunkSize)",
    ):
        if contract not in hidden_akk_output:
            raise ValueError(
                f"A5 fp16 hidden Akk output validation is missing {contract}"
            )

    hidden_w_output = _normalize_source(
        _source_braced_block(tiling, "storeWFp32) {")
    )
    for contract in (
        "context->GetIrOutputInstanceInfo(OUTPUT_W_FP32_IDX)",
        "outputDesc == nullptr || outputShape == nullptr || "
        "outputDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesWFp32Shape(outputShape->GetStorageShape(), shape)",
    ):
        if contract not in hidden_w_output:
            raise ValueError(
                f"A5 fp16 hidden W output validation is missing {contract}"
            )

    h_output_condition = (
        "if (isAscend950 && qDesc->GetDataType() == ge::DT_FLOAT16 && "
        "storeHFp32) {"
    )
    if h_output_condition not in normalized_tiling:
        raise ValueError("hidden h_fp32 output validation must target A5 fp16 storage")
    hidden_h_output = _normalize_source(
        _source_braced_block(tiling, "storeHFp32) {")
    )
    for contract in (
        "context->GetIrOutputInstanceInfo(OUTPUT_H_FP32_IDX)",
        "outputDesc == nullptr || outputShape == nullptr || "
        "outputDesc->GetDataType() != ge::DT_FLOAT",
        "MatchesHFp32Shape(outputShape->GetStorageShape(), shape, totalChunks)",
    ):
        if contract not in hidden_h_output:
            raise ValueError(
                f"A5 fp16 hidden h_fp32 output validation is missing {contract}"
            )

    storage_offset = _source_section(
        tiling, "const uint64_t vNewFp32StorageOffset =", ";"
    )
    expected_storage_offset = _normalize_source(
        """const uint64_t vNewFp32StorageOffset =
        storeVNewFp32 || !useFp32InternalVNew || stage == KDA_STAGE_FINALIZE
            ? 0
            : AllocateWorkspace(cursor, tokenHeads * shape.vDim * sizeof(float))"""
    )
    if storage_offset != expected_storage_offset:
        raise ValueError(
            "Finalize must consume its staged FP32 v_new input without allocating "
            "an unused fallback"
        )

    h_storage_offset = _source_section(
        tiling, "const uint64_t hFp32StorageOffset =", ";"
    )
    expected_h_storage_offset = _normalize_source(
        """const uint64_t hFp32StorageOffset =
        storeHFp32 || !useFp32InternalH || stage == KDA_STAGE_FINALIZE
            ? 0
            : AllocateWorkspace(cursor, hChunkCount * shape.vHeads *
                shape.kDim * shape.vDim * sizeof(float))"""
    )
    if h_storage_offset != expected_h_storage_offset:
        raise ValueError(
            "Finalize must consume its staged FP32 h input without allocating "
            "an unused fallback"
        )
    state_operand_offset = _source_section(
        tiling, "const uint64_t stateOperandFp32Offset =", ";"
    )
    expected_state_operand_offset = _normalize_source(
        """const uint64_t stateOperandFp32Offset = useFp32InternalH
        ? AllocateWorkspace(cursor, tokenHeads * shape.kDim * sizeof(float))
        : 0"""
    )
    if state_operand_offset != expected_state_operand_offset:
        raise ValueError(
            "A5 fp16 must reserve one full FP32 W/QG operand workspace"
        )


def _check_public_aclnn_abi_contract() -> None:
    public_header = KDA_ACLNN_HEADER_PATH.read_text(encoding="utf-8")
    hidden_identifiers = (
        "vNewFp32", "hFp32", "akkFp32", "wFp32",
        "stageVNewFp32", "stageHFp32", "stageAkkFp32", "stageWFp32",
    )
    public_parameters = (
        "q", "k", "v", "g", "beta", "aLogOptional", "dtBiasOptional",
        "initialStateOptional", "cuSeqlensOptional", "chunkIndicesOptional",
        "layout", "scale", "chunkSize", "safeGate", "lowerBound",
        "useGateInKernel", "stateVFirst", "attnOut", "finalStateOut",
        "gkOut", "aqkOut", "akkOut", "wOut", "uOut", "qgOut", "kgOut",
        "vNewOut", "hOut", "workspaceSize", "executor",
    )
    if _cpp_function_parameter_names(
        public_header, "aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize"
    ) != public_parameters:
        raise ValueError(
            "hidden FP32 state must not change the public aclnn workspace API"
        )
    leaked_header_names = tuple(
        name for name in hidden_identifiers if name in public_header
    )
    if leaked_header_names:
        raise ValueError(
            "hidden FP32 carriers leaked into the public aclnn header: "
            f"{leaked_header_names}"
        )

    op_api = _cpp_source_for_aicore(
        KDA_OP_API_PATH.read_text(encoding="utf-8"), 310
    )
    params = _source_braced_block(op_api, "struct ChunkKdaFwdParams")
    tensor_fields = tuple(
        re.findall(r"const\s+aclTensor\s*\*\s*(\w+)\s*=", params)
    )
    expected_tensor_fields = (
        "q", "k", "v", "g", "beta", "aLogOptional", "dtBiasOptional",
        "initialStateOptional", "attnOut", "finalStateOut", "gkOut",
        "aqkOut", "akkOut", "wOut", "uOut", "qgOut", "kgOut", "vNewOut",
        "hOut",
    )
    if tensor_fields != expected_tensor_fields:
        raise ValueError(
            "ChunkKdaFwdParams public tensor fields drifted: "
            f"expected {expected_tensor_fields}, got {tensor_fields}"
        )
    leaked_param_names = tuple(
        name for name in hidden_identifiers if name in params
    )
    if leaked_param_names:
        raise ValueError(
            "hidden FP32 carriers leaked into ChunkKdaFwdParams: "
            f"{leaked_param_names}"
        )

    dfx = _normalize_source(
        _source_section(op_api, "L2_DFX_PHASE_1(", "auto uniqueExecutor")
    )
    expected_dfx_output = (
        "DFX_OUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut, "
        "qgOut, kgOut, vNewOut, hOut)"
    )
    if expected_dfx_output not in dfx:
        raise ValueError("public aclnn DFX outputs must remain the original 11 outputs")
    leaked_dfx_names = tuple(name for name in hidden_identifiers if name in dfx)
    if leaked_dfx_names:
        raise ValueError(
            "hidden FP32 carriers leaked into public aclnn DFX output: "
            f"{leaked_dfx_names}"
        )


def _check_hidden_fp32_h_kernel_contract() -> None:
    common = _normalize_source(
        _cpp_source_for_aicore(KDA_COMMON_PATH.read_text(encoding="utf-8"), 310)
    )
    for contract in (
        "GM_ADDR vNewFp32; GM_ADDR h; GM_ADDR hFp32; GM_ADDR akkFp32; "
        "GM_ADDR wFp32;",
        "ResolveStorage(hFp32, userWorkspace, tiling.hFp32StorageOffset, "
        "tiling.storeHFp32)",
        "ResolveStorage(akkFp32, userWorkspace, tiling.prepareAkkFp32Offset, "
        "tiling.storeAkkFp32)",
        "ResolveStorage(wFp32, userWorkspace, tiling.stateOperandFp32Offset, "
        "tiling.storeWFp32)",
        "addresses.akkFp32, userWorkspace, tiling, pipe, tiling.storeQG",
        "addresses.akkFp32, addresses.wFp32, addresses.vNewFp32, "
        "userWorkspace, tiling, pipe",
        "addresses.kg, addresses.w, addresses.u, addresses.wFp32, "
        "addresses.vNewFp32, addresses.gk, addresses.gk",
        "addresses.vNewFp32, addresses.hFp32, addresses.finalState",
    ):
        if contract not in common:
            raise ValueError(f"hidden FP32 address plumbing is missing {contract}")

    fwd_h = _cpp_source_for_aicore(
        FWD_H_KERNEL_PATHS[1].read_text(encoding="utf-8"), 310
    )
    normalized_fwd_h = _normalize_source(fwd_h)
    fp32_type_contracts = (
        "static constexpr bool FP32_H = FP32_C2;",
        "using ElementC1 = std::conditional_t<FP32_H, float, INPUT_TYPE>;",
        "using ElementWInternal = ElementC1;",
        "using ElementUInternal = std::conditional_t<FP32_C2, float, INPUT_TYPE>;",
        "using UType = Gemm::GemmType<ElementUInternal, layout::RowMajor>;",
        "using ElementHInternal = ElementC1;",
        "using DispatchPolicyTlaMultiC1 = std::conditional_t< "
        "FP32_H, DispatchPolicyTlaMultiC1Fp32, DispatchPolicyTlaMulti>;",
        "using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMultiC1, "
        "L1TileShapeVTla, L0TileShapeC1, ElementC1, ElementC1, "
        "WORKSPACE_TYPE, void, TileCopyWH>;",
    )
    for contract in fp32_type_contracts:
        if contract not in normalized_fwd_h:
            raise ValueError(f"A5 FP32 h C1 MMAD contract is missing {contract}")

    init_from_data = _normalize_source(
        _source_braced_block(fwd_h, "__aicore__ inline void InitFromData(")
    )
    for contract in (
        "GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR w_fp32, GM_ADDR u_fp32,",
        "gmWInternal.SetGlobalBuffer((__gm__ ElementWInternal *)w_fp32);",
        "gmUInternal.SetGlobalBuffer((__gm__ ElementUInternal *)u_fp32);",
        "gmHInternal.SetGlobalBuffer((__gm__ ElementHInternal *)h_fp32);",
    ):
        if contract not in init_from_data:
            raise ValueError(f"A5 FP32 h storage binding is missing {contract}")

    process = _normalize_source(
        _source_braced_block(fwd_h, "__aicore__ inline void Process()")
    )
    if "PrepareFp32W(" in process:
        raise ValueError("FwdH must consume staged raw FP32 W without recasting public W")
    for contract in (
        "tla::MakeTensor(gmWInternal[cube1OffsetW]",
        "gmUInternal[vec1Offsets.uvOffset]",
        "tla::MakeTensor(gmHInternal[cube1OffsetH]",
        "gmHInternal[vec2Offsets.hDstOffset]",
        "gmHInternal[vec2Offsets.hSrcOffset]",
        "gmHInternal[hOffset]",
    ):
        if contract not in process:
            raise ValueError(f"hidden FP32 h recurrence is missing {contract}")

    export_h = _normalize_source(
        _source_braced_block(fwd_h, "__aicore__ inline void ExportPublicH(")
    )
    export_contracts = (
        "constexpr float FP16_MAX = 65504.0f;",
        "DataCopy(source, gmHInternal[offset], count);",
        "Mins(source, source, FP16_MAX, count);",
        "Maxs(source, source, -FP16_MAX, count);",
        "Cast( output, source, AscendC::RoundMode::CAST_RINT, count);",
        "DataCopy(gmH[offset], output, count);",
    )
    positions = []
    for contract in export_contracts:
        if contract not in export_h:
            raise ValueError(f"public FP16 h saturating export is missing {contract}")
        positions.append(export_h.index(contract))
    if positions != sorted(positions):
        raise ValueError("public h must clamp FP32 state before its FP16 cast and write")
    if "ExportPublicH( coreIdx, coreNum, subBlockIdx, subBlockNum);" not in process:
        raise ValueError("FwdH does not export its hidden FP32 h to the public output")

    finalize = _cpp_source_for_aicore(
        KDA_ARCH35_FINALIZE_PATH.read_text(encoding="utf-8"), 310
    )
    normalized_finalize = _normalize_source(finalize)
    for contract in (
        "typename PROP_V_T = T, typename PROP_H_T = T>",
        "static constexpr bool USE_FP32_H = IsSameType<T, half>::value && "
        "IsSameType<PROP_H_T, float>::value;",
        "using PROP_QG_T = std::conditional_t<USE_FP32_H, float, T>;",
        "propagatedH_.SetGlobalBuffer((__gm__ PROP_H_T *)propagatedH);",
        "propagatedQG_.SetGlobalBuffer((__gm__ PROP_QG_T *)propagatedQGFp32);",
        "CopyVectorOut(propagatedQG_, offset, converted, count);",
        "auto layoutQ = tla::MakeLayout<PROP_QG_T, LayoutTagA>(BT_, K_);",
        "auto layoutH = tla::MakeLayout<PROP_H_T, LayoutTagB>(K_, V_);",
        "ComputeOutputStateCube( stateBlockMmad, b, hv, chunkIdx, start, curT);",
        "GM_ADDR stateOperandFp32 = userWorkspace + tiling.stateOperandFp32Offset;",
    ):
        if contract not in normalized_finalize:
            raise ValueError(f"Finalize FP32 h/qg state MMAD is missing {contract}")

    tail_local = _source_braced_block(
        finalize, "__aicore__ inline void ComputeTailLocalRows("
    )
    fp32_branch = _source_braced_block(
        tail_local, "if constexpr (USE_FP32_V_NEW)"
    )
    after_fp32_branch = tail_local[
        tail_local.index(fp32_branch) + len(fp32_branch) :
    ]
    non_fp32_branch = _source_braced_block(after_fp32_branch, "else")
    after_precision_branches = after_fp32_branch[
        after_fp32_branch.index(non_fp32_branch) + len(non_fp32_branch) :
    ]
    scalar_read = "float weight = coefficients.GetValue(j);"
    if scalar_read not in after_precision_branches:
        raise ValueError("Finalize tail Aqk scalar read contract is missing")
    before_scalar_read = after_precision_branches[
        : after_precision_branches.index(scalar_read)
    ]
    if "PipeBarrier<PIPE_ALL>();" not in before_scalar_read:
        raise ValueError(
            "Finalize tail Aqk must drain both precision branches before scalar reads"
        )


def _check_stage_dependency_contract() -> None:
    _check_public_aclnn_abi_contract()
    raw_inputs = (
        "q", "k", "v", "g", "beta", "a_log", "dt_bias", "initial_state",
        "cu_seqlens", "chunk_indices",
    )
    stage_inputs = (
        "stage_gk", "stage_aqk", "stage_akk", "stage_w", "stage_u",
        "stage_kg", "stage_v_new", "stage_h", "stage_qg_scaled",
        "stage_u_seed", "stage_v_new_fp32", "stage_h_fp32",
        "stage_akk_fp32", "stage_w_fp32",
    )
    outputs = (
        "attn_out", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg",
        "kg", "v_new", "h", "qg_scaled", "u_seed", "v_new_fp32", "h_fp32",
        "akk_fp32", "w_fp32",
    )

    op_def = _cpp_source_for_aicore(
        KDA_OP_DEF_PATH.read_text(encoding="utf-8"), 310
    )
    if tuple(re.findall(r'this->Input\("([^"]+)"\)', op_def)) != raw_inputs + stage_inputs:
        raise ValueError("ChunkKdaFwd OpDef input order does not match the staged kernel ABI")
    if tuple(re.findall(r'this->Output\("([^"]+)"\)', op_def)) != outputs:
        raise ValueError("ChunkKdaFwd low-level output order does not match the kernel ABI")
    normalized_op_def = _normalize_source(op_def)
    for name in stage_inputs:
        dtype = (
            "stateTypes"
            if name in (
                "stage_gk", "stage_v_new_fp32", "stage_h_fp32",
                "stage_akk_fp32", "stage_w_fp32",
            )
            else "dataTypes"
        )
        declaration = (
            f'this->Input("{name}").ParamType(OPTIONAL).DataType({dtype})'
        )
        if declaration not in normalized_op_def:
            raise ValueError(f"{name} must remain an optional {dtype} dependency alias")
    for name in ("v_new_fp32", "h_fp32", "akk_fp32", "w_fp32"):
        hidden_output_declaration = (
            f'this->Output("{name}").ParamType(OPTIONAL).DataType(stateTypes)'
        )
        if hidden_output_declaration not in normalized_op_def:
            raise ValueError(f"{name} must remain an optional FP32 state output")

    l0_source = _cpp_source_for_aicore(
        KDA_L0_OP_API_PATH.read_text(encoding="utf-8"), 310
    )
    op_input_match = re.search(
        r"OP_INPUT\((.*?)\),\s*OP_OUTPUT", l0_source, flags=re.DOTALL
    )
    if op_input_match is None:
        raise ValueError("KdaChunkForward must register its L0 inputs")
    l0_inputs = tuple(
        item.strip() for item in op_input_match.group(1).split(",")
    )
    expected_l0_inputs = (
        "q", "k", "v", "g", "beta", "aLogOptional", "dtBiasOptional",
        "initialStateOptional", "actualCuSeqlens", "actualChunkIndices",
        "stageGkInputOptional", "stageAqkInputOptional",
        "stageAkkInputOptional", "stageWInputOptional", "stageUInputOptional",
        "stageKgInputOptional", "stageVNewInputOptional", "stageHInputOptional",
        "stageQgScaledInputOptional", "stageUSeedInputOptional",
        "stageVNewFp32InputOptional", "stageHFp32InputOptional",
        "stageAkkFp32InputOptional", "stageWFp32InputOptional",
    )
    if l0_inputs != expected_l0_inputs:
        raise ValueError("KdaChunkForward OP_INPUT order does not match its OpDef")

    op_output_match = re.search(
        r"OP_OUTPUT\((.*?)\),\s*OP_ATTR", l0_source, flags=re.DOTALL
    )
    if op_output_match is None:
        raise ValueError("KdaChunkForward must register its L0 outputs")
    l0_outputs = tuple(
        item.strip() for item in op_output_match.group(1).split(",")
    )
    expected_l0_outputs = (
        "attnOut", "finalStateOut", "gkOut", "aqkOut", "akkOut", "wOut",
        "uOut", "qgOut", "kgOut", "vNewOut", "hOut", "qgScaledOut",
        "uSeedOut", "vNewFp32Out", "hFp32Out", "akkFp32Out", "wFp32Out",
    )
    if l0_outputs != expected_l0_outputs:
        raise ValueError("KdaChunkForward OP_OUTPUT order does not match its OpDef")

    expected_l0_parameters = expected_l0_inputs[:8] + (
        "cuSeqlensOptional", "chunkIndicesOptional",
    ) + expected_l0_inputs[10:] + (
        "scale", "chunkSize", "safeGate", "inputSequenceMajor",
        "useGateInKernel", "lowerBound", "attnOut", "finalStateOut",
        "gkOut", "aqkOut", "akkOut", "wOut", "uOut", "qgOut",
        "kgOut", "vNewOut", "hOut", "qgScaledOut", "uSeedOut",
        "vNewFp32Out", "hFp32Out", "akkFp32Out", "wFp32Out",
        "stage", "executor",
    )
    l0_header = _cpp_source_for_aicore(
        KDA_L0_HEADER_PATH.read_text(encoding="utf-8"), 310
    )
    declaration_parameters = _cpp_function_parameter_names(
        l0_header, "KdaCoreOutputs KdaChunkForward"
    )
    definition_parameters = _cpp_function_parameter_names(
        l0_source, "KdaCoreOutputs KdaChunkForward"
    )
    if declaration_parameters != expected_l0_parameters:
        raise ValueError(
            "KdaChunkForward declaration parameter ABI drifted: "
            f"expected {expected_l0_parameters}, got {declaration_parameters}"
        )
    if definition_parameters != expected_l0_parameters:
        raise ValueError(
            "KdaChunkForward definition parameter ABI drifted: "
            f"expected {expected_l0_parameters}, got {definition_parameters}"
        )
    l0_dfx_match = re.search(
        r"L0_DFX\((.*?)\);", l0_source, flags=re.DOTALL
    )
    if l0_dfx_match is None:
        raise ValueError("KdaChunkForward must register its complete L0 DFX ABI")
    l0_dfx_parameters = tuple(
        item.strip() for item in l0_dfx_match.group(1).split(",")
    )
    expected_l0_dfx_parameters = (
        "KdaChunkForward", *expected_l0_parameters[:-1]
    )
    if l0_dfx_parameters != expected_l0_dfx_parameters:
        raise ValueError(
            "KdaChunkForward L0_DFX parameter ABI drifted: "
            f"expected {expected_l0_dfx_parameters}, got {l0_dfx_parameters}"
        )
    if re.search(
        r"using\s+KdaCoreOutputs\s*=\s*std::array\s*<\s*"
        r"const\s+aclTensor\s*\*\s*,\s*13\s*>\s*;",
        l0_header,
    ) is None:
        raise ValueError("KdaChunkForward public output tuple size must remain 13")
    l0_body = _normalize_source(
        _source_braced_block(l0_source, "KdaCoreOutputs KdaChunkForward")
    )
    public_return = (
        "return {attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut, "
        "qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut};"
    )
    if public_return not in l0_body:
        raise ValueError("hidden FP32 carriers must not enter the 13-item public output tuple")
    public_return_block = _source_section(l0_body, "return {", "};")
    for hidden_name in (
        "vNewFp32Out", "hFp32Out", "akkFp32Out", "wFp32Out",
    ):
        if hidden_name in public_return_block:
            raise ValueError(
                f"{hidden_name} leaked into the 13-item public output tuple"
            )

    op_api = _cpp_source_for_aicore(
        KDA_OP_API_PATH.read_text(encoding="utf-8"), 310
    )
    expected_stage_fields = {
        "KDA_STAGE_POST_WU": ("gk", "akk", "akkFp32", "w", "uSeed"),
        "KDA_STAGE_FWD_H": (
            "gk", "w", "u", "kg", "uSeed", "vNewFp32", "wFp32",
        ),
        "KDA_STAGE_FINALIZE": (
            "gk", "aqk", "vNew", "h", "qgScaled", "vNewFp32", "hFp32",
        ),
    }
    for stage, expected_fields in expected_stage_fields.items():
        block = _source_braced_block(op_api, f"stage == {stage}")
        fields = tuple(re.findall(r"stageInputs\.(\w+)\s*=", block))
        if fields != expected_fields:
            raise ValueError(f"{stage} dependency aliases drifted: {fields}")
    normalized_op_api = _normalize_source(op_api)
    stage_call = (
        "params.chunkIndicesOptional, stageInputs.gk, stageInputs.aqk, "
        "stageInputs.akk, stageInputs.w, stageInputs.u, stageInputs.kg, "
        "stageInputs.vNew, stageInputs.h, stageInputs.qgScaled, "
        "stageInputs.uSeed, stageInputs.vNewFp32, stageInputs.hFp32, "
        "stageInputs.akkFp32, stageInputs.wFp32, params.scale"
    )
    if stage_call not in normalized_op_api:
        raise ValueError("L2 stage alias call order does not match the L0 input ABI")
    hidden_lifetime_contracts = {
        "canonical [B,HV,T,V] shape": (
            r"const\s+op::Shape\s+vShape4\s*=\s*MakeShape\s*\(\s*\{\s*"
            r"info\.batch\s*,\s*info\.hvNum\s*,\s*info\.seqlen\s*,\s*"
            r"info\.vDim\s*\}\s*\)\s*;"
        ),
        "A5 fp16 selection": (
            r"const\s+bool\s+preserveFp32VNew\s*=\s*IsAscend950\s*\(\s*\)\s*"
            r"&&\s*params\.q->GetDataType\s*\(\s*\)\s*==\s*"
            r"DataType::DT_FLOAT16\s*;"
        ),
        "full-shape FP32 allocation": (
            r"const\s+aclTensor\s*\*\s*vNewFp32Compute\s*=\s*AllocTensor\s*"
            r"\(\s*executorPtr\s*,\s*preserveFp32VNew\s*\?\s*vShape4\s*:\s*"
            r"placeholderShape\s*,\s*DataType::DT_FLOAT\s*\)\s*;"
        ),
        "Finalize input reuse": (
            r"stageInputs\.vNewFp32\s*=\s*vNewFp32Compute\s*;"
        ),
        "canonical [B,HV,NT,K,V] h shape": (
            r"const\s+op::Shape\s+hShape5\s*=\s*MakeShape\s*\(\s*\{\s*"
            r"info\.batch\s*,\s*info\.hvNum\s*,\s*info\.totalChunks\s*,\s*"
            r"info\.kDim\s*,\s*info\.vDim\s*\}\s*\)\s*;"
        ),
        "A5 fp16 h selection": (
            r"const\s+bool\s+preserveFp32H\s*=\s*IsAscend950\s*\(\s*\)\s*"
            r"&&\s*params\.q->GetDataType\s*\(\s*\)\s*==\s*"
            r"DataType::DT_FLOAT16\s*;"
        ),
        "full-shape FP32 h allocation": (
            r"const\s+aclTensor\s*\*\s*hFp32Compute\s*=\s*AllocTensor\s*"
            r"\(\s*executorPtr\s*,\s*preserveFp32H\s*\?\s*hShape5\s*:\s*"
            r"placeholderShape\s*,\s*DataType::DT_FLOAT\s*\)\s*;"
        ),
        "Finalize h input reuse": (
            r"stageInputs\.hFp32\s*=\s*hFp32Compute\s*;"
        ),
        "canonical [B,HV,T,BT] raw Akk shape": (
            r"const\s+op::Shape\s+matrixShape4\s*=\s*MakeShape\s*\(\s*\{\s*"
            r"info\.batch\s*,\s*info\.hvNum\s*,\s*info\.seqlen\s*,\s*"
            r"params\.chunkSize\s*\}\s*\)\s*;"
        ),
        "canonical [B,HV,T,K] raw W shape": (
            r"const\s+op::Shape\s+kShape4\s*=\s*MakeShape\s*\(\s*\{\s*"
            r"info\.batch\s*,\s*info\.hvNum\s*,\s*info\.seqlen\s*,\s*"
            r"info\.kDim\s*\}\s*\)\s*;"
        ),
        "A5 fp16 raw Akk/W/U selection": (
            r"const\s+bool\s+preserveFp32Wu\s*=\s*IsAscend950\s*\(\s*\)\s*"
            r"&&\s*params\.q->GetDataType\s*\(\s*\)\s*==\s*"
            r"DataType::DT_FLOAT16\s*;"
        ),
        "full-shape FP32 Akk allocation": (
            r"const\s+aclTensor\s*\*\s*akkFp32Compute\s*=\s*AllocTensor\s*"
            r"\(\s*executorPtr\s*,\s*preserveFp32Wu\s*\?\s*matrixShape4\s*:\s*"
            r"placeholderShape\s*,\s*DataType::DT_FLOAT\s*\)\s*;"
        ),
        "full-shape FP32 W allocation": (
            r"const\s+aclTensor\s*\*\s*wFp32Compute\s*=\s*AllocTensor\s*"
            r"\(\s*executorPtr\s*,\s*preserveFp32Wu\s*\?\s*kShape4\s*:\s*"
            r"placeholderShape\s*,\s*DataType::DT_FLOAT\s*\)\s*;"
        ),
        "Post-WU raw Akk input reuse": (
            r"stageInputs\.akkFp32\s*=\s*akkFp32Compute\s*;"
        ),
        "FwdH raw W input reuse": (
            r"stageInputs\.wFp32\s*=\s*wFp32Compute\s*;"
        ),
        "FwdH raw U reuses v_new_fp32 carrier": (
            r"stageInputs\.vNewFp32\s*=\s*vNewFp32Compute\s*;"
        ),
        "FwdH output forwarding": (
            r"qgScaledCompute\s*,\s*uSeedCompute\s*,\s*vNewFp32Compute\s*,\s*"
            r"hFp32Compute\s*,\s*akkFp32Compute\s*,\s*wFp32Compute\s*,\s*"
            r"stage\s*,\s*executorPtr\s*\)\s*;"
        ),
    }
    for description, pattern in hidden_lifetime_contracts.items():
        if re.search(pattern, op_api) is None:
            raise ValueError(
                f"A5 fp16 staged hidden FP32-state lifetime is missing {description}"
            )

    kernel = _cpp_source_for_aicore(
        KDA_ENTRY_PATH.read_text(encoding="utf-8"), 310
    )
    kernel_start = kernel.index(
        'extern "C" __global__ __aicore__ void chunk_kda_fwd('
    )
    kernel_open = kernel.index("{", kernel_start)
    kernel_parameters = tuple(
        re.findall(r"GM_ADDR\s+(\w+)", kernel[kernel_start:kernel_open])
    )
    expected_kernel_parameters = raw_inputs + stage_inputs + (
        "attn_out", "final_state", "gk", "aqk", "akk", "w", "u", "qg",
        "kg", "v_new", "h", "qg_scaled", "u_seed", "v_new_fp32",
        "h_fp32", "akk_fp32", "w_fp32", "workspace", "tiling",
    )
    if kernel_parameters != expected_kernel_parameters:
        raise ValueError("chunk_kda_fwd kernel parameter order does not match its OpDef")

    dispatch_prefix = (
        "q", "k", "v", "g", "beta", "aLog", "dtBias", "initialState",
        "cuSeqlens", "chunkIndices",
    )
    stage_camel_inputs = (
        "stageGk", "stageAqk", "stageAkk", "stageW", "stageU",
        "stageKg", "stageVNew", "stageH", "stageQgScaled", "stageUSeed",
        "stageVNewFp32", "stageHFp32", "stageAkkFp32", "stageWFp32",
    )
    dispatch_outputs = (
        "attnOut", "finalState", "gk", "aqk", "akk", "w", "u", "qg",
        "kg", "vNew", "h", "qgScaled", "uSeed", "vNewFp32", "hFp32",
        "akkFp32", "wFp32", "userWorkspace", "tiling",
    )
    expected_stage_dispatch = dispatch_prefix + stage_camel_inputs + dispatch_outputs
    expected_full_dispatch = dispatch_prefix + (
        "attnOut", "finalState", "gk", "aqk", "akk", "w", "u", "qg",
        "kg", "vNew", "vNewFp32", "h", "hFp32", "akkFp32", "wFp32",
        "userWorkspace", "tiling",
    )
    for function_name, expected_parameters in (
        ("DispatchStage", expected_stage_dispatch),
        ("DispatchStageSafeGate", expected_stage_dispatch),
        ("DispatchGeneric", expected_full_dispatch),
        ("DispatchArch35SafeGate", expected_full_dispatch),
        ("DispatchGenericSafeGate", expected_full_dispatch),
    ):
        actual_parameters = _cpp_function_parameter_names(
            kernel, f"__aicore__ inline void {function_name}"
        )
        if actual_parameters != expected_parameters:
            raise ValueError(
                f"{function_name} hidden-carrier ABI drifted: "
                f"expected {expected_parameters}, got {actual_parameters}"
            )

    def require_forwarding(
        function_name: str, arguments: tuple[str, ...], expected_count: int
    ) -> None:
        block = _normalize_source(
            _source_braced_block(
                kernel, f"__aicore__ inline void {function_name}"
            )
        )
        forwarding = ", ".join(arguments)
        if block.count(forwarding) != expected_count:
            raise ValueError(
                f"{function_name} must forward the complete hidden-carrier ABI "
                f"exactly {expected_count} time(s)"
            )

    require_forwarding("DispatchStageSafeGate", expected_stage_dispatch, 2)
    require_forwarding("DispatchGeneric", expected_full_dispatch, 1)
    require_forwarding("DispatchGenericSafeGate", expected_full_dispatch, 2)
    require_forwarding("DispatchArch35SafeGate", expected_full_dispatch, 2)

    arch35_impl = _cpp_source_for_aicore(
        (KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_impl.h").read_text(
            encoding="utf-8"
        ),
        310,
    )
    arch35_parameters = _cpp_function_parameter_names(
        arch35_impl, "__aicore__ inline void Run"
    )
    expected_arch35_parameters = expected_full_dispatch + ("pipe",)
    if arch35_parameters != expected_arch35_parameters:
        raise ValueError(
            "arch35::Run hidden-carrier ABI drifted: "
            f"expected {expected_arch35_parameters}, got {arch35_parameters}"
        )
    arch35_body = _normalize_source(
        _source_braced_block(arch35_impl, "__aicore__ inline void Run")
    )
    resolve_call = (
        "finalState, gk, w, u, qg, kg, vNew, vNewFp32, h, hFp32, "
        "akkFp32, wFp32, userWorkspace, tiling"
    )
    if arch35_body.count(resolve_call) != 1:
        raise ValueError("arch35::Run must bind both hidden FP32 carriers once")

    post_wu = _source_braced_block(kernel, "tiling.stage == KDA_STAGE_POST_WU")
    if tuple(re.findall(r"ResolveStageInput\((stage\w+),", post_wu)) != (
        "stageGk", "stageW", "stageAkk", "stageUSeed", "stageAkkFp32",
    ):
        raise ValueError("Post-WU must read every dependency through its stage alias")

    fwd_h = _source_braced_block(kernel, "tiling.stage == KDA_STAGE_FWD_H")
    fwd_fields = tuple(re.findall(r"fwdInputs\.(\w+)\s*=", fwd_h))
    if fwd_fields != (
        "gk", "w", "u", "kg", "uSeed", "wFp32", "vNewFp32",
    ):
        raise ValueError("FwdH input view must include its public and raw W/U aliases")
    normalized_fwd_h = _normalize_source(fwd_h)
    if normalized_fwd_h.count(
        "initialState, cuSeqlens, chunkIndices, fwdInputs, userWorkspace, tiling"
    ) != 2:
        raise ValueError("both FwdH tile paths must consume the aliased input view")

    finalize = _source_braced_block(kernel, "tiling.stage == KDA_STAGE_FINALIZE")
    if tuple(re.findall(r"ResolveStageInput\((stage\w+),", finalize)) != (
        "stageVNewFp32", "stageVNew", "stageHFp32", "stageH",
        "stageGk", "stageQgScaled", "stageAqk",
    ):
        raise ValueError("Finalize must read every dependency through its stage alias")

    kernel_entry = _normalize_source(kernel[kernel_start:])
    kernel_stage_arguments = raw_inputs + stage_inputs + (
        "attn_out", "final_state", "gk", "aqk", "akk", "w", "u", "qg",
        "kg", "v_new", "h", "qg_scaled", "u_seed", "v_new_fp32",
        "h_fp32", "akk_fp32", "w_fp32", "userWorkspace", "tilingData",
    )
    if kernel_entry.count(", ".join(kernel_stage_arguments)) != 2:
        raise ValueError(
            "both staged tiling keys must forward every dependency alias and "
            "hidden FP32 output"
        )
    kernel_full_arguments = raw_inputs + (
        "attn_out", "final_state", "gk", "aqk", "akk", "w", "u", "qg",
        "kg", "v_new", "v_new_fp32", "h", "h_fp32", "akk_fp32",
        "w_fp32", "userWorkspace", "tilingData",
    )
    if kernel_entry.count(", ".join(kernel_full_arguments)) != 2:
        raise ValueError(
            "both full-stage tiling keys must forward every hidden FP32 output"
        )

    _check_hidden_fp32_h_kernel_contract()
    _check_hidden_fp32_state_tiling_contract()


def _ast_expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _ast_equal(actual: ast.AST, expected: str) -> bool:
    return ast.dump(actual, include_attributes=False) == ast.dump(
        _ast_expression(expected), include_attributes=False
    )


def _assigned_value(statements: list[ast.stmt], name: str) -> ast.expr:
    matches = []
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(statement.value)
    if len(matches) != 1:
        raise ValueError(f"_prepare_inputs must assign {name} exactly once in this branch")
    return matches[0]


def _if_by_test(statements: list[ast.stmt], expected: str) -> ast.If:
    matches = [
        statement
        for statement in statements
        if isinstance(statement, ast.If) and _ast_equal(statement.test, expected)
    ]
    if len(matches) != 1:
        raise ValueError(f"_prepare_inputs must contain one branch for {expected}")
    return matches[0]


def _require_random_range(
    statements: list[ast.stmt], name: str, low: str, high: str
) -> None:
    value = _assigned_value(statements, name)
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_random_quantized"
    ):
        raise ValueError(f"uniform {name} must use _random_quantized")
    keywords = {keyword.arg: keyword.value for keyword in value.keywords}
    if not (
        "low" in keywords
        and "high" in keywords
        and _ast_equal(keywords["low"], low)
        and _ast_equal(keywords["high"], high)
    ):
        raise ValueError(f"uniform {name} range must remain [{low}, {high}]")


def _check_uniform_executor_contract(source: str) -> None:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_inputs"
    ]
    if len(functions) != 1:
        raise ValueError("executor must define exactly one _prepare_inputs")
    prepare = functions[0]
    if not _ast_equal(
        _assigned_value(prepare.body, "data_profile"),
        "str(spec.get('data_profile', 'uniform'))",
    ):
        raise ValueError("_prepare_inputs must read data_profile from the manifest")

    profile_test = "data_profile == 'model_h96'"
    profile_branches = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.If) and _ast_equal(node.test, profile_test)
    ]
    if len(profile_branches) != 3:
        raise ValueError("executor must route q/k/v, A_log and dt_bias by data_profile")
    use_gate_test = "_as_bool(spec['use_gate_in_kernel'])"
    use_gate_branches = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.If) and _ast_equal(node.test, use_gate_test)
    ]
    if len(use_gate_branches) != 2:
        raise ValueError("executor must preserve both positive use_gate_in_kernel branches")

    profile_branch = _if_by_test(prepare.body, profile_test)
    uniform = profile_branch.orelse
    if not _ast_equal(
        _assigned_value(uniform, "data_scale"), "float(spec['data_scale'])"
    ):
        raise ValueError("uniform data_scale must be read without rescaling")
    for name in ("q_bsnd", "k_bsnd", "v_bsnd"):
        _require_random_range(uniform, name, "-data_scale", "data_scale")
    if not _ast_equal(
        _assigned_value(uniform, "gate_scale"), "float(spec['gate_scale'])"
    ):
        raise ValueError("uniform gate_scale must be read without rescaling")
    gate_branch = _if_by_test(uniform, use_gate_test)
    gate_assignments = [
        statement
        for statement in gate_branch.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and ast.unparse(statement.targets[0]) == "(gate_low, gate_high)"
    ]
    if len(gate_assignments) != 1 or not _ast_equal(
        gate_assignments[0].value, "(-gate_scale, gate_scale)"
    ):
        raise ValueError("active uniform gate range must remain symmetric")
    inactive_gate_assignments = [
        statement
        for statement in gate_branch.orelse
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and ast.unparse(statement.targets[0]) == "(gate_low, gate_high)"
    ]
    if len(inactive_gate_assignments) != 1 or not _ast_equal(
        inactive_gate_assignments[0].value,
        "(-0.02 * gate_scale, -0.002 * gate_scale)",
    ):
        raise ValueError("pre-activated uniform gate range drifted")
    _require_random_range(uniform, "g_bsnd", "gate_low", "gate_high")
    _require_random_range(uniform, "beta_bsnd", "0.0", "1.0")

    a_log_branch = _if_by_test(prepare.body, use_gate_test)
    a_log_profile = _if_by_test(a_log_branch.body, profile_test)
    _require_random_range(a_log_profile.orelse, "A_log", "-6.0", "-2.0")
    dt_bias_branch = _if_by_test(prepare.body, "_as_bool(spec['dt_bias'])")
    dt_bias_profile = _if_by_test(dt_bias_branch.body, profile_test)
    _require_random_range(dt_bias_profile.orelse, "dt_bias", "-2.0", "2.0")
    initial_state_branch = _if_by_test(
        prepare.body, "_as_bool(spec['initial_state'])"
    )
    _require_random_range(
        initial_state_branch.body, "initial_state", "-0.02", "0.02"
    )


def _check_kernel_launch_trace_contract(source: str) -> None:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChunkKdaFwdApi"
    ]
    if len(classes) != 1:
        raise ValueError("executor must define exactly one ChunkKdaFwdApi")
    call_methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    ]
    if len(call_methods) != 1:
        raise ValueError("ChunkKdaFwdApi must define exactly one __call__")
    call_method = call_methods[0]
    npu_branches = [
        node
        for node in ast.walk(call_method)
        if isinstance(node, ast.If) and _ast_equal(node.test, "self.device == 'npu'")
    ]
    if len(npu_branches) != 1:
        raise ValueError("executor must contain exactly one positive NPU branch")
    npu_body = npu_branches[0].body
    body_root = ast.Module(body=npu_body, type_ignores=[])
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(body_root):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    trace_assignments = [
        node
        for node in ast.walk(body_root)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "trace_kernel_launch"
    ]
    if len(trace_assignments) != 1 or not _ast_equal(
        trace_assignments[0].value,
        "os.environ.get('KDA_ATK_TRACE_KERNEL_LAUNCH') == '1'",
    ):
        raise ValueError("kernel launch trace must use the explicit environment switch")

    calls = [node for node in ast.walk(body_root) if isinstance(node, ast.Call)]
    npu_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "_run_positive_npu"
    ]
    if len(npu_calls) != 1:
        raise ValueError("NPU branch must invoke _run_positive_npu exactly once")

    def marker(call: ast.Call) -> str | None:
        if not isinstance(call.func, ast.Name) or call.func.id != "print":
            return None
        values = {
            argument.value
            for argument in call.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        matches = values & {
            "KDA_ATK_KERNEL_LAUNCH_BEGIN", "KDA_ATK_KERNEL_LAUNCH_END"
        }
        return next(iter(matches)) if len(matches) == 1 else None

    begin_calls = [
        call for call in calls if marker(call) == "KDA_ATK_KERNEL_LAUNCH_BEGIN"
    ]
    end_calls = [
        call for call in calls if marker(call) == "KDA_ATK_KERNEL_LAUNCH_END"
    ]
    sync_calls = [
        call for call in calls if ast.unparse(call.func) == "torch.npu.synchronize"
    ]
    if len(begin_calls) != 1 or len(end_calls) != 1 or len(sync_calls) != 1:
        raise ValueError("kernel launch trace requires one BEGIN, one synchronize and one END")

    trace_test = ast.dump(_ast_expression("trace_kernel_launch"), include_attributes=False)

    def if_guards(node: ast.AST) -> set[tuple[str, bool]]:
        guards = set()
        child = node
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.If):
                test = ast.dump(parent.test, include_attributes=False)
                if child in parent.body:
                    guards.add((test, True))
                elif child in parent.orelse:
                    guards.add((test, False))
            child = parent
        return guards

    begin, run, synchronize, end = (
        begin_calls[0], npu_calls[0], sync_calls[0], end_calls[0]
    )
    trace_guard = {(trace_test, True)}
    if any(if_guards(node) != trace_guard for node in (begin, synchronize, end)):
        raise ValueError("BEGIN, synchronize and END must share the positive trace guard")
    if if_guards(run):
        raise ValueError("_run_positive_npu must execute independently of launch tracing")

    position = lambda node: (node.lineno, node.col_offset)
    if not (
        position(trace_assignments[0]) < position(begin) < position(run)
        < position(synchronize) < position(end)
    ):
        raise ValueError("kernel launch trace must order switch, BEGIN, launch, synchronize and END")


def _check_l1_clear_barriers() -> None:
    source = KDA_MMAD_MULTI_PATH.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/|//[^\r\n]*", "", source, flags=re.DOTALL)
    start_marker = "CATLASS_DEVICE void operator()("
    start = source.index(start_marker)
    end = source.index("protected:", start)
    operator = source[start:end]

    clear_call = re.compile(
        r"^[ \t]*AscendC::InitConstValue\s*\(\s*"
        r"(l1[AB]TensorList\s*\[\s*[^\]]+\s*\])\s*,\s*"
        r"clearParams\s*\)\s*;",
        flags=re.MULTILINE,
    )
    expected_targets = {
        "l1ATensorList[l1AListId]",
        "l1BTensorList[l1BListId]",
        "l1ATensorList[l1AListIdNext]",
        "l1BTensorList[l1BListIdNext]",
    }
    matches = list(clear_call.finditer(operator))
    actual_targets = {
        re.sub(r"\s+", "", match.group(1)) for match in matches
    }
    if len(matches) != 4 or actual_targets != expected_targets:
        raise ValueError(
            "multi-stage MMAD must clear exactly the initial/preload A/B L1 slots"
        )

    barrier = re.compile(
        r"\s*AscendC::PipeBarrier\s*<\s*PIPE_MTE2\s*>\s*\(\s*\)\s*;"
    )
    for match in matches:
        target = re.sub(r"\s+", "", match.group(1))
        if barrier.match(operator, match.end()) is None:
            raise ValueError(
                f"{target} clear must be immediately followed by a PIPE_MTE2 barrier"
            )


def _check_kernel_utils_ownership() -> None:
    for path in (*KDA_PRIVATE_UTILITY_PATHS, *KDA_PRIVATE_FWD_H_PATHS):
        if not path.is_file():
            raise ValueError(f"missing KDA-private kernel source: {path}")
        source = path.read_text(encoding="utf-8")
        if "fla/ops/ascendc/common/kernel_utils" in source:
            raise ValueError(f"KDA source must not reference common/kernel_utils: {path}")
        if "chunk_gated_delta_rule_fwd_h" in source:
            raise ValueError(f"KDA source must not reference the independent GDN FwdH: {path}")
        if path in KDA_PRIVATE_FWD_H_PATHS and re.search(
            r"\b(?:GDN|Gdn|gdn)FwdH", source
        ):
            raise ValueError(f"KDA-private FwdH must use KDA-owned type names: {path}")

    provider_markers = {
        KDA_MMAD_PATHS[0]: "FLA_NPU_KERNEL_UTIL_MMAD_TLA_PROVIDED",
        KDA_MMAD_PATHS[1]: "FLA_NPU_KERNEL_UTIL_MMAD_MULTI_PROVIDED",
        KDA_MMAD_PATHS[2]: "FLA_NPU_KERNEL_UTIL_MMAD_PRELOAD_PROVIDED",
        KDA_KERNEL_UTILS_ROOT / "tile/copy_l0c_to_ub.hpp": (
            "FLA_NPU_KERNEL_UTIL_COPY_L0C_PROVIDED"
        ),
        KDA_KERNEL_UTILS_ROOT / "vector/regbase.hpp": (
            "FLA_NPU_KERNEL_UTIL_REGBASE_PROVIDED"
        ),
    }
    for path, marker in provider_markers.items():
        source = path.read_text(encoding="utf-8")
        if f"#define {marker} 1" not in source:
            raise ValueError(f"{path.name}: missing private utility provider marker {marker}")

    include_re = re.compile(
        r'^\s*#\s*include\s*[<"]([^">]+)[">]', flags=re.MULTILINE
    )
    for path in KDA_KERNEL_ROOT.rglob("*"):
        if path.suffix not in {".h", ".hpp", ".cpp"}:
            continue
        source = path.read_text(encoding="utf-8")
        for include_target in include_re.findall(source):
            if (
                include_target.startswith("kernel_utils/")
                or "common/kernel_utils/" in include_target
                or "chunk_gdn_fwd/" in include_target
                or "chunk_gated_delta_rule_fwd_h/" in include_target
            ):
                raise ValueError(
                    f"{path}: KDA source must not include public GDN/utility source {include_target}"
                )

    expected_private_includes = {
        KDA_KERNEL_ROOT / "chunk_kda_fwd_prepare.h": (
            '#include "./kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
        ),
        KDA_KERNEL_ROOT / "chunk_kda_fwd_post_wu.h": (
            '#include "./kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
        ),
        KDA_KERNEL_ROOT / "chunk_kda_fwd_finalize.h": (
            '#include "./kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
        ),
        KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_prepare.h": (
            '#include "../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
            '#include "../kernel_utils/tile/copy_l0c_to_ub.hpp"',
            '#include "../kernel_utils/vector/regbase.hpp"',
        ),
        KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_post_wu.h": (
            '#include "../kernel_utils/block/block_mmad_pingpong_tla.hpp"',
            '#include "../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
            '#include "../kernel_utils/vector/regbase.hpp"',
        ),
        KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_finalize.h": (
            '#include "../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
        ),
        KDA_KERNEL_ROOT / "arch35/chunk_kda_fwd_fwd_h.h": (
            '#include "../kernel_utils/tile/copy_l0c_to_ub.hpp"',
        ),
        KDA_COMMON_PATH: (
            '#include "./kernel_utils/vector/regbase.hpp"',
            '#include "./kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"',
            '#include "fwd_h/chunk_kda_fwd_h_struct.h"',
            '#include "fwd_h/arch35/gemm/kernel/kda_fwd_h_kernel.hpp"',
            '#include "fwd_h/gemm/kernel/kda_fwd_h_kernel.hpp"',
        ),
        FWD_H_KERNEL_PATHS[0]: (
            '#include "../../../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
        ),
        FWD_H_KERNEL_PATHS[1]: (
            '#include "../../../../kernel_utils/block/block_mmad_pingpong_tla.hpp"',
            '#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"',
            '#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"',
        ),
        FWD_H_ROOT / "arch35/epilogue/block/block_epilogue_kda_fwdh_regbase.hpp": (
            '#include "../../../../kernel_utils/vector/regbase.hpp"',
        ),
    }
    for path, expected_includes in expected_private_includes.items():
        source = path.read_text(encoding="utf-8")
        for expected_include in expected_includes:
            if expected_include not in source:
                raise ValueError(f"{path.name}: missing private utility include {expected_include}")

    gate_path = KDA_ROOT / "kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
    if "kernel_utils/" in gate_path.read_text(encoding="utf-8"):
        raise ValueError("kda_gate_cumsum must not retain the unused regbase dependency")

    common_source = KDA_COMMON_PATH.read_text(encoding="utf-8")
    if "chunk_gated_delta_rule_fwd_h" in common_source:
        raise ValueError("chunk_kda_fwd_common.h must use only the KDA-private FwdH")

    op_cmake = (
        KDA_KERNEL_ROOT.parent / "op_host/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    if "chunk_gated_delta_rule_fwd_h" in op_cmake:
        raise ValueError("chunk_kda_fwd must not retain a build dependency on GDN FwdH")
    if "fla/ops/ascendc/kda/kda_gate_cumsum" not in op_cmake:
        raise ValueError("chunk_kda_fwd must retain its kda_gate_cumsum dependency")


def _check_kernel_sync_contract() -> None:
    source = _cpp_source_for_aicore(
        PREPARE_PATH.read_text(encoding="utf-8"), 310
    )
    compile_gate = (
        "SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128"
    )

    join = _source_section(
        source,
        "__aicore__ inline void JoinAivMte3()",
        "__aicore__ inline void RunAicAfterBothAivReady",
    )
    for token in (
        "if (!isAivOnly_)",
        "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();",
        "PipeBarrier<PIPE_MTE3>();",
    ):
        if token not in join:
            raise ValueError(f"A5 AIV join is missing {token}")
    if "if (!headPairMode_)" in join:
        raise ValueError("head-pair AIV lanes must join before publishing a shared ready token")

    aiv_direct = _source_section(
        source, "const bool useDirectScoreUb =", "bool firstSolveRowsPrepared"
    )
    aic_direct = _source_section(
        source,
        "__aicore__ inline void ProcessChunkPreAicHeadPairFp32",
        "if (!directScoreDispatched)",
    )
    initializer = _source_section(
        source,
        "__aicore__ inline void ProcessPreAivHeadPair()",
        "for (uint64_t task = coreIdx",
    )
    for name, section in (
        ("AIV direct-score eligibility", aiv_direct),
        ("AIC direct-score eligibility", aic_direct),
        ("direct-score credit initialization", initializer),
    ):
        if compile_gate not in section or "KDA_ARCH35_ENABLE_DIRECT_SCORE_UB" not in section:
            raise ValueError(f"{name} drifted from the key2-only compile gate")
    for token in (
        "curT == 64",
        "scoreBlockSize == KDA_DIRECT_SCORE_ROWS",
        "scoreBlockCount == KDA_SCORE_QUEUE_DEPTH",
    ):
        if token not in aiv_direct or token not in aic_direct:
            raise ValueError(f"A5 direct-score producer/consumer mismatch: {token}")
    if "rowCount == KDA_DIRECT_SCORE_ROWS" not in aic_direct:
        raise ValueError("AIC direct-score dispatch must require one direct-score row block")
    if "InitializeDirectScoreUbArch35();" not in initializer:
        raise ValueError("key2 direct-score queue has no initial free credits")

    aiv_pipeline = _source_section(
        source,
        "__aicore__ inline void ProcessChunkPreAivFp32",
        "__aicore__ inline void ProcessChunkPreAicFp32",
    )
    join_index = aiv_pipeline.index("JoinAivMte3();")
    ready_index = aiv_pipeline.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_)"
    )
    if join_index > ready_index:
        raise ValueError("A5 score-ready token is published before both AIV lanes join")

    gate_score_w = _source_braced_block(
        source, "__aicore__ inline LocalTensor<SCORE_T> GateScoreW"
    )
    low_precision_score = _source_braced_block(
        gate_score_w, "if constexpr (!USE_FP32_SCORE)"
    )
    direct_alias = re.fullmatch(
        r"\s*if\s+constexpr\s*\(\s*!USE_FP32_SCORE\s*\)\s*\{\s*"
        r"return\s+GateKTyped\s*\(\s*slot\s*\)\s*\.\s*template\s+"
        r"ReinterpretCast\s*<\s*SCORE_T\s*>\s*\(\s*\)\s*;\s*\}\s*",
        low_precision_score,
    )
    local_alias = re.fullmatch(
        r"\s*if\s+constexpr\s*\(\s*!USE_FP32_SCORE\s*\)\s*\{\s*"
        r"(?:const\s+)?(?:auto|LocalTensor\s*<\s*T\s*>)\s+([A-Za-z_]\w*)\s*=\s*"
        r"GateKTyped\s*\(\s*slot\s*\)\s*;\s*return\s+\1\s*\.\s*template\s+"
        r"ReinterpretCast\s*<\s*SCORE_T\s*>\s*\(\s*\)\s*;\s*\}\s*",
        low_precision_score,
    )
    if direct_alias is None and local_alias is None:
        raise ValueError(
            "safe low-precision GateScoreW ownership must explicitly alias raw K"
        )

    score_factors_raw = _source_braced_block(
        source, "__aicore__ inline void PrepareScoreFactorsBulk"
    )
    score_factors = _normalize_source(score_factors_raw)
    fused_marker = "if (fuseQwKg) {"
    fused_raw = _source_braced_block(score_factors_raw, fused_marker)
    fused = _normalize_source(fused_raw)
    fused_start = score_factors.index(fused_marker)
    fused_qw_call = _source_section(
        fused,
        "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, false, true>(",
        "if constexpr (exportFinalKg)",
    )
    fused_v_arguments = (
        "(__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()), "
        "(__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()), "
        "(__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()), nullptr,"
    )
    if fused_v_arguments not in fused_qw_call:
        raise ValueError(
            "A5 fused Q/W/Kg helper must consume V and leave finalKg output disabled"
        )
    ordered_final_kg = (
        "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, false, true>(",
        "PipeBarrier<PIPE_V>();",
        "SetFlag<HardEvent::V_MTE2>(vToMte2Event_);",
        "WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);",
        (
            "CopyRowsIn(vTyped, k_, QOffset(b, h, start + tileRow, 0), tileRows, K_, "
            "inputSequenceMajor_ ? H_ * K_ : K_);"
        ),
        "SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);",
        "WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);",
        "PrepareKdaGateKgRegbase<T, T, GK_T, true>(",
    )
    try:
        positions = [fused.index(token) for token in ordered_final_kg]
    except ValueError as error:
        raise ValueError("A5 fused finalKg raw-K reload contract is incomplete") from error
    if positions != sorted(positions):
        raise ValueError("A5 fused finalKg must reload raw K after direct-V consumption")
    score_factor_positions = [fused_start + position for position in positions]
    for token in (
        "SetFlag<HardEvent::V_MTE2>(vToMte2Event_);",
        "WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);",
        "SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);",
        "WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);",
    ):
        if fused.count(token) != 1:
            raise ValueError(f"A5 fused finalKg must contain exactly one {token}")
    if fused.count("PrepareKdaGateKgRegbase<T, T, GK_T, true>(") != 1:
        raise ValueError("A5 fused finalKg must have exactly one typed Kg transform")
    in_place_kg = re.compile(
        r"PrepareKdaGateKgRegbase<T, T, GK_T, true>\(\s*"
        r"\(__ubuf__ T \*\)reinterpret_cast<uint64_t>\(vTyped\.GetPhyAddr\(\)\),\s*"
        r"\(__ubuf__ T \*\)reinterpret_cast<uint64_t>\(vTyped\.GetPhyAddr\(\)\),"
    )
    if in_place_kg.search(fused) is None:
        raise ValueError("A5 fused finalKg must use the reloaded raw-K buffer in place")

    kg_helper = _normalize_source(
        _source_braced_block(
            source, "static __simd_vf__ inline void PrepareKdaGateKgRegbase("
        )
    )
    raw_k_load = "LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);"
    kg_store = "StoreKdaGateRegbasePair<OutputT>( kg + offset"
    if (
        kg_helper.count(raw_k_load) != 1
        or kg_helper.count(kg_store) != 1
        or kg_helper.index(raw_k_load) > kg_helper.index(kg_store)
    ):
        raise ValueError("typed Kg in-place transform must load each raw-K offset before storing it")

    early_prefetch = _source_section(
        score_factors, "uint64_t nextTileRow =", "if (fuseQwKg) {"
    )
    if (
        "const bool deferNextPrefetch = fuseQwKg && exportFinalKg;" not in early_prefetch
        or "if (!deferNextPrefetch && nextTileRow < qwEnd)" not in early_prefetch
        or "PrefetchQKGate(qwSlot ^ 1" not in early_prefetch
    ):
        raise ValueError("finalKg raw-K reload must defer the shared MTE2_V prefetch")
    deferred_index = score_factors.find(
        "if (deferNextPrefetch && nextTileRow < qwEnd)", score_factor_positions[-1]
    )
    previous_output_wait = score_factors.index(
        "if (qwOutputPending) { WaitGateOutputForVector();", score_factor_positions[-1]
    )
    previous_slot_release = score_factors.index(
        "if (qwOutputPending) { WaitGateOutputForMte2();"
    )
    current_output_set = score_factors.index(
        "SetFlag<HardEvent::V_MTE3>(vToMte3Event_);", previous_output_wait
    )
    current_output_wait = score_factors.index(
        "WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);", current_output_set
    )
    current_output_branch = score_factors.index(
        "if (useDirectScoreL1) {", current_output_set
    )
    current_output_sync = score_factors[current_output_set:current_output_branch]
    if (
        current_output_sync.count(
            "SetFlag<HardEvent::V_MTE3>(vToMte3Event_);"
        ) != 1
        or current_output_sync.count(
            "WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);"
        ) != 1
    ):
        raise ValueError("finalKg output must wait for the current vector writes")
    final_kg_output = score_factors.index(
        (
            "CopyVectorOut(finalKg_, KVOffset(b, hv, start + tileRow, 0, K_), "
            "vTyped, elems);"
        ),
        current_output_wait,
    )
    output_complete = score_factors.index("SignalGateOutputDone();", current_output_wait)
    slot_toggle = score_factors.index("qwSlot ^= 1;", output_complete)
    if (
        deferred_index < 0
        or previous_slot_release > score_factor_positions[0]
        or deferred_index > previous_output_wait
        or previous_output_wait > current_output_set
        or current_output_set > current_output_wait
        or current_output_wait > final_kg_output
        or final_kg_output > output_complete
        or output_complete > slot_toggle
        or "PrefetchQKGate(qwSlot ^ 1" not in score_factors[
            deferred_index:previous_output_wait
        ]
    ):
        raise ValueError("finalKg path must prefetch the next tile after raw-K consumption")


def _check_empty_varlen_state_contract() -> None:
    for path in FWD_H_SCHEDULER_PATHS:
        source = path.read_text(encoding="utf-8")
        for token in (
            "stream.batchIdx = b - 1;",
            "newStream.tokenBatchIdx = isVariedLen ? newStream.batchIdx : 0;",
            "ResolveVarlenSequence(newStream.tokenBatchIdx, newStream);",
            "uint64_t hSrcOffset;",
            "uint64_t hDstOffset;",
            "uint64_t uvOffset;",
            "uint64_t wkOffset;",
            "uint64_t wOffset;",
            "uint64_t gOffset;",
            "uint64_t gkOffset;",
            "uint64_t initialStateOffset;",
            "uint64_t finalStateOffset;",
            "uint64_t stateHeadIdx =",
            "uint64_t chunkLinearIdx =",
            "uint64_t tokenLinearV =",
            "uint64_t tokenLinearK =",
        ):
            if token not in source:
                raise ValueError(f"{path.name}: compact/original varlen mapping is missing {token}")
        update_task = _source_section(source, "void UpdateTask", "void InitTasks")
        for token in (
            "static_cast<uint64_t>(stream.batchIdx) * vNumHead",
            "static_cast<uint64_t>(stream.shapeBatchIdx) * vNumHead",
            "static_cast<uint64_t>(stream.shapeBatchIdx) * kNumHead",
            "static_cast<uint64_t>(stream.chunkIdx) * chunkSize",
            "static_cast<uint64_t>(kHeadDim) * vHeadDim",
        ):
            if token not in update_task:
                raise ValueError(f"{path.name}: 64-bit UpdateTask arithmetic is missing {token}")

    for path in FWD_H_KERNEL_PATHS:
        source = path.read_text(encoding="utf-8")
        for token in (
            "PresetEmptyVarlenFinalState();",
            "vecBlockScheduler.inputTokenBatch == vecBlockScheduler.tokenBatch",
            "vecBlockScheduler.ResolveVarlenSequence(batchIdx, resolvedStream);",
            "stateBatchIdx = resolvedStream.batchIdx;",
            "gmInitialState[stateOffset]",
            "static_cast<ElementFinalState>(0)",
            "gmFinalState[stateOffset]",
            "HardEvent::MTE3_MTE2",
            "gmSeqlen.GetValue(batchIdx)",
            "gmSeqlen.GetValue(batchIdx + 1)",
            "if (seqStart != seqEnd)",
            "uint64_t stateBlockSize =",
            "uint64_t stateBaseOffset =",
            "uint64_t stateOffset =",
            "uint64_t hBaseOffset =",
            "uint64_t initialStateBaseOffset =",
            "uint64_t hOffset =",
            "uint64_t initialStateOffset =",
            "static_cast<uint64_t>(shapeBatch) * kNumHead",
            "static_cast<uint64_t>(shapeBatch) * vNumHead",
            "static_cast<uint64_t>(taskIdx) * stateBlockSize",
            "static_cast<uint64_t>(shapeBatchIdx) * vNumHead",
            "static_cast<uint64_t>(stateBatchIdx) * vNumHead",
        ):
            if token not in source:
                raise ValueError(f"{path.name}: empty-varlen state contract is missing {token}")
        preset = _source_section(
            source,
            "__aicore__ inline void PresetEmptyVarlenFinalState",
            "__aicore__ inline void Process",
        )
        seq_start = "gmSeqlen.GetValue(batchIdx)"
        seq_end = "gmSeqlen.GetValue(batchIdx + 1)"
        empty_filter = "if (seqStart != seqEnd)"
        empty_continue = "continue;"
        first_slot_wait = "WaitFlag<AscendC::HardEvent::MTE3_MTE2>"
        final_state_store = "DataCopy(gmFinalState[stateOffset]"
        slot_release = "SetFlag<AscendC::HardEvent::MTE3_MTE2>"
        ordered_tokens = (
            seq_start,
            seq_end,
            empty_filter,
            empty_continue,
            first_slot_wait,
            final_state_store,
        )
        if any(token not in preset for token in ordered_tokens):
            raise ValueError(
                f"{path.name}: empty-varlen preset ordering tokens are incomplete"
            )
        positions = [preset.index(token) for token in ordered_tokens]
        if positions != sorted(positions):
            raise ValueError(
                f"{path.name}: empty tasks must be filtered before consuming an MTE3_MTE2 slot"
            )
        release_pos = preset.find(slot_release, positions[-1])
        if release_pos < 0:
            raise ValueError(
                f"{path.name}: final-state write must release its MTE3_MTE2 slot"
            )

    large_state_elements = 513 * 128 * 256 * 256
    if large_state_elements <= (1 << 32):
        raise ValueError("large-state offset contract no longer exercises the 32-bit boundary")


def _check_post_wu_and_fwd_h_contract() -> None:
    for path in KDA_MMAD_PATHS:
        source = path.read_text(encoding="utf-8")
        if "l0CEventList[0] = 0;" not in source:
            raise ValueError(f"{path.name}: unit-flag L0C event is uninitialized")

    post_wu = POST_WU_PATH.read_text(encoding="utf-8")
    for token in (
        "using KdaDispatchPolicy = Common::MmadPingpong",
        "using KdaWideDispatchPolicy = Common::MmadPingpong",
        "using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy",
        "using UBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy",
        "if (K_ <= 128)",
    ):
        if token not in post_wu:
            raise ValueError(f"A5 Post-WU wide MMAD contract is missing {token}")
    post_aic = _source_section(
        post_wu,
        "__aicore__ inline void ProcessChunkPostAicTyped",
        "__aicore__ inline void ProcessPostAiv",
    )
    tail_guard = "if (curT < BT_) { return; }"
    if tail_guard not in post_aic or post_aic.index(tail_guard) > post_aic.index(
        "ComputePostWuCube"
    ):
        raise ValueError("A5 Post-WU partial chunks must remain AIV-owned")

    prepare = PREPARE_PATH.read_text(encoding="utf-8")
    tiling = KDA_TILING_PATH.read_text(encoding="utf-8")
    common = KDA_COMMON_PATH.read_text(encoding="utf-8")
    entry = KDA_ENTRY_PATH.read_text(encoding="utf-8")
    vnew = VNEW_EPILOGUE_PATH.read_text(encoding="utf-8")
    precision_contracts = {
        "A5 K128 FP32 score host selector": (
            tiling,
            (
                "!useChunk64K128V128Template && !safeGate",
                "qDesc->GetDataType() == ge::DT_BF16 && shape.kDim == 128",
                "shape.vDim >= shape.kDim",
                "useFp32Score ? sizeof(float) : dataBytes",
            ),
        ),
        "A5 K128 FP32 score kernel selector": (
            prepare,
            (
                "COMPILE_BT == 0 && COMPILE_K == 0 && COMPILE_V == 0",
                "!SAFE_GATE && IsSameType<T, bfloat16_t>::value",
                "tiling.kHeadDim == 128 && tiling.vHeadDim >= tiling.kHeadDim",
                "RunChunkKdaPrepareImpl<true",
            ),
        ),
        "KDA residual selector and dispatch": (
            common,
            (
                "struct KgResidualPolicy",
                "!tiling.fusePostWu && !tiling.fusePostWuIntoFwdH",
                "RunChunkKdaPostWu<true",
                "RunFwdHImpl<true",
                "addresses.uSeed",
            ),
        ),
        "split-stage residual carrier": (
            entry,
            (
                "addresses.uSeed = uSeed;",
                "RunPostWu<SAFE_GATE",
                "RunFwdH<SAFE_GATE",
            ),
        ),
        "Post-WU K-gate residual producer": (
            post_wu,
            (
                "PRESERVE_KG_RESIDUAL",
                "ClampScoreExpInput(expLocal",
                "CopyRowsOut(",
                "propagatedVNew_",
                "CrossCoreBarrier<0x1, PIPE_MTE3>()",
                "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>",
                "CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>",
            ),
        ),
        "V1 duplicated-V and K-residual planes": (
            vnew,
            (
                "reductionRows = hiLoC2 ? 2 * mActual : mActual",
                "(mActual + rowBegin) * SIZE_16_NUM_PER_C0",
                "kResidualInput[rowBegin * vHeadDim]",
                "kDecayWorkspace[(mActual + rowBegin) * nkActual]",
            ),
        ),
    }
    for name, (source, tokens) in precision_contracts.items():
        for token in tokens:
            if token not in source:
                raise ValueError(f"{name} is missing {token}")

    arch35_fwd_h = FWD_H_KERNEL_PATHS[1].read_text(encoding="utf-8")
    for token in (
        "ComputeCube2RowTiles",
        "rowOffset += CUBE2_ROW_TILE_M",
        "if (cube1Offsets.blockTokens < 16)",
        "bool useTailVector = cube2Offsets.blockTokens < 16",
        "uint32_t paddedTokens =",
        "HardEvent::MTE3_V",
        "if constexpr (kGated)",
        "gmK[offsets.wkOffset + tokenRow * kHeadDim]",
        "if constexpr (HI_LO_C2)",
        "uint32_t reductionRows = 2 * cube2Offsets.blockTokens",
        "2ULL * cube2Offsets.kDecayWorkOffset",
        "2ULL * cube2Offsets.vWorkOffset",
        "gmKResidual[vec1Offsets.uvOffset]",
    ):
        if token not in arch35_fwd_h:
            raise ValueError(f"A5 FwdH tail/row-tile contract is missing {token}")


def _check_generator(module, manifests: dict[str, list[dict]]) -> None:
    if module.STANDARD != ACCURACY_STANDARD:
        raise ValueError("accuracy generator must use the mixed_tolerance_bm single benchmark")
    if any(case.get("standard") != ACCURACY_STANDARD for case in manifests["accuracy"]):
        raise ValueError("accuracy manifest must use the mixed_tolerance_bm single benchmark")

    expected = {
        "accuracy": module.build_accuracy_specs(),
        "mss": module.build_mss_specs(),
        "perf": module.build_perf_specs(),
    }
    for name, specs in expected.items():
        path = OP_DIR / f"atk_chunk_kda_fwd{'_' + name if name != 'accuracy' else ''}.json"
        actual = manifests[name]
        generated = [module._case_payload(item, manifest=name) for item in specs]
        if actual != generated:
            raise ValueError(f"{path.name}: materialized payloads drifted from generator")


def _contains_exact(specs: list[dict], expected: dict[str, object]) -> bool:
    return any(all(spec.get(name) == value for name, value in expected.items()) for spec in specs)


def _check_accuracy_coverage(specs: list[dict], module) -> None:
    profile_dtypes: dict[str, set[str]] = {}
    for spec in specs:
        profile = str(spec["design_id"]).rsplit("-", 1)[0]
        profile_dtypes.setdefault(profile, set()).add(str(spec["q_dtype"]))
    if len(profile_dtypes) != 100 or any(dtypes != {"bf16", "fp16"} for dtypes in profile_dtypes.values()):
        raise ValueError("accuracy must contain 100 structural profiles paired with BF16/FP16")

    gate_variants = {
        (bool(spec["use_gate_in_kernel"]), bool(spec["safe_gate"]), bool(spec["dt_bias"]))
        for spec in specs
    }
    if gate_variants != GATE_VARIANTS:
        raise ValueError(f"accuracy gate coverage mismatch: {sorted(gate_variants)}")
    if {str(spec["g_dtype"]) for spec in specs} != {"fp32", "bf16"}:
        raise ValueError("accuracy must cover FP32 and BF16 gate tensors")
    gate_dtype_variants = {
        (
            bool(spec["use_gate_in_kernel"]), bool(spec["safe_gate"]), bool(spec["dt_bias"]),
            str(spec["g_dtype"]),
        )
        for spec in specs
    }
    expected_gate_dtype_variants = {
        variant + (g_dtype,)
        for variant in GATE_VARIANTS
        for g_dtype in ("fp32", "bf16")
    }
    if gate_dtype_variants != expected_gate_dtype_variants:
        raise ValueError("each gate variant must cover FP32 and BF16 gate tensors")
    output_policies = {
        (bool(spec["disable_recompute"]), bool(spec["return_intermediate_states"]))
        for spec in specs
    }
    if output_policies != OUTPUT_POLICIES:
        raise ValueError(f"accuracy output-policy coverage mismatch: {sorted(output_policies)}")
    if {bool(spec["state_v_first"]) for spec in specs} != {False, True}:
        raise ValueError("accuracy must cover both state layouts")
    if {int(spec["K"]) for spec in specs} != {16, 128, 256}:
        raise ValueError("accuracy must cover K=16/128/256")
    actual_shapes = {
        (int(spec["chunk_size"]), int(spec["K"]), int(spec["V"])) for spec in specs
    }
    if not SHAPE_VARIANTS.issubset(actual_shapes):
        raise ValueError(f"accuracy shape coverage mismatch: {sorted(actual_shapes)}")
    if not any(
        bool(spec["state_v_first"])
        and int(spec["K"]) != int(spec["V"])
        and any(
            bool(spec[name])
            for name in (
                "initial_state", "output_final_state",
                "disable_recompute", "return_intermediate_states",
            )
        )
        for spec in specs
    ):
        raise ValueError("state_v_first=true must exercise an observable K/V-asymmetric state path")
    if {int(spec["tiling_key"]) for spec in specs} != {1, 2}:
        raise ValueError("accuracy must exercise both tiling keys")

    staged_fallback_common = {
        "B": 1, "H": 1, "HV": 1, "T": 512, "K": 256, "V": 128,
        "chunk_size": 64, "layout": "TND",
        "cu_seqlens": "0,128,341,512", "explicit_chunk_indices": True,
        "g_dtype": "bf16", "beta_dtype": "bf16", "data_profile": "uniform",
        "data_scale": 0.2, "gate_scale": 0.04, "qk_scale": 0.25,
        "v_scale": 0.25, "beta_scale": 0.35, "beta_bias": 1.5,
        "beta_low": 0.2, "beta_high": 0.8, "a_log_scale": 0.12,
        "dt_bias_scale": 0.5, "dt_bias_mean": -3.0,
        "lower_bound": -5.0, "state_scale": 0.02, "scale": 0.0625,
        "initial_state": False, "output_final_state": False,
        "safe_gate": False, "use_gate_in_kernel": True, "dt_bias": False,
        "disable_recompute": True, "return_intermediate_states": False,
        "state_v_first": False, "tiling_key": 1,
    }
    for case_id, q_dtype, seed in (
        (68, "bf16", 20260899),
        (69, "fp16", 20260900),
    ):
        expected = {
            **staged_fallback_common,
            "case_id": case_id,
            "q_dtype": q_dtype,
            "seed": seed,
        }
        spec = specs[case_id]
        if not all(spec.get(name) == value for name, value in expected.items()):
            raise ValueError(
                f"accuracy case {case_id} must preserve the K=256 staged-fallback regression"
            )

    dense_final_kg_common = {
        "q_dtype": "bf16", "g_dtype": "fp32",
        "K": 128, "V": 128, "chunk_size": 64, "cu_seqlens": "",
        "explicit_chunk_indices": False, "safe_gate": True,
        "use_gate_in_kernel": True, "tiling_key": 2,
        "data_profile": "uniform", "qk_scale": 0.25, "v_scale": 0.25,
        "beta_scale": 0.35, "beta_bias": 1.5, "a_log_scale": 0.12,
        "dt_bias_scale": 0.5, "dt_bias_mean": -3.0,
        "lower_bound": -5.0, "state_scale": 0.02,
        "scale": 0.08838834764831843,
    }
    dense_final_kg_cases = {
        56: {
            "B": 1, "H": 1, "HV": 2, "T": 256, "layout": "BSND",
            "beta_dtype": "bf16", "initial_state": False,
            "output_final_state": False, "dt_bias": True,
            "data_scale": 0.03, "gate_scale": 0.01,
            "beta_low": 0.1, "beta_high": 0.9,
            "disable_recompute": False, "return_intermediate_states": False,
            "state_v_first": False, "seed": 20260887,
        },
        130: {
            "B": 2, "H": 2, "HV": 2, "T": 192, "layout": "BNSD",
            "beta_dtype": "fp32", "initial_state": False,
            "output_final_state": True, "dt_bias": False,
            "data_scale": 0.08, "gate_scale": 0.02,
            "beta_low": 0.05, "beta_high": 0.95,
            "disable_recompute": False, "return_intermediate_states": True,
            "state_v_first": True, "seed": 20260961,
        },
        140: {
            "B": 1, "H": 1, "HV": 2, "T": 256, "layout": "TND",
            "beta_dtype": "bf16", "initial_state": True,
            "output_final_state": False, "dt_bias": True,
            "data_scale": 0.2, "gate_scale": 0.04,
            "beta_low": 0.2, "beta_high": 0.8,
            "disable_recompute": True, "return_intermediate_states": False,
            "state_v_first": False, "seed": 20260971,
        },
    }
    for case_id, case_fields in dense_final_kg_cases.items():
        expected = {
            **dense_final_kg_common,
            **case_fields,
            "case_id": case_id,
        }
        spec = specs[case_id]
        if not all(spec.get(name) == value for name, value in expected.items()):
            raise ValueError(
                f"accuracy case {case_id} must preserve the dense finalKg regression"
            )

    a5_fusion = {
        "q_dtype": "bf16", "g_dtype": "fp32", "B": 1, "H": 1, "HV": 2,
        "T": 256, "K": 128, "V": 128, "chunk_size": 64, "layout": "BSND",
        "initial_state": False, "output_final_state": False, "cu_seqlens": "",
        "explicit_chunk_indices": False, "safe_gate": True,
        "use_gate_in_kernel": True, "dt_bias": True, "disable_recompute": False,
        "return_intermediate_states": False, "state_v_first": False, "tiling_key": 2,
    }
    if not _contains_exact(specs, a5_fusion):
        raise ValueError("accuracy is missing the fixed A5 key2 fusion candidate")

    hang_regression = {
        "q_dtype": "bf16", "g_dtype": "fp32", "beta_dtype": "fp32",
        "B": 1, "H": 2, "HV": 2, "T": 65, "K": 128, "V": 256,
        "chunk_size": 64, "layout": "BNSD", "initial_state": False,
        "output_final_state": True, "cu_seqlens": "", "explicit_chunk_indices": False,
        "safe_gate": True, "use_gate_in_kernel": True, "dt_bias": True,
        "disable_recompute": True, "return_intermediate_states": True,
        "state_v_first": False, "tiling_key": 1,
    }
    if not _contains_exact(specs, hang_regression):
        raise ValueError("accuracy is missing the fixed key1 hang regression")

    empty_zero_state = {
        "q_dtype": "bf16", "B": 1, "H": 2, "HV": 2,
        "T": 64, "K": 128, "V": 128, "chunk_size": 128,
        "layout": "BNSD", "initial_state": False,
        "output_final_state": True, "cu_seqlens": "0,0,0,16,16,64,64",
        "explicit_chunk_indices": False, "state_v_first": False, "tiling_key": 1,
    }
    if not _contains_exact(specs, empty_zero_state):
        raise ValueError("accuracy is missing the empty-sequence zero-state regression")

    empty_initial_state = {
        "q_dtype": "bf16", "B": 1, "H": 1, "HV": 4,
        "T": 128, "K": 128, "V": 128, "chunk_size": 64,
        "layout": "BSND", "initial_state": True,
        "output_final_state": True, "cu_seqlens": "0,0,0,32,32,128,128",
        "explicit_chunk_indices": True, "disable_recompute": True,
        "return_intermediate_states": True, "state_v_first": True, "tiling_key": 2,
    }
    if not _contains_exact(specs, empty_initial_state):
        raise ValueError("accuracy is missing the empty-sequence initial-state regression")


def _a5_launch_mode(spec: dict) -> str:
    chunk_size = int(spec["chunk_size"])
    raw_cu = str(spec.get("cu_seqlens", "")).strip()
    if raw_cu:
        cu = [int(value) for value in raw_cu.split(",")]
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != int(spec["T"]):
            raise ValueError(f"case {spec['case_id']} has invalid cu_seqlens")
        if any(end < begin for begin, end in zip(cu, cu[1:])):
            raise ValueError(f"case {spec['case_id']} has decreasing cu_seqlens")
        total_chunks = sum(
            (end - begin + chunk_size - 1) // chunk_size
            for begin, end in zip(cu, cu[1:])
        )
    else:
        total_chunks = (int(spec["T"]) + chunk_size - 1) // chunk_size

    use_dense_a5_fast_path = (
        not raw_cu
        and spec["q_dtype"] == "bf16"
        and chunk_size == 64
        and int(spec["K"]) == 128
        and int(spec["V"]) == 128
        and int(spec["T"]) % chunk_size == 0
    )
    return "staged" if total_chunks > 1 and not use_dense_a5_fast_path else "full"


def _check_mss_coverage(
    specs: list[dict], accuracy_specs: list[dict], module
) -> None:
    base_specs = specs[:4]
    if [int(spec["case_id"]) for spec in base_specs] != list(range(4)):
        raise ValueError("the four original MSS rows must retain IDs 0--3")
    if {
        (int(spec["tiling_key"]), bool(spec["initial_state"]))
        for spec in base_specs
    } != {(key, boundary) for key in (1, 2) for boundary in (False, True)}:
        raise ValueError("MSS must contain ordinary and boundary rows for each key")

    expected_unsafe_sources = tuple(module.MSS_UNSAFE_SOURCE_CASES)
    expected_sources = expected_unsafe_sources + (
        tuple(module.MSS_FP16_K256_SOURCE_CASE),
    ) + tuple(
        (int(module.MSS_DIRECT_C2_SOURCE_CASE), "full")
        for _ in module.MSS_DIRECT_C2_CASE_IDS
    )
    cloned_specs = [spec for spec in specs if "source_accuracy_case_id" in spec]
    actual_sources = [
        (int(spec["source_accuracy_case_id"]), str(spec.get("a5_launch_mode", "")))
        for spec in cloned_specs
    ]
    if sorted(actual_sources) != sorted(expected_sources):
        raise ValueError(
            f"MSS cloned launch coverage drifted: expected {expected_sources}, "
            f"got {sorted(actual_sources)}"
        )

    accuracy_by_id = {int(spec["case_id"]): spec for spec in accuracy_specs}
    identity_fields = {
        "case_id", "case_key", "design_id", "profile", "tags", "manifest",
        "source_accuracy_case_id", "source_accuracy_case_key", "a5_launch_mode",
        "a5_fwd_h_path",
    }
    for local_case_id, (source_case_id, expected_mode) in enumerate(
        expected_unsafe_sources, start=4
    ):
        spec = specs[local_case_id]
        source = accuracy_by_id[source_case_id]
        if int(spec.get("source_accuracy_case_id", -1)) != source_case_id:
            raise ValueError(f"MSS case {local_case_id} source case drifted")
        if spec.get("source_accuracy_case_key") != source["case_key"]:
            raise ValueError(f"MSS case {local_case_id} source key drifted")
        if spec.get("manifest") != "mss" or source.get("manifest") != "accuracy":
            raise ValueError(f"MSS case {local_case_id} manifest identity drifted")
        if set(spec) != set(source) | {
            "source_accuracy_case_id", "source_accuracy_case_key", "a5_launch_mode",
        }:
            raise ValueError(f"MSS case {local_case_id} source field set drifted")
        for name, value in source.items():
            if name not in identity_fields and spec.get(name) != value:
                raise ValueError(
                    f"MSS case {local_case_id} changed canonical source field {name}"
                )
        actual_mode = _a5_launch_mode(spec)
        if spec.get("a5_launch_mode") != expected_mode or actual_mode != expected_mode:
            raise ValueError(
                f"MSS case {local_case_id} launch mode drifted: "
                f"declared={spec.get('a5_launch_mode')}, derived={actual_mode}"
            )
        if not (
            spec["q_dtype"] == "bf16"
            and spec["g_dtype"] == "fp32"
            and not spec["safe_gate"]
            and int(spec["tiling_key"]) == 1
            and int(spec["K"]) == 128
            and int(spec["V"]) >= int(spec["K"])
        ):
            raise ValueError(f"MSS case {local_case_id} misses the A5 precision selector")

    tail_case_id = int(module.MSS_VARLEN_TAIL_CASE_ID)
    tail = specs[tail_case_id]
    expected_tail = {
        "case_id": 6,
        "case_key": "mss_varlen_tail_h2_hv96_t63_key2",
        "design_id": "KDA-FWD-MSS-VARLEN-TAIL-H96",
        "profile": "determinism_regression",
        "soc": "all",
        "B": 1,
        "H": 2,
        "HV": 96,
        "T": 63,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "layout": "BSND",
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "scale": 0.08838834764831843,
        "initial_state": False,
        "output_final_state": True,
        "cu_seqlens": "0,63",
        "explicit_chunk_indices": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": False,
        "negative_case": False,
        "data_profile": "model_h96",
        "data_scale": 0.08,
        "gate_scale": 1.25,
        "qk_scale": 0.05,
        "v_scale": 0.05,
        "beta_scale": 0.35,
        "beta_bias": 1.5,
        "a_log_scale": 0.12,
        "dt_bias_scale": 1.65,
        "dt_bias_mean": -3.0,
        "beta_low": 0.1,
        "beta_high": 0.9,
        "state_scale": 0.02,
        "tiling_key": 2,
        "expected_tiling_key": 2,
        "seed": 4,
    }
    if tail_case_id != 6 or any(
        tail.get(name) != value for name, value in expected_tail.items()
    ):
        raise ValueError("MSS case 6 varlen tail regression drifted")
    if set(tail.get("target_platforms", ())) != SOCS:
        raise ValueError("MSS case 6 must retain cross-SoC coverage")
    required_tags = {
        "mss", "determinism", "mssanitizer", "regression", "boundary",
        "varlen", "tiling_key_2", "issue440",
    }
    if set(str(tail.get("tags", "")).split(",")) != required_tags:
        raise ValueError("MSS case 6 regression tags drifted")

    fp16_case_id = int(module.MSS_FP16_K256_CASE_ID)
    fp16_source_id, fp16_expected_mode = module.MSS_FP16_K256_SOURCE_CASE
    fp16_spec = specs[fp16_case_id]
    fp16_source = accuracy_by_id[int(fp16_source_id)]
    if int(fp16_spec.get("source_accuracy_case_id", -1)) != int(fp16_source_id):
        raise ValueError("MSS FP16 K=256 source case drifted")
    if fp16_spec.get("source_accuracy_case_key") != fp16_source["case_key"]:
        raise ValueError("MSS FP16 K=256 source key drifted")
    if fp16_spec.get("manifest") != "mss" or fp16_source.get("manifest") != "accuracy":
        raise ValueError("MSS FP16 K=256 manifest identity drifted")
    if set(fp16_spec) != set(fp16_source) | {
        "source_accuracy_case_id", "source_accuracy_case_key", "a5_launch_mode",
    }:
        raise ValueError("MSS FP16 K=256 source field set drifted")
    for name, value in fp16_source.items():
        if name not in identity_fields and fp16_spec.get(name) != value:
            raise ValueError(f"MSS FP16 K=256 changed canonical source field {name}")
    actual_mode = _a5_launch_mode(fp16_spec)
    if (
        fp16_spec.get("a5_launch_mode") != fp16_expected_mode
        or actual_mode != fp16_expected_mode
    ):
        raise ValueError(
            "MSS FP16 K=256 launch mode drifted: "
            f"declared={fp16_spec.get('a5_launch_mode')}, derived={actual_mode}"
        )
    if not (
        fp16_spec["q_dtype"] == "fp16"
        and int(fp16_spec["tiling_key"]) == 1
        and int(fp16_spec["K"]) == 256
        and int(fp16_spec["V"]) == 128
    ):
        raise ValueError("MSS FP16 K=256 case misses the A5 FP32-state path")

    direct_source_id = int(module.MSS_DIRECT_C2_SOURCE_CASE)
    direct_source = accuracy_by_id[direct_source_id]
    direct_case_ids = tuple(int(case_id) for case_id in module.MSS_DIRECT_C2_CASE_IDS)
    for local_case_id, output_final_state in zip(direct_case_ids, (True, False)):
        spec = specs[local_case_id]
        if int(spec.get("source_accuracy_case_id", -1)) != direct_source_id:
            raise ValueError(f"MSS direct-C2 case {local_case_id} source case drifted")
        if spec.get("source_accuracy_case_key") != direct_source["case_key"]:
            raise ValueError(f"MSS direct-C2 case {local_case_id} source key drifted")
        if spec.get("manifest") != "mss" or direct_source.get("manifest") != "accuracy":
            raise ValueError(f"MSS direct-C2 case {local_case_id} manifest identity drifted")
        if set(spec) != set(direct_source) | {
            "source_accuracy_case_id", "source_accuracy_case_key", "a5_launch_mode",
            "a5_fwd_h_path",
        }:
            raise ValueError(f"MSS direct-C2 case {local_case_id} field set drifted")
        variant_fields = identity_fields | {"output_final_state", "optional_spec"}
        for name, value in direct_source.items():
            if name not in variant_fields and spec.get(name) != value:
                raise ValueError(
                    f"MSS direct-C2 case {local_case_id} changed source field {name}"
                )
        expected_optional = (
            "initial=False,final=True,varlen=False,indices=False"
            if output_final_state
            else "initial=False,final=False,varlen=False,indices=False"
        )
        if not (
            spec.get("a5_launch_mode") == "full"
            and _a5_launch_mode(spec) == "full"
            and spec.get("a5_fwd_h_path") == "direct_c2_ub"
            and spec["q_dtype"] == "bf16"
            and not spec["cu_seqlens"]
            and int(spec["chunk_size"]) == 64
            and int(spec["K"]) == 128
            and int(spec["V"]) == 128
            and int(spec["T"]) % int(spec["chunk_size"]) == 0
            and int(spec["B"]) * int(spec["HV"]) >= 16
            and bool(spec["output_final_state"]) is output_final_state
            and spec.get("optional_spec") == expected_optional
        ):
            raise ValueError(f"MSS direct-C2 case {local_case_id} selector drifted")


def _check_yaml_input_contract(manifests: dict[str, list[dict]]) -> None:
    design = yaml.safe_load((OP_DIR / "chunk_kda_fwd.yaml").read_text(encoding="utf-8"))
    if design.get("standard") != ACCURACY_STANDARD:
        raise ValueError("chunk_kda_fwd.yaml must use the mixed_tolerance_bm single benchmark")
    yaml_inputs = design.get("inputs")
    if not isinstance(yaml_inputs, list):
        raise ValueError("chunk_kda_fwd.yaml must declare an inputs list")
    yaml_contract = []
    for item in yaml_inputs:
        if not isinstance(item, dict):
            raise ValueError("chunk_kda_fwd.yaml inputs must be mappings")
        dtype_values = item.get("dtypes", {}).get("values", [])
        if not isinstance(dtype_values, list) or not dtype_values:
            raise ValueError(f"YAML input {item.get('name')}: dtypes.values must be non-empty")
        valid_values = item.get("ranges", {}).get("valid", {}).get("values", [])
        if not isinstance(valid_values, list):
            raise ValueError(f"YAML input {item.get('name')}: ranges.valid.values must be a list")
        yaml_contract.append(
            (
                str(item.get("name")),
                str(item.get("type")),
                bool(item.get("required")),
                {str(dtype) for dtype in dtype_values},
                valid_values,
            )
        )
    yaml_names = [item[0] for item in yaml_contract]
    if len(yaml_names) != len(set(yaml_names)):
        raise ValueError("chunk_kda_fwd.yaml contains duplicate input names")
    for manifest_name, cases in manifests.items():
        for case in cases:
            actual_inputs = case.get("inputs", [])
            if not isinstance(actual_inputs, list) or any(
                not isinstance(item, dict) for item in actual_inputs
            ):
                raise ValueError(f"{manifest_name} case {case.get('id')}: invalid inputs list")
            actual_names = [str(item.get("name")) for item in actual_inputs]
            if len(actual_names) != len(set(actual_names)):
                raise ValueError(
                    f"{manifest_name} case {case.get('id')}: duplicate input names"
                )
            if len(actual_inputs) != len(yaml_contract):
                raise ValueError(
                    f"{manifest_name} case {case.get('id')}: input count mismatch; "
                    f"YAML={len(yaml_contract)}, JSON={len(actual_inputs)}"
                )
            for index, (actual, expected) in enumerate(zip(actual_inputs, yaml_contract)):
                expected_name, expected_type, expected_required, allowed_dtypes, valid_values = expected
                actual_name = str(actual.get("name"))
                actual_type = str(actual.get("type"))
                actual_required = bool(actual.get("required"))
                actual_dtype = str(actual.get("dtype"))
                if (
                    actual_name != expected_name
                    or actual_type != expected_type
                    or actual_required != expected_required
                    or actual_dtype not in allowed_dtypes
                ):
                    raise ValueError(
                        f"{manifest_name} case {case.get('id')} input {index}: "
                        f"expected name/type/required/dtype={expected_name}/{expected_type}/"
                        f"{expected_required}/{sorted(allowed_dtypes)}, got "
                        f"{actual_name}/{actual_type}/{actual_required}/{actual_dtype}"
                    )
                if expected_name in ENUMERATED_YAML_INPUTS and actual.get("range_values") not in valid_values:
                    raise ValueError(
                        f"{manifest_name} case {case.get('id')} input {index}: "
                        f"{expected_name} value {actual.get('range_values')!r} is missing from YAML valid values"
                    )


def main() -> int:
    _check_source_evidence()
    _check_kernel_utils_ownership()
    _check_l1_clear_barriers()
    _check_kernel_sync_contract()
    _check_empty_varlen_state_contract()
    _check_post_wu_and_fwd_h_contract()
    module = _load_generator()
    manifests = {
        "accuracy": _read(OP_DIR / "atk_chunk_kda_fwd.json"),
        "mss": _read(OP_DIR / "atk_chunk_kda_fwd_mss.json"),
        "perf": _read(OP_DIR / "atk_chunk_kda_fwd_perf.json"),
    }
    accuracy_specs = _check_manifest(OP_DIR / "atk_chunk_kda_fwd.json", 200, {1, 2})
    mss_specs = _check_manifest(
        OP_DIR / "atk_chunk_kda_fwd_mss.json", module.MSS_COUNT, {1, 2}
    )
    perf_specs = _check_manifest(OP_DIR / "atk_chunk_kda_fwd_perf.json", 2, {1, 2})
    _check_generator(module, manifests)
    _check_stress_driver(module.MSS_COUNT)
    _check_yaml_input_contract(manifests)
    _check_accuracy_coverage(accuracy_specs, module)
    _check_mss_coverage(mss_specs, accuracy_specs, module)
    if {int(item["tiling_key"]) for item in perf_specs} != {1, 2}:
        raise ValueError("performance must exercise both tiling keys")
    print(
        f"chunk_kda_fwd manifests valid: accuracy=200, "
        f"mss={module.MSS_COUNT}, perf=2, "
        f"keys=[1, 2]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
