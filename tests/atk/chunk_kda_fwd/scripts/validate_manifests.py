#!/usr/bin/env python3
"""Validate the checked-in chunk_kda_fwd ATK manifests without an NPU."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import yaml


HERE = Path(__file__).resolve()
OP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
GEN_PATH = OP_DIR / "gen_chunk_kda_fwd.py"
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
KDA_OP_API_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp"
)
KDA_TILING_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp"
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
KDA_BWD_INTRA_REGBASE_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/arch35/kernel_utils/vector/regbase.hpp"
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
    KDA_BWD_INTRA_REGBASE_PATH,
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


def _check_manifest(path: Path, expected_count: int, expected_keys: set[int]) -> list[dict]:
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
        if spec.get("soc") != "all":
            raise ValueError(f"{path.name}: canonical manifests must use soc=all")
        if set(spec.get("target_platforms", [])) != SOCS:
            raise ValueError(f"{path.name}: target platform matrix is incomplete")
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


def _source_section(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return " ".join(source[start:end].split())


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
        KDA_BWD_INTRA_REGBASE_PATH: "FLA_NPU_KERNEL_UTIL_REGBASE_PROVIDED",
    }
    for path, marker in provider_markers.items():
        source = path.read_text(encoding="utf-8")
        if f"#define {marker} 1" not in source:
            raise ValueError(f"{path.name}: missing private utility provider marker {marker}")

    include_re = re.compile(
        r'^\s*#\s*include\s*[<"]([^">]+)[">]', flags=re.MULTILINE
    )
    for path in KDA_ROOT.rglob("*"):
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
        REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/arch35/chunk_kda_bwd_intra_regbase.h": (
            '#include "./kernel_utils/vector/regbase.hpp"',
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

    cmake_source = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if "${_kernel_source_dir}" in cmake_source:
        raise ValueError("RTY private-source install must not change every operator")
    kda_install = _source_section(
        cmake_source,
        'if (_op_name STREQUAL "chunk_kda_fwd" OR',
        "endif ()",
    )
    if '_op_name STREQUAL "chunk_kda_bwd_intra"' not in kda_install:
        raise ValueError("RTY private-source install must remain scoped to KDA operators")
    for install_source in (
        "${_kda_kernel_source_dir}/kernel_utils",
        "${_kda_kernel_source_dir}/fwd_h",
        "${_kda_kernel_source_dir}/${_arch_dir}",
    ):
        if f"install(DIRECTORY {install_source}" not in kda_install:
            raise ValueError(f"RTY package must install KDA-private path {install_source}")
    for public_install_source in (
        "${op_dir}/arch32",
        "${op_dir}/arch35",
        "${op_dir}/arch38",
    ):
        if f"install(DIRECTORY {public_install_source}" not in cmake_source:
            raise ValueError(
                f"non-KDA RTY install behavior must retain {public_install_source}"
            )

    validation_script = (
        REPO_ROOT / "scripts/validate_kda_a5.sh"
    ).read_text(encoding="utf-8")
    default_ops = re.search(r'^ops="([^"]+)"$', validation_script, re.MULTILINE)
    if default_ops is None:
        raise ValueError("A5 validation script must define its default KDA operator set")
    default_op_names = set(default_ops.group(1).split(","))
    if "chunk_gated_delta_rule_fwd_h" in default_op_names:
        raise ValueError("A5 KDA validation must not build the independent GDN FwdH")
    if not {"chunk_kda_fwd", "kda_gate_cumsum"}.issubset(default_op_names):
        raise ValueError("A5 KDA validation must build ChunkKdaFwd and KdaGateCumsum")


def _check_kernel_sync_contract() -> None:
    source = PREPARE_PATH.read_text(encoding="utf-8")
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

    recheck_tag = module.ACCURACY_LT_RECHECK_TAG
    expected_recheck_ids = set(module.ACCURACY_LT_RECHECK_CASE_IDS)
    actual_recheck_ids = {
        int(spec["case_id"])
        for spec in specs
        if recheck_tag in str(spec.get("tags", "")).split(",")
    }
    if actual_recheck_ids != expected_recheck_ids:
        raise ValueError(
            "accuracy_lt recheck IDs drifted: "
            f"expected {sorted(expected_recheck_ids)}, got {sorted(actual_recheck_ids)}"
        )
    documented_recheck_ids = "、".join(
        f"`{case_id}`" for case_id in sorted(expected_recheck_ids)
    )
    readme = (OP_DIR / "README.md").read_text(encoding="utf-8")
    if f"case ID {documented_recheck_ids}" not in readme:
        raise ValueError("README accuracy_lt recheck IDs drifted from the generator")

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
    base_specs = [spec for spec in specs if "source_accuracy_case_id" not in spec]
    if [int(spec["case_id"]) for spec in base_specs] != list(range(4)):
        raise ValueError("the four original MSS rows must retain IDs 0--3")
    if {
        (int(spec["tiling_key"]), bool(spec["initial_state"]))
        for spec in base_specs
    } != {(key, boundary) for key in (1, 2) for boundary in (False, True)}:
        raise ValueError("MSS must contain ordinary and boundary rows for each key")

    expected_sources = tuple(module.MSS_UNSAFE_SOURCE_CASES)
    cloned_specs = [spec for spec in specs if "source_accuracy_case_id" in spec]
    actual_sources = {
        (int(spec["source_accuracy_case_id"]), str(spec.get("a5_launch_mode", "")))
        for spec in cloned_specs
    }
    if len(cloned_specs) != len(expected_sources) or actual_sources != set(expected_sources):
        raise ValueError(
            f"MSS unsafe launch coverage drifted: expected {expected_sources}, "
            f"got {sorted(actual_sources)}"
        )

    accuracy_by_id = {int(spec["case_id"]): spec for spec in accuracy_specs}
    identity_fields = {
        "case_id", "case_key", "design_id", "profile", "tags", "manifest",
        "source_accuracy_case_id", "source_accuracy_case_key", "a5_launch_mode",
    }
    for local_case_id, (source_case_id, expected_mode) in enumerate(
        expected_sources, start=4
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
    _check_yaml_input_contract(manifests)
    _check_accuracy_coverage(accuracy_specs, module)
    _check_mss_coverage(mss_specs, accuracy_specs, module)
    if {int(item["tiling_key"]) for item in perf_specs} != {1, 2}:
        raise ValueError("performance must exercise both tiling keys")
    print(
        f"chunk_kda_fwd manifests valid: accuracy=200, "
        f"mss={module.MSS_COUNT}, perf=2, keys=[1, 2]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
