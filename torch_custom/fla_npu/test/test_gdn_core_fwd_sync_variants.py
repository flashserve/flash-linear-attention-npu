from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL_ROOT = (
    REPO_ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_core_fwd/op_kernel"
)
CORE_KERNEL = KERNEL_ROOT / "chunk_gdn_core_fwd.cpp"
CORE_STRUCT = KERNEL_ROOT / "chunk_gdn_core_fwd_struct.h"
HOST_TILING = (
    KERNEL_ROOT.parent / "op_host/chunk_gdn_core_fwd_tiling.cpp"
)
STATE_UPDATE = (
    KERNEL_ROOT
    / "internal/state_update_output/chunk_gdn_core_state_update_output.cpp"
)
COEFFICIENT = (
    KERNEL_ROOT
    / "internal/coefficient_generation/chunk_gdn_core_coefficient_generation.cpp"
)
FWD_H_ARCH35 = (
    KERNEL_ROOT
    / "internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
FWD_H_UPDATE_ARCH35 = (
    KERNEL_ROOT
    / "internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
)
SOLVE_TRI_64 = (
    KERNEL_ROOT
    / "internal/coefficient_generation/gdn_core_solve_tri/arch35/solve_tri_ascend950_64.h"
)


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class GdnCoreFwdSyncVariantSourceTests(unittest.TestCase):
    def test_fwd_h_variants_narrow_two_bounded_mmad_barriers_independently(self):
        text = source(FWD_H_ARCH35)

        self.assertIn("bool kNarrowCube1ToPipeFix = false", text)
        self.assertIn("bool kNarrowCube2ToPipeFix = false", text)
        self.assertIn("bool kCube1EventOnly = false", text)
        self.assertNotIn("kCube2EventOnly", text)
        self.assertIn("bool kUpdateBarrierToPipeMte3 = false", text)
        self.assertIn("bool kUpdateBarrierEventOnly = false", text)
        self.assertIn("bool kBypassHInitCollective = false", text)
        self.assertIn("bool kEntryLocalPipeDrain = false", text)
        self.assertIn("bool kEntryRolePipeDrain = false", text)
        self.assertEqual(text.count("if constexpr (kNarrowCube1ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (kNarrowCube2ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (!kCube1EventOnly)"), 1)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_ALL>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_FIX>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_MTE3>();"), 1)
        self.assertEqual(text.count("AscendC::SyncAll<false>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_V>();"), 10)

        bounded_patterns = (
            r"blockMmadWH\.finalWaitFlags\(\);\s*"
            r"}\s*(?://[^\n]*\n\s*)*"
            r"if constexpr \(!kCube1EventOnly\) \{\s*"
            r"if constexpr \(kNarrowCube1ToPipeFix\) \{\s*"
            r"AscendC::PipeBarrier<PIPE_FIX>\(\);\s*}\s*else\s*\{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);\s*}\s*}\s*"
            r"Arch::CrossCoreSetFlag<0x2, PIPE_FIX>",
            r"blockMmadKV\.finalWaitFlags\(\);\s*"
            r"}\s*(?://[^\n]*\n\s*)*"
            r"if constexpr \(kNarrowCube2ToPipeFix\) \{\s*"
            r"AscendC::PipeBarrier<PIPE_FIX>\(\);\s*}\s*else\s*\{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);",
        )
        for pattern in bounded_patterns:
            self.assertRegex(text, pattern)

    def test_fwd_h_entry_drain_and_init_bypass_are_mutually_exclusive(self):
        text = source(FWD_H_ARCH35)

        self.assertIn(
            "static_assert(!(kBypassHInitCollective &&\n"
            "                    (kEntryLocalPipeDrain || kEntryRolePipeDrain))",
            text,
        )
        self.assertIn(
            "static_assert(!(kEntryLocalPipeDrain && kEntryRolePipeDrain)", text
        )
        self.assertRegex(
            text,
            r"(?s)__aicore__ inline void Process\(\) \{.*?"
            r"if constexpr \(kEntryRolePipeDrain\) \{.*?"
            r"if ASCEND_IS_AIC \{\s*AscendC::PipeBarrier<PIPE_FIX>\(\);\s*\}"
            r".*?if ASCEND_IS_AIV \{\s*AscendC::PipeBarrier<PIPE_MTE3>\(\);\s*\}"
            r"\s*\} else if constexpr \(kEntryLocalPipeDrain\) \{.*?"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);\s*"
            r"\} else \{\s*AscendC::SyncAll<false>\(\);\s*\}",
        )
        self.assertEqual(
            text.count("if constexpr (!kBypassHInitCollective)"), 2
        )
        self.assertRegex(
            text,
            r"(?s)if ASCEND_IS_AIC \{.*?"
            r"if constexpr \(!kBypassHInitCollective\) \{\s*"
            r"AscendC::SyncAll<false>\(\);\s*\}\s*"
            r"uint32_t currStage = 0;",
        )
        self.assertRegex(
            text,
            r"(?s)WaitFlag<AscendC::HardEvent::MTE3_MTE2>\(EVENT_ID0\);\s*"
            r"AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>\(EVENT_ID1\);\s*"
            r"if constexpr \(!kBypassHInitCollective\) \{\s*"
            r"AscendC::SyncAll<false>\(\);\s*\}.*?"
            r"CrossCoreSetFlag<0x2, PIPE_MTE3>\(vecBlockScheduler.vec2Done\[0\]\);\s*"
            r"Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>\(vecBlockScheduler.vec2Done\[1\]\);",
        )
        self.assertIn(
            "Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);",
            text,
        )
        self.assertIn(
            "Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);", text
        )
        self.assertIn(
            "Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);", text
        )

    def test_update_barrier_variants_keep_the_following_dependency_events(self):
        text = source(FWD_H_UPDATE_ARCH35)

        self.assertIn(
            "KGatedTag::updateBarrierToPipeMte3", text
        )
        self.assertIn(
            "KGatedTag::updateBarrierEventOnly", text
        )
        self.assertRegex(
            text,
            r"CopyUbToGm\(finalStateThisTile, hUpdateUbTensorThisTile,\s*"
            r"rowsThisTile, nActual, outputStride\);\s*"
            r"if constexpr \(!kUpdateBarrierEventOnly\) \{\s*"
            r"if constexpr \(kUpdateBarrierToPipeMte3\) \{\s*"
            r"AscendC::PipeBarrier<PIPE_MTE3>\(\);\s*}\s*else\s*\{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);\s*}\s*}\s*"
            r"AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>\(updateReadyEvent\);\s*"
            r"AscendC::SetFlag<AscendC::HardEvent::MTE3_V>\(updateReadyEvent\);\s*"
            r"AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>\(updateReadyEvent\);",
        )

    def test_fwd_h_wrapper_maps_c1_c2_and_combined_variants_only_on_a5(self):
        text = source(STATE_UPDATE)

        self.assertNotIn("kCube2EventOnly", text)
        self.assertIn(
            "GdnCoreSyncVariant kSyncVariant = GdnCoreSyncVariant::B0", text
        )
        self.assertEqual(text.count("RunFwdH<"), 16)
        self.assertEqual(text.count(", kSyncVariant>("), 16)
        expected_true_sets = {
            "kNarrowCube1ToPipeFix": {
                "B1", "B4", "B5", "B9", "B13", "B15",
                "B16", "B17", "B18", "B20"
            },
            "kNarrowCube2ToPipeFix": {"B2", "B4", "B5", "B17", "B18"},
            "kCube1EventOnly": {"B8", "B10", "B11", "B19", "B21"},
            "kUpdateBarrierToPipeMte3": {"B6", "B9", "B10"},
            "kUpdateBarrierEventOnly": {
                "B7", "B11", "B12", "B13", "B14", "B15",
                "B16", "B17", "B18", "B19", "B20", "B21"
            },
            "kBypassHInitCollective": {"B12", "B13"},
            "kEntryLocalPipeDrain": {"B14", "B15", "B17"},
            "kEntryRolePipeDrain": {"B16", "B18", "B19", "B20", "B21"},
        }
        observed_true_sets = {}
        for flag in expected_true_sets:
            assignment = re.search(
                rf"constexpr bool {flag}\s*=\s*(.*?);", text, re.DOTALL
            )
            self.assertIsNotNone(assignment, flag)
            observed_true_sets[flag] = set(
                re.findall(r"GdnCoreSyncVariant::(B\d+)", assignment.group(1))
            )
        self.assertEqual(observed_true_sets, expected_true_sets)

        expected_matrix = {
            "B0": (False, False, False, False, False, False, False, False),
            "B1": (True, False, False, False, False, False, False, False),
            "B2": (False, True, False, False, False, False, False, False),
            "B3": (False, False, False, False, False, False, False, False),
            "B4": (True, True, False, False, False, False, False, False),
            "B5": (True, True, False, False, False, False, False, False),
            "B6": (False, False, False, True, False, False, False, False),
            "B7": (False, False, False, False, True, False, False, False),
            "B8": (False, False, True, False, False, False, False, False),
            "B9": (True, False, False, True, False, False, False, False),
            "B10": (False, False, True, True, False, False, False, False),
            "B11": (False, False, True, False, True, False, False, False),
            "B12": (False, False, False, False, True, True, False, False),
            "B13": (True, False, False, False, True, True, False, False),
            "B14": (False, False, False, False, True, False, True, False),
            "B15": (True, False, False, False, True, False, True, False),
            "B16": (True, False, False, False, True, False, False, True),
            "B17": (True, True, False, False, True, False, True, False),
            "B18": (True, True, False, False, True, False, False, True),
            "B19": (False, False, True, False, True, False, False, True),
            "B20": (True, False, False, False, True, False, False, True),
            "B21": (False, False, True, False, True, False, False, True),
        }
        observed_matrix = {
            variant: tuple(variant in observed_true_sets[flag]
                           for flag in expected_true_sets)
            for variant in expected_matrix
        }
        self.assertEqual(observed_matrix, expected_matrix)
        self.assertIn(
            "kNarrowCube1ToPipeFix, kNarrowCube2ToPipeFix, kCube1EventOnly,\n"
            "        kUpdateBarrierToPipeMte3, kUpdateBarrierEventOnly,\n"
            "        kBypassHInitCollective, kEntryLocalPipeDrain, "
            "kEntryRolePipeDrain>;", text
        )
        wrapper_start = text.index("__aicore__ inline void RunFwdH(")
        a5_start = text.index(
            "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", wrapper_start
        )
        non_a5_start = text.index("#else", a5_start)
        alias_end = text.index("#endif", non_a5_start)
        non_a5_alias = text[non_a5_start:alias_end]
        self.assertNotIn("kNarrowCube", non_a5_alias)
        self.assertNotIn("kCube1EventOnly", non_a5_alias)
        self.assertNotIn("kUpdateBarrier", non_a5_alias)
        self.assertNotIn("kBypassHInitCollective", non_a5_alias)
        self.assertNotIn("kEntryLocalPipeDrain", non_a5_alias)
        self.assertNotIn("kEntryRolePipeDrain", non_a5_alias)

    def test_solve_wrapper_limits_immediate_event_to_a5_bt64_b3_b5_b20_b21(self):
        text = source(COEFFICIENT)

        self.assertIn(
            "GDN::GdnCoreSyncVariant kSyncVariant,",
            text,
        )
        self.assertRegex(
            text,
            r"kUseMte2Mte1Event\s*=\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B3 \|\|\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B5 \|\|\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B20 \|\|\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B21;",
        )
        self.assertIn(
            "SolveTri64<T, T, kUseMte2Mte1Event, false> solve;",
            text,
        )
        self.assertNotIn("kDeferMte2Mte1Wait =", text)
        self.assertIn("SolveTri128<T, T> solve;", text)

        a5_start = text.index("#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310")
        non_a5_start = text.index("#else", a5_start)
        non_a5_end = text.index("#endif", non_a5_start)
        self.assertNotIn("kSyncVariant", text[non_a5_start:non_a5_end])

    def test_solve_tri64_immediate_event_order_preserves_helper_edge(self):
        text = source(SOLVE_TRI_64)

        self.assertIn(
            "bool kUseMte2Mte1Event = false,\n"
            "          bool kDeferMte2Mte1Wait = false",
            text,
        )
        self.assertIn(
            "static_assert(!kDeferMte2Mte1Wait || kUseMte2Mte1Event,",
            text,
        )
        helper_start = text.index("__aicore__ inline void FixpipeL0cToL1(")
        helper_end = text.index("__aicore__ inline void FixpipeZeroToL1(", helper_start)
        helper = text[helper_start:helper_end]
        self.assertEqual(helper.count("HardEvent::FIX_MTE2"), 2)

        level_start = text.index("__aicore__ inline void MbhLevelAic(")
        level_end = text.index("__aicore__ inline void Process()", level_start)
        level = text[level_start:level_end]
        fix_copy = level.index("FixpipeL0cToL1(l1_Y")
        event_set = level.index("SetFlag<AscendC::HardEvent::MTE2_MTE1>(0)")
        immediate_wait = level.index(
            "WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0)", event_set
        )
        independent_mmad = level.index("MbhMatmulToL0C(l1_I, l1_X", immediate_wait)
        deferred_wait = level.index(
            "WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0)", immediate_wait + 1
        )
        dependent_mmad = level.index("MbhMatmulToL0C(l1_Y, l1_INPUT", deferred_wait)
        self.assertLess(
            fix_copy,
            event_set,
        )
        self.assertLess(event_set, immediate_wait)
        self.assertLess(immediate_wait, independent_mmad)
        self.assertLess(independent_mmad, deferred_wait)
        self.assertLess(deferred_wait, dependent_mmad)
        self.assertIn("if constexpr (!kDeferMte2Mte1Wait)", level)
        self.assertIn(
            "if constexpr (kUseMte2Mte1Event && kDeferMte2Mte1Wait)",
            level,
        )
        self.assertIn("AscendC::PipeBarrier<PIPE_ALL>();", level)
        self.assertNotIn("HardEvent::FIX_MTE1", level)

    def test_top_level_dispatches_all_variants_at_compile_time(self):
        text = source(CORE_KERNEL)

        self.assertEqual(text.count("GDN::GdnCoreSyncVariant::B0>"), 2)
        for variant in tuple(f"B{index}" for index in range(1, 22)):
            self.assertEqual(text.count(f"GDN::GdnCoreSyncVariant::{variant}>"), 1)
        self.assertIn(
            "RunSolvePhase<InputT, 64, kSyncVariant>", text
        )
        self.assertIn(
            "RunSolvePhase<InputT, 128, kSyncVariant>", text
        )
        self.assertIn("DispatchFwdH<TileShapes, kSyncVariant>", text)

    def test_host_variant_router_is_strict_fail_closed_and_abi_neutral(self):
        host = source(HOST_TILING)
        kernel = source(CORE_KERNEL)
        struct = source(CORE_STRUCT)

        resolver_start = host.index("bool ResolveSyncVariant(")
        resolver_end = host.index("bool ResolveT1Diagnostic(", resolver_start)
        resolver = host[resolver_start:resolver_end]
        accepted_values = set(
            re.findall(r'std::strcmp\(value, "([^"]+)"\)', resolver)
        )
        self.assertEqual(
            accepted_values, {f"B{index}" for index in range(22)}
        )
        self.assertIn("value == nullptr", resolver)
        self.assertEqual(resolver.count("return true;"), 22)
        self.assertEqual(resolver.count("return false;"), 1)
        for selector in tuple(f"B{index}" for index in range(22)):
            prefix = r"value == nullptr \|\| " if selector == "B0" else ""
            self.assertRegex(
                resolver,
                rf"if \({prefix}std::strcmp\(value, \"{selector}\"\) == 0\) \{{\s*"
                rf"variant = GDN::GdnCoreSyncVariant::{selector};\s*return true;",
            )

        diagnostic_start = resolver_end
        diagnostic_end = host.index("bool ResolveT65Diagnostic(", diagnostic_start)
        diagnostic = host[diagnostic_start:diagnostic_end]
        self.assertIn('std::getenv("FLA_NPU_GDN_SYNC_T1_DIAGNOSTIC")', diagnostic)
        self.assertEqual(
            set(re.findall(r'std::strcmp\(value, "([^"]+)"\)', diagnostic)),
            {"0", "1"},
        )
        self.assertEqual(diagnostic.count("return true;"), 2)
        self.assertEqual(diagnostic.count("return false;"), 1)
        self.assertIn(
            "FLA_NPU_GDN_SYNC_T1_DIAGNOSTIC must be unset, 0, or 1.", host
        )

        t65_diagnostic_start = diagnostic_end
        t65_diagnostic_end = host.index(
            "bool IsMainShapeOnlySyncVariant(", t65_diagnostic_start
        )
        t65_diagnostic = host[t65_diagnostic_start:t65_diagnostic_end]
        self.assertIn(
            'std::getenv("FLA_NPU_GDN_SYNC_T65_DIAGNOSTIC")', t65_diagnostic
        )
        self.assertEqual(
            set(re.findall(r'std::strcmp\(value, "([^"]+)"\)', t65_diagnostic)),
            {"0", "1"},
        )
        self.assertEqual(t65_diagnostic.count("return true;"), 2)
        self.assertEqual(t65_diagnostic.count("return false;"), 1)
        self.assertIn(
            "FLA_NPU_GDN_SYNC_T65_DIAGNOSTIC must be unset, 0, or 1.", host
        )

        expected_keys = {
            "TILING_KEY_B0_V128": "1",
            "TILING_KEY_B0_V256": "2",
            "TILING_KEY_B1_V128": "11",
            "TILING_KEY_B2_V128": "21",
            "TILING_KEY_B3_V128": "31",
            "TILING_KEY_B4_V128": "41",
            "TILING_KEY_B5_V128": "51",
            "TILING_KEY_B6_V128": "61",
            "TILING_KEY_B7_V128": "71",
            "TILING_KEY_B8_V128": "81",
            "TILING_KEY_B9_V128": "91",
            "TILING_KEY_B10_V128": "101",
            "TILING_KEY_B11_V128": "111",
            "TILING_KEY_B12_V128": "121",
            "TILING_KEY_B13_V128": "131",
            "TILING_KEY_B14_V128": "141",
            "TILING_KEY_B15_V128": "151",
            "TILING_KEY_B16_V128": "161",
            "TILING_KEY_B17_V128": "171",
            "TILING_KEY_B18_V128": "181",
            "TILING_KEY_B19_V128": "191",
            "TILING_KEY_B20_V128": "201",
            "TILING_KEY_B21_V128": "211",
        }
        for name, value in expected_keys.items():
            self.assertIn(f"constexpr uint32_t {name} = {value};", host)
        self.assertRegex(
            host,
            r"const uint32_t b0Key\s*=\s*vDim == SUPPORTED_V_DIM_256 \? "
            r"TILING_KEY_B0_V256 : TILING_KEY_B0_V128;",
        )
        self.assertRegex(
            host,
            r"if \(vDim != SUPPORTED_V_DIM_128\) \{\s*return b0Key;\s*\}",
        )
        for variant, key in {
            "B0": "b0Key",
            "B1": "TILING_KEY_B1_V128",
            "B2": "TILING_KEY_B2_V128",
            "B3": "TILING_KEY_B3_V128",
            "B4": "TILING_KEY_B4_V128",
            "B5": "TILING_KEY_B5_V128",
            "B6": "TILING_KEY_B6_V128",
            "B7": "TILING_KEY_B7_V128",
            "B8": "TILING_KEY_B8_V128",
            "B9": "TILING_KEY_B9_V128",
            "B10": "TILING_KEY_B10_V128",
            "B11": "TILING_KEY_B11_V128",
            "B12": "TILING_KEY_B12_V128",
            "B13": "TILING_KEY_B13_V128",
            "B14": "TILING_KEY_B14_V128",
            "B15": "TILING_KEY_B15_V128",
            "B16": "TILING_KEY_B16_V128",
            "B17": "TILING_KEY_B17_V128",
            "B18": "TILING_KEY_B18_V128",
            "B19": "TILING_KEY_B19_V128",
            "B20": "TILING_KEY_B20_V128",
            "B21": "TILING_KEY_B21_V128",
        }.items():
            self.assertRegex(
                host,
                rf"case GDN::GdnCoreSyncVariant::{variant}:\s*"
                rf"return {key};",
            )
        self.assertIn(
            "context->SetTilingKey(ResolveTilingKey(vDim, effectiveSyncVariant));",
            host,
        )
        self.assertRegex(
            host,
            r"const bool isExperimentalShape\s*=\s*isBf16 && "
            r"initialStateDesc != nullptr &&\s*"
            r"initialStateDesc->GetDataType\(\) == ge::DT_FLOAT &&\s*"
            r"vDim == SUPPORTED_V_DIM_128 && \*chunkSize == CHUNK_64;",
        )
        self.assertIn(
            "context->GetOptionalInputDesc(INPUT_INITIAL_STATE)", host
        )
        main_only_start = host.index("bool IsMainShapeOnlySyncVariant(")
        main_only_end = host.index("bool IsT1DiagnosticSyncVariant(", main_only_start)
        main_only = host[main_only_start:main_only_end]
        self.assertEqual(
            set(re.findall(r"GdnCoreSyncVariant::(B\d+)", main_only)),
            {
                "B12", "B13", "B14", "B15", "B16",
                "B17", "B18", "B19", "B20", "B21",
            },
        )
        t1_variant_start = main_only_end
        t1_variant_end = host.index(
            "bool IsT65DiagnosticSyncVariant(", t1_variant_start
        )
        t1_variants = host[t1_variant_start:t1_variant_end]
        self.assertEqual(
            set(re.findall(r"GdnCoreSyncVariant::(B\d+)", t1_variants)),
            {"B16", "B17", "B18", "B19"},
        )
        t65_variant_start = t1_variant_end
        t65_variant_end = host.index(
            "uint32_t ResolveTilingKey(", t65_variant_start
        )
        t65_variants = host[t65_variant_start:t65_variant_end]
        self.assertEqual(
            set(re.findall(r"GdnCoreSyncVariant::(B\d+)", t65_variants)),
            {"B20", "B21"},
        )
        exact_start = host.index("const bool isExactMainVarlenShape")
        exact_end = host.index(
            "GDN::GdnCoreSyncVariant effectiveSyncVariant", exact_start
        )
        exact_selector = host[exact_start:exact_end]
        for required in (
            "isAscend950",
            "isExperimentalShape",
            "isVarlen",
            "batch == MAIN_MODEL_BATCH",
            "heads == MAIN_MODEL_K_HEADS",
            "valueHeads == MAIN_MODEL_V_HEADS",
            "tokens == MAIN_MODEL_TOKENS",
            "kDim == SUPPORTED_K_DIM",
            "IsShape(cuShape, {2})",
            "varlenChunks == MAIN_MODEL_CHUNKS",
            "*outputFinalState",
        ):
            self.assertIn(required, exact_selector)
        self.assertNotIn("outputMask", exact_selector)
        t1_selector_start = host.index("const bool isExactT1DiagnosticVarlenShape")
        t1_selector_end = host.index(
            "const bool selectT1DiagnosticVariant", t1_selector_start
        )
        t1_selector = host[t1_selector_start:t1_selector_end]
        for required in (
            "isAscend950",
            "isExperimentalShape",
            "isVarlen",
            "batch == MAIN_MODEL_BATCH",
            "heads == MAIN_MODEL_K_HEADS",
            "valueHeads == MAIN_MODEL_V_HEADS",
            "tokens == T1_DIAGNOSTIC_TOKENS",
            "kDim == SUPPORTED_K_DIM",
            "IsShape(cuShape, {2})",
            "varlenChunks == T1_DIAGNOSTIC_CHUNKS",
            "*outputFinalState",
        ):
            self.assertIn(required, t1_selector)
        self.assertNotIn("outputMask", t1_selector)
        t65_selector_start = host.index("const bool isExactT65DiagnosticVarlenShape")
        t65_selector_end = host.index(
            "const bool selectT65DiagnosticVariant", t65_selector_start
        )
        t65_selector = host[t65_selector_start:t65_selector_end]
        for required in (
            "isAscend950",
            "isExperimentalShape",
            "isVarlen",
            "batch == MAIN_MODEL_BATCH",
            "heads == MAIN_MODEL_K_HEADS",
            "valueHeads == MAIN_MODEL_V_HEADS",
            "tokens == T65_DIAGNOSTIC_TOKENS",
            "kDim == SUPPORTED_K_DIM",
            "IsShape(cuShape, {2})",
            "varlenChunks == T65_DIAGNOSTIC_CHUNKS",
            "*outputFinalState",
        ):
            self.assertIn(required, t65_selector)
        self.assertNotIn("outputMask", t65_selector)
        for required in (
            "t1Diagnostic",
            "IsT1DiagnosticSyncVariant(syncVariant)",
            "isExactT1DiagnosticVarlenShape",
            "tokens == T1_DIAGNOSTIC_TOKENS",
            "varlenChunks == T1_DIAGNOSTIC_CHUNKS",
            "t65Diagnostic",
            "IsT65DiagnosticSyncVariant(syncVariant)",
            "isExactT65DiagnosticVarlenShape",
            "tokens == T65_DIAGNOSTIC_TOKENS",
            "varlenChunks == T65_DIAGNOSTIC_CHUNKS",
        ):
            self.assertIn(required, exact_selector)
        self.assertIn("constexpr int64_t T1_DIAGNOSTIC_TOKENS = 1;", host)
        self.assertIn("constexpr uint64_t T1_DIAGNOSTIC_CHUNKS = 1;", host)
        self.assertIn("constexpr int64_t T65_DIAGNOSTIC_TOKENS = 65;", host)
        self.assertIn("constexpr uint64_t T65_DIAGNOSTIC_CHUNKS = 2;", host)
        effective_start = exact_end
        effective_end = host.index(
            "OP_CHECK_IF(Tiling4ChunkGdnCoreStateOutput", effective_start
        )
        effective_selector = host[effective_start:effective_end]
        self.assertIn(
            "effectiveSyncVariant = GDN::GdnCoreSyncVariant::B0",
            effective_selector,
        )
        self.assertIn("if (isExperimentalShape)", effective_selector)
        self.assertRegex(
            effective_selector,
            r"IsMainShapeOnlySyncVariant\(syncVariant\) &&\s*"
            r"!isExactMainVarlenShape && !selectT1DiagnosticVariant &&\s*"
            r"!selectT65DiagnosticVariant\s*\? "
            r"GDN::GdnCoreSyncVariant::B7\s*:\s*syncVariant;",
        )

        # Selector truth table: legacy variants retain their established
        # experimental domain; B12-B21 narrow to the exact main varlen shape.
        # B16-B19 have the exact-T=1 diagnostic exception, while B20/B21 have
        # an independent exact-T=65/two-chunk route that executes SolveTri64.
        def selected(
            requested: str,
            experimental: bool,
            exact_main: bool,
            t1_diagnostic_enabled: bool = False,
            exact_t1: bool = False,
            t65_diagnostic_enabled: bool = False,
            exact_t65: bool = False,
        ) -> str:
            if not experimental:
                return "B0"
            select_t1 = (
                requested in {"B16", "B17", "B18", "B19"}
                and t1_diagnostic_enabled
                and exact_t1
            )
            select_t65 = (
                requested in {"B20", "B21"}
                and t65_diagnostic_enabled
                and exact_t65
            )
            if (
                requested in {
                    "B12", "B13", "B14", "B15", "B16",
                    "B17", "B18", "B19", "B20", "B21",
                }
                and not exact_main
                and not select_t1
                and not select_t65
            ):
                return "B7"
            return requested

        for requested in (f"B{index}" for index in range(12)):
            self.assertEqual(selected(requested, True, False), requested)
            self.assertEqual(selected(requested, True, True), requested)
            self.assertEqual(selected(requested, False, False), "B0")
        for requested in ("B12", "B13", "B14", "B15"):
            self.assertEqual(selected(requested, True, True), requested)
            self.assertEqual(selected(requested, True, False), "B7")
            self.assertEqual(
                selected(requested, True, False, True, True), "B7"
            )
            self.assertEqual(selected(requested, False, False), "B0")
        for requested in ("B16", "B17", "B18", "B19"):
            self.assertEqual(selected(requested, True, True), requested)
            self.assertEqual(selected(requested, True, False), "B7")
            self.assertEqual(
                selected(requested, True, False, True, True), requested
            )
            self.assertEqual(
                selected(requested, True, False, False, True), "B7"
            )
            self.assertEqual(
                selected(requested, True, False, True, False), "B7"
            )
            self.assertEqual(selected(requested, False, False, True, True), "B0")
        for requested in ("B20", "B21"):
            self.assertEqual(selected(requested, True, True), requested)
            self.assertEqual(selected(requested, True, False), "B7")
            self.assertEqual(
                selected(requested, True, False, False, False, True, True),
                requested,
            )
            self.assertEqual(
                selected(requested, True, False, False, False, False, True),
                "B7",
            )
            self.assertEqual(
                selected(requested, True, False, False, False, True, False),
                "B7",
            )
            self.assertEqual(
                selected(requested, True, False, True, True), "B7"
            )
            self.assertEqual(
                selected(requested, False, False, False, False, True, True),
                "B0",
            )

        routed = {
            int(key): (shape, variant)
            for key, shape, variant in re.findall(
                r"(?:if|else if) \(TILING_KEY_IS\((\d+)\)\) \{\s*"
                r"KERNEL_TASK_TYPE\(\1, KERNEL_TYPE_MIX_AIC_1_2\);\s*"
                r"GDN::DispatchPhase6ByDtype<"
                r"Catlass::Gemm::Kernel::(GDNFwdHTileShapes(?:128|256)),\s*"
                r"GDN::GdnCoreSyncVariant::(B\d+)>",
                kernel,
            )
        }
        self.assertEqual(
            routed,
            {
                1: ("GDNFwdHTileShapes128", "B0"),
                2: ("GDNFwdHTileShapes256", "B0"),
                11: ("GDNFwdHTileShapes128", "B1"),
                21: ("GDNFwdHTileShapes128", "B2"),
                31: ("GDNFwdHTileShapes128", "B3"),
                41: ("GDNFwdHTileShapes128", "B4"),
                51: ("GDNFwdHTileShapes128", "B5"),
                61: ("GDNFwdHTileShapes128", "B6"),
                71: ("GDNFwdHTileShapes128", "B7"),
                81: ("GDNFwdHTileShapes128", "B8"),
                91: ("GDNFwdHTileShapes128", "B9"),
                101: ("GDNFwdHTileShapes128", "B10"),
                111: ("GDNFwdHTileShapes128", "B11"),
                121: ("GDNFwdHTileShapes128", "B12"),
                131: ("GDNFwdHTileShapes128", "B13"),
                141: ("GDNFwdHTileShapes128", "B14"),
                151: ("GDNFwdHTileShapes128", "B15"),
                161: ("GDNFwdHTileShapes128", "B16"),
                171: ("GDNFwdHTileShapes128", "B17"),
                181: ("GDNFwdHTileShapes128", "B18"),
                191: ("GDNFwdHTileShapes128", "B19"),
                201: ("GDNFwdHTileShapes128", "B20"),
                211: ("GDNFwdHTileShapes128", "B21"),
            },
        )
        a5_dispatch_start = kernel.index(
            "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 &&",
            kernel.index("TILING_KEY_IS(2)"),
        )
        a5_dispatch_end = kernel.index("#endif", a5_dispatch_start)
        a5_dispatch = kernel[a5_dispatch_start:a5_dispatch_end]
        self.assertIn("ORIG_DTYPE_Q == DT_BF16", a5_dispatch)
        self.assertIn("ORIG_DTYPE_INITIAL_STATE == DT_FLOAT", a5_dispatch)
        for key in (
            11, 21, 31, 41, 51, 61, 71, 81,
            91, 101, 111, 121, 131, 141, 151, 161, 171, 181,
            191, 201, 211,
        ):
            self.assertIn(f"TILING_KEY_IS({key})", a5_dispatch)
        for key in (
            12, 22, 32, 42, 52, 62, 72, 82,
            92, 102, 112, 122, 132, 142, 152, 162, 172, 182,
            192, 202, 212,
        ):
            self.assertNotIn(f"TILING_KEY_IS({key})", kernel)

        self.assertIn(
            "platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950",
            host,
        )
        self.assertIn(
            "syncVariant != GDN::GdnCoreSyncVariant::B0 && !isAscend950",
            host,
        )
        self.assertRegex(
            host,
            r"OP_CHECK_IF\(syncVariant != GDN::GdnCoreSyncVariant::B0 && "
            r"!isAscend950,\s*OP_LOGE\(context->GetNodeName\(\),\s*"
            r'"FLA_NPU_GDN_SYNC_VARIANT B1/B2/B3/B4/B5/B6/B7/B8/B9/B10/B11/B12/B13/B14/B15/B16/B17/B18/B19/B20/B21 is supported only on Ascend950\."\),\s*'
            r"return ge::GRAPH_FAILED\);",
        )

        enum_start = struct.index("enum class GdnCoreSyncVariant")
        enum_end = struct.index("};", enum_start)
        enum_values = {
            name: int(value)
            for name, value in re.findall(
                r"\b(B\d+)\s*=\s*(\d+)", struct[enum_start:enum_end]
            )
        }
        self.assertEqual(
            enum_values, {f"B{index}": index for index in range(22)}
        )

        trailer_start = struct.index("struct ChunkGdnCoreFwdTrailer {")
        trailer_end = struct.index("};", trailer_start)
        trailer = struct[trailer_start:trailer_end]
        self.assertNotIn("GdnCoreSyncVariant", trailer)
        self.assertNotIn("syncVariant", trailer)
        self.assertNotIn("trailer.syncVariant", host)


if __name__ == "__main__":
    unittest.main()
