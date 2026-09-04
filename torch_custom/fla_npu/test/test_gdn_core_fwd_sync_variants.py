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
KKT_EPILOGUE = (
    KERNEL_ROOT
    / "internal/coefficient_generation/chunk_gdn_core_cumsum_kkt.h"
)
FWD_H_ARCH35 = (
    KERNEL_ROOT
    / "internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
FWD_H_UPDATE_ARCH35 = (
    KERNEL_ROOT
    / "internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
)
FWD_O_KERNEL = (
    KERNEL_ROOT
    / "internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
)
SOLVE_TRI_64 = (
    KERNEL_ROOT
    / "internal/coefficient_generation/gdn_core_solve_tri/arch35/solve_tri_ascend950_64.h"
)
RECOMPUTE_W_U_KERNEL = (
    KERNEL_ROOT / "internal/operators/recompute_w_u_fwd/op_kernel"
)
RECOMPUTE_W_U_COMMON = RECOMPUTE_W_U_KERNEL / "recompute_w_u_fwd_common.h"
RECOMPUTE_W_U_CUBE = RECOMPUTE_W_U_KERNEL / "recompute_w_u_fwd_cube.h"
RECOMPUTE_W_U_VECTOR = RECOMPUTE_W_U_KERNEL / "recompute_w_u_fwd_vector.h"


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
        self.assertIn("bool kFwdHVarlenDenseC1FullTiles = false", text)
        self.assertIn("bool kFwdHVarlenDenseC2FullTiles = false", text)
        self.assertEqual(text.count("if constexpr (kNarrowCube1ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (kNarrowCube2ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (!kCube1EventOnly)"), 1)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_ALL>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_FIX>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_MTE3>();"), 1)
        self.assertEqual(text.count("AscendC::SyncAll<false>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_V>();"), 10)

        bounded_protocols = (
            (
                "blockMmadWH.finalWaitFlags();",
                "blockMmadWHTail.finalWaitFlags();",
                "if constexpr (!kCube1EventOnly)",
                "if constexpr (kNarrowCube1ToPipeFix)",
                "cubeBlockScheduler.cube1Done[streamId]",
            ),
            (
                "blockMmadKV.finalWaitFlags();",
                "blockMmadKVTail.finalWaitFlags();",
                None,
                "if constexpr (kNarrowCube2ToPipeFix)",
                "cubeBlockScheduler.cube2Done[streamId]",
            ),
        )
        search_pos = 0
        for final_wait, tail_wait, event_guard, barrier_guard, done_flag in bounded_protocols:
            dense_branch_pos = text.index("if (useDenseFullTile)", search_pos)
            dense_guard_pos = text.index("if (!useDenseFullTile)", dense_branch_pos)
            publish_pos = text.index(done_flag, dense_guard_pos)
            protocol = text[dense_branch_pos:publish_pos]
            self.assertIn(final_wait, protocol)
            self.assertIn(tail_wait, protocol)
            if event_guard is not None:
                self.assertIn(event_guard, protocol)
            self.assertIn(barrier_guard, protocol)
            self.assertIn("AscendC::PipeBarrier<PIPE_FIX>();", protocol)
            self.assertIn("AscendC::PipeBarrier<PIPE_ALL>();", protocol)
            search_pos = publish_pos

        self.assertIn(
            "kFwdHVarlenDenseC1FullTiles && isVariedLen && "
            "tokenBatch == 1 && chunkSize == 64",
            text,
        )
        self.assertIn(
            "kFwdHVarlenDenseC2FullTiles && isVariedLen && "
            "tokenBatch == 1 && chunkSize == 64",
            text,
        )
        self.assertEqual(text.count("bool denseFullGenerationOpen = false;"), 2)
        self.assertEqual(
            text.count("if (denseFullGenerationOpen && !useDenseFullTile)"), 2
        )
        self.assertEqual(text.count("if (!denseFullGenerationOpen)"), 2)
        self.assertEqual(text.count("if (denseFullGenerationOpen)"), 2)
        self.assertIn(
            "useSingleSeqVarlenDenseC1FullTiles &&\n"
            "                                cube1Offsets.blockTokens == chunkSize",
            text,
        )
        self.assertIn(
            "useSingleSeqVarlenDenseC2FullTiles && needProcessStage2 &&\n"
            "                                cube2Offsets.blockTokens == chunkSize",
            text,
        )

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

    def test_fwd_o_mode2_aggregation_traits_are_independent_and_a5_only(self):
        kernel = source(FWD_O_KERNEL)
        wrapper = source(STATE_UPDATE)
        core = source(CORE_KERNEL)

        self.assertIn("bool kFwdOAggregateQkMaskBarrier = false", kernel)
        self.assertIn("bool kFwdOAggregateOutputBarrier = false", kernel)
        self.assertIn("if constexpr (!kFwdOAggregateQkMaskBarrier)", kernel)
        self.assertIn(
            "(isVariedLen == 0 || kFwdOAggregateOutputBarrier)", kernel
        )
        self.assertIn("if constexpr (!kFwdOAggregateOutputBarrier)", kernel)
        self.assertIn(
            "SyncTraits::kFwdOAggregateQkMaskBarrier,\n"
            "        SyncTraits::kFwdOAggregateOutputBarrier>;",
            wrapper,
        )
        self.assertIn(
            "template <GdnCoreSyncVariant kSyncVariant = "
            "GdnCoreSyncVariant::B0>\n"
            "__aicore__ inline void DispatchFwdO",
            wrapper,
        )
        self.assertIn("DispatchFwdO<kSyncVariant>", core)
        run_start = wrapper.index("__aicore__ inline void RunFwdO(")
        a5_start = wrapper.index(
            "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", run_start
        )
        non_a5_start = wrapper.index("#else", a5_start)
        non_a5_end = wrapper.index("#endif", non_a5_start)
        non_a5_alias = wrapper[non_a5_start:non_a5_end]
        self.assertNotIn("kFwdOAggregate", non_a5_alias)

    def test_fwd_h_wrapper_maps_c1_c2_and_combined_variants_only_on_a5(self):
        text = source(STATE_UPDATE)

        self.assertNotIn("kCube2EventOnly", text)
        self.assertIn(
            "GdnCoreSyncVariant kSyncVariant = GdnCoreSyncVariant::B0", text
        )
        self.assertEqual(text.count("RunFwdH<"), 16)
        self.assertEqual(text.count("RunFwdO<"), 4)
        self.assertEqual(text.count(", kSyncVariant>("), 20)
        expected_true_sets = {
            "kNarrowCube1ToPipeFix": {
                "B1", "B4", "B5", "B9", "B13", "B15",
                "B16", "B17", "B18", "B19", "B20", "B21", "B22",
                "B23", "B24", "B25", "B26", "B27", "B28", "B29",
                "B30", "B31"
            },
            "kNarrowCube2ToPipeFix": {"B2", "B4", "B5", "B17", "B18"},
            "kCube1EventOnly": {"B8", "B10", "B11"},
            "kUpdateBarrierToPipeMte3": {"B6", "B9", "B10"},
            "kUpdateBarrierEventOnly": {
                "B7", "B11", "B12", "B13", "B14", "B15",
                "B16", "B17", "B18", "B19", "B20", "B21", "B22",
                "B23", "B24", "B25", "B26", "B27", "B28", "B29",
                "B30", "B31"
            },
            "kBypassHInitCollective": {"B12", "B13"},
            "kEntryLocalPipeDrain": {"B14", "B15", "B17"},
            "kEntryRolePipeDrain": {
                "B16", "B18", "B19", "B20", "B21", "B22", "B23", "B24",
                "B25", "B26", "B27", "B28", "B29", "B30", "B31"
            },
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
            "B19": (True, False, False, False, True, False, False, True),
            "B20": (True, False, False, False, True, False, False, True),
            "B21": (True, False, False, False, True, False, False, True),
            "B22": (True, False, False, False, True, False, False, True),
            "B23": (True, False, False, False, True, False, False, True),
            "B24": (True, False, False, False, True, False, False, True),
            "B25": (True, False, False, False, True, False, False, True),
            "B26": (True, False, False, False, True, False, False, True),
            "B27": (True, False, False, False, True, False, False, True),
            "B28": (True, False, False, False, True, False, False, True),
            "B29": (True, False, False, False, True, False, False, True),
            "B30": (True, False, False, False, True, False, False, True),
            "B31": (True, False, False, False, True, False, False, True),
        }
        observed_matrix = {
            variant: tuple(variant in observed_true_sets[flag]
                           for flag in expected_true_sets)
            for variant in expected_matrix
        }
        self.assertEqual(observed_matrix, expected_matrix)
        self.assertEqual(observed_matrix["B19"], observed_matrix["B16"])
        for variant in (
            "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27",
            "B28", "B29", "B30", "B31",
        ):
            self.assertEqual(observed_matrix[variant], observed_matrix["B16"])
        self.assertIn(
            "kNarrowCube1ToPipeFix, kNarrowCube2ToPipeFix, kCube1EventOnly,\n"
            "        kUpdateBarrierToPipeMte3, kUpdateBarrierEventOnly,\n"
            "        kBypassHInitCollective, kEntryLocalPipeDrain, "
            "kEntryRolePipeDrain,\n"
            "        SyncTraits::kFwdHVarlenDenseC1FullTiles,\n"
            "        SyncTraits::kFwdHVarlenDenseC2FullTiles>;", text
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
        self.assertNotIn("kFwdHVarlenDense", non_a5_alias)

    def test_solve_wrapper_maps_b20_b31_traits_only_to_a5_bt64(self):
        text = source(COEFFICIENT)

        self.assertIn(
            "GDN::GdnCoreSyncVariant kSyncVariant,",
            text,
        )
        self.assertIn(
            "using SyncTraits = GDN::GdnCoreSyncVariantTraits<kSyncVariant>;",
            text,
        )
        self.assertIn(
            "SolveTri64<T, T, SyncTraits::kUseImmediateMte2Mte1,\n"
            "                   SyncTraits::kDeferMte2Mte1Wait,\n"
            "                   SyncTraits::kHeadMajorSolve64Ownership> solve;",
            text,
        )
        self.assertIn("SolveTri128<T, T> solve;", text)

        a5_start = text.index("#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310")
        non_a5_start = text.index("#else", a5_start)
        non_a5_end = text.index("#endif", non_a5_start)
        self.assertNotIn("kSyncVariant", text[non_a5_start:non_a5_end])

    def test_b20_b31_sync_trait_truth_table_is_orthogonal(self):
        text = source(CORE_STRUCT)
        trait_start = text.index("struct GdnCoreSyncVariantTraits")
        trait_end = text.index("struct ChunkGdnCoreCoefficientTiling", trait_start)
        traits = text[trait_start:trait_end]
        expected_true_sets = {
            "kHeadMajorSolve64Ownership": {
                "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27",
                "B28", "B29", "B30", "B31",
            },
            "kKktToSolveGroupHandoff": {
                "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29",
                "B30", "B31",
            },
            "kSolveToWuGroupHandoff": {
                "B21", "B23", "B24", "B25", "B26", "B27", "B28", "B29",
                "B30", "B31",
            },
            "kUseImmediateMte2Mte1": {"B3", "B5", "B19", "B24"},
            "kFwdHVarlenDenseC1FullTiles": {"B25", "B27", "B31"},
            "kFwdHVarlenDenseC2FullTiles": {"B26", "B27", "B31"},
            "kFwdOAggregateQkMaskBarrier": {"B28", "B30", "B31"},
            "kFwdOAggregateOutputBarrier": {"B29", "B30", "B31"},
        }
        observed_true_sets = {}
        for flag in expected_true_sets:
            assignment = re.search(
                rf"static constexpr bool {flag}\s*=\s*(.*?);",
                traits,
                re.DOTALL,
            )
            self.assertIsNotNone(assignment, flag)
            observed_true_sets[flag] = set(
                re.findall(r"GdnCoreSyncVariant::(B\d+)", assignment.group(1))
            )
        self.assertEqual(observed_true_sets, expected_true_sets)
        self.assertIn("kDeferMte2Mte1Wait = false;", traits)
        self.assertIn(
            "!(kKktToSolveGroupHandoff || kSolveToWuGroupHandoff) ||\n"
            "                      kHeadMajorSolve64Ownership",
            traits,
        )
        self.assertIn(
            '"The hardware-rejected deferred Solve64 wait must remain disabled."',
            traits,
        )
        self.assertIn(
            '"H/O experiments must inherit the fully paired B23 coefficient protocol."',
            traits,
        )
        self.assertIn(
            '"H/O experiments must not inherit the precision-risky Solve64 R path."',
            traits,
        )
        for variant in (
            "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28",
            "B29", "B30", "B31",
        ):
            self.assertIn(variant, observed_true_sets["kHeadMajorSolve64Ownership"])
        self.assertIn("B24", observed_true_sets["kUseImmediateMte2Mte1"])
        self.assertTrue(
            observed_true_sets["kFwdHVarlenDenseC1FullTiles"].isdisjoint(
                observed_true_sets["kFwdOAggregateOutputBarrier"] - {"B31"}
            )
        )

    def test_head_major_varlen_solve_uses_total_chunks_not_total_tiles(self):
        text = source(SOLVE_TRI_64)

        self.assertIn("total_chunks = tilingData->totalChunks;", text)
        self.assertIn("chunk_num_total = tilingData->totalTiles;", text)
        self.assertRegex(
            text,
            r"if constexpr \(kHeadMajorVarlenOwnership\) \{\s*"
            r"(?://[^\n]*\n\s*)*"
            r"chunk_idx = loop_idx % total_chunks;\s*"
            r"head_idx = loop_idx / total_chunks;\s*"
            r"\} else \{\s*"
            r"chunk_idx = loop_idx / num_head;\s*"
            r"head_idx = loop_idx % num_head;",
        )
        ownership_start = text.index(
            "if constexpr (kHeadMajorVarlenOwnership)"
        )
        ownership_end = text.index("seq_idx = gm_chunk_indices", ownership_start)
        ownership = text[ownership_start:ownership_end]
        self.assertNotIn("totalTiles", ownership)
        self.assertNotIn("chunk_num_total", ownership)

    def test_kkt_to_solve_p_is_mode2_fanin_then_flag0_release(self):
        text = source(COEFFICIENT)

        self.assertIn("constexpr uint64_t KKT_READY_FLAG = 3;", text)
        self.assertIn("constexpr uint64_t KKT_SOLVE_RELEASE_FLAG = 0;", text)
        self.assertIn(
            "MATRIX_SIZE == 64 && SyncTraits::kKktToSolveGroupHandoff", text
        )
        p_start = text.index("if constexpr (kUseKktToSolveGroupHandoff)")
        p_end = text.index(
            "// KKT uses head-major ownership while legacy varlen SolveTri",
            p_start,
        )
        protocol = text[p_start:p_end]
        self.assertEqual(protocol.count("CrossCoreWaitFlag(KKT_READY_FLAG)"), 1)
        self.assertEqual(
            protocol.count(
                "CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG)"
            ),
            1,
        )
        self.assertEqual(
            protocol.count(
                "CrossCoreSetFlag<0x2, PIPE_FIX>(KKT_SOLVE_RELEASE_FLAG)"
            ),
            1,
        )
        self.assertEqual(
            protocol.count("CrossCoreWaitFlag(KKT_SOLVE_RELEASE_FLAG)"), 1
        )
        aic_start = protocol.index("if ASCEND_IS_AIC")
        aiv_start = protocol.index("if ASCEND_IS_AIV")
        self.assertLess(
            protocol.index("CrossCoreWaitFlag(KKT_READY_FLAG)", aic_start),
            protocol.index(
                "CrossCoreSetFlag<0x2, PIPE_FIX>(KKT_SOLVE_RELEASE_FLAG)",
                aic_start,
            ),
        )
        self.assertLess(
            protocol.index(
                "CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG)", aiv_start
            ),
            protocol.index("CrossCoreWaitFlag(KKT_SOLVE_RELEASE_FLAG)", aiv_start),
        )
        self.assertIn("two AIV notifications are aggregated by FFTS", text)
        self.assertIn("Target CANN reserves flags6/7", text)
        self.assertNotIn("GetSubBlockIdx", protocol)
        self.assertNotIn("PHASE6_SOLVE_DONE_FLAG", protocol)
        self.assertNotRegex(protocol, r"(?:FLAG|Flag)\s*\(?[67]\)?")

    def test_solve_to_wu_q_drains_aic_then_balances_flag5(self):
        text = source(CORE_KERNEL)

        q_start = text.index("if constexpr (SyncTraits::kSolveToWuGroupHandoff)")
        q_end = text.index("} else {\n            AscendC::SyncAll<false>();", q_start)
        protocol = text[q_start:q_end]
        self.assertIn("if (coefficient.BT == 64)", protocol)
        self.assertRegex(
            protocol,
            r"if ASCEND_IS_AIC \{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);\s*"
            r"AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>"
            r"\(PHASE6_SOLVE_DONE_FLAG\);\s*\}",
        )
        self.assertRegex(
            protocol,
            r"if ASCEND_IS_AIV \{\s*"
            r"AscendC::CrossCoreWaitFlag\(PHASE6_SOLVE_DONE_FLAG\);\s*\}",
        )
        self.assertEqual(protocol.count("PHASE6_SOLVE_DONE_FLAG"), 2)
        self.assertIn("both paired AIVs consume that generation", text)

    def test_group_geometry_covers_t1_t65_main_and_idle_participants(self):
        core_groups = 28
        heads = 32
        aiv_subblocks = 2
        coefficient = source(COEFFICIENT)
        core = source(CORE_KERNEL)
        kkt = source(KKT_EPILOGUE)
        solve = source(SOLVE_TRI_64)
        wu_common = source(RECOMPUTE_W_U_COMMON)
        wu_cube = source(RECOMPUTE_W_U_CUBE)
        wu_vector = source(RECOMPUTE_W_U_VECTOR)
        p_start = coefficient.index(
            "if constexpr (kUseKktToSolveGroupHandoff)"
        )
        p_end = coefficient.index(
            "// KKT uses head-major ownership while legacy varlen SolveTri",
            p_start,
        )
        p_protocol = coefficient[p_start:p_end]
        q_start = core.index(
            "if constexpr (SyncTraits::kSolveToWuGroupHandoff)"
        )
        q_end = core.index(
            "} else {\n            AscendC::SyncAll<false>();", q_start
        )
        q_protocol = core[q_start:q_end]
        p_counts = (
            aiv_subblocks * p_protocol.count(
                "CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG)"
            ),
            p_protocol.count("CrossCoreWaitFlag(KKT_READY_FLAG)"),
            p_protocol.count(
                "CrossCoreSetFlag<0x2, PIPE_FIX>(KKT_SOLVE_RELEASE_FLAG)"
            ),
            aiv_subblocks * p_protocol.count(
                "CrossCoreWaitFlag(KKT_SOLVE_RELEASE_FLAG)"
            ),
        )
        q_counts = (
            q_protocol.count(
                "CrossCoreSetFlag<0x2, PIPE_FIX>(PHASE6_SOLVE_DONE_FLAG)"
            ),
            aiv_subblocks * q_protocol.count(
                "CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG)"
            ),
        )
        self.assertEqual(p_counts, (2, 1, 1, 2))
        self.assertEqual(q_counts, (1, 2))
        self.assertNotIn("GetSubBlockIdx", p_protocol)
        self.assertNotIn("GetSubBlockIdx", q_protocol)
        self.assertRegex(
            kkt,
            r"const int64_t begin = aicIdx \* tilesPerAic;\s*"
            r"const int64_t end = MinI64\(begin \+ tilesPerAic, taskNum_\);\s*"
            r"for \(int64_t task = begin \+ subBlockIdx; task < end; "
            r"task \+= subBlockNum\)",
        )
        # Bind the geometry model below to both production schedulers, not just
        # to two equivalent Python ranges.  Solve's AIV0 and AIC sides must use
        # the same logical-core contiguous interval, and WU's cube/vector sides
        # must reproduce that interval before decoding the head-major task.
        self.assertIn(
            "const int64_t begin = contiguous_schedule ? (core_idx / 2) * "
            "tiles_per_core : core_idx / 2;",
            solve,
        )
        self.assertIn(
            "const int64_t begin = contiguous_schedule ? core_idx * "
            "tiles_per_core : core_idx;",
            solve,
        )
        self.assertEqual(
            solve.count(
                "? (begin + tiles_per_core < chunk_num_total ? begin + "
                "tiles_per_core : chunk_num_total)"
            ),
            2,
        )
        self.assertEqual(
            solve.count("const int64_t step = contiguous_schedule ? 1 : num_core;"),
            2,
        )
        self.assertRegex(
            solve,
            r"x_gm_offset = head_idx \* total_tokens \* chunk_size \+\s*"
            r"\(bos \+ chunk_in_seq_idx \* chunk_size\) \* chunk_size;",
        )
        self.assertRegex(
            wu_common,
            r"if \(cuSeqlens != nullptr\) \{\s*"
            r"chunkIdx = loopIdx % static_cast<uint32_t>\(chunkNum\);\s*"
            r"hBegin = loopIdx / static_cast<uint32_t>\(chunkNum\);",
        )
        self.assertRegex(
            wu_cube,
            r"const uint32_t coreLoops = kFlattenHeadTasks \? "
            r"params\.chunkNum \* params\.Hv : params\.chunkNum;\s*"
            r"uint32_t loopBegin = coreIdx;\s*"
            r"uint32_t loopEnd = coreLoops;\s*"
            r"uint32_t loopStep = coreNum;\s*"
            r"if constexpr \(kCoefficientGenerationTaskOrder\) \{\s*"
            r"const uint32_t tasksPerCore = "
            r"\(coreLoops \+ coreNum - 1\) / coreNum;\s*"
            r"loopBegin = coreIdx \* tasksPerCore;\s*"
            r"loopEnd = \(loopBegin \+ tasksPerCore\) < coreLoops \? "
            r"loopBegin \+ tasksPerCore : coreLoops;\s*"
            r"loopStep = 1;",
        )
        self.assertRegex(
            wu_vector,
            r"const uint32_t coreLoops = kFlattenHeadTasks \? "
            r"chunkNum \* Hv : chunkNum;\s*"
            r"const uint32_t coreIdx = GetBlockIdx\(\) / GetSubBlockNum\(\);\s*"
            r"const uint32_t coreNumAic = GetBlockNum\(\);[\s\S]*?"
            r"uint32_t loopBegin = coreIdx;\s*"
            r"uint32_t loopEnd = coreLoops;\s*"
            r"uint32_t loopStep = coreNumAic;\s*"
            r"if constexpr \(kCoefficientGenerationTaskOrder\) \{\s*"
            r"const uint32_t tasksPerCore = "
            r"\(coreLoops \+ coreNumAic - 1\) / coreNumAic;\s*"
            r"loopBegin = coreIdx \* tasksPerCore;\s*"
            r"loopEnd = \(loopBegin \+ tasksPerCore\) < coreLoops \? "
            r"loopBegin \+ tasksPerCore : coreLoops;\s*"
            r"loopStep = 1;",
        )
        for scheduler in (wu_cube, wu_vector):
            self.assertRegex(
                scheduler,
                r"for \(uint32_t loopIdx = loopBegin; loopIdx < loopEnd; "
                r"loopIdx \+= loopStep\) \{[\s\S]*?"
                r"DecodeRecomputeTask<kFlattenHeadTasks, "
                r"kCoefficientGenerationTaskOrder>\(",
            )
        cases = {
            "t1": (1, 16, 12),
            "t65": (2, 22, 6),
            "main": (177, 28, 0),
        }
        for name, (total_chunks, active_expected, idle_expected) in cases.items():
            task_count = heads * total_chunks
            tasks_per_core = (task_count + core_groups - 1) // core_groups
            all_pairs = {
                (head, chunk)
                for head in range(heads)
                for chunk in range(total_chunks)
            }
            decoded_pairs = {
                (task // total_chunks, task % total_chunks)
                for task in range(task_count)
            }
            self.assertEqual(decoded_pairs, all_pairs, name)

            active = 0
            idle = 0
            for core in range(core_groups):
                begin = core * tasks_per_core
                end = min(begin + tasks_per_core, task_count)
                solve_tasks = list(range(begin, end))
                aiv0_tasks = list(range(begin, end, 2))
                aiv1_tasks = list(range(begin + 1, end, 2))
                self.assertEqual(
                    sorted(aiv0_tasks + aiv1_tasks), solve_tasks, (name, core)
                )
                # S gives Solve and WU the same contiguous task interval.
                wu_tasks = list(range(begin, end))
                self.assertEqual(solve_tasks, wu_tasks, (name, core))
                if solve_tasks:
                    active += 1
                else:
                    idle += 1

            self.assertEqual((active, idle), (active_expected, idle_expected), name)

        # T65's last active range contains only one task: AIV0 produces it and
        # AIV1 is empty but must still publish/wait at both P protocol edges.
        task_count = heads * 2
        tasks_per_core = (task_count + core_groups - 1) // core_groups
        last_active_core = (task_count - 1) // tasks_per_core
        begin = last_active_core * tasks_per_core
        end = min(begin + tasks_per_core, task_count)
        self.assertEqual(list(range(begin, end)), [63])
        self.assertEqual(list(range(begin, end, 2)), [63])
        self.assertEqual(list(range(begin + 1, end, 2)), [])

    def test_solve_tri64_immediate_event_order_preserves_helper_edge(self):
        text = source(SOLVE_TRI_64)

        self.assertIn(
            "bool kUseMte2Mte1Event = false,\n"
            "          bool kDeferMte2Mte1Wait = false,\n"
            "          bool kHeadMajorVarlenOwnership = false",
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
        for variant in tuple(f"B{index}" for index in range(1, 32)):
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
            accepted_values, {f"B{index}" for index in range(32)}
        )
        self.assertIn("value == nullptr", resolver)
        self.assertEqual(resolver.count("return true;"), 32)
        self.assertEqual(resolver.count("return false;"), 1)
        for selector in tuple(f"B{index}" for index in range(32)):
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
            "TILING_KEY_B22_V128": "221",
            "TILING_KEY_B23_V128": "231",
            "TILING_KEY_B24_V128": "241",
            "TILING_KEY_B25_V128": "251",
            "TILING_KEY_B26_V128": "261",
            "TILING_KEY_B27_V128": "271",
            "TILING_KEY_B28_V128": "281",
            "TILING_KEY_B29_V128": "291",
            "TILING_KEY_B30_V128": "301",
            "TILING_KEY_B31_V128": "311",
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
            "B22": "TILING_KEY_B22_V128",
            "B23": "TILING_KEY_B23_V128",
            "B24": "TILING_KEY_B24_V128",
            "B25": "TILING_KEY_B25_V128",
            "B26": "TILING_KEY_B26_V128",
            "B27": "TILING_KEY_B27_V128",
            "B28": "TILING_KEY_B28_V128",
            "B29": "TILING_KEY_B29_V128",
            "B30": "TILING_KEY_B30_V128",
            "B31": "TILING_KEY_B31_V128",
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
                "B17", "B18", "B19", "B20", "B21", "B22", "B23",
                "B24", "B25", "B26", "B27", "B28", "B29", "B30",
                "B31",
            },
        )
        t1_variant_start = main_only_end
        t1_variant_end = host.index(
            "bool IsT65DiagnosticSyncVariant(", t1_variant_start
        )
        t1_variants = host[t1_variant_start:t1_variant_end]
        self.assertEqual(
            set(re.findall(r"GdnCoreSyncVariant::(B\d+)", t1_variants)),
            {
                "B16", "B17", "B18", "B20", "B21", "B22", "B23", "B24",
                "B25", "B26", "B27", "B28", "B29", "B30", "B31",
            },
        )
        t65_variant_start = t1_variant_end
        t65_variant_end = host.index(
            "uint32_t ResolveTilingKey(", t65_variant_start
        )
        t65_variants = host[t65_variant_start:t65_variant_end]
        self.assertEqual(
            set(re.findall(r"GdnCoreSyncVariant::(B\d+)", t65_variants)),
            {
                "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26",
                "B27", "B28", "B29", "B30", "B31",
            },
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
        # experimental domain; B12-B31 narrow to the exact main varlen shape.
        # B16-B18 and B20-B31 retain the exact-T=1 diagnostic exception;
        # B19-B31 have an exact-T=65/two-chunk route that executes full+tail.
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
                requested in {
                    "B16", "B17", "B18", "B20", "B21", "B22", "B23", "B24",
                    "B25", "B26", "B27", "B28", "B29", "B30", "B31",
                }
                and t1_diagnostic_enabled
                and exact_t1
            )
            select_t65 = (
                requested in {
                    "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26",
                    "B27", "B28", "B29", "B30", "B31",
                }
                and t65_diagnostic_enabled
                and exact_t65
            )
            if (
                requested in {
                    "B12", "B13", "B14", "B15", "B16",
                    "B17", "B18", "B19", "B20", "B21", "B22", "B23",
                    "B24", "B25", "B26", "B27", "B28", "B29", "B30",
                    "B31",
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
        for requested in ("B16", "B17", "B18"):
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
        self.assertEqual(selected("B19", True, True), "B19")
        self.assertEqual(selected("B19", True, False), "B7")
        self.assertEqual(
            selected("B19", True, False, False, False, True, True), "B19"
        )
        self.assertEqual(
            selected("B19", True, False, False, False, False, True), "B7"
        )
        self.assertEqual(
            selected("B19", True, False, False, False, True, False), "B7"
        )
        self.assertEqual(selected("B19", True, False, True, True), "B7")
        self.assertEqual(
            selected("B19", False, False, False, False, True, True), "B0"
        )
        for requested in (
            "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27",
            "B28", "B29", "B30", "B31",
        ):
            self.assertEqual(selected(requested, True, True), requested)
            self.assertEqual(selected(requested, True, False), "B7")
            self.assertEqual(
                selected(requested, True, False, True, True), requested
            )
            self.assertEqual(
                selected(requested, True, False, False, False, True, True),
                requested,
            )
            self.assertEqual(
                selected(requested, True, False, False, True), "B7"
            )
            self.assertEqual(
                selected(requested, True, False, True, False), "B7"
            )
            self.assertEqual(
                selected(requested, True, False, False, False, False, True),
                "B7",
            )
            self.assertEqual(
                selected(requested, False, False, True, True), "B0"
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
                221: ("GDNFwdHTileShapes128", "B22"),
                231: ("GDNFwdHTileShapes128", "B23"),
                241: ("GDNFwdHTileShapes128", "B24"),
                251: ("GDNFwdHTileShapes128", "B25"),
                261: ("GDNFwdHTileShapes128", "B26"),
                271: ("GDNFwdHTileShapes128", "B27"),
                281: ("GDNFwdHTileShapes128", "B28"),
                291: ("GDNFwdHTileShapes128", "B29"),
                301: ("GDNFwdHTileShapes128", "B30"),
                311: ("GDNFwdHTileShapes128", "B31"),
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
            191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311,
        ):
            self.assertIn(f"TILING_KEY_IS({key})", a5_dispatch)
        for key in (
            12, 22, 32, 42, 52, 62, 72, 82,
            92, 102, 112, 122, 132, 142, 152, 162, 172, 182,
            192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312,
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
            r'"FLA_NPU_GDN_SYNC_VARIANT B1/B2/B3/B4/B5/B6/B7/B8/B9/B10/B11/B12/B13/B14/B15/B16/B17/B18/B19/B20/B21/B22/B23/B24/B25/B26/B27/B28/B29/B30/B31 is supported only on Ascend950\."\),\s*'
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
            enum_values, {f"B{index}": index for index in range(32)}
        )

        trailer_start = struct.index("struct ChunkGdnCoreFwdTrailer {")
        trailer_end = struct.index("};", trailer_start)
        trailer = struct[trailer_start:trailer_end]
        self.assertNotIn("GdnCoreSyncVariant", trailer)
        self.assertNotIn("syncVariant", trailer)
        self.assertNotIn("trailer.syncVariant", host)


if __name__ == "__main__":
    unittest.main()
