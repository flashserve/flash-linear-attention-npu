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
    def test_fwd_h_retains_the_accepted_sync_axes(self):
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
        self.assertNotIn("kFwdHVarlenDense", text)
        self.assertNotIn("useDenseFullTile", text)
        self.assertNotIn("denseFullGenerationOpen", text)
        self.assertEqual(text.count("if constexpr (kNarrowCube1ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (kNarrowCube2ToPipeFix)"), 1)
        self.assertEqual(text.count("if constexpr (!kCube1EventOnly)"), 1)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_ALL>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_FIX>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_MTE3>();"), 1)
        self.assertEqual(text.count("AscendC::SyncAll<false>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_V>();"), 10)

    def test_fwd_h_tail_calls_keep_actual_shapes_and_event_lifetime(self):
        standalone = (REPO_ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd"
                      / "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp")
        for path in (FWD_H_ARCH35, standalone):
            with self.subTest(path=path):
                text = source(path)
                for stage, operand in ((1, "WH"), (2, "KV")):
                    # Tail calls retain the bounded implementation and drain;
                    # neither caller forces whole-L1 initialization anymore.
                    self.assertRegex(
                        text,
                        rf"blockMmad{operand}Tail.preSetFlags\(\);\s*"
                        rf"blockMmad{operand}Tail\([^;]*cube{stage}Shape\);\s*"
                        rf"blockMmad{operand}Tail.finalWaitFlags\(\);",
                    )
                    self.assertNotIn(f"cube{stage}Shape, EmptyClass{{}}, true", text)
                self.assertRegex(
                    text, r"GemmCoord cube1Shape\s*\{\s*"
                    r"cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim\s*\}")
                self.assertRegex(
                    text, r"GemmCoord cube2Shape\s*\{\s*"
                    r"kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens\s*\}")

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

    def test_release_wrapper_maps_only_b30_h_o_traits_on_a5(self):
        text = source(STATE_UPDATE)

        self.assertEqual(text.count("RunFwdH<"), 16)
        self.assertEqual(text.count("RunFwdO<"), 4)
        self.assertEqual(text.count(", kSyncVariant>("), 20)
        self.assertIn(
            "constexpr bool kNarrowCube1ToPipeFix =\n"
            "        kSyncVariant == GdnCoreSyncVariant::B30;",
            text,
        )
        self.assertIn("constexpr bool kNarrowCube2ToPipeFix = false;", text)
        self.assertIn("constexpr bool kCube1EventOnly = false;", text)
        self.assertIn("constexpr bool kUpdateBarrierToPipeMte3 = false;", text)
        self.assertIn(
            "constexpr bool kUpdateBarrierEventOnly =\n"
            "        kSyncVariant == GdnCoreSyncVariant::B30;",
            text,
        )
        self.assertIn("constexpr bool kBypassHInitCollective = false;", text)
        self.assertIn("constexpr bool kEntryLocalPipeDrain = false;", text)
        self.assertIn(
            "constexpr bool kEntryRolePipeDrain =\n"
            "        kSyncVariant == GdnCoreSyncVariant::B30;",
            text,
        )
        self.assertIn(
            "kBypassHInitCollective, kEntryLocalPipeDrain, kEntryRolePipeDrain>;",
            text,
        )
        self.assertNotIn("kFwdHVarlenDense", text)
        self.assertIn(
            "SyncTraits::kFwdOAggregateQkMaskBarrier,\n"
            "        SyncTraits::kFwdOAggregateOutputBarrier>;",
            text,
        )

        wrapper_start = text.index("__aicore__ inline void RunFwdH(")
        a5_start = text.index(
            "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", wrapper_start
        )
        non_a5_start = text.index("#else", a5_start)
        alias_end = text.index("#endif", non_a5_start)
        non_a5_alias = text[non_a5_start:alias_end]
        for flag in (
            "kNarrowCube",
            "kCube1EventOnly",
            "kUpdateBarrier",
            "kBypassHInitCollective",
            "kEntryLocalPipeDrain",
            "kEntryRolePipeDrain",
        ):
            self.assertNotIn(flag, non_a5_alias)

    def test_solve_wrapper_maps_b30_traits_only_to_a5_bt64(self):
        text = source(COEFFICIENT)

        self.assertIn("GDN::GdnCoreSyncVariant kSyncVariant,", text)
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

    def test_release_sync_trait_truth_table_contains_only_b0_b30(self):
        text = source(CORE_STRUCT)
        enum_start = text.index("enum class GdnCoreSyncVariant")
        enum_end = text.index("};", enum_start)
        enum_values = {
            name: int(value)
            for name, value in re.findall(
                r"\b(B\d+)\s*=\s*(\d+)", text[enum_start:enum_end]
            )
        }
        self.assertEqual(enum_values, {"B0": 0, "B30": 30})

        trait_start = text.index("struct GdnCoreSyncVariantTraits")
        trait_end = text.index("struct ChunkGdnCoreCoefficientTiling", trait_start)
        traits = text[trait_start:trait_end]
        for flag in (
            "kHeadMajorSolve64Ownership",
            "kKktToSolveGroupHandoff",
            "kSolveToWuGroupHandoff",
            "kFwdOAggregateQkMaskBarrier",
            "kFwdOAggregateOutputBarrier",
        ):
            assignment = re.search(
                rf"static constexpr bool {flag}\s*=\s*(.*?);",
                traits,
                re.DOTALL,
            )
            self.assertIsNotNone(assignment, flag)
            self.assertEqual(
                set(re.findall(r"GdnCoreSyncVariant::(B\d+)", assignment.group(1))),
                {"B30"},
                flag,
            )
        self.assertIn("kUseImmediateMte2Mte1 = false;", traits)
        self.assertIn("kDeferMte2Mte1Wait = false;", traits)
        self.assertNotIn("kFwdHVarlenDense", traits)
        self.assertIn(
            "kFwdOAggregateQkMaskBarrier || kFwdOAggregateOutputBarrier",
            traits,
        )
        self.assertIn(
            '"B30 FwdO aggregation requires the fully paired coefficient protocol."',
            traits,
        )
        self.assertIn(
            '"B30 FwdO aggregation must not inherit the precision-risky Solve64 R path."',
            traits,
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

    def test_b30_host_and_compiled_state_domains_match(self):
        host = source(HOST_TILING)
        start = host.index("const bool isExperimentalShape =")
        shape_gate = host[start:host.index(";", start)]
        self.assertIn("isBf16 && initialStateDesc != nullptr", shape_gate)
        self.assertEqual(
            set(re.findall(r"initialStateDesc->GetDataType\(\) == ge::(DT_\w+)", shape_gate)),
            {"DT_FLOAT", "DT_BF16"},
        )
        self.assertRegex(
            shape_gate,
            r"\(initialStateDesc->GetDataType\(\) == ge::DT_FLOAT \|\|\s*"
            r"initialStateDesc->GetDataType\(\) == ge::DT_BF16\) &&",
        )
        self.assertIn("vDim == SUPPORTED_V_DIM_128 && *chunkSize == CHUNK_64", shape_gate)
        core = source(CORE_KERNEL)
        key = core.index("} else if (TILING_KEY_IS(301))")
        compile_gate = core[core.rindex("#if", 0, key):key]
        self.assertIn("__CCE_AICORE__ == 310", compile_gate)
        self.assertIn("ORIG_DTYPE_Q == DT_BF16", compile_gate)
        self.assertIn("defined(ORIG_DTYPE_INITIAL_STATE)", compile_gate)
        self.assertEqual(
            set(re.findall(r"ORIG_DTYPE_INITIAL_STATE == (DT_\w+)", compile_gate)),
            {"DT_FLOAT", "DT_BF16"},
        )
        self.assertIn(
            "((ORIG_DTYPE_INITIAL_STATE == DT_FLOAT) || (ORIG_DTYPE_INITIAL_STATE == DT_BF16))",
            compile_gate,
        )
        # BF16 state uses the existing dispatch; no state conversion or separate
        # public entry point is introduced by the broader key compilation domain.
        self.assertIn(
            "RunFwdH<bfloat16_t, float, bfloat16_t, TileShapes, false, kSyncVariant>",
            source(STATE_UPDATE),
        )

    def test_top_level_dispatches_only_b0_and_promoted_b30(self):
        text = source(CORE_KERNEL)

        routed = {
            int(key): (shape, variant)
            for key, shape, variant in re.findall(
                r"(?:if|else if) \(TILING_KEY_IS\((\d+)\)\) \{\s*"
                r"KERNEL_TASK_TYPE\(\1, KERNEL_TYPE_MIX_AIC_1_2\);\s*"
                r"GDN::DispatchPhase6ByDtype<"
                r"Catlass::Gemm::Kernel::(GDNFwdHTileShapes(?:128|256)),\s*"
                r"GDN::GdnCoreSyncVariant::(B\d+)>",
                text,
            )
        }
        self.assertEqual(
            routed,
            {
                1: ("GDNFwdHTileShapes128", "B0"),
                2: ("GDNFwdHTileShapes256", "B0"),
                301: ("GDNFwdHTileShapes128", "B30"),
            },
        )
        self.assertEqual(
            set(map(int, re.findall(r"TILING_KEY_IS\((\d+)\)", text))),
            {1, 2, 301},
        )
        self.assertIn("RunSolvePhase<InputT, 64, kSyncVariant>", text)
        self.assertIn("RunSolvePhase<InputT, 128, kSyncVariant>", text)
        self.assertIn("DispatchFwdH<TileShapes, kSyncVariant>", text)
        self.assertIn("DispatchFwdO<kSyncVariant>", text)

    def test_host_release_router_is_strict_fail_closed_and_abi_neutral(self):
        host = source(HOST_TILING)
        struct = source(CORE_STRUCT)

        resolver_start = host.index("bool ResolveSyncVariant(")
        resolver_end = host.index("uint32_t ResolveTilingKey(", resolver_start)
        resolver = host[resolver_start:resolver_end]
        self.assertIn("explicitSelection = value != nullptr;", resolver)
        self.assertEqual(
            set(re.findall(r'std::strcmp\(value, "([^"]+)"\)', resolver)),
            {"B0", "B30"},
        )
        self.assertEqual(resolver.count("return true;"), 2)
        self.assertEqual(resolver.count("return false;"), 1)
        self.assertRegex(
            resolver,
            r'if \(value == nullptr \|\| std::strcmp\(value, "B0"\) == 0\) '
            r"\{\s*variant = GDN::GdnCoreSyncVariant::B0;\s*return true;",
        )
        self.assertRegex(
            resolver,
            r'if \(std::strcmp\(value, "B30"\) == 0\) '
            r"\{\s*variant = GDN::GdnCoreSyncVariant::B30;\s*return true;",
        )
        self.assertNotIn("FLA_NPU_GDN_SYNC_T1_DIAGNOSTIC", host)
        self.assertNotIn("FLA_NPU_GDN_SYNC_T65_DIAGNOSTIC", host)
        self.assertIn(
            "FLA_NPU_GDN_SYNC_VARIANT must be unset, B0, or B30.", host
        )
        self.assertIn(
            "FLA_NPU_GDN_SYNC_VARIANT B30 is supported only on Ascend950.", host
        )

        constants = {
            name: int(value)
            for name, value in re.findall(
                r"constexpr uint32_t (TILING_KEY_B\w+?) = (\d+);", host
            )
        }
        self.assertEqual(
            constants,
            {
                "TILING_KEY_B0_V128": 1,
                "TILING_KEY_B0_V256": 2,
                "TILING_KEY_B30_V128": 301,
            },
        )
        resolve_start = host.index("uint32_t ResolveTilingKey(")
        resolve_end = host.index("uint64_t CeilDiv(", resolve_start)
        resolve = host[resolve_start:resolve_end]
        self.assertIn(
            "vDim == SUPPORTED_V_DIM_256 ? TILING_KEY_B0_V256 : "
            "TILING_KEY_B0_V128",
            resolve,
        )
        self.assertRegex(
            resolve,
            r"if \(vDim != SUPPORTED_V_DIM_128\) \{\s*return b0Key;\s*\}",
        )
        self.assertRegex(
            resolve,
            r"case GDN::GdnCoreSyncVariant::B0:\s*return b0Key;",
        )
        self.assertRegex(
            resolve,
            r"case GDN::GdnCoreSyncVariant::B30:\s*"
            r"return TILING_KEY_B30_V128;",
        )

        exact_start = host.index("const bool isExactMainVarlenShape")
        effective_end = host.index(
            "OP_CHECK_IF(Tiling4ChunkGdnCoreStateOutput", exact_start
        )
        selector = host[exact_start:effective_end]
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
            self.assertIn(required, selector)
        self.assertNotIn("outputMask", selector)
        self.assertRegex(
            selector,
            r"const GDN::GdnCoreSyncVariant effectiveSyncVariant =\s*"
            r"requestedSyncVariant == GDN::GdnCoreSyncVariant::B30 &&\s*"
            r"isExactMainVarlenShape\s*\? GDN::GdnCoreSyncVariant::B30\s*"
            r": GDN::GdnCoreSyncVariant::B0;",
        )
        self.assertIn(
            "!syncVariantExplicit && isAscend950\n"
            "            ? GDN::GdnCoreSyncVariant::B30\n"
            "            : syncVariant;",
            host,
        )
        self.assertIn(
            "context->SetTilingKey(ResolveTilingKey(vDim, effectiveSyncVariant));",
            host,
        )

        def selected(
            explicit: bool,
            requested: str,
            is_a5: bool,
            exact_main: bool,
        ) -> str:
            requested_variant = "B30" if not explicit and is_a5 else requested
            return "B30" if requested_variant == "B30" and exact_main else "B0"

        self.assertEqual(selected(False, "B0", True, True), "B30")
        self.assertEqual(selected(False, "B0", True, False), "B0")
        self.assertEqual(selected(False, "B0", False, True), "B0")
        self.assertEqual(selected(True, "B0", True, True), "B0")
        self.assertEqual(selected(True, "B30", True, True), "B30")
        self.assertEqual(selected(True, "B30", True, False), "B0")

        enum_start = struct.index("enum class GdnCoreSyncVariant")
        enum_end = struct.index("};", enum_start)
        enum_values = {
            name: int(value)
            for name, value in re.findall(
                r"\b(B\d+)\s*=\s*(\d+)", struct[enum_start:enum_end]
            )
        }
        self.assertEqual(enum_values, {"B0": 0, "B30": 30})

        trailer_start = struct.index("struct ChunkGdnCoreFwdTrailer {")
        trailer_end = struct.index("};", trailer_start)
        trailer = struct[trailer_start:trailer_end]
        self.assertNotIn("GdnCoreSyncVariant", trailer)
        self.assertNotIn("syncVariant", trailer)
        self.assertNotIn("trailer.syncVariant", host)


if __name__ == "__main__":
    unittest.main()
