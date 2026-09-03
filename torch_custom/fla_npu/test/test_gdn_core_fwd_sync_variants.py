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
SOLVE_TRI_64 = (
    KERNEL_ROOT
    / "internal/coefficient_generation/gdn_core_solve_tri/arch35/solve_tri_ascend950_64.h"
)


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class GdnCoreFwdSyncVariantSourceTests(unittest.TestCase):
    def test_fwd_h_variants_only_guard_two_bounded_mmad_barriers(self):
        text = source(FWD_H_ARCH35)

        self.assertIn("bool kSkipBoundedMmadPipeAll = false", text)
        self.assertEqual(text.count("if constexpr (!kSkipBoundedMmadPipeAll)"), 2)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_ALL>();"), 2)
        self.assertEqual(text.count("AscendC::SyncAll<false>();"), 3)
        self.assertEqual(text.count("AscendC::PipeBarrier<PIPE_V>();"), 10)

        bounded_patterns = (
            r"blockMmadWH\.finalWaitFlags\(\);\s*"
            r"}\s*(?://[^\n]*\n\s*)*"
            r"if constexpr \(!kSkipBoundedMmadPipeAll\) \{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);",
            r"blockMmadKV\.finalWaitFlags\(\);\s*"
            r"}\s*(?://[^\n]*\n\s*)*"
            r"if constexpr \(!kSkipBoundedMmadPipeAll\) \{\s*"
            r"AscendC::PipeBarrier<PIPE_ALL>\(\);",
        )
        for pattern in bounded_patterns:
            self.assertRegex(text, pattern)

    def test_fwd_h_wrapper_maps_all_non_b0_variants_only_on_a5(self):
        text = source(STATE_UPDATE)

        self.assertIn(
            "GdnCoreSyncVariant kSyncVariant = GdnCoreSyncVariant::B0", text
        )
        self.assertEqual(text.count("RunFwdH<"), 16)
        self.assertEqual(text.count(", kSyncVariant>("), 16)
        self.assertIn(
            "kSyncVariant != GdnCoreSyncVariant::B0>;", text
        )
        wrapper_start = text.index("__aicore__ inline void RunFwdH(")
        a5_start = text.index(
            "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", wrapper_start
        )
        non_a5_start = text.index("#else", a5_start)
        alias_end = text.index("#endif", non_a5_start)
        self.assertNotIn("kSyncVariant !=", text[non_a5_start:alias_end])

    def test_solve_wrapper_limits_narrow_event_to_a5_bt64_b2_b3(self):
        text = source(COEFFICIENT)

        self.assertIn(
            "GDN::GdnCoreSyncVariant kSyncVariant,",
            text,
        )
        self.assertRegex(
            text,
            r"kUseMte2Mte1Event\s*=\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B2 \|\|\s*"
            r"kSyncVariant == GDN::GdnCoreSyncVariant::B3;",
        )
        self.assertIn(
            "kDeferMte2Mte1Wait =\n            "
            "kSyncVariant == GDN::GdnCoreSyncVariant::B3;",
            text,
        )
        self.assertIn(
            "SolveTri64<T, T, kUseMte2Mte1Event, kDeferMte2Mte1Wait> solve;",
            text,
        )
        self.assertIn("SolveTri128<T, T> solve;", text)

        a5_start = text.index("#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310")
        non_a5_start = text.index("#else", a5_start)
        non_a5_end = text.index("#endif", non_a5_start)
        self.assertNotIn("kSyncVariant", text[non_a5_start:non_a5_end])

    def test_solve_tri64_b2_and_b3_event_order_preserves_helper_edge(self):
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
        for variant in ("B1", "B2", "B3"):
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
        resolver_end = host.index("uint32_t ResolveTilingKey(", resolver_start)
        resolver = host[resolver_start:resolver_end]
        accepted_values = set(
            re.findall(r'std::strcmp\(value, "([^"]+)"\)', resolver)
        )
        self.assertEqual(accepted_values, {"B0", "B1", "B2", "B3"})
        self.assertIn("value == nullptr", resolver)
        self.assertEqual(resolver.count("return true;"), 4)
        self.assertEqual(resolver.count("return false;"), 1)
        for selector in ("B0", "B1", "B2", "B3"):
            prefix = r"value == nullptr \|\| " if selector == "B0" else ""
            self.assertRegex(
                resolver,
                rf"if \({prefix}std::strcmp\(value, \"{selector}\"\) == 0\) \{{\s*"
                rf"variant = GDN::GdnCoreSyncVariant::{selector};\s*return true;",
            )

        expected_keys = {
            "TILING_KEY_B0_V128": "1",
            "TILING_KEY_B0_V256": "2",
            "TILING_KEY_B1_V128": "11",
            "TILING_KEY_B2_V128": "21",
            "TILING_KEY_B3_V128": "31",
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
            r"vDim == SUPPORTED_V_DIM_128;",
        )
        self.assertIn(
            "context->GetOptionalInputDesc(INPUT_INITIAL_STATE)", host
        )
        self.assertRegex(
            host,
            r"const GDN::GdnCoreSyncVariant effectiveSyncVariant\s*=\s*"
            r"isExperimentalShape \? syncVariant : "
            r"GDN::GdnCoreSyncVariant::B0;",
        )

        routed = {
            int(key): (shape, variant)
            for key, shape, variant in re.findall(
                r"(?:if|else if) \(TILING_KEY_IS\((\d+)\)\) \{\s*"
                r"KERNEL_TASK_TYPE\(\1, KERNEL_TYPE_MIX_AIC_1_2\);\s*"
                r"GDN::DispatchPhase6ByDtype<"
                r"Catlass::Gemm::Kernel::(GDNFwdHTileShapes(?:128|256)),\s*"
                r"GDN::GdnCoreSyncVariant::(B[0-3])>",
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
        for key in (11, 21, 31):
            self.assertIn(f"TILING_KEY_IS({key})", a5_dispatch)
        for key in (12, 22, 32):
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
            r'"FLA_NPU_GDN_SYNC_VARIANT B1/B2/B3 is supported only on Ascend950\."\),\s*'
            r"return ge::GRAPH_FAILED\);",
        )

        trailer_start = struct.index("struct ChunkGdnCoreFwdTrailer {")
        trailer_end = struct.index("};", trailer_start)
        trailer = struct[trailer_start:trailer_end]
        self.assertNotIn("GdnCoreSyncVariant", trailer)
        self.assertNotIn("syncVariant", trailer)
        self.assertNotIn("trailer.syncVariant", host)


if __name__ == "__main__":
    unittest.main()
