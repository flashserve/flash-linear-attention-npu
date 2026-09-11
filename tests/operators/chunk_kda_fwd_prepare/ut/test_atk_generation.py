"""检查冻结 ATK 用例与统一 JSON 中的声明完全一致。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
GENERATOR_PATH = (
    ROOT / "tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py"
)
ATK_DIR = GENERATOR_PATH.parent


def _load_generator():
    module_name = "chunk_kda_fwd_prepare_atk_generator_test"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


class ChunkKdaFwdPrepareAtkGenerationTest(unittest.TestCase):
    def test_generation_config_comes_from_canonical_manifest(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(GENERATOR.GENERATION, manifest["atk_generation"])
        source = GENERATOR_PATH.read_text(encoding="utf-8")
        self.assertNotIn("FUNCTIONAL_SPECS", source)
        self.assertNotIn("PERF_SPECS", source)

    def test_frozen_json_matches_generator(self):
        suites = (
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare.json",
                GENERATOR.build_accuracy_specs(),
                200,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_perf.json",
                GENERATOR.build_perf_specs(),
                9,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_mss.json",
                GENERATOR.build_mss_specs(),
                432,
            ),
        )
        for path, specs, expected_count in suites:
            with self.subTest(path=path.name):
                frozen = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(frozen), expected_count)
                self.assertEqual(frozen, GENERATOR._payloads(specs))

    def test_output_modes_cover_accuracy_and_inference_performance(self):
        accuracy_modes = {
            spec["backward_mode"]
            for spec in GENERATOR.build_accuracy_specs()
        }
        self.assertEqual(accuracy_modes, {"none", "recompute", "save"})
        self.assertEqual(
            {
                spec["backward_mode"]
                for spec in GENERATOR.build_perf_specs()
            },
            {"none"},
        )
        self.assertEqual(
            {
                spec["backward_mode"]
                for spec in GENERATOR.build_mss_specs()
            },
            {"none", "recompute", "save"},
        )

    def test_accuracy_has_200_fixed_seed_cases(self):
        specs = GENERATOR.build_accuracy_specs()
        logical_counts = Counter(spec["logical_case_key"] for spec in specs)
        self.assertEqual(len(specs), 200)
        self.assertEqual(len(logical_counts), 56)
        self.assertGreaterEqual(min(logical_counts.values()), 3)
        self.assertEqual(len({spec["seed"] for spec in specs}), 200)

    def test_mss_covers_every_reachable_tiling_key_and_pipeline_profile(self):
        specs = GENERATOR.build_mss_specs()
        keys = {spec["expected_tiling_key"] for spec in specs}
        self.assertEqual(len(specs), 432)
        self.assertEqual(len(keys), 432)
        self.assertEqual(min(keys), 2570)
        self.assertEqual(max(keys), 24452638)
        self.assertEqual(Counter(spec["backward_mode"] for spec in specs), {
            "none": 144,
            "recompute": 144,
            "save": 144,
        })
        self.assertEqual({spec["layout"] for spec in specs}, {
            "BNSD", "BSND", "NTD", "TND",
        })
        self.assertEqual(
            set(Counter(spec["mss_profile"] for spec in specs).values()),
            {36},
        )
        for profile in {spec["mss_profile"] for spec in specs}:
            profile_specs = [
                spec for spec in specs if spec["mss_profile"] == profile
            ]
            self.assertEqual({spec["gate_dtype"] for spec in profile_specs}, {
                "bf16", "fp32",
            })
            self.assertEqual({spec["beta_dtype"] for spec in profile_specs}, {
                "bf16", "fp32",
            })
            self.assertEqual({
                spec["use_qk_l2norm_in_kernel"] for spec in profile_specs
            }, {False, True})
            self.assertEqual({spec["use_exp2"] for spec in profile_specs}, {
                False, True,
            })
            self.assertEqual({
                (
                    spec["use_exp2"],
                    spec["dt_bias"],
                )
                for spec in profile_specs
                if spec["use_gate_in_kernel"]
            }, {
                (False, False), (False, True),
                (True, False), (True, True),
            })
        digest = hashlib.sha256(
            "".join(
                f"{spec['case_id']},{spec['expected_tiling_key']}\n"
                for spec in specs
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            digest,
            "f28a869f07d8e3f65e8d0c85768759bde635759a09d047cd78b4a9b7136d3b19",
        )
        tags = {tag for spec in specs for tag in spec["tags"].split(",")}
        for required in (
            "dense", "varlen", "auto_indices", "explicit_indices",
            "grid_stride", "slot_reuse", "head_split", "gva",
            "active1", "active2", "active3", "active4",
        ):
            self.assertIn(required, tags)

    def test_mss_profiles_cover_real_schedule_branches(self):
        profiles = {
            spec["mss_profile"]: spec
            for spec in GENERATOR.build_mss_specs()
        }

        def schedule(spec, core_count=128):
            if spec["cu_seqlens"]:
                cu = [int(value) for value in spec["cu_seqlens"].split(",")]
                chunks = sum(
                    (end - begin + 63) // 64
                    for begin, end in zip(cu, cu[1:])
                )
            else:
                chunks = int(spec["B"]) * ((int(spec["T"]) + 63) // 64)
            head_split = chunks < core_count and int(spec["HK"]) > 1
            heads_per_partition = (
                int(spec["HV"]) // int(spec["HK"])
                if head_split else int(spec["HV"])
            )
            total_work = chunks * (int(spec["HK"]) if head_split else 1)
            return chunks, heads_per_partition, total_work

        for name in (
            "varlen_ntd_grid_stride_auto",
            "varlen_tnd_grid_stride_explicit",
        ):
            chunks, _, total_work = schedule(profiles[name])
            self.assertEqual(chunks, 129)
            self.assertEqual(total_work, 129)

        chunks, heads_per_partition, total_work = schedule(
            profiles["dense_bnsd_grid_stride_gva_chunk_only"]
        )
        self.assertEqual(chunks, 130)
        self.assertEqual(heads_per_partition, 4)
        self.assertEqual(total_work, 130)

        chunks, heads_per_partition, total_work = schedule(
            profiles["dense_bsnd_gva"]
        )
        self.assertEqual(chunks, 2)
        self.assertEqual(heads_per_partition, 2)
        self.assertEqual(total_work, 4)

    def test_executor_checks_public_output_contract_without_extra_npu_kernels(self):
        source = (
            ATK_DIR / "executor_chunk_kda_fwd_prepare.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("return result.contiguous()"), 2)
        self.assertIn("_validate_output_contract(", source)
        self.assertIn("torch.npu.synchronize()", source)
        self.assertIn("if not with_output:", source)
        self.assertIn("output.detach().cpu()", source)


if __name__ == "__main__":
    unittest.main()
