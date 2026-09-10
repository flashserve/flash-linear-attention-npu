"""检查冻结 ATK 用例与统一 JSON 中的声明完全一致。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
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
                171,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_perf.json",
                GENERATOR.build_perf_specs(),
                7,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_mss.json",
                GENERATOR.build_mss_specs(),
                144,
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


if __name__ == "__main__":
    unittest.main()
