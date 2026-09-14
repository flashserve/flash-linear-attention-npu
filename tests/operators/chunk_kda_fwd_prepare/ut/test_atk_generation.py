"""校验 chunk_kda_fwd_prepare 的 ATK 直参用例合同。"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ATK_DIR = ROOT / "tests/atk/chunk_kda_fwd_prepare"
GENERATOR_PATH = ATK_DIR / "gen_chunk_kda_fwd_prepare.py"
YAML_PATH = ATK_DIR / "chunk_kda_fwd_prepare.yaml"
EXECUTOR_PATH = ATK_DIR / "executor_chunk_kda_fwd_prepare.py"
JSON_PATHS = {
    "accuracy": ATK_DIR / "atk_chunk_kda_fwd_prepare.json",
    "perf": ATK_DIR / "atk_chunk_kda_fwd_prepare_perf.json",
    "mss": ATK_DIR / "atk_chunk_kda_fwd_prepare_mss.json",
}

EXPECTED_CASE_FIELDS = {
    "id",
    "default_seed",
    "name",
    "aclnn_name",
    "triton_name",
    "kernel_name",
    "version",
    "expected_error_msg",
    "api",
    "api_type",
    "aclnn_api_type",
    "triton_api_type",
    "fusion_api_type",
    "fusion_mode",
    "dist_api_type",
    "kernel_api_type",
    "backward",
    "standard",
    "outputs",
    "inputs",
    "acl_json",
    "method_inputs",
    "tensor_input",
    "compute_times",
    "save_name",
    "uuid",
    "downloaded",
    "is_boundary",
    "xrun_cs_name",
    "xrun_data",
    "strategy",
}
EXPECTED_MODEL_SHAPES = (
    (2, 16, 32, 11264),
    (1, 16, 32, 11264),
    (1, 32, 32, 65536),
    (4, 96, 96, 128),
    (1, 32, 32, 160),
    (6, 6, 6, 1084),
    (1, 12, 12, 1084),
    (1, 96, 96, 8192),
    (1, 96, 96, 16384),
    (1, 8, 24, 32768),
)
SPEC_METADATA_FIELDS = {
    "case_key",
    "case_id",
    "seed",
    "suite",
    "distribution",
    "expected_tiling_key",
    "performance_baseline_us",
    "performance_target_us",
    "mss_profile",
}


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


def _read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _logical_input_name(item) -> str:
    if isinstance(item, list):
        if not item:
            raise AssertionError("attrs input must not be empty")
        names = {entry["name"] for entry in item}
        if len(names) != 1:
            raise AssertionError(f"attrs input mixes names: {sorted(names)}")
        return item[0]["name"]
    return item["name"]


def _find_forbidden_keys(value, path: str = "root") -> list[str]:
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"marker", "case_spec"}:
                findings.append(f"{path}.{key}")
            findings.extend(_find_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return findings


def _normalized_case(case: dict) -> dict:
    normalized = deepcopy(case)
    normalized.pop("id", None)
    return normalized


def _cases_by_name(cases: list[dict]) -> dict[str, dict]:
    indexed = {case["name"]: _normalized_case(case) for case in cases}
    if len(indexed) != len(cases):
        raise AssertionError("case names must be unique within a suite")
    return indexed


def _spec_structure(spec: dict) -> str:
    structure = {
        key: value
        for key, value in spec.items()
        if key not in SPEC_METADATA_FIELDS
    }
    return json.dumps(structure, sort_keys=True, separators=(",", ":"))


class TestDirectInputJsonContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = {
            suite: _read_json(path) for suite, path in JSON_PATHS.items()
        }

    def test_suite_sizes(self):
        self.assertEqual(len(self.payloads["accuracy"]), 200)
        self.assertEqual(len(self.payloads["perf"]), 10)
        self.assertEqual(len(self.payloads["mss"]), 432)

    def test_all_cases_use_the_21_direct_inputs(self):
        expected_inputs = list(GENERATOR.EXPECTED_INPUTS)
        self.assertEqual(len(expected_inputs), 21)
        for suite, cases in self.payloads.items():
            for case in cases:
                with self.subTest(suite=suite, case=case["name"]):
                    self.assertEqual(set(case), EXPECTED_CASE_FIELDS)
                    self.assertEqual(len(case["inputs"]), 21)
                    self.assertEqual(
                        [_logical_input_name(item) for item in case["inputs"]],
                        expected_inputs,
                    )
                    self.assertEqual(_find_forbidden_keys(case), [])

    def test_frozen_json_matches_generator_by_name(self):
        expected_specs = {
            "accuracy": list(GENERATOR.build_accuracy_specs()),
            "perf": list(GENERATOR.build_perf_specs()),
            "mss": list(GENERATOR.build_mss_specs()),
        }
        for suite, specs in expected_specs.items():
            with self.subTest(suite=suite):
                expected = GENERATOR._payloads(specs)
                self.assertEqual(
                    _cases_by_name(self.payloads[suite]),
                    _cases_by_name(expected),
                )


class TestAccuracyCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = list(GENERATOR.build_accuracy_specs())

    def test_structures_seeds_and_distributions_are_unique(self):
        self.assertEqual(len(self.specs), 200)
        self.assertEqual(len({spec["case_key"] for spec in self.specs}), 200)
        self.assertEqual(len({_spec_structure(spec) for spec in self.specs}), 200)
        self.assertEqual(len({spec["seed"] for spec in self.specs}), 200)
        self.assertEqual(
            Counter(spec["distribution"] for spec in self.specs),
            {"uniform": 100, "normal": 100},
        )

    def test_all_legal_template_attribute_pairs_are_covered(self):
        matrix_specs = list(GENERATOR._template_specs("mss"))
        legal_pairs = {
            pair
            for spec in matrix_specs
            for pair in GENERATOR._signature_pairs(
                GENERATOR._template_signature(
                    spec, include_output_mode=True
                )
            )
        }
        covered_pairs = {
            pair
            for spec in self.specs
            for pair in GENERATOR._signature_pairs(
                GENERATOR._template_signature(
                    spec, include_output_mode=True
                )
            )
        }
        self.assertEqual(len(legal_pairs), 123)
        self.assertEqual(covered_pairs, legal_pairs)


class TestPerformanceCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accuracy = list(GENERATOR.build_accuracy_specs())
        cls.perf = list(GENERATOR.build_perf_specs())

    def test_model_shapes_and_logical_ids(self):
        self.assertEqual(
            [
                (spec["B"], spec["HK"], spec["HV"], spec["T"])
                for spec in self.perf
            ],
            list(EXPECTED_MODEL_SHAPES),
        )
        self.assertEqual(
            [spec["case_id"] for spec in self.perf], list(range(56, 66))
        )

    def test_performance_cases_are_accuracy_cases(self):
        accuracy_by_key = {spec["case_key"]: spec for spec in self.accuracy}
        self.assertEqual(len(accuracy_by_key), 200)
        for spec in self.perf:
            with self.subTest(case=spec["case_key"]):
                self.assertIn(spec["case_key"], accuracy_by_key)
                self.assertEqual(spec, accuracy_by_key[spec["case_key"]])

        accuracy_json = _cases_by_name(_read_json(JSON_PATHS["accuracy"]))
        perf_json = _cases_by_name(_read_json(JSON_PATHS["perf"]))
        self.assertLessEqual(perf_json.keys(), accuracy_json.keys())
        for name, case in perf_json.items():
            self.assertEqual(case, accuracy_json[name])


class TestMssTilingCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = list(GENERATOR.build_mss_specs())

    def test_all_reachable_tiling_keys_are_unique(self):
        keys = [spec["expected_tiling_key"] for spec in self.specs]
        self.assertEqual(len(keys), 432)
        self.assertEqual(len(set(keys)), 432)
        self.assertEqual(
            keys, [GENERATOR._expected_tiling_key(spec) for spec in self.specs]
        )

    def test_template_axes_and_execution_profiles_are_covered(self):
        axes = GENERATOR.TEMPLATE_MATRIX["axes"]
        expected_signatures = set(
            product(
                axes["gate_dtype"],
                axes["beta_dtype"],
                axes["norm"],
                axes["beta_mode"],
                axes["gate_mode"],
                axes["use_exp2"],
                axes["output_mode"],
            )
        )
        signatures = {
            GENERATOR._template_signature(spec, include_output_mode=True)
            for spec in self.specs
        }
        self.assertEqual(signatures, expected_signatures)
        self.assertEqual(
            Counter(spec["backward_mode"] for spec in self.specs),
            {"none": 144, "recompute": 144, "save": 144},
        )
        self.assertEqual(
            {spec["layout"] for spec in self.specs},
            {"BNSD", "BSND", "NTD", "TND"},
        )
        self.assertEqual(
            {spec["mss_profile"] for spec in self.specs},
            {name for name, _ in GENERATOR.TEMPLATE_MATRIX["profiles"]},
        )
        self.assertTrue(any(spec["cu_seqlens"] is not None for spec in self.specs))
        self.assertTrue(any(spec["explicit_chunk_indices"] for spec in self.specs))


class TestYamlAndExecutorContract(unittest.TestCase):
    def test_yaml_declares_direct_inputs_and_supported_standard(self):
        source = YAML_PATH.read_text(encoding="utf-8")
        dtype_numbers = re.search(r"^dtype_numbers:\s*(\d+)\s*$", source, re.M)
        self.assertIsNotNone(dtype_numbers)
        self.assertEqual(int(dtype_numbers.group(1)), 100)
        self.assertRegex(
            source,
            r"(?m)^standard:\s*$\n^  acc:\s*mixed_tolerance_bm\s*$",
        )
        input_names = re.findall(
            r"(?m)^  - name:\s*([A-Za-z0-9_]+)\s*$", source
        )
        self.assertEqual(input_names, list(GENERATOR.EXPECTED_INPUTS))
        self.assertNotIn("case_spec", source)
        self.assertNotIn("marker", source)

    def test_executor_uses_only_the_stable_fla_npu_path(self):
        source = EXECUTOR_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "from fla_npu.ops.ascendc import chunk_kda_fwd_prepare", source
        )
        self.assertNotIn("torch_npu", source)
        self.assertNotIn("torch.ops", source)
        self.assertNotIn("case_spec", source)
        self.assertNotIn("marker", source)

    def test_no_output_path_returns_before_sync_finite_and_d2h(self):
        source = EXECUTOR_PATH.read_text(encoding="utf-8")
        method_match = re.search(
            r"(?ms)^    def __call__\(.*?(?=^    def export_custom_data\()",
            source,
        )
        self.assertIsNotNone(method_match)
        method_source = method_match.group(0)
        validation = method_source.index("_validate_output_contract(")
        no_output = method_source.index("if not with_output:")
        early_return = method_source.index("return None", no_output)
        synchronize = method_source.index("torch.npu.synchronize()")
        finite = method_source.index("torch.isfinite(")
        self.assertLess(validation, no_output)
        self.assertLess(no_output, early_return)
        self.assertLess(early_return, synchronize)
        self.assertLess(early_return, finite)
        for d2h_token in (".cpu(", '.to("cpu")', ".to('cpu')"):
            for match in re.finditer(re.escape(d2h_token), method_source):
                self.assertGreater(match.start(), early_return)


if __name__ == "__main__":
    unittest.main()
