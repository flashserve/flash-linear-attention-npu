"""PR path selection against a real Git diff and the published smoke inventory."""

import json
import runpy
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests/atk/smoke"


class AtkSmokeSelectionTests(unittest.TestCase):
    def test_registered_roots_are_independent_operator_directories(self):
        for path in sorted((SMOKE / "cases").glob("*/suite.json")):
            op = path.parent.name
            root = json.loads(path.read_bytes())["source_root"]
            with self.subTest(op=op):
                self.assertTrue(root.startswith("fla/ops/ascendc/"))
                self.assertEqual(Path(root).name, op)
                self.assertNotIn("op_kernel", Path(root).parts)
                self.assertTrue((REPO / root / "op_host").is_dir())

    def test_cv_cases_match_operator_generator_on_both_platforms(self):
        for op in ("chunk_fwd_h", "chunk_kda_bwd_recompute"):
            folder = SMOKE / "cases" / op
            generator = runpy.run_path(str(folder / f"gen_{op}.py"))
            platform_cases = {}
            for arch in ("A2", "A5"):
                with self.subTest(op=op, platform=arch):
                    groups = json.loads((folder / arch / "cases.json").read_bytes())
                    group, = (g for g in groups if g["group"] == f"{op}-{arch}-cv200")
                    cases = group["cases"]
                    platform_cases[arch] = cases
                    self.assertEqual([case["id"] for case in cases], list(range(200)))
                    self.assertEqual(group["output"], "native")
                    self.assertTrue(group["require_fp64_golden"])
                    self.assertTrue(all("cv_fused_double_benchmark" in c["standard"]["acc"] for c in cases))
                    if op == "chunk_fwd_h":
                        self.assertEqual(cases, generator["_payloads"](generator["build_accuracy_specs"]()))
                    else:
                        for case in cases:
                            spec = next(i["range_values"] for i in case["inputs"] if i["name"] == "case_spec")
                            self.assertEqual(json.loads(spec), generator["_spec"](case["id"]))
            self.assertEqual(platform_cases["A2"], platform_cases["A5"])
