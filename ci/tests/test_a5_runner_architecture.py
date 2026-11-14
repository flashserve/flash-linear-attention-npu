"""Keep one A5 gate schedulable on either native Linux CPU architecture."""

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class A5RunnerArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        cls.job = workflow["jobs"]["ascend-npu"]
        cls.slots = cls.job["strategy"]["matrix"]["include"]
        cls.platforms = {slot["platform"]: slot for slot in cls.slots}

    def labels(self, platform):
        return {label.casefold() for label in json.loads(self.platforms[platform]["runner_labels"])}

    def matches(self, platform, runner_labels):
        # GitHub requires every runs-on label; label matching ignores case.
        return self.labels(platform) <= {label.casefold() for label in runner_labels}

    def test_matrix_keeps_exactly_one_a2_and_one_a5_slot(self):
        self.assertEqual(len(self.slots), 2)
        self.assertEqual(set(self.platforms), {"a2", "a5"})
        self.assertEqual(self.job["runs-on"], "${{ fromJSON(matrix.runner_labels) }}")

    def test_a5_schedules_on_native_x64_and_arm64_runners(self):
        common = {"self-hosted", "Linux", "npu", "ascend950", "flash-linear-attention-npu-a5"}
        for architecture in ("X64", "ARM64"):
            with self.subTest(architecture=architecture):
                self.assertTrue(self.matches("a5", common | {architecture}))

    def test_a5_rejects_a2_and_runners_missing_its_dedicated_labels(self):
        a2 = {"self-hosted", "Linux", "ARM64", "npu", "flash-linear-attention-npu"}
        self.assertFalse(self.matches("a5", a2))
        for architecture in ("X64", "ARM64"):
            a5 = {"self-hosted", "Linux", architecture, "npu", "ascend950", "flash-linear-attention-npu-a5"}
            for missing in ("ascend950", "flash-linear-attention-npu-a5"):
                with self.subTest(architecture=architecture, missing=missing):
                    self.assertFalse(self.matches("a5", a5 - {missing}))

    def test_a2_keeps_its_existing_arm64_runner_and_image_policy(self):
        self.assertEqual(self.labels("a2"), {
            "self-hosted", "linux", "arm64", "npu", "flash-linear-attention-npu",
        })
        self.assertEqual(self.platforms["a2"]["soc"], "ascend910b")
        self.assertEqual(self.platforms["a2"]["image"], "fla-npu-ci:9.1.0-910b")
        self.assertEqual(self.platforms["a2"]["dockerfile"], "ci/Dockerfile")
        self.assertEqual(self.platforms["a2"]["require_preloaded_image"], "false")

    def test_a5_keeps_its_required_preloaded_native_image_policy(self):
        self.assertEqual(self.platforms["a5"]["soc"], "ascend950")
        self.assertEqual(self.platforms["a5"]["image"], "fla-npu-ci:9.1.0-950")
        self.assertEqual(self.platforms["a5"]["dockerfile"], "ci/Dockerfile.ascend950")
        self.assertEqual(self.platforms["a5"]["require_preloaded_image"], "true")
        for variable, field in (("CI_IMAGE", "image"), ("CI_DOCKERFILE", "dockerfile"),
                                ("CI_REQUIRE_PRELOADED_IMAGE", "require_preloaded_image")):
            self.assertEqual(self.job["env"][variable], "${{ matrix." + field + " }}")


if __name__ == "__main__":
    unittest.main()
