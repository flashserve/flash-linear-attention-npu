"""Check essential Actions wiring; behavior belongs in the runtime test modules."""

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def triggers(workflow):
    # PyYAML's YAML 1.1 parser reads the Actions key `on` as True.
    return workflow.get("on", workflow.get(True))


class WorkflowConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = load_workflow("ci.yml")
        cls.dispatch = load_workflow("npu-ci-comment-dispatch.yml")
        cls.device = cls.ci["jobs"]["ascend-npu"]
        cls.steps = cls.device["steps"]

    def test_entrypoints_and_secret_boundary(self):
        self.assertEqual(set(triggers(self.ci)), {"workflow_dispatch"})
        inputs = triggers(self.ci)["workflow_dispatch"]["inputs"]
        self.assertTrue({"pr_number", "requested_by", "workflow_source", "expected_head_sha",
                         "expected_base_ref", "expected_base_sha", "expected_workflow_sha"} <= set(inputs))
        self.assertEqual(triggers(self.dispatch)["issue_comment"]["types"], ["created"])
        self.assertEqual(self.dispatch["permissions"]["actions"], "write")
        self.assertEqual(self.dispatch["permissions"]["contents"], "write")
        self.assertEqual(self.dispatch["jobs"]["dispatch"]["environment"], "npu-ci-dispatch")
        self.assertTrue(all("environment" not in job for job in self.ci["jobs"].values()))
        # A duplicate request must not cancel the already-running device jobs.
        self.assertFalse(self.ci["concurrency"]["cancel-in-progress"])

    def test_hosted_control_checkout_uses_fixed_workflow_revision(self):
        jobs = [self.dispatch["jobs"]["dispatch"], self.ci["jobs"]["prepare"],
                self.ci["jobs"]["finalize"]]
        for job in jobs:
            with self.subTest(job=job["name"]):
                checkout = next(step for step in job["steps"]
                                if step.get("uses", "").startswith("actions/checkout@"))
                options = checkout["with"]
                self.assertEqual(options["ref"], "${{ github.workflow_sha }}")
                self.assertEqual(options["path"], "ci-control")
                self.assertFalse(options["persist-credentials"])
                for path in options["sparse-checkout"].splitlines():
                    self.assertTrue((ROOT / path).is_file(), path)

    def test_two_platforms_and_native_a5_image_policy(self):
        slots = self.device["strategy"]["matrix"]["include"]
        platforms = {slot["platform"]: slot for slot in slots}
        self.assertEqual(len(slots), 2)
        self.assertEqual(set(platforms), {"a2", "a5"})
        self.assertTrue(self.device["strategy"]["fail-fast"])
        self.assertEqual(self.device["runs-on"], "${{ fromJSON(matrix.runner_labels) }}")
        self.assertEqual({label.lower() for label in json.loads(platforms["a5"]["runner_labels"])},
                         {"self-hosted", "linux", "npu", "ascend950", "flash-linear-attention-npu-a5"})
        for platform, soc, dockerfile, preloaded in (
                ("a2", "ascend910b", "ci/Dockerfile", "false"),
                ("a5", "ascend950", "ci/Dockerfile.ascend950", "true")):
            with self.subTest(platform=platform):
                slot = platforms[platform]
                self.assertEqual((slot["soc"], slot["dockerfile"], slot["require_preloaded_image"]),
                                 (soc, dockerfile, preloaded))
                self.assertTrue((ROOT / dockerfile).is_file())
        for variable, field in (("CI_IMAGE", "image"), ("CI_DOCKERFILE", "dockerfile"),
                                ("CI_REQUIRE_PRELOADED_IMAGE", "require_preloaded_image")):
            self.assertEqual(self.device["env"][variable], "${{ matrix." + field + " }}")

    def test_device_source_checkout_and_offline_guard_order(self):
        checkouts = [step for step in self.steps if step.get("uses", "").startswith("actions/checkout@")]
        self.assertEqual(len(checkouts), 3)
        self.assertEqual([step["id"] for step in checkouts[:2]], ["checkout_pr_1", "checkout_pr_2"])
        self.assertEqual(checkouts[2]["if"],
                         "steps.checkout_pr_1.outcome == 'failure' && steps.checkout_pr_2.outcome == 'failure'")
        for checkout in checkouts:
            self.assertEqual(checkout["with"]["ref"], "${{ needs.prepare.outputs.base_sha }}")
            self.assertEqual(checkout["with"]["path"], "repo")
            self.assertNotIn("matrix.platform", checkout.get("if", ""))
        first = checkouts[0]
        self.assertNotIn("if", first)
        self.assertEqual(first["env"]["no_proxy"], "*")
        for name in ("http_proxy", "https_proxy", "all_proxy", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
            self.assertEqual(first["env"][name], "")
        proxy = next(step for step in self.steps if step.get("id") == "enable_proxy")
        self.assertEqual(proxy["if"], "steps.checkout_pr_1.outcome == 'failure'")
        self.assertLess(self.steps.index(first), self.steps.index(proxy))
        self.assertLess(self.steps.index(proxy), self.steps.index(checkouts[1]))
        merge = next(index for index, step in enumerate(self.steps) if step.get("name") == "现场合并 PR 代码")
        guard = next(index for index, step in enumerate(self.steps) if step.get("id") == "validate_pr")
        run = next(index for index, step in enumerate(self.steps) if "bash ci/run_ci_container.sh" in step.get("run", ""))
        self.assertLess(merge, guard)
        self.assertLess(guard, run)
        self.assertEqual(self.steps[guard]["working-directory"], "repo")
        self.assertFalse(any(step.get("uses", "").startswith("actions/github-script@") for step in self.steps))

    def test_device_storage_and_execution_scope(self):
        env = self.device["env"]
        self.assertEqual((env["CI_JOBS"], env["CI_CPACK_JOBS"]), ("16", "16"))
        identity = "${{ matrix.platform }}-${{ github.run_id }}-${{ github.run_attempt }}"
        self.assertEqual(env["CI_CONTAINER_NAME"], f"fla-npu-ci-{identity}")
        self.assertEqual(env["CI_HOST_TMPDIR"], f"/home/huangjunzhe/ci/tmp/container-{identity}")
        self.assertEqual(env["CI_STAGE"], "${{ needs.prepare.outputs.ops == '' && 'all' || 'opp-package' }}")

    def test_reports_are_uploaded_after_cleanup_and_have_failure_fallback(self):
        cleanup = next(index for index, step in enumerate(self.steps) if step.get("name", "").startswith("清理 "))
        upload = next(index for index, step in enumerate(self.steps)
                      if step.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertIn("always()", self.steps[cleanup]["if"])
        self.assertLess(cleanup, upload)
        paths = self.steps[upload]["with"]["path"]
        for variable in ("CI_ACCURACY_REPORT_FILE", "CI_DIAGNOSTIC_FILE", "CI_STAGE_REPORT_FILE", "CI_SUMMARY_FILE"):
            self.assertIn(variable, paths)
        self.assertNotIn("CI_RAW_LOG_FILE", paths)
        fallback = next(step for step in self.ci["jobs"]["finalize"]["steps"]
                        if step.get("name") == "状态发布异常时收敛全部分项")
        self.assertIn("statuses_published != 'true'", fallback["if"])

    def test_contract_tests_use_hosted_runners(self):
        workflow = load_workflow("ci-contract-tests.yml")
        job = workflow["jobs"]["contract-tests"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("unittest discover -s ci/tests", commands)
        self.assertIn("bash ci/tests/test_run_checks_stage_pipeline.sh", commands)
        self.assertIn("bash ci/tests/test_detect_npu.sh", commands)
        self.assertNotIn("-s ci/tests", (ROOT / "ci/run_checks.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
