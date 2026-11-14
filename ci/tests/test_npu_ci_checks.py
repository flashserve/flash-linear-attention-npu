"""Exercise branch check profiles as data, including legacy release branches."""

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKS_SCRIPT = ROOT / ".github/scripts/npu_ci_checks.js"
DEFINITION_PATH = ".github/scripts/npu_ci_checks.json"
LEGACY_PATH = ".github/workflows/npu-ci-default-status.yml"
DEFINITION = json.loads((ROOT / DEFINITION_PATH).read_text(encoding="utf-8"))


class CiChecksTest(unittest.TestCase):
    def read_branch(self, **options):
        wrapper = r"""
const settings = JSON.parse(process.env.TEST_OPTIONS);
const checks = require(process.env.NPU_CI_CHECKS_SCRIPT);
const calls = [];
const files = settings.files || {};
const github = {rest:{repos:{getContent:async (request) => {
  calls.push(request);
  const file = files[request.path];
  if (!file || file.status) {
    const status = file?.status || 404;
    throw Object.assign(new Error(`HTTP ${status}`), {status});
  }
  if (file.data) return {data:file.data};
  return {data:{type:'file',encoding:'base64',content:Buffer.from(file.text).toString('base64')}};
}}}};
const context = {repo:{owner:'example',repo:'repo'}};
const pr = {base:{ref:'release-v2',sha:settings.base_sha ?? 'c'.repeat(40)}};
(async () => {
  try {
    const contexts = await checks.readBranchContexts({github,context,pr});
    process.stdout.write(JSON.stringify({contexts,calls,executed:globalThis.branchCodeExecuted === true}));
  } catch (error) {
    process.stdout.write(JSON.stringify({error:error.message,calls}));
  }
})();
"""
        env = os.environ.copy()
        env.update(TEST_OPTIONS=json.dumps(options), NPU_CI_CHECKS_SCRIPT=str(CHECKS_SCRIPT))
        completed = subprocess.run(["node"], input=wrapper, text=True, encoding="utf-8",
                                   capture_output=True, cwd=ROOT, env=env, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_new_branch_reads_profile_at_fixed_base_sha(self):
        profile = {**DEFINITION, "required": [
            {**entry, "context": f"NPU CI / release-v2 / {index}"}
            for index, entry in enumerate(DEFINITION["required"][:2], 1)
        ]}
        result = self.read_branch(files={DEFINITION_PATH: {"text": json.dumps(profile)}})
        self.assertEqual(result["contexts"], [entry["context"] for entry in profile["required"]])
        self.assertEqual(result["calls"], [{"owner":"example", "repo":"repo",
                                          "path":DEFINITION_PATH, "ref":"c" * 40}])

    def test_old_branch_falls_back_to_literal_yaml_contexts_without_executing_code(self):
        legacy = """
script: |
  globalThis.branchCodeExecuted = true;
  const defaultContexts = [
    'NPU CI / legacy / 01 build',
    'NPU CI / legacy / 02 test',
    'NPU CI / legacy / 01 build',
  ];
"""
        result = self.read_branch(files={LEGACY_PATH: {"text": legacy}})
        self.assertEqual(result["contexts"], ["NPU CI / legacy / 01 build", "NPU CI / legacy / 02 test"])
        self.assertFalse(result["executed"])
        self.assertEqual([call["path"] for call in result["calls"]], [DEFINITION_PATH, LEGACY_PATH])
        self.assertTrue(all(call["ref"] == "c" * 40 for call in result["calls"]))

    def test_only_missing_profile_uses_legacy_fallback(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                result = self.read_branch(files={DEFINITION_PATH:{"status":status},
                                                 LEGACY_PATH:{"text":"'NPU CI / ignored'"}})
                self.assertIn(f"HTTP {status}", result["error"])
                self.assertEqual(len(result["calls"]), 1)

    def test_invalid_json_or_schema_does_not_fall_back(self):
        profiles = ["{", "null", "{}", json.dumps({"required":[]}),
                    json.dumps({"required":[{"context":"other check"}]}),
                    json.dumps({"required":[{"context":"NPU CI / check "}]}),
                    json.dumps({"required":[{"context":"NPU CI / check"}] * 2})]
        for profile in profiles:
            with self.subTest(profile=profile):
                result = self.read_branch(files={DEFINITION_PATH:{"text":profile},
                                                 LEGACY_PATH:{"text":"'NPU CI / ignored'"}})
                self.assertIn("error", result)
                self.assertEqual(len(result["calls"]), 1)

    def test_profile_must_be_an_api_file(self):
        for data in ([], {"type":"dir", "content":"e30="}, {"type":"file"}):
            with self.subTest(data=data):
                result = self.read_branch(files={DEFINITION_PATH:{"data":data}})
                self.assertIn("error", result)
                self.assertEqual(len(result["calls"]), 1)

    def test_missing_or_unrecognized_legacy_profile_fails(self):
        for files in ({}, {LEGACY_PATH:{"text":"name: no NPU CI profile"}},
                      {LEGACY_PATH:{"status":403}}):
            with self.subTest(files=files):
                result = self.read_branch(files=files)
                self.assertIn("error", result)
                self.assertEqual(len(result["calls"]), 2)

    def test_mutable_or_missing_base_ref_cannot_be_read(self):
        for sha in ("", "release-v2", "c" * 39, "c" * 41, "z" * 40):
            with self.subTest(sha=sha):
                result = self.read_branch(base_sha=sha)
                self.assertIn("error", result)
                self.assertEqual(result["calls"], [])


if __name__ == "__main__":
    unittest.main()
