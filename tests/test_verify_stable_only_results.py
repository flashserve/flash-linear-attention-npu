"""Fail-closed before/after comparison and instrumentation regression tests."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_stable_only_results", ROOT / "scripts/verify_stable_only_results.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def valid_report():
    tensor = {"tensor": True, "sha256": "0" * 64, "finite": True,
              "shape": [1], "dtype": "torch.float32", "requires_grad": False,
              "version": 0, "device": "npu:0", "stride": [1], "storage_offset": 0}
    inputs = {"kind": "tuple", "items": [tensor]}
    return {
        "format": 2, "script": "test.py", "script_sha256": "script-hash",
        "arguments": ["--device", "0"], "torch": "2.7.1", "launcher": "/different/path/lib.so",
        "public_ops": ["npu_probe"], "exit_code": 0,
        "signatures_before": {"npu_probe": "(x)"},
        "signatures_after": {"npu_probe": "(x)"},
        "calls": [{"op": "npu_probe", "inputs": inputs,
                   "input_sha256": VERIFY.digest_description(inputs),
                   "outputs": dict(tensor, sha256="1" * 64),
                   "inputs_after": copy.deepcopy(inputs), "finite": True}],
    }


class ComparisonTests(unittest.TestCase):
    def assert_rejected(self, left, right, required=()):
        result = VERIFY.compare_reports(left, right, required)
        self.assertFalse(result["passed"], result)
        self.assertTrue(result["failures"])

    def test_matching_complete_reports_pass(self):
        report = valid_report()
        self.assertTrue(VERIFY.compare_reports(report, copy.deepcopy(report), ("npu_probe",))["passed"])

    def test_empty_and_missing_call_coverage_fail(self):
        for left_count, right_count in ((0, 0), (1, 0), (0, 1), (1, 2)):
            left, right = valid_report(), valid_report()
            left["calls"] *= left_count
            right["calls"] *= right_count
            with self.subTest(left=left_count, right=right_count):
                self.assert_rejected(left, right)

    def test_matching_caught_operator_errors_still_fail(self):
        report = valid_report()
        report["calls"][0]["error"] = {"type": "RuntimeError", "message": "expected by caller"}
        self.assert_rejected(report, copy.deepcopy(report))

    def test_missing_call_fields_fail_even_on_both_sides(self):
        for key in ("op", "inputs", "input_sha256", "outputs", "inputs_after", "finite"):
            with self.subTest(key=key):
                report = valid_report()
                del report["calls"][0][key]
                self.assert_rejected(report, copy.deepcopy(report))

    def test_report_identity_and_environment_changes_fail(self):
        for key, replacement in (("script", "other.py"), ("script_sha256", "changed"),
                                 ("arguments", []), ("torch", "2.8.0"),
                                 ("public_ops", ["npu_other"])):
            with self.subTest(key=key):
                left, right = valid_report(), valid_report()
                right[key] = replacement
                self.assert_rejected(left, right)

    def test_result_mutation_input_and_call_identity_changes_fail(self):
        for key in ("op", "inputs", "input_sha256", "outputs", "inputs_after"):
            with self.subTest(key=key):
                left, right = valid_report(), valid_report()
                right["calls"][0][key] = "changed"
                self.assert_rejected(left, right)

    def test_nonfinite_output_cannot_hide_behind_finite_flag(self):
        for value in (False, None, "true"):
            report = valid_report()
            report["calls"][0]["finite"] = value
            self.assert_rejected(report, copy.deepcopy(report))
        report = valid_report()
        report["calls"][0]["outputs"]["finite"] = False
        self.assert_rejected(report, copy.deepcopy(report))

    def test_stale_input_hash_and_missing_required_operator_fail(self):
        report = valid_report()
        report["calls"][0]["inputs"]["changed"] = True
        self.assert_rejected(report, copy.deepcopy(report))
        self.assert_rejected(valid_report(), valid_report(), ("npu_missing",))

    def test_tensor_digest_and_metadata_must_be_present(self):
        for key in ("sha256", "shape", "dtype", "version", "device", "stride", "storage_offset", "requires_grad"):
            report = valid_report()
            del report["calls"][0]["outputs"][key]
            with self.subTest(key=key):
                self.assert_rejected(report, copy.deepcopy(report))
        report = valid_report()
        report["calls"][0]["outputs"]["sha256"] = "not-a-byte-hash"
        self.assert_rejected(report, copy.deepcopy(report))

    def test_script_error_or_nonzero_exit_fails(self):
        for key, value in (("test_error", {"type": "AssertionError"}), ("exit_code", 1)):
            report = valid_report()
            report[key] = value
            self.assert_rejected(report, copy.deepcopy(report))

    def test_public_signature_drift_fails(self):
        for key in ("signatures_before", "signatures_after"):
            left, right = valid_report(), valid_report()
            right[key]["npu_probe"] = "(x, new_required)"
            self.assert_rejected(left, right)

    def test_missing_report_metadata_and_old_format_fail(self):
        for key in ("format", "script", "script_sha256", "torch", "public_ops", "signatures_before", "signatures_after"):
            report = valid_report()
            del report[key]
            with self.subTest(key=key):
                self.assert_rejected(report, copy.deepcopy(report))

    def test_invalid_json_and_nonobject_report_return_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for content in ("{broken", "[]", "null"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(VERIFY.compare(path, path), 1)


class RecorderTests(unittest.TestCase):
    def fake_modules(self):
        torch = types.ModuleType("torch")
        torch.__version__ = "2.7.1"
        torch.Tensor = type("Tensor", (), {})
        torch.dtype = type("dtype", (), {})
        torch.device = type("device", (), {})
        package = types.ModuleType("fla_npu")
        ops = types.ModuleType("fla_npu.ops")
        ascendc = types.ModuleType("fla_npu.ops.ascendc")
        stable = types.ModuleType("fla_npu.ops.ascendc._stable")
        stable.load = lambda: None
        stable._op = lambda name: object()
        stable._loaded_path = "launcher.so"

        def implementation(x):
            x.append(2)
            return tuple(x)

        stable.npu_probe = implementation
        ascendc._ASCENDC_OPS = ("npu_probe",)
        ascendc.BACKENDS = {}

        @functools.lru_cache(None)
        def resolver(name):
            ascendc.BACKENDS[name] = "stable"
            implementation = getattr(stable, name)

            @functools.wraps(implementation)
            def mutation_contract(*args, **kwargs):
                result = implementation(*args, **kwargs)
                args[0].append(3)  # a side effect after the Stable wrapper
                return result

            return mutation_contract

        ascendc._get_direct_op = resolver

        def make_raw(name):
            def wrapper(*args, **kwargs):
                fn = ascendc._get_direct_op(name)
                setattr(ascendc, name, fn)
                return fn(*args, **kwargs)

            wrapper.__signature__ = inspect.signature(getattr(stable, name))
            return wrapper

        ascendc._make_raw_wrapper = make_raw
        ascendc.npu_probe = ascendc.probe = make_raw("npu_probe")
        package.ops = ops
        ops.ascendc = ascendc
        ascendc._stable = stable
        return {"torch": torch, "fla_npu": package, "fla_npu.ops": ops,
                "fla_npu.ops.ascendc": ascendc, "fla_npu.ops.ascendc._stable": stable}

    def run_record(self, body, modules=None):
        modules = modules or self.fake_modules()
        with tempfile.TemporaryDirectory() as folder:
            script, output = Path(folder) / "test.py", Path(folder) / "record.json"
            script.write_text("# identical test source\n", encoding="utf-8")
            original_path, original_argv = sys.path[:], sys.argv[:]
            with mock.patch.dict(sys.modules, modules), mock.patch.object(VERIFY.runpy, "run_path", side_effect=lambda *a, **k: body(modules)):
                code = VERIFY.record(output, script, ["--test"])
            self.assertEqual(sys.path, original_path)
            self.assertEqual(sys.argv, original_argv)
            return code, json.loads(output.read_text(encoding="utf-8"))

    def test_public_alias_and_direct_stable_calls_are_recorded_once(self):
        def body(modules):
            modules["fla_npu.ops.ascendc"].probe([1])
            modules["fla_npu.ops.ascendc._stable"].npu_probe([4])

        code, report = self.run_record(body)
        self.assertEqual(code, 0)
        self.assertEqual(len(report["calls"]), 2)
        first = report["calls"][0]
        self.assertNotEqual(first["inputs"], first["inputs_after"])
        # The post-wrapper side effect must be captured, not just kernel work.
        self.assertIn("3", json.dumps(first["inputs_after"]))
        self.assertNotIn("3", json.dumps(first["outputs"]))
        self.assertTrue(VERIFY.compare_reports(report, copy.deepcopy(report))["passed"])

    def test_no_calls_and_caught_operator_failure_return_nonzero(self):
        code, report = self.run_record(lambda modules: None)
        self.assertEqual(code, 1)
        self.assertEqual(report["calls"], [])
        modules = self.fake_modules()

        def fail(x):
            raise ValueError("invalid input")

        modules["fla_npu.ops.ascendc._stable"].npu_probe = fail

        def body(modules):
            try:
                modules["fla_npu.ops.ascendc._stable"].npu_probe([1])
            except ValueError:
                pass

        code, report = self.run_record(body, modules)
        self.assertEqual(code, 1)
        self.assertEqual(report["calls"][0]["error"]["type"], "ValueError")

    def test_setup_failure_is_saved_as_failed_report(self):
        modules = self.fake_modules()
        modules["fla_npu.ops.ascendc._stable"].load = mock.Mock(side_effect=RuntimeError("bad launcher"))
        code, report = self.run_record(lambda modules: None, modules)
        self.assertEqual(code, 1)
        self.assertEqual(report["test_error"]["message"], "bad launcher")

    def test_tensor_bytes_and_post_contract_version_are_recorded(self):
        class Tensor(list):
            requires_grad = False
            dtype = "torch.uint8"
            device = "npu:2"
            _version = 0

            @property
            def shape(self):
                return (len(self),)

            def append(self, value):
                super().append(value)
                self._version += 1

            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def reshape(self, *args): return self
            def view(self, *args): return self
            def numpy(self): return self
            def tobytes(self): return bytes(self)
            def is_floating_point(self): return False
            def is_complex(self): return False
            def stride(self): return (1,)
            def storage_offset(self): return 0

        modules = self.fake_modules()
        modules["torch"].Tensor = Tensor
        modules["torch"].uint8 = object()

        def implementation(x):
            x.append(2)
            return x

        modules["fla_npu.ops.ascendc._stable"].npu_probe = implementation
        code, report = self.run_record(lambda m: m["fla_npu.ops.ascendc"].npu_probe(Tensor([1])), modules)
        self.assertEqual(code, 0)
        entry = report["calls"][0]
        initial = entry["inputs"]["items"][0]["items"][0]
        final = entry["inputs_after"]["items"][0]["items"][0]
        self.assertEqual(initial["sha256"], hashlib.sha256(bytes([1])).hexdigest())
        self.assertEqual(final["sha256"], hashlib.sha256(bytes([1, 2, 3])).hexdigest())
        self.assertEqual(entry["outputs"], final)
        self.assertEqual((initial["version"], final["version"]), (0, 2))
        self.assertTrue(VERIFY.compare_reports(report, copy.deepcopy(report))["passed"])

    def test_instrumentation_is_restored_after_success_and_script_failure(self):
        for fail in (False, True):
            modules = self.fake_modules()
            stable = modules["fla_npu.ops.ascendc._stable"]
            ascendc = modules["fla_npu.ops.ascendc"]
            originals = (stable.npu_probe, ascendc.npu_probe, ascendc.probe, ascendc._get_direct_op)

            def body(modules):
                ascendc.npu_probe([1])
                if fail:
                    raise AssertionError("test failed")

            code, _ = self.run_record(body, modules)
            self.assertEqual(code, int(fail))
            self.assertEqual((stable.npu_probe, ascendc.npu_probe, ascendc.probe, ascendc._get_direct_op), originals)


if __name__ == "__main__":
    unittest.main()
