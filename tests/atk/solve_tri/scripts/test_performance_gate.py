"""Synthetic unit fixtures test the gate, never stand in for NPU acceptance."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import openpyxl

from check_performance import DEVICE_NAMES, check_matrix, contract


class PerformanceGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {"platforms": {}}
        self.specs = {
            208: {"B": 1, "T": 11264, "H": 32, "chunk_size": 64, "dtype": "bf16",
                  "layout": "bsnd", "seed": 208},
            209: {"B": 1, "T": 65536, "H": 32, "chunk_size": 64, "dtype": "bf16",
                  "layout": "tnd", "seqlens": [1024] * 64, "seed": 209},
        }
        for soc in ("A2", "A3", "A5"):
            directory = self.root / soc
            directory.mkdir()
            (directory / "build.exit").write_text("0")
            (directory / "build.log").write_text("unit-test fixture")
            (directory / "cann_version.info").write_text("unit-test")
            (directory / "worker.json").write_text(json.dumps({
                "pid": 1234, "environment": {"ASCEND_HOME_PATH": "/unit-test/cann"},
                "hostnames": ["worker"],
                "mapped_files_sha256": {"/unit-test/libcust_opapi.so": "a" * 64},
                "kernel_assets_sha256": {"/unit-test/solve_tri/SolveTri_bf16.o": "c" * 64}}))
            accuracy_worker = json.loads((directory / "worker.json").read_text())
            accuracy_worker["pid"] = 5678
            (directory / "accuracy_worker.json").write_text(json.dumps(accuracy_worker))
            self.report(directory / "performance.xlsx", accuracy=False)
            self.report(directory / "accuracy_task/report/accuracy.xlsx", accuracy=True)
            self.accuracy_log(directory / "accuracy_task/log/atk.log")
            entry = {"code_id": "synthetic-test", "cann_version": "unit-test", "device_name": DEVICE_NAMES[soc],
                     "opapi_sha256": "a" * 64, "case_ids": {"dense": 208, "varlen": 209}, "node": "npu_dut"}
            for field, name in {"build_exit": "build.exit", "build_log": "build.log", "cann_version_file": "cann_version.info", "runtime_provenance": "worker.json", "accuracy_provenance": "accuracy_worker.json",
                                "performance_report": "performance.xlsx", "accuracy_report": "accuracy_task/report/accuracy.xlsx", "accuracy_log": "accuracy_task/log/atk.log",
                                "profile_root": "profiles", "memory_report": "memory.json"}.items():
                entry[field] = f"{soc}/{name}"
            self.manifest["platforms"][soc] = entry
            records = {}
            for index in self.specs:
                location = directory / "profiles" / "npu_dut" / "unit-test" / str(index) / "worker_1234_profile"
                location.mkdir(parents=True)
                self.profile(location / "op_summary.csv", index)
                spec = contract(self.case(index))
                output = math.prod(spec["shape"]) * 2
                records[str(index)] = {"contract": spec, "baseline_bytes": output,
                                       "peak_bytes": output * 2 + 1024, "output_bytes": output,
                                       "device_name": DEVICE_NAMES[soc],
                                       "device_id": 3, "cann_home": "/unit-test/cann",
                                       "warmup_count": 3, "measurement_count": 1,
                                       "hostnames": ["worker"],
                                       "mapped_files_sha256": {"/unit-test/libcust_opapi.so": "a" * 64},
                                       "kernel_assets_sha256": {"/unit-test/solve_tri/SolveTri_bf16.o": "c" * 64}}
            (directory / "memory.json").write_text(json.dumps({"method": "atk_memory_peak_allocated", "cases": records}))

    def tearDown(self):
        self.temporary.cleanup()

    def case(self, index, api_type="executor_solve_tri_a2"):
        return {"api_type": api_type,
                "inputs": [{"name": "case_spec", "range_values": json.dumps(self.specs[index])}]}

    def report(self, path, accuracy):
        path.parent.mkdir(parents=True, exist_ok=True)
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "statistic"
        sheet.append(["\u7f16\u53f7", "\u8fd0\u884c\u7ed3\u679c", "\u7528\u4f8bjson\u4fe1\u606f",
                      "npu_dut_Device\u6027\u80fd\uff08us\uff09", "cpu_golden_\u7cbe\u5ea6\u901a\u8fc7"])
        for index in self.specs:
            sheet.append([index, "SUCCESS", json.dumps(self.case(index)), 100.0, True if accuracy else None])
        book.save(path)
        book.close()

    def accuracy_log(self, path, pid=5678, device=3, ids=(208, 209), report="accuracy.xlsx", task="accuracy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"[INFO] start atk task: namespace(task=['{task}'])"]
        for index in ids:
            prefix = f"[INFO] [127.0.0.1:9090@device_run_{device}] [{pid}] [trace.py:736] [case {index}]"
            lines.extend([f"{prefix}[RUN OPP TASK] Start Execute OPP on ['{task}']",
                          f"{prefix} SUCCESS run NPU OPP TASK."])
        lines.append(f"save result excel file: /remote/archive/accuracy_task/report/{report}")
        path.write_text("\n".join(lines))

    def profile(self, path, index, value=100.0, samples=20):
        shape = [1, 11264, 32, 64] if index == 208 else [65536, 32, 64]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["OP Type", "Task Duration(us)", "Input Shapes", "Input Data Types", "Device_id"])
            for _ in range(samples):
                writer.writerow(["SolveTri", value, ",".join(map(str, shape)), "DT_BF16", "3"])

    def gate(self, limit=1.0):
        return check_matrix(self.manifest, self.root, limit)

    def test_complete_matrix_passes(self):
        result = self.gate()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(len(result["comparisons"]), 4)

    def test_a2_generator_is_recorded_from_api_type(self):
        self.assertEqual(contract(self.case(208))["generator"], "executor_solve_tri_a2")

    def test_legacy_generator_is_not_relabelled_as_a2(self):
        legacy = contract(self.case(208, api_type="executor_solve_tri"))
        self.assertEqual(legacy["generator"], "executor_solve_tri")
        self.assertNotEqual(legacy, contract(self.case(208)))

    def test_unknown_generator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown SolveTri input generator"):
            contract(self.case(208, api_type="executor_other_operator"))

    def test_legacy_memory_does_not_match_a2_inputs(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["contract"] = contract(self.case(208, api_type="executor_solve_tri"))
        path.write_text(json.dumps(data))
        result = self.gate()
        self.assertFalse(result["passed"])
        self.assertTrue(any("Memory and performance input contracts differ" in error for error in result["errors"]))

    def test_a2_scope_does_not_require_a3_a5(self):
        self.manifest["platforms"] = {"A2": self.manifest["platforms"]["A2"]}
        result = check_matrix(self.manifest, self.root, 1.0, ("A2",))
        self.assertTrue(result["passed"], result["errors"])
        self.assertIsNone(result["max_duration_ratio"])
        self.assertEqual(result["comparisons"], [])

    def test_latency_ratio_fails(self):
        result = self.gate(0.5)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["errors"]), 4)

    def test_missing_a5_fails(self):
        del self.manifest["platforms"]["A5"]
        self.assertFalse(self.gate()["passed"])

    def test_no_memory_evidence_fails(self):
        (self.root / "A2/memory.json").unlink()
        self.assertFalse(self.gate()["passed"])

    def test_memory_above_limit_fails(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["peak_bytes"] += 200_000_000
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_input_seed_mismatch_fails(self):
        self.specs[208]["seed"] += 1
        self.report(self.root / "A5/performance.xlsx", accuracy=False)
        self.report(self.root / "A5/accuracy_task/report/accuracy.xlsx", accuracy=True)
        self.assertFalse(self.gate()["passed"])

    def test_unloaded_operator_artifact_fails(self):
        self.manifest["platforms"]["A2"]["opapi_sha256"] = "b" * 64
        self.assertFalse(self.gate()["passed"])

    def test_profile_xlsx_disagreement_fails(self):
        path = self.root / "A2/profiles/npu_dut/unit-test/208/worker_1234_profile/op_summary.csv"
        self.profile(path, 208, value=101.0)
        self.assertFalse(self.gate()["passed"])

    def test_too_few_profile_samples_fails(self):
        path = self.root / "A2/profiles/npu_dut/unit-test/208/worker_1234_profile/op_summary.csv"
        self.profile(path, 208, samples=19)
        self.assertFalse(self.gate()["passed"])

    def test_missing_accuracy_comparison_fails(self):
        self.report(self.root / "A2/accuracy_task/report/accuracy.xlsx", accuracy=False)
        self.assertFalse(self.gate()["passed"])

    def test_wrong_platform_device_fails(self):
        self.manifest["platforms"]["A3"]["device_name"] = DEVICE_NAMES["A2"]
        self.assertFalse(self.gate()["passed"])

    def test_a2_a3_code_version_mismatch_fails(self):
        self.manifest["platforms"]["A3"]["code_id"] = "different-source"
        self.assertFalse(self.gate()["passed"])

    def test_wrong_performance_worker_provenance_fails(self):
        path = self.root / "A2/worker.json"
        data = json.loads(path.read_text())
        data["pid"] = 9999
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_memory_device_mismatch_fails(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["device_id"] = 4
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_memory_hostname_mismatch_fails(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["hostnames"] = ["different-host"]
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_memory_kernel_change_fails(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["kernel_assets_sha256"]["/unit-test/solve_tri/SolveTri_bf16.o"] = "d" * 64
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_missing_kernel_fingerprint_fails(self):
        path = self.root / "A2/worker.json"
        data = json.loads(path.read_text())
        del data["kernel_assets_sha256"]
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_cann_version_mismatch_fails(self):
        self.manifest["platforms"]["A2"]["cann_version"] = "different-version"
        self.assertFalse(self.gate()["passed"])

    def test_missing_memory_warmup_fails(self):
        path = self.root / "A2/memory.json"
        data = json.loads(path.read_text())
        data["cases"]["208"]["warmup_count"] = 0
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_accuracy_kernel_change_fails(self):
        path = self.root / "A2/accuracy_worker.json"
        data = json.loads((self.root / "A2/worker.json").read_text())
        data["kernel_assets_sha256"]["/unit-test/solve_tri/SolveTri_bf16.o"] = "d" * 64
        path.write_text(json.dumps(data))
        self.manifest["platforms"]["A2"]["accuracy_provenance"] = "A2/accuracy_worker.json"
        self.assertFalse(self.gate()["passed"])

    def test_profile_hostname_mismatch_fails(self):
        path = self.root / "A2/worker.json"
        data = json.loads(path.read_text())
        data["hostnames"] = ["different-host"]
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_wrong_accuracy_worker_provenance_fails(self):
        path = self.root / "A2/accuracy_worker.json"
        data = json.loads(path.read_text())
        data["pid"] = 9999
        path.write_text(json.dumps(data))
        self.assertFalse(self.gate()["passed"])

    def test_missing_accuracy_log_fails(self):
        (self.root / "A2/accuracy_task/log/atk.log").unlink()
        self.assertFalse(self.gate()["passed"])

    def test_accuracy_log_other_report_fails(self):
        self.accuracy_log(self.root / "A2/accuracy_task/log/atk.log", report="other.xlsx")
        self.assertFalse(self.gate()["passed"])

    def test_accuracy_log_missing_case_fails(self):
        self.accuracy_log(self.root / "A2/accuracy_task/log/atk.log", ids=(208,))
        self.assertFalse(self.gate()["passed"])

    def test_accuracy_log_other_device_fails(self):
        self.accuracy_log(self.root / "A2/accuracy_task/log/atk.log", device=4)
        self.assertFalse(self.gate()["passed"])

    def test_performance_log_cannot_replace_accuracy_log(self):
        self.accuracy_log(self.root / "A2/accuracy_task/log/atk.log", task="performance_device")
        self.assertFalse(self.gate()["passed"])

    def test_other_task_log_cannot_replace_accuracy_log(self):
        source = self.root / "A2/accuracy_task/log/atk.log"
        target = self.root / "A2/other_task/log/atk.log"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        self.manifest["platforms"]["A2"]["accuracy_log"] = "A2/other_task/log/atk.log"
        self.assertFalse(self.gate()["passed"])


if __name__ == "__main__":
    unittest.main()
