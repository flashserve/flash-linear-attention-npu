"""Strict CP boundary UT; lower-layer faults and first errors are test doubles.

Requires real target flags/headers/libraries. Compiles both unchanged L2 and L0
sources. No production fault switch or installed operator replacement is used.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


def compile_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


class CpHostFaultContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = Path(os.environ["DFX_BUILD_DIR"])
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        root = Path(__file__).resolve().parents[3]
        commands = json.loads((build / "compile_commands.json").read_text())
        names = ("aclnn_pre_process_bwd_kernel_merged.cpp", "pre_process_bwd_kernel_merged.cpp")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        objects, hashes = [], {}
        compiler = None
        for index, name in enumerate(names):
            entry = next(row for row in commands if row["file"].endswith("/op_api/" + name))
            raw = shlex.split(entry["command"])
            flags, position = [raw[0]], 1
            compiler = raw[0]
            while position < len(raw):
                if raw[position] in ("-c", "-o"):
                    position += 2
                else:
                    flags.append(raw[position])
                    position += 1
            output = Path(cls.temp.name) / f"source{index}.o"
            compile_checked(flags + ["-O0", "-g", "-fPIC", "-c", entry["file"], "-o", str(output)],
                            cwd=entry["directory"])
            objects.append(str(output))
            hashes[name] = hashlib.sha256(Path(entry["file"]).read_bytes()).hexdigest()
            if index == 0:
                probe_flags = flags
                probe_entry = entry
        probe = Path(cls.temp.name) / "probe.o"
        compile_checked(probe_flags + ["-O0", "-g", "-fPIC", f"-I{root}", "-c",
            str(Path(__file__).with_name("host_fault_probe.cpp")), "-o", str(probe)],
            cwd=probe_entry["directory"])
        wraps = ("_ZN14UniqueExecutorC1EPKc", "_Z19CommonOpExecutorRunPvmP13aclOpExecutorS_",
            "_ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor",
            "_ZN13aclOpExecutor15ConvertToTensorEPK11aclIntArrayN2ge8DataTypeE",
            "_Z25CreatAiCoreKernelLauncherPKcjP13aclOpExecutorPN2op12OpArgContextE")
        library = Path(cls.temp.name) / "cp_boundary.so"
        compile_checked([compiler, "-shared", "-Wl,-Bsymbolic-functions", "-Wl,-z,defs",
            *objects, str(probe), *("-Wl,--wrap=" + symbol for symbol in wraps),
            f"-L{cann}/lib64", f"-Wl,-rpath,{cann}/lib64", "-lnnopbase", "-lopapi",
            "-lerror_manager", "-lunified_dlog", "-lascendalog", "-lgraph", "-lgraph_base", "-lmetadef", "-o", str(library)])
        unresolved = subprocess.run(["nm", "-u", str(library)], capture_output=True, text=True, check=True).stdout
        # The constructor wrapper intentionally calls its real constructor on
        # non-failing paths. Other lower calls must bind to the test doubles.
        if any(symbol in unresolved for symbol in wraps[1:]):
            raise AssertionError("an intended fault target was not wrapped: " + unresolved)
        cls.opapi = ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)
        # Registers the operator-common diagnostic catalogue on CANN versions
        # that do not register it from libopapi alone. Calls stay bound to the
        # recompiled test objects through -Bsymbolic-functions.
        cls.custom = ctypes.CDLL(str(build / "libcust_opapi.so"))
        cls.acl = ctypes.CDLL("libascendcl.so")
        cls.acl.aclInit.argtypes = [ctypes.c_char_p]
        cls.acl.aclInit.restype = ctypes.c_int
        rc = cls.acl.aclInit(None)
        if rc not in (0, 100002):
            raise RuntimeError(f"aclInit failed: {rc}")
        cls.acl.aclrtSetDevice.argtypes = [ctypes.c_int]
        cls.acl.aclrtSetDevice.restype = ctypes.c_int
        device = int(os.environ["DFX_DEVICE"])
        rc = cls.acl.aclrtSetDevice(device)
        if rc:
            raise RuntimeError(f"aclrtSetDevice failed: {rc}")
        cls.addClassCleanup(cls.acl.aclrtResetDevice, device)
        cls.probe = ctypes.CDLL(str(library))
        cls.call = cls.probe.cp_boundary_case
        cls.call.argtypes, cls.call.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
        cls.metric = cls.probe.cp_boundary_metric
        cls.metric.argtypes, cls.metric.restype = [ctypes.c_int], ctypes.c_int
        cls.visited = cls.probe.cp_boundary_visited
        cls.visited.argtypes, cls.visited.restype = [ctypes.c_int], ctypes.c_int
        cls.recent = cls.acl.aclGetRecentErrMsg
        cls.recent.argtypes, cls.recent.restype = [], ctypes.c_char_p
        cls.evidence = []
        cls.hashes = hashes

    def check(self, mode, status, expected_counts, report=False, parameter=None, direct=False):
        self.recent()
        actual = self.call(mode, int(report))
        message = (self.recent() or b"").decode(errors="replace")
        metrics = [self.metric(index) for index in range(9)]
        trace = [self.visited(index) for index in range(metrics[1])]
        record = dict(mode=mode, simulated_lower_report=report, status=actual,
                      counters=metrics, parameter_order=trace, message=message)
        self.evidence.append(record)
        print("BOUNDARY_EVIDENCE " + json.dumps(record, ensure_ascii=False), flush=True)
        self.assertEqual(actual, status, record)
        self.assertEqual(metrics[:6], expected_counts, record)
        self.assertEqual(metrics[6:], [0, 0, 0], record)
        codes = re.findall(r"\bE[A-Z0-9]\d{4}\b", message)
        report_records = re.findall(r"\b[A-Za-z_]\w*\((E[A-Z0-9]\d{4})\):", message)
        if direct:
            self.assertIn("aclnnPreProcessBwdKernelMerged", message)
            self.assertIn(parameter, message)
            self.assertIn("writable output pointer", message)
            self.assertEqual(codes, ["EZ0037"], record)
            self.assertEqual(report_records, ["EZ0037"], record)
        elif report:
            self.assertIn("CP_UT_SIMULATED_LOWER", message)
            self.assertIn(parameter, message)
            self.assertIn("boundary evidence only", message)
            self.assertEqual(codes, ["EZ0037"], record)
            self.assertEqual(report_records, ["EZ0037"], record)
        else:
            self.assertEqual(message, "", record)
        return trace

    def test_output_pointer_guards_precede_executor_construction(self):
        for mode, parameter in ((20, "workspaceSize"), (21, "executor")):
            with self.subTest(parameter=parameter):
                self.check(mode, 161001, [0, 0, 0, 0, 0, 0], parameter=parameter, direct=True)

    def test_contiguous_failure_at_each_parameter(self):
        names = ("q", "k", "w", "d_o", "dv", "g", "gk")
        for index, name in enumerate(names):
            order = list(range(index + 1)) if index < 5 else [0, 1, 2, 3, 4, index]
            for report in (False, True):
                with self.subTest(parameter=name, simulated_report=report):
                    trace = self.check(10 + index, 161002, [1, len(order), 0, 0, 0, 1], report, name)
                    self.assertEqual(trace, order)

    def test_convert_failure_stops_l0_before_launcher(self):
        for report in (False, True):
            with self.subTest(simulated_report=report):
                trace = self.check(17, 161001, [1, 5, 1, 0, 0, 1], report, "cu_seqlens")
                self.assertEqual(trace, [0, 1, 2, 3, 4])

    def test_launcher_failure_propagates_l0_null_result(self):
        for mode in (18, 19):
            for report in (False, True):
                with self.subTest(segment=mode == 19, simulated_report=report):
                    trace = self.check(mode, 161001, [1, 5, int(mode == 19), 1, 0, 1], report, "launcher")
                    self.assertEqual(trace, [0, 1, 2, 3, 4])

    def test_execution_failure_keeps_public_status_and_first_error(self):
        for report in (False, True):
            with self.subTest(simulated_report=report):
                self.check(4, 561000, [0, 0, 0, 0, 1, 1], report, "executor_run")

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get("CP_BOUNDARY_EVIDENCE")
        if path:
            Path(path).write_text(json.dumps(dict(scope="boundary UT; lower calls are doubles",
                source_sha256=cls.hashes, cases=cls.evidence), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
