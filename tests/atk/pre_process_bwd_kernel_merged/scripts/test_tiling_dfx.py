"""Compile the actual CP processor with its target flags and verify diagnostics."""

import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
import re


class TilingDfxTest(unittest.TestCase):
    def test_processor(self):
        repo = Path(__file__).resolve().parents[4]
        build = Path(os.environ.get("DFX_BUILD_DIR", repo / "build"))
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        entry = next(row for row in json.loads((build / "compile_commands.json").read_text())
                     if row["file"].endswith("pre_process_bwd_kernel_merged_tiling.cpp"))
        raw = shlex.split(entry["command"])
        args = [raw[0]]
        index = 1
        while index < len(raw):
            if raw[index] in ("-o", "-c"):
                index += 2
            else:
                args.append(raw[index])
                index += 1
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "probe.so"
            subprocess.run(args + ["-shared", "-fPIC", f"-I{repo}",
                str(Path(__file__).with_name("tiling_dfx_probe.cpp")), f"-L{cann}/lib64",
                f"-Wl,-rpath,{cann}/lib64", "-lerror_manager", "-lascendalog", "-lexe_graph",
                "-lgraph_base", "-o", str(library)], cwd=entry["directory"], check=True)
            probe = ctypes.CDLL(str(library))
            acl = ctypes.CDLL("libascendcl.so")
            opapi = ctypes.CDLL("libopapi.so")
            custom = ctypes.CDLL(str(build / "libcust_opapi.so"))
            self.assertIn(acl.aclInit(None), (0, 100002))
            self.assertEqual(acl.aclrtSetDevice(int(os.environ.get("DFX_DEVICE", 0))), 0)
            recent = acl.aclGetRecentErrMsg
            recent.restype = ctypes.c_char_p
            method = probe.dfx_cp_processor_case
            method.argtypes, method.restype = [ctypes.c_int], ctypes.c_uint32
            self.assertEqual(method(0), 0)
            codes = ["EZ0008", "EZ0009", "EZ0009", "EZ0008", "EZ0009", "EZ0027", "EZ0027",
                     "EZ0024", "EZ0027", "EZ0009", "EZ0020", "EZ0009", "EZ0020", "EZ0025",
                     "EZ0024", "EZ0024", "EZ1008", "EZ0004", "EZ0011", "EZ0004", "EZ0004",
                     "EZ0011", "EZ0011", "EZ0011", "EZ0020", "EZ0024"] + ["EZ1008"] * 5 + ["EZ0026"] * 2 + ["EZ0004"] * 5 + ["EZ1008"] * 5 + ["EZ0004"] * 2 + ["EZ1008"] * 2
            params = ["k", "w", "d_o", "dv", "q", "Hk,Hv", "K,V", "chunk_size", "g,gk", "gk",
                      "gk", "g", "g", "cu_seqlens", "B", "B", "0 AIC cores", "q", "q", "g", "gk",
                      "g", "gk", "cu_seqlens", "q", "chunk_size", "context", "tilingData", "attrs",
                      "platformInfo", "workspaceSizes", "4294967232", "4294967232"] + ["q", "k", "w", "d_o", "dv"] * 2 + ["g", "gk"] * 2
            configured = os.environ.get("CP_DFX_CASES", "")
            selected = {int(value) for value in configured.split(",") if value} if configured else set(range(1, len(codes) + 1))
            self.assertTrue(selected and selected <= set(range(1, len(codes) + 1)), selected)
            for case, (code, param) in enumerate(zip(codes, params), 1):
                if case not in selected:
                    continue
                with self.subTest(case=case):
                    recent()
                    self.assertNotEqual(method(case), 0)
                    message = (recent() or b"").decode(errors="replace")
                    self.assertEqual(re.findall(r"\bE[A-Z0-9]\d{4}\b", message), [code], message)
                    self.assertIn(param, message)
                    self.assertIn("dfx_probe_cp", message)
                    self.assertNotIn("EZ9999", message)
                    self.assertNotIn("EZ9903", message)
                    if 39 <= case <= 43 or case >= 46:
                        self.assertIn("shape is present", message)
                        self.assertIn("metadata construction", message)
                    print(f"PASS processor {case}: {message}", flush=True)


if __name__ == "__main__":
    unittest.main()
