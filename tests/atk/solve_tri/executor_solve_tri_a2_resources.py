"""A2 ATK SolveTri peak allocation measurements on the fixed model inputs.

Run atk_solve_tri_a2_resources.json with --task run and set SOLVE_TRI_MEMORY_OUTPUT.
This measures torch_npu allocator bytes, including the stable wrapper workspace;
it does not claim to measure driver/runtime allocations outside that allocator.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import torch
import torch_npu
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor_solve_tri_a2 import build_inputs


def _operator_fingerprints():
    result = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if name.startswith("/") and name.endswith("libcust_opapi.so"):
            path = Path(name)
            if path.is_file():
                result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise RuntimeError("Cannot record the actual loaded SolveTri opapi library")
    return result


def _kernel_fingerprints():
    result = {}
    for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if not value:
            continue
        root = Path(value) / "op_impl" / "ai_core" / "tbe" / "kernel"
        for path in sorted(root.glob("*/solve_tri/*")):
            if path.is_file() and path.suffix in {".o", ".json"}:
                result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(name.endswith(".o") for name in result):
        raise RuntimeError("Cannot record installed SolveTri kernel objects")
    return result


@register("executor_solve_tri_a2_resources")
class FunctionApi(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        if self.device != "npu":
            raise ValueError("SolveTri resource measurements require an ATK NPU node")
        self.case = task_result.case_config
        self.inputs = None
        self.recorded = False

    def init_by_input_data(self, input_data):
        from _ascendc_common_executor import _case_spec

        self.spec = _case_spec(input_data, "solve_tri")
        self.inputs = build_inputs(self.spec, torch.device(f"npu:{self.device_id}"), high_precision=False)
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)

    def _forward(self):
        from fla_npu.ops import ascendc

        return ascendc.solve_tri(self.inputs["x"], cu_seqlens=self.inputs["cu_seqlens"],
                                chunk_indices=self.inputs["chunk_indices"], layout=self.inputs["layout"])

    def __call__(self, input_data, with_output=False):
        if self.inputs is None:
            self.init_by_input_data(input_data)
        device = self.inputs["x"].device
        if not self.recorded:
            destination = os.environ.get("SOLVE_TRI_MEMORY_OUTPUT")
            if not destination:
                raise RuntimeError("Set SOLVE_TRI_MEMORY_OUTPUT to an isolated output directory")
            for _ in range(3):
                warmup = self._forward()
                torch.npu.synchronize(device)
                del warmup
            torch.npu.synchronize(device)
            before = torch.npu.memory_allocated(device)
            torch.npu.reset_peak_memory_stats(device)
            output = self._forward()
            torch.npu.synchronize(device)
            peak = torch.npu.max_memory_allocated(device)
            output_bytes = output.untyped_storage().nbytes()
            shape = list(self.inputs["x"].shape)
            record = {"contract": {"shape": shape, "dtype": self.spec["dtype"], "layout": self.spec["layout"],
                        "seqlens": self.spec.get("seqlens"), "seed": self.spec["seed"], "generator": "executor_solve_tri_a2"},
                      "baseline_bytes": before, "peak_bytes": peak, "output_bytes": output_bytes,
                      "extra_bytes": peak - before - output_bytes,
                      "warmup_count": 3, "measurement_count": 1, "device": str(device),
                      "device_id": device.index,
                      "device_name": torch.npu.get_device_name(device), "torch": torch.__version__,
                      "torch_npu": torch_npu.__version__, "cann_home": os.environ.get("ASCEND_HOME_PATH"),
                      "hostnames": sorted({socket.gethostname(), socket.getfqdn()}),
                      "mapped_files_sha256": _operator_fingerprints(),
                      "kernel_assets_sha256": _kernel_fingerprints(), "pid": os.getpid()}
            if record["extra_bytes"] < 0:
                raise RuntimeError("Peak memory counters do not include the returned output")
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            payload = {"method": "atk_memory_peak_allocated", "cases": {str(self.case.id): record}}
            (path / f"case_{self.case.id}_{os.getpid()}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.recorded = True
        else:
            output = self._forward()
        return (output.detach().cpu(),) if with_output else None
