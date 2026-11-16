"""Explicit inputs and lazy NPU imports for the in-place CP merge."""
import importlib.util
from pathlib import Path
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_path = Path(__file__).resolve().parent / "scripts" / "merge_fwd_bwd_kernel_cpu.py"
_spec = importlib.util.spec_from_file_location("atk_merge_fwd_bwd_kernel_cpu", _path)
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def run_cpu(values):
    return _cpu.merge_fwd_bwd_kernel_cpu(**values)


def run_npu(values):
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc
    return ascendc.npu_merge_fwd_bwd_kernel(**values)


@register("executor_merge_fwd_bwd_kernel")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        if self.device == "cpu":
            return run_cpu(input_data.kwargs)
        if self.device == "npu":
            return run_npu(input_data.kwargs)
        raise RuntimeError(f"Unsupported device: {self.device}")
