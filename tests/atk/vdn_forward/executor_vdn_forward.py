"""ATK harness for the VDN alpha-bridge forward."""
import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device
from cases import expected, make_case, on_device


@register("executor_vdn_forward")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        spec = _case_spec(input_data, "vdn_forward")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[spec["dtype"]]
        case = make_case(spec["frames"], spec["heads"], spec["tokens"], dtype,
                         text=spec["text"], seed=spec["seed"])
        silu, norm = spec["silu"], spec["norm"]
        if self.device == "cpu":
            output = expected(case, silu, norm)
        elif self.device == "npu":
            from fla_npu.ops.vdn import vdn_forward

            if torch.npu.matmul.allow_hf32:
                raise ValueError("VDN precision tests require FP32 matrix multiplication")
            output = vdn_forward(*on_device(case, _marker_device(input_data)),
                                 delta_rule=spec["rule"], frame_tile=spec["tile"], value_first=True,
                                 fuse_silu=silu, fuse_qk_l2norm=norm)
        else:
            raise ValueError(f"unsupported VDN test backend {self.device}")
        return _finite_tuple(output, golden=self.device == "cpu")

