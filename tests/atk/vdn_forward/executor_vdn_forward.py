# SPDX-License-Identifier: BSD-3-Clause
"""ATK CPU oracle and NPU entry for the VDN alpha-bridge forward."""

import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scripts.reference import reference


@register("executor_vdn_forward")
class FunctionApi(BaseApi):
    def init_by_input_data(self, input_data):
        self.arguments = dict(input_data.kwargs)
        # CPU benchmark nodes may widen input tensors. Restore the model's
        # explicit rounding boundaries using the dtype declared by the case.
        dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "int64": torch.int64}
        for field in self.task_result.case_config.inputs:
            tensor = self.arguments.get(field.name)
            if field.type == "tensor" and isinstance(tensor, torch.Tensor):
                self.arguments[field.name] = tensor.to(dtypes[field.dtype])
        state = self.arguments.get("text_state")
        if not isinstance(state, torch.Tensor) or state.numel() == 0:
            self.arguments["text_state"] = None

        # Each row must contain its own frame. ATK samples both endpoints
        # independently. Project them onto the legal interval once, outside
        # the measured call, and pass the resulting CPU metadata to both sides.
        bounds = self.arguments["bounds"].cpu()
        frames = bounds.shape[0]
        row = torch.arange(frames)
        lo, hi = bounds.clamp(0, frames - 1).unbind(dim=1)
        self.arguments["bounds"] = torch.stack((lo.minimum(row), hi.maximum(row)), dim=1)

    def __call__(self, input_data, with_output=False):
        args = self.arguments
        if self.device == "cpu":
            out = reference(
                *(args[key] for key in ("q", "k", "v", "beta", "alpha", "bounds", "text_state")),
                silu=args["fuse_silu"], qk_l2norm=args["fuse_qk_l2norm"],
            )
        elif self.device == "npu":
            from fla_npu.ops.vdn import vdn_forward

            if torch.npu.matmul.allow_hf32:
                raise ValueError("VDN accuracy requires FP32 matrix multiplication")
            out = vdn_forward(**args)
        else:
            raise ValueError(f"Unsupported VDN test backend {self.device!r}")
        if with_output:
            return (out,)
        return None
