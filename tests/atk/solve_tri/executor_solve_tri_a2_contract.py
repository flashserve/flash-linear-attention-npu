"""A2 ATK adapter for real ACLNN errors and legal metadata permutations.

An unexpected status, wrong diagnostic, returned executor, or changed valid
result raises an assertion and fails the ATK case. No CPU golden is fabricated.
"""
from pathlib import Path
import sys

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from check_aclnn_contract import PARAM_INVALID, POSITIVE_CASE, check_execution, check_rejection


@register("executor_solve_tri_a2_contract")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "npu":
            raise RuntimeError("SolveTri ACLNN contract cases require an NPU ATK node")
        name = str(input_data.kwargs["contract_case"])
        expected = int(input_data.kwargs["expected_return_code"])
        device = torch.device(f"npu:{int(self.device_id)}")
        torch.npu.set_device(device)
        if name == POSITIVE_CASE:
            if expected != 0:
                raise AssertionError("positive contract case must expect success")
            status = check_execution(device)
        else:
            if expected != PARAM_INVALID:
                raise AssertionError("negative contract case must expect 161002")
            status = check_rejection(name, device, expected)
        return (torch.tensor([status], dtype=torch.int64),) if with_output else None
