# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops


torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def fast_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    return (x_float / (1 + torch.exp(-1.702 * x_float))).to(x.dtype)


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestFastGelu(unittest.TestCase):
    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertTrue(
            torch.allclose(actual.detach().cpu().float(), expected.detach().cpu().float(), rtol=1e-3, atol=1e-3),
            msg=f"max diff={(actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max().item()}",
        )

    def test_npu_fast_gelu_custom(self):
        values = torch.linspace(-3, 3, steps=3 * 16 * 32, dtype=torch.float32).reshape(3, 16, 32)
        npu_input = values.npu()
        custom_output = ascendc_ops.npu_fast_gelu_custom(npu_input)
        self.assertTensorClose(custom_output, fast_gelu_ref(values).npu())


if __name__ == "__main__":
    unittest.main()
