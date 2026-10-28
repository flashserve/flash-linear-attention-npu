"""Check A2/A3 output allocation without launching a device kernel.

Run with the updated fla_npu package installed:
    python tests/stable_abi/test_chunk_kda_bwd_output_contract.py
"""
import inspect
import itertools
import unittest
from unittest.mock import patch

import torch

from fla_npu.ops.ascendc import _aclnn_ctypes


class KdaBwdOutputContractTest(unittest.TestCase):
    def test_layouts_and_dtypes(self):
        for packed, input_tm, output_tm, saved, norm in itertools.product(
                (False, True), repeat=5):
            if not saved and input_tm:
                continue  # Recompute only accepts head-major external inputs.
            for heads, tokens in ((3, 65), (4, 4)):
                with self.subTest(packed=packed, input_tm=input_tm, output_tm=output_tm,
                                  saved=saved, norm=norm, heads=heads, tokens=tokens):
                    self.check_outputs(packed, input_tm, output_tm, saved, norm, heads, tokens)

    def check_outputs(self, packed, input_tm, output_tm, saved, norm, heads, tokens):
        batch = 2
        head = (heads, tokens, 128) if packed else (batch, heads, tokens, 128)
        gate = (tokens, heads, 128) if packed else (batch, tokens, heads, 128)
        external = gate if input_tm else head
        output = gate if output_tm else head
        chunks = (tokens + 63) // 64
        state = ((chunks, heads, 128, 128) if packed else
                 (batch, chunks, heads, 128, 128))
        layouts = ("NTD", "TND") if packed else ("BNSD", "BSND")
        args = inspect.signature(_aclnn_ctypes.npu_chunk_kda_bwd).bind_partial()
        args.apply_defaults()
        args = args.arguments
        args.update(scale=0.125, input_layout=layouts[input_tm],
                    output_layout=layouts[output_tm], disable_recompute=saved,
                    cu_seqlens=[0, tokens] if packed else None)
        for name in ("q", "k", "v", "d_o"):
            shape = head if norm and name in ("q", "k") else external
            args[name] = torch.empty(shape, dtype=torch.bfloat16)
        args["beta"] = torch.empty(head[:-1], dtype=torch.float32)
        for name in ("Aqk", "Akk"):
            args[name] = torch.empty((*head[:-1], 64), dtype=torch.bfloat16)
        for name in ("gk", "w", "qg", "kg", "v_new", "h"):
            dtype = torch.float32 if name == "gk" else torch.bfloat16
            args[name] = torch.empty(state if name == "h" else head, dtype=dtype) if saved else None
        args["raw_g"] = torch.empty(external, dtype=torch.bfloat16)
        args["A_log"] = torch.empty(heads, dtype=torch.float32)
        for name in ("q_rstd", "k_rstd"):
            args[name] = torch.empty(head[:-1], dtype=torch.float32) if norm else None

        # Only replace device-format inspection, allocation and launch. The
        # actual input validation and output-shape policy run unchanged.
        with patch("torch_npu.get_npu_format", return_value=2), \
                patch.object(_aclnn_ctypes, "_empty",
                             side_effect=lambda shape, ref, dtype: torch.empty(shape, dtype=dtype)), \
                patch.object(_aclnn_ctypes, "_call_aclnn",
                             side_effect=lambda name, build, outputs: outputs):
            dq, dk, dv, db, dg, dh0, da, dbias = _aclnn_ctypes._run_kda_bwd_arch22(args)
        for gradient in (dq, dk, dv):
            self.assertEqual(tuple(gradient.shape), output)
            self.assertEqual(gradient.dtype, torch.bfloat16)
        self.assertEqual(tuple(db.shape), gate[:-1])
        self.assertEqual(tuple(dg.shape), gate)
        self.assertEqual(db.dtype, torch.float32)
        self.assertEqual(dg.dtype, torch.float32)
        self.assertIsNone(dh0)
        self.assertIsNone(dbias)
        self.assertEqual(tuple(da.shape), (heads,))


if __name__ == "__main__":
    unittest.main()
