# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"


def _load_aclnn_ctypes_module():
    package_name = "fla_npu_test_chunk_kda_fwd_prepare"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            ASCENDC_DIR / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        if spec.loader is None:
            raise RuntimeError(f"cannot load {qualified_name}")
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}._aclnn_ctypes"]


ACLNN_CTYPES = _load_aclnn_ctypes_module()

BFLOAT16 = object()
FLOAT16 = object()
FLOAT32 = object()
FAKE_TORCH = types.ModuleType("torch")
FAKE_TORCH.bfloat16 = BFLOAT16
FAKE_TORCH.float16 = FLOAT16
FAKE_TORCH.float32 = FLOAT32


class _FakeDevice:
    type = "npu"
    index = 0


DEVICE = _FakeDevice()


class _FakeTensor:
    def __init__(self, shape, dtype, *, device=DEVICE, contiguous=True, offset=0):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = bool(contiguous)
        self._offset = int(offset)

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def storage_offset(self):
        return self._offset


class _FakeCallContext:
    def __init__(self):
        self.descriptors = []
        self.int_arrays = []

    def tensor(
        self,
        tensor,
        name,
        *,
        acl_format_override=None,
        storage_shape_override=None,
    ):
        self.descriptors.append(
            (name, tensor, acl_format_override, storage_shape_override)
        )
        return ctypes.c_void_p(0x1000 + len(self.descriptors))

    def int_array(self, values):
        values = None if values is None else tuple(values)
        self.int_arrays.append(values)
        return ctypes.c_void_p(0x2000 + len(self.int_arrays))


def _empty(shape, like, *, dtype=None):
    return _FakeTensor(shape, dtype or like.dtype, device=like.device)


def _make_inputs(
    layout,
    *,
    batch=1,
    key_heads=2,
    value_heads=10,
    tokens=65,
    q_dtype=BFLOAT16,
    gate_dtype=FLOAT32,
    beta_dtype=BFLOAT16,
):
    if layout == "BNSD":
        q_shape = (batch, key_heads, tokens, 128)
        value_shape = (batch, value_heads, tokens, 128)
        beta_shape = (batch, value_heads, tokens)
    elif layout == "BSND":
        q_shape = (batch, tokens, key_heads, 128)
        value_shape = (batch, tokens, value_heads, 128)
        beta_shape = (batch, tokens, value_heads)
    elif layout == "NTD":
        q_shape = (key_heads, tokens, 128)
        value_shape = (value_heads, tokens, 128)
        beta_shape = (value_heads, tokens)
    elif layout == "TND":
        q_shape = (tokens, key_heads, 128)
        value_shape = (tokens, value_heads, 128)
        beta_shape = (tokens, value_heads)
    else:
        raise ValueError(layout)

    q = _FakeTensor(q_shape, q_dtype)
    return (
        q,
        _FakeTensor(q_shape, q_dtype),
        _FakeTensor(value_shape, q_dtype),
        _FakeTensor(value_shape, gate_dtype),
        _FakeTensor(beta_shape, beta_dtype),
    )


def _capture_call():
    captured = {}

    def fake_call(name, build_args, outputs):
        context = _FakeCallContext()
        captured["name"] = name
        captured["args"] = build_args(context)
        captured["outputs"] = outputs
        captured["descriptors"] = context.descriptors
        captured["int_arrays"] = context.int_arrays
        return outputs

    return captured, fake_call


class ChunkKdaFwdPrepareWrapperTest(unittest.TestCase):
    def _run(self, inputs, **kwargs):
        captured, fake_call = _capture_call()
        patched_modules = {"torch": FAKE_TORCH}
        if "torch_npu" in sys.modules:
            fake_torch_npu = types.ModuleType("torch_npu")
            fake_torch_npu.get_npu_format = lambda tensor: 2
            patched_modules["torch_npu"] = fake_torch_npu
        with mock.patch.dict(sys.modules, patched_modules):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=_empty):
                with mock.patch.object(ACLNN_CTYPES, "_acl_format", return_value=2):
                    with mock.patch.object(
                        ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call
                    ):
                        outputs = ACLNN_CTYPES.npu_chunk_kda_fwd_prepare(
                            *inputs, **kwargs
                        )
        return outputs, captured

    def test_frozen_aclnn_signature_and_argument_order(self):
        torch_npu_was_loaded = "torch_npu" in sys.modules
        expected_argtypes = [
            *([ctypes.c_void_p] * 9),
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_double,
            *([ctypes.c_bool] * 5),
            ctypes.c_double,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 13),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkKdaFwdPrepare"],
            expected_argtypes,
        )

        inputs = _make_inputs("BNSD", batch=2, key_heads=3, value_heads=15)
        outputs, captured = self._run(
            inputs,
            layout="BNSD",
            scale=0.125,
            epsilon=2e-6,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            lower_bound=-4.0,
            use_exp2=True,
        )

        operator_argtypes = expected_argtypes[:-2]
        self.assertEqual(captured["name"], "aclnnChunkKdaFwdPrepare")
        self.assertEqual(len(captured["args"]), 33)
        self.assertEqual(
            [type(argument) for argument in captured["args"]], operator_argtypes
        )
        self.assertEqual(captured["args"][9].value, b"BNSD")
        self.assertEqual(captured["args"][10].value, 0.125)
        self.assertEqual(captured["args"][11].value, 64)
        self.assertAlmostEqual(captured["args"][12].value, 2e-6)
        self.assertEqual(
            [argument.value for argument in captured["args"][13:18]],
            [True, False, True, True, False],
        )
        self.assertEqual(captured["args"][18].value, -4.0)
        self.assertTrue(captured["args"][19].value)
        self.assertEqual(len(outputs), 13)
        self.assertEqual("torch_npu" in sys.modules, torch_npu_was_loaded)

    def test_all_layouts_produce_fixed_head_major_outputs(self):
        cases = {
            "BNSD": (
                (2, 6, 65, 128),
                (2, 6, 65, 64),
                (2, 2, 65, 128),
                (2, 2, 65),
                (2, 6, 65),
            ),
            "BSND": (
                (2, 6, 65, 128),
                (2, 6, 65, 64),
                (2, 2, 65, 128),
                (2, 2, 65),
                (2, 6, 65),
            ),
            "NTD": (
                (6, 65, 128),
                (6, 65, 64),
                (2, 65, 128),
                (2, 65),
                (6, 65),
            ),
            "TND": (
                (6, 65, 128),
                (6, 65, 64),
                (2, 65, 128),
                (2, 65),
                (6, 65),
            ),
        }
        for layout, expected in cases.items():
            with self.subTest(layout=layout):
                batch = 2 if layout in {"BNSD", "BSND"} else 1
                outputs, captured = self._run(
                    _make_inputs(
                        layout,
                        batch=batch,
                        key_heads=2,
                        value_heads=6,
                    ),
                    layout=layout,
                )
                value_shape, matrix_shape, key_shape, key_stat, value_stat = expected
                self.assertEqual(
                    [output.shape for output in outputs],
                    [
                        value_shape,
                        matrix_shape,
                        matrix_shape,
                        value_shape,
                        value_shape,
                        value_shape,
                        value_shape,
                        value_shape,
                        key_shape,
                        key_shape,
                        key_stat,
                        key_stat,
                        value_stat,
                    ],
                )
                self.assertEqual(
                    [output.dtype for output in outputs],
                    [
                        FLOAT32,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        BFLOAT16,
                        FLOAT32,
                        FLOAT32,
                        FLOAT32,
                    ],
                )
                self.assertEqual(
                    [name for name, *_ in captured["descriptors"]],
                    [
                        "q",
                        "k",
                        "v",
                        "g",
                        "beta",
                        "a_log",
                        "dt_bias",
                        "gk",
                        "aqk",
                        "akk",
                        "w",
                        "u",
                        "qg",
                        "kg",
                        "qg_scaled",
                        "q_hat",
                        "k_hat",
                        "q_rstd",
                        "k_rstd",
                        "beta_eff",
                    ],
                )

    def test_backward_modes_keep_fixed_slots_and_select_expected_outputs(self):
        expected_masks = {
            "none": (
                True, True, False, True, True, False, True, True,
                False, False, False, False, False,
            ),
            "recompute": (
                True, True, True, True, True, False, True, True,
                True, True, True, True, True,
            ),
            "save": (True,) * 13,
        }
        for backward_mode, expected_mask in expected_masks.items():
            with self.subTest(backward_mode=backward_mode):
                outputs, captured = self._run(
                    _make_inputs("BNSD"),
                    layout="BNSD",
                    backward_mode=backward_mode,
                )
                self.assertEqual(len(outputs), 13)
                self.assertEqual(
                    tuple(output is not None for output in outputs),
                    expected_mask,
                )
                self.assertEqual(
                    tuple(
                        tensor is not None
                        for _, tensor, *_ in captured["descriptors"][-13:]
                    ),
                    expected_mask,
                )
                self.assertEqual(len(captured["args"]), 33)

    def test_backward_mode_defaults_to_save(self):
        outputs, _ = self._run(_make_inputs("BNSD"), layout="BNSD")
        self.assertEqual(tuple(output is not None for output in outputs), (True,) * 13)

    def test_invalid_backward_mode_fails_before_launch(self):
        with mock.patch.dict(sys.modules, {"torch": FAKE_TORCH}):
            with mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as call:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "backward_mode must be one of: none, recompute, save",
                ):
                    ACLNN_CTYPES.npu_chunk_kda_fwd_prepare(
                        *_make_inputs("BNSD"),
                        layout="BNSD",
                        backward_mode="invalid",
                    )
        call.assert_not_called()

    def test_varlen_metadata_is_canonical_and_optional(self):
        inputs = _make_inputs("TND", key_heads=1, value_heads=7, tokens=130)
        _, captured = self._run(
            inputs,
            layout="TND",
            cu_seqlens=(0, 1, 65, 130),
        )
        self.assertEqual(captured["int_arrays"][0], (0, 1, 65, 130))
        self.assertEqual(
            captured["int_arrays"][1],
            (0, 0, 1, 0, 2, 0, 2, 1),
        )

        _, dense_captured = self._run(inputs, layout="TND")
        self.assertEqual(dense_captured["int_arrays"], [None, None])

    def test_varlen_sequence_count_has_no_artificial_upper_bound(self):
        inputs = _make_inputs("TND", key_heads=1, value_heads=1, tokens=1)
        cu_seqlens = (*([0] * 1025), 1)
        outputs, captured = self._run(
            inputs,
            layout="TND",
            cu_seqlens=cu_seqlens,
        )
        self.assertEqual(captured["int_arrays"][0], cu_seqlens)
        self.assertEqual(captured["int_arrays"][1], (1024, 0))
        self.assertEqual(outputs[0].shape, (1, 1, 128))

    def test_gva_head_count_has_no_artificial_upper_bound(self):
        outputs, _ = self._run(
            _make_inputs(
                "BNSD",
                key_heads=32,
                value_heads=160,
                tokens=1,
            ),
            layout="BNSD",
        )
        self.assertEqual(outputs[0].shape, (1, 160, 1, 128))
        self.assertEqual(outputs[8].shape, (1, 32, 1, 128))

    def test_gate_inputs_and_attributes_are_forwarded(self):
        inputs = _make_inputs("BSND", key_heads=2, value_heads=8)
        a_log = _FakeTensor((8,), FLOAT32)
        dt_bias = _FakeTensor((8 * 128,), FLOAT32)
        _, captured = self._run(
            inputs,
            layout="BSND",
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-3.0,
            a_log=a_log,
            dt_bias=dt_bias,
        )
        descriptors = {name: tensor for name, tensor, *_ in captured["descriptors"]}
        self.assertIs(descriptors["a_log"], a_log)
        self.assertIs(descriptors["dt_bias"], dt_bias)
        self.assertEqual(
            [argument.value for argument in captured["args"][13:18]],
            [False, True, False, False, True],
        )

    def test_invalid_public_contracts_fail_before_launch(self):
        cases = [
            (
                _make_inputs("BNSD", q_dtype=FLOAT16),
                {"layout": "BNSD"},
                "q, k and v must all use bfloat16",
            ),
            (
                _make_inputs("BNSD", key_heads=3, value_heads=10),
                {"layout": "BNSD"},
                "GVA requires",
            ),
            (
                _make_inputs("BNSD", beta_dtype=FLOAT16),
                {"layout": "BNSD"},
                "beta must use bfloat16 or float32",
            ),
            (
                _make_inputs("BNSD"),
                {"layout": "BNSD", "allow_neg_eigval": True},
                "requires use_beta_sigmoid_in_kernel=True",
            ),
            (
                _make_inputs("BNSD"),
                {
                    "layout": "BNSD",
                    "a_log": _FakeTensor((10,), FLOAT32),
                },
                "require use_gate_in_kernel=True",
            ),
            (
                _make_inputs("TND"),
                {"layout": "TND", "chunk_indices": (0, 0)},
                "can only be provided with cu_seqlens",
            ),
            (
                _make_inputs("TND", tokens=65),
                {
                    "layout": "TND",
                    "cu_seqlens": (0, 65),
                    "chunk_indices": (0, 1),
                },
                "canonical sequence-major order",
            ),
            (
                _make_inputs(
                    "BSND",
                    key_heads=1 << 32,
                    value_heads=1 << 32,
                    tokens=1,
                ),
                {"layout": "BSND"},
                "fit uint32",
            ),
            (
                _make_inputs(
                    "TND",
                    key_heads=8_388_610,
                    value_heads=8_388_610,
                    tokens=1,
                    gate_dtype=FLOAT32,
                ),
                {"layout": "TND"},
                "cross-head DMA strides must fit uint32",
            ),
        ]

        for inputs, kwargs, message in cases:
            with self.subTest(message=message):
                with mock.patch.dict(sys.modules, {"torch": FAKE_TORCH}):
                    with mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as call:
                        with self.assertRaisesRegex(RuntimeError, message):
                            ACLNN_CTYPES.npu_chunk_kda_fwd_prepare(
                                *inputs, **kwargs
                            )
                call.assert_not_called()

    def test_ctypes_entry_is_exported_but_not_installed_as_legacy_op(self):
        package_name = "fla_npu_test_chunk_kda_fwd_prepare_exports"
        spec = importlib.util.spec_from_file_location(
            package_name,
            ASCENDC_DIR / "__init__.py",
            submodule_search_locations=[str(ASCENDC_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        fake_fla_npu = types.ModuleType("fla_npu")
        fake_fla_npu.load_ascendc_opapi_libraries = lambda: ()
        with mock.patch.dict(
            sys.modules,
            {package_name: module, "fla_npu": fake_fla_npu},
        ):
            if spec.loader is None:
                raise RuntimeError(f"cannot load {package_name}")
            spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, "npu_chunk_kda_fwd_prepare"))
        self.assertTrue(hasattr(module, "chunk_kda_fwd_prepare"))
        self.assertIn("npu_chunk_kda_fwd_prepare", module.__all__)
        self.assertIn("chunk_kda_fwd_prepare", module.__all__)
        self.assertNotIn(
            "npu_chunk_kda_fwd_prepare", module._TORCH_NPU_COMPAT_OPS
        )


if __name__ == "__main__":
    unittest.main()
