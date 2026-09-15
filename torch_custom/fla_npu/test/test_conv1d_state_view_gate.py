# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The conv1d state gate: which conv_state layouts may cross as a view.

Runs without an NPU: the question is answered by the toolkit's version file and
by the tensor's own strides, so it can be checked without launching anything.
The measured behaviour behind the boundary (9.1.0 addresses a paged state
densely and writes into the gaps, 9.2.0 is bit-exact) is in
`docs/architecture/stable-abi-host-cost.md`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"


def load_modules():
    package_name = "fla_npu_test_conv1d_gate"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, ASCENDC_DIR / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return (sys.modules[f"{package_name}._runtime"],
            sys.modules[f"{package_name}._aclnn_ctypes"])


RUNTIME, CTYPES = load_modules()


class FakeTensor:
    """The slice of the tensor API the gate reads."""

    def __init__(self, shape, stride, numel=None):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self._numel = (numel if numel is not None
                       else _numel_of(self.shape, self._stride))

    def numel(self):
        return self._numel

    def stride(self):
        return self._stride

    def is_contiguous(self):
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self._stride)):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True


def _numel_of(shape, stride):
    if not shape:
        return 0
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


class Conv1dStateViewGateTest(unittest.TestCase):
    def setUp(self):
        RUNTIME._CONV1D_VIEW_STATE = None
        self.addCleanup(setattr, RUNTIME, "_CONV1D_VIEW_STATE", None)

    def _with_version(self, version, env=None):
        """Point the toolkit variables at a version.info carrying *version*."""

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        home = Path(directory.name)
        (home / "compiler").mkdir()
        (home / "compiler" / "version.info").write_text(
            f"Version={version}\nversion_dir=cann\n", encoding="utf-8")
        variables = {"ASCEND_HOME_PATH": str(home)}
        variables.update(env or {})
        patch = mock.patch.dict(os.environ, variables, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        return patch

    def test_version_info_parses_to_three_components(self):
        self._with_version("9.2.0-beta.1")
        self.assertEqual(RUNTIME.cann_version(), (9, 2, 0))

    def test_version_info_absent_is_undecided(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(RUNTIME.cann_version())

    def test_92_supports_view_and_91_does_not(self):
        self._with_version("9.2.0-beta.1")
        self.assertTrue(RUNTIME.conv1d_view_state_supported())
        RUNTIME._CONV1D_VIEW_STATE = None
        self._with_version("9.1.0-beta.1")
        with self.assertWarns(RuntimeWarning):
            self.assertFalse(RUNTIME.conv1d_view_state_supported())

    def test_env_override_wins_over_the_version(self):
        self._with_version("9.1.0-beta.1",
                           {"FLA_NPU_CONV1D_VIEW_STATE": "1"})
        self.assertTrue(RUNTIME.conv1d_view_state_supported())
        RUNTIME._CONV1D_VIEW_STATE = None
        self._with_version("9.2.0-beta.1",
                           {"FLA_NPU_CONV1D_VIEW_STATE": "0"})
        # An explicit verdict is not second-guessed, and it does not warn: the
        # warning names the override, so warning about the override is noise.
        self.assertFalse(RUNTIME.conv1d_view_state_supported())

    def test_dense_and_offset_states_never_need_a_copy(self):
        for version in ("9.1.0-beta.1", "9.2.0-beta.1"):
            RUNTIME._CONV1D_VIEW_STATE = None
            self._with_version(version)
            dense = FakeTensor((4, 3, 16), (48, 16, 1))
            offset = FakeTensor((4, 3, 16), (48, 16, 1), numel=4 * 48)
            for state in (dense, offset, None):
                self.assertFalse(
                    CTYPES._causal_conv1d_state_needs_dense_copy(state),
                    f"{version}: dense/offset state must cross as-is")

    def test_paged_state_is_staged_only_where_the_runtime_drops_the_view(self):
        paged = FakeTensor((5, 3, 16), (96, 16, 1))
        RUNTIME._CONV1D_VIEW_STATE = None
        self._with_version("9.1.0-beta.1")
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(
                CTYPES._causal_conv1d_state_needs_dense_copy(paged))
        RUNTIME._CONV1D_VIEW_STATE = None
        self._with_version("9.2.0-beta.1")
        self.assertFalse(CTYPES._causal_conv1d_state_needs_dense_copy(paged))

    def test_layout_no_runtime_can_index_is_always_staged(self):
        self._with_version("9.2.0-beta.1")
        transposed = FakeTensor((4, 3, 16), (48, 1, 3))
        negative = FakeTensor((4, 3, 16), (-48, 16, 1))
        two_dim = FakeTensor((3, 16), (16, 1))
        for state in (transposed, negative, two_dim):
            self.assertTrue(
                CTYPES._causal_conv1d_state_needs_dense_copy(state))


if __name__ == "__main__":
    unittest.main()
