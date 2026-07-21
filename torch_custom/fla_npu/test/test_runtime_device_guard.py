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

import contextlib
import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_PATH = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc" / "_runtime.py"
SPEC = importlib.util.spec_from_file_location("fla_npu_test_runtime", RUNTIME_PATH)
RUNTIME = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNTIME)


class FakeDevice:
    def __init__(self, index: int):
        self.type = "npu"
        self.index = index


class FakeTensor:
    def __init__(self, index: int, data_ptr: int = 0x1234):
        self.device = FakeDevice(index)
        self._data_ptr = data_ptr

    def data_ptr(self) -> int:
        return self._data_ptr


class FakeDescriptorTensor(FakeTensor):
    def __init__(self, index: int, shape, strides, data_ptr: int = 0x1234):
        super().__init__(index, data_ptr)
        self.shape = tuple(shape)
        self.dtype = object()
        self._strides = tuple(strides)

    def dim(self) -> int:
        return len(self.shape)

    def stride(self):
        return self._strides

    def storage_offset(self) -> int:
        return 0


class FakeNpu:
    def __init__(self, current_device: int = 0):
        self.current_device_index = current_device
        self.guard_events = []
        self.stream_queries = []

    def current_device(self) -> int:
        return self.current_device_index

    def set_device(self, index: int) -> None:
        self.current_device_index = int(index)

    @contextlib.contextmanager
    def device(self, index: int):
        previous = self.current_device_index
        self.guard_events.append(("enter", int(index), previous))
        self.current_device_index = int(index)
        try:
            yield
        finally:
            self.guard_events.append(("exit", int(index), previous))
            self.current_device_index = previous

    def current_stream(self, device=None):
        index = self.current_device_index if device is None else int(device)
        self.stream_queries.append((index, self.current_device_index))
        return types.SimpleNamespace(npu_stream=0x1000 + index)


def fake_torch(npu: FakeNpu):
    module = types.ModuleType("torch")
    module.npu = npu
    module.uint8 = object()
    module.empty_calls = []

    def empty(shape, *, dtype, device):
        module.empty_calls.append((shape, dtype, device.index, npu.current_device()))
        return FakeTensor(device.index, data_ptr=0xBEEF)

    module.empty = empty
    return module


class RuntimeDeviceGuardTest(unittest.TestCase):
    def setUp(self):
        RUNTIME._RECENT_LAUNCH_STORAGE.clear()

    def test_device_guard_switches_to_target_and_restores_previous_device(self):
        npu = FakeNpu(current_device=0)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(npu)}):
            with RUNTIME._npu_device_guard(FakeDevice(2)) as index:
                self.assertEqual(index, 2)
                self.assertEqual(npu.current_device(), 2)

        self.assertEqual(npu.current_device(), 0)
        self.assertEqual(npu.guard_events, [("enter", 2, 0), ("exit", 2, 0)])

    def test_call_context_rejects_tensors_from_another_device(self):
        npu = FakeNpu(current_device=0)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(npu)}):
            ctx = RUNTIME._CallContext(object(), FakeDevice(1))
            with self.assertRaisesRegex(ValueError, r"input must be on npu:1, got npu:0"):
                ctx.tensor(FakeTensor(0), "input")

    def test_call_device_rejects_outputs_from_different_devices(self):
        with self.assertRaisesRegex(ValueError, r"output\[1\] must be on npu:0, got npu:1"):
            RUNTIME._call_device((FakeTensor(0), FakeTensor(1)))

    def test_runtime_uses_target_device_for_workspace_and_stream(self):
        npu = FakeNpu(current_device=1)
        torch_module = fake_torch(npu)
        calls = {}

        def get_workspace(*args):
            args[-2]._obj.value = 64
            args[-1]._obj.value = 0xCAFE
            return RUNTIME.ACL_SUCCESS

        def launch(workspace, size, executor, stream):
            calls["launch"] = (workspace.value, size.value, executor.value, stream.value, npu.current_device())
            return RUNTIME.ACL_SUCCESS

        runtime = object.__new__(RUNTIME._AclnnRuntime)
        runtime.symbol = lambda name: get_workspace if name.endswith("GetWorkspaceSize") else launch

        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            workspace = runtime.call("aclnnTest", [], FakeDevice(1))

        self.assertIsInstance(workspace, FakeTensor)
        self.assertEqual(torch_module.empty_calls, [((64,), torch_module.uint8, 1, 1)])
        self.assertEqual(npu.stream_queries, [(1, 1)])
        self.assertEqual(calls["launch"], (0xBEEF, 64, 0xCAFE, 0x1001, 1))

    def test_call_aclnn_keeps_build_launch_and_destroy_inside_target_guard(self):
        npu = FakeNpu(current_device=0)
        torch_module = fake_torch(npu)
        events = []

        class FakeDescriptor:
            def __init__(self, runtime, tensor):
                self.ptr = 0xD00D
                events.append(("descriptor", npu.current_device(), tensor.device.index))

            def destroy(self):
                events.append(("destroy", npu.current_device()))

        class FakeRuntime:
            def call(self, name, args, device, *, get_workspace_argtypes=None):
                events.append(("launch", npu.current_device(), device.index))
                return None

        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            with mock.patch.object(RUNTIME, "runtime", return_value=FakeRuntime()):
                with mock.patch.object(RUNTIME, "_AclTensor", FakeDescriptor):
                    output = FakeTensor(2)
                    result = RUNTIME.call_aclnn(
                        "aclnnTest",
                        lambda ctx: [ctx.tensor(FakeTensor(2), "input"), ctx.tensor(output, "output")],
                        output,
                    )

        self.assertIs(result, output)
        self.assertEqual(
            events,
            [
                ("descriptor", 2, 2),
                ("descriptor", 2, 2),
                ("launch", 2, 2),
                ("destroy", 2),
                ("destroy", 2),
            ],
        )
        self.assertEqual(npu.current_device(), 0)

    def test_acl_tensor_preserves_logical_rank_in_storage_shape(self):
        captured = {}

        def acl_create_tensor(
            view_shape,
            view_rank,
            dtype,
            strides,
            storage_offset,
            tensor_format,
            storage_shape,
            storage_rank,
            data_ptr,
        ):
            captured["view_shape"] = tuple(view_shape[index] for index in range(view_rank.value))
            captured["strides"] = tuple(strides[index] for index in range(view_rank.value))
            captured["storage_shape"] = tuple(storage_shape[index] for index in range(storage_rank.value))
            captured["storage_rank"] = storage_rank.value
            return 0xCAFE

        runtime = types.SimpleNamespace(
            acl_create_tensor=acl_create_tensor,
            acl_destroy_tensor=mock.Mock(),
        )
        tensor = FakeDescriptorTensor(0, shape=(2, 3, 5, 7), strides=(105, 35, 7, 1))

        with mock.patch.object(RUNTIME, "dtype_to_acl", return_value=1):
            with mock.patch.object(RUNTIME, "acl_format", return_value=RUNTIME.ACL_FORMAT_ND):
                with mock.patch.object(RUNTIME, "storage_data_ptr", return_value=0x1234):
                    descriptor = RUNTIME._AclTensor(runtime, tensor)

        self.assertEqual(captured["view_shape"], (2, 3, 5, 7))
        self.assertEqual(captured["strides"], (105, 35, 7, 1))
        self.assertEqual(captured["storage_shape"], (2, 3, 5, 7))
        self.assertEqual(captured["storage_rank"], 4)
        descriptor.destroy()
        runtime.acl_destroy_tensor.assert_called_once_with(0xCAFE)


if __name__ == "__main__":
    unittest.main()
