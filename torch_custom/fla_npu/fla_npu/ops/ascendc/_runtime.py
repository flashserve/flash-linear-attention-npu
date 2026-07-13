# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FLA NPU Ascend C 算子 Python wrapper 共用的 ctypes runtime。

具体算子的 wrapper 只需要使用本模块导出的 ``call_aclnn`` 和 tensor/output
辅助函数，不需要直接实例化 descriptor 或 runtime 类。这里的私有类只负责把
torch tensor 转成 aclnn descriptor，并持有一次 launch 期间需要保活的临时资源。
"""

from __future__ import annotations

import ctypes
import sys
from collections import deque
from typing import Iterable, Optional, Sequence


ACL_SUCCESS = 0
ACL_FORMAT_NCHW = 0
ACL_FORMAT_ND = 2
ACL_FORMAT_NCDHW = 30
ACL_FORMAT_NCL = 47

_ACL_FORMAT_BY_NAME = {
    "NCHW": ACL_FORMAT_NCHW,
    "ND": ACL_FORMAT_ND,
    "NCDHW": ACL_FORMAT_NCDHW,
    "NCL": ACL_FORMAT_NCL,
}

# aclnn launch 是异步的。输出 tensor、workspace buffer，以及为可选 int-array
# 输入临时创建的小 tensor，都必须至少存活到队列里的 kernel 消费完成。这里用
# 一个小 ring 保活即可：普通用户 tensor 会由 torch stream 语义保活，而在常见
# test/example 流程里，deque 回绕前旧 launch 通常已经执行完。
_RECENT_LAUNCH_STORAGE = deque(maxlen=128)


def dtype_to_acl(dtype) -> int:
    import torch

    mapping = {
        torch.float32: 0,  # ACL_FLOAT
        torch.float16: 1,  # ACL_FLOAT16
        torch.int8: 2,  # ACL_INT8
        torch.int32: 3,  # ACL_INT32
        torch.uint8: 4,  # ACL_UINT8
        torch.int16: 6,  # ACL_INT16
        torch.int64: 9,  # ACL_INT64
        torch.float64: 11,  # ACL_DOUBLE
        torch.bool: 12,  # ACL_BOOL
        torch.bfloat16: 27,  # ACL_BF16
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype for aclnn tensor descriptor: {dtype}") from exc


def shape(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def stride(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.stride())


def storage_numel(tensor) -> int:
    try:
        nbytes = tensor.untyped_storage().nbytes()
    except AttributeError:
        nbytes = tensor.storage().nbytes()
    return int(nbytes // tensor.element_size())


def storage_data_ptr(tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


def acl_format(tensor) -> int:
    # torch_npu 能拿到内部 NPU layout tensor 的真实 storage format。但解耦后的
    # runtime 默认不能 import torch_npu，因此这里只在别的代码已经加载 torch_npu
    # 时复用它；否则按 tensor 逻辑维度做保守推断。
    try:
        torch_npu = sys.modules.get("torch_npu")
        if torch_npu is None:
            raise RuntimeError("torch_npu is not loaded")
        npu_format = torch_npu.get_npu_format(tensor)
    except Exception:
        npu_format = None

    if isinstance(npu_format, str):
        acl_format_value = _ACL_FORMAT_BY_NAME.get(npu_format)
        if acl_format_value is not None:
            return acl_format_value
    elif npu_format is not None:
        try:
            return int(npu_format)
        except (TypeError, ValueError):
            pass

    dim = tensor.dim()
    if dim == 3:
        return ACL_FORMAT_NCL
    if dim == 4:
        return ACL_FORMAT_NCHW
    if dim == 5:
        return ACL_FORMAT_NCDHW
    return ACL_FORMAT_ND


def ensure_npu_tensor(tensor, name: str):
    if tensor is None:
        return None
    if not hasattr(tensor, "device") or tensor.device.type != "npu":
        raise TypeError(f"{name} must be a torch NPU tensor, got {type(tensor)!r}.")
    return tensor


def optional_bool(value, default: bool) -> bool:
    return default if value is None else bool(value)


def optional_int(value, default: int) -> int:
    return default if value is None else int(value)


def optional_float(value, default: float) -> float:
    return default if value is None else float(value)


def chunk_num(total_tokens: int, chunk_size: int, chunk_indices: Optional[Sequence[int]]) -> int:
    if chunk_indices is not None:
        return len(chunk_indices) // 2
    return (total_tokens + chunk_size - 1) // chunk_size


def current_stream_ptr() -> int:
    import torch

    stream = torch.npu.current_stream()
    return int(getattr(stream, "npu_stream"))


def empty_like(tensor, *, dtype=None):
    import torch

    dtype = dtype or tensor.dtype
    return torch.empty_like(tensor, dtype=dtype)


def empty(shape_: Iterable[int], like, *, dtype=None):
    import torch

    return torch.empty(tuple(int(dim) for dim in shape_), device=like.device, dtype=dtype or like.dtype)


def zeros(shape_: Iterable[int], like, *, dtype=None):
    import torch

    return torch.zeros(tuple(int(dim) for dim in shape_), device=like.device, dtype=dtype or like.dtype)


class _AclTensor:
    """持有一次 aclnn 调用生命周期内的 aclTensor descriptor。"""

    def __init__(self, runtime: "_AclnnRuntime", tensor):
        tensor = ensure_npu_tensor(tensor, "tensor")
        self._runtime = runtime
        self._tensor = tensor
        self._shape = (ctypes.c_int64 * tensor.dim())(*shape(tensor))
        self._stride = (ctypes.c_int64 * tensor.dim())(*stride(tensor))
        self._storage_shape = (ctypes.c_int64 * 1)(storage_numel(tensor))
        self.ptr = runtime.acl_create_tensor(
            self._shape,
            ctypes.c_uint64(tensor.dim()),
            ctypes.c_int(dtype_to_acl(tensor.dtype)),
            self._stride,
            ctypes.c_int64(int(tensor.storage_offset())),
            ctypes.c_int(acl_format(tensor)),
            self._storage_shape,
            ctypes.c_uint64(1),
            ctypes.c_void_p(storage_data_ptr(tensor)),
        )
        if not self.ptr:
            raise RuntimeError("aclCreateTensor returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_tensor(self.ptr)
        self.ptr = None


class _AclIntArray:
    """为可选 Python sequence 输入持有一个 aclIntArray descriptor。"""

    def __init__(self, runtime: "_AclnnRuntime", values: Optional[Sequence[int]]):
        self._runtime = runtime
        self.ptr = None
        if values is None:
            return
        values = tuple(int(value) for value in values)
        if not values:
            return
        self._values = (ctypes.c_int64 * len(values))(*values)
        self.ptr = runtime.acl_create_int_array(self._values, ctypes.c_uint64(len(values)))
        if not self.ptr:
            raise RuntimeError("aclCreateIntArray returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_int_array(self.ptr)
        self.ptr = None


class _CallContext:
    """跟踪构造一次调用时创建的 descriptor 和临时 tensor。"""

    def __init__(self, runtime: "_AclnnRuntime"):
        self.runtime = runtime
        self.resources = []
        self.keepalive_tensors = []

    def tensor(self, tensor, name: str = "tensor") -> ctypes.c_void_p:
        if tensor is None:
            return ctypes.c_void_p()
        desc = _AclTensor(self.runtime, ensure_npu_tensor(tensor, name))
        self.resources.append(desc)
        return ctypes.c_void_p(desc.ptr)

    def int_array(self, values: Optional[Sequence[int]]) -> ctypes.c_void_p:
        desc = _AclIntArray(self.runtime, values)
        self.resources.append(desc)
        return ctypes.c_void_p(desc.ptr or 0)

    def int_tensor(self, values: Optional[Sequence[int]], device) -> ctypes.c_void_p:
        if values is None:
            return ctypes.c_void_p()
        import torch

        # 部分 aclnn API 把可选 index array 建模成 Tensor 而不是 aclIntArray。
        # 这里在目标 device 上创建这些小 tensor，并让它们在异步 launch 期间保活。
        tensor = torch.as_tensor(tuple(int(value) for value in values), dtype=torch.int64, device=device)
        self.keepalive_tensors.append(tensor)
        return self.tensor(tensor, "int tensor")

    def destroy(self) -> None:
        for resource in reversed(self.resources):
            resource.destroy()
        self.resources.clear()


class _AclnnRuntime:
    """从已打包的 custom op_api 动态库里解析并缓存符号。"""

    def __init__(self):
        import fla_npu

        self._libraries = fla_npu.load_ascendc_opapi_libraries()
        self._symbols = {}
        self.acl_create_tensor = self.symbol("aclCreateTensor")
        self.acl_create_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.acl_create_tensor.restype = ctypes.c_void_p

        self.acl_destroy_tensor = self.symbol("aclDestroyTensor")
        self.acl_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.acl_destroy_tensor.restype = ctypes.c_int

        self.acl_create_int_array = self.symbol("aclCreateIntArray")
        self.acl_create_int_array.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
        self.acl_create_int_array.restype = ctypes.c_void_p

        self.acl_destroy_int_array = self.symbol("aclDestroyIntArray")
        self.acl_destroy_int_array.argtypes = [ctypes.c_void_p]
        self.acl_destroy_int_array.restype = ctypes.c_int

    def symbol(self, name: str):
        if name in self._symbols:
            return self._symbols[name]
        for library in self._libraries:
            try:
                symbol = getattr(library, name)
            except AttributeError:
                continue
            self._symbols[name] = symbol
            return symbol
        raise AttributeError(f"Unable to resolve aclnn symbol {name}.")

    def call(
        self,
        name: str,
        args: Sequence[object],
        outputs: Sequence[object],
        *,
        get_workspace_argtypes: Optional[Sequence[object]] = None,
    ):
        # aclnn 调用约定是两段式：第一段创建 executor 并返回 workspace 大小，
        # 第二段传入 workspace 和 stream pointer 发起实际执行。workspace 使用
        # 普通 torch NPU tensor 分配，从而跟随当前 PyTorch allocator 和 device。
        get_workspace = self.symbol(f"{name}GetWorkspaceSize")
        launch = self.symbol(name)
        get_workspace.restype = ctypes.c_int
        if get_workspace_argtypes is not None:
            get_workspace.argtypes = list(get_workspace_argtypes)
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()

        ret = get_workspace(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{name}GetWorkspaceSize failed with aclnnStatus={ret}.")

        workspace = None
        workspace_ptr = ctypes.c_void_p()
        if workspace_size.value:
            import torch

            device = outputs[0].device
            workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=device)
            workspace_ptr = ctypes.c_void_p(int(workspace.data_ptr()))

        ret = launch(
            workspace_ptr,
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(current_stream_ptr()),
        )
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{name} failed with aclnnStatus={ret}.")
        return workspace


_RUNTIME: Optional[_AclnnRuntime] = None


def runtime() -> _AclnnRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _AclnnRuntime()
    return _RUNTIME


def finalize(outputs, workspace, keepalive_tensors):
    _RECENT_LAUNCH_STORAGE.append((tuple(outputs), workspace, tuple(keepalive_tensors)))


def call_aclnn(name: str, build_args, outputs, *, get_workspace_argtypes=None):
    aclnn_runtime = runtime()
    ctx = _CallContext(aclnn_runtime)
    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    try:
        args = build_args(ctx)
        workspace = aclnn_runtime.call(
            name,
            args,
            outputs_tuple,
            get_workspace_argtypes=get_workspace_argtypes,
        )
    finally:
        ctx.destroy()
    finalize(outputs_tuple, workspace, ctx.keepalive_tensors)
    return outputs
