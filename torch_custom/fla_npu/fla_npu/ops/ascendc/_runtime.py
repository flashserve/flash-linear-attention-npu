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
import os
import re
import sys
from contextlib import contextmanager
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


def _npu_device_index(device) -> int:
    if getattr(device, "type", None) != "npu":
        raise TypeError(f"Expected a torch NPU device, got {device!r}.")
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)

    import torch

    return int(torch.npu.current_device())


@contextmanager
def _npu_device_guard(device):
    """在一次 aclnn 调用期间切到 tensor 所在 NPU，并在退出时恢复。"""

    import torch

    device_index = _npu_device_index(device)
    device_guard = getattr(torch.npu, "device", None)
    if device_guard is not None:
        with device_guard(device_index):
            yield device_index
        return

    # 兼容仅提供 current_device/set_device 的早期 NPU Python runtime。
    previous_index = int(torch.npu.current_device())
    if previous_index != device_index:
        torch.npu.set_device(device_index)
    try:
        yield device_index
    finally:
        if previous_index != device_index:
            torch.npu.set_device(previous_index)


def current_stream_ptr(device) -> int:
    import torch

    device_index = _npu_device_index(device)
    try:
        stream = torch.npu.current_stream(device_index)
    except TypeError:
        # 兼容只接受无参数 current_stream() 的早期实现。调用方已经持有
        # device guard；这里保留 guard 使该 helper 单独调用时也不会取错卡。
        with _npu_device_guard(device):
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

    def __init__(
        self,
        runtime: "_AclnnRuntime",
        tensor,
        *,
        acl_format_override: Optional[int] = None,
        storage_shape_override: Optional[Iterable[int]] = None,
    ):
        tensor = ensure_npu_tensor(tensor, "tensor")
        self._runtime = runtime
        self._tensor = tensor
        self._shape = (ctypes.c_int64 * tensor.dim())(*shape(tensor))
        self._stride = (ctypes.c_int64 * tensor.dim())(*stride(tensor))
        descriptor_storage_shape = (
            (storage_numel(tensor),)
            if storage_shape_override is None
            else tuple(int(dim) for dim in storage_shape_override)
        )
        self._storage_shape = (ctypes.c_int64 * len(descriptor_storage_shape))(
            *descriptor_storage_shape
        )
        descriptor_format = (
            acl_format(tensor)
            if acl_format_override is None
            else int(acl_format_override)
        )
        self.ptr = runtime.acl_create_tensor(
            self._shape,
            ctypes.c_uint64(tensor.dim()),
            ctypes.c_int(dtype_to_acl(tensor.dtype)),
            self._stride,
            ctypes.c_int64(int(tensor.storage_offset())),
            ctypes.c_int(descriptor_format),
            self._storage_shape,
            ctypes.c_uint64(len(descriptor_storage_shape)),
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

    def __init__(self, runtime: "_AclnnRuntime", device):
        self.runtime = runtime
        self.device = device
        self.device_index = _npu_device_index(device)
        self.resources = []
        self.keepalive_tensors = []

    def tensor(
        self,
        tensor,
        name: str = "tensor",
        *,
        acl_format_override: Optional[int] = None,
        storage_shape_override: Optional[Iterable[int]] = None,
    ) -> ctypes.c_void_p:
        if tensor is None:
            return ctypes.c_void_p()
        tensor = ensure_npu_tensor(tensor, name)
        tensor_device_index = _npu_device_index(tensor.device)
        if tensor_device_index != self.device_index:
            raise ValueError(
                f"{name} must be on npu:{self.device_index}, got npu:{tensor_device_index}; "
                "one aclnn call cannot mix tensors from different NPU devices."
            )
        if acl_format_override is None and storage_shape_override is None:
            desc = _AclTensor(self.runtime, tensor)
        else:
            desc = _AclTensor(
                self.runtime,
                tensor,
                acl_format_override=acl_format_override,
                storage_shape_override=storage_shape_override,
            )
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
    """从 custom、CANN op_api 句柄中按优先级解析并缓存符号。"""

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
        device,
        *,
        get_workspace_argtypes: Optional[Sequence[object]] = None,
    ):
        # aclnn 调用约定是两段式：第一段创建 executor 并返回 workspace 大小，
        # 第二段传入 workspace 和 stream pointer 发起实际执行。workspace 使用
        # 普通 torch NPU tensor 分配，从而跟随目标 device 的 PyTorch allocator。
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

            workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=device)
            workspace_ptr = ctypes.c_void_p(int(workspace.data_ptr()))

        ret = launch(
            workspace_ptr,
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(current_stream_ptr(device)),
        )
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{name} failed with aclnnStatus={ret}.")
        return workspace


_RUNTIME: Optional[_AclnnRuntime] = None


# ---------------------------------------------------------------------------
# CANN capabilities
# ---------------------------------------------------------------------------
# ``aclnnCausalConv1d`` is handed a descriptor for its optional ``convStates``
# input, but whether its tiling ever sees that descriptor's *view description*
# is decided by the CANN runtime rather than by the operator.  Measured with one
# OPP built from one source revision, swapping only the toolkit:
#
#   9.1.0  the input reaches the tiling with no strides (the operator logs
#          ``isview=0 / stride_null=1``), so a block-strided state is read and
#          written at the dense offsets: the update lands in the gaps between
#          blocks and the caller keeps its stale rows;
#   9.2.0  the same call addresses a block-strided view exactly where the caller
#          keeps it (bit-exact against a dense copy) and rejects a view whose
#          innermost stride is not 1.
#
# The version therefore decides whether a non-dense conv_state may be handed
# over as it is.  ``FLA_NPU_CONV1D_VIEW_STATE`` overrides the verdict for a
# runtime whose version cannot be read; an undecided runtime stages the state
# through a dense copy, which is correct everywhere.
_CANN_VERSION_FILES = ("opp/version.info", "compiler/version.info")
_CANN_VERSION_HOMES = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
_CONV1D_VIEW_STATE_ENV = "FLA_NPU_CONV1D_VIEW_STATE"
_CONV1D_VIEW_STATE_MIN = (9, 2, 0)
_CONV1D_VIEW_STATE: Optional[bool] = None


def cann_version() -> Optional[tuple]:
    """The toolkit version the environment points at, or ``None``.

    Read from the installed ``version.info`` rather than from the runtime: the
    obvious entry point, ``aclsysGetVersionStr``, segfaults on both 9.1.0 and
    9.2.0 when it is called outside ``aclInit``, and the answer has to be
    available from the first operator call.
    """

    for variable in _CANN_VERSION_HOMES:
        home = os.environ.get(variable)
        if not home:
            continue
        for relative in _CANN_VERSION_FILES:
            try:
                with open(os.path.join(home, relative)) as handle:
                    text = handle.read()
            except OSError:
                continue
            match = re.search(r"^Version\s*=\s*([0-9]+(?:\.[0-9]+)*)",
                              text, re.MULTILINE)
            if match:
                parts = tuple(int(part) for part in match.group(1).split("."))
                return (parts + (0, 0, 0))[:3]
    return None


def conv1d_view_state_supported() -> bool:
    """Whether a non-dense conv_state may be passed to the operator as a view."""

    global _CONV1D_VIEW_STATE
    if _CONV1D_VIEW_STATE is None:
        override = os.environ.get(_CONV1D_VIEW_STATE_ENV)
        if override:
            _CONV1D_VIEW_STATE = override.strip().lower() not in (
                "0", "false", "no", "off")
        else:
            version = cann_version()
            _CONV1D_VIEW_STATE = (version is not None
                                  and version >= _CONV1D_VIEW_STATE_MIN)
            if not _CONV1D_VIEW_STATE:
                spelled = (".".join(str(part) for part in version)
                           if version else "unknown")
                import warnings

                warnings.warn(
                    f"the CANN runtime in use ({spelled}) does not hand "
                    "aclnnCausalConv1d the layout of a conv_state, so a "
                    "non-contiguous conv cache is staged through a dense copy "
                    f"on every call; set {_CONV1D_VIEW_STATE_ENV}=1 to skip "
                    "the staging on a runtime that does (CANN >= 9.2.0)",
                    RuntimeWarning,
                    stacklevel=3,
                )
    return _CONV1D_VIEW_STATE


def conv_state_needs_dense_copy(conv_state) -> bool:
    """Whether a causal_conv1d state has to be staged through a dense copy.

    The operator addresses the state as (block stride, row stride, 1), which it
    can only do where the runtime hands its tiling the descriptor's view
    description.  Both layers that build that descriptor (the ctypes reference
    and the stable adapter) ask this one question, so the two cannot drift:

      * a state without a dense innermost dimension, or with non-positive outer
        strides, cannot be addressed by any runtime -> stage it;
      * a non-contiguous state on a runtime that drops the view -> stage it
        (that is what both layers did for *every* non-contiguous state before
        the boundary was measured);
      * everything else crosses as the descriptor's own view.
    """

    if conv_state is None or conv_state.numel() == 0:
        return False
    try:
        dims = len(conv_state.shape)
        stride = tuple(conv_state.stride())
        contiguous = conv_state.is_contiguous()
    except AttributeError:
        return False
    if dims != 3:
        return True
    if not (stride[2] == 1 and stride[0] > 0 and stride[1] > 0):
        return True
    if conv1d_view_state_supported():
        return False
    return not contiguous


def runtime() -> _AclnnRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _AclnnRuntime()
    return _RUNTIME


def _call_device(outputs: Sequence[object]):
    device = None
    device_index = None
    for index, output in enumerate(outputs):
        if output is None:
            continue
        output = ensure_npu_tensor(output, f"output[{index}]")
        output_device_index = _npu_device_index(output.device)
        if device_index is None:
            device = output.device
            device_index = output_device_index
        elif output_device_index != device_index:
            raise ValueError(
                f"output[{index}] must be on npu:{device_index}, got npu:{output_device_index}; "
                "one aclnn call cannot mix tensors from different NPU devices."
            )
    if device is None:
        raise ValueError("An aclnn call must have at least one NPU output tensor.")
    return device


def call_aclnn(name: str, build_args, outputs, *, get_workspace_argtypes=None):
    aclnn_runtime = runtime()
    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    device = _call_device(outputs_tuple)
    ctx = _CallContext(aclnn_runtime, device)
    with _npu_device_guard(device):
        try:
            args = build_args(ctx)
            # runtime.call 在目标 current stream 上分配 workspace 并把 kernel
            # enqueue 到同一 stream。调用返回后可立即释放 Python 引用；NPU
            # caching allocator 会按 stream 生命周期管理底层 block 的安全复用。
            aclnn_runtime.call(
                name,
                args,
                device,
                get_workspace_argtypes=get_workspace_argtypes,
            )
        finally:
            ctx.destroy()
    return outputs
