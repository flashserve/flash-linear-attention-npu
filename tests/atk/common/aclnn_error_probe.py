"""Test-only raw ACLNN descriptor probe for rejected GetWorkspaceSize calls.

This is not an execution backend: it never invokes the second-stage API or
allocates device workspace. It covers malformed output buffers and null output
pointers that the public Stable dispatcher cannot express. Kept outside the
wheel; normal accuracy/performance cases use the public Stable implementation.
"""
from __future__ import annotations

import ctypes


class RawApi:
    def __init__(self):
        from fla_npu import load_ascendc_opapi_libraries
        self.libraries = load_ascendc_opapi_libraries()

    def symbol(self, name):
        for library in self.libraries:
            try:
                return getattr(library, name)
            except AttributeError:
                continue
        raise RuntimeError(f"raw ABI test symbol is missing: {name}")

    def _recent_error_message(self):
        api = ctypes.CDLL("libascendcl.so").aclGetRecentErrMsg
        api.argtypes, api.restype = [], ctypes.c_char_p
        value = api()
        return value.decode("utf-8", errors="replace") if value else ""


def runtime():
    return RawApi()


class _CallContext:
    """Own only descriptors; preserve caller tensors until descriptors die."""
    def __init__(self, api, device=None):
        self.api, self.device = api, device
        self.owned = []
        self.keepalive = []

    def tensor(self, tensor, name, *, acl_format_override=None,
               storage_shape_override=None):
        if tensor is None:
            return None
        import torch
        types = {torch.float32: 0, torch.float16: 1, torch.int8: 2,
                 torch.int32: 3, torch.uint8: 4, torch.int16: 6,
                 torch.int64: 9, torch.float64: 11, torch.bool: 12,
                 torch.bfloat16: 27}
        dims = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
        strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
        storage = tensor.untyped_storage()
        storage_shape = storage_shape_override or (storage.nbytes() // tensor.element_size(),)
        storage_dims = (ctypes.c_int64 * len(storage_shape))(*storage_shape)
        create = self.api.symbol("aclCreateTensor")
        create.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
                           ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
                           ctypes.c_int64, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
                           ctypes.c_void_p]
        create.restype = ctypes.c_void_p
        fmt = acl_format_override
        if fmt is None:
            fmt = {3: 47, 4: 0, 5: 30}.get(tensor.dim(), 2)
        handle = create(dims, tensor.dim(), types[tensor.dtype], strides,
                        tensor.storage_offset(), fmt, storage_dims,
                        len(storage_shape), storage.data_ptr())
        if not handle:
            raise RuntimeError(f"test could not construct {name} descriptor")
        self.owned.append(("aclDestroyTensor", handle))
        self.keepalive.extend((tensor, dims, strides, storage_dims))
        return handle

    def int_array(self, values):
        if values is None:
            return None
        data = (ctypes.c_int64 * len(values))(*values)
        create = self.api.symbol("aclCreateIntArray")
        create.argtypes, create.restype = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64], ctypes.c_void_p
        handle = create(data, len(values))
        if not handle:
            raise RuntimeError("test could not construct int-array descriptor")
        self.owned.append(("aclDestroyIntArray", handle))
        self.keepalive.append(data)
        return handle

    def destroy(self):
        for name, handle in reversed(self.owned):
            destroy = self.api.symbol(name)
            destroy.argtypes, destroy.restype = [ctypes.c_void_p], ctypes.c_int
            destroy(handle)
        self.owned.clear()
        self.keepalive.clear()


def reject_workspace(symbol, build_args, outputs=None):
    """Run only the first phase; success is a test failure, never a launch."""
    api = runtime()
    context = _CallContext(api)
    workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
    try:
        args = build_args(context)
        function = api.symbol(symbol + "GetWorkspaceSize")
        function.argtypes = [type(value) if isinstance(value, ctypes._SimpleCData)
                             else ctypes.c_void_p for value in args] + [
                                 ctypes.POINTER(ctypes.c_uint64),
                                 ctypes.POINTER(ctypes.c_void_p)]
        function.restype = ctypes.c_int
        api._recent_error_message()
        status = function(*args, ctypes.byref(workspace), ctypes.byref(executor))
        message = api._recent_error_message()
        if status == 0:
            raise AssertionError("raw negative ABI probe unexpectedly succeeded; no kernel was launched")
        if executor.value or workspace.value:
            raise AssertionError("rejected ACLNN call returned an executor or workspace")
        raise RuntimeError(f"{symbol} aclnnStatus={status}: {message}")
    finally:
        context.destroy()
