"""Test-only ACL descriptors and symbol inspection; no kernel execution backend.

The negative Host tests must call GetWorkspaceSize directly, before Python
validation and without launching a kernel. Keep this probe outside fla_npu.
"""
import ctypes


class HostProbe:
    def __init__(self):
        from fla_npu import load_ascendc_opapi_libraries
        self.libraries = load_ascendc_opapi_libraries()

    def symbol(self, name):
        # Same overlay -> base -> CANN order as the Stable launcher.
        for library in self.libraries:
            try:
                return getattr(library, name)
            except AttributeError:
                continue
        raise AttributeError(f"Host probe cannot resolve {name}")

    def function(self, name, args, result):
        fn = self.symbol(name)
        fn.argtypes, fn.restype = args, result
        return fn

    def _recent_error_message(self):
        try:
            fn = self.symbol("aclGetRecentErrMsg")
        except AttributeError:
            # Hold the library for the lifetime of the function handle.
            self.error_library = ctypes.CDLL("libascendcl.so")
            fn = self.error_library.aclGetRecentErrMsg
        fn.argtypes, fn.restype = [], ctypes.c_char_p
        message = fn()
        return message.decode("utf-8", errors="replace") if message else ""


class HostDescriptors:
    """Own only the dense ND FP32/BF16 descriptors used by these Host cases."""
    def __init__(self, probe):
        self.probe = probe
        self.resources = []
        self.keepalive = []

    def tensor(self, value, name):
        import torch
        if not value.is_contiguous() or value.storage_offset() != 0:
            raise ValueError(f"{name}: Host probe requires contiguous, offset-zero tensors")
        dtype = {torch.float32: 0, torch.bfloat16: 27}[value.dtype]
        dims = (ctypes.c_int64 * value.ndim)(*value.shape)
        strides = (ctypes.c_int64 * value.ndim)(*value.stride())
        i64p = ctypes.POINTER(ctypes.c_int64)
        create = self.probe.function("aclCreateTensor", [i64p, ctypes.c_uint64,
            ctypes.c_int, i64p, ctypes.c_int64, ctypes.c_int, i64p,
            ctypes.c_uint64, ctypes.c_void_p], ctypes.c_void_p)
        destroy = self.probe.function("aclDestroyTensor", [ctypes.c_void_p], ctypes.c_int)
        ptr = create(dims, value.ndim, dtype, strides, 0, 2, dims, value.ndim,
                     ctypes.c_void_p(value.data_ptr()))
        if not ptr:
            raise RuntimeError(f"aclCreateTensor returned nullptr: {name}")
        self.resources.append((destroy, ptr))
        self.keepalive.extend((value, dims, strides))
        return ctypes.c_void_p(ptr)

    def int_array(self, values):
        if values is None:
            return ctypes.c_void_p()
        array = (ctypes.c_int64 * len(values))(*values)
        create = self.probe.function("aclCreateIntArray",
            [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64], ctypes.c_void_p)
        destroy = self.probe.function("aclDestroyIntArray", [ctypes.c_void_p], ctypes.c_int)
        ptr = create(array, len(values))
        if not ptr:
            raise RuntimeError("aclCreateIntArray returned nullptr")
        self.resources.append((destroy, ptr))
        self.keepalive.append(array)
        return ctypes.c_void_p(ptr)

    def destroy(self):
        for destroy, ptr in reversed(self.resources):
            destroy(ptr)
        self.resources.clear()
        self.keepalive.clear()
