"""Test-only C ABI probe for output descriptors hidden by the public dispatcher."""

import ctypes as C
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from aclnn_error_probe import RawApi, _CallContext as DescriptorContext


def dtype_to_acl(dtype):
    return {torch.float32: 0, torch.float16: 1, torch.bfloat16: 27}[dtype]


def storage_data_ptr(tensor):
    return tensor.untyped_storage().data_ptr()


class ContractApi(RawApi):
    def __init__(self):
        super().__init__()
        self.acl_create_tensor = self.symbol("aclCreateTensor")
        i64p = C.POINTER(C.c_int64)
        self.acl_create_tensor.argtypes = [i64p, C.c_uint64, C.c_int, i64p,
                                         C.c_int64, C.c_int, i64p, C.c_uint64, C.c_void_p]
        self.acl_create_tensor.restype = C.c_void_p
        self.acl_destroy_tensor = self.symbol("aclDestroyTensor")
        self.acl_destroy_tensor.argtypes = [C.c_void_p]
        self.acl_destroy_tensor.restype = C.c_int

    def _aclnn_error(self, name, code):
        return f"{name}: {code}: {self._recent_error_message()}"


def runtime():
    return ContractApi()


class _CallContext(DescriptorContext):
    def __init__(self, api, device):
        super().__init__(api, device)
        self.runtime, self.resources = api, []

    def tensor(self, *args, **kwargs):
        return C.c_void_p(super().tensor(*args, **kwargs))

    def int_array(self, values):
        return C.c_void_p(super().int_array(values))

    def destroy(self):
        for resource in reversed(self.resources):
            resource.destroy()
        self.resources.clear()
        super().destroy()


def call_aclnn(name, build_args, outputs):
    """Launch only valid output-view contract cases; public accuracy uses ascendc."""
    api = runtime()
    device = next(value.device for value in outputs if value is not None)
    ctx = _CallContext(api, device)
    try:
        args = build_args(ctx)
        workspace_size, executor = C.c_uint64(), C.c_void_p()
        get_workspace = api.symbol(name + "GetWorkspaceSize")
        get_workspace.argtypes = [type(value) for value in args] + [
            C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        get_workspace.restype = C.c_int
        status = get_workspace(*args, C.byref(workspace_size), C.byref(executor))
        if status:
            raise RuntimeError(api._aclnn_error(name, status))
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=device)
        launch = api.symbol(name)
        launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        launch.restype = C.c_int
        status = launch(C.c_void_p(workspace.data_ptr()), workspace_size, executor,
                        C.c_void_p(torch.npu.current_stream(device).npu_stream))
        if status:
            raise RuntimeError(api._aclnn_error(name, status))
        torch.npu.synchronize(device)
    finally:
        ctx.destroy()
