"""Regression for a handled CANN query error preceding an inline VDN launch."""
import ctypes
import os
import unittest

import torch


class TestLaunchStatus(unittest.TestCase):
    def test_diagonal_after_handled_memory_query(self):
        try:
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
            self.skipTest("requires torch_npu")
        if not torch.npu.is_available():
            self.skipTest("requires an Ascend NPU")

        from fla_npu.ops.vdn import vdn_native

        device = torch.device("npu", int(os.environ.get("VDN_TEST_DEVICE", 0)))
        with torch.npu.device(device):
            kernels = vdn_native.library(device)
            if kernels is None:
                self.skipTest("requires the optional 910B native kernels")
            two_eye = 2 * torch.eye(128, device=device)
            out = torch.zeros((1, 128, 128), device=device)
            torch.npu.synchronize(device)
            stream = torch.npu.current_stream(device).npu_stream

            acl = ctypes.CDLL("libascendcl.so")
            size_ptr = ctypes.POINTER(ctypes.c_size_t)
            acl.aclrtGetMemInfo.argtypes = [ctypes.c_int, size_ptr, size_ptr]
            acl.aclrtGetMemInfo.restype = ctypes.c_int
            for func in (acl.aclrtPeekAtLastError, acl.aclrtGetLastError):
                func.argtypes = [ctypes.c_int]
                func.restype = ctypes.c_int

            thread_level = 0  # ACL_RT_THREAD_LEVEL
            p2p_huge_1g = 17  # ACL_MEM_P2P_HUGE1G in acl_rt.h
            acl.aclrtGetLastError(thread_level)
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            try:
                status = acl.aclrtGetMemInfo(p2p_huge_1g, ctypes.byref(free), ctypes.byref(total))
                if status == 0:
                    self.skipTest("device supports the memory type used to reproduce the handled error")
                self.assertEqual(acl.aclrtPeekAtLastError(thread_level), status)
                kernels.add_diagonal(out, two_eye, stream)
                torch.testing.assert_close(out[0].cpu(), 2 * torch.eye(128), rtol=0, atol=0)
            finally:
                acl.aclrtGetLastError(thread_level)


if __name__ == "__main__":
    unittest.main()
