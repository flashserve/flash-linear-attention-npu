"""Host-only ATK probes must work without the retired Python runtime."""
import ctypes
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts"
spec = importlib.util.spec_from_file_location("finalize_host_probe", SCRIPTS / "host_probe.py")
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


class FinalizeHostProbeTests(unittest.TestCase):
    def test_overlay_first_with_base_and_cann_fallback(self):
        overlay, base, cann = object(), object(), object()
        libraries = [types.SimpleNamespace(shared=overlay),
                     types.SimpleNamespace(shared=base, base_only=base),
                     types.SimpleNamespace(cann_only=cann)]
        package = types.ModuleType("fla_npu")
        package.load_ascendc_opapi_libraries = lambda: libraries
        with mock.patch.dict(sys.modules, {"fla_npu": package}):
            probe = probe_module.HostProbe()
        self.assertIs(probe.symbol("shared"), overlay)
        self.assertIs(probe.symbol("base_only"), base)
        self.assertIs(probe.symbol("cann_only"), cann)
        with self.assertRaises(AttributeError):
            probe.symbol("missing")
        self.assertFalse(hasattr(probe, "call"))

    def test_nd_descriptor_fields_optional_arrays_and_cleanup(self):
        torch = types.ModuleType("torch")
        torch.float32, torch.bfloat16 = object(), object()
        tensor = types.SimpleNamespace(dtype=torch.bfloat16, ndim=2, shape=(2, 3),
            stride=lambda: (3, 1), is_contiguous=lambda: True,
            storage_offset=lambda: 0, data_ptr=lambda: 1234)
        functions = {name: mock.Mock(return_value=result) for name, result in (
            ("aclCreateTensor", 101), ("aclDestroyTensor", 0),
            ("aclCreateIntArray", 102), ("aclDestroyIntArray", 0))}
        probe = probe_module.HostProbe.__new__(probe_module.HostProbe)
        probe.libraries = [types.SimpleNamespace(**functions)]
        context = probe_module.HostDescriptors(probe)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(context.tensor(tensor, "q").value, 101)
            tensor.dtype = torch.float32
            context.tensor(tensor, "g")
        args = functions["aclCreateTensor"].call_args_list[0].args
        self.assertEqual((list(args[0]), args[1], args[2], list(args[3]),
                          args[4], args[5], list(args[6]), args[7], args[8].value),
                         ([2, 3], 2, 27, [3, 1], 0, 2, [2, 3], 2, 1234))
        self.assertEqual(functions["aclCreateTensor"].call_args.args[2], 0)
        self.assertIsNone(context.int_array(None).value)
        functions["aclCreateIntArray"].assert_not_called()
        self.assertEqual(context.int_array([0, 64]).value, 102)
        self.assertEqual(list(functions["aclCreateIntArray"].call_args.args[0]), [0, 64])
        context.destroy()
        self.assertEqual(functions["aclDestroyTensor"].call_count, 2)
        functions["aclDestroyIntArray"].assert_called_once_with(102)
        self.assertEqual(context.keepalive, [])
        context.destroy()
        self.assertEqual(functions["aclDestroyTensor"].call_count, 2)

    def test_creation_failure_does_not_register_null_resource(self):
        functions = {"aclCreateIntArray": mock.Mock(return_value=None),
                     "aclDestroyIntArray": mock.Mock()}
        probe = probe_module.HostProbe.__new__(probe_module.HostProbe)
        probe.libraries = [types.SimpleNamespace(**functions)]
        context = probe_module.HostDescriptors(probe)
        with self.assertRaisesRegex(RuntimeError, "nullptr"):
            context.int_array([0, 64])
        context.destroy()
        functions["aclDestroyIntArray"].assert_not_called()

    def test_identity_uses_same_ordered_probe(self):
        identity_spec = importlib.util.spec_from_file_location("finalize_identity", SCRIPTS / "arch22_identity.py")
        identity = importlib.util.module_from_spec(identity_spec)
        identity_spec.loader.exec_module(identity)
        probe = mock.Mock()
        with mock.patch.dict(sys.modules, {"host_probe": probe_module}), \
             mock.patch.object(probe_module, "HostProbe", return_value=probe), \
             mock.patch.object(identity, "_arch22_dladdr", side_effect=lambda x: x):
            result = identity._arch22_symbol_sources()
        self.assertEqual(set(result), set(identity._TARGET_SYMBOLS))
        self.assertEqual(probe.symbol.call_args_list,
                         [mock.call(name) for name in identity._TARGET_SYMBOLS])


if __name__ == "__main__":
    unittest.main()
