"""Stable-only dispatch and the complete pre-migration public API contract.

The frozen signatures come from main c9e2f1db plus original overlay 5c53a2a3, not from the implementation under
test. These CPU tests cover Python resolution and binding; NPU numerical tests
remain responsible for actual kernels, gradients, devices and stream ordering.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "torch_custom/fla_npu/fla_npu"
SNAPSHOT = json.loads(
    (ROOT / "tests/stable_public_api.json").read_text(encoding="utf-8")
)["inspect_signatures"]


class StableOnlyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.old_modules = {
            name: module for name, module in sys.modules.items()
            if name == "fla_npu" or name.startswith("fla_npu.")
        }
        for name in self.old_modules:
            sys.modules.pop(name)
        package = types.ModuleType("fla_npu")
        package.__path__ = [str(PACKAGE)]
        package.load_ascendc_opapi_libraries = lambda: []
        sys.modules["fla_npu"] = package
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for name in ("FLA_NPU_STABLE_ABI", "FLA_NPU_STABLE_VALIDATE"):
            os.environ.pop(name, None)
        self.ascendc = importlib.import_module("fla_npu.ops.ascendc")
        self.stable = self.ascendc._stable

    def tearDown(self):
        self.env.stop()
        for name in list(sys.modules):
            if name == "fla_npu" or name.startswith("fla_npu."):
                sys.modules.pop(name)
        sys.modules.update(self.old_modules)

    def test_all_public_signatures_equal_frozen_main(self):
        self.assertEqual(len(self.ascendc._ASCENDC_OPS), 38)
        self.assertEqual(len(SNAPSHOT), 76)
        for name, signature in SNAPSHOT.items():
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(getattr(self.ascendc, name))), signature)
        historical = json.loads((ROOT / "tests/stable_public_api.json").read_text(encoding="utf-8"))
        self.assertEqual(set(historical["operators"]), set(self.ascendc._ASCENDC_OPS))

    def test_all_stable_wrappers_own_frozen_signature(self):
        for name in self.ascendc._ASCENDC_OPS:
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(getattr(self.stable, name))), SNAPSHOT[name])

    def test_all_ops_and_aliases_preserve_signature_after_first_call(self):
        for name in self.ascendc._ASCENDC_OPS:
            with self.subTest(name=name):
                signature = inspect.signature(getattr(self.stable, name))
                received = []

                def resolved(*args, **kwargs):
                    received.append((args, kwargs))
                    return "resolved"

                resolved.__signature__ = signature
                wrapper = getattr(self.ascendc, name)
                with mock.patch.object(self.ascendc, "_get_direct_op", return_value=resolved) as resolve:
                    self.assertEqual(wrapper("sentinel", keyword="unchanged"), "resolved")
                    self.assertEqual(wrapper("again"), "resolved")
                    resolve.assert_called_once_with(name)
                self.assertEqual(received[0], (("sentinel",), {"keyword": "unchanged"}))
                self.assertIs(getattr(self.ascendc, name), resolved)
                self.assertEqual(str(inspect.signature(getattr(self.ascendc, name))), SNAPSHOT[name])
                short_name = name[4:]
                self.assertEqual(str(inspect.signature(getattr(self.ascendc, short_name))), SNAPSHOT[short_name])
                if short_name not in ("fast_gelu_custom", "causal_conv1d"):
                    self.assertIs(getattr(self.ascendc, short_name), resolved)

    def test_all_signatures_bind_defaults_keywords_and_reject_bad_calls(self):
        for name in self.ascendc._ASCENDC_OPS:
            with self.subTest(name=name):
                signature = inspect.signature(getattr(self.stable, name))
                required = {
                    key: object() for key, param in signature.parameters.items()
                    if param.default is inspect.Parameter.empty
                }
                bound = signature.bind(**required)
                bound.apply_defaults()
                self.assertEqual(set(bound.arguments), set(signature.parameters))
                positional, keywords = [], {}
                for key, param in signature.parameters.items():
                    if param.kind == inspect.Parameter.KEYWORD_ONLY:
                        keywords[key] = bound.arguments[key]
                    else:
                        positional.append(bound.arguments[key])
                self.assertEqual(signature.bind(*positional, **keywords).arguments, bound.arguments)
                for key, param in signature.parameters.items():
                    if param.default is not inspect.Parameter.empty:
                        supplied = dict(required, **{key: None})
                        self.assertIsNone(signature.bind(**supplied).arguments[key])
                with self.assertRaises(TypeError):
                    signature.bind(**required, unexpected_keyword=True)
                with self.assertRaises(TypeError):
                    signature.bind(*([object()] * (len(signature.parameters) + 1)))
                if required:
                    incomplete = dict(required)
                    incomplete.pop(next(iter(incomplete)))
                    with self.assertRaises(TypeError):
                        signature.bind(**incomplete)

    def test_first_resolution_requires_exact_registration_and_is_cached(self):
        name = "npu_fast_gelu_custom"
        with mock.patch.object(self.stable, "_op") as registration:
            direct = self.ascendc._get_direct_op(name)
            self.assertIs(direct, self.stable.npu_fast_gelu_custom)
            self.assertIs(self.ascendc._get_direct_op(name), direct)
            registration.assert_called_once_with(name)
        self.assertEqual(self.ascendc.BACKENDS, {name: "stable"})

    def test_resolution_failure_retains_cause_and_never_returns_a_fallback(self):
        for cause in (OSError("library absent"), RuntimeError("stamp mismatch"), AttributeError("operator absent")):
            with self.subTest(cause=str(cause)), mock.patch.object(self.stable, "_op", side_effect=cause):
                with self.assertRaisesRegex(RuntimeError, "required Stable-ABI launcher") as caught:
                    self.ascendc._get_direct_op("npu_fast_gelu_custom")
                self.assertIs(caught.exception.__cause__, cause)
                self.assertIn(str(cause), str(caught.exception))
                self.assertFalse(self.ascendc.BACKENDS)

    def test_removed_modes_fail_before_runtime_loading(self):
        for flag, value in (("FLA_NPU_STABLE_ABI", "ctypes"),
                            ("FLA_NPU_STABLE_ABI", "legacy"),
                            ("FLA_NPU_STABLE_ABI", "0"),
                            ("FLA_NPU_STABLE_ABI", "typo"),
                            ("FLA_NPU_STABLE_VALIDATE", "1"),
                            ("FLA_NPU_STABLE_VALIDATE", "typo")):
            with self.subTest(flag=flag, value=value), mock.patch.dict(os.environ, {flag: value}):
                with mock.patch.object(self.stable, "_op") as load:
                    with self.assertRaisesRegex(RuntimeError, flag):
                        self.ascendc._get_direct_op("npu_fast_gelu_custom")
                    load.assert_not_called()

    def test_stable_compatibility_modes_and_disabled_validation_are_accepted(self):
        policy = importlib.import_module("fla_npu._backend_policy")
        for mode in ("", "stable", "1", "TRUE", " yes ", "on"):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {"FLA_NPU_STABLE_ABI": mode}):
                policy.validate_backend_environment()
        for value in ("", "0", "false", "no", "off"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"FLA_NPU_STABLE_VALIDATE": value}):
                policy.validate_backend_environment()

    def test_removed_modules_and_extension_apis_are_absent(self):
        for name in ("_runtime.py", "_aclnn_ctypes.py"):
            self.assertFalse((PACKAGE / "ops/ascendc" / name).exists())
        for name in ("install_legacy_torch_ops_warning",):
            self.assertFalse(hasattr(self.ascendc, name))
        self.assertEqual(self.ascendc.FALLBACKS, {})
        self.assertTrue(callable(self.ascendc.install_torch_npu_ops_compat))
        root_source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        for old_name in ("load_legacy_torch_ops", "custom_aclnn_extension_lib", "is_legacy_torch_ops_loaded"):
            self.assertNotIn(old_name, root_source)

    def test_public_wrapper_import_does_not_import_torch(self):
        script = """
import importlib, sys, types
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname == 'torch_npu':
            raise AssertionError('wrapper import tried to load ' + fullname)
sys.meta_path.insert(0, RejectTorch())
package = types.ModuleType('fla_npu')
package.__path__ = [sys.argv[1]]
package.load_ascendc_opapi_libraries = lambda: []
sys.modules['fla_npu'] = package
ascendc = importlib.import_module('fla_npu.ops.ascendc')
assert len(ascendc._ASCENDC_OPS) == 38
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script, str(PACKAGE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preprocess_default_and_explicit_none_reach_same_adapter_arguments(self):
        op = self.stable.npu_pre_process_fwd_kernel_merged
        with mock.patch.object(self.stable, "_op") as resolve, \
             mock.patch.object(self.stable, "_current_stream_ptr", return_value=9), \
             mock.patch.object(self.stable, "_host_ints", side_effect=lambda v: v):
            # The wrapper's input checks only need shape/layout metadata here.
            tensor = types.SimpleNamespace(ndim=4, shape=(1, 64, 1, 64))
            try:
                op(tensor, tensor, tensor, tensor, cu_seqlens=[0, 64])
            except (AttributeError, TypeError) as exc:
                self.fail(f"preprocess test input no longer satisfies wrapper: {exc}")
            default_call = resolve.return_value.call_args
            resolve.return_value.reset_mock()
            op(tensor, tensor, tensor, tensor, cu_seqlens=[0, 64], chunk_size=None)
            self.assertEqual(resolve.return_value.call_args, default_call)

    def test_shared_parameter_helpers_keep_existing_semantics(self):
        params = importlib.import_module("fla_npu.ops.ascendc._parameters")
        self.assertIs(params.optional_bool(None, True), True)
        self.assertIs(params.optional_bool(0, True), False)
        self.assertIs(params.optional_bool(2, False), True)
        self.assertEqual(params.optional_float(None, -5.0), -5.0)
        self.assertEqual(params.optional_float("2.5", 0.0), 2.5)
        self.assertEqual([params.ACL_FORMAT_NCHW, params.ACL_FORMAT_ND, params.ACL_FORMAT_NCDHW, params.ACL_FORMAT_NCL], [0, 2, 30, 47])

    def test_overlay_capability_checks_actual_override_library(self):
        class Probe:
            def __init__(self, value): self.value = value
            def __call__(self): return self.value
        env = {"FLA_NPU_OP_API_LIBS": os.pathsep.join(("overlay.so", "base.so")),
               "FLA_NPU_STABLE_LIB": "override-launcher.so"}
        with mock.patch.dict(os.environ, env):
            for value in (0, -1, None):
                library = object() if value is None else types.SimpleNamespace(
                    fla_npu_stable_overlay_order_version=Probe(value))
                with mock.patch("ctypes.CDLL", return_value=library) as load:
                    with self.assertRaisesRegex(RuntimeError, "cannot verify ordered overlay"):
                        self.stable._check_overlay_order_capability("override-launcher.so")
                    load.assert_called_once_with("override-launcher.so")
            library = types.SimpleNamespace(fla_npu_stable_overlay_order_version=Probe(1))
            torch = types.SimpleNamespace(ops=types.SimpleNamespace(load_library=mock.Mock()))
            with mock.patch("ctypes.CDLL", return_value=library) as load, \
                    mock.patch.object(self.stable, "_loaded_path", None), \
                    mock.patch.object(self.stable, "_modules", return_value=(torch, None)), \
                    mock.patch.object(self.stable, "_init_hot"), \
                    mock.patch.object(self.stable, "_check_build_stamp") as stamp:
                self.stable.load()
                self.assertEqual(self.stable._loaded_path, "override-launcher.so")
                torch.ops.load_library.assert_called_once_with("override-launcher.so")
                stamp.assert_called_once_with("override-launcher.so")
                load.assert_called_once_with("override-launcher.so")

    def test_torch_npu_compatibility_aliases_are_stable_public_wrappers(self):
        module = types.ModuleType("torch_npu")
        with mock.patch.dict(sys.modules, {"torch_npu": module}):
            self.ascendc.install_torch_npu_ops_compat()
            for name in self.ascendc._TORCH_NPU_COMPAT_OPS:
                self.assertIs(getattr(module.ops, name), getattr(self.ascendc, name))
                self.assertIs(getattr(module.ops, name[4:]), getattr(self.ascendc, name[4:]))
        self.assertEqual(self.ascendc.FALLBACKS, {})

    def test_build_stamp_missing_unknown_mismatch_and_matching(self):
        class Probe:
            def __init__(self, value):
                self.value = value

            def __call__(self):
                return self.value

        stamp_name = "fla_npu.ops.ascendc._stable_hash"
        with mock.patch.dict(sys.modules, {stamp_name: None}):
            with self.assertRaisesRegex(RuntimeError, "no valid Stable-ABI build stamp"):
                self.stable._check_build_stamp("launcher.so")
        stamp = types.ModuleType(stamp_name)
        stamp.SOURCE_HASH = "expected"
        with mock.patch.dict(sys.modules, {stamp_name: stamp}):
            for actual in (b"unknown", b"different"):
                lib = types.SimpleNamespace(fla_npu_stable_source_hash=Probe(actual))
                with mock.patch("ctypes.CDLL", return_value=lib):
                    with self.assertRaisesRegex(RuntimeError, "different adapter sources"):
                        self.stable._check_build_stamp("launcher.so")
            with mock.patch("ctypes.CDLL", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "cannot report its build stamp"):
                    self.stable._check_build_stamp("launcher.so")
            lib = types.SimpleNamespace(fla_npu_stable_source_hash=Probe(b"expected"))
            with mock.patch("ctypes.CDLL", return_value=lib):
                self.stable._check_build_stamp("launcher.so")


if __name__ == "__main__":
    unittest.main()
