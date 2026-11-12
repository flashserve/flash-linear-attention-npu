"""Stable-only dispatch and consistency with the current Stable wrappers.

These CPU tests cover Python resolution, aliases and binding without freezing
public signatures. NPU numerical tests remain responsible for actual kernels,
gradients, devices and stream ordering.

Chunk metadata tests use real CPU tensors when torch is installed. Set
FLA_NPU_TEST_DEVICE=npu:0 to also exercise device-to-host metadata conversion;
the dispatcher is still mocked, so this is not a kernel numerical test.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "torch_custom/fla_npu/fla_npu"


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

    def test_public_ops_and_generated_aliases_expose_current_stable_signatures(self):
        self.assertTrue(self.ascendc._ASCENDC_OPS)
        for name in self.ascendc._ASCENDC_OPS:
            with self.subTest(name=name):
                signature = inspect.signature(getattr(self.stable, name))
                self.assertEqual(inspect.signature(getattr(self.ascendc, name)), signature)
                # Handwritten convenience wrappers may intentionally rename inputs.
                if name[4:] not in ("fast_gelu_custom", "causal_conv1d"):
                    self.assertEqual(inspect.signature(getattr(self.ascendc, name[4:])), signature)

    def test_all_ops_and_aliases_preserve_signature_after_first_call(self):
        for name in self.ascendc._ASCENDC_OPS:
            with self.subTest(name=name):
                signature = inspect.signature(getattr(self.stable, name))
                short_name = name[4:]
                alias_signature = inspect.signature(getattr(self.ascendc, short_name))
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
                self.assertEqual(inspect.signature(getattr(self.ascendc, name)), signature)
                self.assertEqual(inspect.signature(getattr(self.ascendc, short_name)), alias_signature)
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
assert ascendc._ASCENDC_OPS
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

    def _metadata_torch(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            self.skipTest("chunk metadata conversion requires torch")
        self.metadata_device = os.environ.get("FLA_NPU_TEST_DEVICE", "cpu")
        if self.metadata_device.startswith("npu"):
            import torch_npu  # noqa: F401
        return torch

    def _metadata_forms(self, torch, values):
        # Sliced int64 input also exercises non-contiguous metadata.
        interleaved = [item for value in values for item in (value, -1)]
        return {
            "list": list(values),
            "tuple": tuple(values),
            "int32": torch.tensor(values, dtype=torch.int32, device=self.metadata_device),
            "int64_view": torch.tensor(
                interleaved, dtype=torch.int64, device=self.metadata_device)[::2],
        }

    def _check_chunk_wrapper(self, name, args, offset, generated_indices,
                             **op_kwargs):
        """Check one public wrapper's metadata at its dispatcher boundary."""
        torch = self._metadata_torch()
        cu_values = [0, 65, 128]
        cu_forms = self._metadata_forms(torch, cu_values)
        explicit = list(generated_indices)
        index_forms = self._metadata_forms(torch, explicit)
        empty = self._metadata_forms(torch, [])
        cases = (
            ("list_missing", cu_forms["list"], None, cu_values,
             generated_indices, True),
            ("tuple_missing", cu_forms["tuple"], None, cu_values,
             generated_indices, True),
            ("int32_missing", cu_forms["int32"], None, cu_values,
             generated_indices, True),
            ("int64_view_empty_indices", cu_forms["int64_view"],
             empty["int32"], cu_values, generated_indices, True),
            ("int32_explicit", cu_forms["int32"], index_forms["int32"],
             cu_values, explicit, False),
            ("int64_view_explicit", cu_forms["int64_view"],
             index_forms["int64_view"], cu_values, explicit, False),
            ("tuple_explicit", cu_forms["tuple"], index_forms["tuple"],
             cu_values, explicit, False),
            ("empty_cu", empty["int64_view"], None, [], None, False),
            ("no_metadata", None, None, None, None, False),
        )
        for case, cu, indices, expected_cu, expected_indices, generated in cases:
            with self.subTest(op=name, case=case):
                dispatch = mock.Mock(return_value=(args[0],) * 9)
                kwargs = {"cu_seqlens": cu, "chunk_indices": indices,
                          **op_kwargs}
                with mock.patch.object(self.stable, "_op", return_value=dispatch) as resolve, \
                        mock.patch.object(self.stable, "_current_stream_ptr", return_value=9), \
                        mock.patch.object(self.stable, "_canonical_chunk_indices",
                                          wraps=self.stable._canonical_chunk_indices) as canonical:
                    getattr(self.ascendc, name[4:])(*args, **kwargs)
                resolve.assert_called_with(name)
                dispatch.assert_called_once()
                self.assertEqual(dispatch.call_args.args[-1], 9)
                self.assertEqual(canonical.call_count, int(generated))
                actual = dispatch.call_args.args[offset:offset + 2]
                self.assertEqual(len(actual), 2)
                for value, expected in zip(actual, (expected_cu, expected_indices)):
                    if expected is None:
                        self.assertIsNone(value)
                    else:
                        self.assertIsInstance(value, torch.Tensor)
                        self.assertEqual(value.device.type, "cpu")
                        self.assertEqual(value.dtype, torch.int64)
                        self.assertEqual(value.tolist(), expected)

    def test_chunk_fwd_h_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16)
        g = torch.empty((1, 2, 128), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_fwd_h", (x, x, x), 9,
                                  [0, 0, 0, 1, 1, 0], g=g, chunk_size=64)

    def test_chunk_gated_delta_rule_fwd_h_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16)
        g = torch.empty((1, 2, 128), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_gated_delta_rule_fwd_h",
                                  (x, x, x, g), 8, [0, 0, 0, 1, 1, 0])

    def test_chunk_kda_fwd_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 128, 2, 128), dtype=torch.bfloat16)
        g = torch.empty_like(x, dtype=torch.float32)
        beta = torch.empty((1, 128, 2), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_kda_fwd",
                                  (x, x, x, g, beta, 0.125), 8,
                                  [0, 0, 0, 1, 1, 0])

    def test_chunk_kda_fwd_chunk32_tensor_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 128, 2, 128), dtype=torch.bfloat16)
        g = torch.empty_like(x, dtype=torch.float32)
        beta = torch.empty((1, 128, 2), dtype=torch.float32)
        a_log = torch.empty((2,), dtype=torch.float32)
        cu = torch.tensor([0, 65, 128], dtype=torch.int32,
                          device=self.metadata_device)
        canonical = [0, 0, 0, 1, 0, 2, 1, 0, 1, 1]
        explicit = torch.tensor(canonical, dtype=torch.int64,
                                device=self.metadata_device)
        for label, indices in (("missing", None), ("explicit", explicit)):
            with self.subTest(indices=label):
                dispatch = mock.Mock(return_value=(x,) * 9)
                device_api = types.SimpleNamespace(
                    get_device_name=lambda _: "Ascend950")
                with mock.patch.object(self.stable, "_op", return_value=dispatch), \
                        mock.patch.object(self.stable, "_current_stream_ptr", return_value=9), \
                        mock.patch.object(torch, "npu", device_api, create=True):
                    self.ascendc.chunk_kda_fwd(
                        x, x, x, g, beta, 0.125, chunk_size=32,
                        cu_seqlens=cu, chunk_indices=indices, safe_gate=True,
                        use_gate_in_kernel=True, A_log=a_log,
                        use_qk_l2norm_in_kernel=True,
                        use_beta_sigmoid_in_kernel=True)
                dispatch.assert_called_once()
                for value, expected in zip(dispatch.call_args.args[8:10],
                                           ([0, 65, 128], canonical)):
                    self.assertEqual(value.device.type, "cpu")
                    self.assertEqual(value.dtype, torch.int64)
                    self.assertEqual(value.tolist(), expected)

    def test_chunk_kda_fwd_prepare_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 128, 2, 128), dtype=torch.bfloat16)
        g = torch.empty_like(x, dtype=torch.float32)
        beta = torch.empty((1, 128, 2), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_kda_fwd_prepare",
                                  (x, x, x, g, beta, 0.125), 7,
                                  [0, 0, 0, 1, 1, 0])

    def test_chunk_kda_fwd_finalize_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16)
        self._check_chunk_wrapper("npu_chunk_kda_fwd_finalize",
                                  (x, x, x, x), 4, [0, 0, 0, 1, 1, 0])

    def test_chunk_gated_delta_rule_fwd_prepare_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16)
        g = torch.empty_like(x, dtype=torch.float32)
        beta = torch.empty((1, 2, 128), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_gated_delta_rule_fwd_prepare",
                                  (x, x, x, g, beta), 7, [0, 0, 0, 1, 1, 0])

    def test_chunk_gated_delta_rule_bwd_metadata(self):
        torch = self._metadata_torch()
        x = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16)
        g = torch.empty_like(x, dtype=torch.float32)
        beta = torch.empty((1, 2, 128), dtype=torch.float32)
        self._check_chunk_wrapper("npu_chunk_gated_delta_rule_bwd",
                                  (x, x, x, g, beta, x, x, 0.125), 14,
                                  [0, 0, 0, 1, 1, 0])

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
                    mock.patch.object(self.stable, "_register_launcher") as register, \
                    mock.patch.object(self.stable, "_check_build_stamp") as stamp:
                self.stable.load()
                self.assertEqual(self.stable._loaded_path, "override-launcher.so")
                torch.ops.load_library.assert_called_once_with("override-launcher.so")
                stamp.assert_called_once_with("override-launcher.so")
                register.assert_called_once_with("override-launcher.so")
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
