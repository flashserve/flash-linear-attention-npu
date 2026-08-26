from __future__ import annotations

import importlib
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


try:
    import packaging.version  # noqa: F401
except ImportError:
    from pip._vendor import packaging as vendored_packaging
    from pip._vendor.packaging import version as vendored_packaging_version

    sys.modules["packaging"] = vendored_packaging
    sys.modules["packaging.version"] = vendored_packaging_version


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fla_npu_build_capabilities as capabilities  # noqa: E402

try:
    import setuptools
except ImportError:
    setuptools = None


def _module(path: str, **attributes):
    return types.SimpleNamespace(__file__=path, **attributes)


class LegacyBuildCapabilityTest(unittest.TestCase):
    def test_probes_real_modules_and_cpp_extension_symbols(self) -> None:
        cpp_extension = _module(
            "/fake/torch/utils/cpp_extension.py",
            BuildExtension=object(),
            CppExtension=object(),
        )

        def fake_import(module_name: str):
            if module_name == capabilities.CPP_EXTENSION_MODULE:
                return cpp_extension
            return _module(f"/fake/{module_name.replace('.', '/')}.py")

        with mock.patch.object(
            capabilities.importlib, "import_module", side_effect=fake_import
        ) as import_module:
            probes = capabilities.probe_legacy_build_capabilities()

        self.assertTrue(all(probe.available for probe in probes))
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            [capabilities.CPP_EXTENSION_MODULE, *capabilities.TORCHNPUGEN_MODULES],
        )

    def test_child_module_import_failure_is_reported_with_exception_type(self) -> None:
        broken_module = capabilities.TORCHNPUGEN_MODULES[2]
        cpp_extension = _module(
            "/fake/torch/utils/cpp_extension.py",
            BuildExtension=object(),
            CppExtension=object(),
        )

        def fake_import(module_name: str):
            if module_name == capabilities.CPP_EXTENSION_MODULE:
                return cpp_extension
            if module_name == broken_module:
                raise ModuleNotFoundError("No module named 'missing_dependency'")
            return _module(f"/fake/{module_name.replace('.', '/')}.py")

        with mock.patch.object(
            capabilities.importlib, "import_module", side_effect=fake_import
        ):
            probes = capabilities.probe_legacy_build_capabilities()

        failure = next(
            probe for probe in probes if probe.requirement == broken_module
        )
        self.assertFalse(failure.available)
        self.assertEqual(
            failure.detail,
            "ModuleNotFoundError: No module named 'missing_dependency'",
        )

    def test_missing_cpp_extension_symbol_is_reported(self) -> None:
        cpp_extension = _module(
            "/fake/torch/utils/cpp_extension.py",
            BuildExtension=object(),
        )
        with mock.patch.object(
            capabilities.importlib,
            "import_module",
            return_value=cpp_extension,
        ):
            probes = capabilities.probe_legacy_build_capabilities(
                include_torchnpugen=False
            )

        cpp_extension_probe = next(
            probe
            for probe in probes
            if probe.requirement.endswith(".CppExtension")
        )
        self.assertFalse(cpp_extension_probe.available)
        self.assertIn("required attribute is missing", cpp_extension_probe.detail)

    def test_skipping_torchnpugen_still_checks_cpp_extension(self) -> None:
        cpp_extension = _module(
            "/fake/torch/utils/cpp_extension.py",
            BuildExtension=object(),
            CppExtension=object(),
        )
        with mock.patch.object(
            capabilities.importlib,
            "import_module",
            return_value=cpp_extension,
        ) as import_module:
            probes = capabilities.probe_legacy_build_capabilities(
                include_torchnpugen=False
            )

        self.assertEqual(
            import_module.call_args_list,
            [mock.call(capabilities.CPP_EXTENSION_MODULE)],
        )
        self.assertEqual(len(probes), len(capabilities.CPP_EXTENSION_SYMBOLS))
        self.assertTrue(all(probe.available for probe in probes))

    def test_stream_policy_is_independent_of_import_capabilities(self) -> None:
        for family, minimum in capabilities.KNOWN_GDN_STREAM_FIX_MINIMUMS.items():
            with self.subTest(family=family, minimum=minimum):
                self.assertIsNone(
                    capabilities.torch_npu_gdn_stream_fix_error(minimum)
                )

        self.assertIn(
            "GDN aclnn_extension stream fix",
            capabilities.torch_npu_gdn_stream_fix_error("2.7.1.post4"),
        )
        self.assertIn(
            "unsupported version string",
            capabilities.torch_npu_gdn_stream_fix_error("internal-build"),
        )


class CapabilityIntegrationTest(unittest.TestCase):
    @unittest.skipIf(setuptools is None, "setuptools is required to load setup.py")
    def test_python_only_setup_check_does_not_probe_legacy_dependencies(self) -> None:
        setup_kwargs = {}

        def capture_setup(**kwargs) -> None:
            setup_kwargs.update(kwargs)

        with mock.patch.object(setuptools, "setup", side_effect=capture_setup):
            setup_globals = runpy.run_path(str(REPO_ROOT / "setup.py"))

        check_build_environment = setup_globals["_check_build_environment"]
        function_globals = check_build_environment.__globals__
        legacy_probe = mock.Mock(
            side_effect=AssertionError("legacy capability probe must not run")
        )

        with mock.patch.dict(
            function_globals,
            {"probe_legacy_build_capabilities": legacy_probe},
        ), mock.patch.dict(
            "os.environ",
            {"ASCEND_HOME_PATH": "/fake/cann"},
            clear=True,
        ), mock.patch.object(
            function_globals["shutil"], "which", return_value="/usr/bin/bash"
        ), mock.patch.object(
            function_globals["importlib"],
            "import_module",
            side_effect=AssertionError("Python-only build must not import runtime modules"),
        ):
            check_build_environment()

        legacy_probe.assert_not_called()

    def test_environment_preflight_uses_shared_probe_without_find_spec(self) -> None:
        setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        preflight_source = (SCRIPTS_DIR / "check_npu_env.py").read_text(
            encoding="utf-8"
        )

        for source in (setup_source, preflight_source):
            self.assertIn("probe_legacy_build_capabilities", source)
            self.assertNotIn("find_spec", source)
            self.assertNotIn("TORCHNPUGEN_MODULES =", source)


if __name__ == "__main__":
    unittest.main()
