from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_libcust_opapi_md5.py"

# Relative paths (inside a vendor root) of the artifacts the script compares.
VENDOR_LIB_REL = Path("op_api") / "lib" / "libcust_opapi.so"
VENDOR_KERNEL_REL = Path("op_impl") / "ai_core" / "tbe" / "kernel"


def _fake_vendor(root: Path) -> Path:
    """Create a minimal vendor OPP under ``root`` and return the vendor dir."""
    vendor = root / "vendors" / "fla_npu_transformer"
    lib = vendor / VENDOR_LIB_REL
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"lib-v1")
    kernel_dir = vendor / VENDOR_KERNEL_REL / "ascend910b" / "sample"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "Sample_v1.o").write_bytes(b"kernel-v1")
    return vendor


def _fake_fla_npu_package(root: Path) -> Path:
    """Create a minimal installed fla_npu package (opp/ + wrapper py files)."""
    package = root / "fla_npu"
    package.mkdir(parents=True, exist_ok=True)
    _fake_vendor(package / "opp")
    (package / "__init__.py").write_text("VERSION = '1'\n", encoding="utf-8")
    ops_dir = package / "ops" / "ascendc"
    ops_dir.mkdir(parents=True)
    (ops_dir / "__init__.py").write_text("", encoding="utf-8")
    (ops_dir / "_aclnn_ctypes.py").write_text("", encoding="utf-8")
    (ops_dir / "_runtime.py").write_text("", encoding="utf-8")
    return package


class VerifyLibcustOpapiMd5Test(unittest.TestCase):
    """Test the md5 comparison script in a self-contained fake environment.

    The script must work without a CANN env or an importable/loadable
    fla_npu, so every test runs it in a clean subprocess with only the
    variables it needs and a fake OPP tree on disk.
    """

    def _run_script(self, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
        clean_env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
        clean_env.update(env)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            env=clean_env,
            capture_output=True,
            text=True,
        )

    def _built_lib_arg(self, built_root: Path) -> list[str]:
        """Return --built-lib pointing at a fake built vendor's libcust_opapi.so."""
        return [
            "--built-lib",
            str(built_root / "vendors" / "fla_npu_transformer" / VENDOR_LIB_REL),
        ]

    def _built_kernel_arg(self, built_root: Path) -> list[str]:
        """Return --built-kernel pointing at a fake built kernel .o root."""
        return [
            "--built-kernel",
            str(built_root / "vendors" / "fla_npu_transformer" / VENDOR_KERNEL_REL),
        ]

    def test_default_full_check_ok_when_artifacts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            _fake_vendor(runtime_root)
            _fake_vendor(built_root)

            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                env={"FLA_NPU_OPP_PATH": str(runtime_root)},
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("[OK] loaded libcust_opapi.so", proc.stdout)
            self.assertIn("[OK] all 1 kernel .o files", proc.stdout)

    def test_kernel_only_change_detected(self) -> None:
        """A stale kernel .o must be caught even when libcust_opapi.so matches."""
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            runtime_vendor = _fake_vendor(runtime_root)
            _fake_vendor(built_root)
            # Overwrite the runtime kernel with a different payload (kernel-only change).
            kernel = (
                runtime_vendor
                / VENDOR_KERNEL_REL
                / "ascend910b"
                / "sample"
                / "Sample_v1.o"
            )
            kernel.write_bytes(b"kernel-v2")

            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                env={"FLA_NPU_OPP_PATH": str(runtime_root)},
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("[OK] loaded libcust_opapi.so", proc.stdout)
            self.assertIn("[FAIL]", proc.stdout)
            self.assertIn("1 kernel .o differ", proc.stdout)
            self.assertIn("Sample_v1.o", proc.stdout)

    def test_lib_change_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            runtime_vendor = _fake_vendor(runtime_root)
            _fake_vendor(built_root)
            (runtime_vendor / VENDOR_LIB_REL).write_bytes(b"lib-v2")

            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                env={"FLA_NPU_OPP_PATH": str(runtime_root)},
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("[FAIL] loaded libcust_opapi.so differs", proc.stdout)

    def test_no_kernel_skips_kernel_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            runtime_vendor = _fake_vendor(runtime_root)
            _fake_vendor(built_root)
            (runtime_vendor / VENDOR_KERNEL_REL / "ascend910b" / "sample" / "Sample_v1.o").write_bytes(
                b"kernel-v2"
            )

            proc = self._run_script(
                *self._built_lib_arg(built_root),
                "--no-kernel",
                env={"FLA_NPU_OPP_PATH": str(runtime_root)},
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("kernel", proc.stdout)

    def test_missing_kernel_on_one_side_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            runtime_vendor = _fake_vendor(runtime_root)
            _fake_vendor(built_root)
            # Extra kernel only on the built side.
            extra = (
                built_root
                / "vendors"
                / "fla_npu_transformer"
                / VENDOR_KERNEL_REL
                / "ascend910b"
                / "sample"
                / "Extra.o"
            )
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(b"extra")

            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                env={"FLA_NPU_OPP_PATH": str(runtime_root)},
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("[MISSING] ascend910b/sample/Extra.o (only on built side)", proc.stdout)

    def test_python_wrapper_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            package = _fake_fla_npu_package(runtime_root / "site")
            _fake_vendor(built_root)
            # Stale installed wrapper.
            (package / "__init__.py").write_text("VERSION = '2'\n", encoding="utf-8")

            site_dir = runtime_root / "site"
            sys_path_env = f"{site_dir}{os.pathsep}{Path(__file__).parent}"  # keep script parent off
            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                "--python",
                env={
                    "FLA_NPU_OPP_PATH": str(runtime_root),
                    "PYTHONPATH": sys_path_env,
                },
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("[FAIL] __init__.py", proc.stdout)
            self.assertIn("[FAIL] some installed Python wrapper files", proc.stdout)

    def test_python_wrapper_ok_when_sources_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            built_root = Path(tmp) / "built"
            package = _fake_fla_npu_package(runtime_root / "site")
            _fake_vendor(built_root)
            # Copy the real wrapper sources so installed == sources.
            for rel in (
                "__init__.py",
                "ops/ascendc/__init__.py",
                "ops/ascendc/_aclnn_ctypes.py",
                "ops/ascendc/_runtime.py",
            ):
                src = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu" / rel
                dst = package / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())

            site_dir = runtime_root / "site"
            sys_path_env = f"{site_dir}{os.pathsep}{Path(__file__).parent}"
            proc = self._run_script(
                *self._built_lib_arg(built_root),
                *self._built_kernel_arg(built_root),
                "--python",
                env={
                    "FLA_NPU_OPP_PATH": str(runtime_root),
                    "PYTHONPATH": sys_path_env,
                },
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("[OK] installed Python wrapper files match", proc.stdout)

    def test_opp_path_override_wins_over_installed(self) -> None:
        """FLA_NPU_OPP_PATH must take precedence over the installed package opp."""
        with tempfile.TemporaryDirectory() as tmp:
            override_root = Path(tmp) / "override"
            _fake_vendor(override_root)
            _fake_fla_npu_package(Path(tmp) / "site")

            sys_path_env = f"{Path(tmp) / 'site'}{os.pathsep}{Path(__file__).parent}"
            proc = self._run_script(
                "--no-kernel",
                *self._built_lib_arg(override_root),
                env={"FLA_NPU_OPP_PATH": str(override_root), "PYTHONPATH": sys_path_env},
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            # FLA_NPU_OPP_PATH points at an OPP root containing vendors/; the
            # loaded lib must resolve from there, not from the site-packages opp.
            self.assertIn(str(override_root), proc.stdout)
            self.assertIn("libcust_opapi.so", proc.stdout)

    def test_no_opp_found_reports_fail(self) -> None:
        proc = self._run_script("--no-kernel", env={})
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("[FAIL] runtime libcust_opapi.so not found", proc.stdout)

    def test_help_lists_all_options(self) -> None:
        proc = self._run_script("--help", env={})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for option in ("--built-lib", "--run-package", "--python", "--no-kernel", "--built-kernel"):
            self.assertIn(option, proc.stdout)


if __name__ == "__main__":
    unittest.main()
