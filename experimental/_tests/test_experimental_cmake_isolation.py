"""Exercise the production CMake discovery/closure without CANN or a compiler."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "cmake is required")
class ExperimentalCMakeIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Both names deliberately occur in the absolute source path: only the
        # flat operator marker, not arbitrary path substrings, may classify it.
        self.root = Path(self.temp.name) / "tests" / "op_host_fixture"
        self.root.mkdir(parents=True)
        (self.root / "experimental/ops").mkdir(parents=True)
        self.build = Path(self.temp.name) / "build"
        common = self.root / "experimental" / "_common"
        common.mkdir(parents=True)
        (common / "manifest.txt").write_text("# No required common files.\n")
        self.write_project()

    def write_project(self, extra="", *, validate_common=True):
        common_check = (
            "fla_validate_experimental_common(COMMON_DIR)" if validate_common else ""
        )
        (self.root / "CMakeLists.txt").write_text(
            f'''cmake_minimum_required(VERSION 3.16)
project(experimental_isolation NONE)
option(ENABLE_EXPERIMENTAL "" ON)
set(BUILD_OPEN_PROJECT ON)
set(ENABLE_AICPU ON)
set(ASCEND_OP_NAME ALL CACHE STRING "")
include("{(REPO_ROOT / 'cmake' / 'func.cmake').as_posix()}")
{extra}
if(ENABLE_EXPERIMENTAL)
    {common_check}
endif()
file(WRITE "${{CMAKE_BINARY_DIR}}/mounted.txt" "")
op_add_subdirectory(OP_LIST OP_DIR_LIST)
fla_write_experimental_ops("experimental-selected-ops.txt" ${{OP_LIST}})
foreach(OP_DIR IN LISTS OP_DIR_LIST)
    add_subdirectory("${{OP_DIR}}")
endforeach()
if(ENABLE_EXPERIMENTAL)
    fla_expand_experimental_dependencies(OP_LIST OP_DIR_LIST)
    fla_audit_experimental_targets("${{CMAKE_SOURCE_DIR}}")
endif()
fla_write_experimental_ops("experimental-resolved-ops.txt" ${{OP_LIST}})
''',
            encoding="utf-8",
        )

    def operator(self, name, dependencies=(), *, experimental=True):
        directory = (
            self.root / "experimental/ops" / name
            if experimental
            else self.root / "fla" / "ops" / "ascendc" / "group" / name
        )
        directory.mkdir(parents=True)
        declaration = (
            f'set({name}_depends "{";".join(dependencies)}" CACHE STRING "" FORCE)\n'
            if dependencies
            else ""
        )
        (directory / "CMakeLists.txt").write_text(
            declaration
            + f'file(APPEND "${{CMAKE_BINARY_DIR}}/mounted.txt" "{name}\\n")\n',
            encoding="utf-8",
        )
        return directory

    def configure(self, *options, success=True):
        result = subprocess.run(
            [CMAKE, "-S", str(self.root), "-B", str(self.build), *options],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if success:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        # CMake wraps diagnostics according to path length and terminal width.
        # Preserve their words while making semantic assertions wrap-independent.
        return " ".join(output.split())

    def ops(self, kind="resolved"):
        return (self.build / f"experimental-{kind}-ops.txt").read_text().split()

    def link(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

    def test_flat_discovery_excludes_reserved_framework_directories(self):
        for name in ("op_host_probe", "tests", "scripts", "docs", "cmake", "torch_custom", "zeta"):
            self.operator(name)
        self.operator("stable", experimental=False)
        self.configure()
        self.assertEqual(self.ops(), ["op_host_probe", "zeta"])

    def test_discovery_and_dependencies_do_not_fall_back_outside_ops(self):
        self.operator("alpha")
        legacy = self.root / "experimental/legacy_probe"
        legacy.mkdir()
        (legacy / "CMakeLists.txt").write_text('message(FATAL_ERROR "legacy directory mounted")\n')
        self.configure()
        self.assertEqual(self.ops(), ["alpha"])
        self.assertIn("Requested experimental operator is missing",
                      self.configure("-DASCEND_OP_NAME=legacy_probe", success=False))
        (self.root / "experimental/ops/alpha/CMakeLists.txt").write_text(
            'set(alpha_depends legacy_probe CACHE STRING "" FORCE)\n')
        self.assertIn("Experimental dependency/operator is missing",
                      self.configure("-DASCEND_OP_NAME=alpha", success=False))

    def test_ops_root_cannot_redirect_to_another_experimental_directory(self):
        self.operator("alpha")
        ops = self.root / "experimental/ops"
        relocated = self.root / "experimental/legacy_ops"
        ops.rename(relocated)
        self.link(ops, relocated, directory=True)
        self.assertIn("Experimental ops root must resolve", self.configure(success=False))

    def test_non_reserved_directory_requires_an_entry_point(self):
        (self.root / "experimental/ops/missing_entry").mkdir()
        self.assertIn("missing: 'missing_entry'", self.configure(success=False))

    def test_transitive_diamond_dependency_is_mounted_once(self):
        self.operator("alpha", ["beta", "gamma"])
        self.operator("beta", ["delta"])
        self.operator("gamma", ["delta"])
        self.operator("delta")
        self.configure("-DASCEND_OP_NAME=alpha")
        self.assertEqual(self.ops("selected"), ["alpha"])
        self.assertEqual(self.ops(), ["alpha", "beta", "delta", "gamma"])
        mounted = (self.build / "mounted.txt").read_text().split()
        self.assertCountEqual(mounted, self.ops())

    def test_missing_transitive_dependency_reports_chain(self):
        self.operator("alpha", ["beta"])
        self.operator("beta", ["missing"])
        output = self.configure("-DASCEND_OP_NAME=alpha", success=False)
        self.assertIn("alpha -> beta -> missing", output)
        self.assertIn("Experimental dependency/operator is missing", output)

    def test_dependency_cycle_is_rejected_even_when_all_roots_selected(self):
        self.operator("alpha", ["beta"])
        self.operator("beta", ["alpha"])
        output = self.configure(success=False)
        self.assertIn("Experimental dependency cycle: alpha -> beta -> alpha", output)

    def test_dependency_self_cycle_is_rejected(self):
        self.operator("alpha", ["alpha"])
        self.assertIn("alpha -> alpha", self.configure(success=False))

    def test_false_like_names_are_real_nodes(self):
        self.operator("OFF", ["0"])
        self.operator("0", ["NO"])
        self.operator("NO")
        self.configure("-DASCEND_OP_NAME=OFF")
        self.assertEqual(self.ops(), ["0", "NO", "OFF"])

    def test_invalid_dependency_names_are_rejected(self):
        self.operator("alpha")
        for name in ("../stable", "fla/op", "_common", "bad name", "a$b", "all", "ALL"):
            with self.subTest(name=name):
                (self.root / "experimental/ops" / "alpha" / "CMakeLists.txt").write_text(
                    f'set(alpha_depends "{name}" CACHE STRING "" FORCE)\n'
                )
                self.assertIn("Invalid experimental operator name", self.configure(success=False))

    def test_missing_selected_operator_fails_instead_of_empty_build(self):
        self.operator("alpha")
        self.assertIn(
            "Requested experimental operator is missing",
            self.configure("-DASCEND_OP_NAME=missing", success=False),
        )

    def test_operator_symlink_to_stable_is_rejected(self):
        stable = self.operator("stable", experimental=False)
        self.link(self.root / "experimental/ops" / "alias", stable, directory=True)
        self.assertIn("resolves outside", self.configure(success=False))

    def test_nested_source_symlink_to_stable_is_rejected(self):
        alpha = self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        self.link(alpha / "kernel.cpp", source)
        self.assertIn("resolves outside", self.configure(success=False))

    def test_dependency_symlink_to_stable_is_rejected(self):
        self.operator("alpha", ["beta"])
        stable = self.operator("stable", experimental=False)
        self.link(self.root / "experimental/ops" / "beta", stable, directory=True)
        self.assertIn("resolves outside", self.configure("-DASCEND_OP_NAME=alpha", success=False))

    def test_common_directory_symlink_to_stable_is_rejected(self):
        common = self.root / "experimental" / "_common"
        (common / "manifest.txt").unlink()
        common.rmdir()
        stable = self.operator("stable", experimental=False)
        (stable / "manifest.txt").write_text("")
        self.link(common, stable, directory=True)
        self.assertIn("resolves outside", self.configure(success=False))

    def test_unlisted_common_source_symlink_is_also_rejected(self):
        common = self.root / "experimental" / "_common"
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.hpp"
        source.write_text("// stable header\n")
        self.link(common / "hidden.hpp", source)
        self.assertIn("resolves outside", self.configure(success=False))

    def test_directory_link_cycle_is_rejected(self):
        alpha = self.operator("alpha")
        self.link(alpha / "loop", alpha, directory=True)
        self.assertIn("directory link cycle", self.configure(success=False))

    def test_common_manifest_checks_required_files_and_traversal(self):
        self.operator("alpha")
        common = self.root / "experimental" / "_common"
        manifest = common / "manifest.txt"
        manifest.write_text("missing.hpp\n")
        self.assertIn("missing path", self.configure(success=False))
        manifest.write_text("../alpha/CMakeLists.txt\n")
        self.assertIn("invalid path", self.configure(success=False))
        manifest.write_text("# Empty required-file list is valid.\n\n")
        self.configure()

    def test_reconfigure_clears_removed_dependency_declarations(self):
        alpha = self.operator("alpha", ["beta"])
        self.operator("beta")
        self.configure("-DASCEND_OP_NAME=alpha")
        self.assertEqual(self.ops(), ["alpha", "beta"])
        (alpha / "CMakeLists.txt").write_text("# dependency removed\n")
        self.configure("-DASCEND_OP_NAME=alpha", "-DPREPARE_BUILD=ON")
        self.assertEqual(self.ops(), ["alpha"])

    def test_same_mode_reconfigure_allowed_and_mode_switch_rejected(self):
        self.operator("alpha")
        self.configure()
        self.configure("-DPREPARE_BUILD=ON")
        self.assertIn(
            "Cannot reuse a experimental build directory for stable operators",
            self.configure("-DENABLE_EXPERIMENTAL=OFF", success=False),
        )

    def test_stable_discovery_ignores_invalid_experimental_inputs(self):
        # Preserve the legacy stable path classification, including its tests/
        # exclusion; this assertion is not an experimental discovery test.
        self.root = Path(self.temp.name) / "stable_fixture"
        self.root.mkdir()
        self.write_project()
        self.operator("stable", experimental=False)
        self.operator("bad name")
        self.configure("-DENABLE_EXPERIMENTAL=OFF")
        self.assertEqual(self.ops(), ["stable"])

    def test_unimplemented_experimental_package_modes_are_rejected(self):
        self.operator("alpha")
        for flag in ("BUILD_OPS_RTY_KERNEL", "ENABLE_BUILT_IN", "ENABLE_STATIC"):
            with self.subTest(flag=flag):
                output = self.configure(f"-D{flag}=ON", success=False)
                self.assertIn("only custom OPP packages", output)
                self.configure(f"-D{flag}=OFF")

    def test_explicit_stable_target_source_and_include_are_rejected(self):
        self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        for command in (
            f'target_sources(probe INTERFACE "{source.as_posix()}")',
            f'target_include_directories(probe INTERFACE "{stable.as_posix()}")',
            f'target_include_directories(probe INTERFACE "$<BUILD_INTERFACE:{stable.as_posix()}>")',
        ):
            with self.subTest(command=command):
                self.write_project(extra="add_library(probe INTERFACE)\n" + command)
                self.assertIn("references the stable operator tree", self.configure(success=False))

    def test_explicit_symlink_source_outside_operator_is_audited(self):
        self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        alias = self.root / "bridge.cpp"
        self.link(alias, source)
        self.write_project(
            extra=f'add_library(probe INTERFACE)\ntarget_sources(probe INTERFACE "{alias.as_posix()}")'
        )
        self.assertIn("resolves to the stable operator tree", self.configure(success=False))

    def test_framework_common_sdk_and_generated_inputs_remain_allowed(self):
        self.operator("alpha")
        self.write_project(
            extra='''add_library(probe INTERFACE)
target_include_directories(probe INTERFACE
    "${CMAKE_SOURCE_DIR}/common"
    "${CMAKE_BINARY_DIR}/generated"
    "/opt/ascend/ascend-toolkit/include")
target_sources(probe INTERFACE "${CMAKE_BINARY_DIR}/generated.cpp")
'''
        )
        self.configure()

    def test_nested_interface_relative_inputs_are_resolved_from_owner(self):
        self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        bridge = self.root / "bridge"
        bridge.mkdir()
        self.link(bridge / "alias.cpp", source)
        self.link(bridge / "headers", stable, directory=True)
        self.write_project(extra="add_subdirectory(bridge)")
        for prop, value in (("INTERFACE_SOURCES", "alias.cpp"),
                            ("INTERFACE_INCLUDE_DIRECTORIES", "headers")):
            with self.subTest(property=prop):
                (bridge / "CMakeLists.txt").write_text(
                    f'add_library(probe INTERFACE)\nset_property(TARGET probe PROPERTY {prop} "{value}")\n')
                self.assertIn("resolves to the stable operator tree", self.configure(success=False))

    def test_non_interface_target_inputs_are_still_audited(self):
        self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        for prop, value in (("SOURCES", source), ("INCLUDE_DIRECTORIES", stable)):
            with self.subTest(property=prop):
                self.write_project(extra="add_custom_target(probe)\n"
                    f'set_property(TARGET probe PROPERTY {prop} "{value.as_posix()}")')
                self.assertIn("references the stable operator tree", self.configure(success=False))

    def test_modern_interface_private_inputs_are_still_audited(self):
        version = subprocess.check_output([CMAKE, "--version"], text=True).split()[2]
        if tuple(map(int, version.split(".")[:2])) < (3, 19):
            self.skipTest("Interface private properties require CMake 3.19 or newer")
        self.operator("alpha")
        stable = self.operator("stable", experimental=False)
        source = stable / "kernel.cpp"
        source.write_text("// stable source\n")
        for prop, value in (("SOURCES", source), ("INCLUDE_DIRECTORIES", stable)):
            with self.subTest(property=prop):
                self.write_project(extra="add_library(probe INTERFACE)\n"
                    f'set_property(TARGET probe PROPERTY {prop} "{value.as_posix()}")')
                self.assertIn("references the stable operator tree", self.configure(success=False))


if __name__ == "__main__":
    unittest.main()
