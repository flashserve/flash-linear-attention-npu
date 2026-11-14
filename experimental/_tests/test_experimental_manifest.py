"""Package audit fixtures based on the CANN 9.2 A5 artifact layout."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experimental_fixtures import write_abi


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("experimental_manifest", ROOT / "experimental/scripts/experimental_manifest.py")
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def write(path: Path, content: str | bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def write_json(path: Path, data):
    write(path, json.dumps(data))


class ExperimentalManifestTest(unittest.TestCase):
    def test_provenance_excludes_development_files_but_tracks_build_inputs(self):
        before = self.data["operators"][0]["source_sha256"]
        for name in ("README.md", "docs/design.md", "tests/case.py", "examples/sample.cpp"):
            write(self.op / name, "not a build input")
        create = lambda: manifest.create_manifest(self.repo, self.selected, self.resolved, "ascend950", "arch35;arch22")
        self.assertEqual(before, create()["operators"][0]["source_sha256"])
        write(self.op / "adapter/stable_probe.cpp", "// build input")
        self.assertIn("adapter/stable_probe.cpp", create()["operators"][0]["source_sha256"])
        write(self.op / "op_kernel/probe.cpp", "// changed build input")
        self.assertNotEqual(before["op_kernel/probe.cpp"], create()["operators"][0]["source_sha256"]["op_kernel/probe.cpp"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.op = self.repo / "experimental/ops" / "probe"
        write(self.op / "CMakeLists.txt", "# fixture\n")
        write(self.op / "op_host/probe_def.cpp", 'OP_ADD(Probe); AddConfig("ascend950");\n')
        write(self.op / "op_host/op_api/aclnn_probe.h",
              "int aclnnProbe(); int aclnnProbeGetWorkspaceSize();\n")
        for name in ("probe.cpp", "arch_root.hpp", "arch35/selected.hpp", "arch22/legacy.hpp", "arch32/unselected.hpp"):
            write(self.op / "op_kernel" / name, f"// {name}\n")
        write(self.repo / "experimental/_common/manifest.txt", "shared.hpp\n")
        write(self.repo / "experimental/_common/shared.hpp", "// shared\n")
        write_json(self.repo / ".source-identity.json", {
            "source_commit": "base-commit", "source_dirty": True,
            "source_snapshot_sha256": "snapshot-digest",
        })
        self.selected = self.root / "selected.txt"
        self.resolved = self.root / "resolved.txt"
        write(self.selected, "probe\n")
        write(self.resolved, "probe\n")
        self.data = manifest.create_manifest(self.repo, self.selected, self.resolved, "ascend950", "arch35;arch22")
        self.expected = self.root / "expected.json"
        write_json(self.expected, self.data)
        self.install = self.root / "packages"
        self.vendor = self.install / "vendors" / manifest.VENDOR
        self.tbe = self.vendor / "op_impl/ai_core/tbe"
        self.impl = self.tbe / f"{manifest.VENDOR}_impl"
        self.source = self.impl / "ascendc/probe"
        write_json(self.vendor / manifest.MANIFEST, self.data)
        for name in ("probe.cpp", "arch_root.hpp", "arch35/selected.hpp", "arch22/legacy.hpp"):
            write(self.source / name, (self.op / "op_kernel" / name).read_bytes())
        write(self.impl / "ascendc/common/shared.hpp", "// shared\n")
        write(self.impl / "ascendc/common/manifest.txt", "shared.hpp\n")
        write(self.impl / "dynamic/probe.py", "# adapter\n")
        write(self.vendor / "op_proto/inc/probe_proto.h", "// proto\n")
        write(self.vendor / "op_api/include/aclnnop/aclnn_probe.h", "// aclnn\n")
        write(self.vendor / "op_api/lib/libcust_opapi.so", b"fixture-library")
        write_abi(self.vendor)
        self.kernel = self.tbe / "kernel/ascend950/probe"
        self.object = self.kernel / "Probe_01.o"
        self.metadata = self.kernel / "Probe_01.json"
        write(self.object, b"compiled-kernel-object")
        write_json(self.metadata, {"binFileName": "Probe_01", "binFileSuffix": ".o",
                                  "sha256": hashlib.sha256(self.object.read_bytes()).hexdigest()})
        self.config = self.tbe / "kernel/config/ascend950/probe.json"
        self.aggregate = self.tbe / "kernel/config/ascend950/binary_info_config.json"
        self.ops_info = self.tbe / "config/ascend950/aic-ascend950-ops-info.json"
        write_json(self.config, {"binList": [{"binInfo": {"jsonFilePath": "ascend950/probe/Probe_01.json"}}]})
        write_json(self.aggregate, {"Probe": {"binaryList": [{"binPath": "ascend950/probe/Probe_01.o",
                                                             "jsonPath": "ascend950/probe/Probe_01.json"}]}})
        write_json(self.ops_info, {"Probe": {"opFile": {"value": "probe"}}})
        self.nm = patch.object(manifest.subprocess, "check_output",
                               return_value="000 T aclnnProbe\n001 T aclnnProbeGetWorkspaceSize\n")
        self.nm.start()
        self.addCleanup(self.nm.stop)

    def audit(self):
        return manifest.audit_package(self.install, "ascend950", self.expected)

    def test_valid_package_records_architecture_and_source_identity(self):
        self.assertEqual(self.data["operators"][0]["source"], "experimental/ops/probe")
        result = self.audit()
        self.assertEqual(result["arch_directories"], ["arch22", "arch35"])
        self.assertEqual(result["source_commit"], "base-commit")
        self.assertEqual(result["source_snapshot_sha256"], "snapshot-digest")
        self.assertTrue(result["source_dirty"])
        self.assertFalse(result["runtime_executed"])

    def test_stale_abi_manifest_is_rejected(self):
        write(self.vendor / "op_api/lib/libcust_opapi.so", b"tampered")
        with self.assertRaisesRegex(ValueError, "ABI manifest"):
            self.audit()

    def test_optional_apt_adapter_is_allowed(self):
        write(self.impl / "dynamic/probe_apt.py", "# optional APT\n")
        self.audit()

    def test_apt_adapter_does_not_replace_base_adapter(self):
        (self.impl / "dynamic/probe.py").unlink()
        write(self.impl / "dynamic/probe_apt.py", "# optional APT\n")
        with self.assertRaisesRegex(ValueError, "base adapters required"):
            self.audit()

    def test_other_operator_apt_adapter_is_rejected(self):
        write(self.impl / "dynamic/stable_apt.py", "# unexpected\n")
        with self.assertRaisesRegex(ValueError, "unexpected=.*stable_apt"):
            self.audit()

    def test_operator_name_ending_apt_is_not_removed_as_optional(self):
        write(self.impl / "dynamic/probe_apt.py", "# another operator base adapter\n")
        manifest._audit_dynamic_adapters(self.impl / "dynamic", {"probe", "probe_apt"})
        write(self.impl / "dynamic/probe_apt_apt.py", "# its optional adapter\n")
        manifest._audit_dynamic_adapters(self.impl / "dynamic", {"probe", "probe_apt"})

    def test_operator_ending_apt_still_requires_its_base_adapter(self):
        write(self.impl / "dynamic/probe_apt_apt.py", "# only optional adapter\n")
        with self.assertRaisesRegex(ValueError, "base adapters required.*missing=.*probe_apt"):
            manifest._audit_dynamic_adapters(self.impl / "dynamic", {"probe", "probe_apt"})

    def test_selected_arch_source_cannot_be_missing(self):
        (self.source / "arch35/selected.hpp").unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*arch35/selected.hpp"):
            self.audit()

    def test_root_source_starting_arch_is_still_required(self):
        (self.source / "arch_root.hpp").unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*arch_root.hpp"):
            self.audit()

    def test_unselected_arch_source_is_not_packaged(self):
        write(self.source / "arch32/unselected.hpp", "// arch32/unselected.hpp\n")
        with self.assertRaisesRegex(ValueError, "unexpected=.*arch32/unselected.hpp"):
            self.audit()

    def test_extra_operator_source_is_rejected(self):
        write(self.source / "stable_kernel.hpp", "// unexpected stable code\n")
        with self.assertRaisesRegex(ValueError, "unexpected=.*stable_kernel.hpp"):
            self.audit()

    def test_modified_operator_source_hash_is_rejected(self):
        write(self.source / "probe.cpp", "// modified\n")
        with self.assertRaisesRegex(ValueError, "does not match resolved experimental input"):
            self.audit()

    def test_sdk_common_namespaces_remain_allowed_and_are_reported(self):
        for directory in ("catlass", "tla", "atvoss", "op_kernel"):
            write(self.impl / "ascendc/common" / directory / "sdk.hpp", "// SDK infrastructure\n")
        result = self.audit()
        self.assertEqual(result["sdk_common_roots_outside_source_snapshot"],
                         ["atvoss", "catlass", "op_kernel", "tla"])

    def test_extra_non_sdk_common_source_is_rejected(self):
        write(self.impl / "ascendc/common/stable_utils.hpp", "// unexpected stable common\n")
        with self.assertRaisesRegex(ValueError, "Experimental common source files.*unexpected=.*stable_utils.hpp"):
            self.audit()

    def test_changed_common_manifest_is_rejected(self):
        write(self.impl / "ascendc/common/manifest.txt", "# changed requirements\n")
        with self.assertRaisesRegex(ValueError, "Common source does not match experimental snapshot: manifest.txt"):
            self.audit()

    def test_empty_kernel_directory_is_rejected(self):
        self.object.unlink()
        self.metadata.unlink()
        with self.assertRaisesRegex(ValueError, "must contain nonempty"):
            self.audit()

    def test_zero_length_binary_is_rejected(self):
        write(self.object, b"")
        with self.assertRaisesRegex(ValueError, "empty kernel artifact"):
            self.audit()

    def test_object_without_metadata_is_rejected(self):
        write(self.kernel / "Probe_extra.o", b"orphan-object")
        with self.assertRaisesRegex(ValueError, "object/metadata pairs"):
            self.audit()

    def test_changed_binary_hash_is_rejected(self):
        write(self.object, b"changed-binary")
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            self.audit()

    def test_kernel_metadata_must_name_its_binary(self):
        details = json.loads(self.metadata.read_text())
        details["binFileName"] = "StableKernel"
        write_json(self.metadata, details)
        with self.assertRaisesRegex(ValueError, "names a different binary"):
            self.audit()

    def test_per_operator_config_must_reference_actual_metadata(self):
        write_json(self.config, {"binList": [{"binInfo": {"jsonFilePath": "ascend950/stable/Stable.json"}}]})
        with self.assertRaisesRegex(ValueError, "kernel config JSON references"):
            self.audit()

    def test_empty_per_operator_config_is_rejected(self):
        write_json(self.config, {"binList": []})
        with self.assertRaisesRegex(ValueError, "expected nonempty binList"):
            self.audit()

    def test_aggregate_must_reference_actual_binary(self):
        write_json(self.aggregate, {"Probe": {"binaryList": [{"binPath": "ascend950/probe/missing.o",
                                                             "jsonPath": "ascend950/probe/Probe_01.json"}]}})
        with self.assertRaisesRegex(ValueError, "binary info binPath references"):
            self.audit()

    def test_aggregate_cannot_contain_other_op_types(self):
        data = json.loads(self.aggregate.read_text())
        data["StableOp"] = {"binaryList": []}
        write_json(self.aggregate, data)
        with self.assertRaisesRegex(ValueError, "Binary info operator types"):
            self.audit()

    def test_ops_info_cannot_point_to_another_source(self):
        write_json(self.ops_info, {"Probe": {"opFile": {"value": "stable"}}})
        with self.assertRaisesRegex(ValueError, "different operator source"):
            self.audit()

    def test_installed_manifest_must_match_build_manifest(self):
        altered = dict(self.data, source_commit="different-commit")
        write_json(self.vendor / manifest.MANIFEST, altered)
        with self.assertRaisesRegex(ValueError, "differs from this build"):
            self.audit()

    def test_old_manifest_cannot_guess_architecture_from_missing_files(self):
        altered = dict(self.data, schema_version=1)
        altered.pop("arch_directories")
        write_json(self.vendor / manifest.MANIFEST, altered)
        with self.assertRaisesRegex(ValueError, "CMake architecture selection"):
            self.audit()

    def test_manifest_generation_requires_explicit_architecture_selection(self):
        with self.assertRaisesRegex(ValueError, "architecture directory selection"):
            manifest.create_manifest(self.repo, self.selected, self.resolved, "ascend950", "")

    def test_manifest_generation_rejects_empty_selected_set(self):
        write(self.selected, "")
        with self.assertRaisesRegex(ValueError, "nonempty selected"):
            manifest.create_manifest(self.repo, self.selected, self.resolved, "ascend950", "arch35")


if __name__ == "__main__":
    unittest.main()
