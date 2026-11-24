"""Negative contracts and the recovery CLI, independent of CANN/torch."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from tests.opp_abi_fixtures import write_elf_exports

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "torch_custom/fla_npu"
spec = importlib.util.spec_from_file_location("opp_abi", PACKAGE / "fla_npu/_op_abi.py")
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)


def vendor(path, scalar="double", exports=None, *, name="Demo"):
    (path / abi.LIBRARY).parent.mkdir(parents=True, exist_ok=True)
    (path / abi.LIBRARY).write_bytes(b"fixture ELF, exports supplied by the test")
    header = path / abi.INCLUDE / f"aclnn_{name.lower()}.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        f"aclnnStatus aclnn{name}GetWorkspaceSize(const aclTensor *x, {scalar} scale, "
        "uint64_t *workspaceSize, aclOpExecutor **executor);\n"
        f"aclnnStatus aclnn{name}(void *workspace, uint64_t workspaceSize, "
        "aclOpExecutor *executor, aclrtStream stream);\n", encoding="utf-8")
    exports = exports if exports is not None else set(abi.public_prototypes(path / abi.INCLUDE))
    write_elf_exports(path / abi.LIBRARY, exports)
    with mock.patch.object(abi, "exported_symbols", return_value=exports):
        abi.write_manifest(path)
    return path


class ContractTest(unittest.TestCase):
    def test_elf_reader_supports_classes_and_byte_orders(self):
        for bits in (32, 64):
            for byteorder in ("<", ">"):
                with self.subTest(bits=bits, byteorder=byteorder), tempfile.TemporaryDirectory() as tmp:
                    library = Path(tmp) / "vendor.so"
                    symbols = {"aclnnDemo", "aclnnDemoGetWorkspaceSize"}
                    write_elf_exports(library, symbols | {"private_helper"}, bits=bits, byteorder=byteorder)
                    self.assertEqual(abi.elf_exported_symbols(library), symbols)

    def test_invalid_elf_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "vendor.so"
            write_elf_exports(library, {"aclnnDemo"})
            complete = library.read_bytes()
            for content in (b"not ELF", complete[:20], complete[:-1]):
                library.write_bytes(content)
                with self.assertRaisesRegex(RuntimeError, "Invalid ELF"):
                    abi.elf_exported_symbols(library)

    @unittest.skipUnless(os.name == "posix" and shutil.which("cc") and shutil.which("nm"), "native ELF compiler/nm")
    def test_native_elf_matches_nm(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, library = Path(tmp) / "test.c", Path(tmp) / "test.so"
            source.write_text('extern int puts(const char *);\n'
                              'int aclnnDemo(void) { return puts("unused"); }\n'
                              'int aclnnDemoGetWorkspaceSize(void) { return 0; }\n'
                              'int unrelated(void) { return 0; }\n')
            subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
            self.assertEqual(abi.elf_exported_symbols(library), abi.exported_symbols(library))

    def test_omitted_interface_pair_is_not_treated_as_new(self):
        for omitted in ("base", "overlay"):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as tmp:
                base = vendor(Path(tmp) / "base", "double")
                vendor(base, name="Keep")
                overlay = vendor(Path(tmp) / "overlay", "float")
                vendor(overlay, name="Keep")
                manifest = (base if omitted == "base" else overlay) / abi.MANIFEST_NAME
                data = json.loads(manifest.read_text())
                for symbol in ("aclnnDemo", "aclnnDemoGetWorkspaceSize"):
                    del data["contracts"][symbol]
                manifest.write_text(json.dumps(data))
                problems = abi.compare_overlay_to_base(overlay, base)
                self.assertTrue(any("Incomplete ABI manifest" in p for p in problems), problems)

    def test_extra_unexported_header_declarations_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = vendor(Path(tmp))
            vendor(path, name="DeclarationOnly", exports={"aclnnDemo", "aclnnDemoGetWorkspaceSize"})
            self.assertEqual(set(abi.validated_contracts(path, require_manifest=True)),
                             {"aclnnDemo", "aclnnDemoGetWorkspaceSize"})

    def test_exact_pointer_levels_and_unknown_types(self):
        for left, right in [("double scale", "double *scale"),
                            ("aclTensor *x", "aclTensor **x"),
                            ("int64_t n", "int64_t *n")]:
            self.assertNotEqual(abi.parameter_kind(left), abi.parameter_kind(right))
        for text in ["mystery x", "double x[2]", "void (*callback)(int)", "double &x"]:
            with self.assertRaises(RuntimeError):
                abi.parameter_kind(text)

    def test_valid_manifest_and_legacy_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, overlay = vendor(Path(tmp)/"base"), vendor(Path(tmp)/"overlay")
            (base/abi.MANIFEST_NAME).unlink()
            self.assertEqual(abi.compare_overlay_to_base(overlay, base), [])

    def test_pointer_drift_is_rejected_with_fresh_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, overlay = vendor(Path(tmp)/"base"), vendor(Path(tmp)/"overlay", "double *")
            problems = abi.compare_overlay_to_base(overlay, base)
            self.assertTrue(any("double*" in p for p in problems), problems)

    def test_new_interfaces_are_allowed_with_manifest_or_legacy_base(self):
        for legacy in (False, True):
            for include_existing in (False, True):
                with self.subTest(legacy=legacy, include_existing=include_existing), \
                        tempfile.TemporaryDirectory() as tmp:
                    base = vendor(Path(tmp) / "base")
                    if legacy:
                        (base / abi.MANIFEST_NAME).unlink()
                    overlay = Path(tmp) / "overlay"
                    if include_existing:
                        vendor(overlay)
                    vendor(overlay, name="RecurrentGatedDeltaRuleV2")
                    self.assertEqual(abi.compare_overlay_to_base(overlay, base), [])

    def test_new_interface_does_not_allow_drift_in_existing_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = vendor(Path(tmp) / "base")
            overlay = vendor(Path(tmp) / "overlay", "float")
            vendor(overlay, name="RecurrentGatedDeltaRuleV2")
            problems = abi.compare_overlay_to_base(overlay, base)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("aclnnDemoGetWorkspaceSize", problems[0])
            self.assertIn("float", problems[0])

    def test_new_interface_requires_both_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = vendor(Path(tmp) / "base")
            overlay = vendor(Path(tmp) / "overlay", name="RecurrentGatedDeltaRuleV2")
            manifest = overlay / abi.MANIFEST_NAME
            data = json.loads(manifest.read_text())
            del data["contracts"]["aclnnRecurrentGatedDeltaRuleV2"]
            manifest.write_text(json.dumps(data))
            problems = abi.compare_overlay_to_base(overlay, base)
            self.assertTrue(any("Missing phase-two" in p for p in problems), problems)

    def test_missing_manifest_headers_and_modified_library_fail_closed(self):
        for mutation in ["manifest", "headers", "library"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                base = vendor(Path(tmp) / "base")
                overlay = vendor(Path(tmp) / "overlay", name="RecurrentGatedDeltaRuleV2")
                if mutation == "manifest": (overlay/abi.MANIFEST_NAME).unlink()
                elif mutation == "headers": shutil.rmtree(overlay/abi.INCLUDE)
                else: (overlay/abi.LIBRARY).write_bytes(b"different library")
                self.assertTrue(abi.compare_overlay_to_base(overlay, base))

    def test_uncovered_exports_and_missing_phase_two_rejected(self):
        for exports in [{"aclnnDemoGetWorkspaceSize"},
                        {"aclnnDemo", "aclnnDemoGetWorkspaceSize", "aclnnOther"}]:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
                vendor(Path(tmp), exports=exports)

    def test_malformed_manifest_is_reported_as_invalid(self):
        for invalid in [[], None, {"schema_version": 1, "contracts": {"bad": [None]}}]:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                base, overlay = vendor(Path(tmp)/"base"), vendor(Path(tmp)/"overlay")
                (overlay/abi.MANIFEST_NAME).write_text(json.dumps(invalid))
                self.assertTrue(abi.compare_overlay_to_base(overlay, base))

    def test_tensor_variants_share_only_their_declared_execution_entry(self):
        for variant, execute in abi.EXECUTE_ALIASES.items():
            contracts = {
                variant + "GetWorkspaceSize": (abi.TENSOR, *abi._WORKSPACE_TAIL),
                execute + "GetWorkspaceSize": (abi.INT_ARRAY, *abi._WORKSPACE_TAIL),
                execute: abi._EXECUTE_SIGNATURE,
            }
            abi._validate_exports(set(contracts), contracts)
            with self.assertRaisesRegex(RuntimeError, "Missing phase-two"):
                abi._validate_exports(set(contracts) - {execute}, contracts)

    def test_header_comment_cannot_supply_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = vendor(Path(tmp))
            header = path/abi.INCLUDE/"aclnn_demo.h"
            header.write_text("/*" + header.read_text() + "*/")
            with mock.patch.object(abi, "exported_symbols", return_value={"aclnnDemo", "aclnnDemoGetWorkspaceSize"}):
                with self.assertRaisesRegex(RuntimeError, "Missing public"):
                    abi.write_manifest(path)

    def test_workspace_tail_is_checked_by_type_not_parameter_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            inc = Path(tmp)
            (inc/"aclnn_bad.h").write_text("aclnnStatus aclnnBadGetWorkspaceSize(double x, uint32_t *workspaceSize, aclOpExecutor **executor);")
            with self.assertRaises(RuntimeError): abi.parse_aclnn_signatures(inc)

    def test_management_does_not_import_broken_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root/"fla_npu"
            pkg.mkdir()
            (pkg/"__init__.py").write_text("raise RuntimeError('broken vendor must not be loaded')\n")
            for name in ["install_opp.py", "_op_abi.py"]:
                shutil.copy2(PACKAGE/"fla_npu"/name, pkg/name)
            shutil.copy2(PACKAGE/"fla_npu_opp_manage.py", root)
            vendors = pkg/"opp/vendors"
            base = vendor(vendors/"fla_npu_transformer")
            overlay = vendor(vendors/"fla_npu_overlay_transformer", "float")
            before = (base/abi.LIBRARY).read_bytes()
            for mode in ["--status", "--remove-overlay"]:
                result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", mode],
                    cwd=root, env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(overlay.exists())
            self.assertEqual(before, (base/abi.LIBRARY).read_bytes())


if __name__ == "__main__":
    unittest.main()
