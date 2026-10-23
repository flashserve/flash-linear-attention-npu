"""Tests for the Stable-ABI gates themselves.

A gate that quietly stops working is worse than no gate: it still prints OK
while nothing is checked.  These tests are offline (no torch, no NPU) and cover
both directions -- each gate passes on the tree as it stands, and it fails on an
input that is deliberately wrong.

Usage:  python -m unittest tests.test_stable_gates
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = REPO_ROOT / "torch_custom" / "fla_npu"
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
SRC_DIR = SETUP_DIR / "csrc" / "src"


def _load_tool(name: str):
    path = SETUP_DIR / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stable_tree(tmp: str, *, wrapper_args: int = 2):
    """Small isolated Stable-only tree; tests mutate one link at a time."""
    ops, src = Path(tmp) / "ops", Path(tmp) / "src"
    ops.mkdir(); src.mkdir()
    (ops / "__init__.py").write_text('_ASCENDC_OPS = ("npu_new_op",)\n', encoding="utf-8")
    dispatch = 'a, _current_stream_ptr()' if wrapper_args == 2 else 'a'
    (ops / "_stable.py").write_text(
        '_ENUM = {}\ndef npu_new_op(a):\n    return _op("npu_new_op")(' + dispatch + ')\n', encoding="utf-8")
    (src / "stable_new_op.cpp").write_text(
        'constexpr const char* kSchema_npu_new_op = "npu_new_op(Tensor a, int stream) -> Tensor";\n'
        'Tensor run_npu_new_op(Tensor a, int64_t stream) { return a; }\n', encoding="utf-8")
    (src / "stable_ops.cpp").write_text(
        '#include "stable_new_op.cpp"\n'
        'STABLE_TORCH_LIBRARY(fla_npu_stable, m) { m.def(kSchema_npu_new_op); }\n'
        'STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) {\n'
        'm.impl("npu_new_op", &boxed_adapter<run_npu_new_op>); }\n', encoding="utf-8")
    contract = Path(tmp) / "public.json"
    contract.write_text(json.dumps({"operators": ["npu_new_op"]}), encoding="utf-8")
    return ops, src, contract


class BuildStampTest(unittest.TestCase):
    """The library stamp and the Python side must be produced together."""

    def test_build_stamp_covers_every_adapter_source(self) -> None:
        import importlib.util as util

        path = SETUP_DIR / "csrc" / "build_stable.py"
        spec = util.spec_from_file_location("build_stable", path)
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        first = module.source_hash()
        sources = sorted((SETUP_DIR / "csrc" / "src").glob("*.cpp"))
        self.assertTrue(sources, "no adapter sources found")
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "src"
            shutil.copytree(SRC_DIR, copied)
            target = copied / sources[0].name
            with mock.patch.object(module, "CSRC", copied):
                self.assertEqual(module.source_hash(), first)
                original = target.read_text(encoding="utf-8")
                target.write_text(original + "\n// changed\n", encoding="utf-8")
                self.assertNotEqual(
                    module.source_hash(), first,
                    "editing one adapter must change the stamp")
        self.assertEqual(module.source_hash(), first)

    def test_checked_in_hash_module_matches_the_sources(self) -> None:
        """A stale _stable_hash.py would reject a freshly built library."""

        import importlib.util as util

        path = SETUP_DIR / "csrc" / "build_stable.py"
        spec = util.spec_from_file_location("build_stable2", path)
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        hash_module = OPS_DIR / "_stable_hash.py"
        if not hash_module.is_file():
            self.skipTest("no _stable_hash.py in this tree (never built here)")
        match = re.search(r'SOURCE_HASH = "([0-9a-f]{32})"',
                          hash_module.read_text(encoding="utf-8"))
        self.assertIsNotNone(match, "_stable_hash.py carries no SOURCE_HASH")
        self.assertEqual(match.group(1), module.source_hash(),
                         "the checked-in stamp does not match the adapters; "
                         "rebuild with python csrc/build_stable.py")


class CoverageGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool("stable_coverage.py")

    def test_current_tree_has_no_unexplained_gap(self) -> None:
        report = self.tool.evaluate()
        self.assertEqual(report["blockers"], [])
        self.assertEqual(set(self.tool.required_public_ops()), {row["op"] for row in report["rows"]})
        self.assertEqual(report["adapter_count"], len(self.tool.required_public_ops()))

    def test_missing_wrapper_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ops = Path(tmp) / "ops"
            ops.mkdir()
            original = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
            # Drop one wrapper: the operator must then be reported, not skipped.
            stripped = re.sub(
                r"^def npu_chunk_fwd_o\(.*?(?=^def )", "", original,
                flags=re.S | re.M)
            self.assertNotEqual(stripped, original)
            (ops / "_stable.py").write_text(stripped, encoding="utf-8")
            (ops / "__init__.py").write_text(
                (OPS_DIR / "__init__.py").read_text(encoding="utf-8"),
                encoding="utf-8")
            with mock.patch.object(self.tool, "OPS_DIR", ops):
                report = self.tool.evaluate()
            self.assertTrue(any("npu_chunk_fwd_o: no wrapper" in item
                                for item in report["blockers"]),
                            report["blockers"])

    def test_enum_table_order_drift_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            target = src / "stable_chunk_gated_delta_rule_fwd.cpp"
            text = target.read_text(encoding="utf-8")
            # Swap two layout names in the C++ table only: the Python table no
            # longer agrees, which is exactly the silent-layout-change bug.
            changed = text.replace(
                'kGdnFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"}',
                'kGdnFwdLayoutNames[] = {"BNSD", "BSND", "TND", "NTD"}')
            self.assertNotEqual(changed, text)
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("kGdnFwdLayoutNames order" in item
                                for item in report["blockers"]),
                            report["blockers"])

    def test_adapter_source_nobody_includes_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            # Adapter sources are #included into stable_ops.cpp, never compiled
            # on their own: a file nobody includes is dead code and its operator
            # silently has no adapter, while the rest of this gate stays green.
            (src / "stable_orphan_op.cpp").write_text("// orphaned adapter\n",
                                                      encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("never compiled" in item
                                for item in report["blockers"]),
                            report["blockers"])

    def test_enum_numeric_drift_without_key_order_change_is_reported(self) -> None:
        original = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
        before = '_LAYOUT_CODES = {"BSND": 0, "BNSD": 1, "TND": 2, "NTD": 3}'
        self.assertIn(before, original)
        # All retain the identical string key order: only numeric semantics
        # change. Bool/float also compare equal to integers in Python and must
        # not slip through a plain list equality check.
        for values in ((1, 0, 2, 3), (0, 0, 2, 3), (0, 1, 2, 4),
                       (False, 1, 2, 3), (0.0, 1, 2, 3)):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as tmp:
                ops = Path(tmp)
                (ops / "__init__.py").write_text(
                    (OPS_DIR / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
                after = "_LAYOUT_CODES = " + repr(dict(zip(("BSND", "BNSD", "TND", "NTD"), values)))
                (ops / "_stable.py").write_text(original.replace(before, after), encoding="utf-8")
                with mock.patch.object(self.tool, "OPS_DIR", ops):
                    report = self.tool.evaluate()
                self.assertTrue(any("enum numeric codes" in item for item in report["blockers"]), report)

    def test_adapter_in_a_shared_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            moved = "stable_chunk_fwd_h.cpp"
            for path in SRC_DIR.glob("stable_*.cpp"):
                if path.name != moved:
                    (src / path.name).write_text(
                        path.read_text(encoding="utf-8"), encoding="utf-8")
            # A branch that predates the split adds its adapter to the shared
            # file it already has, so the operator's run_ lands in
            # stable_<something else>.cpp while still being registered and
            # included -- every other check in the gate stays green.
            (src / "stable_chunk.cpp").write_text(
                (SRC_DIR / moved).read_text(encoding="utf-8"),
                encoding="utf-8")
            (src / "stable_ops.cpp").write_text(
                (SRC_DIR / "stable_ops.cpp").read_text(encoding="utf-8")
                .replace(f'#include "{moved}"', '#include "stable_chunk.cpp"'),
                encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("one operator per file" in item
                                for item in report["blockers"]),
                            report["blockers"])


class AbiParityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool("op_abi_parity.py")

    def test_current_tree_matches(self) -> None:
        report = self.tool.evaluate()
        self.assertEqual(report["problems"], [])
        self.assertGreater(report["checked"], 25)
        # The two hand-written recurrent entry points consume 13 required
        # tensor slots between them; a drop here means a slot stopped being
        # unboxed into an owning Tensor and is leaking again.
        self.assertGreaterEqual(report["owned_slots"], 13)

    def test_parameter_reorder_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            target = src / "stable_kda_gate_cumsum.cpp"
            text = target.read_text(encoding="utf-8")
            changed = text.replace(
                "run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,\n"
                "                               std::optional<Tensor> dt_bias,\n"
                "                               std::optional<Tensor> cu_seqlens,\n"
                "                               int64_t chunk_size,",
                "run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,\n"
                "                               std::optional<Tensor> dt_bias,\n"
                "                               std::optional<Tensor> cu_seqlens,\n"
                "                               bool chunk_size,")
            self.assertNotEqual(changed, text, "test setup did not apply")
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("chunk_size" in item for item in report["problems"]),
                            report["problems"])

    def test_unconsumed_stack_slot_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            target = src / "stable_recurrent_gated_delta_rule.cpp"
            text = target.read_text(encoding="utf-8")
            # The leak that OOMed the conc32 service: reading a required tensor
            # slot as a raw handle consumes nothing (library.h expects the kernel
            # to steal it).
            changed = text.replace(
                "const Tensor t_query = to<Tensor>(stack[0]);",
                "const AtenTensorHandle query = "
                "to<AtenTensorHandle>(stack[0]);")
            self.assertNotEqual(changed, text, "test setup did not apply")
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(
                any("AtenTensorHandle" in item for item in report["problems"]),
                report["problems"])
            self.assertEqual(report["owned_slots"], 12)


class FallbackGateTest(unittest.TestCase):
    def test_no_adapter_reaches_the_ctypes_reference(self) -> None:
        tool = _load_tool("stable_ctypes_fallbacks.py")
        text = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
        self.assertEqual(tool.delegating_ops(text), [],
                         "an adapter delegates to ctypes again")

    def test_delegation_is_detected(self) -> None:
        tool = _load_tool("stable_ctypes_fallbacks.py")
        sample = ("def npu_x(a):\n"
                  "    from . import _aclnn_ctypes as ct\n"
                  "    return ct.npu_x(a)\n")
        self.assertEqual(tool.delegating_ops(sample), ["npu_x"])


class RemovedBackendGateTest(unittest.TestCase):
    def test_forbidden_imports_and_calls(self):
        tool = _load_tool("stable_ctypes_fallbacks.py")
        for code in ("from . import _runtime", "from ._aclnn_ctypes import npu_x",
                     "import fla_npu.ops.ascendc._runtime", "fla_npu.load_legacy_torch_ops()",
                     "torch.ops.npu.npu_x(a)", "call_aclnn(a)",
                     "def load_legacy_torch_ops(): pass",
                     "importlib.import_module('fla_npu.ops.ascendc._runtime')",
                     "torch.ops.load_library('custom_aclnn_extension_lib.so')"):
            with self.subTest(code=code):
                self.assertTrue(tool.violations(code))
        self.assertEqual(tool.violations('import ctypes; ctypes.CDLL("lib.so")'), [])

    def test_empty_package_and_removed_files_fail(self):
        tool = _load_tool("stable_ctypes_fallbacks.py")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tool, "PACKAGE_DIR", Path(tmp)):
            self.assertTrue(tool.evaluate()["problems"])
            (Path(tmp) / "_runtime.py").write_text("# old execution backend")
            self.assertTrue(any("removed backend file" in item for item in tool.evaluate()["problems"]))


class HeaderKindGateTest(unittest.TestCase):
    """`op_abi_validate` reads the OPP header, the only source of truth left."""

    def test_header_kinds_are_recognised(self) -> None:
        tool = _load_tool("op_abi_validate.py")
        self.assertEqual(tool.header_kind("const aclTensor *q"), tool.WILDCARD)
        self.assertEqual(tool.header_kind("const aclIntArray *cu"), tool.WILDCARD)
        self.assertEqual(tool.header_kind("int64_t chunkSize"), "int64")
        self.assertEqual(tool.header_kind("bool useExp2"), "bool")
        self.assertEqual(tool.header_kind("double scale"), "double")
        self.assertEqual(tool.header_kind("const char *layout"), "char_ptr")

    def test_a_comment_inside_a_call_is_not_an_argument(self) -> None:
        """A comma inside `// ...` must not split the parameter list.

        The adapters comment individual arguments ("// the fused kernel takes
        no cu_seqlens/chunk_indices"); with a comma in that comment the call
        site parsed one argument too long, which is what the
        chunk_gated_delta_rule_bwd_dhu row of this gate used to report.
        """

        tool = _load_tool("op_abi_validate.py")
        params = tool.split_params(
            "Tensor a, // one, two\n"
            "tensor(b), /* three, four */ scalar(c)")
        self.assertEqual(len(params), 3, params)
        # String literals are left alone: a layout name can hold anything.
        self.assertEqual(tool.split_params('cstr("a//b"), scalar(c)'),
                         ['cstr("a//b")', 'scalar(c)'])


class StableOnlyCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool("stable_coverage.py")

    def check_tree(self, mutation=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            ops, src, contract = _stable_tree(tmp, **kwargs)
            if mutation:
                mutation(ops, src)
            with mock.patch.object(self.tool, "OPS_DIR", ops), \
                    mock.patch.object(self.tool, "SRC_DIR", src), \
                    mock.patch.object(self.tool, "PUBLIC_CONTRACT", contract):
                return self.tool.evaluate()

    def assert_mutation(self, file, before, after, expected):
        def mutate(ops, src):
            path = (ops if file.endswith(".py") else src) / file
            original = path.read_text(encoding="utf-8")
            self.assertIn(before, original)
            path.write_text(original.replace(before, after), encoding="utf-8")
        report = self.check_tree(mutate)
        self.assertTrue(any(expected in item for item in report["blockers"]), report)

    def test_stable_only_operator_needs_no_fallback(self):
        report = self.check_tree()
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["rows"][0]["backend"], "stable")

    def test_wrapper_argument_count_drift(self):
        self.assertTrue(any("wrapper passes 1 arguments" in item
                            for item in self.check_tree(wrapper_args=1)["blockers"]))

    def test_missing_run(self):
        self.assert_mutation("stable_new_op.cpp", "run_npu_new_op(", "missing_run(", "missing its run")

    def test_missing_definition(self):
        self.assert_mutation("stable_ops.cpp", "m.def(kSchema_npu_new_op);", "", "missing its schema")

    def test_missing_implementation(self):
        self.assert_mutation("stable_ops.cpp", 'm.impl("npu_new_op", &boxed_adapter<run_npu_new_op>);', "", "missing its impl")

    def test_wrong_implementation_name(self):
        self.assert_mutation("stable_ops.cpp", 'm.impl("npu_new_op"', 'm.impl("npu_wrong_op"', "implementation has no registered schema")

    def test_missing_schema(self):
        self.assert_mutation("stable_new_op.cpp", "kSchema_npu_new_op", "kSchema_removed", "registered schema is not defined")

    def test_missing_include(self):
        self.assert_mutation("stable_ops.cpp", '#include "stable_new_op.cpp"', "", "never compiled")

    def test_missing_source(self):
        report = self.check_tree(lambda ops, src: (src / "stable_new_op.cpp").unlink())
        self.assertTrue(any("included adapter source is missing" in item for item in report["blockers"]))

    def test_empty_inventory_and_deleted_public_api(self):
        self.assert_mutation("__init__.py", '"npu_new_op",', "", "operator inventory is empty")
        self.assert_mutation("__init__.py", '"npu_new_op",', "", "historical public API was removed")

    def test_duplicate_registration(self):
        self.assert_mutation("stable_ops.cpp", "m.def(kSchema_npu_new_op);", "m.def(kSchema_npu_new_op); m.def(kSchema_npu_new_op);", "duplicate schema")

    def test_same_type_argument_reorder_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            ops, src, contract = _stable_tree(tmp)
            frozen = {"operators": ["npu_new_op"], "dispatch_arguments": {
                "npu_new_op": [{"op": "npu_new_op", "expressions": ["a", "_current_stream_ptr()"]}]}}
            contract.write_text(json.dumps(frozen), encoding="utf-8")
            path = ops / "_stable.py"
            path.write_text(path.read_text().replace('(a, _current_stream_ptr())', '(_current_stream_ptr(), a)'), encoding="utf-8")
            with mock.patch.object(self.tool, "OPS_DIR", ops), \
                    mock.patch.object(self.tool, "SRC_DIR", src), \
                    mock.patch.object(self.tool, "PUBLIC_CONTRACT", contract):
                report = self.tool.evaluate()
            self.assertTrue(any("argument order/expression" in item for item in report["blockers"]), report)

    def test_json_cli_fails_without_strict_for_unexplained_gap(self):
        with mock.patch.object(self.tool, "evaluate", return_value={"rows": [], "blockers": ["broken: missing wrapper"]}), \
                mock.patch.object(self.tool, "load_baseline", return_value={}), \
                mock.patch.object(sys, "argv", ["coverage", "--json"]), \
                mock.patch("builtins.print"):
            self.assertEqual(self.tool.main(), 1)


class TranslationUnitTest(unittest.TestCase):
    """The gates and the build have to agree on what gets compiled.

    The adapter sources are only self-contained as a set: stable_ops.cpp
    includes the rest, and a using-declaration in one file's anonymous
    namespace is what the next file relies on.  A gate that compiles the
    ``.cpp`` files one at a time therefore checks something the build never
    does, and reports failures the build does not have.
    """

    def test_the_gates_check_what_the_builder_compiles(self) -> None:
        builder = _load_module(
            "fla_build_stable_gate", SETUP_DIR / "csrc" / "build_stable.py")
        audit = _load_tool("stable_abi_audit.py")
        vendor = _load_tool("vendor_stable_headers.py")
        self.assertEqual(builder.TRANSLATION_UNITS, audit.translation_units())
        self.assertEqual(builder.TRANSLATION_UNITS, vendor.translation_units())

    def test_the_builder_uses_that_list(self) -> None:
        source = (SETUP_DIR / "csrc" / "build_stable.py").read_text(
            encoding="utf-8")
        self.assertIn("[str(path) for path in TRANSLATION_UNITS]", source,
                      "build_stable.py stopped building the declared units")


class VendoredHeaderTest(unittest.TestCase):
    """The vendored torch headers are a copy, and a copy has to stay one.

    Everything else in this suite guards the adapters' shape; this guards the
    interface they are compiled against.  A header edited in place would move
    the runtime floor, or the meaning of a symbol, without a single line of
    adapter source changing -- which is exactly what pinning the copy to 2.9
    was for.
    """

    def setUp(self) -> None:
        self.tool = _load_tool("vendor_stable_headers.py")

    def _tree(self, tmp: str, *, count: int = 1,
              contents: str = "// upstream\n"):
        root = Path(tmp)
        for index in range(count):
            path = root / f"torch/csrc/stable/header{index}.h"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        manifest = "".join(
            f"{self.tool._digest(path)}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in sorted(root.rglob("*.h")))
        (root / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        return root

    def test_current_tree_matches_the_manifest(self) -> None:
        self.assertEqual([], self.tool.verify_tree())

    def test_an_edited_header_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            self.assertEqual([], self.tool.verify_tree(root))
            (root / "torch/csrc/stable/header0.h").write_text(
                "// locally patched\n", encoding="utf-8")
            problems = self.tool.verify_tree(root)
        self.assertTrue(any("edited:" in line for line in problems), problems)

    def test_an_unlisted_header_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            extra = root / "torch/csrc/stable/extra.h"
            extra.write_text("// new\n", encoding="utf-8")
            problems = self.tool.verify_tree(root)
        self.assertTrue(any("not listed" in line for line in problems), problems)

    def test_a_deleted_header_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, count=2)
            (root / "torch/csrc/stable/header1.h").unlink()
            problems = self.tool.verify_tree(root)
        self.assertTrue(any("missing from the tree" in line for line in problems),
                        problems)


if __name__ == "__main__":
    unittest.main()
