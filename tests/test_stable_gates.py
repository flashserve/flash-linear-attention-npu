"""Tests for the Stable-ABI gates themselves.

A gate that quietly stops working is worse than no gate: it still prints OK
while nothing is checked.  These tests are offline (no torch, no NPU) and cover
both directions -- each gate passes on the tree as it stands, and it fails on an
input that is deliberately wrong.

Usage:  python -m unittest tests.test_stable_gates
"""
from __future__ import annotations

import importlib.util
import re
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


def _launcher_only_tree(tmp: str, *, declare: bool,
                        ctypes_defines: bool = False,
                        wrapper_args: int = 2):
    """A minimal tree holding one operator that has no ctypes wrapper.

    Everything the two gates read is synthesised: the published list, the
    launcher wrapper, the adapter and its registration.  Returning the paths
    lets a test point OPS_DIR/SRC_DIR (or REFERENCE/BACKENDS) at it.
    """

    ops = Path(tmp) / "ops"
    src = Path(tmp) / "src"
    ops.mkdir()
    src.mkdir()
    declaration = ('_LAUNCHER_ONLY_OPS: tuple[str, ...] = ("npu_new_op",)'
                   if declare else "_LAUNCHER_ONLY_OPS: tuple[str, ...] = ()")
    (ops / "__init__.py").write_text(
        '_ASCENDC_OPS = (\n    "npu_new_op",\n)\n\n' + declaration + "\n",
        encoding="utf-8")
    dispatch = ('    return _op("npu_new_op")(a, _current_stream_ptr())\n'
                if wrapper_args == 2 else '    return _op("npu_new_op")(a)\n')
    (ops / "_stable.py").write_text(
        "_ENUM = {}\n\n\n"
        "def _op(name):\n"
        "    return globals()[name]\n\n\n"
        "def _current_stream_ptr():\n"
        "    return 0\n\n\n"
        "def npu_new_op(a):\n" + dispatch,
        encoding="utf-8")
    (ops / "_aclnn_ctypes.py").write_text(
        "def npu_new_op(a):\n    pass\n" if ctypes_defines
        else "# no ctypes reference for this operator\n",
        encoding="utf-8")
    (src / "stable_new.cpp").write_text(
        "constexpr const char* kSchema_npu_new_op =\n"
        '    "npu_new_op(Tensor a, int stream) -> Tensor";\n\n'
        "Tensor run_npu_new_op(Tensor a, int64_t stream) {\n"
        "  return a;\n"
        "}\n",
        encoding="utf-8")
    (src / "stable_ops.cpp").write_text(
        "STABLE_TORCH_LIBRARY(fla_npu_stable, m) {\n"
        "  m.def(kSchema_npu_new_op);\n"
        "}\n\n"
        "STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, "
        "CompositeExplicitAutograd, m) {\n"
        '  m.impl("npu_new_op", &boxed_adapter<run_npu_new_op>);\n'
        "}\n",
        encoding="utf-8")
    return ops, src


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
            target = sources[0]
            original = target.read_text(encoding="utf-8")
            try:
                target.write_text(original + "\n// changed\n", encoding="utf-8")
                self.assertNotEqual(
                    module.source_hash(), first,
                    "editing one adapter must change the stamp")
            finally:
                target.write_text(original, encoding="utf-8")
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
        # One adapter per published operator on this branch; the floor is the
        # count itself so a silently dropped adapter still fails the test.
        self.assertGreaterEqual(report["adapter_count"], 23)

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
            (ops / "_aclnn_ctypes.py").write_text(
                (OPS_DIR / "_aclnn_ctypes.py").read_text(encoding="utf-8"),
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
            target = src / "stable_kda.cpp"
            text = target.read_text(encoding="utf-8")
            # Swap two layout names in the C++ table only: the Python table no
            # longer agrees, which is exactly the silent-layout-change bug.
            changed = text.replace(
                'kChunkKdaFwdLayoutNames[] = {"BSND", "BNSD", "TND",\n'
                '                                                   "NTD"}',
                'kChunkKdaFwdLayoutNames[] = {"BNSD", "BSND", "TND",\n'
                '                                                   "NTD"}')
            self.assertNotEqual(changed, text)
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("kChunkKdaFwdLayoutNames order" in item
                                for item in report["blockers"]),
                            report["blockers"])


class AbiParityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool("op_abi_parity.py")

    def test_current_tree_matches(self) -> None:
        report = self.tool.evaluate()
        self.assertEqual(report["problems"], [])
        self.assertGreaterEqual(report["checked"], 23)
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
            target = src / "stable_kda.cpp"
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
            target = src / "stable_recurrent_gdr.cpp"
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


class CtypesTableGateTest(unittest.TestCase):
    """The ctypes argument table is what the OPP headers are compared against."""

    def test_every_entry_ends_with_workspace_and_executor(self) -> None:
        tool = _load_tool("op_abi_validate.py")
        table = tool.parse_ctypes_table(OPS_DIR / "_aclnn_ctypes.py")
        # This branch's ctypes module keeps a static argtype table only for the
        # entries that predate the per-call form; every other call site is
        # parsed from its own list, so the floor is what the table holds here.
        self.assertGreaterEqual(len(table), 12)
        for symbol, kinds in table.items():
            with self.subTest(symbol=symbol):
                # The trailing pair is dropped by the parser, so what is left
                # must not contain a pointer-to-out-parameter.
                self.assertNotIn("_pointer", kinds)
                self.assertTrue(kinds, f"{symbol} parsed to nothing")

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


class LauncherOnlyCoverageTest(unittest.TestCase):
    """An operator with no ctypes wrapper has to be a declared state.

    The point is that a new operator can ship without the ctypes adaptation at
    all: nothing else in the tree is allowed to assume it exists.  These tests
    build a one-operator tree so the behaviour does not depend on how many real
    operators happen to be launcher-only today.
    """

    def setUp(self) -> None:
        self.tool = _load_tool("stable_coverage.py")

    def _evaluate(self, tmp: str, **kwargs) -> dict:
        ops, src = _launcher_only_tree(tmp, **kwargs)
        with mock.patch.object(self.tool, "OPS_DIR", ops), \
                mock.patch.object(self.tool, "SRC_DIR", src):
            return self.tool.evaluate()

    def test_declared_launcher_only_operator_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=True)
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["rows"][0]["reference"], "launcher-only")
        self.assertEqual(report["launcher_only"], ["npu_new_op"])

    def test_undeclared_operator_without_a_reference_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=False)
        self.assertTrue(any("_LAUNCHER_ONLY_OPS" in item
                            for item in report["blockers"]), report["blockers"])

    def test_stale_declaration_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=True, ctypes_defines=True)
        self.assertTrue(any("still defines it" in item
                            for item in report["blockers"]), report["blockers"])

    def test_wrapper_argument_count_drift_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=True, wrapper_args=1)
        self.assertTrue(any("wrapper passes 1 arguments" in item
                            for item in report["blockers"]), report["blockers"])


class LauncherOnlySignatureTest(unittest.TestCase):
    """op_api_parity must not silently skip an operator ctypes does not define."""

    def setUp(self) -> None:
        self.tool = _load_tool("op_api_parity.py")

    def _evaluate(self, tmp: str, **kwargs) -> dict:
        ops, _src = _launcher_only_tree(tmp, **kwargs)
        with mock.patch.object(self.tool, "OPS_DIR", ops), \
                mock.patch.object(self.tool, "REFERENCE",
                                  ops / "_aclnn_ctypes.py"), \
                mock.patch.object(self.tool, "BACKENDS",
                                  {"stable": ops / "_stable.py"}):
            return self.tool.evaluate()

    def test_declared_operator_is_recorded_as_launcher_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=True)
        self.assertEqual([row["problems"] for row in report["rows"]], [[]])
        self.assertEqual(report["rows"][0]["reference"], "launcher-only")

    def test_undeclared_operator_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evaluate(tmp, declare=False)
        self.assertTrue(any("_LAUNCHER_ONLY_OPS" in problem
                            for row in report["rows"]
                            for problem in row["problems"]),
                        report["rows"])


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
