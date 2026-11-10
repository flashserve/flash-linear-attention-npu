"""Offline tests for the recurrent GDN capability publication.

The value has to travel with the OPP, not with the Python wheel: a caller that
hard-codes the wheel's window keeps sending 9+ token sequences to a kernel that
silently skips them.  These tests cover both halves -- the publication in the
operator directory and the reader that finds it in the winning vendor.

Usage:  python -m unittest tests.test_rgdr_capabilities
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu" / "ops" / "ascendc"
GDN_ROOT = (
    REPO_ROOT
    / "fla"
    / "ops"
    / "ascendc"
    / "gdn"
    / "recurrent_gdn"
    / "recurrent_gated_delta_rule"
    / "op_kernel"
)
IMPL_RELATIVE = Path("op_impl") / "ai_core" / "tbe" / "fla_npu_transformer_impl"
OP = "recurrent_gated_delta_rule"
# The capability file deliberately does not name its kernel headers: the window
# size lives only in those headers, and this mapping is the test's own.
KERNELS = {
    "arch22": GDN_ROOT / "arch22" / f"{OP}.h",
    "arch35": GDN_ROOT / "arch35" / f"{OP}.h",
}


def _load_capabilities():
    path = OPS_DIR / "_capabilities.py"
    spec = importlib.util.spec_from_file_location("fla_npu_rgdr_capabilities", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublishedCapabilitiesTest(unittest.TestCase):
    """The checked-in window must describe the kernels next to it."""

    def test_published_values_match_kernel_headers(self):
        published = json.loads((GDN_ROOT / "capabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(published["schema"], 1)
        self.assertEqual(published["op"], OP)

        arch22 = published["arches"]["arch22"]
        self.assertEqual(set(arch22), {"max_verify_len"})
        header22 = KERNELS["arch22"].read_text(encoding="utf-8")
        if arch22["max_verify_len"] is None:
            # A tiled loop consumes the whole sequence; no per-sequence cap.
            self.assertIn("position += kWindowTokens", header22)
            self.assertNotIn("MAX_MTP", header22)
        else:
            self.assertIn(f"MAX_MTP = {arch22['max_verify_len']};", header22)

        arch35 = published["arches"]["arch35"]
        self.assertEqual(set(arch35), {"max_verify_len"})
        header35 = KERNELS["arch35"].read_text(encoding="utf-8")
        self.assertIn("TOKEN_TILE", header35)
        self.assertNotIn("MAX_MTP", header35)
        self.assertIsNone(arch35["max_verify_len"])


class ReadCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_capabilities()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _op_dir(self, vendor: str) -> Path:
        path = self.root / vendor / IMPL_RELATIVE / "ascendc" / OP
        path.mkdir(parents=True)
        return path

    def _publish(self, op_dir: Path, arch: str, value) -> None:
        (op_dir / "capabilities.json").write_text(
            json.dumps({"schema": 1, "op": OP, "arches": {arch: {"max_verify_len": value}}}),
            encoding="utf-8",
        )

    def test_reads_published_cap(self):
        op_dir = self._op_dir("vendor")
        self._publish(op_dir, "arch22", 8)
        capability = self.module._resolve_capability("arch22", [self.root / "vendor"])
        self.assertEqual(capability.max_verify_len, 8)
        self.assertTrue(capability.published)
        self.assertEqual(capability.source, "opp-capabilities")

    def test_null_means_unbounded(self):
        op_dir = self._op_dir("vendor")
        self._publish(op_dir, "arch22", None)
        capability = self.module._resolve_capability("arch22", [self.root / "vendor"])
        self.assertIsNone(capability.max_verify_len)
        self.assertTrue(capability.published)

    def test_legacy_single_file_kernel_is_parsed(self):
        op_dir = self._op_dir("vendor")
        (op_dir / f"{OP}.h").write_text(
            "constexpr uint64_t MAX_MTP = 8;\n", encoding="utf-8"
        )
        capability = self.module._resolve_capability("arch22", [self.root / "vendor"])
        self.assertEqual(capability.max_verify_len, 8)
        self.assertEqual(capability.source, "legacy-kernel-header")

    def test_legacy_kernel_cannot_describe_arch35(self):
        op_dir = self._op_dir("vendor")
        (op_dir / f"{OP}.h").write_text(
            "constexpr uint64_t MAX_MTP = 8;\n", encoding="utf-8"
        )
        capability = self.module._resolve_capability("arch35", [self.root / "vendor"])
        self.assertFalse(capability.published)

    def test_overlay_without_publication_does_not_fall_through(self):
        overlay = self._op_dir("overlay")
        (overlay / "kernel.o").write_bytes(b"")
        base = self._op_dir("base")
        self._publish(base, "arch22", 8)
        capability = self.module._resolve_capability(
            "arch22", [self.root / "overlay", self.root / "base"]
        )
        self.assertFalse(capability.published)
        self.assertIsNone(capability.max_verify_len)

    def test_missing_operator_is_unpublished(self):
        capability = self.module._resolve_capability("arch22", [self.root / "empty"])
        self.assertFalse(capability.published)
        self.assertIsNone(capability.path)


class VendorResolverTest(unittest.TestCase):
    """Old wheels predate the overlay chain and expose only one resolver."""

    def setUp(self):
        self.module = _load_capabilities()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _inject(self, **attributes):
        module = type(sys)("fla_npu")
        for name, value in attributes.items():
            setattr(module, name, value)
        original = sys.modules.get("fla_npu")
        sys.modules["fla_npu"] = module
        self.addCleanup(self._restore, original)

    def _restore(self, original):
        if original is None:
            sys.modules.pop("fla_npu", None)
        else:
            sys.modules["fla_npu"] = original

    def test_prefers_the_overlay_chain(self):
        self._inject(
            _resolve_vendor_dirs=lambda: [self.tmp.name, "second"],
            _resolve_vendor_dir=lambda: "unused",
        )
        self.assertEqual(self.module._vendor_dirs(), [Path(self.tmp.name), Path("second")])

    def test_falls_back_to_the_single_vendor_resolver(self):
        self._inject(_resolve_vendor_dir=lambda: self.tmp.name)
        self.assertEqual(self.module._vendor_dirs(), [Path(self.tmp.name)])


class ValidateLengthsTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_capabilities()

    def _patch(self, capability):
        original = self.module.recurrent_gdn_capability
        self.module.recurrent_gdn_capability = lambda device_index=None: capability
        self.addCleanup(setattr, self.module, "recurrent_gdn_capability", original)

    def _capability(self, value, published=True):
        return self.module.RecurrentGdnCapability(
            "arch22", value, published, "opp-capabilities", "capabilities.json"
        )

    def test_rejects_a_sequence_over_the_published_window(self):
        self._patch(self._capability(8))
        with self.assertRaisesRegex(ValueError, r"at most 8 tokens.*got 9"):
            self.module.validate_recurrent_gdn_lengths([4, 9])

    def test_accepts_the_published_window(self):
        self._patch(self._capability(8))
        self.module.validate_recurrent_gdn_lengths([4, 8])

    def test_unbounded_accepts_any_length(self):
        self._patch(self._capability(None))
        self.module.validate_recurrent_gdn_lengths([4, 64])

    def test_unpublished_is_left_to_the_caller_probe(self):
        self._patch(self._capability(None, published=False))
        self.module.validate_recurrent_gdn_lengths([4, 64])


class ArchSelectionTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_capabilities()

    def test_known_device_names(self):
        for name in ("Ascend910B3", "Ascend910C", "Ascend910_93"):
            self.assertEqual(self.module._arch_from_device_name(name), "arch22")
        self.assertEqual(self.module._arch_from_device_name("Ascend950PR"), "arch35")

    def test_unknown_device_is_a_lookup_error(self):
        with self.assertRaises(LookupError):
            self.module._arch_from_device_name("Ascend310P")


if __name__ == "__main__":
    unittest.main()
