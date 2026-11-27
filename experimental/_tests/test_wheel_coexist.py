"""File-level collision policy for independently built operator wheels."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from experimental_fixtures import ABI_HELPER
from tests.opp_abi_fixtures import write_elf_exports

sys.path.insert(0, str(ABI_HELPER.parents[1]))
try:
    from fla_npu_coexist import InstalledPair
finally:
    sys.path.pop(0)


class CoexistTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.formal = root / "fla_npu"
        self.experimental = root / "fla_npu_experimental"
        self.formal.mkdir()
        shutil.copyfile(ABI_HELPER, self.formal / "_op_abi.py")
        env = patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": str(root / "external")})
        env.start()
        self.addCleanup(env.stop)

    def vendor(self, root, name, op_type, symbol):
        path = root / "opp/vendors" / name
        config = path / "op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({op_type: {}}))
        write_elf_exports(path / "op_api/lib/libcust_opapi.so", [symbol])
        return path

    def test_uses_real_exports_and_keeps_winner_on_independent_conflicts(self):
        for kind, formal_first in (("CANN opType", True), ("CANN opType", False),
                                   ("ACLNN", True), ("ACLNN", False)):
            with self.subTest(kind=kind, formal_first=formal_first):
                tag = kind.replace(" ", "") + str(formal_first)
                base = self.vendor(self.formal, tag, "A", "aclnnA")
                peer = self.vendor(self.experimental, tag, "A" if kind == "CANN opType" else "B",
                                   "aclnnA" if kind == "ACLNN" else "aclnnB")
                first, second = self.formal, self.experimental
                if not formal_first:
                    first, second = second, first
                pair = InstalledPair(self.formal, self.experimental, [base], lambda _: [peer])
                pair.startup()
                external = str(self.formal.parent / "external")
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep), [str(base), external])
                with pair.activating(first):
                    pass
                selected = os.environ["ASCEND_CUSTOM_OPP_PATH"]
                self.assertEqual(selected.split(os.pathsep), [str(base if formal_first else peer), external])
                with self.assertRaises(ImportError) as error:
                    with pair.activating(second):
                        self.fail("conflicting package started loading")
                name = "A" if kind == "CANN opType" else "aclnnA"
                self.assertIn(f"{kind} {name}: {base} and {peer}", str(error.exception))
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], selected)
                self.assertEqual(pair.active, {first})

    def test_native_formal_vendor_keeps_selection_before_python_import(self):
        base = self.vendor(self.formal, "base", "A", "aclnnA")
        peer = self.vendor(self.experimental, "exp", "A", "aclnnB")
        pair = InstalledPair(self.formal, self.experimental, [base], lambda _: [peer])
        pair.startup()
        previous = os.environ["ASCEND_CUSTOM_OPP_PATH"]
        with patch.object(pair, "_loaded_formal_vendor", return_value=str(base / "op_proto/lib/proto.so")):
            with self.assertRaisesRegex(ImportError, "CANN opType A"):
                with pair.activating(self.experimental):
                    self.fail("CANN already registered the formal vendor")
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], previous)
        with pair.activating(self.formal):
            pass
        self.assertEqual(pair.active, {self.formal})

    def test_disjoint_vendors_keep_local_overlay_order(self):
        base = self.vendor(self.formal, "base", "A", "aclnnA")
        update = self.vendor(self.formal, "update", "A", "aclnnA")
        peer = self.vendor(self.experimental, "exp", "B", "aclnnB")
        pair = InstalledPair(self.formal, self.experimental, [update, base], lambda _: [peer])
        pair.startup()
        for root in (self.experimental, self.formal):
            with pair.activating(root):
                pass
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)[:3],
                         list(map(str, (update, base, peer))))

    def test_failed_first_import_releases_selection_for_retry(self):
        a = self.vendor(self.formal, "a", "A", "aclnnA")
        b = self.vendor(self.experimental, "b", "A", "aclnnB")
        pair = InstalledPair(self.formal, self.experimental, [a], lambda _: [b])
        pair.startup()
        initial = os.environ["ASCEND_CUSTOM_OPP_PATH"]
        with self.assertRaises(ValueError):
            with pair.activating(self.experimental):
                raise ValueError("invalid package")
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], initial)
        with pair.activating(self.formal):
            pass
        self.assertEqual(pair.active, {self.formal})

    def test_broken_experimental_layer_does_not_disable_formal(self):
        base = self.vendor(self.formal, "a", "A", "aclnnA")

        def missing_layer(_):
            raise RuntimeError("Selected OPP is missing")

        pair = InstalledPair(self.formal, self.experimental, [base], missing_layer)
        pair.startup()
        with self.assertRaisesRegex(ImportError, "Selected OPP is missing"):
            with pair.activating(self.experimental):
                self.fail("damaged side was enabled")
        with pair.activating(self.formal):
            self.assertIn(str(base), os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep))
