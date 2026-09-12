"""由统一 JSON 驱动 ChunkKdaFwdPrepare 的公开参数负向测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
WRAPPER_FIXTURE = (
    ROOT / "torch_custom/fla_npu/test/test_chunk_kda_fwd_prepare_wrapper.py"
)


def _load_wrapper_fixture():
    module_name = "chunk_kda_fwd_prepare_negative_fixture"
    spec = importlib.util.spec_from_file_location(module_name, WRAPPER_FIXTURE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {WRAPPER_FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


FIXTURE = _load_wrapper_fixture()
DTYPES = {
    "bfloat16": FIXTURE.BFLOAT16,
    "float16": FIXTURE.FLOAT16,
    "float32": FIXTURE.FLOAT32,
}


def _decode(value):
    if not isinstance(value, dict) or "fake_tensor" not in value:
        return value
    tensor = value["fake_tensor"]
    return FIXTURE._FakeTensor(tensor["shape"], DTYPES[tensor["dtype"]])


class ChunkKdaFwdPrepareNegativeContractTest(unittest.TestCase):
    def test_manifest_negative_cases_fail_before_aclnn_launch(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = manifest["negative_cases"]
        self.assertTrue(cases)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

        for case in cases:
            if case.get("raw_only", False):
                continue
            fixture_args = dict(case["fixture"])
            for name in ("q_dtype", "gate_dtype", "beta_dtype"):
                if name in fixture_args:
                    fixture_args[name] = DTYPES[fixture_args[name]]
            inputs = FIXTURE._make_inputs(**fixture_args)
            kwargs = {
                name: _decode(value)
                for name, value in case.get("kwargs", {}).items()
            }
            expected = case["expect"]
            self.assertIn(
                expected["return_code"],
                {"ACLNN_ERR_PARAM_INVALID", "ACLNN_ERR_PARAM_NULLPTR"},
            )
            self.assertEqual(expected["python_exception"], "RuntimeError")

            with self.subTest(case=case["id"]):
                with mock.patch.dict(sys.modules, {"torch": FIXTURE.FAKE_TORCH}):
                    with mock.patch.object(
                        FIXTURE.ACLNN_CTYPES, "_call_aclnn"
                    ) as aclnn_call:
                        with self.assertRaisesRegex(
                            RuntimeError, expected["message"]
                        ):
                            FIXTURE.ACLNN_CTYPES.npu_chunk_kda_fwd_prepare(
                                *inputs, **kwargs
                            )
                    aclnn_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
