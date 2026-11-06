"""Regression for the public int64 -> uint32 tiling conversion boundary."""

import importlib.util
from pathlib import Path
import unittest


CONTRACT = Path(__file__).resolve().parents[1] / "fla_npu_experimental" / "_contract.py"
spec = importlib.util.spec_from_file_location("_experimental_contract_under_test", CONTRACT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class ChunkLenTest(unittest.TestCase):
    def test_representable_positive_lengths(self):
        for length in (1, 16, (1 << 32) - 1):
            with self.subTest(length=length):
                self.assertEqual(contract.validate_chunk_len(length), length)

    def test_rejects_zero_negative_and_uint32_wraparound(self):
        for length in (-1, 0, 1 << 32, (1 << 32) + 1, (1 << 63) - 1):
            with self.subTest(length=length):
                with self.assertRaisesRegex(RuntimeError, r"chunk_len must be in \[1, 4294967295\]"):
                    contract.validate_chunk_len(length)


if __name__ == "__main__":
    unittest.main()
