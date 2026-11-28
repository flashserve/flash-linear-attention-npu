"""KDA and GDN model catalog integrity."""
from pathlib import Path
import json
import unittest


class ModelCatalogTest(unittest.TestCase):
    def test_model_case_counts_and_distinct_ids(self):
        root = Path(__file__).resolve().parents[1] / "model_cases"
        for suite, field, expected in (("kda", "id", 39), ("gdn", "name", 5)):
            with self.subTest(suite=suite):
                rows = json.loads((root / f"{suite}_cases.json").read_bytes())["cases"]
                self.assertEqual(len(rows), expected)
                self.assertEqual(len({row[field] for row in rows}), expected)
