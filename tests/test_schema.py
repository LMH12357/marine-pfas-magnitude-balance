"""Tests for the agreed model-panel and feature-label contract."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pfas_schema as schema


class SchemaTests(unittest.TestCase):
    def test_membership(self):
        self.assertEqual(len(schema.MODEL_COMPOUNDS), 19)
        self.assertEqual(len(set(schema.MODEL_COMPOUNDS)), 19)
        self.assertNotIn("PFNS", schema.MODEL_COMPOUNDS)
        self.assertNotIn("PFECHS", schema.MODEL_COMPOUNDS)

    def test_missing_is_not_zero(self):
        frame = pd.DataFrame(
            {"PFOS": [np.nan, 0, 2], "PFHxS": [np.nan, 1, 3], "PFECHS": [999] * 3}
        )
        result = schema.sum_group(frame, "Long_PFSA")
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(result.iloc[1:].tolist(), [1, 5])

    def test_label_mapping_does_not_change_values(self):
        result = schema.normalize_feature_columns(
            pd.DataFrame({"Salinity (‰)": [30.1]})
        )
        self.assertEqual(result.Salinity.tolist(), [30.1])

    def test_alias_conflict_is_rejected(self):
        with self.assertRaises(ValueError):
            schema.normalize_feature_columns(
                pd.DataFrame({"Salinity (‰)": [30.1], "Salinity": [31]})
            )

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            schema.sum_group(pd.DataFrame({"PFOS": [-1]}), "Long_PFSA")


if __name__ == "__main__":
    unittest.main()
