"""Small synthetic regression fixtures; never production observations."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "code" / "02_modeling_and_validation" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


composition = load("composition_validation_test", "05_compositional_validation.py")
magnitude = load("magnitude_validation_test", "06_magnitude_balance_validation.py")


class ValidationContracts(unittest.TestCase):
    def test_ilr_and_gdr_round_trip(self):
        values = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3]])
        np.testing.assert_allclose(
            composition.ilr_inverse(composition.ilr_transform(values)), values
        )
        np.testing.assert_allclose(
            composition.log_gdr_to_composition(
                np.log(composition.composition_to_gdr(values))
            ),
            values,
        )

    def test_magnitude_balance_round_trip(self):
        values = np.log([[2.0, 8.0], [0.5, 3.0]])
        coordinates = magnitude.component_to_magnitude_balance(values)
        np.testing.assert_allclose(
            magnitude.magnitude_balance_to_components(coordinates), values
        )
        outputs = magnitude.derived_outputs(values)
        np.testing.assert_allclose(
            outputs["Log_Long_Ratio"], np.sqrt(2) * coordinates[:, 1]
        )

    def test_model_panel_excludes_unassigned_analytes(self):
        self.assertNotIn("PFNS", composition.GROUP_DEFINITIONS["Long_PFSA"])
        self.assertNotIn("PFECHS", magnitude.LONG_PFSA)
        self.assertEqual(sum(map(len, composition.GROUP_DEFINITIONS.values())), 19)

    def test_alignment_rejects_reordered_or_stale_inputs(self):
        raw = pd.DataFrame(
            {
                "Sampling_Group_ID": [1, 2],
                "Longitude": [120.0, 121.0],
                "Latitude": [30.0, 31.0],
                "PFOS": [1.0, 2.0],
                "Temperature": [20.0, 21.0],
            }
        )
        composition.validate_input_alignment(raw, raw.copy())
        with self.assertRaisesRegex(ValueError, "differ"):
            composition.validate_input_alignment(
                raw, raw.iloc[::-1].reset_index(drop=True)
            )
        changed = raw.copy()
        changed.loc[0, "PFOS"] = 999
        with self.assertRaisesRegex(ValueError, "PFOS"):
            composition.validate_input_alignment(raw, changed)

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            composition.ilr_transform([[1, 2, np.inf, 3]])
        with self.assertRaises(ValueError):
            magnitude.component_to_magnitude_balance([[1, np.nan]])

    def test_group_holdouts_do_not_overlap(self):
        frame = pd.DataFrame({"Sampling_Group": np.repeat(["a", "b", "c"], 3)})
        for train, test, _ in magnitude.make_splits(frame, "sampling_group", 3, 42):
            self.assertFalse(
                set(frame.iloc[train].Sampling_Group)
                & set(frame.iloc[test].Sampling_Group)
            )


if __name__ == "__main__":
    unittest.main()
