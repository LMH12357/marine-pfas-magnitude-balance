"""Source preparation checks using synthetic fixtures only."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "code" / "01_data_preparation" / "02_source_covariates"
sys.path.insert(0, str(MODULE_DIR))
import source_common as common


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fingerprint = load_module("fingerprint_test", "02_river_fingerprint_construction.py")
coastline = load_module("coastline_test", "04_coastline_source_descriptors.py")


class SourcePreparationTests(unittest.TestCase):
    def test_agreed_taxonomy_boundaries(self):
        self.assertEqual(len(common.PFAS_COMPOUNDS), 19)
        self.assertEqual(common.TAXONOMY["LC_PFCA"][0], "PFOA")
        self.assertEqual(common.TAXONOMY["LC_PFSA"][0], "PFHxS")
        self.assertFalse(set(("PFNS", "PFECHS", "PFOSA")) & set(common.PFAS_COMPOUNDS))

    def test_case_distinct_rivers_remain_distinct(self):
        rivers = pd.DataFrame(
            {
                "ID": [66, 67],
                "River Name": ["AO River", "Ao River"],
                "Longitude": [120.581] * 2,
                "Latitude": [27.562] * 2,
            }
        )
        first = pd.Series({"River": "AO River", "Orig_Lon": 120.581, "Orig_Lat": 27.562})
        second = pd.Series({"River": "Ao River", "Orig_Lon": 120.581, "Orig_Lat": 27.562})
        self.assertEqual(fingerprint._match_inventory(first, rivers)[0], [0])
        self.assertEqual(fingerprint._match_inventory(second, rivers)[0], [1])
        with self.assertRaises(ValueError):
            fingerprint._match_inventory(
                pd.Series({"River": "ao river", "Orig_Lon": 120.581, "Orig_Lat": 27.562}),
                rivers,
            )

    def test_conflicting_known_seas_fail(self):
        frame = pd.DataFrame(
            {"River": ["A", "A"], "District I": ["Bohai Sea", "Yellow Sea"]}
        )
        with self.assertRaises(ValueError):
            fingerprint._known_seas(frame)

    def test_observed_borrowed_and_missing_are_distinct(self):
        records = []
        for value in (1.0, 3.0, 5.0):
            record = {compound: np.nan for compound in common.PFAS_COMPOUNDS}
            record.update(
                {
                    "River": "Synthetic River",
                    "District I": "East China Sea",
                    "PFOA": value,
                    "PFOS": 2.0,
                    "PFBA": 4.0,
                    "PFBS": 1.0,
                }
            )
            records.append(record)
        with tempfile.TemporaryDirectory(prefix="source_fixture_") as temporary:
            path = Path(temporary) / "fingerprints.csv"
            with (
                patch.object(
                    fingerprint,
                    "read_estuary_inventory",
                    return_value=pd.DataFrame(records),
                ),
                patch.object(fingerprint, "write_manifest"),
            ):
                frame = fingerprint.extract_regional_fingerprints(
                    Path("synthetic.xlsx"), path
                ).set_index("Sea_Area")
            self.assertEqual(frame.loc["East China Sea", "PFOA_Mean"], 3.0)
            self.assertEqual(frame.loc["East China Sea", "PFOA_Std"], 2.0)
            self.assertEqual(frame.loc["South China Sea", "PFOA_Count"], 0)
            self.assertEqual(frame.loc["South China Sea", "PFOA_Donor_Count"], 3)
            self.assertEqual(
                frame.loc["South China Sea", "PFOA_Origin"], "Adjacent_Borrowing"
            )
            self.assertTrue(pd.isna(frame.loc["South China Sea", "PFODA_Mean"]))
            self.assertEqual(frame.loc["South China Sea", "PFODA_Origin"], "Unavailable")
            self.assertTrue(frame.loc["South China Sea", "Source_Eligible"])
            common.read_fingerprints(path)

    def test_source_ratios_conserve_full_precision(self):
        compounds = {
            f"{compound}_Flux_Kg": 0.123456789123 for compound in common.PFAS_COMPOUNDS
        }
        row = {
            "Year": 2018,
            "Grid_Lon": 120.0,
            "Grid_Lat": 30.0,
            "Expected_Total_Flux_Kg": sum(compounds.values()),
            **compounds,
        }
        with tempfile.TemporaryDirectory(prefix="source_fixture_") as temporary:
            input_file = Path(temporary) / "flux.csv"
            output_file = Path(temporary) / "predictors.csv"
            pd.DataFrame([row]).to_csv(input_file, index=False)
            frame = coastline.build_coastline_source_load(input_file, output_file)
            self.assertAlmostEqual(
                frame.LC_PFCA_Flux_Kg.iloc[0], 9 * 0.123456789123, places=12
            )
            self.assertAlmostEqual(
                frame.Source_Ratio_LC_PFSA_PFCA.iloc[0], 4 / 9, places=12
            )
            self.assertAlmostEqual(
                frame.Load_Total_Legacy_Kg.iloc[0], sum(compounds.values()), places=12
            )
            row["Expected_Total_Flux_Kg"] += 1
            pd.DataFrame([row]).to_csv(input_file, index=False)
            with self.assertRaises(ValueError):
                coastline.build_coastline_source_load(input_file, output_file)


if __name__ == "__main__":
    unittest.main()
