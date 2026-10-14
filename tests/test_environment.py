"""Environmental extraction checks using synthetic fixtures only."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/01_data_preparation/01_environmental_covariates"))
from environment_core import (
    FIELDS,
    Observations,
    MonthlyProduct,
    canonical_source,
    pad_and_sample,
    median_fallback,
    sha256,
)


class ExtractionTests(unittest.TestCase):
    def baseline_config(self, directory):
        """Create a minimal workbook without local datasets or product files."""
        directory = Path(directory)
        frame = pd.DataFrame(
            {
                "Number": ["synthetic-1", "synthetic-2", "synthetic-3"],
                "Samplingtime": ["2020.07.31", "2020.07.15", "2020.08.01"],
                "Longitude": [120.0, 120.1, 120.2],
                "Latitude": [38.0, 38.1, 38.2],
                **{field: [np.nan] * 3 for field in FIELDS},
            }
        )
        frame["Temperature"] = [12.0, np.nan, 14.0]
        frame["Temperature_Source"] = ["Original Data (fixture)", None, None]
        workbook = directory / "observations.xlsx"
        frame.to_excel(workbook, sheet_name="Spatiotemporal identification", index=False)
        return {
            "workbook": str(workbook),
            "sheet": "Spatiotemporal identification",
            "output_directory": str(directory / "outputs"),
        }

    def test_original_labels_normalize(self):
        for value in [
            "Original Data",
            "Original Data (D04)",
            "Original Data; sampled layer average",
            None,
        ]:
            self.assertEqual(canonical_source(value, True), "Original Data")
        self.assertEqual(
            canonical_source("NASA_Bilinear_Downscaled", True), "NASA Extraction"
        )
        self.assertEqual(canonical_source(None, False), "Unfilled")
        with self.assertRaises(ValueError):
            canonical_source("unknown method", True)

    def test_grid_orientation_and_bounds(self):
        da = xr.DataArray(
            [[2.0, 3.0], [0.0, 1.0]],
            dims=["lat", "lon"],
            coords={"lat": [1.0, 0.0], "lon": [0.0, 1.0]},
        )
        frame = pd.DataFrame({"Latitude": [0.5, 3.0], "Longitude": [0.5, 0.5]})
        result = pad_and_sample(da, frame, padding=0)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(
            np.isnan(pad_and_sample(da, frame, method="nearest", padding=0)[1])
        )

    def test_month_end_matches_same_month(self):
        product = MonthlyProduct.__new__(MonthlyProduct)
        ds = xr.Dataset(
            {
                "x": (
                    ("time", "latitude", "longitude"),
                    np.array([np.ones((2, 2)), np.ones((2, 2)) * 9]),
                )
            },
            coords={
                "time": pd.to_datetime(["2020-07-01", "2020-08-01"]),
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            },
        )
        product.variable = "x"
        product.periods = {(2020, 7): (ds, 0, "fixture"), (2020, 8): (ds, 1, "fixture")}
        frame = pd.DataFrame(
            {"Latitude": [0.5], "Longitude": [0.5], "_year": [2020], "_month": [7]}
        )
        values, details = product.sample(frame, padding=0)
        self.assertEqual(values[0], 1.0)
        self.assertEqual(details[0]["months"], ["2020-07"])
        frame["_year"] = 2030
        self.assertTrue(np.isnan(product.sample(frame, padding=0)[0][0]))

    def test_synthetic_input_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.baseline_config(temporary)
            original_hash = sha256(config["workbook"])
            obs = Observations(config)
            self.assertEqual(len(obs.frame), 3)
            self.assertEqual(obs.frame.POC.notna().sum(), 0)
            self.assertEqual(obs.frame.Temperature.notna().sum(), 2)
            self.assertEqual(
                obs.sources.loc[obs.frame.Temperature.notna(), "Temperature"]
                .unique()
                .tolist(),
                ["Original Data"],
            )
            index = obs.frame.index[obs.frame.Temperature.notna()][0]
            with self.assertRaises(ValueError):
                obs.set(index, "Temperature", 20.0, "CMEMS Extraction")
            self.assertEqual(sha256(config["workbook"]), original_hash)

    def test_empty_fallback_stays_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.baseline_config(temporary)
            obs = Observations(config)
            median_fallback(obs, "POC", monthly=True)
            self.assertTrue(obs.frame.POC.isna().all())
            self.assertEqual(set(obs.sources.POC), {"Unfilled"})


if __name__ == "__main__":
    unittest.main()
