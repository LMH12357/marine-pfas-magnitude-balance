"""Synthetic checks for stream reproducibility, expectations and flux conservation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/01_data_preparation/02_source_covariates"
sys.path.insert(0, str(CODE))
SPEC = importlib.util.spec_from_file_location(
    "source_mc", CODE / "03_monte_carlo_river_flux.py"
)
MC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MC)
from source_common import PFAS_COMPOUNDS, SEA_AREAS


def synthetic_fingerprint(sea: str) -> pd.Series:
    record = {"Sea_Area": sea, "Source_Eligible": True}
    for compound in PFAS_COMPOUNDS:
        record.update(
            {
                f"{compound}_Mean": 1.0,
                f"{compound}_Std": 0.0,
                f"{compound}_Count": 3,
                f"{compound}_Origin": "Observed",
                f"{compound}_Donor_Count": 0,
                f"{compound}_Donor_Sea": "",
            }
        )
    record.update(
        {
            "PFNA_Mean": np.nan,
            "PFNA_Std": np.nan,
            "PFNA_Count": 0,
            "PFNA_Origin": "Unavailable",
        }
    )
    return pd.Series(record)


class SourceMonteCarloTests(unittest.TestCase):
    def test_clipped_normal_mean_matches_independent_quadrature(self):
        for mean, sd in [(0.0, 1.0), (1.0, 1.0), (4.0, 0.8)]:
            x = np.linspace(0.0, mean + 10 * sd, 300_001)
            density = np.exp(-0.5 * np.square((x - mean) / sd)) / (
                sd * np.sqrt(2 * np.pi)
            )
            reference = np.trapezoid(x * density, x)
            self.assertAlmostEqual(
                MC.clipped_normal_mean(mean, sd), reference, places=8
            )
        self.assertEqual(MC.clipped_normal_mean(3.2, 0.0), 3.2)

    def test_keyed_stream_does_not_depend_on_other_streams(self):
        first = MC.keyed_generator(42, "fingerprint", "Bohai Sea", "PFOS").normal(
            size=17
        )
        MC.keyed_generator(42, "fingerprint", "Yellow Sea", "PFBA").normal(size=999)
        again = MC.keyed_generator(42, "fingerprint", "Bohai Sea", "PFOS").normal(
            size=17
        )
        np.testing.assert_array_equal(first, again)
        self.assertFalse(
            np.array_equal(
                first,
                MC.keyed_generator(43, "fingerprint", "Bohai Sea", "PFOS").normal(
                    size=17
                ),
            )
        )

    def test_missing_component_has_no_allocation_and_constant_shares_are_exact(self):
        fractions, diagnostics = MC.integrate_regional_fingerprint(
            synthetic_fingerprint("Bohai Sea"), 31, 42
        )
        self.assertEqual(fractions["PFNA"], 0.0)
        self.assertAlmostEqual(sum(fractions.values()), 1.0)
        self.assertTrue(
            all(
                abs(value - 1 / 18) < 1e-14
                for key, value in fractions.items()
                if key != "PFNA"
            )
        )
        missing = next(row for row in diagnostics if row["Compound"] == "PFNA")
        self.assertEqual(missing["Allocation_Status"], "Unavailable")
        self.assertEqual(missing["Observed_Count"], 0)

    def test_zero_fingerprint_draw_is_rejected_without_mass_loss(self):
        row = synthetic_fingerprint("Bohai Sea")
        for compound in PFAS_COMPOUNDS:
            if row[f"{compound}_Origin"] != "Unavailable":
                row[f"{compound}_Mean"] = 0.0
        with self.assertRaisesRegex(ValueError, "no positive mass"):
            MC.integrate_regional_fingerprint(row, 3, 42)

    def test_shared_grid_is_not_double_counted_and_order_is_irrelevant(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            runoff = pd.DataFrame(
                {
                    "River": ["A", "B", "C"],
                    "Grid_Lon": [120.0, 120.0, 121.0],
                    "Grid_Lat": [38.0, 38.0, 38.0],
                    "Year": [2018] * 3,
                    "Annual_Q_10^8m3": [1.0, 1.0, 2.0],
                    "Q_SD_10^8m3": [0.0] * 3,
                }
            )
            runoff_file = folder / "Step1_Annual_Runoff_2014_2025.csv"
            runoff.to_csv(runoff_file, index=False)
            mapping = pd.DataFrame(
                {
                    "Grid_Lon": [120.0, 121.0],
                    "Grid_Lat": [38.0, 38.0],
                    "Sea_Area": ["Bohai Sea"] * 2,
                    "∑PFAAs (ng/L)": [10.0, 20.0],
                }
            )
            mapping.to_csv(folder / "Step2_River_Sea_Mapping.csv", index=False)
            pd.DataFrame([synthetic_fingerprint(sea) for sea in SEA_AREAS]).to_csv(
                folder / "Step2_Regional_Fingerprints.csv", index=False
            )
            pd.DataFrame(
                {
                    "River Name": ["A", "B", "C"],
                    "Longitude": [120.0, 120.0, 121.0],
                    "Latitude": [38.0] * 3,
                    "Runoff": [1.0, 1.0, 2.0],
                    "∑PFAAs (ng/L)": [10.0, 10.0, 20.0],
                }
            ).to_excel(folder / "100_Rivers_2018.xlsx", index=False)
            first_file = folder / "first.csv"
            first = MC.run_monte_carlo_baseline_flux(folder, first_file, iterations=31)
            self.assertEqual(len(first), 2)
            np.testing.assert_array_equal(first.Expected_Total_Flux_Kg, [1.0, 4.0])
            columns = [f"{compound}_Flux_Kg" for compound in PFAS_COMPOUNDS]
            np.testing.assert_allclose(
                first[columns].sum(axis=1), first.Expected_Total_Flux_Kg, rtol=1e-13
            )
            shares = first[columns].div(first.Expected_Total_Flux_Kg, axis=0).to_numpy()
            np.testing.assert_array_equal(shares[0], shares[1])
            runoff.iloc[::-1].to_csv(runoff_file, index=False)
            second = MC.run_monte_carlo_baseline_flux(
                folder, folder / "second.csv", iterations=31
            )
            pd.testing.assert_frame_equal(first, second)
            manifest = json.loads(
                first_file.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["fingerprint_iterations_per_region"], 31)
            runoff.loc[1, "Annual_Q_10^8m3"] = 1.1
            runoff.to_csv(runoff_file, index=False)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                MC.run_monte_carlo_baseline_flux(
                    folder, folder / "invalid.csv", iterations=31
                )


if __name__ == "__main__":
    unittest.main()
