"""Synthetic geography and frozen-routing contract tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, LineString

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import coastal_routing as routing


class CoastalRoutingTests(unittest.TestCase):
    def test_land_interior_not_boundary_touch(self):
        land = box(0, 0, 1, 1)
        lines = np.array(
            [
                LineString([(-1, 0.5), (2, 0.5)]),
                LineString([(-1, 0), (2, 0)]),
                LineString([(-1, 1), (0, 2)]),
            ],
            dtype=object,
        )
        self.assertEqual(
            routing.crosses_land(lines, land).tolist(), [True, False, False]
        )

    def test_local_refinement_resolves_real_channel_without_land_shortcut(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "land.geojson"
            gpd.GeoDataFrame(
                geometry=[box(0.4, -0.1, 0.6, 0.47), box(0.4, 0.53, 0.6, 1.1)], crs=4326
            ).to_file(path, driver="GeoJSON")
            config = {"bounds": [0, 0, 1, 1], "base_resolution": 0.2}
            coarse = routing.build_routing_network(path, config)
            self.assertGreater(coarse.audit["components"], 1)
            config["refinement_patches"] = [
                {"name": "channel", "bounds": [0.2, 0.2, 0.8, 0.8], "resolution": 0.025}
            ]
            fine = routing.build_routing_network(path, config)
            self.assertEqual(fine.audit["components"], 1)
            self.assertEqual(fine.audit["land_crossing_edges_remaining"], 0)
            self.assertEqual(
                len(fine.coordinates), len(set(map(tuple, fine.coordinates)))
            )
            with self.assertRaisesRegex(ValueError, "No wet"):
                routing.nearest_node(0.5, 0.1, fine, [0.49, 0.09, 0.51, 0.11])

    def fixture_bundle(self, directory):
        source = pd.DataFrame(
            {"source_id": ["a", "b"], "Grid_Lon": [0.0, 0.01], "Grid_Lat": [0.0, 0.0]}
        )
        obs = pd.DataFrame(
            {"row_index": [4, 7], "Longitude": [0.2, 0.3], "Latitude": [0.2, 0.3]}
        )
        source.to_csv(directory / "Source_Ocean_Node_Mapping.csv", index=False)
        obs.to_csv(directory / "Observation_Ocean_Node_Mapping.csv", index=False)
        # Both distinct original sources are mapped to the same numerical anchor.
        np.savez_compressed(
            directory / "Source_Observation_Distances.npz",
            distance_km=np.array([[1.0, 3.0], [1.0, 3.0]]),
            row_index=np.array([4, 7]),
            source_id=np.array(["a", "b"]),
        )
        manifest = {
            "schema": routing.SCHEMA,
            "geography_review_complete": True,
            "products": {p.name: routing.sha256(p) for p in directory.iterdir()},
        }
        (directory / "Routing_Manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        source["Load_Total_Legacy_Kg"] = [2.0, 6.0]
        for ratio in routing.RATIOS:
            source[ratio] = [1.0, 3.0]
        return source, obs

    def test_water_attachment_cannot_cross_a_peninsula(self):
        coordinates = np.array([[0.6, 0.4], [0.1, 0.9]])
        network = routing.RoutingNetwork(
            None, coordinates, box(0.4, 0, 0.5, 0.8), np.zeros(2, dtype=int), 0, {}
        )
        node, _ = routing.nearest_node(0.3, 0.4, network)
        self.assertEqual(node, 1)
        with self.assertRaisesRegex(ValueError, "No water-only attachment"):
            routing.nearest_node(0.3, 0.4, network, [0.55, 0.35, 0.65, 0.45])

    def test_disconnected_nearest_node_is_not_silently_skipped(self):
        network = routing.RoutingNetwork(
            None,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            shapely.Polygon(),
            np.array([1, 0]),
            0,
            {},
        )
        with self.assertRaisesRegex(ValueError, "disconnected"):
            routing.nearest_node(0.01, 0.01, network)

    def test_out_of_domain_point_is_rejected(self):
        network = routing.RoutingNetwork(
            None, np.array([[0.0, 0.0]]), shapely.Polygon(), np.array([0]), 0, {}
        )
        source = pd.DataFrame({"Grid_Lon": [2.0], "Grid_Lat": [0.0]})
        with self.assertRaisesRegex(ValueError, "outside the routing domain"):
            routing.map_points(source, network, {"bounds": [-1, -1, 1, 1]}, "source")

    def test_distance_ties_are_resolved_by_coordinates(self):
        xy = np.array([[1.0, 0.0], [-1.0, 0.0]])
        network = routing.RoutingNetwork(
            None, xy, shapely.Polygon(), np.zeros(2, dtype=int), 0, {}
        )
        node, _ = routing.nearest_node(0, 0, network)
        self.assertEqual(tuple(xy[node]), (-1.0, 0.0))

    def test_frozen_routes_preserve_source_loads_and_subset_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source, obs = self.fixture_bundle(path)
            result = routing.compute_routed_features(
                obs.iloc[::-1], source.iloc[::-1], path
            )
            np.testing.assert_allclose(result[routing.FEATURES[0]], [2.0, 4.0])
            np.testing.assert_allclose(result[routing.FEATURES[1]], 2.5)
            subset = routing.compute_routed_features(obs.iloc[[1]], source, path)
            self.assertEqual(subset.row_index.tolist(), [7])
            source["Load_Total_Legacy_Kg"] *= 0.5
            reduced = routing.compute_routed_features(obs, source, path)
            np.testing.assert_allclose(reduced[routing.FEATURES[0]], [2.0, 1.0])

    def test_coordinate_drift_missing_sources_and_tampering_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source, obs = self.fixture_bundle(path)
            with self.assertRaisesRegex(ValueError, "source set"):
                routing.compute_routed_features(obs, source.iloc[:1], path)
            changed = obs.copy()
            changed.loc[0, "Latitude"] += 0.001
            with self.assertRaisesRegex(ValueError, "Observation coordinates"):
                routing.compute_routed_features(changed, source, path)
            with (path / "Source_Ocean_Node_Mapping.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                routing.load_routing_bundle(path)

    def test_duplicate_original_source_coordinates_are_not_merged(self):
        source = pd.DataFrame({"Grid_Lon": [0.0, 0.0], "Grid_Lat": [0.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "unique original"):
            routing.map_points(source, None, {}, "source")


if __name__ == "__main__":
    unittest.main()
