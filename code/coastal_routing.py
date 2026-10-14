"""Deterministic coastal routing with frozen source and observation assignments.

Routing positions are numerical anchors, not replacements for measured coordinates.
The graph is a geographic distance proxy, not a hydrodynamic transport model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import shapely
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse.csgraph import connected_components, dijkstra

SCHEMA = "marine-pfas-coastal-routing/1"
GEOD = Geod(ellps="WGS84")
RATIOS = [
    "Source_Ratio_SC_LC",
    "Source_Ratio_LC_PFSA_PFCA",
    "Source_Ratio_SC_PFSA_PFCA",
]
FEATURES = [
    "Diffused_Total_Legacy_Force",
    "Mixed_Source_Ratio_SC_LC",
    "Mixed_Source_Ratio_LC_PFSA_PFCA",
    "Mixed_Source_Ratio_SC_PFSA_PFCA",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crosses_land(segments, geometry):
    """Reject positive-length intersection with land interior, not coast tangency."""
    segments = np.asarray(segments, dtype=object)
    hit = shapely.intersects(geometry, segments)
    result = np.zeros(len(segments), dtype=bool)
    result[hit] = shapely.relate_pattern(segments[hit], geometry, "1********")
    return result


def coordinate_keys(frame, x, y):
    values = frame[[x, y]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Coordinates must be finite")
    if (np.abs(values[:, 0]) > 180).any() or (np.abs(values[:, 1]) > 90).any():
        raise ValueError("Coordinates must be valid longitude/latitude values")
    return [(round(lon, 9), round(lat, 9)) for lon, lat in values]


@dataclass
class RoutingNetwork:
    graph: object
    coordinates: np.ndarray
    land: object
    node_components: np.ndarray
    main_component: int
    audit: dict


def regular_mesh(bounds, step):
    xmin, ymin, xmax, ymax = bounds
    if not (0 < step and xmin < xmax and ymin < ymax):
        raise ValueError("Invalid routing mesh bounds or resolution")
    xs = np.round(
        np.arange(np.ceil((xmin - 1e-10) / step), np.floor((xmax + 1e-10) / step) + 1)
        * step,
        9,
    )
    ys = np.round(
        np.arange(np.ceil((ymin - 1e-10) / step), np.floor((ymax + 1e-10) / step) + 1)
        * step,
        9,
    )
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel())), xx.shape


def spherical_distance(a, b):
    """Retain the historical graph's Haversine edge-distance definition (km)."""
    a, b = np.radians(a), np.radians(b)
    delta = b - a
    h = (
        np.sin(delta[:, 1] / 2) ** 2
        + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin(delta[:, 0] / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build_routing_network(boundary_path, config):
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        if not config.get("boundary_crs"):
            raise ValueError(
                "A land polygon without CRS metadata requires explicit boundary_crs in the routing configuration"
            )
        boundary = boundary.set_crs(config["boundary_crs"])
    boundary = boundary.to_crs(4326)
    land = boundary.geometry.union_all()
    if not land.is_valid:
        raise ValueError(
            "Invalid land geometry requires explicit review; not repaired automatically"
        )
    shapely.prepare(land)
    base = {
        "name": "base",
        "bounds": config["bounds"],
        "resolution": config["base_resolution"],
    }
    meshes = [base, *config.get("refinement_patches", [])]
    index, coordinates, edge_set, mesh_audit = {}, [], set(), []
    for mesh in meshes:
        if mesh is not base:
            ratio = float(config["base_resolution"]) / float(mesh["resolution"])
            if ratio < 1 or not np.isclose(ratio, round(ratio)):
                raise ValueError(
                    "Refinement resolution must divide the base resolution"
                )
            if any(mesh["bounds"][i] < config["bounds"][i] for i in (0, 1)) or any(
                mesh["bounds"][i] > config["bounds"][i] for i in (2, 3)
            ):
                raise ValueError("Refinement patch lies outside the routing domain")
        xy, shape = regular_mesh(mesh["bounds"], mesh["resolution"])
        wet = ~shapely.covers(land, shapely.points(xy))
        local_to_global = np.full(len(xy), -1, dtype=int)
        for position in np.flatnonzero(wet):
            key = tuple(xy[position])
            if key not in index:
                index[key] = len(coordinates)
                coordinates.append(key)
            local_to_global[position] = index[key]
        ids = np.arange(len(xy)).reshape(shape)
        candidates, rejected = 0, 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = ids[: shape[0] - dr, max(0, -dc) : min(shape[1], shape[1] - dc)].ravel()
            b = ids[dr:, max(0, dc) : min(shape[1], shape[1] + dc)].ravel()
            keep = wet[a] & wet[b]
            a, b = a[keep], b[keep]
            segments = shapely.linestrings(np.stack((xy[a], xy[b]), axis=1))
            invalid = crosses_land(segments, land)
            candidates += len(a)
            rejected += int(invalid.sum())
            for first, second in zip(
                local_to_global[a[~invalid]], local_to_global[b[~invalid]]
            ):
                edge_set.add((min(first, second), max(first, second)))
        mesh_audit.append(
            {
                "name": mesh["name"],
                "resolution": mesh["resolution"],
                "wet_nodes": int(wet.sum()),
                "candidate_edges": candidates,
                "land_crossing_edges_rejected": rejected,
            }
        )
    xy = np.asarray(coordinates, dtype=float)
    edges = np.asarray(sorted(edge_set), dtype=int).reshape(-1, 2)
    lengths = spherical_distance(xy[edges[:, 0]], xy[edges[:, 1]])
    graph = coo_matrix(
        (
            np.r_[lengths, lengths],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(xy), len(xy)),
    ).tocsr()
    count, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    main = int(np.argmax(sizes))
    audit = {
        "meshes": mesh_audit,
        "nodes": len(xy),
        "undirected_edges": len(edges),
        "components": int(count),
        "main_component_nodes": int(sizes[main]),
        "land_crossing_edges_remaining": int(
            crosses_land(shapely.linestrings(xy[edges]), land).sum()
        ),
    }
    if audit["land_crossing_edges_remaining"]:
        raise RuntimeError("The completed network contains land-crossing edges")
    return RoutingNetwork(graph, xy, land, components, main, audit)


def nearest_node(lon, lat, network, bounds=None, max_distance_km=None):
    candidates = np.ones(len(network.coordinates), dtype=bool)
    xy = network.coordinates
    if bounds is not None:
        candidates &= (
            (xy[:, 0] >= bounds[0])
            & (xy[:, 1] >= bounds[1])
            & (xy[:, 0] <= bounds[2])
            & (xy[:, 1] <= bounds[3])
        )
    ids = np.flatnonzero(candidates)
    if not len(ids):
        raise ValueError(
            "No wet routing node exists inside the reviewed geographic bounds"
        )
    if len(ids) == 1:
        distances = np.array([GEOD.inv(lon, lat, *xy[ids[0]])[2] / 1000])
    else:
        distances = (
            GEOD.inv(
                np.full(len(ids), lon), np.full(len(ids), lat), xy[ids, 0], xy[ids, 1]
            )[2]
            / 1000
        )
    if max_distance_km is not None:
        close = distances <= max_distance_km
        ids, distances = ids[close], distances[close]
        if not len(ids):
            raise ValueError(
                f"No wet node within {max_distance_km:g} km; geographic review is required"
            )
    # Sort all wet candidates first; do not silently jump over an isolated bay.
    order = np.lexsort((ids, xy[ids, 1], xy[ids, 0], np.round(distances, 9)))
    selected_position = int(order[0])
    if not shapely.covers(network.land, shapely.Point(lon, lat)):
        # A water point must not attach across a peninsula or island. Test
        # distance-ordered candidates without preferring a different component.
        for start in range(len(order)):
            positions = order[start : start + 1]
            segments = shapely.linestrings(
                np.stack(
                    (np.tile([lon, lat], (len(positions), 1)), xy[ids[positions]]),
                    axis=1,
                )
            )
            visible = ~crosses_land(segments, network.land)
            if visible.any():
                selected_position = int(positions[np.flatnonzero(visible)[0]])
                break
        else:
            raise ValueError(
                "No water-only attachment exists inside the geographic bounds; review the local waterway"
            )
    selected = int(ids[selected_position])
    if network.node_components[selected] != network.main_component:
        raise ValueError(
            f"Nearest wet node for ({lon}, {lat}) is disconnected; review the local waterway instead of moving the point to another bay"
        )
    return selected, float(distances[selected_position])


def map_points(frame, network, config, kind):
    if kind not in {"source", "observation"}:
        raise ValueError("Routing point kind must be source or observation")
    x, y = ("Grid_Lon", "Grid_Lat") if kind == "source" else ("Longitude", "Latitude")
    keys = coordinate_keys(frame, x, y)
    if kind == "source" and len(set(keys)) != len(keys):
        raise ValueError(
            "The merged source inventory must have unique original coordinates; no duplicate sources are silently merged"
        )
    if kind == "observation":
        validate_rows(frame)
    constraints = config.get(f"{kind}_constraints", [])
    records = []
    coordinate_index = {
        tuple(np.round(xy, 9)): i for i, xy in enumerate(network.coordinates)
    }
    for position, ((lon, lat), (_, original)) in enumerate(zip(keys, frame.iterrows())):
        domain = config["bounds"]
        if not (domain[0] <= lon <= domain[2] and domain[1] <= lat <= domain[3]):
            raise ValueError(
                f"{kind} coordinate ({lon}, {lat}) lies outside the routing domain"
            )
        original_on_land = bool(shapely.covers(network.land, shapely.Point(lon, lat)))
        matching = [
            item
            for item in constraints
            if (
                kind == "source"
                and np.allclose(
                    item["original_coordinates"], [lon, lat], rtol=0, atol=1e-9
                )
            )
            or (
                kind == "observation"
                and item["row_index"] == int(original["row_index"])
            )
        ]
        if len(matching) > 1:
            raise ValueError("Duplicate geographic constraints for a routing point")
        constraint = matching[0] if matching else {}
        bounds = constraint.get("bounds")
        inside_patch = any(
            p["bounds"][0] <= lon <= p["bounds"][2]
            and p["bounds"][1] <= lat <= p["bounds"][3]
            for p in config.get("refinement_patches", [])
        )
        base_xy = tuple(
            np.round(
                np.rint(np.array([lon, lat]) / config["base_resolution"])
                * config["base_resolution"],
                9,
            )
        )
        retained = coordinate_index.get(base_xy)
        if retained is not None and not original_on_land:
            connector = shapely.LineString([(lon, lat), network.coordinates[retained]])
            if crosses_land([connector], network.land)[0]:
                retained = None
        if (
            retained is not None
            and network.node_components[retained] == network.main_component
            and not inside_patch
            and not constraint
        ):
            node = retained
            distance = GEOD.inv(lon, lat, *network.coordinates[node])[2] / 1000
            method = "retained_valid_base_node"
        else:
            node, distance = nearest_node(
                lon,
                lat,
                network,
                bounds,
                None if constraint else config.get("displacement_review_km", 20),
            )
            method = "reviewed_outlet_bounds" if bounds else "nearest_wet_node"
        if distance > config.get("displacement_review_km", 20) and not constraint.get(
            "reviewed", False
        ):
            raise ValueError(
                f"{kind} at {lon}, {lat}: displacement {distance:.2f} km requires recorded geographic review"
            )
        if constraint and not constraint.get("reviewed", False):
            raise ValueError(
                f"Unresolved geographic constraint for {kind} at {lon}, {lat}"
            )
        records.append(
            {
                "source_id": f"source_{position:03d}",
                "source_position": position,
                "Grid_Lon": float(original[x]),
                "Grid_Lat": float(original[y]),
                "node_id": node,
                "routing_lon": network.coordinates[node, 0],
                "routing_lat": network.coordinates[node, 1],
                "displacement_km": distance,
                "assignment_method": method,
                "original_on_land_mask": original_on_land,
                "water_connector_checked": not original_on_land,
                "review_basis": constraint.get("review_basis", ""),
                **(
                    {
                        "row_index": int(original["row_index"]),
                        "Longitude": float(original[x]),
                        "Latitude": float(original[y]),
                    }
                    if kind != "source"
                    else {}
                ),
            }
        )
    result = pd.DataFrame(records)
    if kind != "source":
        result = result.drop(
            columns=["source_id", "source_position", "Grid_Lon", "Grid_Lat"]
        )
    return result


def validate_rows(frame):
    if "row_index" not in frame:
        raise ValueError("Observation row_index is required")
    ids = pd.to_numeric(frame.row_index, errors="raise").to_numpy(float)
    if (
        not np.isfinite(ids).all()
        or (ids < 0).any()
        or not np.equal(ids, np.floor(ids)).all()
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Observation row_index must be unique nonnegative integers")


@dataclass
class RoutingBundle:
    manifest: dict
    sources: pd.DataFrame
    observations: pd.DataFrame
    distances: np.ndarray


def load_routing_bundle(directory):
    directory = Path(directory)
    manifest = json.loads(
        (directory / "Routing_Manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != SCHEMA or not manifest.get(
        "geography_review_complete"
    ):
        raise ValueError("Routing artifact is not a geography-reviewed release")
    required = {
        "Source_Ocean_Node_Mapping.csv",
        "Observation_Ocean_Node_Mapping.csv",
        "Source_Observation_Distances.npz",
    }
    if not required.issubset(manifest.get("products", {})):
        raise ValueError("Routing manifest lacks required product hashes")
    for name, digest in manifest["products"].items():
        path = directory / name
        if path.parent.resolve() != directory.resolve() or sha256(path) != digest:
            raise ValueError(f"Routing product hash mismatch: {name}")
    sources = pd.read_csv(directory / "Source_Ocean_Node_Mapping.csv")
    observations = pd.read_csv(directory / "Observation_Ocean_Node_Mapping.csv")
    keys = coordinate_keys(sources, "Grid_Lon", "Grid_Lat")
    if (
        len(set(keys)) != len(keys)
        or sources.source_id.isna().any()
        or not sources.source_id.is_unique
    ):
        raise ValueError("Frozen source coordinates and identifiers must be unique")
    validate_rows(observations)
    with np.load(
        directory / "Source_Observation_Distances.npz", allow_pickle=False
    ) as archive:
        distances = archive["distance_km"]
        if not np.array_equal(
            archive["row_index"], observations.row_index.to_numpy()
        ) or not np.array_equal(archive["source_id"], sources.source_id.to_numpy(str)):
            raise ValueError("Routing distance axes do not match the mapping tables")
    if (
        distances.shape != (len(sources), len(observations))
        or not np.isfinite(distances).all()
        or (distances < 0).any()
    ):
        raise ValueError(
            "Every source-observation routing distance must be finite and nonnegative"
        )
    return RoutingBundle(manifest, sources, observations, distances)


def compute_routed_features(frame, source_load, routing_directory):
    bundle = load_routing_bundle(routing_directory)
    validate_rows(frame)
    expected = coordinate_keys(bundle.sources, "Grid_Lon", "Grid_Lat")
    supplied = coordinate_keys(source_load, "Grid_Lon", "Grid_Lat")
    if (
        len(set(supplied)) != len(supplied)
        or len(supplied) != len(expected)
        or set(supplied) != set(expected)
    ):
        raise ValueError(
            "Scenario source set or original source coordinates differ from the frozen routing inventory"
        )
    lookup = {key: i for i, key in enumerate(supplied)}
    source = source_load.iloc[[lookup[key] for key in expected]]
    values = (
        source[["Load_Total_Legacy_Kg", *RATIOS]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Source loads and ratios must be finite and nonnegative")
    observations = bundle.observations.set_index("row_index", drop=False)
    if not set(frame.row_index).issubset(observations.index):
        raise ValueError("Observation IDs are absent from the frozen routing artifact")
    selected = observations.loc[frame.row_index]
    if not np.allclose(
        frame[["Longitude", "Latitude"]].to_numpy(float),
        selected[["Longitude", "Latitude"]].to_numpy(float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError(
            "Observation coordinates differ from the frozen routing artifact"
        )
    column_lookup = {int(row): i for i, row in enumerate(bundle.observations.row_index)}
    distance = bundle.distances[
        :, [column_lookup[int(row)] for row in frame.row_index]
    ].T
    weights = values[:, 0][None, :] / (distance + 1.0)
    total = weights.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("All-zero source forcing leaves mixture ratios undefined")
    mixed = weights @ values[:, 1:] / total[:, None]
    result = frame.copy()
    result[FEATURES] = np.column_stack((total, mixed))
    return result


def prepare_routing(data_path, source_path, boundary_path, config_path, output_dir):
    paths = {
        "data": Path(data_path),
        "source_load": Path(source_path),
        "routing_config": Path(config_path),
        **{
            f"boundary_{p.suffix.lstrip('.')}": p
            for p in sorted(
                Path(boundary_path).parent.glob(f"{Path(boundary_path).stem}.*")
            )
        },
    }
    provenance = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not config.get("geography_review_complete"):
        raise ValueError(
            "Complete the geographic review before producing model-ready routing"
        )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty routing output directory")
    data = (
        pd.read_excel(data_path)
        if Path(data_path).suffix.lower() != ".csv"
        else pd.read_csv(data_path)
    )
    if "row_index" not in data:
        data["row_index"] = np.arange(len(data))
    validate_rows(data)
    source = pd.read_csv(source_path)
    source_values = (
        source[["Load_Total_Legacy_Kg", *RATIOS]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    if (
        not np.isfinite(source_values).all()
        or (source_values < 0).any()
        or source_values[:, 0].sum() <= 0
    ):
        raise ValueError(
            "Source inventory must contain finite nonnegative loads and ratios, with positive total load"
        )
    print("Building the water-only coastal network...", flush=True)
    network = build_routing_network(boundary_path, config)
    print(
        f"Network complete: {len(network.coordinates)} nodes. Mapping sources and observations...",
        flush=True,
    )
    source_mapping = map_points(source, network, config, "source")
    observation_mapping = map_points(data, network, config, "observation")
    mapped_load = source_mapping.merge(
        source[["Grid_Lon", "Grid_Lat", "Load_Total_Legacy_Kg"]],
        on=["Grid_Lon", "Grid_Lat"],
        how="left",
        validate="one_to_one",
    )
    if mapped_load.Load_Total_Legacy_Kg.isna().any() or len(mapped_load) != len(source):
        raise ValueError(
            "Routing assignments do not preserve the complete source inventory"
        )
    source_mapping["baseline_load_kg"] = mapped_load.Load_Total_Legacy_Kg.to_numpy()
    distance = dijkstra(
        network.graph, directed=False, indices=source_mapping.node_id.to_numpy()
    )[:, observation_mapping.node_id.to_numpy()]
    if not np.isfinite(distance).all():
        raise ValueError(
            "Disconnected source-observation pair; no source may be silently discarded"
        )
    for name, record in provenance.items():
        if sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"Input changed during routing: {name}")
    output.mkdir(parents=True, exist_ok=True)
    source_mapping.to_csv(output / "Source_Ocean_Node_Mapping.csv", index=False)
    observation_mapping.to_csv(
        output / "Observation_Ocean_Node_Mapping.csv", index=False
    )
    pd.DataFrame(network.coordinates, columns=["Longitude", "Latitude"]).to_csv(
        output / "Routing_Nodes.csv", index_label="node_id"
    )
    save_npz(output / "Routing_Graph.npz", network.graph)
    np.savez_compressed(
        output / "Source_Observation_Distances.npz",
        distance_km=distance,
        row_index=observation_mapping.row_index.to_numpy(),
        source_id=source_mapping.source_id.to_numpy(str),
    )
    manifest = {
        "schema": SCHEMA,
        "geography_review_complete": True,
        "input_provenance": provenance,
        "config": config,
        "code_sha256": sha256(__file__),
        "network_audit": network.audit,
        "sources": len(source_mapping),
        "observations": len(observation_mapping),
        "finite_pairs": int(np.isfinite(distance).sum()),
        "input_source_load_kg": float(source.Load_Total_Legacy_Kg.sum()),
        "mapped_source_load_kg": float(source_mapping.baseline_load_kg.sum()),
        "distance_definition": "Haversine-weighted shortest water-only grid path; anchor displacement excluded",
        "products": {
            p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
        },
    }
    (output / "Routing_Manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    load_routing_bundle(output)
    print(
        json.dumps(
            {
                "sources": len(source_mapping),
                "observations": len(observation_mapping),
                "network": network.audit,
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "source-load", "boundary", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare_routing(
        args.data, args.source_load, args.boundary, args.config, args.output_dir
    )


if __name__ == "__main__":
    main()
