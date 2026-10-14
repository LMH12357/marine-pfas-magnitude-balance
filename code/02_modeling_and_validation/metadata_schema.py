from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

SPATIAL_BLOCK_DEGREES = 2.0
BOUNDARY_QA_DISTANCE_KM = 15.0
COASTAL_SNAP_TOLERANCE_KM = 1.0
CANONICAL_SEA_REGIONS = ("Bohai Sea", "Yellow Sea", "East China Sea", "South China Sea")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEA_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "geospatial"
    / "china_marginal_sea_regions.geojson"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampling_group_source(frame: pd.DataFrame) -> str:
    present = [
        column
        for column in ("Sampling_Group_ID", "Sampling_Group", "Study")
        if column in frame.columns
    ]
    if not present:
        raise ValueError(
            "A sampling-group column is required: Sampling_Group_ID (preferred), Sampling_Group, or legacy Study"
        )
    return present[0]


def normalize_sampling_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers without renumbering groups or requiring a full dataset.

    A filtered table can contain gaps and a resampled table can reorder rows.
    Neither operation changes the identity of a sampling group.
    """
    result = frame.copy()
    source = _sampling_group_source(result)
    raw = result[source].ffill()
    if raw.isna().any():
        raise ValueError(
            "Sampling-group identifiers cannot be reconstructed by forward filling"
        )

    if (
        source == "Sampling_Group"
        and raw.astype(str).str.fullmatch(r"Sampling_Group_\d+").all()
    ):
        identifiers = pd.to_numeric(
            raw.astype(str).str.extract(r"(\d+)$", expand=False), errors="raise"
        )
    elif source == "Study" and raw.astype(str).str.fullmatch(r"Study_\d+").all():
        identifiers = pd.to_numeric(
            raw.astype(str).str.extract(r"(\d+)$", expand=False), errors="raise"
        )
    else:
        identifiers = pd.to_numeric(raw, errors="raise")
    if (
        not np.isfinite(identifiers.to_numpy(dtype=float)).all()
        or not np.equal(identifiers, np.floor(identifiers)).all()
    ):
        raise ValueError("Sampling-group identifiers must be finite integers")
    identifiers = identifiers.astype(int)
    if (identifiers < 1).any():
        raise ValueError("Sampling-group identifiers must be positive")

    canonical_labels = identifiers.map(lambda value: f"Sampling_Group_{value:02d}")
    if "Sampling_Group" in result.columns and source != "Sampling_Group":
        supplied_labels = result["Sampling_Group"].ffill().astype(str)
        mismatch = supplied_labels.to_numpy() != canonical_labels.astype(str).to_numpy()
        if mismatch.any():
            raise ValueError(
                f"Sampling_Group conflicts with Sampling_Group_ID in {int(mismatch.sum())} rows"
            )
    result["Sampling_Group_ID"] = identifiers.to_numpy()
    result["Sampling_Group"] = canonical_labels.to_numpy()
    for obsolete in ("Study", "District"):
        if obsolete in result.columns:
            result = result.drop(columns=obsolete)
    return result


def assign_canonical_sea_regions(
    frame: pd.DataFrame,
    sea_reference: Path = DEFAULT_SEA_REFERENCE,
    boundary_qa_distance_km: float = BOUNDARY_QA_DISTANCE_KM,
    coastal_snap_tolerance_km: float = COASTAL_SNAP_TOLERANCE_KM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Longitude", "Latitude"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    if not sea_reference.exists():
        raise FileNotFoundError(f"Sea-region reference not found: {sea_reference}")

    result = frame.copy()
    longitude = pd.to_numeric(result["Longitude"], errors="raise")
    latitude = pd.to_numeric(result["Latitude"], errors="raise")
    if not np.isfinite(np.column_stack([longitude, latitude])).all():
        raise ValueError("Longitude and latitude must be finite")
    if not longitude.between(-180, 180).all() or not latitude.between(-90, 90).all():
        raise ValueError("Longitude and latitude are outside geographic bounds")
    if result.empty:
        raise ValueError("A nonempty observation table is required")
    points = gpd.GeoDataFrame(
        {"row_index": result.index.to_numpy()},
        geometry=gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    )
    seas = gpd.read_file(sea_reference).to_crs("EPSG:4326")
    if set(seas["Sea_Region"]) != set(CANONICAL_SEA_REGIONS):
        raise ValueError(
            "Sea-region reference does not contain the four canonical regions"
        )

    centre_lon, centre_lat = float(longitude.mean()), float(latitude.mean())
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre_lat:.8f} +lon_0={centre_lon:.8f} +datum=WGS84 +units=m +no_defs"
    )
    projected_points = points.to_crs(local_crs)
    projected_seas = seas.to_crs(local_crs).set_index("Sea_Region")

    assignments = []
    assignment_methods = []
    snap_distances_km = []
    for point, projected_point in zip(points.geometry, projected_points.geometry):
        matches = seas.loc[seas.geometry.covers(point), "Sea_Region"].tolist()
        if len(matches) == 1:
            assignments.append(matches)
            assignment_methods.append("polygon_cover")
            snap_distances_km.append(0.0)
            continue
        if not matches:
            distances = projected_seas.geometry.distance(projected_point).sort_values()
            if float(distances.iloc[0]) <= float(coastal_snap_tolerance_km) * 1000.0:
                assignments.append([str(distances.index[0])])
                assignment_methods.append("nearest_polygon_within_coastal_tolerance")
                snap_distances_km.append(float(distances.iloc[0] / 1000.0))
                continue
        assignments.append(matches)
        assignment_methods.append("unresolved")
        snap_distances_km.append(np.nan)
    ambiguous = [
        position for position, matches in enumerate(assignments) if len(matches) > 1
    ]
    unassigned = [
        position for position, matches in enumerate(assignments) if len(matches) == 0
    ]
    if ambiguous or unassigned:
        details = {
            "ambiguous_row_indices": result.index[
                np.asarray(ambiguous, dtype=int)
            ].tolist(),
            "unassigned_row_indices": result.index[
                np.asarray(unassigned, dtype=int)
            ].tolist(),
        }
        raise ValueError(
            f"Canonical sea assignment was not unique for every point: {details}"
        )
    result["Sea_Region"] = [matches[0] for matches in assignments]
    result["Sea_Assignment_Method"] = assignment_methods
    result["Sea_Coastal_Snap_Distance_km"] = snap_distances_km
    distances_km = []
    for point, sea_region in zip(projected_points.geometry, result["Sea_Region"]):
        competing = projected_seas.drop(index=sea_region).geometry
        distances_km.append(float(competing.distance(point).min() / 1000.0))
    result["Sea_Boundary_Distance_km"] = distances_km
    result["Sea_Boundary_QA_Flag"] = result["Sea_Boundary_Distance_km"] <= float(
        boundary_qa_distance_km
    )

    audit = result[
        [
            "Sampling_Group_ID",
            "Sampling_Group",
            "Longitude",
            "Latitude",
            "Sea_Region",
            "Sea_Assignment_Method",
            "Sea_Coastal_Snap_Distance_km",
            "Sea_Boundary_Distance_km",
            "Sea_Boundary_QA_Flag",
        ]
    ].copy()
    audit.insert(0, "row_index", result.index)
    audit["Sea_Reference_File"] = sea_reference.name
    audit["Sea_Reference_SHA256"] = sha256(sea_reference)
    audit["Boundary_QA_Threshold_km"] = float(boundary_qa_distance_km)
    audit["Coastal_Snap_Tolerance_km"] = float(coastal_snap_tolerance_km)
    return result, audit


def normalize_metadata(
    frame: pd.DataFrame,
    sea_reference: Path = DEFAULT_SEA_REFERENCE,
    boundary_qa_distance_km: float = BOUNDARY_QA_DISTANCE_KM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = normalize_sampling_groups(frame)
    if "Sea_Region" in result.columns:
        unknown = sorted(
            set(result["Sea_Region"].dropna()) - set(CANONICAL_SEA_REGIONS)
        )
        if unknown or result["Sea_Region"].isna().any():
            raise ValueError(f"Invalid canonical Sea_Region values: {unknown}")
        # Recompute from coordinates and compare, rather than trusting a stale export.
        supplied = result["Sea_Region"].copy()
        result = result.drop(columns="Sea_Region")
        result, audit = assign_canonical_sea_regions(
            result, sea_reference, boundary_qa_distance_km
        )
        mismatch = (
            supplied.astype(str).to_numpy()
            != result["Sea_Region"].astype(str).to_numpy()
        )
        if mismatch.any():
            raise ValueError(
                f"Supplied Sea_Region conflicts with coordinates in {int(mismatch.sum())} rows"
            )
    else:
        result, audit = assign_canonical_sea_regions(
            result, sea_reference, boundary_qa_distance_km
        )
    if "Samplingtime" in result.columns:
        result["SamplingYear"] = pd.to_numeric(
            result["Samplingtime"].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        ).astype("Int64")
    elif "SamplingYear" not in result.columns:
        raise ValueError("Samplingtime or SamplingYear is required")
    result["Spatial_Block"] = (
        np.floor(pd.to_numeric(result["Longitude"]) / SPATIAL_BLOCK_DEGREES)
        .astype(int)
        .astype(str)
        + "_"
        + np.floor(pd.to_numeric(result["Latitude"]) / SPATIAL_BLOCK_DEGREES)
        .astype(int)
        .astype(str)
    )
    return result, audit
