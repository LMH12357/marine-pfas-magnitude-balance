"""Shared observation, extraction, interpolation and provenance utilities."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import logging
import math
import re

import numpy as np
import pandas as pd
import xarray as xr
from pykrige.ok import OrdinaryKriging
from sklearn.neighbors import BallTree, KNeighborsRegressor

LOG = logging.getLogger(__name__)
FIELDS = (
    "Temperature",
    "Salinity",
    "pH",
    "Chla",
    "POC",
    "SPM",
    "DO",
    "TOC",
    "Hydro_Bottom_U",
    "Hydro_Bottom_V",
    "Hydro_MLD",
)
ALIASES = {
    "Temperature (°C)": "Temperature",
    "Salinity (‰)": "Salinity",
    "POC (mg/m3)": "POC",
    "SPM (mg/L)": "SPM",
    "TOC (%)": "TOC",
    "Chl-a": "Chla",
}
SOURCE_ALIASES = {
    "L1_CMEMS_Reanalysis": "CMEMS Extraction",
    "L1_CMEMS_Bilinear_Downscaled": "CMEMS Extraction",
    "CMEMS_Bilinear_Downscaled": "CMEMS Extraction",
    "Direct_Extraction": "CMEMS Extraction",
    "NASA_Bilinear_Downscaled": "NASA Extraction",
    "Climatological_Mean": "NASA Climatology",
    "Rrs_Bilinear_Inverted": "Rrs Estimate",
    "Spatial_KNN": "Spatial KNN",
    "Regional_Median": "Regional Median",
    "Local_Kriging": "Spatial Kriging",
    "Kriging_Interpolation": "Spatial Kriging",
    "L2_Isolated_Kriging": "Spatial Kriging",
    "Extended_Kriging": "TOC Reference Kriging",
    "L3_Climatological_Fallback": "CMEMS Climatology",
    "Climatological_Bilinear_Fallback": "CMEMS Climatology",
    "Neighbor_Expanded": "CMEMS Neighbor",
    "L1.5_Estuarine_Temporal_Relaxation": "Estuarine Median",
}
SOURCES = set(SOURCE_ALIASES.values()) | {
    "Original Data",
    "Unfilled",
    "CMEMS Climatology Neighbor",
}
BOUNDS = {
    "Bohai": (117, 122.5, 37, 42),
    "Yellow": (118, 127, 31, 37),
    "East": (116, 131, 23, 31),
    "South": (105, 121, 3, 23),
}
EARTH_RADIUS_KM = 6371.01
SEARCH_RADIUS_RAD = 0.5 * 111.32 / EARTH_RADIUS_KM


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def canonical_source(value, numeric_present):
    if not numeric_present:
        return "Unfilled"
    if pd.isna(value) or not str(value).strip():
        return "Original Data"
    value = str(value).strip()
    if value.startswith("Original Data") or value == "Original":
        return "Original Data"
    if value.startswith("Climatology_Month_"):
        return "CMEMS Climatology"
    value = SOURCE_ALIASES.get(value, value)
    if value not in SOURCES or value == "Unfilled":
        raise ValueError(f"Nonmissing value has an unrecognized source: {value!r}")
    return value


def region(lon, lat, toc=False):
    for name, (lo, hi, bottom, top) in BOUNDS.items():
        if toc and name in ("Yellow", "East"):
            lo += 1
        if lo <= lon <= hi and bottom <= lat <= top:
            return name
    return "OpenOcean"


class Observations:
    def __init__(self, config):
        self.config = config
        self.path = Path(config["workbook"])
        self.input_hash = sha256(self.path)
        self.frame = pd.read_excel(
            self.path, sheet_name=config.get("sheet", "Spatiotemporal identification")
        )
        self.frame.columns = [str(c).strip() for c in self.frame.columns]
        for old, new in ALIASES.items():
            if old in self.frame and new in self.frame:
                raise ValueError(f"Duplicate aliases: {old}, {new}")
        self.frame.rename(columns=ALIASES, inplace=True)
        self.frame = self.frame.loc[self.frame["Number"].notna()].copy()
        self.frame["_excel_row"] = self.frame.index + 2
        self.frame["_date"] = pd.to_datetime(
            self.frame["Samplingtime"].astype(str).str.replace(".", "-", regex=False),
            format="mixed",
            errors="raise",
        )
        self.frame["_month"] = self.frame["_date"].dt.month
        self.frame["_year"] = self.frame["_date"].dt.year
        self.frame["_ym"] = self.frame["_date"].dt.to_period("M")
        for field in ("Latitude", "Longitude"):
            self.frame[field] = pd.to_numeric(self.frame[field], errors="raise")
        if not np.isfinite(self.frame[["Latitude", "Longitude"]].to_numpy()).all():
            raise ValueError("Nonfinite coordinates")
        self.frame["_region"] = [
            region(lon, lat)
            for lon, lat in self.frame[["Longitude", "Latitude"]].itertuples(
                index=False, name=None
            )
        ]
        self.before = self.frame.copy(deep=True)
        self.sources = pd.DataFrame(index=self.frame.index)
        self.initial_sources = {}
        self.events = {}
        self.products = {}
        self.completed = []
        for field in FIELDS:
            if field not in self.frame:
                raise ValueError(f"Required field missing: {field}")
            self.frame[field] = pd.to_numeric(self.frame[field], errors="raise")
            source_col = (
                "Hydro_Source" if field.startswith("Hydro_") else field + "_Source"
            )
            old_sources = self.frame.get(
                source_col, pd.Series(None, index=self.frame.index, dtype=object)
            )
            self.initial_sources[field] = old_sources.to_dict()
            self.sources[field] = [
                canonical_source(s, pd.notna(v))
                for s, v in zip(old_sources, self.frame[field])
            ]

    def missing(self, field):
        return self.frame.index[self.frame[field].isna()]

    def set(self, i, field, value, source, **detail):
        if not np.isfinite(value):
            return False
        if pd.notna(self.before.at[i, field]):
            raise ValueError(f"Attempt to overwrite existing {field} at row {i+2}")
        if source not in SOURCES:
            raise ValueError(source)
        self.frame.at[i, field] = float(value)
        self.sources.at[i, field] = source
        self.events[(i, field)] = detail
        return True

    def product(self, path):
        path = Path(path).resolve()
        self.products[str(path)] = {
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }

    def rows(self):
        result = []
        for field in FIELDS:
            for i, row in self.frame.iterrows():
                v, old = row[field], self.before.at[i, field]
                result.append(
                    {
                        "excel_row": int(row["_excel_row"]),
                        "sample": str(row["Number"]),
                        "variable": field,
                        "value": float(v) if pd.notna(v) else None,
                        "previous_value": float(old) if pd.notna(old) else None,
                        "source": self.sources.at[i, field],
                        "previous_source": (
                            None
                            if pd.isna(self.initial_sources[field][i])
                            else str(self.initial_sources[field][i])
                        ),
                        "sampling_date": str(self.before.at[i, "Samplingtime"]),
                        "detail": self.events.get((i, field), {}),
                    }
                )
        return result

    def save(self, name="environment_results"):
        output = Path(self.config["output_directory"])
        records = self.rows()
        dump_json(
            output / (name + ".json"),
            {
                "input_path": str(self.path),
                "input_sha256": self.input_hash,
                "completed": self.completed,
                "products": self.products,
                "records": records,
            },
        )
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "detail"} for r in records]
        ).to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig")
        summary = {
            f: {
                "original_nonmissing": int(self.before[f].notna().sum()),
                "current_nonmissing": int(self.frame[f].notna().sum()),
                "sources": {
                    str(k): int(v) for k, v in self.sources[f].value_counts().items()
                },
            }
            for f in FIELDS
        }
        dump_json(output / "environment_summary.json", summary)
        return summary


def ordered_slice(values, low, high):
    return (
        slice(low, high) if float(values[0]) < float(values[-1]) else slice(high, low)
    )


def normalize_grid(da):
    rename = {
        k: v for k, v in [("lat", "latitude"), ("lon", "longitude")] if k in da.dims
    }
    return da.rename(rename)


def pad_and_sample(da, frame, method="linear", padding=1):
    da = normalize_grid(da)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    for dim in list(da.dims):
        if dim not in ("latitude", "longitude"):
            if da.sizes[dim] != 1:
                raise ValueError(f"Unresolved dimension {dim}")
            da = da.isel({dim: 0})
    lo, hi = frame.Longitude.min() - 2, frame.Longitude.max() + 2
    bottom, top = frame.Latitude.min() - 2, frame.Latitude.max() + 2
    da = da.sel(
        longitude=ordered_slice(da.longitude.values, lo, hi),
        latitude=ordered_slice(da.latitude.values, bottom, top),
    ).load()
    if not da.sizes["latitude"] or not da.sizes["longitude"]:
        return dict.fromkeys(frame.index, np.nan)
    da = da.sortby("latitude").sortby("longitude")
    if padding:
        da = (
            da.ffill("longitude", limit=padding)
            .bfill("longitude", limit=padding)
            .ffill("latitude", limit=padding)
            .bfill("latitude", limit=padding)
        )
    y = xr.DataArray(frame.Latitude.to_numpy(), dims="station")
    x = xr.DataArray(frame.Longitude.to_numpy(), dims="station")
    if method == "nearest":
        values = da.sel(latitude=y, longitude=x, method="nearest").values
        inside = (
            (y.values >= da.latitude.values[0])
            & (y.values <= da.latitude.values[-1])
            & (x.values >= da.longitude.values[0])
            & (x.values <= da.longitude.values[-1])
        )
        values = np.where(inside, values, np.nan)
    else:
        values = da.interp(latitude=y, longitude=x, method="linear").values
    return dict(zip(frame.index, np.asarray(values, dtype=float)))


class MonthlyProduct:
    def __init__(self, paths, variable, units, context):
        if not paths:
            raise FileNotFoundError(f"No product files for {variable}")
        self.variable, self.context = variable, context
        self.datasets = []
        self.periods = {}
        self.units = units
        for path in sorted(paths):
            ds = xr.open_dataset(path)
            self.datasets.append(ds)
            context.product(path)
            if variable not in ds:
                raise ValueError(f"{path}: missing {variable}")
            actual = ds[variable].attrs.get("units")
            if actual not in units:
                raise ValueError(f"{path}: unsupported {variable} units {actual!r}")
            for position, t in enumerate(pd.to_datetime(ds.time.values)):
                key = (t.year, t.month)
                if key in self.periods:
                    old_ds, old_pos, _ = self.periods[key]
                    if (
                        not old_ds[variable]
                        .isel(time=old_pos)
                        .equals(ds[variable].isel(time=position))
                    ):
                        raise ValueError(
                            f"Conflicting duplicate product month {key}: {variable}"
                        )
                    continue
                self.periods[key] = (ds, position, str(path))

    def sample(self, frame, method="linear", padding=1, climatology=False):
        values, metadata = {}, {}
        for (year, month), group in frame.groupby(["_year", "_month"]):
            keys = (
                sorted(k for k in self.periods if k[1] == month)
                if climatology
                else [(int(year), int(month))]
            )
            valid = [k for k in keys if k in self.periods]
            if not valid:
                values.update(dict.fromkeys(group.index, np.nan))
                continue
            if climatology:
                arrays = [
                    self.periods[k][0][self.variable].isel(
                        time=self.periods[k][1], drop=True
                    )
                    for k in valid
                ]
                da = xr.concat(arrays, dim="climatology_year").mean("climatology_year")
            else:
                ds, pos, _ = self.periods[valid[0]]
                da = ds[self.variable].isel(time=pos, drop=True)
            values.update(pad_and_sample(da, group, method, padding))
            detail = {
                "files": sorted({self.periods[k][2] for k in valid}),
                "product_variable": self.variable,
                "months": [f"{y:04d}-{m:02d}" for y, m in valid],
                "spatial_method": method,
                "padding_pixels": padding,
                "depth_scope": "surface_product",
                "time_method": (
                    "same_calendar_month_climatology"
                    if climatology
                    else "same_year_and_calendar_month"
                ),
            }
            metadata.update({i: detail for i in group.index})
        return values, metadata

    def close(self):
        for ds in self.datasets:
            ds.close()


def nasa_index(directory):
    result = {}
    for path in sorted(Path(directory).glob("*.nc")):
        match = re.search(r"\.(\d{4})(\d{2})\d{2}_\d{8}\.", path.name)
        if match:
            key = tuple(map(int, match.groups()))
        else:
            with xr.open_dataset(path) as ds:
                t = pd.Timestamp(ds.attrs["time_coverage_start"])
                key = (t.year, t.month)
        if key in result:
            raise ValueError(f"Multiple NASA products for {key} in {directory}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"No NASA monthly products in {directory}")
    return result


def nasa_sample(path, frame, variable, ctx):
    if path is None:
        return dict.fromkeys(frame.index, np.nan)
    ctx.product(path)
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"{path}: missing exact band/variable {variable}")
        expected = {"chlor_a": {"mg m^-3", "mg m-3"}, "poc": {"mg m^-3", "mg m-3"}}
        allowed = expected.get(variable, {"sr^-1", "sr-1"})
        if ds[variable].attrs.get("units") not in allowed:
            raise ValueError(f"Unexpected units for {variable}")
        return pad_and_sample(ds[variable], frame, padding=1)


def local_interpolation(
    ctx, field, method, temporal=False, external=None, toc=False, excluded=()
):
    frame = ctx.frame
    pool = (
        (external if external is not None else frame)
        .dropna(subset=[field, "Latitude", "Longitude"])
        .copy()
    )
    if pool.empty:
        return
    for i in list(ctx.missing(field)):
        if i in excluded:
            continue
        row = frame.loc[i]
        candidates = pool.loc[pool["_ym"] == row["_ym"]] if temporal else pool
        # Co-located records contribute one spatial location to geostatistics.
        refs = candidates.groupby(["Latitude", "Longitude"], as_index=False)[
            field
        ].mean()
        if len(refs) < 3:
            continue
        xy = np.radians(refs[["Latitude", "Longitude"]].to_numpy())
        query = np.radians([[row.Latitude, row.Longitude]])
        idx = BallTree(xy, metric="haversine").query_radius(query, r=SEARCH_RADIUS_RAD)[
            0
        ]
        if len(idx) < 3:
            continue
        vals = refs[field].to_numpy()[idx]
        try:
            if np.ptp(vals) < 1e-12:
                value = float(vals[0])
            elif method == "knn":
                model = KNeighborsRegressor(
                    n_neighbors=min(5, len(idx)),
                    weights="distance",
                    algorithm="ball_tree",
                    metric="haversine",
                )
                value = float(model.fit(xy[idx], vals).predict(query)[0])
            else:
                coords = refs[["Longitude", "Latitude"]].to_numpy()[idx]
                model = OrdinaryKriging(
                    coords[:, 0],
                    coords[:, 1],
                    vals,
                    variogram_model="linear",
                    coordinates_type="geographic",
                    verbose=False,
                )
                pred, _ = model.execute("points", [row.Longitude], [row.Latitude])
                value = float(pred[0])
        except (ValueError, np.linalg.LinAlgError) as exc:
            LOG.warning(
                "%s row %s: spatial interpolation unavailable: %s", field, i + 2, exc
            )
            continue
        low, high = {"Temperature": (-2, 45), "Salinity": (0, 42), "pH": (6, 9)}.get(
            field, (0, float("inf"))
        )
        if np.isfinite(value) and low <= value <= high and (not toc or value > 0):
            label = (
                "TOC Reference Kriging"
                if external is not None
                else "Spatial KNN" if method == "knn" else "Spatial Kriging"
            )
            ctx.set(
                i,
                field,
                value,
                label,
                donor_coordinates=refs.loc[idx, ["Longitude", "Latitude"]]
                .to_numpy()
                .tolist(),
                donor_values=vals.tolist(),
                requires_fold_refit=external is None,
                temporal_scope=(
                    "same_year_month" if temporal else "available_observations"
                ),
                radius_km=0.5 * 111.32,
                reference_file=(
                    str(ctx.config["toc_reference"]) if external is not None else None
                ),
            )


def median_fallback(ctx, field, monthly=False, toc=False):
    frame = ctx.frame
    pool = frame.loc[frame[field].notna()].copy()
    if toc:
        pool["_region"] = [
            region(lon, lat, toc=True)
            for lon, lat in pool[["Longitude", "Latitude"]].itertuples(
                index=False, name=None
            )
        ]
    for i in ctx.missing(field):
        row = frame.loc[i]
        area = region(row.Longitude, row.Latitude, toc=True) if toc else row["_region"]
        donors = (
            pool.loc[(pool["_region"] == area) & (pool["_month"] == row["_month"])]
            if monthly
            else pool.iloc[:0]
        )
        scope = "region_month"
        if donors.empty:
            donors = pool.loc[pool["_region"] == area]
            scope = "region"
        if donors.empty:
            donors = pool
            scope = "available_pool"
        if donors.empty:
            continue
        value = float(donors[field].median())
        if np.isfinite(value) and value >= 0:
            ctx.set(
                i,
                field,
                value,
                "Regional Median",
                scope=scope,
                donor_rows=(donors.index + 2).tolist(),
                requires_fold_refit=True,
            )


def apply_product(
    ctx,
    field,
    product,
    method="linear",
    padding=1,
    climatology=False,
    factor=1.0,
    excluded=(),
):
    missing = [i for i in ctx.missing(field) if i not in excluded]
    if not missing:
        return
    vals, details = product.sample(ctx.frame.loc[missing], method, padding, climatology)
    low, high = {"Temperature": (-2, 45), "Salinity": (0, 42), "pH": (6, 9)}.get(
        field, (0, float("inf"))
    )
    for i, v in vals.items():
        value = v * factor
        if np.isfinite(value) and low <= value <= high:
            ctx.set(
                i,
                field,
                value,
                "CMEMS Climatology" if climatology else "CMEMS Extraction",
                **details[i],
                unit_factor=factor,
                requires_fold_refit=False,
            )


def physicochemical(ctx):
    base = Path(ctx.config["provider_root"])
    configs = {
        "Temperature": ("physical_file", "thetao", {"degrees_C"}, "nearest", 2),
        "Salinity": ("physical_file", "so", {"1e-3"}, "nearest", 0),
        "pH": ("ph_file", "ph", {"1"}, "linear", 2),
    }
    for field, (file_key, var, units, method, padding) in configs.items():
        product = MonthlyProduct([base / ctx.config[file_key]], var, units, ctx)
        isolated = set()
        try:
            if field == "Salinity":
                measured = ctx.frame.loc[ctx.sources[field] == "Original Data"]
                prior = measured.groupby(["Latitude", "Longitude"])[field].agg(
                    ["min", "median"]
                )
                vals, details = product.sample(
                    ctx.frame.loc[ctx.missing(field)], method, padding
                )
                for i, v in vals.items():
                    row = ctx.frame.loc[i]
                    key = (row.Latitude, row.Longitude)
                    estuary = key in prior.index and prior.loc[key, "min"] < 10
                    if estuary:
                        isolated.add(i)
                    if estuary and (not np.isfinite(v) or v >= 25):
                        donor = measured.loc[
                            (measured.Latitude == key[0])
                            & (measured.Longitude == key[1])
                        ]
                        ctx.set(
                            i,
                            field,
                            float(prior.loc[key, "median"]),
                            "Estuarine Median",
                            donor_rows=(donor.index + 2).tolist(),
                            requires_fold_refit=True,
                        )
                    elif np.isfinite(v) and 0 <= v <= 42:
                        ctx.set(
                            i,
                            field,
                            v,
                            "CMEMS Extraction",
                            **details[i],
                            requires_fold_refit=False,
                        )
            else:
                apply_product(ctx, field, product, method, padding)
            local_interpolation(ctx, field, "kriging", temporal=True, excluded=isolated)
            apply_product(
                ctx, field, product, "nearest", 2, climatology=True, excluded=isolated
            )
        finally:
            product.close()
        LOG.info(
            "%s complete: %s/%s populated",
            field,
            ctx.frame[field].notna().sum(),
            len(ctx.frame),
        )


def chlorophyll(ctx):
    index = nasa_index(Path(ctx.config["provider_root"]) / "NASA_Chl-a_Monthly_Data")
    groups = ctx.frame.loc[ctx.missing("Chla")].groupby(["_year", "_month"])
    for (y, m), group in groups:
        path = index.get((int(y), int(m)))
        values = nasa_sample(path, group, "chlor_a", ctx)
        for i, v in values.items():
            if np.isfinite(v) and v >= 0:
                ctx.set(
                    i,
                    "Chla",
                    v,
                    "NASA Extraction",
                    file=str(path),
                    variable="chlor_a",
                    year=int(y),
                    month=int(m),
                    padding_pixels=1,
                    requires_fold_refit=False,
                )
        remaining = group.index.intersection(ctx.missing("Chla"))
        if len(remaining):
            parts = []
            paths = []
            for (cy, cm), cp in index.items():
                if cm != m:
                    continue
                parts.append(nasa_sample(cp, group.loc[remaining], "chlor_a", ctx))
                paths.append(str(cp))
            for i in remaining:
                vals = [p[i] for p in parts if np.isfinite(p[i]) and p[i] >= 0]
                if vals:
                    ctx.set(
                        i,
                        "Chla",
                        np.mean(vals),
                        "NASA Climatology",
                        files=paths,
                        month=int(m),
                        requires_fold_refit=False,
                    )
        LOG.info("Chla %s-%02d processed", y, m)
    local_interpolation(ctx, "Chla", "knn")
    median_fallback(ctx, "Chla", monthly=True)


def poc(ctx):
    base = Path(ctx.config["provider_root"])
    index = nasa_index(base / "NASA_POC_Monthly_Data")
    bands = {
        w: nasa_index(base / f"NASA_Rrs{w}_Monthly_Data") for w in (488, 547, 645, 678)
    }
    for (y, m), group in ctx.frame.loc[ctx.missing("POC")].groupby(["_year", "_month"]):
        key = (int(y), int(m))
        path = index.get(key)
        values = nasa_sample(path, group, "poc", ctx)
        for i, v in values.items():
            if np.isfinite(v) and v >= 0:
                ctx.set(
                    i,
                    "POC",
                    v,
                    "NASA Extraction",
                    file=str(path),
                    variable="poc",
                    year=int(y),
                    month=int(m),
                    requires_fold_refit=False,
                )
        rem = group.index.intersection(ctx.missing("POC"))
        if len(rem):
            rrs = {
                w: nasa_sample(bands[w].get(key), group.loc[rem], f"Rrs_{w}", ctx)
                for w in bands
            }
            for i in rem:
                v = {w: rrs[w][i] for w in bands}
                if not all(np.isfinite(a) and a >= 0 for a in v.values()):
                    continue
                if v[488] >= v[547]:
                    ci = v[547] - (
                        v[488] + (547 - 488) / (678 - 488) * (v[678] - v[488])
                    )
                    logpoc = 171.30 * ci + 1.93
                    algorithm = "colour_index"
                elif v[547] > 0:
                    logpoc = 1.78 * (v[645] / v[547]) + 1.89
                    algorithm = "band_ratio"
                else:
                    continue
                if not -300 < logpoc < 300:
                    continue
                ctx.set(
                    i,
                    "POC",
                    10.0**logpoc,
                    "Rrs Estimate",
                    bands=v,
                    files={w: str(bands[w].get(key)) for w in bands},
                    algorithm=algorithm,
                    requires_fold_refit=False,
                )
        LOG.info("POC %s-%02d processed", y, m)
    median_fallback(ctx, "POC", monthly=True)


def spm(ctx):
    p = MonthlyProduct(
        list((Path(ctx.config["provider_root"]) / "CEMES_SPM").glob("*.nc")),
        "SPM",
        {"g/m3", "mg/L"},
        ctx,
    )
    try:
        apply_product(ctx, "SPM", p)
        local_interpolation(ctx, "SPM", "kriging", temporal=True)
        apply_product(ctx, "SPM", p, climatology=True)
    finally:
        p.close()


def oxygen(ctx):
    p = MonthlyProduct(
        list((Path(ctx.config["provider_root"]) / "CEMES_DO").glob("*.nc")),
        "o2",
        {"mmol m-3"},
        ctx,
    )
    try:
        apply_product(ctx, "DO", p, factor=0.032)
    finally:
        p.close()
    local_interpolation(ctx, "DO", "kriging")
    median_fallback(ctx, "DO")


def toc(ctx):
    ref = pd.read_excel(ctx.config["toc_reference"]).rename(columns=ALIASES)
    for c in ["TOC", "Latitude", "Longitude"]:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
    ref = ref.dropna(subset=["TOC", "Latitude", "Longitude"])
    ctx.product(ctx.config["toc_reference"])
    local_interpolation(ctx, "TOC", "kriging", external=ref, toc=True)
    local_interpolation(ctx, "TOC", "knn", toc=True)
    median_fallback(ctx, "TOC", toc=True)


def hydro(ctx):
    path = Path(ctx.config["provider_root"]) / "CMEMS_Data" / "CMEMS_MY_2014_2022.nc"
    ctx.product(path)
    with xr.open_dataset(path) as ds:
        for v, u in [("uo", "m s-1"), ("vo", "m s-1"), ("mlotst", "m")]:
            if ds[v].attrs.get("units") != u:
                raise ValueError(f"Hydro unit mismatch {v}")
        times = pd.to_datetime(ds.time.values)
        periods = {(t.year, t.month): i for i, t in enumerate(times)}
        missing = ctx.frame.index[
            ctx.frame[["Hydro_Bottom_U", "Hydro_Bottom_V", "Hydro_MLD"]]
            .isna()
            .any(axis=1)
        ]
        for (y, m), group in ctx.frame.loc[missing].groupby(["_year", "_month"]):
            key = (int(y), int(m))
            clim = key not in periods
            if clim:
                positions = [i for i, t in enumerate(times) if t.month == m]
                if not positions:
                    continue
                field = ds.isel(time=positions).mean("time").load()
            else:
                field = ds.isel(time=periods[key], drop=True).load()
            field = field.sortby("latitude").sortby("longitude").sortby("depth")
            u = field.uo.to_numpy()
            v = field.vo.to_numpy()
            depth = field.depth.to_numpy()
            for i, row in group.iterrows():
                iy = int(np.argmin(abs(field.latitude.values - row.Latitude)))
                ix = int(np.argmin(abs(field.longitude.values - row.Longitude)))
                dy = abs(float(field.latitude.values[iy]) - row.Latitude)
                dx = abs(float(field.longitude.values[ix]) - row.Longitude)
                if dy > 0.25 or dx > 0.25:
                    continue
                valid = np.isfinite(u[:, iy, ix]) & np.isfinite(v[:, iy, ix])
                neighbor = False
                if not valid.any():
                    ys = np.where(abs(field.latitude.values - row.Latitude) <= 0.25)[0]
                    xs = np.where(abs(field.longitude.values - row.Longitude) <= 0.25)[
                        0
                    ]
                    choices = []
                    for yy in ys:
                        for xx in xs:
                            if not (
                                np.isfinite(u[:, yy, xx]) & np.isfinite(v[:, yy, xx])
                            ).any():
                                continue
                            p1, p2 = np.radians(row.Latitude), np.radians(
                                float(field.latitude.values[yy])
                            )
                            dlon = np.radians(
                                float(field.longitude.values[xx]) - row.Longitude
                            )
                            a = (
                                np.sin((p2 - p1) / 2) ** 2
                                + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
                            )
                            dist = (
                                EARTH_RADIUS_KM
                                * 2
                                * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                            )
                            if dist <= 0.25 * 111.32:
                                choices.append((dist, yy, xx))
                    if not choices:
                        continue
                    _, iy, ix = min(choices)
                    neighbor = True
                    valid = np.isfinite(u[:, iy, ix]) & np.isfinite(v[:, iy, ix])
                bottom = int(np.where(valid)[0][-1])
                raw_mld = float(field.mlotst.values[iy, ix])
                mld = (
                    min(raw_mld, float(depth[bottom]))
                    if np.isfinite(raw_mld) and raw_mld >= 0
                    else np.nan
                )
                values = {
                    "Hydro_Bottom_U": float(u[bottom, iy, ix]),
                    "Hydro_Bottom_V": float(v[bottom, iy, ix]),
                    "Hydro_MLD": mld,
                }
                source = "CMEMS Climatology" if clim else "CMEMS Extraction"
                if neighbor:
                    source = "CMEMS Climatology Neighbor" if clim else "CMEMS Neighbor"
                detail = {
                    "file": str(path),
                    "year": int(y),
                    "month": int(m),
                    "temporal_method": (
                        "2014_2022_monthly_climatology" if clim else "same_year_month"
                    ),
                    "product_longitude": float(field.longitude.values[ix]),
                    "product_latitude": float(field.latitude.values[iy]),
                    "bottom_depth_m": float(depth[bottom]),
                    "mld_before_depth_cap": raw_mld if np.isfinite(raw_mld) else None,
                    "requires_fold_refit": False,
                }
                for f, val in values.items():
                    if pd.isna(ctx.frame.at[i, f]):
                        ctx.set(i, f, val, source, **detail)
            LOG.info(
                "Hydro %s-%02d processed (%s)",
                y,
                m,
                "climatology" if clim else "matched month",
            )


MODULES = OrderedDict(
    [
        ("01", physicochemical),
        ("02", chlorophyll),
        ("03", poc),
        ("04", spm),
        ("05", oxygen),
        ("06", toc),
        ("07", hydro),
    ]
)
