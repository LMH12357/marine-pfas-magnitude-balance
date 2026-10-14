"""Extract annual river discharge from local GloFAS products and bulletins."""

import argparse
import hashlib
import json
import logging
import xarray as xr
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path

LOG = logging.getLogger(__name__)


def download_glofas_discharge(year, output_file):
    import cdsapi

    c = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
    print(f"INFO: Requesting GloFAS discharge data for year {year} from EWDS...")
    try:
        c.retrieve(
            "cems-glofas-historical",
            {
                "system_version": "version_4_0",
                "hydrological_model": "lisflood",
                "product_type": "consolidated",
                "variable": "river_discharge_in_the_last_24_hours",
                "hyear": [str(year)],
                "hmonth": [f"{month:02d}" for month in range(1, 13)],
                "hday": [f"{day:02d}" for day in range(1, 32)],
                "area": [45.0, 105.0, 18.0, 125.0],
                "data_format": "netcdf",
            },
            output_file,
        )
        print(f"INFO: Download completed -> {output_file}")
        return True
    except Exception as e:
        LOG.error("Download failed for %s: %s", year, e)
        return False


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.01 * 2 * math.asin(math.sqrt(a))


def perform_pure_spatial_clustering(df_coords, threshold_km=16.6):
    clusters = []
    for idx, row in df_coords.iterrows():
        r_name = str(row["River Name"]).strip()
        if r_name.lower() in ["nan", "none", ""]:
            continue
        lon, lat = float(row["Longitude"]), float(row["Latitude"])

        placed = False
        for cluster in clusters:
            c_lon = np.mean([m["orig_lon"] for m in cluster["members"]])
            c_lat = np.mean([m["orig_lat"] for m in cluster["members"]])

            if haversine_distance(lon, lat, c_lon, c_lat) <= threshold_km:
                cluster["members"].append(
                    {"name": r_name, "orig_lon": lon, "orig_lat": lat}
                )
                placed = True
                break

        if not placed:
            clusters.append(
                {"members": [{"name": r_name, "orig_lon": lon, "orig_lat": lat}]}
            )

    MACRO_NAMES = [
        "yangtze river",
        "changjiang river",
        "pearl river",
        "zhujiang river",
        "yellow river",
        "huaihe river",
        "hai river",
        "liao river",
        "qiantang river",
        "min river",
    ]

    result_rows = []
    for cluster in clusters:
        members = cluster["members"]

        macro_member = next(
            (m for m in members if m["name"].lower() in MACRO_NAMES), None
        )

        if macro_member:
            c_lon, c_lat = macro_member["orig_lon"], macro_member["orig_lat"]
        else:
            c_lon = np.mean([m["orig_lon"] for m in members])
            c_lat = np.mean([m["orig_lat"] for m in members])

        for m in members:
            result_rows.append(
                {
                    "River Name": m["name"],
                    "Orig_Lon": m["orig_lon"],
                    "Orig_Lat": m["orig_lat"],
                    "Grid_Lon": c_lon,
                    "Grid_Lat": c_lat,
                }
            )

    return pd.DataFrame(result_rows)


def get_dt_pwm_target(
    mean_field,
    orig_lon,
    orig_lat,
    search_radius,
    sigma,
    claimed_grids,
    threshold=0.5,
    max_allowable_q=float("inf"),
):
    lon_dim = [d for d in mean_field.dims if "lon" in d.lower()][0]
    lat_dim = [d for d in mean_field.dims if "lat" in d.lower()][0]

    lon_start, lon_end = orig_lon - search_radius, orig_lon + search_radius
    lat_start, lat_end = orig_lat - search_radius, orig_lat + search_radius

    if mean_field[lon_dim].values[0] > mean_field[lon_dim].values[-1]:
        lon_start, lon_end = lon_end, lon_start
    if mean_field[lat_dim].values[0] > mean_field[lat_dim].values[-1]:
        lat_start, lat_end = lat_end, lat_start

    slice_kwargs = {
        lon_dim: slice(lon_start, lon_end),
        lat_dim: slice(lat_start, lat_end),
    }

    window = mean_field.sel(**slice_kwargs)

    if window.size == 0 or np.isnan(window).all():
        return None, None

    lats_val = window[lat_dim].values
    lons_val = window[lon_dim].values
    vals = window.values

    best_score = -1.0
    target_lon, target_lat = None, None

    for i in range(len(lats_val)):
        for j in range(len(lons_val)):
            val = vals[i, j]

            if np.isnan(val) or val < threshold:
                continue

            q_converted = (
                val * 24 * 3600 * mean_field.attrs.get("annual_days", 365)
            ) / (10**8)
            if q_converted > max_allowable_q:
                continue

            g_lon = float(lons_val[j])
            g_lat = float(lats_val[i])
            grid_key = (round(g_lon, 4), round(g_lat, 4))

            if grid_key not in claimed_grids:
                dist_km = haversine_distance(orig_lon, orig_lat, g_lon, g_lat)
                dist_eq_deg = dist_km / 111.32
                dist_sq = dist_eq_deg**2

                w_dist = math.exp(-dist_sq / (2 * sigma**2))
                t_rank = math.log10(val + 1.0)
                composite_score = w_dist * t_rank

                if composite_score > best_score:
                    best_score = composite_score
                    target_lon = g_lon
                    target_lat = g_lat

    return target_lon, target_lat


def extract_and_integrate_flux_twopass(nc_file, coords_file, year, df_bulletin):
    if not os.path.exists(coords_file):
        raise FileNotFoundError(
            f"Required spatial baseline file not found: '{coords_file}'"
        )

    if Path(coords_file).suffix.lower() == ".xlsx":
        df_coords = pd.read_excel(coords_file)
    else:
        df_coords = pd.read_csv(coords_file)
    df_coords.columns = [str(c).strip() for c in df_coords.columns]
    df_coords = df_coords.loc[df_coords["River Name"].notna()].copy()
    for c in ["Longitude", "Latitude"]:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="raise")
    if not np.isfinite(df_coords[["Longitude", "Latitude"]].to_numpy()).all():
        raise ValueError("River coordinates must be finite")
    if "ID" in df_coords:
        df_coords = df_coords.sort_values("ID", kind="stable")
    else:
        df_coords = df_coords.sort_values(
            ["Latitude", "Longitude", "River Name"], kind="stable"
        )

    missing_majors = {
        "Huaihe River": {"Longitude": 120.25, "Latitude": 34.00},
        "Hai River": {"Longitude": 117.70, "Latitude": 38.90},
        "Liao River": {"Longitude": 121.80, "Latitude": 40.80},
    }
    existing_rivers = df_coords["River Name"].astype(str).str.strip().str.lower().values

    for m_name, m_coords in missing_majors.items():
        if m_name.lower() not in existing_rivers:
            new_row = pd.DataFrame(
                [
                    {
                        "River Name": m_name,
                        "Longitude": m_coords["Longitude"],
                        "Latitude": m_coords["Latitude"],
                    }
                ]
            )
            df_coords = pd.concat([df_coords, new_row], ignore_index=True)

    df_topo = perform_pure_spatial_clustering(df_coords, threshold_km=16.6)

    ds = xr.open_dataset(nc_file)
    if "dis24" not in ds:
        ds.close()
        raise ValueError("Expected GloFAS dis24 product")
    var_name = "dis24"
    lon_dim = [d for d in ds.dims if "lon" in d.lower()][0]
    lat_dim = [d for d in ds.dims if "lat" in d.lower()][0]
    time_dim = [d for d in ds.dims if "time" in d.lower()][0]

    units = ds[var_name].attrs.get("units")
    if units not in ("m**3 s**-1", "m3 s-1", "m3/s"):
        ds.close()
        raise ValueError(f"Unexpected GloFAS units: {units}")
    # dis24 is averaged over the preceding 24 hours; validity marks the end.
    valid_times = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values))
    daily_dates = valid_times - pd.Timedelta(days=1)
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if not daily_dates.equals(expected):
        ds.close()
        raise ValueError(f"Incomplete or mismatched daily coverage for {year}")
    days = len(expected)
    mean_field = (
        ds[var_name]
        .transpose(time_dim, lat_dim, lon_dim)
        .mean(dim=time_dim, skipna=True)
        .load()
    )
    mean_field.attrs["annual_days"] = days

    def annual_volume(t_lon, t_lat):
        ts = (
            ds[var_name]
            .sel({lon_dim: t_lon, lat_dim: t_lat}, method="nearest")
            .to_numpy()
        )
        if len(ts) != days or not np.isfinite(ts).all() or (ts < 0).any():
            return np.nan
        return float(ts.sum(dtype=np.float64) * 86400 / 1e8)

    claimed_grids = set()

    MACRO_BBOX = {
        "Yangtze River": {"lon": (120.5, 122.5), "lat": (30.5, 32.5)},
        "Changjiang River": {"lon": (120.5, 122.5), "lat": (30.5, 32.5)},
        "Pearl River": {"lon": (113.0, 114.5), "lat": (22.0, 23.5)},
        "Zhujiang River": {"lon": (113.0, 114.5), "lat": (22.0, 23.5)},
        "Yellow River": {"lon": (118.0, 119.5), "lat": (37.0, 38.5)},
        "Huaihe River": {"lon": (119.5, 120.5), "lat": (33.5, 34.5)},
        "Hai River": {"lon": (117.5, 118.5), "lat": (38.5, 39.5)},
        "Liao River": {"lon": (121.5, 122.5), "lat": (40.5, 41.5)},
    }

    macro_anchors = {}
    for name, bounds in MACRO_BBOX.items():
        lons, lats = sorted(bounds["lon"]), sorted(bounds["lat"])
        if ds[lon_dim].values[0] > ds[lon_dim].values[-1]:
            lons = lons[::-1]
        if ds[lat_dim].values[0] > ds[lat_dim].values[-1]:
            lats = lats[::-1]

        box_data = mean_field.sel(
            {lon_dim: slice(lons[0], lons[1]), lat_dim: slice(lats[0], lats[1])}
        )
        if box_data.size > 0 and not np.isnan(box_data).all():
            stacked = box_data.stack(space=[lat_dim, lon_dim])
            best_idx = int(np.nanargmax(stacked.to_numpy()))
            b_coords = stacked["space"][best_idx].values.item()
            macro_anchors[name] = (float(b_coords[1]), float(b_coords[0]))
            claimed_grids.add(
                (round(float(b_coords[1]), 4), round(float(b_coords[0]), 4))
            )

    P1_RADIUS, P1_SIGMA = 0.15, 0.1
    P2_RADIUS, P2_SIGMA = 0.30, 0.3
    ANOMALY_THRESHOLD = 10.0
    SCALE_SHIFT_FACTOR = 5.0
    ABSOLUTE_CEILING = 300.0

    df_bull_year = pd.DataFrame()
    if not df_bulletin.empty and "year" in df_bulletin.columns:
        df_bull_year = df_bulletin[df_bulletin["year"] == year]

    grid_results = {}

    unique_grids = df_topo[["Grid_Lon", "Grid_Lat"]].drop_duplicates()
    for _, grid_row in unique_grids.iterrows():
        g_lon, g_lat = grid_row["Grid_Lon"], grid_row["Grid_Lat"]

        associated_rivers = (
            df_topo[(df_topo["Grid_Lon"] == g_lon) & (df_topo["Grid_Lat"] == g_lat)][
                "River Name"
            ]
            .str.lower()
            .tolist()
        )

        official_q = np.nan
        is_official = False
        primary_river = associated_rivers[0]

        for r_name in associated_rivers:
            if not df_bull_year.empty:
                match = df_bull_year[
                    df_bull_year["river"].str.strip().str.lower() == r_name
                ]
                if not match.empty:
                    vals = pd.to_numeric(match["Q"], errors="raise").unique()
                    if len(vals) != 1 or not np.isfinite(vals[0]) or vals[0] < 0:
                        raise ValueError(
                            f"Inconsistent official discharge: {r_name}, {year}"
                        )
                    official_q = float(vals[0])
                    is_official = True
                    primary_river = r_name
                    break

        if primary_river in [k.lower() for k in macro_anchors.keys()]:
            macro_key = next(
                k for k in macro_anchors.keys() if k.lower() == primary_river
            )
            t_lon, t_lat = macro_anchors[macro_key]
        else:
            t_lon, t_lat = get_dt_pwm_target(
                mean_field,
                g_lon,
                g_lat,
                P1_RADIUS,
                P1_SIGMA,
                claimed_grids,
                max_allowable_q=ABSOLUTE_CEILING,
            )
            if t_lon is not None:
                claimed_grids.add((round(t_lon, 4), round(t_lat, 4)))

        q_val = official_q if is_official else np.nan
        if t_lon is not None:
            if not is_official:
                q_val = annual_volume(t_lon, t_lat)

        grid_results[(g_lon, g_lat)] = {
            "t_lon": t_lon,
            "t_lat": t_lat,
            "q": q_val,
            "official": is_official,
        }

    for g_key, g_data in grid_results.items():
        if g_data["official"] or pd.isna(g_data["q"]):
            continue

        if g_data["q"] < ANOMALY_THRESHOLD:
            p1_key = (round(g_data["t_lon"], 4), round(g_data["t_lat"], 4))
            if p1_key in claimed_grids:
                claimed_grids.remove(p1_key)

            t_lon_2, t_lat_2 = get_dt_pwm_target(
                mean_field,
                g_key[0],
                g_key[1],
                P2_RADIUS,
                P2_SIGMA,
                claimed_grids,
                max_allowable_q=ABSOLUTE_CEILING,
            )

            overwrite_approved = False
            if t_lon_2 is not None:
                q_val_2 = annual_volume(t_lon_2, t_lat_2)

                if q_val_2 > g_data["q"] * SCALE_SHIFT_FACTOR:
                    overwrite_approved = True
                    grid_results[g_key]["q"] = q_val_2
                    grid_results[g_key]["t_lon"] = t_lon_2
                    grid_results[g_key]["t_lat"] = t_lat_2
                    claimed_grids.add((round(t_lon_2, 4), round(t_lat_2, 4)))

            if not overwrite_approved:
                claimed_grids.add(p1_key)

    final_results = []
    for _, row in df_topo.iterrows():
        g_lon, g_lat = row["Grid_Lon"], row["Grid_Lat"]
        g_data = grid_results.get((g_lon, g_lat), {})
        q_val = g_data.get("q", np.nan)
        is_official = g_data.get("official", False)

        if pd.isna(q_val):
            continue

        sd_val = q_val * 0.05 if is_official else q_val * 0.20

        final_results.append(
            {
                "River": row["River Name"],
                "Year": year,
                "Orig_Lon": row["Orig_Lon"],
                "Orig_Lat": row["Orig_Lat"],
                "Grid_Lon": g_lon,
                "Grid_Lat": g_lat,
                "Annual_Q_10^8m3": float(q_val),
                "Q_SD_10^8m3": float(sd_val),
                "Q_Source": "Bulletin" if is_official else "GloFAS",
                "Product_Lon": g_data.get("t_lon"),
                "Product_Lat": g_data.get("t_lat"),
                "Daily_Count": days,
                "Relative_SD_Assumption": 0.05 if is_official else 0.20,
            }
        )

    ds.close()
    return pd.DataFrame(final_results)


def run_discharge(config):
    coords = Path(config["rivers_file"])
    bulletin = Path(config["bulletin_file"])
    provider = Path(config["discharge_directory"])
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    if not bulletin.is_file():
        raise FileNotFoundError(bulletin)
    df_bulletin = pd.read_excel(bulletin)
    df_bulletin.columns = [str(c).strip() for c in df_bulletin.columns]
    rows = []
    products = []
    for year in range(config.get("start_year", 2014), config.get("end_year", 2025) + 1):
        path = provider / f"GloFAS_Discharge_China_Coast_{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        LOG.info("START GloFAS annual extraction %s", year)
        annual = extract_and_integrate_flux_twopass(
            str(path), str(coords), year, df_bulletin
        )
        if annual.empty:
            raise ValueError(f"No discharge nodes for {year}")
        rows.append(annual)
        products.append(
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        LOG.info(
            "DONE %s: %s river records, %s source nodes",
            year,
            len(annual),
            len(annual[["Grid_Lon", "Grid_Lat"]].drop_duplicates()),
        )
    result = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["Year", "Grid_Lat", "Grid_Lon", "River"], kind="stable")
        .reset_index(drop=True)
    )
    target = output / "Step1_Annual_Runoff_2014_2025.csv"
    result.to_csv(target, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "Annual river discharge extraction",
        "start_year": config.get("start_year", 2014),
        "end_year": config.get("end_year", 2025),
        "rows": len(result),
        "time_rule": "dis24 interval ending at valid_time; associated date is valid_time minus 24 hours",
        "relative_sd_assumptions": {"Bulletin": 0.05, "GloFAS": 0.20},
        "spatial_clustering_km": 16.6,
        "inputs": [
            {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (coords, bulletin)
        ]
        + products,
        "output": {
            "file": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_discharge(json.loads(args.config.read_text(encoding="utf-8")))
