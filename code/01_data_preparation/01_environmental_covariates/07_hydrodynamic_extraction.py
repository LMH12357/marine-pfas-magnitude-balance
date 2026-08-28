import xarray as xr
import pandas as pd
import numpy as np
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ================= 1. Configurable project/data roots =================
PROJECT_ROOT = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("PFAS_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("PFAS_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()
DIR_CMEMS = DATA_ROOT / "raw" / "CMEMS_Data"
NC_MY = DIR_CMEMS / "CMEMS_MY_2014_2022.nc"

INPUT_EXCEL = DATA_ROOT / "raw" / "DATA.xlsx"
SHEET_ENV = "Spatiotemporal identification"
OUTPUT_FILE = OUTPUT_ROOT / "Final_Dataset_With_Real_Hydro.xlsx"
SEARCH_RADIUS_DEG = 0.25


# ================= 2. Spatial Core =================
def parse_dms(s):
    """Parse degree-minute-second coordinate format."""
    if pd.isnull(s): return None
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    s_clean = re.sub(r"[°'′\"″]", " ", s)
    parts = [float(x) for x in s_clean.split() if x]
    if not parts: return None
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return parts[0] + parts[1] / 60.0
    elif len(parts) >= 3:
        return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    return None


def calculate_haversine_matrix(lat1, lon1, lat_grid, lon_grid):
    """Calculate Great-Circle Distance (Haversine formula) in kilometers."""
    R = 6371.01
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat_grid)
    dphi = phi2 - phi1
    dlambda = np.radians(lon_grid) - np.radians(lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# ================= 3. Benthic & MLD Feature Extractor =================
def extract_hydro_profile(ds_point):
    """Extract hydrodynamic parameters (bottom u and v only) and MLD."""
    u_prof = ds_point['uo'].values
    v_prof = ds_point['vo'].values
    mld = float(ds_point['mlotst'].values) if 'mlotst' in ds_point.data_vars else np.nan

    if u_prof.ndim == 0:
        u_prof, v_prof = np.array([u_prof]), np.array([v_prof])

    valid_mask = ~np.isnan(u_prof) & ~np.isnan(v_prof)

    if np.any(valid_mask):
        u_bottom = float(u_prof[valid_mask][-1])
        v_bottom = float(v_prof[valid_mask][-1])

        if 'depth' in ds_point.coords:
            bottom_depth = float(ds_point['depth'].values[valid_mask][-1])
            if not np.isnan(mld) and mld > bottom_depth:
                mld = bottom_depth
    else:
        u_bottom, v_bottom, mld = np.nan, np.nan, np.nan

    return u_bottom, v_bottom, mld


# ================= 4. Climatology-based extraction =================
def process_hydro_dynamics():
    if not INPUT_EXCEL.is_file():
        raise FileNotFoundError(f"Source file not found: {INPUT_EXCEL}")
    if not NC_MY.is_file():
        raise FileNotFoundError(f"Hydrodynamic input not found: {NC_MY}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Initializing hydrodynamic vector extraction pipeline (Historical & Climatology Routing)...")
    if not os.path.exists(INPUT_EXCEL) or not os.path.exists(NC_MY):
        raise FileNotFoundError(f"Required files are missing. Please verify {BASE_DIR} and {NC_MY}.")

    try:
        print("  -> Loading historical dataset and pre-computing monthly climatology...")
        ds_my = xr.open_dataset(NC_MY)
        # Precompute monthly climatological means for 2014–2022
        ds_climatology = ds_my.groupby('time.month').mean('time')
    except Exception as e:
        raise RuntimeError(f"Failed to load NetCDF files or compute climatology: {e}")

    df = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_ENV)

    df['Latitude'] = df['Latitude'].apply(parse_dms)
    df['Longitude'] = df['Longitude'].apply(parse_dms)
    df['Samplingtime'] = pd.to_datetime(df['Samplingtime'], errors='coerce')

    hydro_results = []
    audit_stats = {'Direct_Extraction': 0, 'Climatology_Direct': 0, 'Neighbor_Expanded': 0, 'Missing_or_Error': 0}

    lat_k = 'latitude' if 'latitude' in ds_my.coords else 'lat'
    lon_k = 'longitude' if 'longitude' in ds_my.coords else 'lon'

    total_rows = len(df)
    for idx, row in df.iterrows():
        lat, lon, time = row['Latitude'], row['Longitude'], row['Samplingtime']

        if pd.isna(lat) or pd.isna(lon) or pd.isna(time):
            hydro_results.append({
                'Hydro_Bottom_U': np.nan, 'Hydro_Bottom_V': np.nan,
                'Hydro_MLD': np.nan, 'Hydro_Source': 'Invalid_Coordinate'
            })
            audit_stats['Missing_or_Error'] += 1
            continue

        try:
            # Routing Logic: Determine Tier 1 (Historical) or Tier 2 (Climatology) source slice
            if time.year <= 2022:
                ds_time = ds_my.sel(time=time, method='nearest')
                tier_direct_label = 'Direct_Extraction'
            else:
                target_month = time.month
                ds_time = ds_climatology.sel(month=target_month)
                tier_direct_label = f'Climatology_Month_{target_month:02d}'

            # Execute Direct Extraction (Tier 1 & Tier 2)
            native_point = ds_time.sel({lat_k: lat, lon_k: lon}, method='nearest')
            u_native = native_point['uo'].values
            v_native = native_point['vo'].values

            u_surf_test = u_native[0] if u_native.ndim > 0 else u_native
            v_surf_test = v_native[0] if v_native.ndim > 0 else v_native

            if not np.isnan(u_surf_test) and not np.isnan(v_surf_test):
                ub, vb, mld = extract_hydro_profile(native_point)
                hydro_results.append({
                    'Hydro_Bottom_U': ub, 'Hydro_Bottom_V': vb,
                    'Hydro_MLD': mld, 'Hydro_Source': tier_direct_label
                })
                if time.year <= 2022:
                    audit_stats['Direct_Extraction'] += 1
                else:
                    audit_stats['Climatology_Direct'] += 1
                continue

            # Execute Tier 3: Unified 0.25° Neighbor Expansion for missing values
            lat_min, lat_max = lat - SEARCH_RADIUS_DEG, lat + SEARCH_RADIUS_DEG
            lon_min, lon_max = lon - SEARCH_RADIUS_DEG, lon + SEARCH_RADIUS_DEG

            subset = ds_time.sel({lat_k: slice(lat_min, lat_max), lon_k: slice(lon_min, lon_max)})
            if subset[lat_k].size == 0 or subset[lon_k].size == 0:
                raise ValueError("Spatial subset out of bounds")

            lats_arr = subset[lat_k].values
            lons_arr = subset[lon_k].values

            if lats_arr.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lons_arr, lats_arr)
            else:
                lat_grid, lon_grid = lats_arr, lons_arr

            dist_km = calculate_haversine_matrix(lat, lon, lat_grid, lon_grid)

            u_surf_mat = subset['uo'].isel(depth=0).values if 'depth' in subset.dims else subset['uo'].values
            v_surf_mat = subset['vo'].isel(depth=0).values if 'depth' in subset.dims else subset['vo'].values

            valid_mask_mat = ~np.isnan(u_surf_mat) & ~np.isnan(v_surf_mat)
            dist_km_masked = np.where(valid_mask_mat, dist_km, np.inf)

            min_dist = np.min(dist_km_masked)
            search_radius_km = SEARCH_RADIUS_DEG * 111.32

            if min_dist <= search_radius_km:
                min_idx = np.unravel_index(np.argmin(dist_km_masked), dist_km_masked.shape)
                found_lat = lat_grid[min_idx]
                found_lon = lon_grid[min_idx]

                valid_point = subset.sel({lat_k: found_lat, lon_k: found_lon}, method='nearest')
                ub, vb, mld = extract_hydro_profile(valid_point)
                hydro_results.append({
                    'Hydro_Bottom_U': ub, 'Hydro_Bottom_V': vb,
                    'Hydro_MLD': mld, 'Hydro_Source': 'Neighbor_Expanded'
                })
                audit_stats['Neighbor_Expanded'] += 1
            else:
                hydro_results.append({
                    'Hydro_Bottom_U': np.nan, 'Hydro_Bottom_V': np.nan,
                    'Hydro_MLD': np.nan, 'Hydro_Source': 'Land_Mask_or_Missing'
                })
                audit_stats['Missing_or_Error'] += 1

        except (KeyError, IndexError, ValueError):
            hydro_results.append({
                'Hydro_Bottom_U': np.nan, 'Hydro_Bottom_V': np.nan,
                'Hydro_MLD': np.nan, 'Hydro_Source': 'Extraction_Error'
            })
            audit_stats['Missing_or_Error'] += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total_rows:
            print(f"Processed {idx + 1}/{total_rows} points")

    res_df = pd.DataFrame(hydro_results)

    cols_to_drop = [c for c in ['Hydro_Bottom_Speed', 'Hydro_Surface_Speed', 'Hydro_Bottom_U', 'Hydro_Bottom_V', 'Hydro_Surface_U', 'Hydro_Surface_V', 'Hydro_MLD', 'Hydro_Source'] if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    df_final = pd.concat([df, res_df], axis=1)

    print(f"Exporting dataset to {OUTPUT_FILE}...")
    df_final.to_excel(OUTPUT_FILE, sheet_name=SHEET_ENV, index=False)

    print("-" * 55)
    print("Hydrodynamics Climatology & Assimilation Audit")
    print("-" * 55)
    print(f"Tier 1 (Historical Direct)  : {audit_stats['Direct_Extraction']}")
    print(f"Tier 2 (Climatology Direct) : {audit_stats['Climatology_Direct']}")
    print(f"Tier 3 (Neighbor Expanded)  : {audit_stats['Neighbor_Expanded']}")
    print(f"Missing / Error             : {audit_stats['Missing_or_Error']}")
    print("-" * 55)

if __name__ == "__main__":
    process_hydro_dynamics()
