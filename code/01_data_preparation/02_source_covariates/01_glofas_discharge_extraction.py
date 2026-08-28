import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import os
import math
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def download_glofas_discharge(year, output_file):
    c = cdsapi.Client(url='https://ewds.climate.copernicus.eu/api')
    print(f"INFO: Requesting GloFAS discharge data for year {year} from EWDS...")
    try:
        c.retrieve(
            'cems-glofas-historical',
            {
                'system_version': 'version_4_0',
                'hydrological_model': 'lisflood',
                'product_type': 'consolidated',
                'variable': 'river_discharge_in_the_last_24_hours',
                'hyear': [str(year)],
                'hmonth': [f"{month:02d}" for month in range(1, 13)],
                'hday': [f"{day:02d}" for day in range(1, 32)],
                'area': [45.0, 105.0, 18.0, 125.0],
                'data_format': 'netcdf',
            },
            output_file)
        print(f"INFO: Download completed -> {output_file}")
        return True
    except Exception as e:
        print(f"ERROR: Download failed for {year}. Exception: e")
        return False


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.01 * 2 * math.asin(math.sqrt(a))


def perform_pure_spatial_clustering(df_coords, threshold_km=16.6):
    clusters = []
    for idx, row in df_coords.iterrows():
        r_name = str(row['River Name']).strip()
        if r_name.lower() in ['nan', 'none']: continue
        lon, lat = float(row['Longitude']), float(row['Latitude'])

        placed = False
        for cluster in clusters:
            # FIX: Used exact dictionary keys 'orig_lon' and 'orig_lat'
            c_lon = np.mean([m['orig_lon'] for m in cluster['members']])
            c_lat = np.mean([m['orig_lat'] for m in cluster['members']])

            if haversine_distance(lon, lat, c_lon, c_lat) <= threshold_km:
                cluster['members'].append({'name': r_name, 'orig_lon': lon, 'orig_lat': lat})
                placed = True
                break

        if not placed:
            clusters.append({'members': [{'name': r_name, 'orig_lon': lon, 'orig_lat': lat}]})

    MACRO_NAMES = ['yangtze river', 'changjiang river', 'pearl river', 'zhujiang river',
                   'yellow river', 'huaihe river', 'hai river', 'liao river',
                   'qiantang river', 'min river']

    result_rows = []
    for cluster in clusters:
        members = cluster['members']

        macro_member = next((m for m in members if m['name'].lower() in MACRO_NAMES), None)

        if macro_member:
            c_lon, c_lat = macro_member['orig_lon'], macro_member['orig_lat']
        else:
            c_lon = np.mean([m['orig_lon'] for m in members])
            c_lat = np.mean([m['orig_lat'] for m in members])

        for m in members:
            result_rows.append({
                'River Name': m['name'],
                'Orig_Lon': m['orig_lon'],
                'Orig_Lat': m['orig_lat'],
                'Grid_Lon': c_lon,
                'Grid_Lat': c_lat
            })

    return pd.DataFrame(result_rows)


def get_dt_pwm_target(mean_field, orig_lon, orig_lat, search_radius, sigma, claimed_grids, threshold=0.5,
                      max_allowable_q=float('inf')):
    lon_dim = [d for d in mean_field.dims if 'lon' in d.lower()][0]
    lat_dim = [d for d in mean_field.dims if 'lat' in d.lower()][0]

    lon_start, lon_end = orig_lon - search_radius, orig_lon + search_radius
    lat_start, lat_end = orig_lat - search_radius, orig_lat + search_radius

    if mean_field[lon_dim].values[0] > mean_field[lon_dim].values[-1]:
        lon_start, lon_end = lon_end, lon_start
    if mean_field[lat_dim].values[0] > mean_field[lat_dim].values[-1]:
        lat_start, lat_end = lat_end, lat_start

    slice_kwargs = {
        lon_dim: slice(lon_start, lon_end),
        lat_dim: slice(lat_start, lat_end)
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

            q_converted = (val * 24 * 3600 * 365) / (10 ** 8)
            if q_converted > max_allowable_q:
                continue

            g_lon = float(lons_val[j])
            g_lat = float(lats_val[i])
            grid_key = (round(g_lon, 4), round(g_lat, 4))

            if grid_key not in claimed_grids:
                dist_km = haversine_distance(orig_lon, orig_lat, g_lon, g_lat)
                dist_eq_deg = dist_km / 111.32
                dist_sq = dist_eq_deg ** 2

                w_dist = math.exp(-dist_sq / (2 * sigma ** 2))
                t_rank = math.log10(val + 1.0)
                composite_score = w_dist * t_rank

                if composite_score > best_score:
                    best_score = composite_score
                    target_lon = g_lon
                    target_lat = g_lat

    return target_lon, target_lat


def extract_and_integrate_flux_twopass(nc_file, coords_file, year, df_bulletin):
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Required spatial baseline file not found: '{coords_file}'")

    if coords_file.endswith('.xlsx'):
        df_coords = pd.read_excel(coords_file)
    else:
        df_coords = pd.read_csv(coords_file)

    missing_majors = {
        'Huaihe River': {'Longitude': 120.25, 'Latitude': 34.00},
        'Hai River': {'Longitude': 117.70, 'Latitude': 38.90},
        'Liao River': {'Longitude': 121.80, 'Latitude': 40.80}
    }
    existing_rivers = df_coords['River Name'].astype(str).str.strip().str.lower().values

    for m_name, m_coords in missing_majors.items():
        if m_name.lower() not in existing_rivers:
            new_row = pd.DataFrame(
                [{'River Name': m_name, 'Longitude': m_coords['Longitude'], 'Latitude': m_coords['Latitude']}])
            df_coords = pd.concat([df_coords, new_row], ignore_index=True)

    df_topo = perform_pure_spatial_clustering(df_coords, threshold_km=16.6)

    ds = xr.open_dataset(nc_file)
    var_name = [v for v in ds.data_vars if 'dis' in v.lower()][0]
    lon_dim = [d for d in ds.dims if 'lon' in d.lower()][0]
    lat_dim = [d for d in ds.dims if 'lat' in d.lower()][0]
    time_dim = [d for d in ds.dims if 'time' in d.lower()][0]

    mean_field = ds[var_name].mean(dim=time_dim, skipna=True)
    claimed_grids = set()

    MACRO_BBOX = {
        'Yangtze River': {'lon': (120.5, 122.5), 'lat': (30.5, 32.5)},
        'Changjiang River': {'lon': (120.5, 122.5), 'lat': (30.5, 32.5)},
        'Pearl River': {'lon': (113.0, 114.5), 'lat': (22.0, 23.5)},
        'Zhujiang River': {'lon': (113.0, 114.5), 'lat': (22.0, 23.5)},
        'Yellow River': {'lon': (118.0, 119.5), 'lat': (37.0, 38.5)},
        'Huaihe River': {'lon': (119.5, 120.5), 'lat': (33.5, 34.5)},
        'Hai River': {'lon': (117.5, 118.5), 'lat': (38.5, 39.5)},
        'Liao River': {'lon': (121.5, 122.5), 'lat': (40.5, 41.5)}
    }

    macro_anchors = {}
    for name, bounds in MACRO_BBOX.items():
        lons, lats = sorted(bounds['lon']), sorted(bounds['lat'])
        if ds[lon_dim].values[0] > ds[lon_dim].values[-1]: lons = lons[::-1]
        if ds[lat_dim].values[0] > ds[lat_dim].values[-1]: lats = lats[::-1]

        box_data = mean_field.sel({lon_dim: slice(lons[0], lons[1]), lat_dim: slice(lats[0], lats[1])})
        if box_data.size > 0 and not np.isnan(box_data).all():
            stacked = box_data.stack(space=[lat_dim, lon_dim])
            best_idx = int(stacked.argmax(skipna=True))
            b_coords = stacked['space'][best_idx].values.item()
            macro_anchors[name] = (float(b_coords[1]), float(b_coords[0]))
            claimed_grids.add((round(float(b_coords[1]), 4), round(float(b_coords[0]), 4)))

    P1_RADIUS, P1_SIGMA = 0.15, 0.1
    P2_RADIUS, P2_SIGMA = 0.30, 0.3
    ANOMALY_THRESHOLD = 10.0
    SCALE_SHIFT_FACTOR = 5.0
    ABSOLUTE_CEILING = 300.0

    df_bull_year = pd.DataFrame()
    if not df_bulletin.empty and 'year' in df_bulletin.columns:
        df_bull_year = df_bulletin[df_bulletin['year'] == year]

    grid_results = {}

    unique_grids = df_topo[['Grid_Lon', 'Grid_Lat']].drop_duplicates()
    for _, grid_row in unique_grids.iterrows():
        g_lon, g_lat = grid_row['Grid_Lon'], grid_row['Grid_Lat']

        associated_rivers = df_topo[(df_topo['Grid_Lon'] == g_lon) & (df_topo['Grid_Lat'] == g_lat)][
            'River Name'].str.lower().tolist()

        official_q = np.nan
        is_official = False
        primary_river = associated_rivers[0]

        for r_name in associated_rivers:
            if not df_bull_year.empty:
                match = df_bull_year[df_bull_year['river'].str.strip().str.lower() == r_name]
                if not match.empty:
                    official_q = float(match.iloc[0]['Q'])
                    is_official = True
                    primary_river = r_name
                    break

        if primary_river in [k.lower() for k in macro_anchors.keys()]:
            macro_key = next(k for k in macro_anchors.keys() if k.lower() == primary_river)
            t_lon, t_lat = macro_anchors[macro_key]
        else:
            t_lon, t_lat = get_dt_pwm_target(
                mean_field, g_lon, g_lat, P1_RADIUS, P1_SIGMA, claimed_grids, max_allowable_q=ABSOLUTE_CEILING
            )
            if t_lon is not None:
                claimed_grids.add((round(t_lon, 4), round(t_lat, 4)))

        q_val = np.nan
        if t_lon is not None:
            if is_official:
                q_val = official_q
            else:
                ts_data = ds[var_name].sel({lon_dim: t_lon, lat_dim: t_lat}, method='nearest').values
                q_val = np.nansum(ts_data * 24 * 3600) / (10 ** 8)

        grid_results[(g_lon, g_lat)] = {'t_lon': t_lon, 't_lat': t_lat, 'q': q_val, 'official': is_official}

    for g_key, g_data in grid_results.items():
        if g_data['official'] or pd.isna(g_data['q']):
            continue

        if g_data['q'] < ANOMALY_THRESHOLD:
            p1_key = (round(g_data['t_lon'], 4), round(g_data['t_lat'], 4))
            if p1_key in claimed_grids:
                claimed_grids.remove(p1_key)

            t_lon_2, t_lat_2 = get_dt_pwm_target(
                mean_field, g_key[0], g_key[1], P2_RADIUS, P2_SIGMA, claimed_grids, max_allowable_q=ABSOLUTE_CEILING
            )

            overwrite_approved = False
            if t_lon_2 is not None:
                ts_data_2 = ds[var_name].sel({lon_dim: t_lon_2, lat_dim: t_lat_2}, method='nearest').values
                q_val_2 = np.nansum(ts_data_2 * 24 * 3600) / (10 ** 8)

                if q_val_2 > g_data['q'] * SCALE_SHIFT_FACTOR:
                    overwrite_approved = True
                    grid_results[g_key]['q'] = q_val_2
                    claimed_grids.add((round(t_lon_2, 4), round(t_lat_2, 4)))

            if not overwrite_approved:
                claimed_grids.add(p1_key)

    final_results = []
    for _, row in df_topo.iterrows():
        g_lon, g_lat = row['Grid_Lon'], row['Grid_Lat']
        g_data = grid_results.get((g_lon, g_lat), {})
        q_val = g_data.get('q', np.nan)
        is_official = g_data.get('official', False)

        if pd.isna(q_val):
            continue

        sd_val = q_val * 0.05 if is_official else q_val * 0.20

        final_results.append({
            'River': row['River Name'],
            'Year': year,
            'Orig_Lon': row['Orig_Lon'],
            'Orig_Lat': row['Orig_Lat'],
            'Grid_Lon': g_lon,
            'Grid_Lat': g_lat,
            'Annual_Q_10^8m3': round(q_val, 2),
            'Q_SD_10^8m3': round(sd_val, 2)
        })

    ds.close()
    return pd.DataFrame(final_results)


if __name__ == '__main__':
    project_root = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data")).expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    COORDS_FILE = data_root / 'raw' / '100_Rivers_2018.xlsx'
    START_YEAR = 2014
    END_YEAR = 2025
    NC_DATA_DIR = data_root / 'raw' / 'GloFAS_Discharge_China_Coast'

    NC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    exogenous_output = output_root / "exogenous"
    exogenous_output.mkdir(parents=True, exist_ok=True)

    bulletin_file_xlsx = data_root / 'raw' / 'China Sediment Bulletin.xlsx'
    df_bulletin = pd.DataFrame()

    if os.path.exists(bulletin_file_xlsx):
        df_bulletin = pd.read_excel(bulletin_file_xlsx)
    else:
        print(f"WARNING: {bulletin_file_xlsx} not found.")

    all_years_flux = []

    for year in range(START_YEAR, END_YEAR + 1):
        nc_filename = NC_DATA_DIR / f'GloFAS_Discharge_China_Coast_{year}.nc'

        if not os.path.exists(nc_filename):
            success = download_glofas_discharge(year, nc_filename)
            if not success: continue

        print(f"INFO: Processing {year} spatial routing...")
        df_year = extract_and_integrate_flux_twopass(nc_filename, COORDS_FILE, year, df_bulletin)

        if df_year is not None and not df_year.empty:
            all_years_flux.append(df_year)

    if all_years_flux:
        df_final = pd.concat(all_years_flux, ignore_index=True)
        output_filename = exogenous_output / 'Step1_Annual_Runoff_2014_2025.csv'
        df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"SUCCESS: Integration completed. Output: {output_filename}")
    else:
        print("ERROR: Integration failed. Matrix empty.")
