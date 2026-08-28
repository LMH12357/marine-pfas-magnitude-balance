import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ================= 1. Global Spatial & Chemical Configuration =================
SEA_BOUNDARIES = {
    'Bohai Sea': {'lat': (37.0, 42.0), 'lon': (117.0, 122.5)},
    'Yellow Sea': {'lat': (31.0, 37.0), 'lon': (118.0, 127.0)},
    'East China Sea': {'lat': (23.0, 31.0), 'lon': (116.0, 131.0)},
    'South China Sea': {'lat': (3.0, 23.0), 'lon': (105.0, 121.0)}
}

# Compound groups for the 2 x 2 chain-length and functional-group framework
LEGACY_PFAS_WHITELIST = [
    'PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 'PFHxDA', 'PFODA',
    'PFHxS', 'PFHpS', 'PFOS', 'PFNS', 'PFDS', 'PFECHS',
    'PFBA', 'PFPeA', 'PFHxA', 'PFHpA',
    'PFBS', 'PFPeS'
]


def get_sea_area_by_topology(lon, lat):
    for sea, bounds in SEA_BOUNDARIES.items():
        if bounds['lat'][0] <= lat <= bounds['lat'][1] and bounds['lon'][0] <= lon <= bounds['lon'][1]:
            return sea
    return 'OpenOcean'


def build_topology_mapping(estuary_file, rivers_file, step1_file, output_file):
    if not os.path.exists(rivers_file):
        raise FileNotFoundError(f"Required baseline file not found: '{rivers_file}'")
    if not os.path.exists(step1_file):
        raise FileNotFoundError(f"Required Step 1 file not found: '{step1_file}'")

    df_rivers = pd.read_excel(rivers_file)
    df_rivers.columns = [str(c).strip() for c in df_rivers.columns]

    df_step1 = pd.read_csv(step1_file)

    # Scenario A: Filter for 2018 baseline year to collapse temporal dimension
    df_step1 = df_step1[df_step1['Year'] == 2018]

    known_mapping = {}
    if os.path.exists(estuary_file):
        df_est = pd.read_excel(estuary_file)
        for _, row in df_est.iterrows():
            r_name = str(row['River']).strip()
            sea = str(row['District I']).strip()
            if r_name.lower() not in ['nan', 'none'] and sea.lower() not in ['nan', 'none']:
                known_mapping[r_name.lower()] = sea

    q_cols = [c for c in df_rivers.columns if 'runoff' in c.lower()]
    q_col = q_cols[0] if q_cols else None

    # Restrict topology mapping to the predefined compound groups
    pfas_cols = [c for c in df_rivers.columns if c in LEGACY_PFAS_WHITELIST]
    if '∑PFAAs (ng/L)' in df_rivers.columns and '∑PFAAs (ng/L)' not in pfas_cols:
        pfas_cols.append('∑PFAAs (ng/L)')

    grid_groups = df_step1.groupby(['Grid_Lon', 'Grid_Lat'])['River'].apply(lambda x: list(set(x))).reset_index()
    mapping_results = []

    for _, row in grid_groups.iterrows():
        g_lon = row['Grid_Lon']
        g_lat = row['Grid_Lat']
        rivers_in_grid = row['River']

        sea_area = None
        for r in rivers_in_grid:
            if str(r).lower() in known_mapping:
                sea_area = known_mapping[str(r).lower()]
                break

        if not sea_area:
            sea_area = get_sea_area_by_topology(g_lon, g_lat)

        emp_mask = df_rivers['River Name'].str.strip().str.lower().isin(
            [str(r).strip().lower() for r in rivers_in_grid])
        emp_matches = df_rivers[emp_mask]

        profile = {p: np.nan for p in pfas_cols}

        if not emp_matches.empty:
            if q_col is not None and emp_matches[q_col].sum() > 0:
                weights = emp_matches[q_col]
                for p in pfas_cols:
                    valid_mask = emp_matches[p].notna() & weights.notna()
                    if valid_mask.sum() > 0:
                        mass_sum = (emp_matches.loc[valid_mask, p] * weights[valid_mask]).sum()
                        profile[p] = mass_sum / weights[valid_mask].sum()
            else:
                for p in pfas_cols:
                    profile[p] = emp_matches[p].mean()

        entry = {
            'Grid_Lon': g_lon,
            'Grid_Lat': g_lat,
            'Included_Rivers': ", ".join(sorted([str(r) for r in rivers_in_grid])),
            'Sea_Area': sea_area
        }
        entry.update(profile)
        mapping_results.append(entry)

    df_mapping = pd.DataFrame(mapping_results)
    df_mapping.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"INFO: Topology mapping and FWMC fusion completed -> {output_file}")


def extract_regional_fingerprints(estuary_file, output_file):
    if not os.path.exists(estuary_file):
        raise FileNotFoundError(f"Required estuary data not found: '{estuary_file}'")

    df = pd.read_excel(estuary_file)
    df.rename(columns=lambda x: str(x).strip(), inplace=True)

    # Apply the predefined compound-group filter
    pfas_cols = [c for c in df.columns if c in LEGACY_PFAS_WHITELIST]

    # Convert to numeric but preserve NaN (No zero-filling)
    for p in pfas_cols:
        df[p] = pd.to_numeric(df[p], errors='coerce')

    # Calculate Global RSD Prior (Skip NaN to ensure valid statistical representation)
    global_rsd = {}
    for p in pfas_cols:
        g_mean = df[p].mean(skipna=True)
        g_std = df[p].std(skipna=True)
        if pd.notna(g_mean) and g_mean > 0 and pd.notna(g_std):
            global_rsd[p] = g_std / g_mean
        else:
            global_rsd[p] = 0.10

    fingerprint_records = []
    sea_areas = ['Bohai Sea', 'Yellow Sea', 'East China Sea', 'South China Sea']

    for sea in sea_areas:
        df_sea = df[df['District I'].str.contains(sea, na=False, case=False)]
        n_samples_total = len(df_sea)

        record = {
            'Sea_Area': sea,
            'Sample_Count': n_samples_total
        }

        if n_samples_total == 0:
            record['Data_Status'] = 'No_Data'
            for p in pfas_cols:
                record[f'{p}_Mean'] = 0.0
                record[f'{p}_Std'] = 0.0
        else:
            record['Data_Status'] = 'Independent_Statistics'
            for p in pfas_cols:
                # Isolate valid data for specific compound
                valid_data = df_sea[p].dropna()
                n_valid = len(valid_data)

                if n_valid == 0:
                    mean_val = 0.0
                    std_val = 0.0
                else:
                    mean_val = valid_data.mean()
                    if n_valid >= 3:
                        std_val = valid_data.std()
                        if pd.isna(std_val):
                            std_val = mean_val * 0.10
                    else:
                        std_val = mean_val * global_rsd[p]

                record[f'{p}_Mean'] = mean_val
                record[f'{p}_Std'] = std_val

        fingerprint_records.append(record)

    df_fingerprints = pd.DataFrame(fingerprint_records)

    # Element-wise Adjacent Basin Borrowing to prevent full-matrix sparsity
    borrowing_rules = {
        'South China Sea': 'East China Sea'
    }

    for target_sea, donor_sea in borrowing_rules.items():
        target_mask = df_fingerprints['Sea_Area'] == target_sea
        donor_mask = df_fingerprints['Sea_Area'] == donor_sea

        if target_mask.sum() > 0 and donor_mask.sum() > 0:
            target_idx = df_fingerprints[target_mask].index[0]
            donor_idx = df_fingerprints[donor_mask].index[0]

            borrowed_any = False
            for p in pfas_cols:
                if df_fingerprints.loc[target_idx, f'{p}_Mean'] == 0.0 and df_fingerprints.loc[
                    target_idx, f'{p}_Std'] == 0.0:
                    if df_fingerprints.loc[donor_idx, f'{p}_Mean'] > 0.0:
                        df_fingerprints.loc[target_idx, f'{p}_Mean'] = df_fingerprints.loc[donor_idx, f'{p}_Mean']
                        df_fingerprints.loc[target_idx, f'{p}_Std'] = df_fingerprints.loc[donor_idx, f'{p}_Std']
                        borrowed_any = True

            if borrowed_any:
                current_status = df_fingerprints.loc[target_idx, 'Data_Status']
                if current_status == 'No_Data':
                    df_fingerprints.loc[
                        target_idx, 'Data_Status'] = f'Adjacent_Borrowing_From_{donor_sea.replace(" ", "_")}'
                elif 'Borrowing' not in current_status:
                    df_fingerprints.loc[target_idx, 'Data_Status'] += f'_&_Partial_Borrowing'

    df_fingerprints.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"INFO: Regional fingerprints extracted -> {output_file}")


if __name__ == '__main__':
    project_root = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data")).expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    exogenous_output = output_root / "exogenous"
    exogenous_output.mkdir(parents=True, exist_ok=True)
    ESTUARY_FILE = data_root / 'raw' / 'estuary.xlsx'
    RIVERS_FILE = data_root / 'raw' / '100_Rivers_2018.xlsx'
    STEP1_FILE = exogenous_output / 'Step1_Annual_Runoff_2014_2025.csv'
    MAPPING_OUTPUT = exogenous_output / 'Step2_River_Sea_Mapping.csv'
    FINGERPRINT_OUTPUT = exogenous_output / 'Step2_Regional_Fingerprints.csv'

    build_topology_mapping(ESTUARY_FILE, RIVERS_FILE, STEP1_FILE, MAPPING_OUTPUT)
    extract_regional_fingerprints(ESTUARY_FILE, FINGERPRINT_OUTPUT)
