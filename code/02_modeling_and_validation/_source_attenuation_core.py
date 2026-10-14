from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfas_schema import MODEL_GROUPS, schema_digest

VERSION = "v1.3"
TAXONOMY = MODEL_GROUPS
SOURCE_COLUMNS = (
    "Grid_Lon",
    "Grid_Lat",
    "Load_Total_Legacy_Kg",
    "Source_Ratio_SC_LC",
    "Source_Ratio_LC_PFSA_PFCA",
    "Source_Ratio_SC_PFSA_PFCA",
)
LEGACY_MODES = {"reference", "synchronized", "class_resolved"}
SHORT_MODES = {
    "reference",
    "observed_transition",
    "post_plateau_common",
    "post_plateau_class_resolved",
}
APPLICATIONS = ("OCPSF", "MF", "AFFFs", "Txl", "PI", "PPP")


def flux_column(compound: str) -> str:
    return f"{compound}_Flux_Kg"


def all_compounds() -> tuple[str, ...]:
    return tuple(compound for compounds in TAXONOMY.values() for compound in compounds)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_configuration(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        configuration = json.load(stream)
    evidence = configuration.get("evidence", {})
    required_evidence = {
        "usage_data_file",
        "compound_mapping_file",
        "release_factor_file",
        "reference_year",
        "comparison_year",
        "multiplier_basis",
    }
    missing_evidence = sorted(required_evidence - set(evidence))
    if missing_evidence:
        raise ValueError(
            f"Evidence configuration is missing fields: {missing_evidence}"
        )
    if evidence["multiplier_basis"] != "industrial_use_mass_ratio":
        raise ValueError(
            "The primary source-boundary multiplier must be industrial_use_mass_ratio"
        )
    if int(evidence["comparison_year"]) <= int(evidence["reference_year"]):
        raise ValueError("comparison_year must be later than reference_year")
    scenarios = configuration.get("scenarios", [])
    identifiers = [item.get("scenario_id") for item in scenarios]
    if (
        not identifiers
        or any(not item for item in identifiers)
        or len(identifiers) != len(set(identifiers))
    ):
        raise ValueError("Scenarios require unique non-empty identifiers")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in identifiers):
        raise ValueError("Scenario identifiers must be safe file-name tokens")
    references = [item for item in scenarios if item["scenario_id"] == "reference"]
    if len(references) != 1 or any(
        references[0].get(key) != "reference" for key in ("legacy_mode", "short_mode")
    ):
        raise ValueError(
            "The reference scenario must leave both source classes unchanged"
        )
    for scenario in scenarios:
        if scenario.get("legacy_mode") not in LEGACY_MODES:
            raise ValueError(f"{scenario['scenario_id']}: invalid legacy_mode")
        if scenario.get("short_mode") not in SHORT_MODES:
            raise ValueError(f"{scenario['scenario_id']}: invalid short_mode")
        intensity = float(scenario.get("transition_intensity", 1.0))
        if not np.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError(
                f"{scenario['scenario_id']}: transition_intensity must be between zero and one"
            )
    contrasts = configuration.get("contrasts", [])
    contrast_ids = [item.get("contrast_id") for item in contrasts]
    if any(not item for item in contrast_ids) or len(contrast_ids) != len(
        set(contrast_ids)
    ):
        raise ValueError("Contrasts require unique non-empty identifiers")
    scenario_ids = set(identifiers)
    for contrast in contrasts:
        if (
            contrast.get("scenario_id") not in scenario_ids
            or contrast.get("comparator_id") not in scenario_ids
        ):
            raise ValueError(
                f"{contrast.get('contrast_id')}: contrast scenarios are undefined"
            )
    return configuration


def evidence_path(configuration_path: Path, value: str) -> Path:
    return (configuration_path.resolve().parent / value).resolve()


def load_usage_evidence(
    configuration: dict, configuration_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = configuration["evidence"]
    usage = pd.read_csv(evidence_path(configuration_path, evidence["usage_data_file"]))
    mapping = pd.read_csv(
        evidence_path(configuration_path, evidence["compound_mapping_file"])
    )
    release = pd.read_csv(
        evidence_path(configuration_path, evidence["release_factor_file"])
    )
    years = [
        str(year)
        for year in range(
            int(evidence["reference_year"]), int(evidence["comparison_year"]) + 1
        )
    ]
    required_usage = {"compound", "application", *years}
    if not required_usage.issubset(usage.columns):
        raise ValueError(
            f"Table S8 data are missing columns: {sorted(required_usage - set(usage.columns))}"
        )
    if usage.duplicated(["compound", "application"]).any():
        raise ValueError("Table S8 contains duplicate compound-application rows")
    compounds = set(usage["compound"])
    expected_pairs = pd.MultiIndex.from_product([compounds, APPLICATIONS])
    observed_pairs = pd.MultiIndex.from_frame(usage[["compound", "application"]])
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != set(
        expected_pairs
    ):
        raise ValueError(
            "Table S8 must contain one row for every compound-application combination"
        )
    values = usage[years].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any().any():
        raise ValueError("Table S8 use masses must be finite and non-negative")
    usage[years] = values
    if mapping["compound"].duplicated().any() or set(mapping["compound"]) != compounds:
        raise ValueError(
            "The GDR mapping must contain each Table S8 compound exactly once"
        )
    if set(mapping["gdr_group"]) != set(TAXONOMY):
        raise ValueError("The GDR mapping must contain all four source groups")
    if release["application"].duplicated().any() or set(release["application"]) != set(
        APPLICATIONS
    ):
        raise ValueError(
            "The release-factor table must contain each application exactly once"
        )
    for column in (
        "total_environment_release_fraction",
        "water_release_fraction_of_use",
    ):
        release[column] = pd.to_numeric(release[column], errors="coerce")
        non_missing = release[column].dropna()
        if ((non_missing < 0) | (non_missing > 1)).any():
            raise ValueError(f"{column} must be between zero and one")
    return usage, mapping, release


def transition_metric(
    reference: float, comparison: float, interval_years: int
) -> tuple[float, float]:
    if reference <= 0 or comparison <= 0:
        raise ValueError("Transition metrics require positive endpoint values")
    multiplier = comparison / reference
    annual_rate_percent = 100.0 * (multiplier ** (1.0 / interval_years) - 1.0)
    return multiplier, annual_rate_percent


def derive_evidence_tables(
    usage: pd.DataFrame,
    mapping: pd.DataFrame,
    release: pd.DataFrame,
    reference_year: int,
    comparison_year: int,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    year_columns = [column for column in usage.columns if str(column).isdigit()]
    long = usage.melt(
        id_vars=["compound", "application"],
        value_vars=year_columns,
        var_name="year",
        value_name="industrial_use_mass_t",
    )
    long["year"] = long["year"].astype(int)
    long = long.merge(mapping, on="compound", how="left", validate="many_to_one")
    long = long.merge(release, on="application", how="left", validate="many_to_one")
    long["water_factor_available"] = long["water_release_fraction_of_use"].notna()
    long["use_mass_with_water_factor_t"] = np.where(
        long["water_factor_available"], long["industrial_use_mass_t"], 0.0
    )
    long["attributed_water_release_t"] = (
        long["industrial_use_mass_t"] * long["water_release_fraction_of_use"]
    )

    compound_annual = long.groupby(
        ["compound", "gdr_group", "year"], as_index=False
    ).agg(
        industrial_use_mass_t=("industrial_use_mass_t", "sum"),
        use_mass_with_water_factor_t=("use_mass_with_water_factor_t", "sum"),
        attributed_water_release_t=(
            "attributed_water_release_t",
            lambda values: values.sum(min_count=1),
        ),
    )
    compound_annual["water_factor_use_coverage"] = (
        compound_annual["use_mass_with_water_factor_t"]
        / compound_annual["industrial_use_mass_t"]
    )
    group_annual = compound_annual.groupby(["gdr_group", "year"], as_index=False).agg(
        industrial_use_mass_t=("industrial_use_mass_t", "sum"),
        use_mass_with_water_factor_t=("use_mass_with_water_factor_t", "sum"),
        attributed_water_release_t=("attributed_water_release_t", "sum"),
    )
    group_annual["water_factor_use_coverage"] = (
        group_annual["use_mass_with_water_factor_t"]
        / group_annual["industrial_use_mass_t"]
    )

    interval = comparison_year - reference_year
    metric_rows = []
    for level, frame, identifier in (
        ("compound", compound_annual, "compound"),
        ("gdr_group", group_annual, "gdr_group"),
    ):
        for name, subset in frame.groupby(identifier, sort=True):
            reference = subset.loc[subset["year"] == reference_year].iloc[0]
            comparison = subset.loc[subset["year"] == comparison_year].iloc[0]
            use_multiplier, use_cagr = transition_metric(
                float(reference["industrial_use_mass_t"]),
                float(comparison["industrial_use_mass_t"]),
                interval,
            )
            water_multiplier, water_cagr = transition_metric(
                float(reference["attributed_water_release_t"]),
                float(comparison["attributed_water_release_t"]),
                interval,
            )
            metric_rows.append(
                {
                    "aggregation_level": level,
                    "identifier": name,
                    "reference_year": reference_year,
                    "comparison_year": comparison_year,
                    "reference_use_mass_t": float(reference["industrial_use_mass_t"]),
                    "comparison_use_mass_t": float(comparison["industrial_use_mass_t"]),
                    "use_mass_multiplier": use_multiplier,
                    "use_mass_cagr_percent": use_cagr,
                    "reference_attributed_water_release_t": float(
                        reference["attributed_water_release_t"]
                    ),
                    "comparison_attributed_water_release_t": float(
                        comparison["attributed_water_release_t"]
                    ),
                    "attributed_water_release_multiplier": water_multiplier,
                    "attributed_water_release_cagr_percent": water_cagr,
                    "reference_water_factor_use_coverage": float(
                        reference["water_factor_use_coverage"]
                    ),
                    "comparison_water_factor_use_coverage": float(
                        comparison["water_factor_use_coverage"]
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    group_metrics = metrics[metrics["aggregation_level"] == "gdr_group"].set_index(
        "identifier"
    )
    multipliers = group_metrics["use_mass_multiplier"].to_dict()
    long_reference = group_metrics.loc[
        ["Long_PFSA", "Long_PFCA"], "reference_use_mass_t"
    ].sum()
    long_comparison = group_metrics.loc[
        ["Long_PFSA", "Long_PFCA"], "comparison_use_mass_t"
    ].sum()
    multipliers["Long_Synchronized"] = float(long_comparison / long_reference)
    return {
        "application_annual": long,
        "compound_annual": compound_annual,
        "group_annual": group_annual,
        "metrics": metrics,
        "multipliers": multipliers,
    }


def validate_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Grid_Lon",
        "Grid_Lat",
        *(flux_column(compound) for compound in all_compounds()),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Compound source inventory is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Compound source inventory contains no records")
    validated = frame.copy()
    for column in required:
        validated[column] = pd.to_numeric(validated[column], errors="raise")
    values = validated[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Compound source inventory contains non-finite required values"
        )
    compound_columns = [flux_column(compound) for compound in all_compounds()]
    if (validated[compound_columns] < 0).any().any():
        raise ValueError("Compound fluxes must be non-negative")
    if validated.duplicated(["Grid_Lon", "Grid_Lat"]).any():
        raise ValueError("Grid_Lon and Grid_Lat must uniquely identify source nodes")
    return validated.sort_values(["Grid_Lat", "Grid_Lon"]).reset_index(drop=True)


def group_masses(frame: pd.DataFrame) -> pd.DataFrame:
    groups = pd.DataFrame(index=frame.index)
    for group, compounds in TAXONOMY.items():
        groups[group] = frame[[flux_column(compound) for compound in compounds]].sum(
            axis=1
        )
    groups["Long_Total"] = groups["Long_PFSA"] + groups["Long_PFCA"]
    groups["Short_Total"] = groups["Short_PFSA"] + groups["Short_PFCA"]
    groups["Legacy_Total"] = groups["Long_Total"] + groups["Short_Total"]
    return groups


def safe_ratio(numerator: pd.Series, denominator: pd.Series, label: str) -> pd.Series:
    if (
        not np.isfinite(numerator.to_numpy(dtype=float)).all()
        or not np.isfinite(denominator.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"{label} contains non-finite component masses")
    if (denominator <= 0).any():
        raise ValueError(
            f"{label} is undefined for {int((denominator <= 0).sum())} source nodes"
        )
    ratio = numerator / denominator
    if not np.isfinite(ratio.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite ratios")
    return ratio


def aggregate_source_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = group_masses(frame)
    source = frame[["Grid_Lon", "Grid_Lat"]].copy()
    source["Load_Total_Legacy_Kg"] = groups["Legacy_Total"]
    source["Source_Ratio_SC_LC"] = safe_ratio(
        groups["Short_Total"], groups["Long_Total"], "SC/LC"
    )
    source["Source_Ratio_LC_PFSA_PFCA"] = safe_ratio(
        groups["Long_PFSA"], groups["Long_PFCA"], "LC PFSA/PFCA"
    )
    source["Source_Ratio_SC_PFSA_PFCA"] = safe_ratio(
        groups["Short_PFSA"], groups["Short_PFCA"], "SC PFSA/PFCA"
    )
    return source, groups


def scenario_multipliers(
    scenario: dict, evidence: dict[str, float]
) -> dict[str, float]:
    multipliers = {group: 1.0 for group in TAXONOMY}
    legacy_mode = scenario["legacy_mode"]
    short_mode = scenario["short_mode"]
    if legacy_mode == "synchronized":
        multipliers["Long_PFSA"] = evidence["Long_Synchronized"]
        multipliers["Long_PFCA"] = evidence["Long_Synchronized"]
    elif legacy_mode == "class_resolved":
        multipliers["Long_PFSA"] = evidence["Long_PFSA"]
        multipliers["Long_PFCA"] = evidence["Long_PFCA"]
    if short_mode == "observed_transition":
        multipliers["Short_PFSA"] = evidence["Short_PFSA"]
        multipliers["Short_PFCA"] = evidence["Short_PFCA"]
    elif short_mode == "post_plateau_common":
        multipliers["Short_PFSA"] = (
            evidence["Short_PFSA"] * evidence["Long_Synchronized"]
        )
        multipliers["Short_PFCA"] = (
            evidence["Short_PFCA"] * evidence["Long_Synchronized"]
        )
    elif short_mode == "post_plateau_class_resolved":
        multipliers["Short_PFSA"] = evidence["Short_PFSA"] * evidence["Long_PFSA"]
        multipliers["Short_PFCA"] = evidence["Short_PFCA"] * evidence["Long_PFCA"]
    intensity = float(scenario.get("transition_intensity", 1.0))
    return {
        group: float(multiplier**intensity) for group, multiplier in multipliers.items()
    }


def apply_multipliers(
    frame: pd.DataFrame, multipliers: dict[str, float]
) -> pd.DataFrame:
    result = frame.copy()
    for group, compounds in TAXONOMY.items():
        result[[flux_column(compound) for compound in compounds]] *= multipliers[group]
    return result


def relative_change(new: float, old: float) -> float | None:
    return None if old == 0 else 100.0 * (new / old - 1.0)


def evidence_coverage(frame: pd.DataFrame, mapping: pd.DataFrame) -> dict[str, float]:
    coverage = {}
    for group in TAXONOMY:
        covered_compounds = mapping.loc[
            mapping["gdr_group"] == group, "compound"
        ].tolist()
        covered = (
            frame[[flux_column(compound) for compound in covered_compounds]]
            .to_numpy()
            .sum()
        )
        total = (
            frame[[flux_column(compound) for compound in TAXONOMY[group]]]
            .to_numpy()
            .sum()
        )
        coverage[group] = float(covered / total) if total > 0 else 0.0
    return coverage


def summarize(
    scenario: dict,
    multipliers: dict[str, float],
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    baseline_source: pd.DataFrame,
    source: pd.DataFrame,
    baseline_groups: pd.DataFrame,
    groups: pd.DataFrame,
) -> dict:
    row = {
        "scenario_id": scenario["scenario_id"],
        "analysis_role": scenario.get("analysis_role", "sensitivity"),
        "claim_scope": scenario.get("claim_scope", ""),
        "transition_intensity": float(scenario.get("transition_intensity", 1.0)),
        "legacy_mode": scenario["legacy_mode"],
        "short_mode": scenario["short_mode"],
        "source_nodes": int(len(baseline)),
    }
    for group, multiplier in multipliers.items():
        row[f"{group.lower()}_multiplier"] = multiplier
    for group in (*TAXONOMY, "Long_Total", "Short_Total", "Legacy_Total"):
        before = float(baseline_groups[group].sum())
        after = float(groups[group].sum())
        row[f"{group.lower()}_base_kg"] = before
        row[f"{group.lower()}_perturbed_kg"] = after
        row[f"{group.lower()}_change_percent"] = relative_change(after, before)
    for column in SOURCE_COLUMNS[3:]:
        before = float(baseline_source[column].median())
        after = float(source[column].median())
        label = column.replace("Source_Ratio_", "").lower()
        row[f"median_{label}_base"] = before
        row[f"median_{label}_perturbed"] = after
        row[f"median_{label}_change_percent"] = relative_change(after, before)
    compound_total = perturbed[
        [flux_column(compound) for compound in all_compounds()]
    ].sum(axis=1)
    row["maximum_mass_closure_error_kg"] = float(
        np.max(
            np.abs(
                source["Load_Total_Legacy_Kg"].to_numpy() - compound_total.to_numpy()
            )
        )
    )
    return row


def run(
    compound_inventory_path: Path,
    configuration_path: Path,
    output_directory: Path,
    overwrite: bool,
) -> None:
    compound_inventory_path = compound_inventory_path.resolve()
    configuration_path = configuration_path.resolve()
    output_directory = output_directory.resolve()
    manifest_path = output_directory / "Legacy_Class_Transition_Manifest.json"
    baseline = validate_inventory(pd.read_csv(compound_inventory_path))
    configuration = load_configuration(configuration_path)
    protected_inputs = [compound_inventory_path, configuration_path] + [
        evidence_path(configuration_path, configuration["evidence"][key])
        for key in ("usage_data_file", "compound_mapping_file", "release_factor_file")
    ]
    if any(
        path == output_directory or output_directory in path.parents
        for path in protected_inputs
    ):
        raise ValueError("Scenario output directory must not contain any input file")
    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. Use --overwrite explicitly"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    usage, mapping, release = load_usage_evidence(configuration, configuration_path)
    reference_year = int(configuration["evidence"]["reference_year"])
    comparison_year = int(configuration["evidence"]["comparison_year"])
    derived = derive_evidence_tables(
        usage, mapping, release, reference_year, comparison_year
    )
    evidence = derived["multipliers"]
    coverage = evidence_coverage(baseline, mapping)
    baseline_source, baseline_groups = aggregate_source_table(baseline)
    summaries = []
    evidence_products = {
        "application_annual": output_directory
        / "Cai_Table_S8_Application_Annual_Calculation.csv",
        "compound_annual": output_directory
        / "Cai_Table_S8_Compound_Annual_Calculation.csv",
        "group_annual": output_directory
        / "Cai_Table_S8_GDR_Group_Annual_Calculation.csv",
        "metrics": output_directory / "Cai_Table_S8_Transition_Metrics.csv",
        "release_factors": output_directory
        / "Cai_Application_Release_Factor_Audit.csv",
    }
    for key in ("application_annual", "compound_annual", "group_annual", "metrics"):
        derived[key].to_csv(evidence_products[key], index=False, encoding="utf-8-sig")
    release.to_csv(
        evidence_products["release_factors"], index=False, encoding="utf-8-sig"
    )
    products = [path.name for path in evidence_products.values()]

    for scenario in configuration["scenarios"]:
        multipliers = scenario_multipliers(scenario, evidence)
        perturbed = apply_multipliers(baseline, multipliers)
        source, groups = aggregate_source_table(perturbed)
        if scenario["legacy_mode"] == "synchronized" and not np.allclose(
            source["Source_Ratio_LC_PFSA_PFCA"],
            baseline_source["Source_Ratio_LC_PFSA_PFCA"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"{scenario['scenario_id']}: synchronized attenuation changed a source-node long-chain ratio"
            )
        summary = summarize(
            scenario,
            multipliers,
            baseline,
            perturbed,
            baseline_source,
            source,
            baseline_groups,
            groups,
        )
        identifier = scenario["scenario_id"]
        compound_path = output_directory / f"Compound_Source_Flux_{identifier}.csv"
        source_path = output_directory / f"Coastline_Source_Load_{identifier}.csv"
        perturbed.insert(0, "Scenario_ID", identifier)
        perturbed.to_csv(compound_path, index=False, encoding="utf-8-sig")
        source.to_csv(source_path, index=False, encoding="utf-8-sig")
        summary.update(
            {
                "compound_source_file": compound_path.name,
                "compound_source_sha256": sha256(compound_path),
                "aggregated_source_file": source_path.name,
                "aggregated_source_sha256": sha256(source_path),
            }
        )
        summaries.append(summary)
        products.extend([compound_path.name, source_path.name])
        print(
            f"{identifier}: long PFSA {summary['long_pfsa_change_percent']:+.2f}%; "
            f"long PFCA {summary['long_pfca_change_percent']:+.2f}%; "
            f"short total {summary['short_total_change_percent']:+.2f}%"
        )

    summary_path = output_directory / "Legacy_Class_Transition_Summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    if max(row["maximum_mass_closure_error_kg"] for row in summaries) > 1e-10:
        raise ValueError("Mass closure tolerance was exceeded")
    manifest = {
        "script_version": VERSION,
        "pfas_schema_sha256": schema_digest(),
        "model_groups": {key: list(value) for key, value in TAXONOMY.items()},
        "analysis_label": "Literature-scaled PFAS class-transition experiment",
        "claim_scope": (
            "Observed 2015-2022 Chinese group-level industrial-use mass ratios are transferred to the 2018 "
            "riverine source boundary under a stable use-to-river-release proportionality assumption. Outputs "
            "are counterfactual perturbations, not absolute emission estimates or calendar-year forecasts. "
            "Application-weighted water releases exclude textile and pharmaceutical-intermediate use because "
            "their reported release factors are not allocated to water; those estimates are diagnostic only."
        ),
        "calculation_chain": [
            "Read compound-by-application annual use masses from Cai et al. Table S8",
            "Sum the six applications to obtain each compound's annual use mass",
            "Sum mapped compounds to obtain each GDR group's annual use mass",
            "Divide comparison-year group mass by reference-year group mass",
            "Apply the resulting dimensionless ratio to every inventory compound in that GDR group",
        ],
        "group_aggregation_rule": (
            "GDR group multipliers are ratios of summed group masses. Compound-specific growth rates are not averaged."
        ),
        "use_to_source_equation": (
            "river_source_group_scenario = river_source_group_baseline * "
            "(group_use_mass_comparison_year / group_use_mass_reference_year)"
        ),
        "water_release_diagnostic_equation": (
            "attributed_water_release = sum(application_use_mass * explicit_water_release_fraction_of_use)"
        ),
        "compound_inventory": str(compound_inventory_path),
        "compound_inventory_sha256": sha256(compound_inventory_path),
        "configuration": str(configuration_path),
        "configuration_sha256": sha256(configuration_path),
        "raw_evidence_files": {
            key: {
                "path": str(
                    evidence_path(configuration_path, configuration["evidence"][key])
                ),
                "sha256": sha256(
                    evidence_path(configuration_path, configuration["evidence"][key])
                ),
            }
            for key in (
                "usage_data_file",
                "compound_mapping_file",
                "release_factor_file",
            )
        },
        "evidence_multipliers": evidence,
        "scenario_definitions": configuration["scenarios"],
        "contrasts": configuration.get("contrasts", []),
        "baseline_flux_fraction_covered_by_evidence": coverage,
        "products": products + [summary_path.name],
        "product_sha256": {
            name: sha256(output_directory / name)
            for name in products + [summary_path.name]
        },
    }
    write_json(manifest_path, manifest)
    print(f"Transition products written to {output_directory}")


def parse_arguments() -> argparse.Namespace:
    project_root = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs"))
        .expanduser()
        .resolve()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compound-inventory",
        type=Path,
        required=True,
        help="Current Step3 compound flux CSV; no frozen-input fallback",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root
        / "code"
        / "02_modeling_and_validation"
        / "config"
        / "source_attenuation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "02_source_attenuation_boundary",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run(
        arguments.compound_inventory,
        arguments.config,
        arguments.output_dir,
        arguments.overwrite,
    )
