#!/usr/bin/env python3
"""Audit the existing CMR split without modifying any dataset file.

The report is descriptive. It verifies the saved split against the exact
``grouped_split`` implementation used to create it, then summarizes property
and chemistry coverage. It does not regenerate, rebalance, or score a model on
any split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "data" / "processed" / "cmr_reconstructed"
DEFAULT_REPORT = WORKSPACE_ROOT / "reports" / "split_balance.md"
DEFAULT_TABLE_DIR = WORKSPACE_ROOT / "reports" / "tables"
DEFAULT_JSON = WORKSPACE_ROOT / "reports" / "split_balance.json"
SPLITS = ("train", "val", "test")

EXPECTED_HASHES = {
    "train.csv": "af4b03a5860c76bb629df9fd6b46eb313b15880fc8cc5b601c6dc894ec4a05b1",
    "val.csv": "01666020b7327a108279b4eefea3092ded429809c39e0dc4c11160bbb00d3041",
    "test.csv": "c97e67afd29e93ba1f5427b91d40a52dc21623aba3259b3953cfeee91c60fa0d",
    "audit.csv": "afab4524786ef88c82ee310d36bf9a0b6ccd2011142fc695934522cab2efa804",
}

BOOLEAN_COLUMNS = (
    "mixed_anion",
    "single_anion",
    "generation_scope_compatible",
)
REQUIRED_COLUMNS = (
    "material_id",
    "heat_all",
    "dir_gap",
    "source_combination",
    "source_A_ion",
    "source_B_ion",
    "source_anion",
    "composition_group",
    "source_site_symbols",
    *BOOLEAN_COLUMNS,
)

GAP_BIN_LABELS = ("zero", "(0,1]", "(1,2]", "(2,3]", ">3")
HEAT_BIN_LABELS = (
    "<=-0.50",
    "(-0.50,-0.25]",
    "(-0.25,0)",
    "zero",
    "(0,0.50]",
    "(0.50,1]",
    "(1,2]",
    ">2",
)
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

# Descriptive neighborhoods around targets shipped in demo.py and the batch
# generation script. They are not acceptance thresholds and are never used to
# change split membership.
TARGET_GAP_HALF_WIDTH_EV = 0.25
TARGET_HEAT_HALF_WIDTH_EV_PER_ATOM = 0.05
REQUESTED_TARGETS = (
    {
        "configuration": "demo_halide",
        "source": "demo.py",
        "family": "halide",
        "dir_gap": 2.0,
        "heat_all": -0.10,
        "source_anion": None,
    },
    {
        "configuration": "demo_oxide",
        "source": "demo.py",
        "family": "oxide",
        "dir_gap": 2.5,
        "heat_all": -0.20,
        "source_anion": "O3",
    },
    {
        "configuration": "demo_chalcogenide",
        "source": "demo.py",
        "family": "chalcogenide",
        "dir_gap": 1.5,
        "heat_all": -0.15,
        "source_anion": None,
    },
    {
        "configuration": "demo_nitride",
        "source": "demo.py",
        "family": "nitride",
        "dir_gap": 2.0,
        "heat_all": -0.15,
        "source_anion": "N3",
    },
    {
        "configuration": "batch_halide_1.5",
        "source": "scripts/generate_candidates.py",
        "family": "halide",
        "dir_gap": 1.5,
        "heat_all": -0.10,
        "source_anion": None,
    },
    {
        "configuration": "batch_halide_2.5",
        "source": "scripts/generate_candidates.py",
        "family": "halide",
        "dir_gap": 2.5,
        "heat_all": -0.10,
        "source_anion": None,
    },
    {
        "configuration": "batch_halide_3.5",
        "source": "scripts/generate_candidates.py",
        "family": "halide",
        "dir_gap": 3.5,
        "heat_all": -0.10,
        "source_anion": None,
    },
    {
        "configuration": "batch_oxide_2.0",
        "source": "scripts/generate_candidates.py",
        "family": "oxide",
        "dir_gap": 2.0,
        "heat_all": -0.20,
        "source_anion": "O3",
    },
    {
        "configuration": "batch_oxide_3.5",
        "source": "scripts/generate_candidates.py",
        "family": "oxide",
        "dir_gap": 3.5,
        "heat_all": -0.20,
        "source_anion": "O3",
    },
    {
        "configuration": "batch_chalcogenide_1.0",
        "source": "scripts/generate_candidates.py",
        "family": "chalcogenide",
        "dir_gap": 1.0,
        "heat_all": -0.15,
        "source_anion": None,
    },
    {
        "configuration": "batch_chalcogenide_2.0",
        "source": "scripts/generate_candidates.py",
        "family": "chalcogenide",
        "dir_gap": 2.0,
        "heat_all": -0.15,
        "source_anion": None,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_boolean(value: Any) -> bool:
    """Parse a serialized Boolean without treating non-empty strings as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    token = str(value).strip().casefold()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"Expected an explicit Boolean, got {value!r}")


def gap_bin(value: float) -> str:
    if value == 0:
        return "zero"
    if value < 0:
        raise ValueError(f"Direct gap cannot be negative: {value}")
    if value <= 1:
        return "(0,1]"
    if value <= 2:
        return "(1,2]"
    if value <= 3:
        return "(2,3]"
    return ">3"


def heat_bin(value: float) -> str:
    if value <= -0.50:
        return "<=-0.50"
    if value <= -0.25:
        return "(-0.50,-0.25]"
    if value < 0:
        return "(-0.25,0)"
    if value == 0:
        return "zero"
    if value <= 0.50:
        return "(0,0.50]"
    if value <= 1:
        return "(0.50,1]"
    if value <= 2:
        return "(1,2]"
    return ">2"


def heat_sign(value: float) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def read_frame(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={column: "string" for column in REQUIRED_COLUMNS},
        keep_default_na=False,
    )
    for column in ("heat_all", "dir_gap"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: {column} contains a non-finite value")
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(parse_boolean).astype(bool)
    for column in REQUIRED_COLUMNS:
        if column not in ("heat_all", "dir_gap", *BOOLEAN_COLUMNS):
            frame[column] = frame[column].astype(str)
    if frame["material_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate material_id")
    frame["split"] = split
    return frame


def scope_views(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str, pd.DataFrame]]:
    full = pd.concat([frames[name] for name in SPLITS], ignore_index=True)
    views: list[tuple[str, str, pd.DataFrame]] = []
    for split, frame in [*(list(frames.items())), ("all", full)]:
        views.append(("all", split, frame))
        views.append(
            (
                "generation_scope",
                split,
                frame.loc[frame["generation_scope_compatible"]].copy(),
            )
        )
    return views


def verify_hashes(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for filename, expected in EXPECTED_HASHES.items():
        path = dataset_dir / filename
        actual = sha256_file(path)
        rows.append(
            {
                "file": filename,
                "size_bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected,
                "matches_setup_snapshot": actual == expected,
            }
        )
    failures = [row["file"] for row in rows if not row["matches_setup_snapshot"]]
    if failures:
        raise AssertionError(f"Dataset bytes differ from the setup snapshot: {failures}")
    return rows


def verify_grouping_and_assignments(
    frames: dict[str, pd.DataFrame], manifest: dict[str, Any]
) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from prepare_cmr_dataset import composition_group, grouped_split

    id_sets = {name: set(frame["material_id"]) for name, frame in frames.items()}
    group_sets = {name: set(frame["composition_group"]) for name, frame in frames.items()}
    overlaps: dict[str, dict[str, int]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            label = f"{left}__{right}"
            overlaps[label] = {
                "material_ids": len(id_sets[left] & id_sets[right]),
                "composition_groups": len(group_sets[left] & group_sets[right]),
            }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Cross-split overlap found: {overlaps}")

    full = pd.concat([frames[name] for name in SPLITS], ignore_index=True)
    recalculated_group_mismatches = []
    for row in full.itertuples(index=False):
        symbols = json.loads(row.source_site_symbols)
        _, recalculated = composition_group(symbols)
        if recalculated != row.composition_group:
            recalculated_group_mismatches.append(row.material_id)
    if recalculated_group_mismatches:
        raise AssertionError(
            "Stored composition_group differs from the converter's composition_group "
            f"for {len(recalculated_group_mismatches)} rows"
        )

    split_config = manifest["split"]
    seed = int(split_config["seed"])
    fractions = tuple(float(split_config["requested_fractions"][name]) for name in SPLITS)
    records = full[["material_id", "composition_group"]].to_dict("records")
    expected = grouped_split(records, seed=seed, fractions=fractions)
    actual = dict(zip(full["material_id"], full["split"]))
    mismatches = [material_id for material_id in sorted(actual) if expected[material_id] != actual[material_id]]
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} rows differ from grouped_split(seed={seed}); "
            f"first={mismatches[0]}"
        )
    for split in SPLITS:
        if len(frames[split]) != int(split_config["counts"][split]):
            raise AssertionError(f"{split} row count differs from manifest.json")
        if frames[split]["composition_group"].nunique() != int(
            split_config["composition_groups"][split]
        ):
            raise AssertionError(f"{split} group count differs from manifest.json")

    return {
        "group_key": "composition_group (reduced composition from source site symbols)",
        "recalculated_group_mismatches": 0,
        "algorithm": split_config["algorithm"],
        "assignment_rule_verified": (
            "each seeded-shuffled whole group is assigned to the split with the "
            "largest target_rows-minus-current_rows capacity; ties follow train/val/test order"
        ),
        "seed": seed,
        "requested_fractions": dict(zip(SPLITS, fractions)),
        "assignment_mismatches": 0,
        "overlaps": overlaps,
    }


def count_rows(
    frame: pd.DataFrame,
    category: str,
    ordered_labels: tuple[str, ...],
    classifier: Callable[[float], str],
) -> list[dict[str, Any]]:
    classified = frame[category].map(classifier)
    result = []
    total = len(frame)
    for label in ordered_labels:
        mask = classified == label
        selected = frame.loc[mask]
        result.append(
            {
                "bin": label,
                "rows": int(mask.sum()),
                "fraction": float(mask.mean()) if total else None,
                "composition_groups": int(selected["composition_group"].nunique()),
            }
        )
    return result


def build_tables(frames: dict[str, pd.DataFrame], audit: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    full = pd.concat([frames[name] for name in SPLITS], ignore_index=True)
    total_rows = len(full)
    total_groups = full["composition_group"].nunique()
    tables: dict[str, list[dict[str, Any]]] = {
        "split_summary": [],
        "group_sizes": [],
        "gap_bins": [],
        "heat_quantiles": [],
        "heat_signs": [],
        "heat_histogram": [],
        "chemistry_flags": [],
        "anion_combinations": [],
        "species_coverage": [],
        "category_group_support": [],
        "two_property_occupancy": [],
        "target_windows": [],
        "audit_summary": [],
    }

    for split, frame in [*((name, frames[name]) for name in SPLITS), ("all", full)]:
        group_sizes = frame.groupby("composition_group", sort=True).size()
        tables["split_summary"].append(
            {
                "split": split,
                "rows": len(frame),
                "row_fraction_of_all": len(frame) / total_rows,
                "composition_groups": frame["composition_group"].nunique(),
                "group_fraction_of_all": frame["composition_group"].nunique() / total_groups,
                "minimum_group_size": int(group_sizes.min()),
                "maximum_group_size": int(group_sizes.max()),
                "mean_group_size": float(group_sizes.mean()),
                "median_group_size": float(group_sizes.median()),
            }
        )
        for size, count in group_sizes.value_counts().sort_index().items():
            tables["group_sizes"].append(
                {
                    "split": split,
                    "group_size_rows": int(size),
                    "composition_groups": int(count),
                    "rows_in_these_groups": int(size * count),
                }
            )

    for scope, split, frame in scope_views(frames):
        for row in count_rows(frame, "dir_gap", GAP_BIN_LABELS, gap_bin):
            tables["gap_bins"].append({"scope": scope, "split": split, **row})

        for quantile in QUANTILES:
            tables["heat_quantiles"].append(
                {
                    "scope": scope,
                    "split": split,
                    "quantile": quantile,
                    "heat_all_eV_per_atom": (
                        float(frame["heat_all"].quantile(quantile)) if len(frame) else None
                    ),
                }
            )
        signs = frame["heat_all"].map(heat_sign)
        for label in ("negative", "zero", "positive"):
            selected = frame.loc[signs == label]
            tables["heat_signs"].append(
                {
                    "scope": scope,
                    "split": split,
                    "sign": label,
                    "rows": len(selected),
                    "fraction": len(selected) / len(frame) if len(frame) else None,
                    "composition_groups": selected["composition_group"].nunique(),
                }
            )
        for row in count_rows(frame, "heat_all", HEAT_BIN_LABELS, heat_bin):
            tables["heat_histogram"].append({"scope": scope, "split": split, **row})

        flag_masks = {
            "single_anion": frame["single_anion"],
            "mixed_anion": frame["mixed_anion"],
            "generation_scope_compatible": frame["generation_scope_compatible"],
        }
        for flag, mask in flag_masks.items():
            selected = frame.loc[mask]
            tables["chemistry_flags"].append(
                {
                    "scope": scope,
                    "split": split,
                    "category": flag,
                    "rows": len(selected),
                    "fraction": len(selected) / len(frame) if len(frame) else None,
                    "composition_groups": selected["composition_group"].nunique(),
                }
            )

        grouped_anions = frame.groupby(
            ["source_combination", "source_anion"], sort=True, dropna=False
        )
        for (combination, anion), selected in grouped_anions:
            tables["anion_combinations"].append(
                {
                    "scope": scope,
                    "split": split,
                    "source_combination": combination,
                    "source_anion": anion,
                    "rows": len(selected),
                    "fraction": len(selected) / len(frame) if len(frame) else None,
                    "composition_groups": selected["composition_group"].nunique(),
                }
            )

        for site_column, site in (("source_A_ion", "A"), ("source_B_ion", "B")):
            for species, selected in frame.groupby(site_column, sort=True):
                tables["species_coverage"].append(
                    {
                        "scope": scope,
                        "split": split,
                        "site": site,
                        "species": species,
                        "rows": len(selected),
                        "composition_groups": selected["composition_group"].nunique(),
                    }
                )

        gap_labels = frame["dir_gap"].map(gap_bin)
        heat_labels = frame["heat_all"].map(heat_bin)
        for combination in sorted(frame["source_combination"].unique()):
            family_mask = frame["source_combination"] == combination
            for label in GAP_BIN_LABELS:
                selected = frame.loc[family_mask & (gap_labels == label)]
                tables["category_group_support"].append(
                    {
                        "scope": scope,
                        "split": split,
                        "source_combination": combination,
                        "gap_bin": label,
                        "rows": len(selected),
                        "composition_groups": selected["composition_group"].nunique(),
                        "sparse_under_5_groups": selected["composition_group"].nunique() < 5,
                    }
                )
        for gap_label in GAP_BIN_LABELS:
            for heat_label in HEAT_BIN_LABELS:
                selected = frame.loc[(gap_labels == gap_label) & (heat_labels == heat_label)]
                tables["two_property_occupancy"].append(
                    {
                        "scope": scope,
                        "split": split,
                        "gap_bin": gap_label,
                        "heat_bin": heat_label,
                        "rows": len(selected),
                        "fraction": len(selected) / len(frame) if len(frame) else None,
                        "composition_groups": selected["composition_group"].nunique(),
                    }
                )

    for split, frame in [*((name, frames[name]) for name in SPLITS), ("all", full)]:
        for target in REQUESTED_TARGETS:
            gap_low = target["dir_gap"] - TARGET_GAP_HALF_WIDTH_EV
            gap_high = target["dir_gap"] + TARGET_GAP_HALF_WIDTH_EV
            heat_low = target["heat_all"] - TARGET_HEAT_HALF_WIDTH_EV_PER_ATOM
            heat_high = target["heat_all"] + TARGET_HEAT_HALF_WIDTH_EV_PER_ATOM
            in_window = frame["dir_gap"].between(gap_low, gap_high, inclusive="both") & frame[
                "heat_all"
            ].between(heat_low, heat_high, inclusive="both")
            populations = {
                "all_chemistry": pd.Series(True, index=frame.index),
                "matching_source_family": (
                    frame["source_anion"] == target["source_anion"]
                    if target["source_anion"] is not None
                    else pd.Series(False, index=frame.index)
                ),
                "generation_scope_matching_family": (
                    frame["generation_scope_compatible"]
                    & (frame["source_anion"] == target["source_anion"])
                    if target["source_anion"] is not None
                    else pd.Series(False, index=frame.index)
                ),
            }
            for population, mask in populations.items():
                gap_selected = frame.loc[
                    mask & frame["dir_gap"].between(gap_low, gap_high, inclusive="both")
                ]
                heat_selected = frame.loc[
                    mask & frame["heat_all"].between(heat_low, heat_high, inclusive="both")
                ]
                selected = frame.loc[mask & in_window]
                population_frame = frame.loc[mask]
                tables["target_windows"].append(
                    {
                        "split": split,
                        "configuration": target["configuration"],
                        "source": target["source"],
                        "family": target["family"],
                        "population": population,
                        "target_dir_gap_eV": target["dir_gap"],
                        "gap_window_low_eV": gap_low,
                        "gap_window_high_eV": gap_high,
                        "target_heat_eV_per_atom": target["heat_all"],
                        "heat_window_low_eV_per_atom": heat_low,
                        "heat_window_high_eV_per_atom": heat_high,
                        "population_rows": len(population_frame),
                        "gap_window_rows": len(gap_selected),
                        "gap_window_composition_groups": gap_selected[
                            "composition_group"
                        ].nunique(),
                        "heat_window_rows": len(heat_selected),
                        "heat_window_composition_groups": heat_selected[
                            "composition_group"
                        ].nunique(),
                        "window_rows": len(selected),
                        "window_fraction": (
                            len(selected) / len(population_frame) if len(population_frame) else None
                        ),
                        "window_composition_groups": selected["composition_group"].nunique(),
                    }
                )

    eligible = frames["val"].loc[frames["val"]["generation_scope_compatible"]]
    for name, frame in (("eligible_validation_pool", eligible), ("audit", audit)):
        tables["audit_summary"].append(
            {
                "population": name,
                "rows": len(frame),
                "composition_groups": frame["composition_group"].nunique(),
                "zero_gap_rows": int((frame["dir_gap"] == 0).sum()),
                "positive_gap_rows": int((frame["dir_gap"] > 0).sum()),
                "heat_min_eV_per_atom": float(frame["heat_all"].min()),
                "heat_max_eV_per_atom": float(frame["heat_all"].max()),
                "gap_min_eV": float(frame["dir_gap"].min()),
                "gap_max_eV": float(frame["dir_gap"].max()),
            }
        )
    return tables


def verify_audit(
    frames: dict[str, pd.DataFrame], audit: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from prepare_cmr_dataset import select_audit

    val = frames["val"]
    val_by_id = val.set_index("material_id", drop=False)
    audit_ids = list(audit["material_id"])
    if len(audit_ids) != len(set(audit_ids)):
        raise AssertionError("audit.csv contains duplicate material IDs")
    if not set(audit_ids).issubset(set(val_by_id.index)):
        raise AssertionError("audit.csv is not a validation subset")
    for row in audit.itertuples(index=False):
        source = val_by_id.loc[row.material_id]
        for column in REQUIRED_COLUMNS:
            left = getattr(row, column)
            right = source[column]
            if isinstance(left, float):
                if not math.isclose(left, float(right), rel_tol=0, abs_tol=0):
                    raise AssertionError(f"audit.csv differs from val.csv: {row.material_id} {column}")
            elif left != right:
                raise AssertionError(f"audit.csv differs from val.csv: {row.material_id} {column}")

    eligible = val.loc[val["generation_scope_compatible"]]
    records = eligible.to_dict("records")
    expected_records = select_audit(records, int(manifest["audit"]["requested_size"]))
    expected_ids = [record["material_id"] for record in expected_records]
    if audit_ids != expected_ids:
        raise AssertionError("audit.csv does not match deterministic maximin selection")
    if audit_ids != manifest["audit"]["material_ids"]:
        raise AssertionError("audit.csv material IDs differ from manifest.json")
    if not audit["generation_scope_compatible"].all():
        raise AssertionError("audit.csv contains an out-of-generation-scope record")

    zero = int((audit["dir_gap"] == 0).sum())
    positive = int((audit["dir_gap"] > 0).sum())
    eligible_zero = int((eligible["dir_gap"] == 0).sum())
    eligible_positive = int((eligible["dir_gap"] > 0).sum())
    if (len(audit), zero, positive) != (20, 19, 1):
        raise AssertionError(
            f"Unexpected audit coverage: rows={len(audit)}, zero={zero}, positive={positive}"
        )
    if (len(eligible), eligible_zero, eligible_positive) != (32, 31, 1):
        raise AssertionError(
            "Unexpected eligible validation coverage: "
            f"rows={len(eligible)}, zero={eligible_zero}, positive={eligible_positive}"
        )
    return {
        "selection": "deterministic normalized two-property maximin",
        "source_split": "val",
        "is_exact_validation_subset": True,
        "matches_recomputed_selection": True,
        "audit_rows": len(audit),
        "audit_zero_gap_rows": zero,
        "audit_positive_gap_rows": positive,
        "eligible_validation_rows": len(eligible),
        "eligible_validation_zero_gap_rows": eligible_zero,
        "eligible_validation_positive_gap_rows": eligible_positive,
    }


def write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def table_lookup(
    rows: list[dict[str, Any]], **criteria: Any
) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise KeyError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def render_report(result: dict[str, Any]) -> str:
    tables = result["tables"]
    verification = result["verification"]
    split_rows = []
    for split in (*SPLITS, "all"):
        row = table_lookup(tables["split_summary"], split=split)
        split_rows.append(
            [
                split,
                row["rows"],
                row["row_fraction_of_all"],
                row["composition_groups"],
                row["minimum_group_size"],
                row["maximum_group_size"],
                row["mean_group_size"],
            ]
        )

    group_size_rows = []
    for split in (*SPLITS, "all"):
        by_size = {
            row["group_size_rows"]: row
            for row in tables["group_sizes"]
            if row["split"] == split
        }
        group_size_rows.append(
            [
                split,
                by_size.get(1, {}).get("composition_groups", 0),
                by_size.get(2, {}).get("composition_groups", 0),
            ]
        )

    gap_rows = []
    for scope in ("all", "generation_scope"):
        for split in (*SPLITS, "all"):
            bins = {
                row["bin"]: row
                for row in tables["gap_bins"]
                if row["scope"] == scope and row["split"] == split
            }
            gap_rows.append(
                [
                    scope,
                    split,
                    bins["zero"]["rows"],
                    bins["zero"]["fraction"],
                    sum(bins[label]["rows"] for label in GAP_BIN_LABELS[1:]),
                    *[bins[label]["rows"] for label in GAP_BIN_LABELS[1:]],
                ]
            )

    heat_rows = []
    for scope in ("all", "generation_scope"):
        for split in (*SPLITS, "all"):
            quantiles = {
                row["quantile"]: row["heat_all_eV_per_atom"]
                for row in tables["heat_quantiles"]
                if row["scope"] == scope and row["split"] == split
            }
            signs = {
                row["sign"]: row["rows"]
                for row in tables["heat_signs"]
                if row["scope"] == scope and row["split"] == split
            }
            heat_rows.append(
                [
                    scope,
                    split,
                    quantiles[0.0],
                    quantiles[0.25],
                    quantiles[0.5],
                    quantiles[0.75],
                    quantiles[1.0],
                    signs["negative"],
                    signs["zero"],
                    signs["positive"],
                ]
            )

    chemistry_rows = []
    for split in (*SPLITS, "all"):
        values = {
            row["category"]: row
            for row in tables["chemistry_flags"]
            if row["scope"] == "all" and row["split"] == split
        }
        chemistry_rows.append(
            [
                split,
                values["single_anion"]["rows"],
                values["mixed_anion"]["rows"],
                values["generation_scope_compatible"]["rows"],
                values["generation_scope_compatible"]["composition_groups"],
            ]
        )

    species_rows = []
    for scope in ("all", "generation_scope"):
        for split in (*SPLITS, "all"):
            selected = [
                row
                for row in tables["species_coverage"]
                if row["scope"] == scope and row["split"] == split
            ]
            species_rows.append(
                [
                    scope,
                    split,
                    len({row["species"] for row in selected if row["site"] == "A"}),
                    len({row["species"] for row in selected if row["site"] == "B"}),
                ]
            )

    anion_rows = [
        [
            row["source_combination"],
            row["source_anion"],
            row["rows"],
            row["fraction"],
            row["composition_groups"],
        ]
        for row in tables["anion_combinations"]
        if row["scope"] == "all" and row["split"] == "all"
    ]

    target_rows = []
    for target in REQUESTED_TARGETS:
        generation = table_lookup(
            tables["target_windows"],
            split="all",
            configuration=target["configuration"],
            population="generation_scope_matching_family",
        )
        all_chemistry = table_lookup(
            tables["target_windows"],
            split="all",
            configuration=target["configuration"],
            population="all_chemistry",
        )
        target_rows.append(
            [
                target["configuration"],
                target["family"],
                target["dir_gap"],
                target["heat_all"],
                generation["population_rows"],
                generation["gap_window_rows"],
                generation["heat_window_rows"],
                generation["window_rows"],
                generation["window_composition_groups"],
                all_chemistry["window_rows"],
            ]
        )

    audit = verification["audit"]
    hash_rows = [
        [row["file"], row["size_bytes"], f"`{row['sha256']}`", row["matches_setup_snapshot"]]
        for row in result["input_hashes"]
    ]
    rare_all = sum(
        row["sparse_under_5_groups"]
        for row in tables["category_group_support"]
        if row["scope"] == "all" and row["split"] == "all"
    )
    rare_generation = sum(
        row["sparse_under_5_groups"]
        for row in tables["category_group_support"]
        if row["scope"] == "generation_scope" and row["split"] == "all"
    )
    positive_fractions = {}
    single_fractions = {}
    heat_medians = {}
    generation_positive = {}
    for split in SPLITS:
        zero_row = table_lookup(
            tables["gap_bins"], scope="all", split=split, bin="zero"
        )
        positive_fractions[split] = 1.0 - zero_row["fraction"]
        single_fractions[split] = table_lookup(
            tables["chemistry_flags"], scope="all", split=split, category="single_anion"
        )["fraction"]
        heat_medians[split] = table_lookup(
            tables["heat_quantiles"], scope="all", split=split, quantile=0.5
        )["heat_all_eV_per_atom"]
        generation_bins = [
            row
            for row in tables["gap_bins"]
            if row["scope"] == "generation_scope" and row["split"] == split
        ]
        generation_positive[split] = sum(
            row["rows"] for row in generation_bins if row["bin"] != "zero"
        )
    positive_spread_pp = 100 * (max(positive_fractions.values()) - min(positive_fractions.values()))
    single_spread_pp = 100 * (max(single_fractions.values()) - min(single_fractions.values()))

    return f"""# CMR reconstructed split-balance audit

Generated by `scripts/audit_split_balance.py` from the unchanged reconstructed
CSV files. This report is descriptive; it did not regenerate a split, train a
model, run a model on the test set, or change any dataset byte.

`original_splits_recovered: false`

`dataset_status: CMR-derived MEIDNet-format dataset; newly generated splits`

`pretrained_training_overlap: unknown`

CMR `heat_all` compatibility with the pretrained checkpoint remains provisional.

## Verification result

- Status: **passed**.
- Group key: `{verification['grouping']['group_key']}`.
- Algorithm: `{verification['grouping']['algorithm']}`.
- Assignment rule checked: `{verification['grouping']['assignment_rule_verified']}`.
- Seed: `{verification['grouping']['seed']}` with requested fractions
  `{verification['grouping']['requested_fractions']}`.
- Recomputed group-key mismatches: `0`.
- Recomputed assignment mismatches: `0`.
- Cross-split material-ID overlaps: `0`; cross-split composition-group overlaps: `0`.
- All four input hashes matched the setup snapshot before and after analysis.

{markdown_table(['Input', 'Bytes', 'SHA-256', 'Matches setup'], hash_rows)}

## Rows and composition groups

{markdown_table(['Split', 'Rows', 'Row fraction', 'Groups', 'Min group', 'Max group', 'Mean group'], split_rows)}

{markdown_table(['Split', 'One-row groups', 'Two-row groups'], group_size_rows)}

The full dataset has only one- and two-row reduced-composition groups; the
machine-readable counts are in `reports/tables/split_group_sizes.csv`.

## Direct-gap balance

The bins are fixed at exactly zero, `(0,1]`, `(1,2]`, `(2,3]`, and `>3` eV and
are reused for every split and subset.

{markdown_table(['Scope', 'Split', 'Zero', 'Zero fraction', 'Positive', '(0,1]', '(1,2]', '(2,3]', '>3'], gap_rows)}

A zero `gllbsc_dir_gap` is a legitimate retained source value. It says that
the source calculation reported a zero direct gap; it is **not by itself proof
that the material is metallic**, and this report does not relabel it as such.

## Heat-label balance

Values are the unchanged CMR finite-reference-pool labels in eV/atom. They are
not conventional elemental formation enthalpies or energy above hull, and
checkpoint-label compatibility remains unresolved. Fixed histogram bins are
`<=-0.50`, `(-0.50,-0.25]`, `(-0.25,0)`, exact zero, `(0,0.50]`, `(0.50,1]`,
`(1,2]`, and `>2`; full quantiles, signs, histograms, and group support are in
the numeric tables.

{markdown_table(['Scope', 'Split', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Negative', 'Zero', 'Positive'], heat_rows)}

## Chemistry coverage

{markdown_table(['Split', 'Single anion', 'Mixed anion', 'Generation scope', 'Generation groups'], chemistry_rows)}

{markdown_table(['Scope', 'Split', 'Unique A species', 'Unique B species'], species_rows)}

{markdown_table(['Source family', 'Source anion', 'Rows', 'Fraction', 'Groups'], anion_rows)}

The A- and B-site species counts and composition-group support are in
`reports/tables/split_species_coverage.csv`. Gap-by-family group support is in
`split_category_group_support.csv`: {rare_all} full-data cells and
{rare_generation} generation-scope cells have fewer than five supporting
composition groups. Empty or sparse cells are retained as evidence of limited
support rather than filled or merged after inspecting model behavior.

## Two-property occupancy and requested-target neighborhoods

`split_two_property_occupancy.csv` crosses the fixed gap and heat bins and
reports both row counts and unique composition-group support. Target windows
below are descriptive neighborhoods around the unmodified targets shipped in
`MEIDNet-main/demo.py` and `MEIDNet-main/scripts/generate_candidates.py`: gap target +/-{TARGET_GAP_HALF_WIDTH_EV} eV and heat target
+/-{TARGET_HEAT_HALF_WIDTH_EV_PER_ATOM} eV/atom. They are not acceptance
tolerances and were not used to change membership. CMR has no matching
single-anion halide or chalcogenide family, so those matching-family populations
are explicitly zero.

{markdown_table(['Configuration', 'Family', 'Gap target', 'Heat target', 'In-scope family rows', 'Gap-window rows', 'Heat-window rows', 'Joint-window rows', 'Joint-window groups', 'All-chem joint rows'], target_rows)}

## Audit sample verification

- The eligible validation pool has **{audit['eligible_validation_rows']}** rows:
  **{audit['eligible_validation_zero_gap_rows']}** zero direct gaps and
  **{audit['eligible_validation_positive_gap_rows']}** positive direct gap.
- `audit.csv` has **{audit['audit_rows']}** rows: **{audit['audit_zero_gap_rows']}**
  zero and **{audit['audit_positive_gap_rows']}** positive direct gap.
- Every audit ID and value is an exact validation-row subset, all rows are in
  generation scope, and recomputing the converter's normalized two-property
  maximin rule reproduces the saved ordered IDs.

This audit sample was intentionally selected for property coverage from the
eligible validation pool. It is not a random sample, a representative sample,
or evidence that the underlying split was stratified.

## Assessment and later split proposal

The saved split is group-aware and exactly reproducible, but it is **not
property-stratified**. At full-dataset scale the positive-gap fractions are
{positive_fractions['train']:.4%} train, {positive_fractions['val']:.4%}
validation, and {positive_fractions['test']:.4%} test (a {positive_spread_pp:.3f}
percentage-point range), while heat medians are {heat_medians['train']:.2f},
{heat_medians['val']:.2f}, and {heat_medians['test']:.2f} eV/atom. Those broad
property summaries are close. Chemistry balance is looser: single-anion
fractions are {single_fractions['train']:.4%}, {single_fractions['val']:.4%},
and {single_fractions['test']:.4%} (a {single_spread_pp:.3f} percentage-point
range). Most importantly, positive-gap generation-scope support is only
{generation_positive['train']}/{generation_positive['val']}/{generation_positive['test']}
rows in train/validation/test. That sparse and asymmetric support limits
property-coverage claims, especially for the one positive eligible validation
example, but it does not invalidate the current files for interface and decoder
diagnostics. Aggregate test statistics are reported only as dataset facts; no
test example was decoded and no seed, threshold, or model was selected from
test performance.

For a future, separately versioned training experiment, a grouped-stratified
alternative could assign whole reduced-composition groups while approximately
balancing: (1) exact-zero versus positive direct gap and the fixed positive-gap
bins, (2) the fixed heat bins, (3) source anion family, and (4) generation-scope
compatibility. Assignment should optimize row and stratum target deficits with
a fixed seed after sorting group IDs. Strata with too few independent groups
for all splits must be merged before assignment or marked unsupported. Groups
must never be split, and no positive example should be manufactured. This is a
proposal only; the existing assignments were not overwritten or regenerated.

## Machine-readable outputs

- `reports/split_balance.json`
- `reports/tables/split_input_hashes.csv`
- `reports/tables/split_summary.csv`
- `reports/tables/split_group_sizes.csv`
- `reports/tables/split_gap_bins.csv`
- `reports/tables/split_heat_quantiles.csv`
- `reports/tables/split_heat_signs.csv`
- `reports/tables/split_heat_histogram.csv`
- `reports/tables/split_chemistry_flags.csv`
- `reports/tables/split_anion_combinations.csv`
- `reports/tables/split_species_coverage.csv`
- `reports/tables/split_category_group_support.csv`
- `reports/tables/split_two_property_occupancy.csv`
- `reports/tables/split_target_windows.csv`
- `reports/tables/split_audit_summary.csv`
"""


def audit_dataset(dataset_dir: Path) -> dict[str, Any]:
    input_hashes = verify_hashes(dataset_dir)
    with (dataset_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = {name: read_frame(dataset_dir / f"{name}.csv", name) for name in SPLITS}
    audit = read_frame(dataset_dir / "audit.csv", "audit")

    grouping = verify_grouping_and_assignments(frames, manifest)
    audit_verification = verify_audit(frames, audit, manifest)
    tables = build_tables(frames, audit)
    after_hashes = verify_hashes(dataset_dir)
    if [row["sha256"] for row in input_hashes] != [row["sha256"] for row in after_hashes]:
        raise AssertionError("A dataset input changed while it was being audited")

    return {
        "status": "passed",
        "dataset_dir": str(dataset_dir.resolve()),
        "provenance": {
            "original_splits_recovered": False,
            "dataset_status": "CMR-derived MEIDNet-format dataset; newly generated splits",
            "pretrained_training_overlap": "unknown",
            "property_checkpoint_compatibility": "provisional",
        },
        "input_hashes": input_hashes,
        "input_hashes_unchanged_during_audit": True,
        "verification": {"grouping": grouping, "audit": audit_verification},
        "tables": tables,
        "test_set_use": "aggregate dataset statistics only; no model execution or selection",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    result = audit_dataset(dataset_dir)

    table_files = {"input_hashes": result["input_hashes"], **result["tables"]}
    for name, rows in table_files.items():
        filename = f"{name}.csv" if name.startswith("split_") else f"split_{name}.csv"
        write_csv_table(args.table_dir.resolve() / filename, rows)

    args.json_report.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.json_report.resolve().open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(render_report(result), encoding="utf-8", newline="\n")
    total_rows = table_lookup(result["tables"]["split_summary"], split="all")["rows"]
    print(f"Split-balance audit passed: {total_rows} rows")
    print(f"Report: {args.report.resolve()}")
    print(f"Machine-readable JSON: {args.json_report.resolve()}")
    print(f"Tables: {args.table_dir.resolve()}")


if __name__ == "__main__":
    main()
