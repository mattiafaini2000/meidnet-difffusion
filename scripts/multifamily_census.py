"""Count CMR source conformity under frozen multifamily_v1 policy scopes.

This is a source-data census. It never loads a model or encodes the test split.
The raw ASE database supplies source geometry; the existing CSVs supply the
unchanged split, labels, roles, and verified CIF/parser correspondence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase.db import connect
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "MEIDNet-main"))
sys.path.insert(0, str(ROOT / "scripts"))

from meidnet import design as legacy  # noqa: E402
from audit_template_reachability import oracle_species_record  # noqa: E402


POLICY_SCHEMA_VERSION = "multifamily_v1"
DEFAULT_DATA = ROOT / "data" / "processed" / "cmr_reconstructed"
DEFAULT_RAW_DB = ROOT / "data" / "raw" / "cmr" / "cubic_perovskites.db"
DEFAULT_OUTPUT = ROOT / "reports" / "multifamily_rule_relaxation"
SPLITS = ("train", "val", "test")
ANION_ORDER = ("O", "F", "N", "S", "Cl", "Br", "I", "Se", "Te")
HALIDES = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"S", "Se", "Te"}
HOMOGENEOUS = {
    "O": ("oxide", "homogeneous_oxide", "oxide"),
    "N": ("nitride", "homogeneous_nitride", "nitride"),
    "F": ("fluoride", "homogeneous_halide", "halide"),
    "Cl": ("chloride", "homogeneous_halide", "halide"),
    "Br": ("bromide", "homogeneous_halide", "halide"),
    "I": ("iodide", "homogeneous_halide", "halide"),
    "S": ("sulfide", "homogeneous_nonoxygen_chalcogenide", "chalcogenide"),
    "Se": ("selenide", "homogeneous_nonoxygen_chalcogenide", "chalcogenide"),
    "Te": ("telluride", "homogeneous_nonoxygen_chalcogenide", "chalcogenide"),
}
FAMILY_ORDER = (
    "oxide", "nitride", "fluoride", "chloride", "bromide", "iodide",
    "sulfide", "selenide", "telluride", "mixed_halide",
    "mixed_chalcogenide", "oxynitride", "oxyhalide",
    "oxychalcogenide", "other_mixed", "other_unresolved",
)
POLICY_WATERFALL_STATUSES = (
    "missing_paired_labels", "ambiguous_roles", "source_structural_failure",
    "outside_legacy_scope", "legacy_role_out_of_support",
    "legacy_oracle_rejected", "legacy_oracle_unknown", "shared_X_required",
    "role_out_of_training_support", "eligible_source", "eligible_oracle",
)
MF_P4_WATERFALL_STEPS = (
    "missing_paired_labels", "ambiguous_roles", "source_structural_failure",
    "role_out_of_training_support", "eligible_MF_P4",
)


def anion_key(x_sites: tuple[str, ...]) -> str:
    """Canonical count key; ordered X slots are retained in a separate field."""
    counts = Counter(x_sites)
    symbols = [symbol for symbol in ANION_ORDER if symbol in counts]
    symbols += sorted(set(counts) - set(symbols))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "") for symbol in symbols)


def classify_family(x_sites: tuple[str, ...]) -> tuple[str, str, str]:
    """Detailed family, nonoverlapping display category, legacy family."""
    if len(x_sites) != 3 or any(not symbol for symbol in x_sites):
        return "other_unresolved", "other_unresolved", ""
    species = set(x_sites)
    if len(species) == 1:
        return HOMOGENEOUS.get(next(iter(species)), ("other_unresolved", "other_unresolved", ""))
    if species <= HALIDES:
        return "mixed_halide", "mixed_halide", ""
    if species <= CHALCOGENS:
        return "mixed_chalcogenide", "mixed_chalcogenide", ""
    if species <= {"O", "N"}:
        return "oxynitride", "oxynitride", ""
    if "O" in species and species <= ({"O"} | HALIDES):
        return "oxyhalide", "oxyhalide", ""
    if "O" in species and species <= ({"O"} | CHALCOGENS):
        return "oxychalcogenide", "oxychalcogenide", ""
    return "other_mixed", "other_mixed", ""


def load_source_rows(
    data_root: Path,
    splits: tuple[str, ...] = SPLITS,
) -> list[dict[str, str]]:
    """Read saved splits as strings, omitting only the large embedded CIF text."""
    rows: list[dict[str, str]] = []
    for split in splits:
        with (data_root / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row.pop("cif", None)
                row["split"] = split
                rows.append(row)
    if len({row["material_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate material_id across saved splits")
    return rows


def source_roles(row: Mapping[str, str]) -> dict[str, Any]:
    """Verify A/B/X slots against source labels and the stored parser permutation."""
    reasons: list[str] = []
    try:
        source = json.loads(row["source_site_symbols"])
        parsed = json.loads(row["meidnet_site_symbols"])
        permutation = json.loads(row["site_permutation"])
    except (KeyError, ValueError, TypeError) as error:
        source, parsed, permutation = [], [], []
        reasons.append(f"invalid_site_metadata:{type(error).__name__}")
    if len(source) != 5:
        reasons.append("source_site_count_not_five")
    if len(parsed) != 5 or sorted(permutation) != list(range(5)):
        reasons.append("invalid_parser_correspondence")
    elif len(source) == 5 and any(source[i] != parsed[permutation[i]] for i in range(5)):
        reasons.append("parser_species_mismatch")
    a = source[0] if len(source) == 5 else ""
    b = source[1] if len(source) == 5 else ""
    x = tuple(source[2:5]) if len(source) == 5 else ()
    if a != row.get("source_A_ion") or b != row.get("source_B_ion"):
        reasons.append("source_role_label_mismatch")
    if x:
        try:
            source_anion_counts = Counter({
                symbol: round(float(amount))
                for symbol, amount in Composition(row["source_anion"]).get_el_amt_dict().items()
            })
            if source_anion_counts != Counter(x):
                reasons.append("source_anion_label_mismatch")
        except (KeyError, ValueError, TypeError):
            reasons.append("invalid_source_anion_label")
    family, display, legacy_family = classify_family(x)
    return {
        "A": a,
        "B": b,
        "X": x,
        "anion_key": anion_key(x),
        "family": family,
        "display_category": display,
        "legacy_family": legacy_family,
        "role_status": "VERIFIED" if not reasons else "AMBIGUOUS",
        "role_reason": ";".join(reasons),
    }


def training_support(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    """Freeze observed site vocabularies using verified TRAIN rows only."""
    training = [row for row in rows if row["split"] == "train"]
    if not training:
        raise ValueError("Training rows are required to freeze role vocabularies")
    role_counts = {role: Counter() for role in ("A", "B", "X")}
    family_role_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {role: Counter() for role in ("A", "B", "X")}
    )
    family_counts: Counter[str] = Counter()
    anion_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    ambiguous = 0
    known = set(legacy.SPECIES_LIST)
    for row in training:
        roles = source_roles(row)
        if roles["role_status"] != "VERIFIED":
            ambiguous += 1
            continue
        a, b, x = roles["A"], roles["B"], roles["X"]
        for role, symbols in (("A", (a,)), ("B", (b,)), ("X", x)):
            role_counts[role].update(symbol for symbol in symbols if symbol in known)
            family_role_counts[roles["family"]][role].update(
                symbol for symbol in symbols if symbol in known
            )
        family_counts[roles["family"]] += 1
        anion_counts[roles["anion_key"]] += 1
        composition_counts[row["composition_group"]] += 1
    return {
        "V_A": set(role_counts["A"]),
        "V_B": set(role_counts["B"]),
        "V_X": set(role_counts["X"]),
        "role_counts": role_counts,
        "family_role_counts": dict(family_role_counts),
        "family_counts": family_counts,
        "anion_counts": anion_counts,
        "composition_counts": composition_counts,
        "training_rows": len(training),
        "ambiguous_training_rows": ambiguous,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_minimum_distance(atoms: Any) -> float:
    """Shortest periodic pair or nonzero self-image distance in angstroms."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    pair = atoms.get_all_distances(mic=True)
    np.fill_diagonal(pair, np.inf)
    pair_min = float(np.min(pair))
    singular_min = float(np.linalg.svd(cell, compute_uv=False).min())
    if singular_min <= 1e-8:
        raise ValueError("singular_cell")
    neighbors = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    neighbors = neighbors[np.any(neighbors != 0, axis=1)]
    self_min = float(np.linalg.norm(neighbors @ cell, axis=1).min())
    # Any unseen integer image has max component >=2 and length >=2*sigma_min.
    if self_min >= 2 * singular_min:
        extent = math.ceil(self_min / singular_min)
        if extent > 12:
            raise ValueError("self_image_search_exceeds_finite_guard")
        neighbors = np.array(
            list(itertools.product(range(-extent, extent + 1), repeat=3)), dtype=float
        )
        neighbors = neighbors[np.any(neighbors != 0, axis=1)]
        self_min = float(np.linalg.norm(neighbors @ cell, axis=1).min())
    return min(pair_min, self_min)


def source_geometry(atoms: Any) -> dict[str, Any]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    metric = cell @ cell.T
    reasons = []
    if len(atoms) != 5:
        reasons.append("site_count_not_five")
    if not np.asarray(atoms.pbc, dtype=bool).all():
        reasons.append("not_fully_periodic")
    finite = bool(np.isfinite(cell).all() and np.isfinite(positions).all())
    if not finite:
        reasons.append("nonfinite_cell_or_position")
    volume = float(abs(np.linalg.det(cell))) if np.isfinite(cell).all() else math.nan
    volume_valid = math.isfinite(volume) and volume > 1e-8
    if not volume_valid:
        reasons.append("invalid_volume")
    metric_valid = bool(
        np.isfinite(metric).all() and float(np.linalg.eigvalsh(metric).min()) > 1e-10
    )
    if not metric_valid:
        reasons.append("invalid_metric")
    minimum = None
    if not reasons:
        try:
            minimum = source_minimum_distance(atoms)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            reasons.append(f"distance_unknown:{error}")
        else:
            if not math.isfinite(minimum) or minimum < 0.8:
                reasons.append("periodic_distance_below_0p8_angstrom")
    return {
        "source_site_count": len(atoms),
        "source_fully_occupied_ase_sites": len(atoms) == 5,
        "source_fully_periodic": bool(np.asarray(atoms.pbc, dtype=bool).all()),
        "source_finite_cell_and_positions": finite,
        "source_positive_volume": volume_valid,
        "source_valid_metric": metric_valid,
        "source_volume_angstrom3": volume,
        "source_min_periodic_distance_angstrom": minimum,
        "source_minimal_valid": not reasons,
        "source_geometry_reasons": ";".join(reasons),
    }


def legacy_scope(roles: Mapping[str, Any]) -> bool:
    if roles["role_status"] != "VERIFIED" or len(set(roles["X"])) != 1:
        return False
    family = roles["legacy_family"]
    return bool(
        family in legacy.VALENCE_B_BY_FAMILY
        and roles["A"] in legacy.A_CATIONS
        and roles["B"] in legacy.VALENCE_B_BY_FAMILY[family]
        and roles["X"][0] in legacy.ANIONS_ALL
    )


def p1_radius_projected_minimal(roles: Mapping[str, Any]) -> str:
    if not legacy_scope(roles):
        return "OUTSIDE_LEGACY_SCOPE" if len(set(roles["X"])) > 1 else "OUT_OF_SUPPORT"
    b, x = roles["B"], roles["X"][0]
    length = float(np.clip(2 * (legacy.get_ionic_radius(b) + legacy.get_ionic_radius(x)), 3, 8))
    if not math.isfinite(length) or length <= 0:
        return "INVALID_CELL"
    # Every distinct canonical five-site pair is at least half the cubic length.
    return "PASS" if length / 2 >= 0.8 else "GROSS_OVERLAP"


def legacy_source_diagnostics(atoms: Any, roles: Mapping[str, Any]) -> dict[str, Any]:
    default = "NOT_APPLICABLE" if len(set(roles["X"])) > 1 else "OUT_OF_SUPPORT"
    result = {
        "legacy_source_charge": default,
        "legacy_source_t_mu": default,
        "legacy_source_BX_distance": default,
        "legacy_source_t": "",
        "legacy_source_mu": "",
        "legacy_source_diagnostic_error": "",
    }
    if not legacy_scope(roles):
        return result
    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        charge = legacy.check_charge_balance_existential(structure)
        factors, t, mu = legacy.goldschmidt_and_mu_ok(structure)
        distance = legacy.check_BX_distances_family(structure)
        result.update({
            "legacy_source_charge": "PASS" if not charge else "FAIL",
            "legacy_source_t_mu": "PASS" if not factors else "FAIL",
            "legacy_source_BX_distance": "PASS" if not distance else "FAIL",
            "legacy_source_t": float(t),
            "legacy_source_mu": float(mu),
        })
    except (ValueError, RuntimeError, IndexError, KeyError) as error:
        result.update({
            "legacy_source_charge": "UNKNOWN",
            "legacy_source_t_mu": "UNKNOWN",
            "legacy_source_BX_distance": "UNKNOWN",
            "legacy_source_diagnostic_error": f"{type(error).__name__}:{error}",
        })
    return result


def legacy_oracle_result(
    roles: Mapping[str, Any],
    in_scope: bool,
    cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Cache the original legacy filter verdict for one homogeneous A/B/X tuple."""
    x = roles["X"]
    oracle = {
        "status": "OUTSIDE_LEGACY_SCOPE" if not (len(x) == 3 and len(set(x)) == 1) else "OUT_OF_SUPPORT",
        "first_failure": "", "error": "", "lattice_a": "", "charge": "NOT_APPLICABLE",
        "t_mu": "NOT_APPLICABLE", "distance": "NOT_APPLICABLE",
        "refinement": "NOT_APPLICABLE", "realism": "NOT_APPLICABLE",
    }
    if in_scope:
        key = (roles["A"], roles["B"], x[0])
        if key not in cache:
            try:
                finding, _ = oracle_species_record(*key, refine=True)
                cache[key] = {
                    "status": "PASS" if finding["deterministically_exportable"] else "FAIL",
                    "first_failure": finding["first_rejection_stage"] or "",
                    "error": "",
                    "lattice_a": finding["projected_lattice_a_angstrom"],
                    "charge": "PASS" if finding["charge_filter_passed"] else "FAIL",
                    "t_mu": "PASS" if finding["tolerance_mu_filter_passed"] else "FAIL",
                    "distance": "PASS" if finding["B_X_distance_filter_passed"] else "FAIL",
                    "refinement": "PASS" if finding["refinement_passed"] else "FAIL",
                    "realism": "PASS" if finding["realism_passed"] else "FAIL",
                }
            except (ValueError, RuntimeError, IndexError, KeyError) as error:
                cache[key] = {
                    "status": "UNKNOWN", "first_failure": "",
                    "error": f"{type(error).__name__}:{error}",
                    "lattice_a": "", "charge": "UNKNOWN", "t_mu": "UNKNOWN",
                    "distance": "UNKNOWN", "refinement": "UNKNOWN", "realism": "UNKNOWN",
                }
        oracle = cache[key]
    return oracle


def census_entry(
    row: Mapping[str, str],
    atoms: Any | None,
    support: Mapping[str, Any],
    oracle_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Record source geometry and policy scope without a model forward pass."""
    roles = source_roles(row)
    source = source_geometry(atoms) if atoms is not None else {
        "source_site_count": "",
        "source_fully_occupied_ase_sites": False,
        "source_fully_periodic": False,
        "source_finite_cell_and_positions": False,
        "source_positive_volume": False,
        "source_valid_metric": False,
        "source_volume_angstrom3": "",
        "source_min_periodic_distance_angstrom": "",
        "source_minimal_valid": False,
        "source_geometry_reasons": "raw_source_id_missing",
    }
    try:
        paired = all(math.isfinite(float(row[name])) for name in ("heat_all", "dir_gap"))
    except (KeyError, ValueError, TypeError):
        paired = False
    a, b, x = roles["A"], roles["B"], roles["X"]
    role_supported = {
        "A": a in support["V_A"],
        "B": b in support["V_B"],
        "X": all(symbol in support["V_X"] for symbol in x) and len(x) == 3,
    }
    all_role_supported = all(role_supported.values()) and roles["role_status"] == "VERIFIED"
    family_support = support["family_role_counts"].get(roles["family"], {})
    per_family_role = {
        "A": a in family_support.get("A", {}),
        "B": b in family_support.get("B", {}),
        "X": bool(x) and all(symbol in family_support.get("X", {}) for symbol in x),
    }
    single = len(x) == 3 and len(set(x)) == 1
    minimal = bool(source["source_minimal_valid"] and paired and roles["role_status"] == "VERIFIED")
    old_scope = legacy_scope(roles)
    oracle = legacy_oracle_result(roles, old_scope, oracle_cache)
    p0 = old_scope and minimal
    p1 = p0
    p2 = p0
    p3 = single and minimal and all_role_supported
    p4 = minimal and all_role_supported
    common_failure = (
        "missing_paired_labels" if not paired else
        "ambiguous_roles" if roles["role_status"] != "VERIFIED" else
        "source_structural_failure" if not source["source_minimal_valid"] else ""
    )
    legacy_failure = "outside_legacy_scope" if not single else "legacy_role_out_of_support"
    if common_failure:
        p0_waterfall = common_failure
    elif not old_scope:
        p0_waterfall = legacy_failure
    elif oracle["status"] == "PASS":
        p0_waterfall = "eligible_oracle"
    elif oracle["status"] == "UNKNOWN":
        p0_waterfall = "legacy_oracle_unknown"
    else:
        p0_waterfall = "legacy_oracle_rejected"
    policy_waterfall = {
        "MF_P0": p0_waterfall,
        "MF_P1": common_failure or ("eligible_source" if old_scope else legacy_failure),
        "MF_P2": common_failure or ("eligible_source" if old_scope else legacy_failure),
        "MF_P3": common_failure or (
            "shared_X_required" if not single else
            "role_out_of_training_support" if not all_role_supported else "eligible_source"
        ),
        "MF_P4": common_failure or (
            "role_out_of_training_support" if not all_role_supported else "eligible_source"
        ),
    }
    if not paired:
        first_failure = "missing_paired_labels"
    elif roles["role_status"] != "VERIFIED":
        first_failure = "ambiguous_roles"
    elif not source["source_minimal_valid"]:
        first_failure = "source_structural_failure"
    elif not all_role_supported:
        first_failure = "role_out_of_training_support"
    else:
        first_failure = "eligible_MF_P4"
    output = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "split": row["split"],
        "material_id": row["material_id"],
        "source_id": row["source_id"],
        "source_unique_id": row["source_unique_id"],
        "composition_group": row["composition_group"],
        "source_formula": row["source_formula"],
        "source_anion": row["source_anion"],
        "A": a, "B": b,
        "X_ordered_json": json.dumps(x),
        "X_multiset_json": json.dumps(dict(sorted(Counter(x).items()))),
        "anion_key": roles["anion_key"],
        "family": roles["family"],
        "display_category": roles["display_category"],
        "legacy_family": roles["legacy_family"],
        "single_anion": single,
        "A_equals_B": a == b,
        "role_status": roles["role_status"],
        "role_reason": roles["role_reason"],
        "paired_labels_available": paired,
        "heat_all": row["heat_all"],
        "dir_gap": row["dir_gap"],
        **source,
        "legacy_scope": old_scope,
        "legacy_source_charge": "NOT_APPLICABLE" if not single else "OUT_OF_SUPPORT",
        "legacy_source_t_mu": "NOT_APPLICABLE" if not single else "OUT_OF_SUPPORT",
        "legacy_source_BX_distance": "NOT_APPLICABLE" if not single else "OUT_OF_SUPPORT",
        "legacy_source_t": "",
        "legacy_source_mu": "",
        "legacy_source_diagnostic_error": "",
        "legacy_oracle_status": oracle["status"],
        "legacy_oracle_first_failure": oracle["first_failure"],
        "legacy_oracle_error": oracle["error"],
        "legacy_oracle_projected_lattice_a_angstrom": oracle["lattice_a"],
        "legacy_oracle_charge": oracle["charge"],
        "legacy_oracle_t_mu": oracle["t_mu"],
        "legacy_oracle_BX_distance": oracle["distance"],
        "legacy_oracle_refinement": oracle["refinement"],
        "legacy_oracle_realism": oracle["realism"],
        "global_A_supported": role_supported["A"],
        "global_B_supported": role_supported["B"],
        "global_X_supported": role_supported["X"],
        "global_elements_supported": all(
            symbol in support["V_A"] | support["V_B"] | support["V_X"]
            for symbol in (a, b, *x)
        ),
        "all_roles_supported": all_role_supported,
        "family_A_seen_in_train": per_family_role["A"],
        "family_B_seen_in_train": per_family_role["B"],
        "family_X_seen_in_train": per_family_role["X"],
        "exact_anion_seen_in_train": roles["anion_key"] in support["anion_counts"],
        "full_composition_seen_in_train": row["composition_group"] in support["composition_counts"],
        "MF_P0_source_eligible": p0,
        "MF_P0_legacy_oracle_exportable": p0 and oracle["status"] == "PASS",
        "MF_P1_source_eligible": p1,
        "MF_P1_radius_projected_minimal_status": p1_radius_projected_minimal(roles),
        "MF_P2_source_eligible": p2,
        "MF_P3_source_eligible": p3,
        "MF_P4_source_eligible": p4,
        "MF_P2_to_P4_learned_exportability": "NOT_EVALUATED",
        "newly_eligible_after_legacy_veto_removal": p1 and oracle["status"] == "FAIL",
        "newly_eligible_after_global_vocabulary": p3 and not p2,
        "newly_eligible_after_mixed_X": p4 and not p3,
        "first_MF_P4_waterfall_status": first_failure,
        **{policy + "_waterfall_status": status for policy, status in policy_waterfall.items()},
    }
    if old_scope and atoms is not None:
        output.update(legacy_source_diagnostics(atoms, roles))
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]],
    all_count: int,
    family_totals: Counter[str],
) -> dict[str, Any]:
    count = len(rows)
    family = rows[0]["family"] if rows and len({row["family"] for row in rows}) == 1 else "ALL"
    gaps = [float(row["dir_gap"]) for row in rows if row["paired_labels_available"]]
    heats = [float(row["heat_all"]) for row in rows if row["paired_labels_available"]]
    result: dict[str, Any] = {
        "records": count,
        "pct_all_CMR": 100 * count / all_count,
        "pct_family_all_splits": 100 * count / family_totals[family] if family != "ALL" and family_totals[family] else "",
        "composition_groups": len({row["composition_group"] for row in rows}),
        "zero_direct_gap_records": sum(gap == 0 for gap in gaps),
        "positive_direct_gap_records": sum(gap > 0 for gap in gaps),
        "negative_direct_gap_records": sum(gap < 0 for gap in gaps),
        "direct_gap_min_eV": min(gaps) if gaps else "",
        "direct_gap_max_eV": max(gaps) if gaps else "",
        "heat_all_min_eV_per_atom": min(heats) if heats else "",
        "heat_all_max_eV_per_atom": max(heats) if heats else "",
    }
    for column in (
        "paired_labels_available", "source_minimal_valid", "legacy_scope",
        "all_roles_supported", "global_A_supported", "global_B_supported",
        "global_X_supported", "exact_anion_seen_in_train",
        "full_composition_seen_in_train", "MF_P0_source_eligible",
        "MF_P0_legacy_oracle_exportable", "MF_P1_source_eligible",
        "MF_P2_source_eligible", "MF_P3_source_eligible",
        "MF_P4_source_eligible", "newly_eligible_after_legacy_veto_removal",
        "newly_eligible_after_global_vocabulary", "newly_eligible_after_mixed_X",
    ):
        result[column + "_count"] = sum(bool(row[column]) for row in rows)
    result["ambiguous_role_count"] = sum(row["role_status"] != "VERIFIED" for row in rows)
    result["structural_failure_count"] = sum(not row["source_minimal_valid"] for row in rows)
    result["missing_paired_label_count"] = sum(not row["paired_labels_available"] for row in rows)
    result["legacy_oracle_unknown_count"] = sum(row["legacy_oracle_status"] == "UNKNOWN" for row in rows)
    for policy in ("MF_P0", "MF_P1", "MF_P2", "MF_P3", "MF_P4"):
        for status in POLICY_WATERFALL_STATUSES:
            result[policy + "_waterfall_" + status + "_count"] = sum(
                row[policy + "_waterfall_status"] == status for row in rows
            )
        assert sum(
            result[policy + "_waterfall_" + status + "_count"]
            for status in POLICY_WATERFALL_STATUSES
        ) == count
    for step in MF_P4_WATERFALL_STEPS:
        result["waterfall_" + step + "_count"] = sum(
            row["first_MF_P4_waterfall_status"] == step for row in rows
        )
    assert sum(result["waterfall_" + step + "_count"] for step in MF_P4_WATERFALL_STEPS) == count
    return result


def aggregate_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_count = len(rows)
    family_totals = Counter(row["family"] for row in rows)
    table = []
    exact = []
    for split in (*SPLITS, "total"):
        selected = [row for row in rows if split == "total" or row["split"] == split]
        split_family_counts = Counter(row["family"] for row in selected)
        strata = [("all", "ALL", selected)]
        strata += [("family", family, [row for row in selected if row["family"] == family])
                   for family in FAMILY_ORDER if split_family_counts[family]]
        strata += [("exact_anion", key, [row for row in selected if row["anion_key"] == key])
                   for key in sorted({row["anion_key"] for row in selected})]
        for level, key, subset in strata:
            result = {
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "split": split,
                "stratum_type": level,
                "stratum": key,
                "family": subset[0]["family"] if subset and len({row["family"] for row in subset}) == 1 else "ALL",
                "single_or_mixed": "single" if subset and all(row["single_anion"] for row in subset) else (
                    "mixed" if subset and all(not row["single_anion"] for row in subset) else "both"
                ),
                **aggregate(subset, all_count, family_totals),
            }
            result["pct_split_family"] = (
                100 * len(subset) / split_family_counts[result["family"]]
                if result["family"] != "ALL" else ""
            )
            table.append(result)
            if level == "exact_anion":
                exact.append({
                    "split": split, "anion_key": key, "family": result["family"],
                    "single_or_mixed": result["single_or_mixed"],
                    "records": result["records"],
                    "composition_groups": result["composition_groups"],
                    "pct_all_CMR": result["pct_all_CMR"],
                    "pct_split_family": result["pct_split_family"],
                })
    return table, exact


def support_serializable(support: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "support_definition": "Verified TRAIN source roles intersected with checkpoint's 118-element mapping; no frequency cutoff",
        "training_rows": support["training_rows"],
        "ambiguous_training_rows": support["ambiguous_training_rows"],
        "V_A": sorted(support["V_A"]),
        "V_B": sorted(support["V_B"]),
        "V_X": sorted(support["V_X"]),
        "role_counts": {role: dict(sorted(counts.items())) for role, counts in support["role_counts"].items()},
        "family_counts": dict(sorted(support["family_counts"].items())),
        "family_role_counts": {
            family: {role: dict(sorted(counts.items())) for role, counts in role_counts.items()}
            for family, role_counts in sorted(support["family_role_counts"].items())
        },
        "exact_anion_counts": dict(sorted(support["anion_counts"].items())),
        "full_composition_counts": dict(sorted(support["composition_counts"].items())),
        "analytic_combinatorial_support": {
            "MF_P3_A_B_X_role_tuples": len(support["V_A"]) * len(support["V_B"]) * len(support["V_X"]),
            "MF_P4_ordered_A_B_X1_X2_X3_role_tuples": (
                len(support["V_A"]) * len(support["V_B"]) * len(support["V_X"]) ** 3
            ),
            "note": "Role tuples are finite categorical supports, not unique compositions, valid structures, or stable materials.",
        },
    }


def family_support_table(rows: list[dict[str, Any]], support: Mapping[str, Any]) -> list[dict[str, Any]]:
    table = []
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        counts = Counter(row["split"] for row in family_rows)
        if family_rows:
            assessment = "PAIRED_SOURCE_AVAILABLE_VALIDATION" if counts["val"] else "NOT_ASSESSABLE_NO_VALIDATION_DATA"
        else:
            assessment = "NOT_ASSESSABLE_NO_MATCHING_DATA"
        table.append({
            "family": family,
            "train_count": counts["train"],
            "val_count": counts["val"],
            "test_count": counts["test"],
            "total_count": len(family_rows),
            "train_roles_supported": family in support["family_role_counts"],
            "global_role_supported_count": sum(row["all_roles_supported"] for row in family_rows),
            "inference_count": "NOT_RUN_BY_SOURCE_CENSUS",
            "assessment_status": assessment,
        })
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    train = load_source_rows(args.data_root, ("train",))
    support = training_support(train)  # Frozen before reading validation/test rows.
    rows = train + load_source_rows(args.data_root, ("val", "test"))
    by_source_id = {int(row["source_id"]): row for row in rows}
    if len(by_source_id) != len(rows):
        raise ValueError("Duplicate source_id across saved splits")
    oracle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    findings: dict[str, dict[str, Any]] = {}
    database = connect(str(args.raw_db))
    raw_count = 0
    for raw in database.select():
        raw_count += 1
        row = by_source_id.get(int(raw.id))
        if row is None:
            continue
        if str(raw.unique_id) != row["source_unique_id"]:
            raise ValueError(f"Raw source identity mismatch at ID {raw.id}")
        findings[row["material_id"]] = census_entry(row, raw.toatoms(), support, oracle_cache)
    if len(findings) != len(rows):
        raise ValueError(f"Raw database only matched {len(findings)}/{len(rows)} retained records")
    entry_rows = [findings[row["material_id"]] for row in rows]
    aggregate_rows, exact_rows = aggregate_tables(entry_rows)
    family_rows = family_support_table(entry_rows, support)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "census_per_entry.csv", entry_rows)
    write_csv(args.output_dir / "dataset_conformity_by_split_and_family.csv", aggregate_rows)
    write_csv(args.output_dir / "exact_anion_inventory.csv", exact_rows)
    write_csv(args.output_dir / "family_support.csv", family_rows)
    (args.output_dir / "training_support.json").write_text(
        json.dumps(support_serializable(support), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    input_paths = {
        **{split + "_csv": args.data_root / f"{split}.csv" for split in SPLITS},
        "raw_db": args.raw_db,
    }
    manifest = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " .venv/Scripts/python.exe scripts/multifamily_census.py",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "raw_database_rows": raw_count,
        "retained_rows": len(entry_rows),
        "raw_reference_or_other_excluded_rows": raw_count - len(entry_rows),
        "split_counts": dict(Counter(row["split"] for row in entry_rows)),
        "legacy_oracle_unique_role_tuples": len(oracle_cache),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "census_per_entry.csv", "dataset_conformity_by_split_and_family.csv",
                "exact_anion_inventory.csv", "family_support.csv", "training_support.json",
            )
        },
        "note": "Source census only; no model inference was performed on any split.",
    }
    (args.output_dir / "census_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = next(row for row in aggregate_rows if row["split"] == "total" and row["stratum_type"] == "all")
    print(json.dumps({
        "retained": total["records"],
        "family_counts": dict(Counter(row["family"] for row in entry_rows)),
        "p0_legacy_oracle": total["MF_P0_legacy_oracle_exportable_count"],
        "p1_source": total["MF_P1_source_eligible_count"],
        "p3_source": total["MF_P3_source_eligible_count"],
        "p4_source": total["MF_P4_source_eligible_count"],
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
