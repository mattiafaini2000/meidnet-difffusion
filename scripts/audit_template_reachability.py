#!/usr/bin/env python3
"""Audit fixed-template decoder inputs and current ABX3 constraint reachability.

This follow-up is deliberately additive. It imports the completed decoder-audit
harness and the repository's current tables/functions without modifying either.
CMR properties are provisional numerical inputs, not validated target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

WORKSPACE = Path(__file__).resolve().parents[1]
MEIDNET_ROOT = WORKSPACE / "MEIDNet-main"
SCRIPTS_ROOT = WORKSPACE / "scripts"
DATA_ROOT = WORKSPACE / "data" / "processed" / "cmr_reconstructed"
DEFAULT_CHECKPOINT = (
    MEIDNET_ROOT
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)
DEFAULT_REPORT_DIR = WORKSPACE / "reports" / "template_reachability"
PROTOCOL_PATH = DEFAULT_REPORT_DIR / "protocol.md"
OLD_RAW_PATH = WORKSPACE / "reports" / "decoder_audit" / "raw_heads.csv"
OLD_SAMPLE_PATH = WORKSPACE / "reports" / "decoder_audit" / "per_sample.csv"
OLD_YIELD_PATH = WORKSPACE / "reports" / "decoder_audit" / "yield_by_condition.csv"
OLD_SUMMARY_PATH = WORKSPACE / "reports" / "decoder_audit_summary.json"
REVIEWED_COMMIT = "52fd7cdcba7c392f302c0b71f82d339bcd904367"

for import_root in (SCRIPTS_ROOT, MEIDNET_ROOT, WORKSPACE):
    value = str(import_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import audit_decoder as audit  # noqa: E402
from meidnet import design as generation  # noqa: E402
from meidnet.model import parse_cif_to_dense  # noqa: E402
from meidnet_audits.files import package_version, sha256_file, write_csv  # noqa: E402
from meidnet_audits.template_report import fmt, markdown_report as render_markdown_report  # noqa: E402


LATENT_ARMS = ("joint", "fixed_real", "property")
GEOMETRY_MODES = ("zero", "template")
EXPECTED_AUDIT_SHA256 = audit.EXPECTED_AUDIT_SHA256
EXPECTED_CHECKPOINT_SHA256 = audit.EXPECTED_CHECKPOINT_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DATA_ROOT / "audit.csv")
    parser.add_argument("--cif-root", type=Path, default=DATA_ROOT / "cifs" / "val")
    parser.add_argument("--train-csv", type=Path, default=DATA_ROOT / "train.csv")
    parser.add_argument("--train-cif-root", type=Path, default=DATA_ROOT / "cifs" / "train")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--decode-temperature", type=float, default=1.25)
    parser.add_argument("--decode-topk", type=int, default=12)
    parser.add_argument("--decode-tries", type=int, default=12)
    return parser.parse_args()


def sha256_tensor_bytes(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": run_git("status", "--short").splitlines(),
        "commits_since_reviewed": run_git(
            "rev-list", "--count", f"{REVIEWED_COMMIT}..HEAD"
        ),
        "changed_files_since_reviewed": run_git(
            "diff", "--name-only", f"{REVIEWED_COMMIT}..HEAD"
        ).splitlines(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        return str(path.resolve())


def protected_files(
    audit_csv: Path,
    cif_root: Path,
    checkpoint: Path,
) -> list[Path]:
    paths = [
        DATA_ROOT / "train.csv",
        DATA_ROOT / "val.csv",
        DATA_ROOT / "test.csv",
        audit_csv,
        checkpoint,
        *sorted(path for path in MEIDNET_ROOT.rglob("*") if path.is_file()),
        *sorted(cif_root / f"{material_id}.cif" for material_id in pd.read_csv(
            audit_csv,
            usecols=["material_id"],
            dtype={"material_id": "string"},
        )["material_id"].astype(str)),
        WORKSPACE / "reports" / "decoder_audit.md",
        WORKSPACE / "reports" / "decoder_audit_protocol.md",
        WORKSPACE / "reports" / "decoder_audit_summary.json",
        WORKSPACE / "reports" / "decoder_audit_raw.csv",
        WORKSPACE / "reports" / "decoder_audit_results.csv",
        WORKSPACE / "reports" / "decoder_audit_failures.csv",
        *sorted(
            path
            for path in (WORKSPACE / "reports" / "decoder_audit").rglob("*")
            if path.is_file()
        ),
    ]
    unique = {path.resolve() for path in paths}
    missing = [path for path in sorted(unique) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected inputs are missing: {missing}")
    return sorted(unique)


def file_hashes(paths: list[Path]) -> dict[Path, str]:
    return {path: sha256_file(path) for path in paths}


def derive_training_template_convention(
    train_csv: Path,
    train_cif_root: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Choose one X-slot order from training data, before audit inference."""
    frame = pd.read_csv(train_csv, dtype={"material_id": "string"})
    eligible = frame[frame["generation_scope_compatible"].map(audit.parse_bool)].copy()
    eligible = eligible.sort_values("material_id").reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No generation-scope-compatible training rows")

    template = generation.TEMPLATE.detach().cpu().float()
    candidates = [(0, 1, *permutation) for permutation in itertools.permutations((2, 3, 4))]
    errors: dict[tuple[int, ...], list[float]] = {candidate: [] for candidate in candidates}
    center_x: list[float] = []
    wrapped = 0
    role_order_valid = 0
    site_order_preserved = 0

    coordinate_offset = 6 + generation.MAX_SITES * generation.MAX_SITES + (
        generation.MAX_SITES * generation.NUM_SPECIES
    )
    for _, row in eligible.iterrows():
        material_id = str(row["material_id"])
        cif_path = train_cif_root / str(row["cif_filename"])
        if not cif_path.is_file():
            cif_path = train_cif_root / f"{material_id}.cif"
        dense = parse_cif_to_dense(str(cif_path), max_sites=generation.MAX_SITES)
        coordinates = torch.from_numpy(
            dense[coordinate_offset:].reshape(generation.MAX_SITES, 3)[:5]
        ).float()
        source_symbols = json.loads(str(row["source_site_symbols"]))
        parsed_symbols = json.loads(str(row["meidnet_site_symbols"]))
        expected_symbols = [
            str(row["source_A_ion"]),
            str(row["source_B_ion"]),
            *[str(source_symbols[2])] * 3,
        ]
        if parsed_symbols == expected_symbols:
            role_order_valid += 1
        if audit.parse_bool(row["site_order_preserved"]):
            site_order_preserved += 1
        center_x.append(float(coordinates[:, 0].mean()))
        wrapped += int(audit.parse_bool(row["positions_wrapped_by_meidnet_parser"]))
        for candidate in candidates:
            candidate_coordinates = template[list(candidate)]
            delta = coordinates - candidate_coordinates
            delta = delta - torch.round(delta)
            errors[candidate].extend(
                float(value) for value in torch.linalg.vector_norm(delta, dim=1).tolist()
            )

    candidate_metrics = []
    for candidate in candidates:
        values = np.asarray(errors[candidate], dtype=float)
        candidate_metrics.append(
            {
                "slot_from_generation_template": list(candidate),
                "mean_periodic_fractional_displacement": float(values.mean()),
                "median_periodic_fractional_displacement": float(np.median(values)),
                "maximum_periodic_fractional_displacement": float(values.max()),
            }
        )
    candidate_metrics.sort(
        key=lambda item: (
            item["mean_periodic_fractional_displacement"],
            item["slot_from_generation_template"],
        )
    )
    selected = candidate_metrics[0]["slot_from_generation_template"]
    if role_order_valid != len(eligible):
        raise AssertionError(
            f"Training A/B/X role order is not uniform: {role_order_valid}/{len(eligible)}"
        )
    if candidate_metrics[0]["mean_periodic_fractional_displacement"] == candidate_metrics[1][
        "mean_periodic_fractional_displacement"
    ]:
        raise AssertionError("Training-derived template permutation is not unique")

    coordinates = torch.zeros(generation.MAX_SITES, 3, dtype=torch.float32)
    coordinates[: generation.N_TEMPLATE] = template[selected]
    centers = np.asarray(center_x, dtype=float)
    convention = {
        "selection_data": display_path(train_csv),
        "selection_cif_root": display_path(train_cif_root),
        "selection_uses_audit_outcomes": False,
        "eligible_training_rows": int(len(eligible)),
        "role_order_A_B_X_X_X_count": role_order_valid,
        "site_order_preserved_count": site_order_preserved,
        "parser_wrapped_count": wrapped,
        "candidate_metrics": candidate_metrics,
        "selected_slot_from_generation_template": selected,
        "selected_because": "lowest mean periodic fractional site displacement on training rows",
        "training_center_x": {
            "minimum": float(centers.min()),
            "median": float(np.median(centers)),
            "maximum": float(centers.max()),
            "count_below_0p4": int((centers < 0.4).sum()),
            "count_0p4_to_below_0p6": int(((centers >= 0.4) & (centers < 0.6)).sum()),
            "count_at_least_0p6": int((centers >= 0.6).sum()),
        },
        "periodic_origin_limitation": (
            "Parser wrapping creates multiple ordinary-Euclidean coordinate branches; one "
            "universal template cannot reproduce every branch, and the checkpoint's missing "
            "historical training data cannot be inspected."
        ),
    }
    return coordinates, convention


def template_input_manifest(
    coordinates: torch.Tensor,
    convention: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    center = coordinates[: generation.N_TEMPLATE].mean(dim=0)
    relative = coordinates - center
    occupied_pairwise = torch.cdist(
        coordinates[: generation.N_TEMPLATE],
        coordinates[: generation.N_TEMPLATE],
    )
    combined = torch.cat((coordinates.reshape(-1), center.reshape(-1)))
    return center, {
        "coordinate_convention": "dimensionless fractional coordinates",
        "slot_roles": ["A", "B", "X", "X", "X"] + ["padding"] * 15,
        "shape": list(coordinates.shape),
        "dtype": str(coordinates.dtype),
        "coordinates": coordinates.tolist(),
        "center_from_first_five": center.tolist(),
        "all_twenty_mean_not_used": coordinates.mean(dim=0).tolist(),
        "relative_coordinates": relative.tolist(),
        "occupied_pairwise_distance_matrix_fractional": occupied_pairwise.tolist(),
        "padding": "slots 5-19 are zeros, as in parse_cif_to_dense",
        "adjacency": "decoder retains its all-ones 20x20 adjacency",
        "species_or_occupancy_input": False,
        "tensor_hash_algorithm": "SHA-256 of little-endian float32 C-order bytes",
        "coordinates_sha256": sha256_tensor_bytes(coordinates),
        "center_sha256": sha256_tensor_bytes(center),
        "relative_coordinates_sha256": sha256_tensor_bytes(relative),
        "coordinates_then_center_sha256": sha256_tensor_bytes(combined),
        "training_convention": convention,
    }


class GeometryInputAdapter:
    """Replace only decoder coordinates/center while leaving the model untouched."""

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        template_coordinates: torch.Tensor,
        template_center: torch.Tensor,
    ) -> None:
        if mode not in GEOMETRY_MODES:
            raise ValueError(f"Unknown geometry mode: {mode}")
        self.model = model
        self.mode = mode
        self.template_coordinates = template_coordinates.detach().clone()
        self.template_center = template_center.detach().clone()
        self.property_decoder = model.property_decoder
        self.call_count = 0

    def crystal_decoder(
        self,
        latent: torch.Tensor,
        input_coords: torch.Tensor | None = None,
        center: torch.Tensor | None = None,
        species_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.call_count += 1
        batch = latent.shape[0]
        if self.mode == "template":
            input_coords = self.template_coordinates.to(
                device=latent.device,
                dtype=latent.dtype,
            ).unsqueeze(0).expand(batch, -1, -1)
            center = self.template_center.to(
                device=latent.device,
                dtype=latent.dtype,
            ).unsqueeze(0).expand(batch, -1)
        else:
            if input_coords is None:
                input_coords = torch.zeros(
                    batch,
                    generation.MAX_SITES,
                    3,
                    device=latent.device,
                    dtype=latent.dtype,
                )
            if center is None:
                center = torch.zeros(
                    batch,
                    3,
                    device=latent.device,
                    dtype=latent.dtype,
                )
        return self.model.crystal_decoder(
            latent,
            input_coords=input_coords,
            center=center,
            species_mask=species_mask,
        )


def ionic_radius_metadata(symbol: str) -> dict[str, Any]:
    try:
        radii = Element(symbol).ionic_radii or {}
        if isinstance(radii, dict):
            values = [
                {"oxidation_state": str(key), "radius_angstrom": float(value)}
                for key, value in sorted(radii.items(), key=lambda item: str(item[0]))
                if value is not None
            ]
        else:
            values = [
                {"oxidation_state": None, "radius_angstrom": float(value)}
                for value in radii
                if value is not None
            ]
    except Exception as error:
        values = []
        radius_error = f"{type(error).__name__}: {error}"
    else:
        radius_error = None
    return {
        "symbol": symbol,
        "pymatgen_ionic_radius_available": bool(values),
        "pymatgen_ionic_radii": values,
        "generation_radius_angstrom": float(generation.get_ionic_radius(symbol)),
        "generation_fallback_possible": not bool(values),
        "error": radius_error,
    }


def permitted_charge_tuples(a_symbol: str, b_symbol: str, x_symbol: str) -> dict[str, Any]:
    family = generation.FAMILY_OF_X.get(x_symbol)
    a_values = generation.VALENCE_A.get(a_symbol, set())
    b_values = generation.VALENCE_B_BY_FAMILY.get(family, {}).get(b_symbol, set())
    x_values = generation.VALENCE_X.get(x_symbol, set())
    all_tuples = [
        {"qA": int(q_a), "qB": int(q_b), "qX": int(q_x), "total": int(q_a + q_b + 3 * q_x)}
        for q_a, q_b, q_x in itertools.product(
            sorted(a_values),
            sorted(b_values),
            sorted(x_values),
        )
    ]
    neutral = [item for item in all_tuples if item["total"] == 0]
    return {
        "family": family,
        "permitted": all_tuples,
        "neutral": neutral,
        "existential_neutrality_from_tables": bool(neutral),
    }


def oracle_species_record(
    a_symbol: str,
    b_symbol: str,
    x_symbol: str,
    refine: bool,
) -> tuple[dict[str, Any], Structure | None]:
    family = generation.FAMILY_OF_X.get(x_symbol)
    radii = {
        role: ionic_radius_metadata(symbol)
        for role, symbol in (("A", a_symbol), ("B", b_symbol), ("X", x_symbol))
    }
    charge = permitted_charge_tuples(a_symbol, b_symbol, x_symbol)
    b_pool = generation.target_conditioned_B_set(
        0.0,
        family,
        use_target_filter=False,
    )
    compatible_a = compatible_a_symbols(b_symbol, x_symbol, family)
    raw_projected_lattice_a = float(
        2.0
        * (
            generation.get_ionic_radius(b_symbol)
            + generation.get_ionic_radius(x_symbol)
        )
    )
    projected_lattice_a = float(np.clip(raw_projected_lattice_a, 3.0, 8.0))
    symbols = [a_symbol, b_symbol, x_symbol, x_symbol, x_symbol]
    structure = Structure(
        Lattice.cubic(projected_lattice_a),
        symbols,
        generation.TEMPLATE.detach().cpu().numpy().tolist(),
        coords_are_cartesian=False,
    )
    pre_refine_formula = structure.composition.reduced_formula
    pre_refine_species = [site.specie.symbol for site in structure]
    pre_refine_fractional = structure.frac_coords.copy()
    stoichiometry_ok = bool(generation.is_ABX3(structure))
    realism_failures = generation.realism(structure)
    decoded_family = generation.family_from_structure(structure)
    refinement_error = None
    refined = structure
    if refine:
        try:
            refined = SpacegroupAnalyzer(structure, symprec=0.05).get_refined_structure()
            if np.isnan(refined.cart_coords).any():
                refinement_error = "refined Cartesian coordinates are NaN"
        except Exception as error:
            refinement_error = f"{type(error).__name__}: {error}"
            refined = None

    charge_failures: list[str] = []
    distance_failures: list[str] = []
    factor_failures: list[str] = []
    tolerance = None
    mu = None
    if refined is not None and refinement_error is None:
        charge_failures = generation.check_charge_balance_existential(refined)
        distance_failures = generation.check_BX_distances_family(refined)
        factor_failures, tolerance, mu = generation.goldschmidt_and_mu_ok(refined)

    post_refine_species = (
        [site.specie.symbol for site in refined] if refined is not None else None
    )
    post_refine_formula = (
        refined.composition.reduced_formula if refined is not None else None
    )
    refinement_coordinate_max_abs = None
    if refined is not None and len(refined) == len(structure):
        refinement_coordinate_max_abs = float(
            np.max(np.abs(refined.frac_coords - pre_refine_fractional))
        )

    ordered_failures: list[tuple[str, list[str]]] = [
        ("stoichiometry", [] if stoichiometry_ok else ["is_ABX3 returned false"]),
        ("realism", list(realism_failures)),
        ("family", [] if decoded_family == family else [f"decoded family {decoded_family!r}"]),
        ("refinement", [] if refinement_error is None else [refinement_error]),
        ("charge", list(charge_failures)),
        ("B-X distance", list(distance_failures)),
        ("tolerance/mu", list(factor_failures)),
    ]
    first_failure = next(
        (
            {"stage": stage, "reasons": reasons}
            for stage, reasons in ordered_failures
            if reasons
        ),
        None,
    )
    exportable = first_failure is None
    if b_symbol not in b_pool:
        sampler_first_failure = "B_pool"
    elif not compatible_a:
        sampler_first_failure = "charge_compatible_A"
    elif a_symbol not in compatible_a:
        sampler_first_failure = "source_A_excluded_by_source_B_X"
    elif a_symbol == b_symbol:
        sampler_first_failure = "distinct_roles"
    elif first_failure is not None:
        sampler_first_failure = first_failure["stage"]
    else:
        sampler_first_failure = None
    record = {
        "A": a_symbol,
        "B": b_symbol,
        "X": x_symbol,
        "family": family,
        "declared_role_scope_compatible": (
            a_symbol in generation.A_CATIONS
            and b_symbol in generation.B_CATIONS
            and x_symbol in generation.ANIONS_ALL
            and family in generation.VALENCE_B_BY_FAMILY
        ),
        "ionic_radius_A_available": radii["A"]["pymatgen_ionic_radius_available"],
        "ionic_radius_B_available": radii["B"]["pymatgen_ionic_radius_available"],
        "ionic_radius_X_available": radii["X"]["pymatgen_ionic_radius_available"],
        "all_required_ionic_radii_available": all(
            item["pymatgen_ionic_radius_available"] for item in radii.values()
        ),
        "ionic_radii_json": json.dumps(radii, sort_keys=True),
        "permitted_charge_tuples_json": json.dumps(charge["permitted"], sort_keys=True),
        "neutral_charge_tuples_json": json.dumps(charge["neutral"], sort_keys=True),
        "existential_charge_neutrality_from_tables": charge[
            "existential_neutrality_from_tables"
        ],
        "B_pool_json": json.dumps(b_pool),
        "source_B_in_family_pool": b_symbol in b_pool,
        "source_conditional_compatible_A_json": json.dumps(compatible_a),
        "source_A_in_conditional_compatible_pool": a_symbol in compatible_a,
        "A_equals_B": a_symbol == b_symbol,
        "raw_projected_lattice_a_angstrom": raw_projected_lattice_a,
        "projected_lattice_a_angstrom": projected_lattice_a,
        "lattice_clipped_low": raw_projected_lattice_a < 3.0,
        "lattice_clipped_high": raw_projected_lattice_a > 8.0,
        "oracle_slot_order": "A,B,X,X,X",
        "oracle_fractional_coordinates_json": json.dumps(
            generation.TEMPLATE.detach().cpu().tolist()
        ),
        "stoichiometry_ok": stoichiometry_ok,
        "realism_passed": not realism_failures,
        "realism_failures_json": json.dumps(realism_failures),
        "decoded_family": decoded_family,
        "family_passed": decoded_family == family,
        "refinement_requested": refine,
        "refinement_passed": refinement_error is None,
        "refinement_error": refinement_error,
        "pre_refine_formula": pre_refine_formula,
        "pre_refine_site_count": len(structure),
        "pre_refine_species_json": json.dumps(pre_refine_species),
        "post_refine_formula": post_refine_formula,
        "refined_site_count": len(refined) if refined is not None else None,
        "refined_species_json": json.dumps(post_refine_species),
        "refinement_changed_composition": post_refine_formula != pre_refine_formula,
        "refinement_changed_site_count": (
            len(refined) != len(structure) if refined is not None else None
        ),
        "refinement_changed_species_order": post_refine_species != pre_refine_species,
        "refinement_coordinate_max_abs_fractional": refinement_coordinate_max_abs,
        "charge_filter_passed": not charge_failures,
        "charge_filter_failures_json": json.dumps(charge_failures),
        "B_X_distance_filter_passed": not distance_failures,
        "B_X_distance_filter_failures_json": json.dumps(distance_failures),
        "tolerance_mu_filter_passed": not factor_failures,
        "tolerance_mu_filter_failures_json": json.dumps(factor_failures),
        "goldschmidt_t": float(tolerance) if tolerance is not None else None,
        "octahedral_mu": float(mu) if mu is not None else None,
        "first_rejection_stage": first_failure["stage"] if first_failure else None,
        "first_rejection_reasons_json": (
            json.dumps(first_failure["reasons"]) if first_failure else "[]"
        ),
        "sampler_first_rejection_stage": sampler_first_failure,
        "deterministically_exportable": exportable,
        "formal_valence_scope_note": (
            "Repository-table admissibility only; rejection is not proof of physical impossibility."
        ),
    }
    return record, refined if exportable else None


def reference_reachability_rows(
    frame: pd.DataFrame,
    refine: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for _, source in frame.iterrows():
        context = audit.source_context(source)
        oracle, _ = oracle_species_record(
            context["A"],
            context["B"],
            context["X"],
            refine,
        )
        row = {
            "material_id": str(source["material_id"]),
            "composition_group": str(source["composition_group"]),
            "source_formula": str(source["source_formula"]),
            "source_A": context["A"],
            "source_B": context["B"],
            "source_X": context["X"],
            "family": context["family"],
            "csv_generation_scope_compatible": audit.parse_bool(
                source["generation_scope_compatible"]
            ),
            **oracle,
        }
        rows.append(row)
        by_id[row["material_id"]] = row
    return rows, by_id


def family_charge_rows(refine: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, x_symbol in (("oxide", "O"), ("nitride", "N")):
        b_elements = generation.target_conditioned_B_set(
            0.0,
            family,
            use_target_filter=False,
        )
        for b_symbol in b_elements:
            compatible_a: list[str] = []
            neutral_tuples: list[dict[str, Any]] = []
            exportable_a: list[str] = []
            first_rejections: Counter[str] = Counter()
            for a_symbol in generation.A_CATIONS:
                charge = permitted_charge_tuples(a_symbol, b_symbol, x_symbol)
                if not charge["neutral"]:
                    continue
                compatible_a.append(a_symbol)
                neutral_tuples.extend(
                    {"A": a_symbol, **item} for item in charge["neutral"]
                )
                oracle, _ = oracle_species_record(a_symbol, b_symbol, x_symbol, refine)
                if oracle["deterministically_exportable"]:
                    exportable_a.append(a_symbol)
                else:
                    first_rejections[str(oracle["first_rejection_stage"])] += 1
            rows.append(
                {
                    "family": family,
                    "X": x_symbol,
                    "B": b_symbol,
                    "B_valences_json": json.dumps(
                        sorted(generation.VALENCE_B_BY_FAMILY[family][b_symbol])
                    ),
                    "compatible_A_count": len(compatible_a),
                    "compatible_A_json": json.dumps(compatible_a),
                    "neutral_charge_tuples_json": json.dumps(
                        neutral_tuples,
                        sort_keys=True,
                    ),
                    "has_any_charge_compatible_A": bool(compatible_a),
                    "deterministically_exportable_A_count": len(exportable_a),
                    "deterministically_exportable_A_json": json.dumps(exportable_a),
                    "later_filter_first_rejections_json": json.dumps(
                        dict(first_rejections),
                        sort_keys=True,
                    ),
                }
            )
    return rows


def compatible_a_symbols(b_symbol: str, x_symbol: str, family: str) -> list[str]:
    q_b = generation.VALENCE_B_BY_FAMILY.get(family, {}).get(b_symbol, set())
    q_x = generation.VALENCE_X.get(x_symbol, set())
    required = {-(b_charge + 3 * x_charge) for b_charge in q_b for x_charge in q_x}
    return [
        symbol
        for symbol in generation.A_CATIONS
        if any(charge in generation.VALENCE_A.get(symbol, set()) for charge in required)
    ]


def rank_and_probability(
    values: torch.Tensor,
    symbols: list[str],
    target: str,
    topk: int,
) -> dict[str, Any]:
    values = values.detach().cpu().float()
    order = torch.argsort(values, descending=True).tolist()
    ranked_symbols = [symbols[index] for index in order]
    rank = ranked_symbols.index(target) + 1 if target in ranked_symbols else None
    count = min(topk, len(symbols))
    top_values, top_indices = torch.topk(values, k=count)
    top_probabilities = F.softmax(top_values, dim=-1)
    top_symbols = [symbols[int(index)] for index in top_indices.tolist()]
    probability = 0.0
    if target in top_symbols:
        probability = float(top_probabilities[top_symbols.index(target)])
    return {
        "rank": rank,
        "topk_included": target in top_symbols,
        "topk_probability": probability,
        "topk_symbols": top_symbols,
        "topk_probabilities": [float(value) for value in top_probabilities.tolist()],
    }


def sampling_reachability(
    unmasked_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    context: dict[str, Any],
    temperature: float,
    topk: int,
) -> dict[str, Any]:
    family = context["family"]
    x_symbol = context["X"]
    source_b = context["B"]
    source_a = context["A"]
    b_symbols = generation.target_conditioned_B_set(
        0.0,
        family,
        use_target_filter=False,
    )
    b_indices = [generation.SPECIES_LIST.index(symbol) for symbol in b_symbols]
    b_raw_all = F.softmax(unmasked_logits[0, 1].detach().cpu().float(), dim=-1)
    b_pool_raw_values = unmasked_logits[0, 1, b_indices].detach().cpu().float()
    b_pool_raw_probabilities = F.softmax(b_pool_raw_values, dim=-1)
    b_pool_temperature_values = (
        masked_logits[0, 1, b_indices].detach().cpu().float() / max(temperature, 1e-6)
    )
    b_pool_temperature_probabilities = F.softmax(b_pool_temperature_values, dim=-1)
    b_actual = rank_and_probability(
        b_pool_temperature_values,
        b_symbols,
        source_b,
        topk,
    )
    incompatible_b = [
        symbol
        for symbol in b_symbols
        if not compatible_a_symbols(symbol, x_symbol, family)
    ]
    incompatible_positions = [b_symbols.index(symbol) for symbol in incompatible_b]
    incompatible_indices = [generation.SPECIES_LIST.index(symbol) for symbol in incompatible_b]
    actual_no_a_mass = sum(
        probability
        for symbol, probability in zip(
            b_actual["topk_symbols"],
            b_actual["topk_probabilities"],
        )
        if symbol in incompatible_b
    )

    compatible_a = compatible_a_symbols(source_b, x_symbol, family)
    source_a_allowed = source_a in compatible_a
    if compatible_a:
        a_indices = [generation.SPECIES_LIST.index(symbol) for symbol in compatible_a]
        a_temperature_values = (
            masked_logits[0, 0, a_indices].detach().cpu().float()
            / max(temperature, 1e-6)
        )
        a_actual = rank_and_probability(
            a_temperature_values,
            compatible_a,
            source_a,
            topk,
        )
    else:
        a_actual = {
            "rank": None,
            "topk_included": False,
            "topk_probability": 0.0,
            "topk_symbols": [],
            "topk_probabilities": [],
        }
    source_b_pool_probability = (
        float(b_pool_raw_probabilities[b_symbols.index(source_b)])
        if source_b in b_symbols
        else 0.0
    )
    source_b_temperature_probability = (
        float(b_pool_temperature_probabilities[b_symbols.index(source_b)])
        if source_b in b_symbols
        else 0.0
    )
    source_b_all_probability = (
        float(b_raw_all[generation.SPECIES_LIST.index(source_b)])
        if source_b in generation.SPECIES_LIST
        else 0.0
    )
    source_a_all_probability = float(
        F.softmax(unmasked_logits[0, 0].detach().cpu().float(), dim=-1)[
            generation.SPECIES_LIST.index(source_a)
        ]
    )
    return {
        "source_B_rank_in_family_pool": b_actual["rank"],
        "source_B_in_actual_topk": b_actual["topk_included"],
        "source_B_raw_all_species_probability": source_b_all_probability,
        "source_B_raw_family_pool_probability": source_b_pool_probability,
        "source_B_temperature_family_pool_probability": source_b_temperature_probability,
        "source_B_actual_post_topk_probability": b_actual["topk_probability"],
        "B_actual_topk_symbols_json": json.dumps(b_actual["topk_symbols"]),
        "B_actual_topk_probabilities_json": json.dumps(b_actual["topk_probabilities"]),
        "B_choices_without_compatible_A_json": json.dumps(incompatible_b),
        "B_raw_all_species_mass_without_compatible_A": float(
            b_raw_all[incompatible_indices].sum()
        ) if incompatible_indices else 0.0,
        "B_raw_family_pool_mass_without_compatible_A": float(
            b_pool_raw_probabilities[incompatible_positions].sum()
        ) if incompatible_positions else 0.0,
        "B_temperature_pre_topk_mass_without_compatible_A": float(
            b_pool_temperature_probabilities[incompatible_positions].sum()
        ) if incompatible_positions else 0.0,
        "B_actual_post_topk_mass_without_compatible_A": float(actual_no_a_mass),
        "source_conditional_compatible_A_json": json.dumps(compatible_a),
        "source_A_charge_compatible_given_source_B_X": source_a_allowed,
        "source_A_rank_in_conditional_pool": a_actual["rank"],
        "source_A_in_actual_conditional_topk": a_actual["topk_included"],
        "source_A_raw_all_species_probability": source_a_all_probability,
        "source_A_actual_conditional_post_topk_probability": a_actual[
            "topk_probability"
        ],
        "A_actual_topk_symbols_json": json.dumps(a_actual["topk_symbols"]),
        "A_actual_topk_probabilities_json": json.dumps(a_actual["topk_probabilities"]),
        "source_AB_actual_sampling_probability": float(
            b_actual["topk_probability"] * a_actual["topk_probability"]
        ),
        "source_AB_topk_reachable": bool(
            b_actual["topk_included"] and a_actual["topk_included"]
        ),
        "anti_repeat_state": "fresh empty counts; all B weights equal 1",
    }


def finite_heads(outputs: tuple[torch.Tensor, ...]) -> bool:
    return all(audit.finite_tensor(output) for output in outputs)


def trace_equal(
    left_structure: Structure | None,
    left: dict[str, Any],
    right_structure: Structure | None,
    right: dict[str, Any],
) -> bool:
    return bool(
        audit.structures_match_trace(left_structure, right_structure)
        and left == right
    )


def normal_csv_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def previous_zero_matches(
    previous: pd.Series,
    structure: Structure | None,
    trace: dict[str, Any],
) -> bool:
    formula = structure.composition.reduced_formula if structure is not None else None
    return bool(
        audit.parse_bool(previous["accepted"]) == bool(trace["accepted"])
        and int(previous["attempts_used"]) == int(trace["attempts_used"])
        and normal_csv_value(previous["sampled_A"]) == trace["A"]
        and normal_csv_value(previous["sampled_B"]) == trace["B"]
        and normal_csv_value(previous["sampled_X"]) == trace["X"]
        and normal_csv_value(previous["formula"]) == formula
    )


def effect_summary(
    rows: list[dict[str, Any]],
    left_arm: str,
    left_mode: str,
    right_arm: str,
    right_mode: str,
    metric: str,
    left_minus_right: bool = True,
    family: str | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if family is None or row["family"] == family]
    indexed = {
        (row["material_id"], row["latent_arm"], row["geometry_mode"]): row
        for row in selected
    }
    groups: dict[str, list[float]] = {}
    material_ids = sorted({row["material_id"] for row in selected})
    missing = 0
    for material_id in material_ids:
        left = indexed.get((material_id, left_arm, left_mode))
        right = indexed.get((material_id, right_arm, right_mode))
        if left is None or right is None:
            continue
        left_value = left.get(metric)
        right_value = right.get(metric)
        if left_value is None or right_value is None:
            missing += 1
            continue
        difference = float(left_value) - float(right_value)
        if not left_minus_right:
            difference = -difference
        groups.setdefault(str(left["composition_group"]), []).append(difference)
    result = audit.bootstrap_group_mean(groups)
    result["missing_pairs"] = missing
    result["metric"] = metric
    result["comparison"] = (
        f"{left_arm}/{left_mode} - {right_arm}/{right_mode}"
        if left_minus_right
        else f"{right_arm}/{right_mode} - {left_arm}/{left_mode}"
    )
    return result


def old_a_to_template_effect(
    raw_rows: list[dict[str, Any]],
    old_raw: pd.DataFrame,
    template_metric: str,
    old_metric: str,
    template_minus_a: bool,
) -> dict[str, Any]:
    template = {
        row["material_id"]: row
        for row in raw_rows
        if row["latent_arm"] == "joint" and row["geometry_mode"] == "template"
    }
    old_joint = old_raw[old_raw["latent_path"] == "joint"].set_index("material_id")
    groups: dict[str, list[float]] = {}
    for material_id, row in template.items():
        a_value = old_joint.loc[material_id, old_metric]
        t_value = row[template_metric]
        if pd.isna(a_value) or t_value is None:
            continue
        difference = float(t_value) - float(a_value)
        if not template_minus_a:
            difference = -difference
        groups.setdefault(str(row["composition_group"]), []).append(difference)
    return audit.bootstrap_group_mean(groups)


def raw_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in LATENT_ARMS:
        for mode in GEOMETRY_MODES:
            for family in ("oxide", "nitride"):
                chosen = [
                    row
                    for row in rows
                    if row["latent_arm"] == arm
                    and row["geometry_mode"] == mode
                    and row["family"] == family
                ]
                if not chosen:
                    continue
                matcher_status = Counter(row["structure_matcher_status"] for row in chosen)
                output.append(
                    {
                        "latent_arm": arm,
                        "geometry_mode": mode,
                        "family": family,
                        "material_count": len(chosen),
                        "mean_unmasked_nll": float(
                            np.mean([row["unmasked_nll_nats_per_site"] for row in chosen])
                        ),
                        "mean_masked_ab_nll": float(
                            np.mean([row["masked_ab_nll_nats_per_site"] for row in chosen])
                        ),
                        "mean_unmasked_top1_accuracy": float(
                            np.mean([row["unmasked_top1_accuracy"] for row in chosen])
                        ),
                        "mean_masked_ab_top1_accuracy": float(
                            np.mean([row["masked_ab_top1_accuracy"] for row in chosen])
                        ),
                        "mean_unmasked_true_AB_probability_product": float(
                            np.mean(
                                [
                                    row["unmasked_true_AB_probability_product"]
                                    for row in chosen
                                ]
                            )
                        ),
                        "mean_masked_true_AB_probability_product": float(
                            np.mean(
                                [
                                    row["masked_true_AB_probability_product"]
                                    for row in chosen
                                ]
                            )
                        ),
                        "raw_source_AB_top1_count": sum(
                            bool(row["raw_source_AB_top1"]) for row in chosen
                        ),
                        "raw_exact_composition_count": sum(
                            bool(row["unmasked_exact_composition"]) for row in chosen
                        ),
                        "matcher_status_counts_json": json.dumps(
                            dict(matcher_status),
                            sort_keys=True,
                        ),
                        "scale_false_match_count": sum(
                            bool(row["structure_matcher_scale_false"]) for row in chosen
                        ),
                        "mean_lattice_length_mae_angstrom": float(
                            np.mean([row["lattice_length_mae_angstrom"] for row in chosen])
                        ),
                        "mean_periodic_coordinate_rms_angstrom": float(
                            np.mean([row["same_slot_periodic_rms_angstrom"] for row in chosen])
                        ),
                        "mean_minimum_periodic_distance_angstrom": float(
                            np.mean(
                                [
                                    row["minimum_periodic_distance_angstrom"]
                                    for row in chosen
                                    if row["minimum_periodic_distance_angstrom"] is not None
                                ]
                            )
                        ),
                    }
                )
    return output


def sampling_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in LATENT_ARMS:
        for mode in GEOMETRY_MODES:
            for family in ("oxide", "nitride"):
                chosen = [
                    row
                    for row in rows
                    if row["latent_arm"] == arm
                    and row["geometry_mode"] == mode
                    and row["family"] == family
                ]
                if not chosen:
                    continue
                output.append(
                    {
                        "latent_arm": arm,
                        "geometry_mode": mode,
                        "family": family,
                        "material_count": len(chosen),
                        "mean_B_raw_all_species_mass_without_compatible_A": float(
                            np.mean(
                                [
                                    row[
                                        "B_raw_all_species_mass_without_compatible_A"
                                    ]
                                    for row in chosen
                                ]
                            )
                        ),
                        "mean_B_raw_family_pool_mass_without_compatible_A": float(
                            np.mean(
                                [
                                    row[
                                        "B_raw_family_pool_mass_without_compatible_A"
                                    ]
                                    for row in chosen
                                ]
                            )
                        ),
                        "mean_B_temperature_pre_topk_mass_without_compatible_A": float(
                            np.mean(
                                [
                                    row[
                                        "B_temperature_pre_topk_mass_without_compatible_A"
                                    ]
                                    for row in chosen
                                ]
                            )
                        ),
                        "mean_B_actual_post_topk_mass_without_compatible_A": float(
                            np.mean(
                                [
                                    row[
                                        "B_actual_post_topk_mass_without_compatible_A"
                                    ]
                                    for row in chosen
                                ]
                            )
                        ),
                        "source_B_in_topk_count": sum(
                            bool(row["source_B_in_actual_topk"]) for row in chosen
                        ),
                        "source_A_charge_compatible_count": sum(
                            bool(row["source_A_charge_compatible_given_source_B_X"])
                            for row in chosen
                        ),
                        "source_A_in_conditional_topk_count": sum(
                            bool(row["source_A_in_actual_conditional_topk"])
                            for row in chosen
                        ),
                        "source_AB_topk_reachable_count": sum(
                            bool(row["source_AB_topk_reachable"]) for row in chosen
                        ),
                    }
                )
    return output


def generation_summary_rows(
    calls: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    oracle_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in LATENT_ARMS:
        for family in ("oxide", "nitride", "all"):
            chosen_calls = [
                row
                for row in calls
                if row["latent_arm"] == arm
                and (family == "all" or row["family"] == family)
            ]
            chosen_attempts = [
                row
                for row in attempts
                if row["latent_arm"] == arm
                and (family == "all" or row["family"] == family)
            ]
            if not chosen_calls:
                continue
            accepted = [row for row in chosen_calls if row["accepted"]]
            formulas = Counter(row["formula"] for row in accepted)
            rejection_counts: Counter[str] = Counter()
            for row in chosen_calls:
                rejection_counts.update(json.loads(row["rejection_stage_counts_json"]))
            material_ids = sorted({row["material_id"] for row in chosen_calls})
            exportable_ids = {
                material_id
                for material_id in material_ids
                if oracle_by_id[material_id]["deterministically_exportable"]
            }
            charge_ids = {
                material_id
                for material_id in material_ids
                if oracle_by_id[material_id]["existential_charge_neutrality_from_tables"]
            }
            exportable_attempts = [
                row for row in chosen_attempts if row["material_id"] in exportable_ids
            ]
            exportable_calls = [
                row for row in chosen_calls if row["material_id"] in exportable_ids
            ]
            output.append(
                {
                    "latent_arm": arm,
                    "family": family,
                    "unique_source_materials": len(material_ids),
                    "charge_neutral_source_materials": len(charge_ids),
                    "deterministically_exportable_source_materials": len(exportable_ids),
                    "decode_calls": len(chosen_calls),
                    "consumed_attempts": len(chosen_attempts),
                    "constructed_attempts": sum(
                        bool(row["constructed"]) for row in chosen_attempts
                    ),
                    "accepted_calls": len(accepted),
                    "accepted_calls_per_consumed_attempt": (
                        len(accepted) / len(chosen_attempts) if chosen_attempts else None
                    ),
                    "accepted_call_fraction": len(accepted) / len(chosen_calls),
                    "unique_accepted_compositions": len(formulas),
                    "accepted_composition_counts_json": json.dumps(
                        dict(formulas),
                        sort_keys=True,
                    ),
                    "source_materials_with_accepted_output": len(
                        {row["material_id"] for row in accepted}
                    ),
                    "attempt_source_AB_recovery_count": sum(
                        bool(row["source_AB_sampled"]) for row in chosen_attempts
                    ),
                    "attempt_source_AB_recovery_rate": (
                        sum(bool(row["source_AB_sampled"]) for row in chosen_attempts)
                        / len(chosen_attempts)
                        if chosen_attempts
                        else None
                    ),
                    "accepted_source_AB_recovery_count": sum(
                        bool(row["passed_source_AB"]) for row in chosen_attempts
                    ),
                    "accepted_source_AB_recovery_rate_per_attempt": (
                        sum(bool(row["passed_source_AB"]) for row in chosen_attempts)
                        / len(chosen_attempts)
                        if chosen_attempts
                        else None
                    ),
                    "endpoint_source_AB_recovery_count": sum(
                        bool(row["source_AB_recovered"]) for row in chosen_calls
                    ),
                    "oracle_exportable_decode_calls": len(exportable_calls),
                    "oracle_exportable_consumed_attempts": len(exportable_attempts),
                    "oracle_exportable_attempt_source_AB_recovery_count": sum(
                        bool(row["source_AB_sampled"]) for row in exportable_attempts
                    ),
                    "oracle_exportable_accepted_source_AB_recovery_count": sum(
                        bool(row["passed_source_AB"]) for row in exportable_attempts
                    ),
                    "oracle_exportable_endpoint_source_AB_recovery_count": sum(
                        bool(row["source_AB_recovered"]) for row in exportable_calls
                    ),
                    "rejection_stage_counts_json": json.dumps(
                        dict(rejection_counts),
                        sort_keys=True,
                    ),
                }
            )
    return output


def paired_generation_effect(
    calls: list[dict[str, Any]],
    control_arm: str,
    oracle_only: bool,
) -> dict[str, Any]:
    selected = [
        row for row in calls if not oracle_only or row["source_deterministically_exportable"]
    ]
    known = {
        (row["material_id"], row["seed"]): row
        for row in selected
        if row["latent_arm"] == "joint"
    }
    control = {
        (row["material_id"], row["seed"]): row
        for row in selected
        if row["latent_arm"] == control_arm
    }
    if set(known) != set(control):
        raise AssertionError(f"Unpaired final calls for {control_arm}")
    groups: dict[str, list[float]] = {}
    endpoints_changed = 0
    both_passed = 0
    formula_changed = 0
    by_seed: dict[int, list[float]] = {}
    for key in sorted(known):
        left = known[key]
        right = control[key]
        difference = float(left["source_AB_recovered"]) - float(
            right["source_AB_recovered"]
        )
        groups.setdefault(str(left["composition_group"]), []).append(difference)
        by_seed.setdefault(int(left["seed"]), []).append(difference)
        left_endpoint = (left["accepted"], left["formula"])
        right_endpoint = (right["accepted"], right["formula"])
        endpoints_changed += left_endpoint != right_endpoint
        if left["accepted"] and right["accepted"]:
            both_passed += 1
            formula_changed += left["formula"] != right["formula"]
    effect = audit.bootstrap_group_mean(groups)
    return {
        "control": control_arm,
        "oracle_exportable_only": oracle_only,
        "paired_calls": len(known),
        "joint_minus_control_endpoint_source_AB_recovery": effect,
        "difference_by_seed": {
            str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())
        },
        "endpoint_changed_fraction": endpoints_changed / len(known) if known else None,
        "both_passed": both_passed,
        "formula_changed_given_both_passed_fraction": (
            formula_changed / both_passed if both_passed else None
        ),
        "zero_interval_interpretation": (
            "If [0,0], this is a degenerate resample of observed zero paired differences, "
            "not evidence of a precisely zero population effect."
        ),
    }


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def markdown_report(summary: dict[str, Any]) -> str:
    return render_markdown_report(summary, LATENT_ARMS)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    audit_csv = args.audit_csv.resolve()
    cif_root = args.cif_root.resolve()
    train_csv = args.train_csv.resolve()
    train_cif_root = args.train_cif_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    report_dir = args.report_dir.resolve()

    required = [
        audit_csv,
        cif_root,
        train_csv,
        train_cif_root,
        checkpoint_path,
        PROTOCOL_PATH,
        OLD_RAW_PATH,
        OLD_SAMPLE_PATH,
        OLD_YIELD_PATH,
        OLD_SUMMARY_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required follow-up assets are missing: {missing}")
    frozen_configuration = {
        "seeds": list(args.seeds) == [0, 1, 2],
        "decode_temperature": args.decode_temperature == 1.25,
        "decode_topk": args.decode_topk == 12,
        "decode_tries": args.decode_tries == 12,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED") == "0",
    }
    if not all(frozen_configuration.values()):
        raise AssertionError(f"Frozen follow-up configuration failed: {frozen_configuration}")
    if sha256_file(audit_csv) != EXPECTED_AUDIT_SHA256:
        raise AssertionError("Audit CSV hash differs from the completed audit")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("Checkpoint hash differs from the completed audit")

    report_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash_before_run = sha256_file(PROTOCOL_PATH)
    protected = protected_files(audit_csv, cif_root, checkpoint_path)
    protected_before = file_hashes(protected)
    split_hashes_before = {
        name: sha256_file(DATA_ROOT / f"{name}.csv")
        for name in ("train", "val", "test")
    }

    frame, loaded = audit.load_audit_inputs(audit_csv, cif_root)
    template_coordinates, training_convention = derive_training_template_convention(
        train_csv,
        train_cif_root,
    )
    template_center, input_manifest = template_input_manifest(
        template_coordinates,
        training_convention,
    )
    write_json(report_dir / "input_manifest.json", input_manifest)

    reachability_rows, oracle_by_id = reference_reachability_rows(frame, refine=True)
    charge_rows = family_charge_rows(refine=True)
    write_csv(report_dir / "reference_reachability.csv", reachability_rows)
    write_csv(report_dir / "allowed_combinations.csv", charge_rows)

    checkpoint, checkpoint_report = audit.checkpoint_summary(checkpoint_path)
    main_model, main_model_report = audit.instantiate_model(checkpoint)
    generation_model, generation_model_report = audit.load_generation_model(checkpoint)
    main_model.eval().requires_grad_(False)
    generation_model.eval().requires_grad_(False)
    model_checks = {
        "training_model_eval": not main_model.training,
        "generation_model_eval": not generation_model.training,
        "training_model_frozen": not any(
            parameter.requires_grad for parameter in main_model.parameters()
        ),
        "generation_model_frozen": not any(
            parameter.requires_grad for parameter in generation_model.parameters()
        ),
        "training_model_strict_load": bool(main_model_report["strict_state_dict_load"]),
        "generation_model_strict_load": bool(
            generation_model_report["strict_state_dict_load"]
        ),
    }
    if not all(model_checks.values()):
        raise AssertionError(f"Model load/frozen checks failed: {model_checks}")

    prepared: dict[str, dict[str, Any]] = {}
    latent_stats: dict[str, Any] = {}
    with torch.inference_mode():
        for _, row in frame.iterrows():
            material_id = str(row["material_id"])
            sample = loaded[material_id]
            crystal = sample["crystal_vec"].unsqueeze(0)
            heat = sample["heat_all"].reshape(1)
            gap = sample["dir_gap"].reshape(1)
            paths, metadata, _, _, _ = audit.latent_bundle(
                main_model,
                crystal,
                heat,
                gap,
                material_id,
            )
            prepared[material_id] = {
                "row": row,
                "crystal": crystal,
                "heat": heat,
                "gap": gap,
                "context": audit.source_context(row),
                "paths": paths,
            }
            latent_stats[material_id] = metadata
        fixed_donor_id = str(frame.iloc[0]["material_id"])
        fixed_real = prepared[fixed_donor_id]["paths"]["joint"].detach().clone()
        for material_id, item in prepared.items():
            item["paths"]["fixed_real"] = fixed_real.clone()
            latent_stats[material_id]["fixed_real"] = audit.tensor_stats(fixed_real)

    old_raw = pd.read_csv(OLD_RAW_PATH, dtype={"material_id": "string"})
    old_samples = pd.read_csv(OLD_SAMPLE_PATH, dtype={"material_id": "string"})
    old_yield = pd.read_csv(OLD_YIELD_PATH)
    old_summary = json.loads(OLD_SUMMARY_PATH.read_text(encoding="utf-8"))
    old_sample_index = {
        (str(row["material_id"]), str(row["latent_path"]), int(row["seed"])): row
        for _, row in old_samples.iterrows()
        if str(row["latent_path"]) in LATENT_ARMS
    }
    old_raw_index = {
        (str(row["material_id"]), str(row["latent_path"])): row
        for _, row in old_raw.iterrows()
        if str(row["latent_path"]) in LATENT_ARMS
    }

    adapters = {
        mode: GeometryInputAdapter(
            generation_model,
            mode,
            template_coordinates,
            template_center,
        )
        for mode in GEOMETRY_MODES
    }
    raw_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    all_raw_finite = True
    all_zero_raw_equal = True
    all_zero_metrics_match_previous = True
    lattice_unchanged = True

    with torch.inference_mode():
        for material_id in frame["material_id"].astype(str):
            item = prepared[material_id]
            row = item["row"]
            crystal = item["crystal"]
            context = item["context"]
            family_mask = generation.build_species_mask(context["allowed_anions"])
            oracle = oracle_by_id[material_id]
            for arm in LATENT_ARMS:
                latent = item["paths"][arm]
                direct_zero_unmasked = audit.zero_reference_decode(
                    generation_model,
                    latent,
                    None,
                )
                mode_outputs: dict[str, dict[str, tuple[torch.Tensor, ...]]] = {}
                for mode in GEOMETRY_MODES:
                    adapter = adapters[mode]
                    mode_outputs[mode] = {
                        "unmasked": audit.zero_reference_decode(adapter, latent, None),
                        "masked": audit.zero_reference_decode(
                            adapter,
                            latent,
                            family_mask,
                        ),
                    }
                zero_equal = all(
                    torch.equal(left, right)
                    for left, right in zip(
                        direct_zero_unmasked,
                        mode_outputs["zero"]["unmasked"],
                    )
                )
                all_zero_raw_equal = all_zero_raw_equal and zero_equal
                lattice_unchanged = lattice_unchanged and torch.equal(
                    mode_outputs["zero"]["unmasked"][0],
                    mode_outputs["template"]["unmasked"][0],
                )
                zero_pairwise = torch.cdist(
                    mode_outputs["zero"]["unmasked"][3][0, :5],
                    mode_outputs["zero"]["unmasked"][3][0, :5],
                )
                template_pairwise = torch.cdist(
                    mode_outputs["template"]["unmasked"][3][0, :5],
                    mode_outputs["template"]["unmasked"][3][0, :5],
                )

                for mode in GEOMETRY_MODES:
                    unmasked_outputs = mode_outputs[mode]["unmasked"]
                    masked_outputs = mode_outputs[mode]["masked"]
                    all_raw_finite = all_raw_finite and finite_heads(unmasked_outputs)
                    unmasked_information, unmasked_sites = audit.species_information_metrics(
                        unmasked_outputs[2],
                        crystal,
                    )
                    masked_information, masked_sites = audit.species_information_metrics(
                        masked_outputs[2],
                        crystal,
                        family_mask,
                        scored_slots=[0, 1],
                    )
                    reconstruction = audit.reconstruction_metrics(
                        crystal,
                        unmasked_outputs[0],
                        unmasked_outputs[2],
                        unmasked_outputs[3],
                        context,
                    )
                    raw_geometry = reconstruction["raw_geometry"]
                    zero_to_mode_tv = audit.occupied_site_total_variation(
                        mode_outputs["zero"]["unmasked"][2],
                        unmasked_outputs[2],
                        crystal,
                    )
                    zero_to_mode_masked_tv = audit.occupied_site_total_variation(
                        mode_outputs["zero"]["masked"][2],
                        masked_outputs[2],
                        crystal,
                        family_mask,
                        scored_slots=[0, 1],
                    )
                    unmasked_ab_product = float(
                        unmasked_sites[0]["true_probability"]
                        * unmasked_sites[1]["true_probability"]
                    )
                    masked_ab_product = float(
                        masked_sites[0]["true_probability"]
                        * masked_sites[1]["true_probability"]
                    )
                    previous = old_raw_index[(material_id, arm)]
                    previous_match = bool(
                        math.isclose(
                            float(previous["unmasked_nll_nats_per_site"]),
                            float(unmasked_information["nll_nats_per_site"]),
                            rel_tol=audit.RTOL,
                            abs_tol=audit.ATOL,
                        )
                        and math.isclose(
                            float(previous["masked_ab_nll_nats_per_site"]),
                            float(masked_information["nll_nats_per_site"]),
                            rel_tol=audit.RTOL,
                            abs_tol=audit.ATOL,
                        )
                    ) if mode == "zero" else None
                    if mode == "zero":
                        all_zero_metrics_match_previous = (
                            all_zero_metrics_match_previous and previous_match
                        )
                    raw_row = {
                        "material_id": material_id,
                        "composition_group": str(row["composition_group"]),
                        "latent_arm": arm,
                        "geometry_mode": mode,
                        "family": context["family"],
                        "source_A": context["A"],
                        "source_B": context["B"],
                        "source_X": context["X"],
                        "source_charge_neutral": oracle[
                            "existential_charge_neutrality_from_tables"
                        ],
                        "source_deterministically_exportable": oracle[
                            "deterministically_exportable"
                        ],
                        "no_reference_coordinates_or_species_supplied": True,
                        "input_tensor_is_global": True,
                        "input_coordinates_sha256": (
                            input_manifest["coordinates_sha256"]
                            if mode == "template"
                            else sha256_tensor_bytes(torch.zeros_like(template_coordinates))
                        ),
                        "input_center_sha256": (
                            input_manifest["center_sha256"]
                            if mode == "template"
                            else sha256_tensor_bytes(torch.zeros_like(template_center))
                        ),
                        "latent_l2_norm": float(torch.linalg.vector_norm(latent)),
                        "all_unmasked_heads_finite": finite_heads(unmasked_outputs),
                        "zero_adapter_raw_heads_exact": zero_equal if mode == "zero" else None,
                        "zero_metrics_match_previous_audit": previous_match,
                        "unmasked_nll_nats_per_site": unmasked_information[
                            "nll_nats_per_site"
                        ],
                        "unmasked_top1_accuracy": unmasked_information["top1_accuracy"],
                        "unmasked_exact_composition": unmasked_information[
                            "exact_composition"
                        ],
                        "unmasked_predicted_reduced_formula": unmasked_information[
                            "predicted_reduced_formula"
                        ],
                        "unmasked_true_A_probability": unmasked_sites[0][
                            "true_probability"
                        ],
                        "unmasked_true_B_probability": unmasked_sites[1][
                            "true_probability"
                        ],
                        "unmasked_true_AB_probability_product": unmasked_ab_product,
                        "masked_ab_nll_nats_per_site": masked_information[
                            "nll_nats_per_site"
                        ],
                        "masked_ab_top1_accuracy": masked_information["top1_accuracy"],
                        "masked_true_A_probability": masked_sites[0]["true_probability"],
                        "masked_true_B_probability": masked_sites[1]["true_probability"],
                        "masked_true_AB_probability_product": masked_ab_product,
                        "raw_source_A_top1": reconstruction["role_exact"]["A"],
                        "raw_source_B_top1": reconstruction["role_exact"]["B"],
                        "raw_source_AB_top1": bool(
                            reconstruction["role_exact"]["A"]
                            and reconstruction["role_exact"]["B"]
                        ),
                        "raw_predicted_roles_json": json.dumps(
                            reconstruction["role_predictions"],
                            sort_keys=True,
                        ),
                        "structure_matcher_status": reconstruction[
                            "structure_matcher_status"
                        ],
                        "structure_matcher_scale_false": reconstruction[
                            "structure_matcher_scale_false"
                        ],
                        "structure_matcher_scale_true": reconstruction[
                            "structure_matcher_scale_true"
                        ],
                        "lattice_length_mae_angstrom": reconstruction[
                            "mean_lattice_length_absolute_error_angstrom"
                        ],
                        "lattice_angle_mae_degrees": reconstruction[
                            "mean_lattice_angle_absolute_error_degrees"
                        ],
                        "relative_volume_error": reconstruction["relative_volume_error"],
                        "same_slot_periodic_rms_angstrom": reconstruction[
                            "same_slot_periodic_rms_angstrom"
                        ],
                        "species_matched_periodic_rms_angstrom": reconstruction[
                            "species_matched_periodic_rms_angstrom"
                        ],
                        "minimum_periodic_distance_angstrom": raw_geometry[
                            "first_five_minimum_periodic_distance_angstrom"
                        ],
                        "legal_cell": raw_geometry["legal_cell"],
                        "zero_to_mode_unmasked_probability_tv": zero_to_mode_tv["mean"],
                        "zero_to_mode_masked_ab_probability_tv": zero_to_mode_masked_tv[
                            "mean"
                        ],
                        "zero_to_mode_species_logits_max_abs": audit.difference_stats(
                            mode_outputs["zero"]["unmasked"][2],
                            unmasked_outputs[2],
                        )["maximum_absolute"],
                        "zero_to_mode_coordinate_head_max_abs": audit.difference_stats(
                            mode_outputs["zero"]["unmasked"][3],
                            unmasked_outputs[3],
                        )["maximum_absolute"],
                        "zero_to_mode_output_pairwise_mean_abs": float(
                            torch.abs(template_pairwise - zero_pairwise).mean()
                        ) if mode == "template" else 0.0,
                        "lattice_head_zero_to_mode_max_abs": audit.difference_stats(
                            mode_outputs["zero"]["unmasked"][0],
                            unmasked_outputs[0],
                        )["maximum_absolute"],
                    }
                    raw_rows.append(raw_row)
                    sampling_rows.append(
                        {
                            "material_id": material_id,
                            "composition_group": str(row["composition_group"]),
                            "latent_arm": arm,
                            "geometry_mode": mode,
                            "family": context["family"],
                            "source_A": context["A"],
                            "source_B": context["B"],
                            "source_X": context["X"],
                            "source_charge_neutral": oracle[
                                "existential_charge_neutrality_from_tables"
                            ],
                            "source_deterministically_exportable": oracle[
                                "deterministically_exportable"
                            ],
                            "decode_temperature": args.decode_temperature,
                            "decode_topk": args.decode_topk,
                            "x_prior_strength": 0.0,
                            **sampling_reachability(
                                unmasked_outputs[2],
                                masked_outputs[2],
                                context,
                                args.decode_temperature,
                                args.decode_topk,
                            ),
                        }
                    )

    write_csv(report_dir / "per_example_metrics.csv", raw_rows)
    write_csv(report_dir / "sampling_reachability.csv", sampling_rows)
    raw_by_family = raw_family_rows(raw_rows)
    sampling_by_family = sampling_family_rows(sampling_rows)
    write_csv(report_dir / "raw_by_family.csv", raw_by_family)
    write_csv(report_dir / "sampling_by_family.csv", sampling_by_family)

    call_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    zero_trace_parity = True
    template_baseline_parity = True
    previous_zero_reproduced = True
    attempt_accounting = True
    accepted_structures: list[tuple[dict[str, Any], Structure]] = []

    with torch.inference_mode():
        for material_id in frame["material_id"].astype(str):
            item = prepared[material_id]
            row = item["row"]
            context = item["context"]
            family_mask = generation.build_species_mask(context["allowed_anions"])
            oracle = oracle_by_id[material_id]
            for arm in LATENT_ARMS:
                latent = item["paths"][arm]
                for base_seed in args.seeds:
                    resolved_seed = generation.seed_from_target(
                        int(base_seed),
                        float(item["gap"].item()),
                        float(item["heat"].item()),
                    )
                    audit.reset_random_state(resolved_seed)
                    original_zero_structure, original_zero_trace = (
                        audit.traced_generation_decode(
                            generation_model,
                            latent,
                            context["allowed_anions"],
                            family_mask,
                            np.random.RandomState(resolved_seed),
                            float(item["gap"].item()),
                            context["family"],
                            args.decode_temperature,
                            args.decode_topk,
                            args.decode_tries,
                            refine=True,
                        )
                    )
                    audit.reset_random_state(resolved_seed)
                    adapter_zero_structure, adapter_zero_trace = audit.traced_generation_decode(
                        adapters["zero"],
                        latent,
                        context["allowed_anions"],
                        family_mask,
                        np.random.RandomState(resolved_seed),
                        float(item["gap"].item()),
                        context["family"],
                        args.decode_temperature,
                        args.decode_topk,
                        args.decode_tries,
                        refine=True,
                    )
                    current_zero_equal = trace_equal(
                        original_zero_structure,
                        original_zero_trace,
                        adapter_zero_structure,
                        adapter_zero_trace,
                    )
                    zero_trace_parity = zero_trace_parity and current_zero_equal

                    audit.reset_random_state(resolved_seed)
                    template_structure, template_trace = audit.traced_generation_decode(
                        adapters["template"],
                        latent,
                        context["allowed_anions"],
                        family_mask,
                        np.random.RandomState(resolved_seed),
                        float(item["gap"].item()),
                        context["family"],
                        args.decode_temperature,
                        args.decode_topk,
                        args.decode_tries,
                        refine=True,
                    )
                    audit.reset_random_state(resolved_seed)
                    (
                        baseline_template_structure,
                        baseline_template_info,
                        baseline_template_gap,
                        baseline_template_heat,
                    ) = audit.baseline_generation_decode(
                        adapters["template"],
                        latent,
                        context["allowed_anions"],
                        family_mask,
                        resolved_seed,
                        float(item["gap"].item()),
                        context["family"],
                        args.decode_temperature,
                        args.decode_topk,
                        args.decode_tries,
                        refine=True,
                    )
                    current_template_parity = audit.endpoint_matches_baseline(
                        template_structure,
                        template_trace,
                        baseline_template_structure,
                        baseline_template_info,
                        baseline_template_gap,
                        baseline_template_heat,
                    )
                    template_baseline_parity = (
                        template_baseline_parity and current_template_parity
                    )
                    previous = old_sample_index[(material_id, arm, int(base_seed))]
                    current_previous_match = previous_zero_matches(
                        previous,
                        original_zero_structure,
                        original_zero_trace,
                    )
                    previous_zero_reproduced = (
                        previous_zero_reproduced and current_previous_match
                    )
                    accounting_ok = (
                        len(template_trace["attempt_records"])
                        == int(template_trace["attempts_used"])
                        and int(template_trace["constructed_count"])
                        == sum(
                            bool(record["constructed"])
                            for record in template_trace["attempt_records"]
                        )
                    )
                    attempt_accounting = attempt_accounting and accounting_ok
                    accepted = bool(template_trace["accepted"])
                    source_a = accepted and template_trace["A"] == context["A"]
                    source_b = accepted and template_trace["B"] == context["B"]
                    formula = (
                        template_structure.composition.reduced_formula
                        if template_structure is not None
                        else None
                    )
                    zero_formula = (
                        original_zero_structure.composition.reduced_formula
                        if original_zero_structure is not None
                        else None
                    )
                    template_sequence = [
                        (
                            record.get("sampled_A"),
                            record.get("sampled_B"),
                            record.get("sampled_X"),
                            record.get("rejection_stage"),
                            bool(record.get("passed")),
                        )
                        for record in template_trace["attempt_records"]
                    ]
                    zero_sequence = [
                        (
                            record.get("sampled_A"),
                            record.get("sampled_B"),
                            record.get("sampled_X"),
                            record.get("rejection_stage"),
                            bool(record.get("passed")),
                        )
                        for record in original_zero_trace["attempt_records"]
                    ]
                    rejection_counts = Counter(
                        event["stage"] for event in template_trace["rejection_trace"]
                    )
                    result = {
                        "material_id": material_id,
                        "composition_group": str(row["composition_group"]),
                        "latent_arm": arm,
                        "seed": int(base_seed),
                        "resolved_target_seed": int(resolved_seed),
                        "family": context["family"],
                        "source_A": context["A"],
                        "source_B": context["B"],
                        "source_X": context["X"],
                        "source_charge_neutral": oracle[
                            "existential_charge_neutrality_from_tables"
                        ],
                        "source_deterministically_exportable": oracle[
                            "deterministically_exportable"
                        ],
                        "accepted": accepted,
                        "attempt_budget": args.decode_tries,
                        "attempts_used": int(template_trace["attempts_used"]),
                        "constructed_count": int(template_trace["constructed_count"]),
                        "sampled_A": template_trace["A"],
                        "sampled_B": template_trace["B"],
                        "sampled_X": template_trace["X"],
                        "source_A_recovered": source_a,
                        "source_B_recovered": source_b,
                        "source_AB_recovered": bool(source_a and source_b),
                        "formula": formula,
                        "structure_signature": (
                            generation.struct_signature(template_structure)
                            if template_structure is not None
                            else None
                        ),
                        "rejection_stage_counts_json": json.dumps(
                            dict(rejection_counts),
                            sort_keys=True,
                        ),
                        "zero_adapter_trace_exact": current_zero_equal,
                        "zero_original_matches_previous_audit": current_previous_match,
                        "template_trace_matches_unmodified_generation_function": current_template_parity,
                        "zero_endpoint_formula": zero_formula,
                        "template_endpoint_changed_from_zero": (
                            (accepted, formula)
                            != (bool(original_zero_trace["accepted"]), zero_formula)
                        ),
                        "attempt_sequence_changed_from_zero": template_sequence
                        != zero_sequence,
                        "branch_dependent_rng_divergence_recorded": True,
                    }
                    call_rows.append(result)
                    if template_structure is not None:
                        accepted_structures.append((result, template_structure))
                    for record in template_trace["attempt_records"]:
                        sampled_a = record.get("sampled_A")
                        sampled_b = record.get("sampled_B")
                        sampled_x = record.get("sampled_X")
                        source_ab_sampled = (
                            sampled_a == context["A"] and sampled_b == context["B"]
                        )
                        attempt_rows.append(
                            {
                                "material_id": material_id,
                                "composition_group": str(row["composition_group"]),
                                "latent_arm": arm,
                                "seed": int(base_seed),
                                "resolved_target_seed": int(resolved_seed),
                                "family": context["family"],
                                "attempt": int(record["attempt"]),
                                "constructed": bool(record["constructed"]),
                                "passed": bool(record["passed"]),
                                "rejection_stage": record.get("rejection_stage"),
                                "reason": record.get("reason"),
                                "sampled_A": sampled_a,
                                "sampled_B": sampled_b,
                                "sampled_X": sampled_x,
                                "source_AB_sampled": source_ab_sampled,
                                "passed_source_AB": bool(
                                    record["passed"] and source_ab_sampled
                                ),
                                "source_charge_neutral": oracle[
                                    "existential_charge_neutrality_from_tables"
                                ],
                                "source_deterministically_exportable": oracle[
                                    "deterministically_exportable"
                                ],
                            }
                        )

    write_csv(report_dir / "per_seed_metrics.csv", call_rows)
    write_csv(report_dir / "attempts.csv", attempt_rows)
    write_csv(
        report_dir / "failures.csv",
        [row for row in attempt_rows if not row["passed"]],
    )

    generation_yield = generation_summary_rows(call_rows, attempt_rows, oracle_by_id)
    write_csv(report_dir / "yield_by_family.csv", generation_yield)

    raw_comparisons = {
        "zero_minus_template_joint_unmasked_nll": effect_summary(
            raw_rows,
            "joint",
            "zero",
            "joint",
            "template",
            "unmasked_nll_nats_per_site",
        ),
        "zero_minus_template_joint_masked_ab_nll": effect_summary(
            raw_rows,
            "joint",
            "zero",
            "joint",
            "template",
            "masked_ab_nll_nats_per_site",
        ),
        "template_minus_zero_joint_masked_ab_accuracy": effect_summary(
            raw_rows,
            "joint",
            "template",
            "joint",
            "zero",
            "masked_ab_top1_accuracy",
        ),
        "template_fixed_minus_joint_unmasked_nll": effect_summary(
            raw_rows,
            "fixed_real",
            "template",
            "joint",
            "template",
            "unmasked_nll_nats_per_site",
        ),
        "template_fixed_minus_joint_masked_ab_nll": effect_summary(
            raw_rows,
            "fixed_real",
            "template",
            "joint",
            "template",
            "masked_ab_nll_nats_per_site",
        ),
        "template_property_minus_joint_unmasked_nll": effect_summary(
            raw_rows,
            "property",
            "template",
            "joint",
            "template",
            "unmasked_nll_nats_per_site",
        ),
        "template_property_minus_joint_masked_ab_nll": effect_summary(
            raw_rows,
            "property",
            "template",
            "joint",
            "template",
            "masked_ab_nll_nats_per_site",
        ),
        "template_joint_minus_fixed_masked_AB_probability": effect_summary(
            raw_rows,
            "joint",
            "template",
            "fixed_real",
            "template",
            "masked_true_AB_probability_product",
        ),
        "template_joint_minus_property_masked_AB_probability": effect_summary(
            raw_rows,
            "joint",
            "template",
            "property",
            "template",
            "masked_true_AB_probability_product",
        ),
        "template_minus_A_joint_unmasked_nll": old_a_to_template_effect(
            raw_rows,
            old_raw,
            "unmasked_nll_nats_per_site",
            "reference_nll_nats_per_site",
            template_minus_a=True,
        ),
        "template_minus_A_joint_masked_ab_nll": old_a_to_template_effect(
            raw_rows,
            old_raw,
            "masked_ab_nll_nats_per_site",
            "reference_masked_ab_nll_nats_per_site",
            template_minus_a=True,
        ),
        "joint_template_vs_zero_unmasked_tv_mean": mean_or_none(
            [
                row["zero_to_mode_unmasked_probability_tv"]
                for row in raw_rows
                if row["latent_arm"] == "joint"
                and row["geometry_mode"] == "template"
            ]
        ),
        "joint_template_vs_zero_masked_ab_tv_mean": mean_or_none(
            [
                row["zero_to_mode_masked_ab_probability_tv"]
                for row in raw_rows
                if row["latent_arm"] == "joint"
                and row["geometry_mode"] == "template"
            ]
        ),
    }

    reference_summary: dict[str, Any] = {
        "all_count": len(reachability_rows),
        "declared_scope_count": sum(
            bool(row["csv_generation_scope_compatible"]) for row in reachability_rows
        ),
        "charge_neutral_count": sum(
            bool(row["existential_charge_neutrality_from_tables"])
            for row in reachability_rows
        ),
        "deterministically_exportable_count": sum(
            bool(row["deterministically_exportable"]) for row in reachability_rows
        ),
        "all_required_radii_available_count": sum(
            bool(row["all_required_ionic_radii_available"])
            for row in reachability_rows
        ),
        "charge_neutral_roles": [
            f"{row['source_A']}{row['source_B']}{row['source_X']}3"
            for row in reachability_rows
            if row["existential_charge_neutrality_from_tables"]
        ],
        "deterministically_exportable_roles": [
            f"{row['source_A']}{row['source_B']}{row['source_X']}3"
            for row in reachability_rows
            if row["deterministically_exportable"]
        ],
        "by_family": {},
        "first_rejection_counts": dict(
            Counter(
                str(row["first_rejection_stage"])
                for row in reachability_rows
                if not row["deterministically_exportable"]
            )
        ),
    }
    for family in ("oxide", "nitride"):
        selected = [row for row in reachability_rows if row["family"] == family]
        reference_summary["by_family"][family] = {
            "count": len(selected),
            "charge_neutral": sum(
                bool(row["existential_charge_neutrality_from_tables"])
                for row in selected
            ),
            "deterministically_exportable": sum(
                bool(row["deterministically_exportable"]) for row in selected
            ),
        }

    enumerated_space: dict[str, Any] = {}
    for family in ("oxide", "nitride"):
        selected = [row for row in charge_rows if row["family"] == family]
        enumerated_space[family] = {
            "B_choices": len(selected),
            "B_with_compatible_A": sum(
                bool(row["has_any_charge_compatible_A"]) for row in selected
            ),
            "charge_compatible_AB_pairs": sum(
                int(row["compatible_A_count"]) for row in selected
            ),
            "B_with_exportable_A": sum(
                int(row["deterministically_exportable_A_count"]) > 0
                for row in selected
            ),
            "exportable_AB_pairs": sum(
                int(row["deterministically_exportable_A_count"]) for row in selected
            ),
        }

    previous_zero_by_family = {}
    for family in ("oxide", "nitride"):
        selected = old_yield[
            (old_yield["latent_path"] == "joint")
            & (old_yield["grouping"] == "family")
            & (old_yield["condition"] == family)
        ]
        if len(selected) != 1:
            raise AssertionError(f"Missing prior joint family baseline for {family}")
        old_row = selected.iloc[0]
        previous_zero_by_family[family] = {
            "decode_calls": int(old_row["decode_calls"]),
            "N_attempt": int(old_row["N_attempt"]),
            "N_pass": int(old_row["N_pass"]),
            "N_unique_composition": int(old_row["N_unique_composition"]),
        }

    sampling_summary: dict[str, Any] = {"joint_template": {}}
    for family in ("oxide", "nitride"):
        selected = [
            row
            for row in sampling_rows
            if row["latent_arm"] == "joint"
            and row["geometry_mode"] == "template"
            and row["family"] == family
        ]
        sampling_summary["joint_template"][
            f"{family}_no_compatible_A_mass"
        ] = mean_or_none(
            [row["B_actual_post_topk_mass_without_compatible_A"] for row in selected]
        )
        sampling_summary["joint_template"][f"{family}_source_B_topk_count"] = sum(
            bool(row["source_B_in_actual_topk"]) for row in selected
        )
        sampling_summary["joint_template"][f"{family}_source_AB_reachable_count"] = sum(
            bool(row["source_AB_topk_reachable"]) for row in selected
        )

    fixed_nll_effect = raw_comparisons[
        "template_fixed_minus_joint_masked_ab_nll"
    ]
    interface_nll_effect = raw_comparisons[
        "zero_minus_template_joint_masked_ab_nll"
    ]
    joint_template_rows = [
        row
        for row in raw_rows
        if row["latent_arm"] == "joint" and row["geometry_mode"] == "template"
    ]
    joint_attempts = [row for row in attempt_rows if row["latent_arm"] == "joint"]
    recovery_evidence = bool(
        any(row["raw_source_AB_top1"] for row in joint_template_rows)
        or any(row["source_AB_sampled"] for row in joint_attempts)
    )
    restoration_supported = bool(
        interface_nll_effect["ci95_low"] is not None
        and interface_nll_effect["ci95_low"] > 0
        and fixed_nll_effect["ci95_low"] is not None
        and fixed_nll_effect["ci95_low"] > 0
        and recovery_evidence
    )
    if restoration_supported:
        fixed_geometry_decision = "SUPPORTED WITHIN THE FROZEN TEMPLATE SCOPE"
        fixed_explanation = (
            "T met the pre-run joint-versus-zero, joint-versus-fixed, and recovery gates. "
            "This supports a template-interface effect only."
        )
    else:
        fixed_geometry_decision = "NOT ESTABLISHED"
        failed = []
        if interface_nll_effect["ci95_low"] is None or interface_nll_effect["ci95_low"] <= 0:
            failed.append("zero-to-T masked NLL interval did not exclude zero in T's favor")
        if fixed_nll_effect["ci95_low"] is None or fixed_nll_effect["ci95_low"] <= 0:
            failed.append("joint T did not show a group-level masked-NLL advantage over fixed-real")
        if not recovery_evidence:
            failed.append("no deterministic raw or sampled source-A/B recovery was observed")
        fixed_explanation = "The pre-run restoration gate failed: " + "; ".join(failed) + "."

    joint_oxide_yield = next(
        row
        for row in generation_yield
        if row["latent_arm"] == "joint" and row["family"] == "oxide"
    )
    oxide_noncollapsed = bool(
        joint_oxide_yield["accepted_calls"] > 0
        and joint_oxide_yield["unique_accepted_compositions"] > 1
    )
    pilot_justified = restoration_supported and oxide_noncollapsed
    if pilot_justified:
        next_experiment = "SMALL EXPLORATORY OXIDE-ONLY TEMPLATE-CONSTRAINED PILOT"
        pilot_explanation = (
            "A small exploratory oxide-only pilot is justified under the shared T interface and "
            "unchanged rules because raw reference information is restored and one of the two "
            "exportable references is recovered in every seed. Final paired recovery intervals "
            "still include zero because only two references are eligible, so this is a bounded "
            "mechanistic pilot, not a population-level performance claim. Nitrides are excluded "
            "because the frozen deterministic rule intersection is empty."
        )
    else:
        next_experiment = "TRAIN-ONLY FROZEN-LATENT COMPOSITION READOUT PROBE"
        pilot_explanation = (
            "A diffusion pilot is not justified by this diagnostic. The smallest next experiment "
            "is a train-only linear or shallow composition readout on frozen latents, evaluated on "
            "the unchanged validation audit, to separate latent representation from decoder-interface loss."
        )

    protected_after = file_hashes(protected)
    protected_unchanged = protected_before == protected_after
    split_hashes_after = {
        name: sha256_file(DATA_ROOT / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    hard_checks = {
        **frozen_configuration,
        **model_checks,
        "audit_hash_frozen": sha256_file(audit_csv) == EXPECTED_AUDIT_SHA256,
        "checkpoint_hash_frozen": sha256_file(checkpoint_path)
        == EXPECTED_CHECKPOINT_SHA256,
        "protocol_unchanged_during_run": sha256_file(PROTOCOL_PATH)
        == protocol_hash_before_run,
        "protected_old_assets_unchanged": protected_unchanged,
        "split_hashes_unchanged": split_hashes_before == split_hashes_after,
        "all_raw_heads_finite": all_raw_finite,
        "zero_adapter_raw_heads_exact": all_zero_raw_equal,
        "zero_metrics_match_previous_audit": all_zero_metrics_match_previous,
        "zero_adapter_traces_exact": zero_trace_parity,
        "zero_original_reproduces_previous_audit": previous_zero_reproduced,
        "template_trace_matches_unmodified_generation_function": template_baseline_parity,
        "attempt_accounting_complete": attempt_accounting
        and len(attempt_rows) == sum(row["attempts_used"] for row in call_rows),
        "lattice_head_unchanged_by_geometry_input": lattice_unchanged,
        "all_reference_radii_available": reference_summary[
            "all_required_radii_available_count"
        ]
        == reference_summary["all_count"],
        "no_final_test_inference": True,
    }
    execution_integrity = "PASSED" if all(hard_checks.values()) else "FAILED"

    final_effects = {
        control: {
            "all_examples": paired_generation_effect(call_rows, control, False),
            "oracle_exportable_only": paired_generation_effect(call_rows, control, True),
        }
        for control in ("fixed_real", "property")
    }
    summary = {
        "status": "complete" if execution_integrity == "PASSED" else "failed_integrity",
        "execution_integrity": execution_integrity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "criteria_timing": (
            "Protocol recorded before this follow-up inference, after the completed decoder "
            "audit was inspected; informed follow-up, not retroactive preregistration."
        ),
        "label_compatibility": "unresolved; CMR labels used as provisional inputs only",
        "pretrained_training_overlap": "unknown",
        "test_set_model_inference": False,
        "environment": {
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "scipy", "pymatgen", "torch")
            },
            "device": "cpu",
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "git": git_state(),
        },
        "configuration": {
            "audit_examples": len(frame),
            "seeds": list(args.seeds),
            "latent_arms": list(LATENT_ARMS),
            "geometry_modes": list(GEOMETRY_MODES),
            "decode_temperature": args.decode_temperature,
            "decode_topk": args.decode_topk,
            "decode_tries_maximum": args.decode_tries,
            "refine": True,
            "x_prior_strength": 0.0,
            "target_B_filter": False,
            "fixed_real_donor_id": fixed_donor_id,
            "rng_history_reset_per_call": True,
            "branch_dependent_rng_divergence": (
                "Same initial seeds; changed categorical outcomes/rejections can desynchronize "
                "later draws, so post-branch attempts are not draw-matched."
            ),
        },
        "assets": {
            "audit_csv": display_path(audit_csv),
            "audit_csv_sha256": sha256_file(audit_csv),
            "train_csv": display_path(train_csv),
            "train_csv_sha256": sha256_file(train_csv),
            "checkpoint": checkpoint_report,
            "protocol": display_path(PROTOCOL_PATH),
            "protocol_sha256": protocol_hash_before_run,
            "previous_audit_summary": display_path(OLD_SUMMARY_PATH),
            "previous_audit_summary_sha256": sha256_file(OLD_SUMMARY_PATH),
            "script": display_path(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "template_input": input_manifest,
        "reference_reachability": reference_summary,
        "enumerated_constraint_space": enumerated_space,
        "raw_comparisons": raw_comparisons,
        "raw_by_family": raw_by_family,
        "sampling_by_family": sampling_by_family,
        "previous_reference_geometry_controls": {
            "A_reference_assisted": old_summary["raw_geometry"]["A_reference_assisted"],
            "B_reference_free": old_summary["raw_geometry"]["B_reference_free"],
        },
        "sampling_summary": sampling_summary,
        "template_generation": {
            "yield_by_family": generation_yield,
            "paired_control_effects": final_effects,
            "call_count": len(call_rows),
            "consumed_attempt_count": len(attempt_rows),
            "accepted_structure_count": len(accepted_structures),
        },
        "previous_zero_baseline_by_family": previous_zero_by_family,
        "decisions": {
            "fixed_geometry_restoration": fixed_geometry_decision,
            "fixed_geometry_explanation": fixed_explanation,
            "constraint_explanation": (
                f"Only {reference_summary['deterministically_exportable_count']}/20 references "
                "survive the current deterministic rules, and the enumerated nitride output "
                "space is empty after tolerance/μ filtering. Source-B/A top-k pruning is reported "
                "separately from those hard constraints."
            ),
            "small_diffusion_pilot_justified": pilot_justified,
            "next_experiment": next_experiment,
            "pilot_explanation": pilot_explanation,
        },
        "hard_checks": hard_checks,
        "old_audit_integrity_record": {
            "execution_integrity": old_summary["execution_integrity"],
            "all_instrumented_endpoints_match_baseline": old_summary["harness_checks"][
                "all_instrumented_endpoints_match_baseline"
            ],
            "old_reports_preserved": protected_unchanged,
        },
        "protected_asset_hashes_unchanged": protected_unchanged,
        "protected_asset_sha256": {
            display_path(path): digest for path, digest in protected_before.items()
        },
    }

    write_json(report_dir / "summary.json", summary)
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "environment": summary["environment"],
        "configuration": summary["configuration"],
        "assets": summary["assets"],
        "template_input_file": display_path(report_dir / "input_manifest.json"),
        "template_input_file_sha256": sha256_file(report_dir / "input_manifest.json"),
        "protocol_frozen_before_inference": True,
        "criteria_informed_by_previous_audit": True,
        "ordinary_generation_received_source_species": False,
        "original_generation_function_modified": False,
        "adapter_changes_only_decoder_coordinates_and_center": True,
        "hard_checks": hard_checks,
    }
    write_json(report_dir / "run_manifest.json", manifest)
    report_path = report_dir / "report.md"
    report_path.write_text(markdown_report(summary), encoding="utf-8", newline="\n")

    artifact_hashes = {
        display_path(path): sha256_file(path)
        for path in sorted(report_dir.iterdir())
        if path.is_file() and path.name != "summary.json"
    }
    summary["artifact_sha256"] = artifact_hashes
    write_json(report_dir / "summary.json", summary)

    print(f"Template reachability follow-up complete: {report_path}")
    print(
        "  reference reachability:",
        {
            "charge_neutral": reference_summary["charge_neutral_count"],
            "deterministically_exportable": reference_summary[
                "deterministically_exportable_count"
            ],
        },
    )
    print(
        "  template joint accepted by family:",
        {
            family: next(
                row["accepted_calls"]
                for row in generation_yield
                if row["latent_arm"] == "joint" and row["family"] == family
            )
            for family in ("oxide", "nitride")
        },
    )
    print(f"  decision: {next_experiment}")
    if execution_integrity != "PASSED":
        print(f"  failed hard checks: {[name for name, value in hard_checks.items() if not value]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
