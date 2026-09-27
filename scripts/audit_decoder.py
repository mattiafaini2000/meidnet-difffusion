#!/usr/bin/env python3
"""Run a small, frozen decoder-interface audit on known validation latents.

The CMR labels are used as provisional numerical inputs.  This script does not
claim label calibration, held-out generalization, stability, or property
accuracy.  It never sends reference coordinates, species, centers, masks, or
atom counts into the reference-free generation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
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
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

WORKSPACE = Path(__file__).resolve().parents[1]
MEIDNET_ROOT = WORKSPACE / "MEIDNet-main"
SCRIPTS_ROOT = WORKSPACE / "scripts"
DEFAULT_DATA_ROOT = WORKSPACE / "data" / "processed" / "cmr_reconstructed"
DEFAULT_CHECKPOINT = (
    MEIDNET_ROOT
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)
PROTOCOL_PATH = WORKSPACE / "reports" / "decoder_audit_protocol.md"

for import_root in (SCRIPTS_ROOT, MEIDNET_ROOT, WORKSPACE):
    value = str(import_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from validate_setup import checkpoint_summary, instantiate_model  # noqa: E402
from meidnet.model import (  # noqa: E402
    MAX_SITES,
    NUM_SPECIES,
    SPECIES_LIST,
    TripleModalityDataset,
)
from meidnet import design as generation  # noqa: E402
from meidnet_audits.files import package_version, sha256_file, write_csv  # noqa: E402
from meidnet_audits.decoder_report import (  # noqa: E402
    _legacy_markdown_report as render_legacy_markdown_report,
    markdown_report as render_markdown_report,
)


LATENT_PATHS = (
    "joint",
    "fixed_real",
    "property",
    "crystal",
    "joint_perturb_0p01",
    "joint_perturb_0p05",
)
PERTURBATION_RATIOS = (0.01, 0.05)
RTOL = 1e-5
ATOL = 1e-6
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_RESAMPLES = 10_000
FAMILY_ANIONS = {"oxide": {"O"}, "nitride": {"N"}}
EXPECTED_AUDIT_SHA256 = "afab4524786ef88c82ee310d36bf9a0b6ccd2011142fc695934522cab2efa804"
EXPECTED_CHECKPOINT_SHA256 = "f9493781d5bbb05dfe106874269496c4c1c0364ae9e543703625c8d1efd60c87"
EXPECTED_PROTOCOL_SHA256 = "ce6b21e5b933cde0ab57c4783f6564ab57c6bff58bcc2bb65281d8273c53c769"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_DATA_ROOT / "audit.csv")
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_DATA_ROOT / "cifs" / "val")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=WORKSPACE / "reports")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--decode-temperature", type=float, default=1.25)
    parser.add_argument("--decode-topk", type=int, default=12)
    parser.add_argument("--decode-tries", type=int, default=12)
    parser.add_argument("--representative-cifs", type=int, default=8)
    parser.add_argument("--no-refine", action="store_true")
    return parser.parse_args()


def git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu().float()
    finite = torch.isfinite(detached)
    return {
        "shape": list(detached.shape),
        "finite": bool(finite.all().item()),
        "finite_fraction": float(finite.float().mean().item()),
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
        "l2_norm": float(torch.linalg.vector_norm(detached)),
    }


def difference_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {tuple(left.shape)} and {tuple(right.shape)}")
    difference = (left.detach().cpu().float() - right.detach().cpu().float()).abs()
    return {
        "mean_absolute": float(difference.mean()),
        "maximum_absolute": float(difference.max()),
        "root_mean_square": float(torch.sqrt(torch.mean(difference.square()))),
        "exactly_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
    }


DECODER_HEAD_NAMES = ("lattice", "adjacency", "species_logits", "coordinates")


def new_repeat_summary() -> dict[str, Any]:
    return {
        "call_pairs": 0,
        "all_heads_within_tolerance": True,
        "maximum_absolute_difference": 0.0,
        "heads": {
            name: {
                "within_tolerance": True,
                "maximum_absolute_difference": 0.0,
            }
            for name in DECODER_HEAD_NAMES
        },
    }


def update_repeat_summary(
    summary: dict[str, Any],
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
) -> None:
    """Accumulate repeated-call agreement without conflating batch duplication."""
    if len(first) != len(DECODER_HEAD_NAMES) or len(second) != len(DECODER_HEAD_NAMES):
        raise ValueError("Decoder repeat comparison requires all four decoder heads")
    summary["call_pairs"] += 1
    for name, left, right in zip(DECODER_HEAD_NAMES, first, second):
        difference = difference_stats(left, right)
        within_tolerance = bool(
            torch.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=False)
        )
        head = summary["heads"][name]
        head["within_tolerance"] = bool(head["within_tolerance"] and within_tolerance)
        maximum = difference["maximum_absolute"]
        if (
            math.isfinite(maximum)
            and head["maximum_absolute_difference"] is not None
            and summary["maximum_absolute_difference"] is not None
        ):
            head["maximum_absolute_difference"] = max(
                float(head["maximum_absolute_difference"]),
                maximum,
            )
            summary["maximum_absolute_difference"] = max(
                float(summary["maximum_absolute_difference"]),
                maximum,
            )
        else:
            head["maximum_absolute_difference"] = None
            summary["maximum_absolute_difference"] = None
        summary["all_heads_within_tolerance"] = bool(
            summary["all_heads_within_tolerance"] and within_tolerance
        )


def load_audit_inputs(audit_csv: Path, cif_root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = pd.read_csv(audit_csv, dtype={"material_id": "string"})
    if len(frame) != 20:
        raise ValueError(f"Expected the documented 20-row audit sample, found {len(frame)}")
    if frame["material_id"].duplicated().any():
        raise ValueError("Audit material IDs are not unique")
    if not frame["generation_scope_compatible"].map(parse_bool).all():
        raise ValueError("Every audit row must be generation-scope compatible")

    dataset = TripleModalityDataset(str(cif_root), str(audit_csv))
    loaded: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        material_id = str(sample["material_id"])
        if material_id in loaded:
            raise ValueError(f"Duplicate loader material ID: {material_id}")
        loaded[material_id] = sample
    expected_ids = set(frame["material_id"].astype(str))
    if set(loaded) != expected_ids:
        raise ValueError(
            f"Loader membership differs from audit CSV: missing={sorted(expected_ids-set(loaded))}, "
            f"extra={sorted(set(loaded)-expected_ids)}"
        )
    frame = frame.sort_values("material_id").reset_index(drop=True)
    return frame, loaded


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def load_generation_model(checkpoint: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = generation.DualAutoencoderModel(
        generation.MAX_SITES,
        generation.NUM_SPECIES,
        latent_dim=int(checkpoint["latent_dim_common"]),
    )
    state = generation.remap_checkpoint_keys(checkpoint["model_state_dict"])
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Generation model strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval().requires_grad_(False)
    return model, {
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def latent_bundle(
    model: torch.nn.Module,
    crystal: torch.Tensor,
    heat: torch.Tensor,
    gap: torch.Tensor,
    material_id: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_crystal, center = model.crystal_encoder(crystal)
    raw_property = model.property_encoder(torch.stack((heat, gap), dim=1))
    common_crystal = F.normalize(model.proj_crystal(F.normalize(raw_crystal, dim=1)), dim=1)
    common_property = F.normalize(model.proj_prop(F.normalize(raw_property, dim=1)), dim=1)
    joint = (common_crystal + common_property) / 2.0

    perturbations: dict[str, torch.Tensor] = {}
    perturbation_metadata: dict[str, Any] = {}
    joint_norm = torch.linalg.vector_norm(joint, dim=1, keepdim=True)
    if bool((joint_norm <= 0).any()):
        raise ValueError(f"Cannot make relative perturbation of zero latent for {material_id}")
    seed_text = f"perturb-v1:{material_id}"
    direction_seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator(device="cpu").manual_seed(direction_seed)
    direction = torch.randn(joint.shape, generator=generator, dtype=joint.dtype)
    direction = F.normalize(direction, dim=1).to(joint.device)
    for ratio in PERTURBATION_RATIOS:
        label = f"joint_perturb_0p{int(round(ratio * 100)):02d}"
        delta = ratio * joint_norm * direction
        perturbations[label] = joint + delta
        perturbation_metadata[label] = {
            "direction_seed": direction_seed,
            "requested_relative_l2": ratio,
            "delta_l2": float(torch.linalg.vector_norm(delta).item()),
            "observed_relative_l2": float(
                (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(joint)).item()
            ),
            "renormalized": False,
        }

    species_offset = 6 + MAX_SITES * MAX_SITES
    species_length = MAX_SITES * NUM_SPECIES
    coordinates_offset = species_offset + species_length
    input_species = crystal[:, species_offset:coordinates_offset].reshape(
        crystal.shape[0], MAX_SITES, NUM_SPECIES
    )
    input_coordinates = crystal[:, coordinates_offset:].reshape(crystal.shape[0], MAX_SITES, 3)
    paths = {
        "joint": joint,
        "property": common_property,
        "crystal": common_crystal,
        **perturbations,
    }
    metadata = {
        "raw_crystal": tensor_stats(raw_crystal),
        "raw_property": tensor_stats(raw_property),
        "common_crystal": tensor_stats(common_crystal),
        "common_property": tensor_stats(common_property),
        "joint": tensor_stats(joint),
        **{name: tensor_stats(value) for name, value in perturbations.items()},
        "crystal_property_cosine": float(F.cosine_similarity(common_crystal, common_property).item()),
        "crystal_property_l2": float(torch.linalg.vector_norm(common_crystal - common_property).item()),
        "perturbations": perturbation_metadata,
        "joint_was_renormalized": False,
    }
    return paths, metadata, center, input_species, input_coordinates


def source_context(row: pd.Series) -> dict[str, Any]:
    source_symbols = json.loads(str(row["source_site_symbols"]))
    parsed_symbols = json.loads(str(row["meidnet_site_symbols"]))
    permutation = [int(value) for value in json.loads(str(row["site_permutation"]))]
    if len(source_symbols) != 5 or len(parsed_symbols) != 5 or len(permutation) != 5:
        raise ValueError(f"Unexpected five-site metadata for {row['material_id']}")
    if sorted(permutation) != list(range(5)):
        raise ValueError(f"Invalid site permutation for {row['material_id']}")
    mapping_valid = all(
        source_symbols[source_index] == parsed_symbols[parser_index]
        for source_index, parser_index in enumerate(permutation)
    )
    if not mapping_valid:
        raise ValueError(f"Recorded source-to-parser mapping is invalid for {row['material_id']}")
    anions = source_symbols[2:5]
    if len(set(anions)) != 1:
        raise ValueError(f"Audit record is not single-anion: {row['material_id']}")
    anion = anions[0]
    canonical = [
        str(row["source_A_ion"]),
        str(row["source_B_ion"]),
        anion,
        anion,
        anion,
    ]
    if source_symbols != canonical:
        raise ValueError(f"Source symbols do not follow recorded A/B/X roles for {row['material_id']}")
    family = generation.FAMILY_OF_X.get(anion)
    if family not in FAMILY_ANIONS:
        raise ValueError(f"Unsupported audit family {family!r} for {row['material_id']}")
    if anion not in FAMILY_ANIONS[family]:
        raise ValueError(f"Source anion is inconsistent with declared family for {row['material_id']}")
    return {
        "source_symbols": source_symbols,
        "parsed_symbols": parsed_symbols,
        "source_to_parser": permutation,
        "site_mapping_valid": mapping_valid,
        "site_mapping_identity": permutation == list(range(5)),
        "A": str(row["source_A_ion"]),
        "B": str(row["source_B_ion"]),
        "X": anion,
        "family": family,
        "allowed_anions": set(FAMILY_ANIONS[family]),
    }


def predicted_symbols(logits: torch.Tensor) -> list[str]:
    indices = logits.detach().cpu().argmax(dim=-1).tolist()
    return [SPECIES_LIST[int(index)] for index in indices]


def target_site_tensors(crystal: torch.Tensor) -> tuple[torch.Tensor, list[int], list[int]]:
    species_offset = 6 + MAX_SITES * MAX_SITES
    species_length = MAX_SITES * NUM_SPECIES
    target = crystal[0, species_offset : species_offset + species_length].reshape(
        MAX_SITES,
        NUM_SPECIES,
    )
    occupied = torch.nonzero(target.sum(dim=1) > 0.5, as_tuple=False).reshape(-1).tolist()
    true_indices = [int(target[index].argmax()) for index in occupied]
    return target, occupied, true_indices


def species_information_metrics(
    logits: torch.Tensor,
    crystal: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
    scored_slots: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score occupied parser slots; padding never enters the denominator."""
    target, occupied, _ = target_site_tensors(crystal)
    if scored_slots is not None:
        if not scored_slots or not set(scored_slots).issubset(occupied):
            raise ValueError("Scored slots must be a non-empty subset of occupied parser slots")
        occupied = list(scored_slots)
    true_indices = [int(target[index].argmax()) for index in occupied]
    selected = logits[0, occupied].detach().cpu().float()
    recorded_logits = selected.clone()
    restriction_failures = torch.zeros(len(occupied), dtype=torch.bool)
    empty_allowed_rows = torch.zeros(len(occupied), dtype=torch.bool)
    if allowed_mask is not None:
        selected_mask = allowed_mask[occupied].detach().cpu().bool()
        empty_allowed_rows = selected_mask.sum(dim=-1) == 0
        true_tensor = torch.tensor(true_indices, dtype=torch.long)
        restriction_failures = ~selected_mask[torch.arange(len(occupied)), true_tensor]
        selected = selected.masked_fill(~selected_mask, float("-inf"))
    log_probabilities = F.log_softmax(selected, dim=-1)
    probabilities = log_probabilities.exp()
    probability_sums = probabilities.sum(dim=-1)
    true_tensor = torch.tensor(true_indices, dtype=torch.long)
    true_probabilities = probabilities[torch.arange(len(occupied)), true_tensor]
    true_log_probabilities = log_probabilities[torch.arange(len(occupied)), true_tensor]
    predicted_indices = probabilities.argmax(dim=-1)
    probability_rows_valid = bool(
        torch.isfinite(probabilities).all()
        and not bool(empty_allowed_rows.any())
        and torch.allclose(
            probability_sums,
            torch.ones_like(probability_sums),
            rtol=RTOL,
            atol=ATOL,
        )
    )
    nll = None
    nll_valid_sites = None
    valid_reference_sites = ~restriction_failures
    if probability_rows_valid and not bool(restriction_failures.any()):
        nll = float((-true_log_probabilities).mean().item())
    if probability_rows_valid and bool(valid_reference_sites.any()):
        nll_valid_sites = float((-true_log_probabilities[valid_reference_sites]).mean().item())

    true_symbols = [SPECIES_LIST[index] for index in true_indices]
    predicted = [SPECIES_LIST[int(index)] for index in predicted_indices.tolist()]
    exact_composition = Counter(true_symbols) == Counter(predicted)
    if bool(restriction_failures.any()):
        nll_status = "positive_infinity_restriction_failure"
    elif probability_rows_valid:
        nll_status = "finite"
    else:
        nll_status = "undefined_invalid_probability_rows"
    summary = {
        "site_count": len(occupied),
        "nll_nats_per_site": nll,
        "nll_status": nll_status,
        "nll_is_infinite_due_to_restriction": bool(restriction_failures.any()),
        "nll_valid_sites_only": nll_valid_sites,
        "top1_accuracy": float((predicted_indices == true_tensor).float().mean().item()),
        "zero_true_probability_sites": int(restriction_failures.sum().item()),
        "restriction_failure_sites": int(restriction_failures.sum().item()),
        "empty_allowed_rows": int(empty_allowed_rows.sum().item()),
        "probability_rows_valid": probability_rows_valid,
        "exact_composition": exact_composition,
        "true_reduced_formula": Composition(Counter(true_symbols)).reduced_formula,
        "predicted_reduced_formula": Composition(Counter(predicted)).reduced_formula,
    }
    site_rows = []
    for local_index, slot in enumerate(occupied):
        true_probability = float(true_probabilities[local_index].item())
        restriction_failure = bool(restriction_failures[local_index].item())
        row_probability_valid = bool(
            torch.isfinite(probabilities[local_index]).all()
            and not bool(empty_allowed_rows[local_index].item())
            and torch.allclose(
                probability_sums[local_index],
                torch.ones_like(probability_sums[local_index]),
                rtol=RTOL,
                atol=ATOL,
            )
        )
        site_rows.append(
            {
                "parser_slot": int(slot),
                "true_symbol": true_symbols[local_index],
                "predicted_symbol": predicted[local_index],
                "true_probability": true_probability,
                "top1_probability": float(probabilities[local_index].max().item()),
                "negative_log_likelihood": (
                    float(-true_log_probabilities[local_index].item())
                    if not restriction_failure and row_probability_valid
                    else None
                ),
                "negative_log_likelihood_status": (
                    "positive_infinity_restriction_failure"
                    if restriction_failure
                    else (
                        "finite"
                        if row_probability_valid
                        else "undefined_invalid_probability_row"
                    )
                ),
                "zero_true_probability": restriction_failure,
                "restriction_failure": restriction_failure,
                "logits_json": json.dumps(
                    [float(value) for value in recorded_logits[local_index].tolist()]
                ),
                "probabilities_json": json.dumps(
                    [float(value) for value in probabilities[local_index].tolist()]
                ),
            }
        )
    return summary, site_rows


def occupied_site_total_variation(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    crystal: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
    scored_slots: list[int] | None = None,
) -> dict[str, Any]:
    _, occupied, _ = target_site_tensors(crystal)
    if scored_slots is not None:
        if not scored_slots or not set(scored_slots).issubset(occupied):
            raise ValueError("Scored slots must be a non-empty subset of occupied parser slots")
        occupied = list(scored_slots)
    left_selected = left_logits[0, occupied].detach().cpu().float()
    right_selected = right_logits[0, occupied].detach().cpu().float()
    if allowed_mask is not None:
        selected_mask = allowed_mask[occupied].detach().cpu().bool()
        if bool((selected_mask.sum(dim=-1) == 0).any()):
            raise ValueError("Cannot compute masked TV for a site with no allowed species")
        left_selected = left_selected.masked_fill(~selected_mask, float("-inf"))
        right_selected = right_selected.masked_fill(~selected_mask, float("-inf"))
    left = F.softmax(left_selected, dim=-1)
    right = F.softmax(right_selected, dim=-1)
    site_values = 0.5 * torch.abs(left - right).sum(dim=-1)
    return {
        "mean": float(site_values.mean().item()),
        "minimum": float(site_values.min().item()),
        "maximum": float(site_values.max().item()),
        "per_site": [float(value) for value in site_values.tolist()],
    }


def matcher(scale: bool) -> StructureMatcher:
    return StructureMatcher(
        ltol=0.3,
        stol=0.5,
        angle_tol=10,
        primitive_cell=False,
        scale=scale,
        attempt_supercell=False,
        allow_subset=False,
        comparator=ElementComparator(),
    )


def lattice_parameters(latent_lattice: torch.Tensor) -> list[float]:
    values = latent_lattice.detach().cpu().reshape(-1).numpy().astype(float)
    return [
        float(values[0] * 20.0),
        float(values[1] * 20.0),
        float(values[2] * 20.0),
        float(values[3] * 180.0),
        float(values[4] * 180.0),
        float(values[5] * 180.0),
    ]


def raw_geometry_metrics(latent_lattice: torch.Tensor, coordinates: torch.Tensor) -> dict[str, Any]:
    parameters = lattice_parameters(latent_lattice)
    lengths = parameters[:3]
    angles = parameters[3:]
    finite_parameters = all(math.isfinite(value) for value in parameters)
    legal_parameters = (
        finite_parameters
        and all(value > 0 for value in lengths)
        and all(0 < value < 180 for value in angles)
    )
    plausible = legal_parameters and all(2.0 <= value <= 12.0 for value in lengths) and all(
        30.0 <= value <= 150.0 for value in angles
    )
    result: dict[str, Any] = {
        "parameters": parameters,
        "finite_parameters": finite_parameters,
        "legal_cell": False,
        "finite_positive_volume": False,
        "simple_plausibility_window": plausible,
        "coordinate_min": float(coordinates.min()),
        "coordinate_max": float(coordinates.max()),
        "coordinate_finite_fraction": float(torch.isfinite(coordinates).float().mean().item()),
        "first_five_minimum_periodic_distance_angstrom": None,
        "cell_volume_angstrom3": None,
    }
    if not legal_parameters:
        return result
    try:
        lattice = Lattice.from_parameters(*parameters)
        result["legal_cell"] = bool(
            np.isfinite(lattice.matrix).all() and float(lattice.volume) > 0
        )
        result["finite_positive_volume"] = bool(
            math.isfinite(float(lattice.volume)) and float(lattice.volume) > 0
        )
        distance_matrix = lattice.get_all_distances(
            coordinates[:5].detach().cpu().numpy(),
            coordinates[:5].detach().cpu().numpy(),
        )
        distance_matrix[np.eye(5, dtype=bool)] = np.inf
        result["first_five_minimum_periodic_distance_angstrom"] = float(distance_matrix.min())
        result["cell_volume_angstrom3"] = float(lattice.volume)
    except Exception as error:
        result["geometry_error"] = f"{type(error).__name__}: {error}"
        result["legal_cell"] = False
        result["finite_positive_volume"] = False
    return result


def reconstruction_metrics(
    crystal: torch.Tensor,
    lattice_output: torch.Tensor,
    species_logits: torch.Tensor,
    coordinates_output: torch.Tensor,
    context: dict[str, Any],
) -> dict[str, Any]:
    crystal = crystal.detach().cpu()
    species_offset = 6 + MAX_SITES * MAX_SITES
    species_length = MAX_SITES * NUM_SPECIES
    target_species = crystal[0, species_offset : species_offset + species_length].reshape(
        MAX_SITES, NUM_SPECIES
    )
    target_coordinates = crystal[0, species_offset + species_length :].reshape(MAX_SITES, 3)
    occupied = torch.nonzero(
        target_species.sum(dim=1) > 0.5,
        as_tuple=False,
    ).reshape(-1).tolist()
    target_symbols = [SPECIES_LIST[int(target_species[index].argmax())] for index in occupied]
    all_predictions = predicted_symbols(species_logits[0])
    predicted_occupied = [all_predictions[index] for index in occupied]

    source_to_parser = context["source_to_parser"]
    role_predictions = {
        "A": all_predictions[source_to_parser[0]],
        "B": all_predictions[source_to_parser[1]],
        "X": [all_predictions[index] for index in source_to_parser[2:5]],
    }
    role_exact = {
        "A": role_predictions["A"] == context["A"],
        "B": role_predictions["B"] == context["B"],
        "X": all(symbol == context["X"] for symbol in role_predictions["X"]),
    }

    target_lattice = lattice_parameters(crystal[0, :6])
    lattice = Lattice.from_parameters(*target_lattice)
    target_fractional = target_coordinates[occupied].numpy()
    predicted_fractional = coordinates_output[0, occupied].detach().cpu().numpy()
    periodic_distances = lattice.get_all_distances(target_fractional, predicted_fractional)
    same_slot_distances = np.diag(periodic_distances)

    matched_rms = None
    matched_maximum = None
    if Counter(target_symbols) == Counter(predicted_occupied):
        costs = np.full((len(occupied), len(occupied)), 1e9, dtype=float)
        for source_index, source_symbol in enumerate(target_symbols):
            for predicted_index, predicted_symbol in enumerate(predicted_occupied):
                if source_symbol != predicted_symbol:
                    continue
                costs[source_index, predicted_index] = periodic_distances[
                    source_index,
                    predicted_index,
                ]
        source_indices, predicted_indices = linear_sum_assignment(costs)
        selected = costs[source_indices, predicted_indices]
        if np.all(selected < 1e8):
            matched_rms = float(np.sqrt(np.mean(selected**2)))
            matched_maximum = float(selected.max())

    predicted_lattice = lattice_parameters(lattice_output[0])
    raw_geometry = raw_geometry_metrics(lattice_output[0], coordinates_output[0])
    target_volume = float(lattice.volume)
    predicted_volume = raw_geometry["cell_volume_angstrom3"]
    relative_volume_error = None
    if predicted_volume is not None and target_volume > 0:
        relative_volume_error = abs(float(predicted_volume) - target_volume) / target_volume

    matcher_scale_false = False
    matcher_scale_true = False
    matcher_error = None
    if raw_geometry["legal_cell"] and Counter(target_symbols) == Counter(predicted_occupied):
        try:
            reference_structure = Structure(
                lattice,
                target_symbols,
                target_fractional,
                coords_are_cartesian=False,
            )
            predicted_structure = Structure(
                Lattice.from_parameters(*predicted_lattice),
                predicted_occupied,
                predicted_fractional,
                coords_are_cartesian=False,
            )
            matcher_scale_false = bool(matcher(scale=False).fit(reference_structure, predicted_structure))
            matcher_scale_true = bool(matcher(scale=True).fit(reference_structure, predicted_structure))
        except Exception as error:
            matcher_error = f"{type(error).__name__}: {error}"
    if not raw_geometry["legal_cell"]:
        matcher_status = "invalid"
    elif Counter(target_symbols) != Counter(predicted_occupied):
        matcher_status = "composition_incompatible"
    elif matcher_error is not None:
        matcher_status = "invalid"
    elif matcher_scale_false:
        matcher_status = "matched"
    else:
        matcher_status = "unmatched"
    return {
        "occupied_site_count": len(occupied),
        "target_parser_symbols": target_symbols,
        "predicted_parser_symbols": predicted_occupied,
        "slot_species_accuracy": sum(
            left == right for left, right in zip(target_symbols, predicted_occupied)
        )
        / len(occupied),
        "composition_exact": Counter(target_symbols) == Counter(predicted_occupied),
        "role_predictions": role_predictions,
        "role_exact": role_exact,
        "all_roles_exact": all(role_exact.values()),
        "same_slot_periodic_rms_angstrom": float(np.sqrt(np.mean(same_slot_distances**2))),
        "same_slot_periodic_maximum_angstrom": float(same_slot_distances.max()),
        "species_matched_periodic_rms_angstrom": matched_rms,
        "species_matched_periodic_maximum_angstrom": matched_maximum,
        "target_lattice_parameters": target_lattice,
        "predicted_lattice_parameters": predicted_lattice,
        "lattice_absolute_errors": [
            abs(predicted - target)
            for predicted, target in zip(predicted_lattice, target_lattice)
        ],
        "mean_lattice_length_absolute_error_angstrom": float(
            np.mean(np.abs(np.asarray(predicted_lattice[:3]) - np.asarray(target_lattice[:3])))
        ),
        "mean_lattice_angle_absolute_error_degrees": float(
            np.mean(np.abs(np.asarray(predicted_lattice[3:]) - np.asarray(target_lattice[3:])))
        ),
        "reference_volume_angstrom3": target_volume,
        "relative_volume_error": relative_volume_error,
        "structure_matcher_scale_false": matcher_scale_false,
        "structure_matcher_scale_true": matcher_scale_true,
        "structure_matcher_error": matcher_error,
        "structure_matcher_status": matcher_status,
        "raw_geometry": raw_geometry,
    }


def design_lattice_summary(latent_lattice: torch.Tensor) -> dict[str, Any]:
    scaled = torch.sigmoid(latent_lattice)
    a, b, c, alpha, beta, gamma = generation.unscale_lat(scaled)
    values = [a, b, c, alpha, beta, gamma]
    parameters = [float(value.reshape(-1)[0].detach().cpu()) for value in values]
    return {
        "raw_decoder_output": [float(value) for value in latent_lattice.detach().cpu().reshape(-1)],
        "sigmoid_then_generation_scale": parameters,
        "mean_predicted_length_before_projection": float(np.mean(parameters[:3])),
        "used_by_final_generation_structure": False,
    }


def zero_reference_decode(
    model: torch.nn.Module,
    latent: torch.Tensor,
    species_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the generation decoder with zeros only; no reference input is accepted."""
    batch = latent.shape[0]
    return model.crystal_decoder(
        latent,
        input_coords=torch.zeros(batch, MAX_SITES, 3, device=latent.device),
        center=torch.zeros(batch, 3, device=latent.device),
        species_mask=species_mask,
    )


def reject(
    trace: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    record: dict[str, Any],
    stage: str,
    reason: str,
) -> None:
    event = {"attempt": record["attempt"], "stage": stage, "reason": reason}
    trace.append(event)
    record.update({"outcome": "rejected", "rejection_stage": stage, "reason": reason})
    attempts.append(record)


@torch.inference_mode()
def traced_generation_decode(
    model: torch.nn.Module,
    latent: torch.Tensor,
    allowed_anions: set[str],
    species_mask: torch.Tensor,
    rng: np.random.RandomState,
    bg_target: float,
    family: str,
    decode_temperature: float,
    decode_topk: int,
    decode_tries: int,
    refine: bool,
) -> tuple[Structure | None, dict[str, Any]]:
    """Mirror design.decode_and_filter for one latent while recording failures."""
    if latent.shape[0] != 1:
        raise ValueError("The traced decoder accepts exactly one latent at a time")
    property_output = model.property_decoder(latent)
    lattice_output, _, species_logits, coordinates_output = zero_reference_decode(
        model, latent, species_mask.to(latent.device)
    )
    a, b, c, _, _, _ = generation.unscale_lat(torch.sigmoid(lattice_output))
    ignored_decoder_a0 = float(((a + b + c) / 3.0).cpu().item())

    logits = species_logits[0]
    x_prior = generation.target_guided_x_prior(bg_target, family, set(allowed_anions))
    anion_indices = [generation.SPECIES_LIST.index(symbol) for symbol in allowed_anions]
    b_elements = generation.target_conditioned_B_set(
        bg_target, family, use_target_filter=False
    )
    b_indices = torch.tensor(
        [generation.SPECIES_LIST.index(symbol) for symbol in b_elements],
        device=latent.device,
    )
    rejection_trace: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []

    def sample_topk(values: torch.Tensor) -> int:
        count = min(decode_topk, values.numel())
        top_values, top_indices = torch.topk(values, k=count)
        probabilities = F.softmax(top_values, dim=-1).cpu().numpy()
        return int(rng.choice(top_indices.cpu().numpy(), p=probabilities))

    for attempt_index in range(decode_tries):
        attempt = attempt_index + 1
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "constructed": False,
            "passed": False,
        }
        b_values = logits[1, b_indices] / max(decode_temperature, 1e-6)
        b_relative = sample_topk(b_values)
        b_index = int(b_indices[b_relative].item())
        b_symbol = generation.SPECIES_LIST[b_index]
        attempt_record["sampled_B"] = b_symbol

        base_scores = logits[2:5, anion_indices].sum(dim=0) / max(
            decode_temperature, 1e-6
        )
        prior = torch.tensor(
            [float(x_prior.get(generation.SPECIES_LIST[index], 1.0)) for index in anion_indices],
            device=latent.device,
            dtype=base_scores.dtype,
        )
        effective_scores = base_scores + 0.0 * torch.log(prior.clamp_min(1e-6))
        x_values, x_relative_indices = torch.topk(
            effective_scores,
            k=min(decode_topk, effective_scores.numel()),
        )
        x_probabilities = F.softmax(x_values, dim=-1).cpu().numpy()
        chosen_relative = int(rng.choice(x_relative_indices.cpu().numpy(), p=x_probabilities))
        x_index = int(torch.tensor(anion_indices, device=latent.device)[chosen_relative])
        x_symbol = generation.SPECIES_LIST[x_index]
        attempt_record["sampled_X"] = x_symbol

        family_for_b = family if family in generation.VALENCE_B_BY_FAMILY else "oxide"
        q_b = generation.VALENCE_B_BY_FAMILY.get(family_for_b, {}).get(b_symbol, {+2})
        q_x = generation.VALENCE_X.get(x_symbol, {-1})
        required_a_charges = {-(b_charge + 3 * x_charge) for b_charge in q_b for x_charge in q_x}
        compatible_a = [
            symbol
            for symbol in generation.A_CATIONS
            if any(
                charge in generation.VALENCE_A.get(symbol, set())
                for charge in required_a_charges
            )
        ]
        if not compatible_a:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "charge_compatible_A",
                "no compatible A species",
            )
            continue
        a_indices = torch.tensor(
            [generation.SPECIES_LIST.index(symbol) for symbol in compatible_a],
            device=latent.device,
        )
        a_relative = sample_topk(logits[0, a_indices] / max(decode_temperature, 1e-6))
        a_index = int(a_indices[a_relative].item())
        a_symbol = generation.SPECIES_LIST[a_index]
        attempt_record["sampled_A"] = a_symbol
        if a_index == b_index:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "distinct_roles",
                "sampled A equals sampled B",
            )
            continue

        symbols = [a_symbol, b_symbol, x_symbol, x_symbol, x_symbol]
        radius_b = generation.get_ionic_radius(b_symbol)
        radius_x = generation.get_ionic_radius(x_symbol)
        projected_a = float(np.clip(2.0 * (radius_b + radius_x), 3.0, 8.0))
        structure = Structure(
            Lattice.cubic(projected_a),
            symbols,
            generation.TEMPLATE.cpu().numpy().tolist(),
            coords_are_cartesian=False,
        )
        attempt_record.update(
            {
                "constructed": True,
                "sampled_A": a_symbol,
                "sampled_B": b_symbol,
                "sampled_X": x_symbol,
                "projected_lattice_a": projected_a,
            }
        )
        if not generation.is_ABX3(structure):
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "stoichiometry",
                "is_ABX3 returned false",
            )
            continue
        realism_failures = generation.realism(structure)
        if realism_failures:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "realism",
                "; ".join(realism_failures),
            )
            continue
        decoded_family = generation.family_from_structure(structure)
        if decoded_family is None:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "family",
                "family_from_structure returned None",
            )
            continue
        if refine:
            try:
                structure = SpacegroupAnalyzer(structure, symprec=0.05).get_refined_structure()
            except Exception as error:
                reject(
                    rejection_trace,
                    attempt_records,
                    attempt_record,
                    "refinement",
                    f"{type(error).__name__}: {error}",
                )
                continue
            if np.isnan(structure.cart_coords).any():
                reject(
                    rejection_trace,
                    attempt_records,
                    attempt_record,
                    "refinement",
                    "refined Cartesian coordinates are NaN",
                )
                continue
        charge_failures = generation.check_charge_balance_existential(structure)
        if charge_failures:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "charge",
                "; ".join(charge_failures),
            )
            continue
        distance_failures = generation.check_BX_distances_family(structure)
        if distance_failures:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "B-X distance",
                "; ".join(distance_failures),
            )
            continue
        factor_failures, tolerance, mu = generation.goldschmidt_and_mu_ok(structure)
        if factor_failures:
            reject(
                rejection_trace,
                attempt_records,
                attempt_record,
                "tolerance/mu",
                "; ".join(factor_failures),
            )
            continue

        attempt_record.update(
            {
                "passed": True,
                "outcome": "passed",
                "rejection_stage": None,
                "reason": None,
            }
        )
        attempt_records.append(attempt_record)
        return structure, {
            "accepted": True,
            "attempts_used": attempt,
            "rejection_trace": rejection_trace,
            "attempt_records": attempt_records,
            "constructed_count": sum(record["constructed"] for record in attempt_records),
            "A": a_symbol,
            "B": b_symbol,
            "X": x_symbol,
            "projected_lattice_a": projected_a,
            "tolerance_factor": float(tolerance),
            "octahedral_factor": float(mu),
            "ignored_raw_decoder_lattice_a": ignored_decoder_a0,
            "ignored_raw_coordinate_range": [
                float(coordinates_output.min()),
                float(coordinates_output.max()),
            ],
            "provisional_head_heat": float(property_output[0, 0]),
            "provisional_head_gap": float(property_output[0, 1]),
        }

    return None, {
        "accepted": False,
        "attempts_used": decode_tries,
        "rejection_trace": rejection_trace,
        "attempt_records": attempt_records,
        "constructed_count": sum(record["constructed"] for record in attempt_records),
        "A": None,
        "B": None,
        "X": None,
        "projected_lattice_a": None,
        "tolerance_factor": None,
        "octahedral_factor": None,
        "ignored_raw_decoder_lattice_a": ignored_decoder_a0,
        "ignored_raw_coordinate_range": [
            float(coordinates_output.min()),
            float(coordinates_output.max()),
        ],
        "provisional_head_heat": float(property_output[0, 0]),
        "provisional_head_gap": float(property_output[0, 1]),
    }


@torch.inference_mode()
def baseline_generation_decode(
    model: torch.nn.Module,
    latent: torch.Tensor,
    allowed_anions: set[str],
    species_mask: torch.Tensor,
    rng_seed: int,
    bg_target: float,
    family: str,
    decode_temperature: float,
    decode_topk: int,
    decode_tries: int,
    refine: bool,
) -> tuple[Structure | None, dict[str, Any] | None, float, float]:
    structures, infos, gaps, heats = generation.decode_and_filter(
        model,
        latent,
        allowed_anions,
        species_mask,
        decimals=3,
        refine=refine,
        decode_temp=decode_temperature,
        decode_topk=decode_topk,
        decode_tries=decode_tries,
        rng=np.random.RandomState(rng_seed),
        ab_pair_counts=Counter(),
        ab_pair_counts_local=Counter(),
        anti_repeat_alpha=0.6,
        bg_target=bg_target,
        family_str=family,
        x_prior_strength=0.0,
        use_target_b_filter=False,
    )
    return structures[0], infos[0], float(gaps[0]), float(heats[0])


def structures_match_trace(left: Structure | None, right: Structure | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return generation.struct_signature(left) == generation.struct_signature(right)


def reset_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def endpoint_matches_baseline(
    traced_structure: Structure | None,
    traced: dict[str, Any],
    baseline_structure: Structure | None,
    baseline_info: dict[str, Any] | None,
    baseline_gap: float,
    baseline_heat: float,
) -> bool:
    if not structures_match_trace(traced_structure, baseline_structure):
        return False
    if traced_structure is None:
        return baseline_info is None and not traced["accepted"]
    if baseline_info is None or not traced["accepted"]:
        return False
    symbolic_equal = all(
        traced[key] == baseline_info[key]
        for key in ("A", "B", "X")
    )
    numeric_equal = all(
        math.isclose(float(left), float(right), rel_tol=RTOL, abs_tol=ATOL)
        for left, right in (
            (traced["projected_lattice_a"], baseline_info["a0"]),
            (traced["tolerance_factor"], baseline_info["t"]),
            (traced["octahedral_factor"], baseline_info["mu"]),
            (traced["provisional_head_gap"], baseline_gap),
            (traced["provisional_head_heat"], baseline_heat),
        )
    )
    return symbolic_equal and numeric_equal


def projected_structure_metrics(
    structure: Structure | None,
    reference_path: Path,
    context: dict[str, Any],
    sampled_roles: dict[str, Any],
) -> dict[str, Any]:
    if structure is None:
        return {
            "cif_constructed": False,
            "formula": None,
            "source_roles_recovered": False,
            "structure_matcher_scale_false": False,
            "structure_matcher_scale_true": False,
            "structure_matcher_status": "invalid",
            "minimum_distance_angstrom": None,
        }
    reference = CifParser(str(reference_path)).parse_structures(primitive=False)[0]
    distance_matrix = structure.distance_matrix.copy()
    distance_matrix[np.eye(len(structure), dtype=bool)] = np.inf
    formula = structure.composition.reduced_formula
    role_recovered = (
        sampled_roles["A"] == context["A"]
        and sampled_roles["B"] == context["B"]
        and sampled_roles["X"] == context["X"]
    )
    matcher_scale_false = False
    matcher_scale_true = False
    matcher_status = "composition_incompatible"
    if structure.composition == reference.composition:
        matcher_scale_false = bool(matcher(scale=False).fit(reference, structure))
        matcher_scale_true = bool(matcher(scale=True).fit(reference, structure))
        matcher_status = "matched" if matcher_scale_false else "unmatched"
    return {
        "cif_constructed": True,
        "formula": formula,
        "source_roles_recovered": role_recovered,
        "structure_matcher_scale_false": matcher_scale_false,
        "structure_matcher_scale_true": matcher_scale_true,
        "structure_matcher_status": matcher_status,
        "minimum_distance_angstrom": float(distance_matrix.min()),
        "lattice_parameters": [
            *[float(value) for value in structure.lattice.abc],
            *[float(value) for value in structure.lattice.angles],
        ],
        "site_count": len(structure),
    }


def select_representatives(
    accepted: list[tuple[dict[str, Any], Structure]],
    limit: int,
) -> list[tuple[dict[str, Any], Structure]]:
    """Select deterministic, structurally unique examples with path coverage first."""
    if limit <= 0:
        return []
    ordered = sorted(
        accepted,
        key=lambda item: (
            0 if item[0]["source_gap_provisional"] > 0 else 1,
            str(item[0]["latent_path"]),
            str(item[0]["material_id"]),
            int(item[0]["seed"]),
        ),
    )
    selected: list[tuple[dict[str, Any], Structure]] = []
    seen_signatures: set[str] = set()

    def add(item: tuple[dict[str, Any], Structure]) -> None:
        signature = str(item[0]["structure_signature"])
        if signature not in seen_signatures and len(selected) < limit:
            selected.append(item)
            seen_signatures.add(signature)

    for path_name in LATENT_PATHS:
        for item in ordered:
            if item[0]["latent_path"] == path_name:
                before = len(selected)
                add(item)
                if len(selected) > before:
                    break
    for item in ordered:
        add(item)
    return selected


def assign_structure_clusters(
    rows: list[dict[str, Any]],
    accepted: list[tuple[dict[str, Any], Structure]],
) -> None:
    """Assign deterministic scale=False StructureMatcher clusters within each latent arm."""
    for row in rows:
        row["matcher_structure_cluster"] = None
    for path_name in LATENT_PATHS:
        clusters: list[Structure] = []
        selected = sorted(
            [item for item in accepted if item[0]["latent_path"] == path_name],
            key=lambda item: (
                str(item[0]["material_id"]),
                int(item[0]["seed"]),
            ),
        )
        for row, structure in selected:
            cluster_index = None
            for index, representative in enumerate(clusters):
                if structure.composition != representative.composition:
                    continue
                if matcher(scale=False).fit(representative, structure):
                    cluster_index = index
                    break
            if cluster_index is None:
                clusters.append(structure)
                cluster_index = len(clusters) - 1
            row["matcher_structure_cluster"] = f"{path_name}:{cluster_index}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def bootstrap_group_mean(
    group_values: dict[str, list[float]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    group_means = np.asarray(
        [np.mean(group_values[group]) for group in sorted(group_values)],
        dtype=float,
    )
    if group_means.size == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "independent_groups": 0,
            "bootstrap_resamples": resamples,
            "median": None,
            "range": None,
        }
    rng = np.random.RandomState(seed)
    sampled = rng.choice(group_means, size=(resamples, len(group_means)), replace=True).mean(axis=1)
    return {
        "mean": float(group_means.mean()),
        "median": float(np.median(group_means)),
        "range": [float(group_means.min()), float(group_means.max())],
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "independent_groups": int(group_means.size),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def yield_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(row["attempts_used"]) for row in rows)
    constructed = sum(int(row["constructed_count"]) for row in rows)
    passed = [row for row in rows if row["accepted"]]
    formulas = Counter(str(row["formula"]) for row in passed)
    accepted_ab = Counter(
        (str(row["sampled_A"]), str(row["sampled_B"])) for row in passed
    )
    signature_structures = {row["structure_signature"] for row in passed}
    matcher_structures = {row["matcher_structure_cluster"] for row in passed}
    unique_compositions = len(formulas)
    n_pass = len(passed)
    dominant_fraction = max(formulas.values()) / n_pass if n_pass else None
    dominant_ab_fraction = max(accepted_ab.values()) / n_pass if n_pass else None
    attempt_ab_recoveries = sum(
        int(row["attempt_source_AB_recovery_count"]) for row in rows
    )
    passed_ab_recoveries = sum(
        int(row["passed_source_AB_recovery_count"]) for row in rows
    )
    attempt_exact_recoveries = sum(
        int(row["attempt_source_exact_composition_recovery_count"]) for row in rows
    )
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        rejection_counts.update(json.loads(row["rejection_stage_counts"]))
    return {
        "decode_calls": len(rows),
        "N_attempt": attempts,
        "N_constructed": constructed,
        "N_pass": n_pass,
        "N_unique_composition": unique_compositions,
        "N_unique_accepted_AB": len(accepted_ab),
        "N_unique_structure_signature": len(signature_structures),
        "N_unique_structure_matcher_scale_false": len(matcher_structures),
        "acceptance_yield": n_pass / attempts if attempts else None,
        "unique_accepted_yield": unique_compositions / attempts if attempts else None,
        "unique_accepted_AB_yield": len(accepted_ab) / attempts if attempts else None,
        "conditional_uniqueness": unique_compositions / n_pass if n_pass else None,
        "conditional_AB_uniqueness": len(accepted_ab) / n_pass if n_pass else None,
        "dominant_accepted_composition_fraction": dominant_fraction,
        "dominant_accepted_AB_fraction": dominant_ab_fraction,
        "accepted_AB_counts": {
            f"{a}|{b}": count for (a, b), count in sorted(accepted_ab.items())
        },
        "attempt_source_AB_recovery_count": attempt_ab_recoveries,
        "attempt_source_AB_recovery_rate": (
            attempt_ab_recoveries / attempts if attempts else None
        ),
        "passed_source_AB_recovery_count": passed_ab_recoveries,
        "passed_source_AB_recovery_rate": (
            passed_ab_recoveries / attempts if attempts else None
        ),
        "attempt_source_exact_composition_recovery_count": attempt_exact_recoveries,
        "attempt_source_exact_composition_recovery_rate": (
            attempt_exact_recoveries / attempts if attempts else None
        ),
        "rejection_stage_counts": dict(rejection_counts),
        "source_A_recoveries": sum(bool(row["source_A_recovered"]) for row in passed),
        "source_B_recoveries": sum(bool(row["source_B_recovered"]) for row in passed),
        "source_AB_recoveries": sum(bool(row["source_AB_recovered"]) for row in passed),
        "source_exact_composition_recoveries": sum(
            bool(row["source_exact_composition_recovered"]) for row in passed
        ),
        "all_endpoints_match_baseline": all(row["endpoint_matches_baseline"] for row in rows),
    }


def summarize_stochastic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for path_name in LATENT_PATHS:
        selected = [row for row in rows if row["latent_path"] == path_name]
        summary[path_name] = yield_summary(selected)
    return summary


def grouped_yield_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path_name in LATENT_PATHS:
        path_rows = [row for row in rows if row["latent_path"] == path_name]
        for grouping, values in (
            ("pooled", ["all"]),
            ("family", sorted({str(row["family"]) for row in path_rows})),
            ("material_id", sorted({str(row["material_id"]) for row in path_rows})),
        ):
            for value in values:
                selected = path_rows
                if grouping != "pooled":
                    selected = [row for row in path_rows if str(row[grouping]) == value]
                item = yield_summary(selected)
                output.append(
                    {
                        "latent_path": path_name,
                        "grouping": grouping,
                        "condition": value,
                        **{
                            key: (json.dumps(metric, sort_keys=True) if isinstance(metric, dict) else metric)
                            for key, metric in item.items()
                        },
                    }
                )
    return output


def condition_level_ab_summary(
    rows: list[dict[str, Any]],
    latent_path: str,
) -> dict[str, Any]:
    """Describe accepted A/B collapse without treating forced X as diversity."""
    selected = [row for row in rows if row["latent_path"] == latent_path]
    conditions: list[dict[str, Any]] = []
    for material_id in sorted({str(row["material_id"]) for row in selected}):
        item = yield_summary(
            [row for row in selected if str(row["material_id"]) == material_id]
        )
        conditions.append(
            {
                "material_id": material_id,
                "N_pass": item["N_pass"],
                "N_unique_accepted_AB": item["N_unique_accepted_AB"],
                "dominant_accepted_AB_fraction": item["dominant_accepted_AB_fraction"],
                "accepted_AB_counts": item["accepted_AB_counts"],
            }
        )
    passing = [item for item in conditions if item["N_pass"] > 0]
    single_ab = [item for item in passing if item["N_unique_accepted_AB"] == 1]
    multiple_ab = [item for item in passing if item["N_unique_accepted_AB"] > 1]
    return {
        "latent_path": latent_path,
        "condition_count": len(conditions),
        "conditions_with_no_accepted_output": len(conditions) - len(passing),
        "conditions_with_accepted_output": len(passing),
        "conditions_with_single_accepted_AB": len(single_ab),
        "conditions_with_multiple_accepted_AB": len(multiple_ab),
        "all_passing_conditions_single_AB": (
            len(single_ab) == len(passing) if passing else None
        ),
        "conditions": conditions,
    }


def paired_final_summary(
    rows: list[dict[str, Any]],
    control_path: str,
) -> dict[str, Any]:
    known = {
        (str(row["material_id"]), int(row["seed"])): row
        for row in rows
        if row["latent_path"] == "joint"
    }
    control = {
        (str(row["material_id"]), int(row["seed"])): row
        for row in rows
        if row["latent_path"] == control_path
    }
    if set(known) != set(control):
        raise AssertionError(f"Unpaired stochastic rows for joint versus {control_path}")
    endpoint_recovery_by_group: dict[str, list[float]] = {}
    attempt_recovery_by_group: dict[str, list[float]] = {}
    passed_recovery_by_group: dict[str, list[float]] = {}
    seed_differences: dict[int, list[float]] = {}
    attempt_seed_differences: dict[int, list[float]] = {}
    passed_seed_differences: dict[int, list[float]] = {}
    endpoints_changed = 0
    both_passed = 0
    passed_formula_changed = 0
    for key in sorted(known):
        left = known[key]
        right = control[key]
        difference = float(left["source_AB_recovered"]) - float(right["source_AB_recovered"])
        group = str(left["composition_group"])
        endpoint_recovery_by_group.setdefault(group, []).append(difference)
        attempt_recovery_by_group.setdefault(group, []).append(
            float(left["attempt_source_AB_recovery_rate"])
            - float(right["attempt_source_AB_recovery_rate"])
        )
        passed_recovery_by_group.setdefault(group, []).append(
            float(left["passed_source_AB_recovery_rate"])
            - float(right["passed_source_AB_recovery_rate"])
        )
        seed_differences.setdefault(int(left["seed"]), []).append(difference)
        attempt_seed_differences.setdefault(int(left["seed"]), []).append(
            float(left["attempt_source_AB_recovery_rate"])
            - float(right["attempt_source_AB_recovery_rate"])
        )
        passed_seed_differences.setdefault(int(left["seed"]), []).append(
            float(left["passed_source_AB_recovery_rate"])
            - float(right["passed_source_AB_recovery_rate"])
        )
        left_endpoint = (bool(left["accepted"]), left["formula"])
        right_endpoint = (bool(right["accepted"]), right["formula"])
        endpoints_changed += left_endpoint != right_endpoint
        if left["accepted"] and right["accepted"]:
            both_passed += 1
            passed_formula_changed += left["formula"] != right["formula"]
    return {
        "control": control_path,
        "paired_draws": len(known),
        "joint_minus_control_endpoint_source_AB_recovery": bootstrap_group_mean(
            endpoint_recovery_by_group
        ),
        "joint_minus_control_attempt_source_AB_recovery_rate": bootstrap_group_mean(
            attempt_recovery_by_group
        ),
        "joint_minus_control_passed_source_AB_recovery_rate": bootstrap_group_mean(
            passed_recovery_by_group
        ),
        "source_AB_recovery_difference_by_seed": {
            str(seed): float(np.mean(values)) for seed, values in sorted(seed_differences.items())
        },
        "attempt_source_AB_recovery_rate_difference_by_seed": {
            str(seed): float(np.mean(values))
            for seed, values in sorted(attempt_seed_differences.items())
        },
        "passed_source_AB_recovery_rate_difference_by_seed": {
            str(seed): float(np.mean(values))
            for seed, values in sorted(passed_seed_differences.items())
        },
        "endpoint_changed_fraction": endpoints_changed / len(known),
        "both_passed": both_passed,
        "formula_changed_given_both_passed_fraction": (
            passed_formula_changed / both_passed if both_passed else None
        ),
    }


def paired_information_summary(
    rows: list[dict[str, Any]],
    control_path: str,
    metric_prefix: str,
) -> dict[str, Any]:
    by_path = {str(row["latent_path"]): row for row in rows}
    known = by_path["joint"]
    control = by_path[control_path]
    known_nll = known[f"{metric_prefix}_nll_nats_per_site"]
    control_nll = control[f"{metric_prefix}_nll_nats_per_site"]
    return {
        "delta_nll_control_minus_joint": (
            float(control_nll) - float(known_nll)
            if known_nll is not None and control_nll is not None
            else None
        ),
        "delta_top1_joint_minus_control": float(
            known[f"{metric_prefix}_top1_accuracy"]
            - control[f"{metric_prefix}_top1_accuracy"]
        ),
        "mean_total_variation": control[f"{metric_prefix}_tv_from_joint"],
    }


def aggregate_information_comparison(
    raw_rows: list[dict[str, Any]],
    control_path: str,
    metric_prefix: str,
) -> dict[str, Any]:
    by_material: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_material.setdefault(str(row["material_id"]), []).append(row)
    delta_nll_by_group: dict[str, list[float]] = {}
    delta_top1_by_group: dict[str, list[float]] = {}
    tv_values: list[float] = []
    paired = 0
    unavailable_nll = 0
    for material_id in sorted(by_material):
        rows = by_material[material_id]
        indexed = {str(row["latent_path"]): row for row in rows}
        if "joint" not in indexed or control_path not in indexed:
            raise AssertionError(f"Missing information pair for {material_id}, {control_path}")
        result = paired_information_summary(rows, control_path, metric_prefix)
        group = str(indexed["joint"]["composition_group"])
        if result["delta_nll_control_minus_joint"] is None:
            unavailable_nll += 1
        else:
            delta_nll_by_group.setdefault(group, []).append(
                float(result["delta_nll_control_minus_joint"])
            )
        delta_top1_by_group.setdefault(group, []).append(
            float(result["delta_top1_joint_minus_control"])
        )
        tv_values.append(float(result["mean_total_variation"]))
        paired += 1
    return {
        "control": control_path,
        "stage": metric_prefix,
        "paired_examples": paired,
        "nll_unavailable_examples": unavailable_nll,
        "delta_nll_control_minus_joint": bootstrap_group_mean(delta_nll_by_group),
        "delta_top1_joint_minus_control": bootstrap_group_mean(delta_top1_by_group),
        "mean_total_variation": float(np.mean(tv_values)),
        "median_total_variation": float(np.median(tv_values)),
        "total_variation_range": [float(np.min(tv_values)), float(np.max(tv_values))],
    }


def _legacy_markdown_report(summary: dict[str, Any]) -> str:
    return render_legacy_markdown_report(summary, LATENT_PATHS)


def markdown_report(summary: dict[str, Any]) -> str:
    return render_markdown_report(summary, LATENT_PATHS)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    audit_csv = args.audit_csv.resolve()
    cif_root = args.cif_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    report_dir = args.report_dir.resolve()
    if not audit_csv.is_file() or not checkpoint_path.is_file() or not cif_root.is_dir():
        raise FileNotFoundError("Audit CSV, validation CIF directory, and checkpoint must exist")
    if args.decode_tries <= 0 or args.decode_topk <= 0:
        raise ValueError("Decode tries and top-k must be positive")
    if args.representative_cifs < 0:
        raise ValueError("Representative CIF count cannot be negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")

    frozen_preflight = {
        "frozen_audit_hash": sha256_file(audit_csv) == EXPECTED_AUDIT_SHA256,
        "frozen_checkpoint_hash": sha256_file(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256,
        "frozen_protocol_hash": (
            PROTOCOL_PATH.is_file()
            and sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256
        ),
        "frozen_protocol_configuration": (
            list(args.seeds) == [0, 1, 2]
            and args.decode_temperature == 1.25
            and args.decode_topk == 12
            and args.decode_tries == 12
            and not args.no_refine
        ),
        "python_hash_seed_is_zero": os.environ.get("PYTHONHASHSEED") == "0",
    }
    if not all(frozen_preflight.values()):
        raise AssertionError(
            "Frozen audit preflight failed before model execution: "
            f"{frozen_preflight}"
        )

    frame, loaded = load_audit_inputs(audit_csv, cif_root)
    audit_cif_paths = [
        cif_root / f"{material_id}.cif"
        for material_id in frame["material_id"].astype(str)
    ]
    protected_paths = {
        path: sha256_file(path)
        for path in [
            DEFAULT_DATA_ROOT / "train.csv",
            DEFAULT_DATA_ROOT / "val.csv",
            DEFAULT_DATA_ROOT / "test.csv",
            audit_csv,
            checkpoint_path,
            MEIDNET_ROOT / "meidnet" / "model.py",
            MEIDNET_ROOT / "meidnet" / "design.py",
            PROTOCOL_PATH,
            Path(__file__).resolve(),
            *audit_cif_paths,
        ]
    }
    checkpoint, checkpoint_report = checkpoint_summary(checkpoint_path)
    main_model, main_model_report = instantiate_model(checkpoint)
    generation_model, generation_model_report = load_generation_model(checkpoint)
    main_model.eval().requires_grad_(False)
    model_state_checks = {
        "training_model_eval": not main_model.training,
        "generation_model_eval": not generation_model.training,
        "training_model_all_frozen": not any(
            parameter.requires_grad for parameter in main_model.parameters()
        ),
        "generation_model_all_frozen": not any(
            parameter.requires_grad for parameter in generation_model.parameters()
        ),
    }
    if not all(model_state_checks.values()):
        raise AssertionError(f"Frozen/eval model check failed: {model_state_checks}")

    raw_rows: list[dict[str, Any]] = []
    stochastic_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    information_site_rows: list[dict[str, Any]] = []
    latent_metadata: dict[str, Any] = {}
    accepted_structures: list[tuple[dict[str, Any], Structure]] = []
    repeated_latent_exact = True
    repeated_latent_maximum_absolute = 0.0
    repeat_by_stage = {
        "A_reference_assisted": new_repeat_summary(),
        "B_canonical_reference_free": new_repeat_summary(),
        "C_generation_reference_free": new_repeat_summary(),
    }
    decoder_implementations_agree = True
    decoder_implementation_maximum_absolute = 0.0
    maximum_input_species_effect = 0.0
    all_unmasked_outputs_finite = True
    all_latents_finite = True
    all_probability_rows_valid = True

    prepared: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for _, row in frame.iterrows():
            material_id = str(row["material_id"])
            sample = loaded[material_id]
            crystal = sample["crystal_vec"].unsqueeze(0)
            heat = sample["heat_all"].reshape(1)
            gap = sample["dir_gap"].reshape(1)
            context = source_context(row)
            _, occupied_slots, true_indices = target_site_tensors(crystal)
            dense_symbols = [SPECIES_LIST[index] for index in true_indices]
            if occupied_slots != list(range(5)) or dense_symbols != context["parsed_symbols"]:
                raise ValueError(
                    f"Dense parser slots differ from recorded metadata for {material_id}: "
                    f"slots={occupied_slots}, symbols={dense_symbols}"
                )
            paths, latent_info, center, input_species, input_coordinates = latent_bundle(
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
                "context": context,
                "paths": paths,
                "center": center,
                "input_species": input_species,
                "input_coordinates": input_coordinates,
            }
            latent_metadata[material_id] = latent_info

        fixed_donor_id = str(frame.iloc[0]["material_id"])
        fixed_real_latent = prepared[fixed_donor_id]["paths"]["joint"].detach().clone()
        fixed_donor_condition = {
            "material_id": fixed_donor_id,
            "heat_provisional": float(prepared[fixed_donor_id]["heat"].item()),
            "gap_provisional": float(prepared[fixed_donor_id]["gap"].item()),
            "family": prepared[fixed_donor_id]["context"]["family"],
        }
        for material_id, item in prepared.items():
            item["paths"]["fixed_real"] = fixed_real_latent.clone()
            latent_metadata[material_id]["fixed_real"] = tensor_stats(fixed_real_latent)
            latent_metadata[material_id]["fixed_real_donor_id"] = fixed_donor_id

    with torch.inference_mode():
        for material_id in frame["material_id"].astype(str):
            item = prepared[material_id]
            row = item["row"]
            crystal = item["crystal"]
            heat = item["heat"]
            gap = item["gap"]
            context = item["context"]
            family_mask = generation.build_species_mask(context["allowed_anions"])
            paths = item["paths"]
            center = item["center"]
            input_species = item["input_species"]
            input_coordinates = item["input_coordinates"]

            reference_outputs = main_model.crystal_decoder(
                paths["joint"],
                input_species=input_species,
                input_coords=input_coordinates,
                center=center,
            )
            repeated_reference_outputs = main_model.crystal_decoder(
                paths["joint"],
                input_species=input_species,
                input_coords=input_coordinates,
                center=center,
            )
            update_repeat_summary(
                repeat_by_stage["A_reference_assisted"],
                reference_outputs,
                repeated_reference_outputs,
            )
            same_aux_without_species = main_model.crystal_decoder(
                paths["joint"],
                input_species=None,
                input_coords=input_coordinates,
                center=center,
            )
            reference_free_outputs = main_model.crystal_decoder(paths["joint"])
            repeated_reference_free_outputs = main_model.crystal_decoder(paths["joint"])
            update_repeat_summary(
                repeat_by_stage["B_canonical_reference_free"],
                reference_free_outputs,
                repeated_reference_free_outputs,
            )
            all_unmasked_outputs_finite = all_unmasked_outputs_finite and all(
                finite_tensor(output) for output in (*reference_outputs, *reference_free_outputs)
            )
            input_species_effect = max(
                difference_stats(left, right)["maximum_absolute"]
                for left, right in zip(reference_outputs, same_aux_without_species)
            )
            maximum_input_species_effect = max(
                maximum_input_species_effect,
                input_species_effect,
            )

            reconstruction = reconstruction_metrics(
                crystal,
                reference_outputs[0],
                reference_outputs[2],
                reference_outputs[3],
                context,
            )
            reference_information, reference_sites = species_information_metrics(
                reference_outputs[2],
                crystal,
            )
            reference_free_information, reference_free_sites = species_information_metrics(
                reference_free_outputs[2],
                crystal,
            )
            reference_masked_ab_information, reference_masked_ab_sites = (
                species_information_metrics(
                    reference_outputs[2],
                    crystal,
                    family_mask,
                    scored_slots=[0, 1],
                )
            )
            reference_free_masked_ab_information, reference_free_masked_ab_sites = (
                species_information_metrics(
                    reference_free_outputs[2],
                    crystal,
                    family_mask,
                    scored_slots=[0, 1],
                )
            )
            all_probability_rows_valid = all_probability_rows_valid and all(
                information["probability_rows_valid"]
                for information in (
                    reference_information,
                    reference_free_information,
                    reference_masked_ab_information,
                    reference_free_masked_ab_information,
                )
            )
            reference_to_reference_free_tv = occupied_site_total_variation(
                reference_outputs[2],
                reference_free_outputs[2],
                crystal,
            )
            reference_to_reference_free_masked_ab_tv = occupied_site_total_variation(
                reference_outputs[2],
                reference_free_outputs[2],
                crystal,
                family_mask,
                scored_slots=[0, 1],
            )
            reference_free_geometry = reconstruction_metrics(
                crystal,
                reference_free_outputs[0],
                reference_free_outputs[2],
                reference_free_outputs[3],
                context,
            )
            auxiliary_differences = {
                "lattice": difference_stats(reference_outputs[0], reference_free_outputs[0]),
                "adjacency": difference_stats(reference_outputs[1], reference_free_outputs[1]),
                "species_logits": difference_stats(reference_outputs[2], reference_free_outputs[2]),
                "coordinates": difference_stats(reference_outputs[3], reference_free_outputs[3]),
            }

            for stage, sites in (
                ("A_reference_assisted", reference_sites),
                ("B_canonical_reference_free", reference_free_sites),
                ("A_reference_assisted_family_masked_AB_only", reference_masked_ab_sites),
                (
                    "B_canonical_reference_free_family_masked_AB_only",
                    reference_free_masked_ab_sites,
                ),
            ):
                for site in sites:
                    information_site_rows.append(
                        {
                            "material_id": material_id,
                            "composition_group": str(row["composition_group"]),
                            "latent_path": "joint",
                            "stage": stage,
                            **site,
                        }
                    )

            decoded: dict[str, dict[str, Any]] = {}
            for path_name in LATENT_PATHS:
                latent = paths[path_name]
                all_latents_finite = all_latents_finite and finite_tensor(latent)
                unmasked_outputs = zero_reference_decode(generation_model, latent, None)
                masked_outputs = zero_reference_decode(generation_model, latent, family_mask)
                main_free = main_model.crystal_decoder(latent)
                repeated_outputs = zero_reference_decode(generation_model, latent, None)
                update_repeat_summary(
                    repeat_by_stage["C_generation_reference_free"],
                    unmasked_outputs,
                    repeated_outputs,
                )
                repeated_differences = [
                    difference_stats(left, right)
                    for left, right in zip(unmasked_outputs, repeated_outputs)
                ]
                repeated_latent_maximum_absolute = max(
                    repeated_latent_maximum_absolute,
                    *(difference["maximum_absolute"] for difference in repeated_differences),
                )
                repeated_latent_exact = repeated_latent_exact and all(
                    torch.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=False)
                    for left, right in zip(unmasked_outputs, repeated_outputs)
                )
                compatibility = {
                    "lattice": difference_stats(main_free[0], unmasked_outputs[0]),
                    "species_logits": difference_stats(main_free[2], unmasked_outputs[2]),
                    "coordinates": difference_stats(main_free[3], unmasked_outputs[3]),
                    "adjacency_comparable": False,
                }
                compatible = all(
                    torch.allclose(main_free[index], unmasked_outputs[index], rtol=RTOL, atol=ATOL)
                    for index in (0, 2, 3)
                )
                decoder_implementations_agree = decoder_implementations_agree and compatible
                decoder_implementation_maximum_absolute = max(
                    decoder_implementation_maximum_absolute,
                    *(compatibility[key]["maximum_absolute"] for key in (
                        "lattice",
                        "species_logits",
                        "coordinates",
                    )),
                )
                unmasked_information, unmasked_sites = species_information_metrics(
                    unmasked_outputs[2],
                    crystal,
                )
                masked_all_five_information, masked_all_five_sites = species_information_metrics(
                    masked_outputs[2],
                    crystal,
                    family_mask,
                )
                masked_ab_information, masked_ab_sites = species_information_metrics(
                    masked_outputs[2],
                    crystal,
                    family_mask,
                    scored_slots=[0, 1],
                )
                all_probability_rows_valid = all_probability_rows_valid and bool(
                    unmasked_information["probability_rows_valid"]
                    and masked_all_five_information["probability_rows_valid"]
                    and masked_ab_information["probability_rows_valid"]
                )
                property_output = generation_model.property_decoder(latent)
                heads = {
                    "generation_lattice": unmasked_outputs[0],
                    "generation_adjacency": unmasked_outputs[1],
                    "generation_species_logits": unmasked_outputs[2],
                    "generation_coordinates": unmasked_outputs[3],
                    "canonical_lattice": main_free[0],
                    "canonical_adjacency": main_free[1],
                    "canonical_species_logits": main_free[2],
                    "canonical_coordinates": main_free[3],
                    "property": property_output,
                }
                path_finite = all(finite_tensor(output) for output in heads.values())
                all_unmasked_outputs_finite = all_unmasked_outputs_finite and path_finite
                decoded[path_name] = {
                    "unmasked": unmasked_outputs,
                    "masked": masked_outputs,
                    "unmasked_information": unmasked_information,
                    "masked_all_five_information": masked_all_five_information,
                    "masked_ab_information": masked_ab_information,
                    "property_output": property_output,
                    "head_stats": {name: tensor_stats(output) for name, output in heads.items()},
                }
                for stage, sites in (
                    ("C_generation_unmasked", unmasked_sites),
                    (
                        "C_generation_family_masked_all_five_forced_X_including",
                        masked_all_five_sites,
                    ),
                    ("C_generation_family_masked_AB_only", masked_ab_sites),
                ):
                    for site in sites:
                        information_site_rows.append(
                            {
                                "material_id": material_id,
                                "composition_group": str(row["composition_group"]),
                                "latent_path": path_name,
                                "stage": stage,
                                **site,
                            }
                        )

            joint_unmasked_logits = decoded["joint"]["unmasked"][2]
            joint_masked_logits = decoded["joint"]["masked"][2]
            for path_name in LATENT_PATHS:
                latent = paths[path_name]
                unmasked_outputs = decoded[path_name]["unmasked"]
                masked_outputs = decoded[path_name]["masked"]
                unmasked_information = decoded[path_name]["unmasked_information"]
                masked_all_five_information = decoded[path_name][
                    "masked_all_five_information"
                ]
                masked_ab_information = decoded[path_name]["masked_ab_information"]
                compatibility = {
                    "lattice": difference_stats(main_model.crystal_decoder(latent)[0], unmasked_outputs[0]),
                    "species_logits": difference_stats(
                        main_model.crystal_decoder(latent)[2], unmasked_outputs[2]
                    ),
                    "coordinates": difference_stats(
                        main_model.crystal_decoder(latent)[3], unmasked_outputs[3]
                    ),
                }
                unmasked_tv = occupied_site_total_variation(
                    joint_unmasked_logits,
                    unmasked_outputs[2],
                    crystal,
                )
                masked_all_five_tv = occupied_site_total_variation(
                    joint_masked_logits,
                    masked_outputs[2],
                    crystal,
                    family_mask,
                )
                masked_ab_tv = occupied_site_total_variation(
                    joint_masked_logits,
                    masked_outputs[2],
                    crystal,
                    family_mask,
                    scored_slots=[0, 1],
                )
                generation_reference_geometry = reconstruction_metrics(
                    crystal,
                    unmasked_outputs[0],
                    unmasked_outputs[2],
                    unmasked_outputs[3],
                    context,
                )
                unmasked_top = predicted_symbols(unmasked_outputs[2][0])[:5]
                masked_top = predicted_symbols(masked_outputs[2][0])[:5]
                raw_rows.append(
                    {
                        "material_id": material_id,
                        "composition_group": str(row["composition_group"]),
                        "latent_path": path_name,
                        "source_heat_provisional": float(heat.item()),
                        "source_gap_provisional": float(gap.item()),
                        "family": context["family"],
                        "fixed_real_donor_id": fixed_donor_id if path_name == "fixed_real" else None,
                        "fixed_control_same_as_recipient": (
                            material_id == fixed_donor_id if path_name == "fixed_real" else None
                        ),
                        "fixed_control_family_mismatch": (
                            context["family"] != fixed_donor_condition["family"]
                            if path_name == "fixed_real"
                            else None
                        ),
                        "family_mask_forces_source_X": True,
                        "site_mapping_valid": context["site_mapping_valid"],
                        "site_mapping_identity": context["site_mapping_identity"],
                        "latent_l2_norm": float(torch.linalg.vector_norm(latent)),
                        "latent_finite_fraction": tensor_stats(latent)["finite_fraction"],
                        "all_unmasked_heads_finite": all(
                            value["finite"] for value in decoded[path_name]["head_stats"].values()
                        ),
                        "unmasked_head_stats": json.dumps(
                            decoded[path_name]["head_stats"], sort_keys=True
                        ),
                        "unmasked_top_first_five": json.dumps(unmasked_top),
                        "masked_top_first_five": json.dumps(masked_top),
                        "mask_changed_first_five_count": sum(
                            left != right for left, right in zip(unmasked_top, masked_top)
                        ),
                        "unmasked_nll_nats_per_site": unmasked_information["nll_nats_per_site"],
                        "unmasked_nll_status": unmasked_information["nll_status"],
                        "unmasked_top1_accuracy": unmasked_information["top1_accuracy"],
                        "unmasked_exact_composition": unmasked_information["exact_composition"],
                        "unmasked_zero_true_probability_sites": unmasked_information[
                            "zero_true_probability_sites"
                        ],
                        "unmasked_tv_from_joint": unmasked_tv["mean"],
                        "unmasked_tv_from_joint_per_site_json": json.dumps(
                            unmasked_tv["per_site"]
                        ),
                        "masked_all_five_nll_nats_per_site": masked_all_five_information[
                            "nll_nats_per_site"
                        ],
                        "masked_all_five_nll_status": masked_all_five_information[
                            "nll_status"
                        ],
                        "masked_all_five_top1_accuracy": masked_all_five_information[
                            "top1_accuracy"
                        ],
                        "masked_all_five_exact_composition": masked_all_five_information[
                            "exact_composition"
                        ],
                        "masked_all_five_zero_true_probability_sites": masked_all_five_information[
                            "zero_true_probability_sites"
                        ],
                        "masked_all_five_tv_from_joint": masked_all_five_tv["mean"],
                        "masked_all_five_tv_from_joint_per_site_json": json.dumps(
                            masked_all_five_tv["per_site"]
                        ),
                        "masked_ab_nll_nats_per_site": masked_ab_information[
                            "nll_nats_per_site"
                        ],
                        "masked_ab_nll_status": masked_ab_information["nll_status"],
                        "masked_ab_top1_accuracy": masked_ab_information["top1_accuracy"],
                        "masked_ab_exact_composition": masked_ab_information[
                            "exact_composition"
                        ],
                        "masked_ab_zero_true_probability_sites": masked_ab_information[
                            "zero_true_probability_sites"
                        ],
                        "masked_ab_tv_from_joint": masked_ab_tv["mean"],
                        "masked_ab_tv_from_joint_per_site_json": json.dumps(
                            masked_ab_tv["per_site"]
                        ),
                        "raw_lattice": json.dumps(design_lattice_summary(unmasked_outputs[0])),
                        "raw_geometry": json.dumps(
                            raw_geometry_metrics(unmasked_outputs[0][0], unmasked_outputs[3][0])
                        ),
                        "main_generation_lattice_max_abs": compatibility["lattice"][
                            "maximum_absolute"
                        ],
                        "main_generation_species_max_abs": compatibility["species_logits"][
                            "maximum_absolute"
                        ],
                        "main_generation_coords_max_abs": compatibility["coordinates"][
                            "maximum_absolute"
                        ],
                        "reference_nll_nats_per_site": (
                            reference_information["nll_nats_per_site"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_top1_accuracy": (
                            reference_information["top1_accuracy"] if path_name == "joint" else None
                        ),
                        "reference_masked_ab_nll_nats_per_site": (
                            reference_masked_ab_information["nll_nats_per_site"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_masked_ab_nll_status": (
                            reference_masked_ab_information["nll_status"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_masked_ab_top1_accuracy": (
                            reference_masked_ab_information["top1_accuracy"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_masked_ab_zero_true_probability_sites": (
                            reference_masked_ab_information["zero_true_probability_sites"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_nll_nats_per_site": (
                            reference_free_information["nll_nats_per_site"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_top1_accuracy": (
                            reference_free_information["top1_accuracy"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_masked_ab_nll_nats_per_site": (
                            reference_free_masked_ab_information["nll_nats_per_site"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_masked_ab_nll_status": (
                            reference_free_masked_ab_information["nll_status"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_masked_ab_top1_accuracy": (
                            reference_free_masked_ab_information["top1_accuracy"]
                            if path_name == "joint"
                            else None
                        ),
                        "canonical_reference_free_masked_ab_zero_true_probability_sites": (
                            reference_free_masked_ab_information[
                                "zero_true_probability_sites"
                            ]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_to_reference_free_mean_tv": (
                            reference_to_reference_free_tv["mean"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_to_reference_free_tv_per_site_json": (
                            json.dumps(reference_to_reference_free_tv["per_site"])
                            if path_name == "joint"
                            else None
                        ),
                        "reference_to_reference_free_masked_ab_mean_tv": (
                            reference_to_reference_free_masked_ab_tv["mean"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_to_reference_free_masked_ab_tv_per_site_json": (
                            json.dumps(reference_to_reference_free_masked_ab_tv["per_site"])
                            if path_name == "joint"
                            else None
                        ),
                        "reference_species_logit_mean_abs_difference": (
                            auxiliary_differences["species_logits"]["mean_absolute"]
                            if path_name == "joint"
                            else None
                        ),
                        "reference_coordinate_mean_abs_difference": (
                            auxiliary_differences["coordinates"]["mean_absolute"]
                            if path_name == "joint"
                            else None
                        ),
                        "input_species_effect_max_abs": (
                            input_species_effect if path_name == "joint" else None
                        ),
                        "reconstruction_metrics": (
                            json.dumps(reconstruction) if path_name == "joint" else None
                        ),
                        "reference_free_geometry_metrics": (
                            json.dumps(reference_free_geometry) if path_name == "joint" else None
                        ),
                        "generation_reference_geometry_metrics": json.dumps(
                            generation_reference_geometry
                        ),
                    }
                )

                for base_seed in args.seeds:
                    resolved_seed = generation.seed_from_target(
                        int(base_seed),
                        float(gap.item()),
                        float(heat.item()),
                    )
                    reset_random_state(resolved_seed)
                    traced_structure, traced = traced_generation_decode(
                        generation_model,
                        latent,
                        context["allowed_anions"],
                        family_mask,
                        np.random.RandomState(resolved_seed),
                        float(gap.item()),
                        context["family"],
                        args.decode_temperature,
                        args.decode_topk,
                        args.decode_tries,
                        refine=not args.no_refine,
                    )
                    reset_random_state(resolved_seed)
                    (
                        baseline_structure,
                        baseline_info,
                        baseline_gap,
                        baseline_heat,
                    ) = baseline_generation_decode(
                        generation_model,
                        latent,
                        context["allowed_anions"],
                        family_mask,
                        resolved_seed,
                        float(gap.item()),
                        context["family"],
                        args.decode_temperature,
                        args.decode_topk,
                        args.decode_tries,
                        refine=not args.no_refine,
                    )
                    endpoint_matches = endpoint_matches_baseline(
                        traced_structure,
                        traced,
                        baseline_structure,
                        baseline_info,
                        baseline_gap,
                        baseline_heat,
                    )
                    if not endpoint_matches:
                        raise AssertionError(
                            f"Instrumented generation endpoint diverged from baseline for "
                            f"{material_id}, {path_name}, seed {base_seed}"
                        )
                    if len(traced["attempt_records"]) != traced["attempts_used"]:
                        raise AssertionError("Attempt accounting differs from attempts_used")
                    projected = projected_structure_metrics(
                        traced_structure,
                        cif_root / f"{material_id}.cif",
                        context,
                        traced,
                    )
                    signature = (
                        generation.struct_signature(traced_structure)
                        if traced_structure is not None
                        else None
                    )
                    accepted = bool(traced["accepted"])
                    source_a_recovered = accepted and traced["A"] == context["A"]
                    source_b_recovered = accepted and traced["B"] == context["B"]
                    source_x_matches = accepted and traced["X"] == context["X"]
                    rejection_counts = Counter(
                        rejection["stage"] for rejection in traced["rejection_trace"]
                    )
                    attempt_source_ab_recoveries = sum(
                        record.get("sampled_A") == context["A"]
                        and record.get("sampled_B") == context["B"]
                        for record in traced["attempt_records"]
                    )
                    passed_source_ab_recoveries = sum(
                        bool(record.get("passed"))
                        and record.get("sampled_A") == context["A"]
                        and record.get("sampled_B") == context["B"]
                        for record in traced["attempt_records"]
                    )
                    attempt_source_exact_recoveries = sum(
                        record.get("sampled_A") == context["A"]
                        and record.get("sampled_B") == context["B"]
                        and record.get("sampled_X") == context["X"]
                        for record in traced["attempt_records"]
                    )
                    result_row = {
                        "material_id": material_id,
                        "composition_group": str(row["composition_group"]),
                        "latent_path": path_name,
                        "seed": int(base_seed),
                        "resolved_target_seed": int(resolved_seed),
                        "family": context["family"],
                        "fixed_real_donor_id": fixed_donor_id if path_name == "fixed_real" else None,
                        "fixed_control_same_as_recipient": (
                            material_id == fixed_donor_id if path_name == "fixed_real" else None
                        ),
                        "fixed_control_family_mismatch": (
                            context["family"] != fixed_donor_condition["family"]
                            if path_name == "fixed_real"
                            else None
                        ),
                        "source_A": context["A"],
                        "source_B": context["B"],
                        "source_X": context["X"],
                        "source_heat_provisional": float(heat.item()),
                        "source_gap_provisional": float(gap.item()),
                        "accepted": accepted,
                        "attempt_budget": int(args.decode_tries),
                        "attempts_used": int(traced["attempts_used"]),
                        "constructed_count": int(traced["constructed_count"]),
                        "attempt_source_AB_recovery_count": int(
                            attempt_source_ab_recoveries
                        ),
                        "attempt_source_exact_composition_recovery_count": int(
                            attempt_source_exact_recoveries
                        ),
                        "attempt_source_AB_recovery_rate": (
                            attempt_source_ab_recoveries / int(traced["attempts_used"])
                        ),
                        "passed_source_AB_recovery_count": int(
                            passed_source_ab_recoveries
                        ),
                        "passed_source_AB_recovery_rate": (
                            passed_source_ab_recoveries / int(traced["attempts_used"])
                        ),
                        "sampled_A": traced["A"],
                        "sampled_B": traced["B"],
                        "sampled_X": traced["X"],
                        "source_A_recovered": source_a_recovered,
                        "source_B_recovered": source_b_recovered,
                        "source_AB_recovered": source_a_recovered and source_b_recovered,
                        "source_X_matches_forced_family": source_x_matches,
                        "source_exact_composition_recovered": (
                            source_a_recovered and source_b_recovered and source_x_matches
                        ),
                        "formula": projected["formula"],
                        "structure_matcher_scale_false": projected[
                            "structure_matcher_scale_false"
                        ],
                        "structure_matcher_scale_true": projected[
                            "structure_matcher_scale_true"
                        ],
                        "structure_matcher_status": projected[
                            "structure_matcher_status"
                        ],
                        "minimum_distance_angstrom": projected["minimum_distance_angstrom"],
                        "projected_lattice_a": traced["projected_lattice_a"],
                        "ignored_raw_decoder_lattice_a": traced[
                            "ignored_raw_decoder_lattice_a"
                        ],
                        "provisional_head_heat": traced["provisional_head_heat"],
                        "provisional_head_gap": traced["provisional_head_gap"],
                        "structure_signature": signature,
                        "endpoint_matches_baseline": endpoint_matches,
                        "rejection_stage_counts": json.dumps(
                            dict(rejection_counts), sort_keys=True
                        ),
                        "rejection_trace": json.dumps(traced["rejection_trace"]),
                    }
                    stochastic_rows.append(result_row)
                    for attempt_record in traced["attempt_records"]:
                        attempt_source_a_recovered = (
                            attempt_record.get("sampled_A") == context["A"]
                        )
                        attempt_source_b_recovered = (
                            attempt_record.get("sampled_B") == context["B"]
                        )
                        attempt_source_x_matches = (
                            attempt_record.get("sampled_X") == context["X"]
                        )
                        attempt_row = {
                            "material_id": material_id,
                            "composition_group": str(row["composition_group"]),
                            "latent_path": path_name,
                            "seed": int(base_seed),
                            "resolved_target_seed": int(resolved_seed),
                            "family": context["family"],
                            "attempt": int(attempt_record["attempt"]),
                            "constructed": bool(attempt_record["constructed"]),
                            "passed": bool(attempt_record["passed"]),
                            "outcome": attempt_record.get("outcome"),
                            "rejection_stage": attempt_record.get("rejection_stage"),
                            "reason": attempt_record.get("reason"),
                            "sampled_A": attempt_record.get("sampled_A"),
                            "sampled_B": attempt_record.get("sampled_B"),
                            "sampled_X": attempt_record.get("sampled_X"),
                            "source_A_recovered": attempt_source_a_recovered,
                            "source_B_recovered": attempt_source_b_recovered,
                            "source_AB_recovered": (
                                attempt_source_a_recovered and attempt_source_b_recovered
                            ),
                            "source_X_matches_forced_family": attempt_source_x_matches,
                            "source_exact_composition_recovered": (
                                attempt_source_a_recovered
                                and attempt_source_b_recovered
                                and attempt_source_x_matches
                            ),
                            "projected_lattice_a": attempt_record.get("projected_lattice_a"),
                            "eventually_accepted": accepted,
                        }
                        attempt_rows.append(attempt_row)
                        if not attempt_record["passed"]:
                            failure_rows.append(attempt_row.copy())
                    if traced_structure is not None:
                        accepted_structures.append((result_row, traced_structure))

    repeated_latent_exact = all(
        bool(stage["all_heads_within_tolerance"])
        for stage in repeat_by_stage.values()
    )
    repeat_maxima = [
        stage["maximum_absolute_difference"] for stage in repeat_by_stage.values()
    ]
    repeated_latent_maximum_absolute = (
        max(float(value) for value in repeat_maxima)
        if all(value is not None for value in repeat_maxima)
        else None
    )

    harness_checks = {
        **frozen_preflight,
        "strict_training_checkpoint_load": bool(main_model_report["strict_state_dict_load"]),
        "strict_generation_checkpoint_load": bool(
            generation_model_report["strict_state_dict_load"]
        ),
        "frozen_eval_models": all(model_state_checks.values()),
        "site_mappings_valid": all(
            bool(item["context"]["site_mapping_valid"]) for item in prepared.values()
        ),
        "all_instrumented_endpoints_match_baseline": all(
            bool(row["endpoint_matches_baseline"]) for row in stochastic_rows
        ),
        "attempt_accounting_complete": len(attempt_rows)
        == sum(int(row["attempts_used"]) for row in stochastic_rows),
        "passed_source_AB_accounting_consistent": all(
            int(row["passed_source_AB_recovery_count"])
            == int(bool(row["source_AB_recovered"]))
            for row in stochastic_rows
        ),
    }
    scientific_checks = {
        "all_latents_finite": all_latents_finite,
        "all_unmasked_heads_finite": all_unmasked_outputs_finite,
        "all_probability_rows_valid": all_probability_rows_valid,
        "repeated_call_within_tolerance": repeated_latent_exact,
        "equivalent_B_C_within_tolerance": decoder_implementations_agree,
        "input_species_argument_has_no_numerical_effect": maximum_input_species_effect == 0.0,
    }
    hard_checks = {**harness_checks, **scientific_checks}
    if not all(harness_checks.values()):
        raise AssertionError(f"Decoder-audit harness checks failed: {harness_checks}")

    assign_structure_clusters(stochastic_rows, accepted_structures)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir = report_dir / "decoder_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "decoder_audit_raw.csv", raw_rows)
    write_csv(report_dir / "decoder_audit_results.csv", stochastic_rows)
    write_csv(report_dir / "decoder_audit_failures.csv", failure_rows)
    write_csv(run_dir / "raw_heads.csv", raw_rows)
    write_csv(run_dir / "per_sample.csv", stochastic_rows)
    write_csv(run_dir / "attempts.csv", attempt_rows)
    write_csv(run_dir / "failures.csv", failure_rows)
    write_csv(run_dir / "information_sites.csv", information_site_rows)

    representative_dir = run_dir / "representative_cifs"
    representative_dir.mkdir(parents=True, exist_ok=True)
    representatives = select_representatives(
        accepted_structures,
        args.representative_cifs,
    )
    representative_manifest = []
    for index, (result_row, structure) in enumerate(representatives, start=1):
        filename = (
            f"audit_{index:02d}_{result_row['latent_path']}_"
            f"{result_row['material_id']}_seed{result_row['seed']}.cif"
        )
        path = representative_dir / filename
        CifWriter(structure).write_file(path)
        parsed = CifParser(str(path)).parse_structures(primitive=False)[0]
        if parsed.composition != structure.composition:
            raise AssertionError(f"Representative CIF composition changed: {path}")
        representative_manifest.append(
            {
                "path": display_path(path),
                "material_id": result_row["material_id"],
                "latent_path": result_row["latent_path"],
                "seed": result_row["seed"],
                "formula": result_row["formula"],
                "sha256": sha256_file(path),
            }
        )

    joint_raw_rows = [row for row in raw_rows if row["latent_path"] == "joint"]
    reconstruction_values = [json.loads(row["reconstruction_metrics"]) for row in joint_raw_rows]
    reference_free_values = [
        json.loads(row["reference_free_geometry_metrics"]) for row in joint_raw_rows
    ]
    stochastic_summary = summarize_stochastic(stochastic_rows)
    yield_rows = grouped_yield_rows(stochastic_rows)
    write_csv(run_dir / "yield_by_condition.csv", yield_rows)

    information_comparisons: dict[str, dict[str, Any]] = {}
    for stage in ("unmasked", "masked_ab"):
        information_comparisons[stage] = {
            control: aggregate_information_comparison(raw_rows, control, stage)
            for control in LATENT_PATHS
            if control != "joint"
        }
    a_to_b_nll: dict[str, list[float]] = {}
    a_to_b_top1: dict[str, list[float]] = {}
    a_to_b_masked_ab_nll: dict[str, list[float]] = {}
    a_to_b_masked_ab_top1: dict[str, list[float]] = {}
    for row in joint_raw_rows:
        group = str(row["composition_group"])
        a_nll = row["reference_nll_nats_per_site"]
        b_nll = row["canonical_reference_free_nll_nats_per_site"]
        if a_nll is not None and b_nll is not None:
            a_to_b_nll.setdefault(group, []).append(float(b_nll) - float(a_nll))
        a_to_b_top1.setdefault(group, []).append(
            float(row["reference_top1_accuracy"])
            - float(row["canonical_reference_free_top1_accuracy"])
        )
        a_masked_nll = row["reference_masked_ab_nll_nats_per_site"]
        b_masked_nll = row["canonical_reference_free_masked_ab_nll_nats_per_site"]
        if a_masked_nll is not None and b_masked_nll is not None:
            a_to_b_masked_ab_nll.setdefault(group, []).append(
                float(b_masked_nll) - float(a_masked_nll)
            )
        a_to_b_masked_ab_top1.setdefault(group, []).append(
            float(row["reference_masked_ab_top1_accuracy"])
            - float(row["canonical_reference_free_masked_ab_top1_accuracy"])
        )
    final_control_comparisons = {
        control: paired_final_summary(stochastic_rows, control)
        for control in LATENT_PATHS
        if control != "joint"
    }

    def geometry_aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        def finite_values(key: str) -> list[float]:
            return [
                float(value[key])
                for value in values
                if value.get(key) is not None and math.isfinite(float(value[key]))
            ]

        def mean_or_none(items: list[float]) -> float | None:
            return float(np.mean(items)) if items else None

        length_errors = finite_values("mean_lattice_length_absolute_error_angstrom")
        angle_errors = finite_values("mean_lattice_angle_absolute_error_degrees")
        relative_volumes = finite_values("relative_volume_error")
        same_slot_displacements = finite_values("same_slot_periodic_rms_angstrom")
        matched_displacements = finite_values("species_matched_periodic_rms_angstrom")
        minimum_distances = [
            float(value["raw_geometry"]["first_five_minimum_periodic_distance_angstrom"])
            for value in values
            if value["raw_geometry"].get("first_five_minimum_periodic_distance_angstrom")
            is not None
            and math.isfinite(
                float(value["raw_geometry"]["first_five_minimum_periodic_distance_angstrom"])
            )
        ]
        matcher_status_counts = Counter(
            str(value.get("structure_matcher_status", "unknown")) for value in values
        )
        return {
            "examples": len(values),
            "legal_cell_count": sum(bool(value["raw_geometry"]["legal_cell"]) for value in values),
            "finite_positive_volume_count": sum(
                bool(value["raw_geometry"]["finite_positive_volume"]) for value in values
            ),
            "simple_plausibility_count": sum(
                bool(value["raw_geometry"]["simple_plausibility_window"]) for value in values
            ),
            "valid_lattice_length_error_count": len(length_errors),
            "mean_lattice_length_absolute_error_angstrom": mean_or_none(length_errors),
            "valid_lattice_angle_error_count": len(angle_errors),
            "mean_lattice_angle_absolute_error_degrees": mean_or_none(angle_errors),
            "valid_relative_volume_error_count": len(relative_volumes),
            "median_relative_volume_error": (
                float(np.median(relative_volumes)) if relative_volumes else None
            ),
            "valid_same_slot_displacement_count": len(same_slot_displacements),
            "mean_same_slot_periodic_rms_angstrom": mean_or_none(same_slot_displacements),
            "valid_species_matched_displacement_count": len(matched_displacements),
            "mean_species_matched_periodic_rms_angstrom": mean_or_none(matched_displacements),
            "structure_matcher_scale_false_count": sum(
                bool(value["structure_matcher_scale_false"]) for value in values
            ),
            "structure_matcher_scale_true_count": sum(
                bool(value["structure_matcher_scale_true"]) for value in values
            ),
            "structure_matcher_status_counts": dict(sorted(matcher_status_counts.items())),
            "valid_minimum_periodic_distance_count": len(minimum_distances),
            "minimum_periodic_distance_range_angstrom": (
                [float(min(minimum_distances)), float(max(minimum_distances))]
                if minimum_distances
                else None
            ),
        }

    raw_geometry_summary = {
        "A_reference_assisted": geometry_aggregate(reconstruction_values),
        "B_reference_free": geometry_aggregate(reference_free_values),
        "C_reference_free_by_latent_path": {
            path_name: geometry_aggregate(
                [
                    json.loads(row["generation_reference_geometry_metrics"])
                    for row in raw_rows
                    if row["latent_path"] == path_name
                ]
            )
            for path_name in LATENT_PATHS
        },
        "matcher_configuration": {
            "ltol": 0.3,
            "stol": 0.5,
            "angle_tol": 10,
            "primitive_cell": False,
            "scale_primary": False,
            "scale_topology_supplement": True,
            "attempt_supercell": False,
            "allow_subset": False,
            "comparator": "ElementComparator",
        },
        "periodic_distance_method": "pymatgen Lattice.get_all_distances; Hungarian species matching",
    }

    fixed_information = information_comparisons["unmasked"]["fixed_real"][
        "delta_nll_control_minus_joint"
    ]
    fixed_final = final_control_comparisons["fixed_real"][
        "joint_minus_control_passed_source_AB_recovery_rate"
    ]
    joint_yield = stochastic_summary["joint"]
    joint_condition_ab = condition_level_ab_summary(stochastic_rows, "joint")
    latent_nll_advantage = (
        fixed_information["ci95_low"] is not None and fixed_information["ci95_low"] > 0
    )
    final_ab_advantage = fixed_final["ci95_low"] is not None and fixed_final["ci95_low"] > 0
    final_seed_consistent = all(
        value > 0
        for value in final_control_comparisons["fixed_real"][
            "passed_source_AB_recovery_rate_difference_by_seed"
        ].values()
    )
    noncollapsed = (
        joint_yield["N_unique_accepted_AB"] > 1
        and joint_yield["dominant_accepted_AB_fraction"] is not None
        and joint_yield["dominant_accepted_AB_fraction"] < 1.0
    )
    execution_integrity_passed = all(harness_checks.values()) and all(
        scientific_checks.values()
    )
    numerical_verdict = (
        "PASSED" if execution_integrity_passed else "BLOCKED BY A SPECIFIC ISSUE"
    )
    scientific_failures = [
        name for name, passed in scientific_checks.items() if not passed
    ]
    if not execution_integrity_passed:
        composition_verdict = "BLOCKED BY A SPECIFIC ISSUE"
    elif latent_nll_advantage and final_ab_advantage and final_seed_consistent and noncollapsed:
        composition_verdict = "READY FOR A SMALL, TEMPLATE-CONSTRAINED DIFFUSION PILOT"
    elif joint_yield["N_pass"] > 1 and not noncollapsed:
        composition_verdict = "BLOCKED BY A SPECIFIC ISSUE"
    else:
        composition_verdict = "INSUFFICIENT EVIDENCE"

    reference_free_geometry_summary = raw_geometry_summary["B_reference_free"]
    if not execution_integrity_passed:
        raw_geometry_verdict = "BLOCKED BY A SPECIFIC ISSUE"
    elif reference_free_geometry_summary["structure_matcher_scale_false_count"] == 0:
        raw_geometry_verdict = "BLOCKED BY A SPECIFIC ISSUE"
    elif (
        reference_free_geometry_summary["structure_matcher_scale_false_count"]
        == len(reference_free_values)
    ):
        raw_geometry_verdict = "READY FOR A SMALL DIFFUSION PILOT"
    else:
        raw_geometry_verdict = "INSUFFICIENT EVIDENCE"

    scorecard = [
        {
            "metric": "unmasked finite fraction",
            "arm": "all latents/heads",
            "control_or_paired_difference": "1.0 required",
            "denominator": len(raw_rows),
            "uncertainty": "none; hard integrity check",
            "conclusion": "passed" if all_unmasked_outputs_finite else "failed",
        },
        {
            "metric": "B/C maximum absolute difference",
            "arm": "canonical versus generation, equivalent heads",
            "control_or_paired_difference": decoder_implementation_maximum_absolute,
            "denominator": len(raw_rows),
            "uncertainty": f"rtol={RTOL}, atol={ATOL}",
            "conclusion": "passed" if decoder_implementations_agree else "failed",
        },
        {
            "metric": "occupied-site NLL delta (control - joint)",
            "arm": "unmasked joint",
            "control_or_paired_difference": fixed_information["mean"],
            "denominator": fixed_information["independent_groups"],
            "uncertainty": (
                f"group bootstrap 95% CI [{fixed_information['ci95_low']}, "
                f"{fixed_information['ci95_high']}]"
            ),
            "conclusion": "supports joint" if latent_nll_advantage else "inconclusive",
        },
        {
            "metric": "accepted-output source A/B yield difference (joint - control)",
            "arm": "composition after projection and common filters",
            "control_or_paired_difference": fixed_final["mean"],
            "denominator": fixed_final["independent_groups"],
            "uncertainty": (
                f"group bootstrap 95% CI [{fixed_final['ci95_low']}, {fixed_final['ci95_high']}]"
            ),
            "conclusion": "supports joint" if final_ab_advantage else "inconclusive",
        },
        {
            "metric": "accepted-output source A/B direction by seed",
            "arm": "joint minus fixed-real after projection and common filters",
            "control_or_paired_difference": json.dumps(
                final_control_comparisons["fixed_real"][
                    "passed_source_AB_recovery_rate_difference_by_seed"
                ],
                sort_keys=True,
            ),
            "denominator": len(
                final_control_comparisons["fixed_real"][
                    "passed_source_AB_recovery_rate_difference_by_seed"
                ]
            ),
            "uncertainty": "three fixed seeds; positive required in each seed for readiness",
            "conclusion": "consistent" if final_seed_consistent else "not consistent",
        },
        {
            "metric": "accepted unique A/B pairs",
            "arm": "joint final output",
            "control_or_paired_difference": joint_yield["N_unique_accepted_AB"],
            "denominator": joint_yield["N_attempt"],
            "uncertainty": (
                "descriptive fixed-budget audit; forced family X excluded; "
                f"dominant accepted A/B fraction={joint_yield['dominant_accepted_AB_fraction']}"
            ),
            "conclusion": (
                "non-collapsed"
                if noncollapsed
                else ("collapsed" if joint_yield["N_pass"] > 1 else "too few accepted outputs")
            ),
        },
        {
            "metric": "material conditions with multiple accepted A/B pairs",
            "arm": "joint final output by requested material condition",
            "control_or_paired_difference": joint_condition_ab[
                "conditions_with_multiple_accepted_AB"
            ],
            "denominator": joint_condition_ab["conditions_with_accepted_output"],
            "uncertainty": "descriptive; three fixed seeds per material condition",
            "conclusion": (
                "reported separately; repeated recovery of one A/B can be expected for a fixed "
                "known-latent condition"
            ),
        },
        {
            "metric": "scale=False StructureMatcher fits",
            "arm": "reference-free raw geometry",
            "control_or_paired_difference": reference_free_geometry_summary[
                "structure_matcher_scale_false_count"
            ],
            "denominator": len(reference_free_values),
            "uncertainty": "fixed matcher settings",
            "conclusion": raw_geometry_verdict,
        },
    ]
    a_to_b_nll_summary = bootstrap_group_mean(a_to_b_nll)
    masked_fixed_information = information_comparisons["masked_ab"]["fixed_real"][
        "delta_nll_control_minus_joint"
    ]
    property_information = information_comparisons["unmasked"]["property"]
    perturbation_information = {
        name: information_comparisons["unmasked"][name]
        for name in ("joint_perturb_0p01", "joint_perturb_0p05")
    }
    scorecard.extend(
        [
            {
                "metric": "harness provenance/load/accounting checks",
                "arm": "audit harness",
                "control_or_paired_difference": sum(harness_checks.values()),
                "denominator": len(harness_checks),
                "uncertainty": "exact Boolean checks; failure invalidates the harness",
                "conclusion": "passed" if all(harness_checks.values()) else "failed",
            },
            {
                "metric": "separate-call deterministic repeats",
                "arm": "A, B, and C by decoder head",
                "control_or_paired_difference": json.dumps(
                    {
                        name: value["maximum_absolute_difference"]
                        for name, value in repeat_by_stage.items()
                    },
                    sort_keys=True,
                ),
                "denominator": sum(value["call_pairs"] for value in repeat_by_stage.values()),
                "uncertainty": f"rtol={RTOL}, atol={ATOL}; per-head records in summary JSON",
                "conclusion": "passed" if repeated_latent_exact else "failed",
            },
            {
                "metric": "A-to-B occupied-site raw NLL change",
                "arm": "joint; B minus A",
                "control_or_paired_difference": a_to_b_nll_summary["mean"],
                "denominator": a_to_b_nll_summary["independent_groups"],
                "uncertainty": (
                    f"group bootstrap 95% CI [{a_to_b_nll_summary['ci95_low']}, "
                    f"{a_to_b_nll_summary['ci95_high']}]"
                ),
                "conclusion": "positive means reference-free decoding is worse",
            },
            {
                "metric": "family-masked A/B-only NLL delta (control - joint)",
                "arm": "fixed-real versus joint",
                "control_or_paired_difference": masked_fixed_information["mean"],
                "denominator": masked_fixed_information["independent_groups"],
                "uncertainty": (
                    f"group bootstrap 95% CI [{masked_fixed_information['ci95_low']}, "
                    f"{masked_fixed_information['ci95_high']}]"
                ),
                "conclusion": "positive favors the per-example joint latent",
            },
            {
                "metric": "raw NLL delta and TV (property - joint)",
                "arm": "property-only common latent",
                "control_or_paired_difference": json.dumps(
                    {
                        "delta_nll": property_information[
                            "delta_nll_control_minus_joint"
                        ]["mean"],
                        "mean_tv": property_information["mean_total_variation"],
                    },
                    sort_keys=True,
                ),
                "denominator": property_information["paired_examples"],
                "uncertainty": "whole-group NLL bootstrap; TV descriptive",
                "conclusion": "mechanistic comparison; not an automatic readiness veto",
            },
            {
                "metric": "raw probability sensitivity to relative perturbations",
                "arm": "0.01 and 0.05 relative L2",
                "control_or_paired_difference": json.dumps(
                    {
                        name: {
                            "delta_nll": value["delta_nll_control_minus_joint"]["mean"],
                            "mean_tv": value["mean_total_variation"],
                        }
                        for name, value in perturbation_information.items()
                    },
                    sort_keys=True,
                ),
                "denominator": len(joint_raw_rows),
                "uncertainty": "same deterministic direction at both magnitudes per example",
                "conclusion": "sensitivity diagnostic only; TV is not usefulness",
            },
            {
                "metric": "attempt accounting",
                "arm": "all stochastic arms",
                "control_or_paired_difference": len(attempt_rows),
                "denominator": sum(int(row["attempts_used"]) for row in stochastic_rows),
                "uncertainty": "exact ledger equality",
                "conclusion": (
                    "passed" if harness_checks["attempt_accounting_complete"] else "failed"
                ),
            },
            {
                "metric": "joint final acceptance yield",
                "arm": "unchanged projection and common filters",
                "control_or_paired_difference": joint_yield["acceptance_yield"],
                "denominator": joint_yield["N_attempt"],
                "uncertainty": "descriptive fixed audit budget; no universal yield threshold",
                "conclusion": f"N_pass={joint_yield['N_pass']}",
            },
            {
                "metric": "raw reference-free legal cells / positive volumes",
                "arm": "B canonical reference-free",
                "control_or_paired_difference": (
                    f"{reference_free_geometry_summary['legal_cell_count']} / "
                    f"{reference_free_geometry_summary['finite_positive_volume_count']}"
                ),
                "denominator": reference_free_geometry_summary["examples"],
                "uncertainty": "documented inverse scaling; legality is not physical validity",
                "conclusion": raw_geometry_verdict,
            },
            {
                "metric": "scale=True supplementary topology fits",
                "arm": "B canonical reference-free",
                "control_or_paired_difference": reference_free_geometry_summary[
                    "structure_matcher_scale_true_count"
                ],
                "denominator": reference_free_geometry_summary["examples"],
                "uncertainty": "fixed matcher; scale=False remains primary",
                "conclusion": "supplementary topology diagnostic",
            },
        ]
    )
    write_csv(run_dir / "scorecard.csv", scorecard)

    final_hashes = {path: sha256_file(path) for path in protected_paths}
    changed = [str(path) for path, digest in protected_paths.items() if final_hashes[path] != digest]
    if changed:
        raise AssertionError(f"Protected source/data/checkpoint files changed: {changed}")

    summary = {
        "status": "passed_audit_complete",
        "audit_completion": "COMPLETE",
        "execution_integrity": numerical_verdict,
        "template_composition_readiness": composition_verdict,
        "raw_geometry_readiness": raw_geometry_verdict,
        "execution_success": True,
        "criteria_timing": (
            "Metrics addendum was read after a seed-0 debug run was inspected and before this "
            "three-seed run; added criteria are post-inspection, not preregistered."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_compatibility": "unresolved; CMR labels used only as provisional numerical inputs",
        "pretrained_training_overlap": "unknown",
        "test_set_model_inference": False,
        "input_count": len(frame),
        "independent_composition_groups": int(frame["composition_group"].nunique()),
        "runtime_seconds": time.perf_counter() - started,
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
            "seeds": [int(seed) for seed in args.seeds],
            "decode_temperature": args.decode_temperature,
            "decode_topk": args.decode_topk,
            "decode_tries_maximum_per_call": args.decode_tries,
            "perturbation_relative_l2": list(PERTURBATION_RATIOS),
            "refine": not args.no_refine,
            "history_reset_between_methods": True,
            "python_numpy_torch_rng_reset_between_methods": True,
            "anti_repeat_counts_reset_between_methods": True,
            "x_prior_strength": 0.0,
            "target_B_filter": False,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "numerical_tolerances": {"rtol": RTOL, "atol": ATOL},
            "fixed_real_donor_id": fixed_donor_id,
            "fixed_real_donor_condition": fixed_donor_condition,
            "fixed_real_condition_mismatch_possible": True,
            "rng_pairing_limit": (
                "Calls begin from matched fresh seeds; branch-dependent rejection can consume "
                "different draws, so later attempts are not draw-by-draw paired across arms."
            ),
        },
        "assets": {
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": sha256_file(audit_csv),
            "audit_material_ids": frame["material_id"].astype(str).tolist(),
            "audit_cif_sha256": {
                path.name: sha256_file(path) for path in audit_cif_paths
            },
            "cif_root": str(cif_root),
            "checkpoint": checkpoint_report,
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "protocol_expected_sha256": EXPECTED_PROTOCOL_SHA256,
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "hard_checks": hard_checks,
        "harness_checks": harness_checks,
        "scientific_checks": scientific_checks,
        "model_state_checks": model_state_checks,
        "model_compatibility": {
            "training_model": main_model_report,
            "generation_model": generation_model_report,
            "generation_is_separate_implementation": True,
            "B_C_maximum_absolute_difference": decoder_implementation_maximum_absolute,
            "deliberate_adjacency_difference": (
                "Canonical decoder returns learned node-dot-product adjacency; generation decoder "
                "returns all ones, so adjacency is not included in B/C equivalence."
            ),
            "source_references": {
                "common_latent": "MEIDNet-main/meidnet/model.py:604-629",
                "full_forward": "MEIDNet-main/meidnet/model.py:631-655",
                "generation_decoder": "MEIDNet-main/meidnet/design.py:401-447",
                "decode_and_filter": "MEIDNet-main/meidnet/design.py:934-1088",
            },
        },
        "controls": {
            "repeated_latent_exact": repeated_latent_exact,
            "repeated_latent_maximum_absolute_difference": repeated_latent_maximum_absolute,
            "repeat_by_stage": repeat_by_stage,
            "both_decoder_implementations_agree": decoder_implementations_agree,
            "input_species_numerical_effect_max_abs": maximum_input_species_effect,
            "joint_was_renormalized": False,
            "fixed_real_control": True,
            "property_only_control": True,
            "projected_crystal_only_control": True,
            "relative_perturbation_controls": list(PERTURBATION_RATIOS),
        },
        "reference_information": {
            "A_to_B_delta_nll_B_minus_A": bootstrap_group_mean(a_to_b_nll),
            "A_to_B_delta_top1_A_minus_B": bootstrap_group_mean(a_to_b_top1),
            "A_to_B_masked_AB_only_delta_nll_B_minus_A": bootstrap_group_mean(
                a_to_b_masked_ab_nll
            ),
            "A_to_B_masked_AB_only_delta_top1_A_minus_B": bootstrap_group_mean(
                a_to_b_masked_ab_top1
            ),
            "A_to_B_mean_total_variation": float(
                np.mean([row["reference_to_reference_free_mean_tv"] for row in joint_raw_rows])
            ),
            "A_to_B_masked_AB_only_mean_total_variation": float(
                np.mean(
                    [
                        row["reference_to_reference_free_masked_ab_mean_tv"]
                        for row in joint_raw_rows
                    ]
                )
            ),
            "control_comparisons": information_comparisons,
            "A_masked_AB_only_zero_true_probability_sites": sum(
                int(row["reference_masked_ab_zero_true_probability_sites"])
                for row in joint_raw_rows
            ),
            "B_masked_AB_only_zero_true_probability_sites": sum(
                int(row["canonical_reference_free_masked_ab_zero_true_probability_sites"])
                for row in joint_raw_rows
            ),
            "C_masked_AB_only_zero_true_probability_sites": sum(
                int(row["masked_ab_zero_true_probability_sites"]) for row in raw_rows
            ),
            "masked_all_five_forced_X_including_zero_true_probability_sites": sum(
                int(row["masked_all_five_zero_true_probability_sites"])
                for row in raw_rows
            ),
            "site_mapping_identity_count": sum(
                bool(row["site_mapping_identity"])
                for row in joint_raw_rows
            ),
            "padding_scored": False,
            "primary_metric": "unmasked occupied-site NLL in nats/site",
            "masked_metric": (
                "family-masked A/B-only NLL in nats/site; forced X slots are excluded"
            ),
            "full_site_evidence_file": display_path(run_dir / "information_sites.csv"),
        },
        "raw_geometry": raw_geometry_summary,
        "stochastic_generation": stochastic_summary,
        "accepted_AB_by_material_condition": joint_condition_ab,
        "final_control_comparisons": final_control_comparisons,
        "yield_by_condition_file": display_path(run_dir / "yield_by_condition.csv"),
        "attempt_count": len(attempt_rows),
        "failure_attempt_count": len(failure_rows),
        "representative_cif_count": len(representative_manifest),
        "representative_cifs": representative_manifest,
        "generation_projection": {
            "raw_lattice_used_in_final_cif": False,
            "raw_coordinates_used_in_final_cif": False,
            "final_coordinates": "fixed five-site cubic Pm-3m template",
            "final_lattice": "ionic-radius clip(2*(r_B+r_X), 3, 8) cubic cell",
            "source_X_forced_by_declared_family_mask": True,
            "X_recovery_credited_as_learned": False,
            "deduplication_in_per_call_audit": False,
        },
        "readiness": {
            "numerical_execution": numerical_verdict,
            "execution_integrity_passed": execution_integrity_passed,
            "scientific_failures": scientific_failures,
            "raw_geometry_generation": raw_geometry_verdict,
            "template_constrained_composition_generation": composition_verdict,
            "template_composition_evidence": {
                "execution_integrity_passed": execution_integrity_passed,
                "unmasked_NLL_advantage_over_fixed_real": latent_nll_advantage,
                "accepted_output_AB_yield_advantage_over_fixed_real": final_ab_advantage,
                "accepted_output_AB_direction_positive_in_every_seed": final_seed_consistent,
                "accepted_AB_noncollapsed": noncollapsed,
                "joint_N_pass": joint_yield["N_pass"],
                "joint_N_unique_accepted_AB": joint_yield["N_unique_accepted_AB"],
                "joint_dominant_accepted_AB_fraction": joint_yield[
                    "dominant_accepted_AB_fraction"
                ],
            },
            "decision_logic": (
                "Composition readiness requires the frozen protocol's hard checks, a group-bootstrap "
                "NLL advantage over fixed-real control, a group-bootstrap accepted-output A/B-recovery "
                "yield advantage after projection and common filters, the same positive final direction "
                "in every seed, and accepted A/B non-collapse. Forced family X and pre-filter samples "
                "cannot satisfy this gate. No yield cutoff was introduced."
            ),
        },
        "scorecard": scorecard,
        "latent_metadata": latent_metadata,
        "protected_asset_hashes_unchanged": True,
        "protected_asset_sha256": {
            display_path(path): digest for path, digest in protected_paths.items()
        },
    }

    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "command": summary["environment"]["command"],
        "environment": summary["environment"],
        "configuration": summary["configuration"],
        "assets": summary["assets"],
        "comparison_arms": {
            "A": "canonical decoder with source reconstruction coordinates and center",
            "B": "same canonical decoder with zero coordinates and center",
            "C": "separate generation decoder, unmasked then declared family mask",
            "D": "unchanged stochastic composition rules, template projection, and filters",
        },
        "latent_paths": list(LATENT_PATHS),
        "hard_checks": hard_checks,
        "harness_checks": harness_checks,
        "scientific_checks": scientific_checks,
        "source_family_is_declared_condition": True,
        "source_X_is_forced_and_not_credited": True,
        "criteria_post_inspection": True,
        "protocol": display_path(PROTOCOL_PATH),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = report_dir / "decoder_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = report_dir / "decoder_audit.md"
    markdown_path.write_text(markdown_report(summary), encoding="utf-8", newline="\n")
    print(f"Decoder audit complete: {markdown_path}")
    print(
        "  final N_pass by latent path:",
        {name: summary["stochastic_generation"][name]["N_pass"] for name in LATENT_PATHS},
    )
    print(f"  raw geometry verdict: {raw_geometry_verdict}")
    print(f"  constrained composition verdict: {composition_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
