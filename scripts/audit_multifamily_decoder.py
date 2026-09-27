#!/usr/bin/env python3
"""Frozen, validation-only multifamily decoder and output-policy audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pymatgen.core import Composition, Lattice, Structure

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "MEIDNet-main"), str(ROOT)]

from meidnet_audits.multifamily_accounting import (  # noqa: E402
    ATTEMPTS, LATENT_ARMS, assert_replay_and_budget, paired_group_effect,
    policy_pair_comparisons,
)
import audit_decoder as previous  # noqa: E402
import audit_template_reachability as template_audit  # noqa: E402
import multifamily_census as census  # noqa: E402
import multifamily_panel as panel  # noqa: E402
import multifamily_policies as policies  # noqa: E402
from meidnet import design as legacy  # noqa: E402
from meidnet.model import (  # noqa: E402
    MAX_SITES, NUM_SPECIES, SPECIES_LIST, TripleModalityDataset,
    parse_cif_to_dense,
)

DATA_ROOT = ROOT / "data" / "processed" / "cmr_reconstructed"
REPORT_ROOT = ROOT / "reports" / "multifamily_rule_relaxation"
INPUT_MANIFEST = ROOT / "reports" / "template_reachability" / "input_manifest.json"
OLD_AUDIT = DATA_ROOT / "audit.csv"
CHECKPOINT = previous.DEFAULT_CHECKPOINT
SEEDS = (0, 1, 2)
TEMPERATURE = 1.25
EXPECTED_COORD_HASH = "2b92f48bde0008fe445fd81bc63a032b6121116e2080810acaf3ae6bd7315459"
EXPECTED_CENTER_HASH = "c4be0f035fc77ce3b797c6acf4ead43bec2d631e9e4eae87a5a18397bfe5e312"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--max-panel", type=int, default=384)
    parser.add_argument("--max-stochastic", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(itertools.chain.from_iterable(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def current_git() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "status_short": git("status", "--short").splitlines()}


def template_input() -> tuple[torch.Tensor, torch.Tensor, dict]:
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    coordinates = torch.tensor(manifest["coordinates"], dtype=torch.float32)
    center = torch.tensor(manifest["center_from_first_five"], dtype=torch.float32)
    assert coordinates.shape == (MAX_SITES, 3) and center.shape == (3,)
    assert template_audit.sha256_tensor_bytes(coordinates) == manifest["coordinates_sha256"] == EXPECTED_COORD_HASH
    assert template_audit.sha256_tensor_bytes(center) == manifest["center_sha256"] == EXPECTED_CENTER_HASH
    assert torch.equal(center, coordinates[:5].mean(0))
    return coordinates, center, manifest


def checked_roles(row: dict) -> dict:
    roles = census.source_roles(row)
    source = json.loads(row["source_site_symbols"])
    parsed = json.loads(row["meidnet_site_symbols"])
    permutation = [int(index) for index in json.loads(row["site_permutation"])]
    valid = (len(source) == len(parsed) == len(permutation) == 5
             and sorted(permutation) == list(range(5))
             and all(source[i] == parsed[permutation[i]] for i in range(5))
             and source[:2] == [row["source_A_ion"], row["source_B_ion"]])
    if roles["role_status"] != "VERIFIED" or not valid:
        raise ValueError(f"Unusable source/parser role map for {row['material_id']}: {roles['role_reason']}")
    return roles | {
        "source_symbols": source,
        "parsed_symbols": parsed,
        "source_to_parser": permutation,
        "parser_ab_aligned": permutation[:2] == [0, 1],
        "parser_all_roles_aligned": permutation == list(range(5)),
    }


def reference_from_dense(crystal: torch.Tensor, roles: dict) -> Structure:
    vector = crystal.detach().cpu().numpy()
    parameters = previous.lattice_parameters(torch.from_numpy(vector[:6]))
    coordinates = vector[6 + MAX_SITES * MAX_SITES + MAX_SITES * NUM_SPECIES:].reshape(MAX_SITES, 3)
    return Structure(Lattice.from_parameters(*parameters), roles["parsed_symbols"], coordinates[:5],
                     coords_are_cartesian=False)


def load_selected(data_root: Path, selected_ids: list[str]) -> dict[str, dict]:
    validation = pd.read_csv(data_root / "val.csv", dtype={"material_id": "string"})
    selected = validation[validation["material_id"].isin(selected_ids)].copy()
    if len(selected) != len(selected_ids) or selected["material_id"].duplicated().any():
        raise AssertionError("Selected validation IDs missing or duplicated")
    with tempfile.TemporaryDirectory(prefix="meidnet_multifamily_") as scratch:
        csv_path = Path(scratch) / "selected.csv"
        selected.to_csv(csv_path, index=False)
        dataset = TripleModalityDataset(str(data_root / "cifs" / "val"), str(csv_path))
        loaded = {str(dataset[i]["material_id"]): dataset[i] for i in range(len(dataset))}
    if set(loaded) != set(selected_ids) or len(loaded) != len(selected_ids):
        raise AssertionError(f"Dataset loader skipped selected IDs: {sorted(set(selected_ids)-set(loaded))}")
    rows = {str(row["material_id"]): row.to_dict() for _, row in selected.iterrows()}
    return {material_id: {"row": rows[material_id], "sample": loaded[material_id],
                          "roles": checked_roles(rows[material_id])}
            for material_id in selected_ids}


def donor_joint(model: torch.nn.Module, data_root: Path) -> tuple[str, torch.Tensor]:
    frame = pd.read_csv(OLD_AUDIT, dtype={"material_id": "string"}).sort_values("material_id")
    donor_id = str(frame.iloc[0]["material_id"])
    donor_path = data_root / "cifs" / "val" / f"{donor_id}.cif"
    crystal = torch.from_numpy(parse_cif_to_dense(str(donor_path))).float().unsqueeze(0)
    heat = torch.tensor([float(frame.iloc[0]["heat_all"])], dtype=torch.float32)
    gap = torch.tensor([float(frame.iloc[0]["dir_gap"])], dtype=torch.float32)
    with torch.inference_mode():
        _, _, joint, _, _, _ = model.encode_modalities(crystal, heat, gap)
    return donor_id, joint.detach().cpu()


def encode_and_decode(
    model: torch.nn.Module, selected: dict[str, dict], coordinates: torch.Tensor,
    center: torch.Tensor, batch_size: int, fixed_joint: torch.Tensor,
) -> dict[str, dict]:
    """Cache three latent arms and their unmasked template heads once."""
    ids = list(selected)
    cache: dict[str, dict] = {}
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            crystal = torch.stack([selected[mid]["sample"]["crystal_vec"] for mid in chunk])
            heat = torch.stack([selected[mid]["sample"]["heat_all"] for mid in chunk])
            gap = torch.stack([selected[mid]["sample"]["dir_gap"] for mid in chunk])
            zc, zp, joint, _, _, _ = model.encode_modalities(crystal, heat, gap)
            latents = {"joint": joint, "fixed_real": fixed_joint.expand(len(chunk), -1),
                       "property": zp}
            for arm, latent in latents.items():
                coords = coordinates.unsqueeze(0).expand(len(chunk), -1, -1)
                centers = center.unsqueeze(0).expand(len(chunk), -1)
                heads = model.crystal_decoder(latent, input_coords=coords, center=centers)
                if not all(bool(torch.isfinite(head).all()) for head in heads):
                    raise ArithmeticError(f"Nonfinite unmasked head in {arm} batch {start}")
                property_head = model.property_decoder(latent)
                if not bool(torch.isfinite(property_head).all()):
                    raise ArithmeticError(f"Nonfinite property head in {arm} batch {start}")
                for i, material_id in enumerate(chunk):
                    cache.setdefault(material_id, {})[arm] = {
                        "latent": latent[i:i + 1].detach().cpu(),
                        "heads": tuple(head[i:i + 1].detach().cpu() for head in heads),
                        "property": property_head[i:i + 1].detach().cpu(),
                    }
    assert set(cache) == set(ids) and all(set(value) == set(LATENT_ARMS) for value in cache.values())
    return cache


def counter_key(symbols: list[str] | tuple[str, ...]) -> str:
    return "|".join(f"{element}:{count}" for element, count in sorted(Counter(symbols).items()))


def sampled_training_presence(symbols: tuple[str, ...] | None, training: dict) -> tuple[bool | None, bool | None]:
    """Check generated combinations against training, without filtering them."""
    if symbols is None:
        return None, None
    return (census.anion_key(symbols[2:]) in training["anion_counts"],
            counter_key(symbols) in training["composition_counts"])


def role_probabilities(logits: torch.Tensor, support: tuple[str, ...], slot: int) -> dict[str, float]:
    if not support:
        return {}
    indices = [SPECIES_LIST.index(symbol) for symbol in support]
    scores = logits[slot, indices].double() / TEMPERATURE
    probabilities = torch.softmax(scores, dim=0)
    if not bool(torch.isfinite(probabilities).all()) or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-8):
        raise ArithmeticError(f"Invalid role distribution in slot {slot}")
    return {symbol: float(probability) for symbol, probability in zip(support, probabilities)}


def multiset_probability(probabilities: list[dict[str, float]], source_x: tuple[str, ...]) -> float:
    """Sum distinct X-site assignments; never choose a favorable permutation."""
    result = 0.0
    for assignment in set(itertools.permutations(source_x)):
        result += math.prod(probabilities[slot].get(symbol, 0.0) for slot, symbol in enumerate(assignment))
    return result


def match_structure(reference: Structure, candidate: Structure | None) -> tuple[bool, bool]:
    if candidate is None or reference.composition != candidate.composition:
        return False, False
    try:
        return (bool(previous.matcher(False).fit(reference, candidate)),
                bool(previous.matcher(True).fit(reference, candidate)))
    except (ValueError, RuntimeError, OverflowError) as error:
        MATCHER_FAILURES.append({"reference": reference.formula,
                                 "candidate": candidate.formula,
                                 "error": f"{type(error).__name__}: {error}"})
        return False, False


MATCHER_FAILURES: list[dict] = []


def raw_candidate(heads: tuple[torch.Tensor, ...]) -> tuple[Structure | None, dict]:
    lattice_head, _, logits, coordinates = heads
    parameters = previous.lattice_parameters(lattice_head[0])
    details = {"raw_lengths_angstrom": parameters[:3], "raw_angles_degrees": parameters[3:]}
    if (not all(math.isfinite(x) for x in parameters)
            or any(x <= 0 for x in parameters[:3])
            or any(not 0 < x < 180 for x in parameters[3:])):
        return None, details | {"reason": "INVALID_RAW_CELL"}
    symbols = [SPECIES_LIST[int(index)] for index in logits[0, :5].argmax(-1).tolist()]
    try:
        structure = Structure(Lattice.from_parameters(*parameters), symbols,
                              coordinates[0, :5].numpy(), coords_are_cartesian=False)
    except (ValueError, TypeError) as error:
        return None, details | {"reason": f"RAW_STRUCTURE_ERROR:{type(error).__name__}"}
    return structure, details | {"reason": None, "symbols": symbols}


def top1_roles(logits: torch.Tensor, policy: policies.PolicyConfig,
               vocab: dict, declared_anions: tuple[str, ...] | None = None) -> tuple[str, ...]:
    support = policies.allowed_roles(policy, vocab, declared_anions)
    def winner(slot: int, symbols: tuple[str, ...]) -> str:
        indices = [SPECIES_LIST.index(symbol) for symbol in symbols]
        return symbols[int(logits[slot, indices].argmax())]
    a, b = winner(0, support["A"]), winner(1, support["B"])
    if policy.policy_id == "MF_P4":
        x = tuple(winner(slot, support["X"]) for slot in (2, 3, 4))
    else:
        indices = [SPECIES_LIST.index(symbol) for symbol in support["X"]]
        pooled = logits[2:5, indices].sum(0)
        symbol = support["X"][int(pooled.argmax())]
        x = (symbol, symbol, symbol)
    return (a, b, *x)


def deterministic_rows(selected: dict[str, dict], cache: dict[str, dict], vocab: dict) -> list[dict]:
    results: list[dict] = []
    all_elements = {role: tuple(SPECIES_LIST) for role in ("A", "B", "X")}
    for material_id, item in selected.items():
        role_info = item["roles"]
        source_symbols = tuple(role_info["source_symbols"])
        source_x = tuple(role_info["X"])
        reference = reference_from_dense(item["sample"]["crystal_vec"], role_info)
        for arm in LATENT_ARMS:
            heads = cache[material_id][arm]["heads"]
            lattice_head, _, logits_batch, _ = heads
            logits = logits_batch[0]
            parser_slots = role_info["source_to_parser"]
            parser_top1 = [SPECIES_LIST[int(logits[slot].argmax())] for slot in parser_slots]
            parser_probabilities = [
                role_probabilities(logits, tuple(SPECIES_LIST), slot)
                for slot in parser_slots
            ]
            parser_nll = [
                -math.log(parser_probabilities[i][symbol])
                if parser_probabilities[i][symbol] > 0 else None
                for i, symbol in enumerate(source_symbols)
            ]
            masked = [
                role_probabilities(logits, tuple(vocab[role]), slot)
                for role, slot in zip(("A", "B", "X", "X", "X"), parser_slots)
            ]
            masked_nll = [(-math.log(probabilities[symbol]) if probabilities.get(symbol, 0) > 0 else None)
                          for probabilities, symbol in zip(masked, source_symbols)]
            x_multiset_p = multiset_probability(masked[2:], source_x)
            fixed_a = role_probabilities(logits, tuple(vocab["A"]), 0)
            fixed_b = role_probabilities(logits, tuple(vocab["B"]), 1)
            fixed_x = [role_probabilities(logits, tuple(vocab["X"]), slot)
                       for slot in (2, 3, 4)]
            fixed_x_multiset_p = multiset_probability(fixed_x, source_x)
            raw, raw_details = raw_candidate(heads)
            raw_match, raw_topology_match = match_structure(reference, raw)
            raw_minimal = (policies.minimal_check(raw, raw_details.get("symbols", []), all_elements)
                           if raw is not None else {"valid": False, "minimum_periodic_distance_angstrom": None,
                                                      "failure_reasons": [raw_details["reason"]]})
            parameters = previous.lattice_parameters(lattice_head[0])
            reference_lengths = reference.lattice.abc
            cubic_length = float(np.mean(parameters[:3]))
            base = {
                "material_id": material_id,
                "panel_membership": item["panel_membership"],
                "composition_group": item["row"]["composition_group"],
                "source_family": role_info["family"],
                "source_anion_key": role_info["anion_key"],
                "source_mixed": len(set(source_x)) > 1,
                "latent_arm": arm,
                "parser_ab_aligned": role_info["parser_ab_aligned"],
                "parser_permutation": parser_slots,
                "source_roles": source_symbols,
                "parser_top1_roles": parser_top1,
                "parser_A_top1": parser_top1[0] == source_symbols[0],
                "parser_B_top1": parser_top1[1] == source_symbols[1],
                "parser_X_ordered_top1": parser_top1[2:] == list(source_x),
                "parser_X_multiset_top1": Counter(parser_top1[2:]) == Counter(source_x),
                "parser_full_composition_top1": Counter(parser_top1) == Counter(source_symbols),
                "parser_A_B_nll_all118": (float(np.mean(parser_nll[:2]))
                                              if all(value is not None for value in parser_nll[:2]) else None),
                "parser_A_B_nll_all118_status": ("FINITE" if all(value is not None for value in parser_nll[:2])
                                                  else "ZERO_NUMERICAL_PROBABILITY"),
                "parser_X_nll_all118": (float(np.mean(parser_nll[2:]))
                                           if all(value is not None for value in parser_nll[2:]) else None),
                "parser_X_nll_all118_status": ("FINITE" if all(value is not None for value in parser_nll[2:])
                                                else "ZERO_NUMERICAL_PROBABILITY"),
                "parser_A_B_nll_role_vocab": (float(np.mean(masked_nll[:2]))
                                                   if all(value is not None for value in masked_nll[:2]) else None),
                "parser_A_B_nll_role_vocab_status": ("FINITE" if all(value is not None for value in masked_nll[:2])
                                                      else "OUT_OF_SUPPORT_OR_ZERO_PROBABILITY"),
                "parser_X_nll_role_vocab": (float(np.mean(masked_nll[2:]))
                                                 if all(value is not None for value in masked_nll[2:]) else None),
                "parser_X_nll_role_vocab_status": ("FINITE" if all(value is not None for value in masked_nll[2:])
                                                    else "OUT_OF_SUPPORT_OR_ZERO_PROBABILITY"),
                "parser_X_multiset_probability": x_multiset_p,
                "parser_X_multiset_nll": (-math.log(x_multiset_p) if x_multiset_p > 0 else None),
                "parser_X_multiset_nll_status": ("FINITE" if x_multiset_p > 0 else "OUT_OF_SUPPORT_OR_ZERO_PROBABILITY"),
                "fixed_role_A_B_probability": fixed_a.get(source_symbols[0], 0.0)
                                                * fixed_b.get(source_symbols[1], 0.0),
                "fixed_role_A_B_nll": (-math.log(fixed_a[source_symbols[0]])
                                       - math.log(fixed_b[source_symbols[1]])
                                       if fixed_a.get(source_symbols[0], 0) > 0
                                       and fixed_b.get(source_symbols[1], 0) > 0 else None),
                "fixed_X_multiset_probability": fixed_x_multiset_p,
                "fixed_X_multiset_nll": (-math.log(fixed_x_multiset_p) if fixed_x_multiset_p > 0 else None),
                "raw_lattice_lengths_angstrom": parameters[:3],
                "raw_lattice_angles_degrees": parameters[3:],
                "raw_lattice_length_mae_angstrom": float(np.mean(np.abs(np.asarray(parameters[:3]) - reference_lengths))),
                "raw_mean_cubic_length_angstrom": cubic_length,
                "learned_cubic_length_mae_angstrom": float(np.mean(
                    np.abs(cubic_length - np.asarray(reference_lengths)))),
                "learned_cubic_volume_relative_error": (abs(cubic_length ** 3 / reference.volume - 1)
                                                        if cubic_length > 0 else None),
                "raw_geometry_minimal_only": raw_minimal["valid"],
                "raw_geometry_check_role_support": "ALL_118_ONLY_GEOMETRY_DIAGNOSTIC",
                "raw_minimum_periodic_distance_angstrom": raw_minimal["minimum_periodic_distance_angstrom"],
                "raw_geometry_failures": raw_minimal["failure_reasons"],
                "raw_scale_false_match": raw_match,
                "raw_scale_true_match": raw_topology_match,
                "provisional_property_head": cache[material_id][arm]["property"][0].tolist(),
            }
            for policy_id in ("MF_P3", "MF_P4"):
                config = policies.PolicyConfig(policy_id)
                source_scope = policies.source_scope(source_symbols, config, vocab)
                top_roles = top1_roles(logits, config, vocab)
                constructed, details = policies.construct_candidate(top_roles, config, lattice_head[0])
                minimal = policies.minimal_check(constructed, top_roles,
                                                 policies.allowed_roles(config, vocab))
                scale_false, scale_true = match_structure(reference, constructed)
                family, _, _ = census.classify_family(top_roles[2:])
                generated_legacy_family = census.classify_family(top_roles[2:])[2]
                legacy_annotation = legacy_rescore(constructed, top_roles, generated_legacy_family, vocab)
                prefix = policy_id.lower()
                base.update({
                    f"{prefix}_learned_top1_roles": top_roles,
                    f"{prefix}_source_scope_eligible": source_scope["eligible"],
                    f"{prefix}_source_scope_reasons": source_scope["reasons"],
                    f"{prefix}_learned_family": family,
                    f"{prefix}_learned_A_B_exact": top_roles[:2] == source_symbols[:2],
                    f"{prefix}_learned_X_ordered_exact": top_roles[2:] == source_x,
                    f"{prefix}_learned_X_multiset_exact": Counter(top_roles[2:]) == Counter(source_x),
                    f"{prefix}_learned_full_composition_exact": Counter(top_roles) == Counter(source_symbols),
                    f"{prefix}_learned_constructed": constructed is not None,
                    f"{prefix}_learned_minimal_valid": minimal["valid"],
                    f"{prefix}_learned_minimum_distance_angstrom": minimal["minimum_periodic_distance_angstrom"],
                    f"{prefix}_learned_scale_false_match": scale_false,
                    f"{prefix}_learned_scale_true_match": scale_true,
                    f"{prefix}_learned_cell_reason": details["reason"],
                    f"{prefix}_learned_legacy_rescore_status": legacy_annotation["status"],
                    f"{prefix}_learned_legacy_rescore_first_failure": legacy_annotation["first_failure"],
                })
            results.append(base)
    return results


class CachedLegacyModel:
    """Expose cached unmasked heads to the untouched one-attempt legacy path."""

    def __init__(self, heads: tuple[torch.Tensor, ...], property_head: torch.Tensor):
        self.heads = heads
        self.property_head = property_head

    def property_decoder(self, latent: torch.Tensor) -> torch.Tensor:
        return self.property_head

    def crystal_decoder(self, latent: torch.Tensor, input_coords=None, center=None,
                        species_mask=None) -> tuple[torch.Tensor, ...]:
        lattice, adjacency, logits, coordinates = self.heads
        if species_mask is not None:
            masked = ~species_mask.bool().unsqueeze(0)
            logits = logits.masked_fill(masked, torch.finfo(logits.dtype).min)
        return lattice, adjacency, logits, coordinates


def legacy_rescore(structure: Structure | None, roles: tuple[str, ...],
                   legacy_family: str | None, vocab: dict) -> dict:
    """Annotate identical constructed geometry; no reconstruction or refinement."""
    if structure is None:
        return {"status": "NOT_CONSTRUCTED", "first_failure": None}
    if len(set(roles[2:])) != 1:
        return {"status": "OUTSIDE_LEGACY_SCOPE", "first_failure": "mixed_X",
                "tolerance_mu_status": "NOT_APPLICABLE"}
    if legacy_family not in policies.LEGACY_FAMILY_X:
        return {"status": "OUTSIDE_LEGACY_SCOPE", "first_failure": "family_unavailable"}
    config = policies.PolicyConfig("MF_P0", family=legacy_family)
    scope = policies.source_scope(roles, config, vocab)
    if not scope["eligible"]:
        return {"status": "OUTSIDE_LEGACY_SCOPE", "first_failure": scope["reasons"][0]}
    checks = [
        ("is_ABX3", not legacy.is_ABX3(structure)),
        ("realism", bool(legacy.realism(structure))),
        ("family", legacy.family_from_structure(structure) != legacy_family),
        ("charge", bool(legacy.check_charge_balance_existential(structure))),
        ("B_X_distance", bool(legacy.check_BX_distances_family(structure))),
    ]
    tolerance_failures, t_value, mu_value = legacy.goldschmidt_and_mu_ok(structure)
    checks.append(("tolerance_mu", bool(tolerance_failures)))
    first = next((name for name, failed in checks if failed), None)
    return {"status": "PASS" if first is None else "FAIL", "first_failure": first,
            "tolerance_mu_status": "FAIL" if tolerance_failures else "PASS",
            "tolerance_factor": float(t_value), "octahedral_factor": float(mu_value)}


def policy_variants(role_info: dict) -> list[tuple[policies.PolicyConfig, tuple[str, ...] | None]]:
    """Global learned-anion tasks always run; family tasks need homogeneous X."""
    variants: list[tuple[policies.PolicyConfig, tuple[str, ...] | None]] = [
        (policies.PolicyConfig("MF_P3", anion_setting="LEARNED_ANION"), None),
        (policies.PolicyConfig("MF_P4", anion_setting="LEARNED_ANION"), None),
    ]
    family = role_info["legacy_family"]
    if family in policies.LEGACY_FAMILY_X and len(set(role_info["X"])) == 1:
        declared = policies.LEGACY_FAMILY_X[family]
        for policy_id in ("MF_P0", "MF_P1", "MF_P2"):
            variants.append((policies.PolicyConfig(policy_id, family=family), None))
        for policy_id in ("MF_P3", "MF_P4"):
            variants.append((policies.PolicyConfig(policy_id, family=family,
                                                   anion_setting="DECLARED_DOMAIN"), declared))
    return variants


def legacy_one_draw(cache_item: dict, family: str, seed: int,
                    source_gap: float) -> tuple[Structure | None, dict]:
    allowed_x = set(policies.LEGACY_FAMILY_X[family])
    mask = legacy.build_species_mask(allowed_x)
    wrapper = CachedLegacyModel(cache_item["heads"], cache_item["property"])
    result, trace = previous.traced_generation_decode(
        wrapper, cache_item["latent"], allowed_x, mask,
        np.random.RandomState(seed % (2**32)), source_gap, family,
        TEMPERATURE, 12, 1, True,
    )
    if len(trace["attempt_records"]) != 1:
        raise AssertionError("Legacy one-draw trace did not account for one attempt")
    return result, trace["attempt_records"][0]


def verify_legacy_cache_parity(generation_model: torch.nn.Module, selected: dict[str, dict],
                               cache: dict[str, dict], coordinates: torch.Tensor,
                               center: torch.Tensor) -> dict:
    """Small control against original generation with the same fixed input."""
    adapter = template_audit.GeometryInputAdapter(generation_model, "template", coordinates, center)
    eligible = [mid for mid in selected if selected[mid]["roles"]["legacy_family"] in
                policies.LEGACY_FAMILY_X and len(set(selected[mid]["roles"]["X"])) == 1]
    chosen = []
    for family in sorted({selected[mid]["roles"]["legacy_family"] for mid in eligible}):
        chosen.append(next(mid for mid in eligible if selected[mid]["roles"]["legacy_family"] == family))
    chosen += [mid for mid in eligible if mid not in chosen][:max(0, 4 - len(chosen))]
    checks = []
    with torch.inference_mode():
        for material_id in chosen:
            roles = selected[material_id]["roles"]
            item = cache[material_id]["joint"]
            family = roles["legacy_family"]
            allowed = set(policies.LEGACY_FAMILY_X[family])
            mask = legacy.build_species_mask(allowed)
            original_heads = adapter.crystal_decoder(item["latent"], species_mask=None)
            # Training decoder returns node-dot adjacency logits; design.py
            # returns its fixed adjacency. Generation consumes neither value.
            heads_equal = all(torch.allclose(original_heads[index], item["heads"][index],
                                             rtol=1e-5, atol=1e-6)
                              for index in (0, 2, 3))
            rng_seed = policies.proposal_seed("legacy-parity", material_id) % (2**32)
            original, _, _, _ = previous.baseline_generation_decode(
                adapter, item["latent"], allowed, mask, rng_seed,
                float(selected[material_id]["row"]["dir_gap"]), family,
                TEMPERATURE, 12, 1, True,
            )
            cached, trace = legacy_one_draw(item, family, rng_seed,
                                            float(selected[material_id]["row"]["dir_gap"]))
            endpoint_equal = previous.structures_match_trace(original, cached)
            checks.append({"material_id": material_id, "head_parity": heads_equal,
                           "endpoint_parity": endpoint_equal, "stage": trace["rejection_stage"]})
    if not checks or not all(row["head_parity"] and row["endpoint_parity"] for row in checks):
        raise AssertionError(f"Legacy cache parity failed: {checks}")
    return {"checked": len(checks), "rows": checks, "passed": True,
            "adjacency_head_not_compared": "training and generation expose different unused adjacency outputs"}


def stochastic_rows(selected: dict[str, dict], cache: dict[str, dict],
                    stochastic_ids: list[str], vocab: dict, training: dict,
                    population: str = "balanced_validation_stochastic_panel") -> tuple[list[dict], dict[int, Structure]]:
    ledger: list[dict] = []
    accepted_structures: dict[int, Structure] = {}
    for material_id in stochastic_ids:
        item = selected[material_id]
        roles_info = item["roles"]
        source = tuple(roles_info["source_symbols"])
        reference = reference_from_dense(item["sample"]["crystal_vec"], roles_info)
        variants = policy_variants(roles_info)
        for arm in LATENT_ARMS:
            cached = cache[material_id][arm]
            lat_out, _, logits_batch, _ = cached["heads"]
            for config, declared in variants:
                source_scope = policies.source_scope(source, config, vocab, declared)
                support = policies.allowed_roles(config, vocab, declared)
                for replicate in SEEDS:
                    for attempt in range(ATTEMPTS):
                        family = roles_info["legacy_family"]
                        shared_seed_label = ("P1P2" if config.policy_id in ("MF_P1", "MF_P2")
                                             else "P3P4" if config.anion_setting == "LEARNED_ANION"
                                             and config.policy_id in ("MF_P3", "MF_P4")
                                             else config.policy_id + "_" + config.anion_setting)
                        draw_seed = policies.proposal_seed("multifamily_v1", material_id, arm,
                                                           replicate, attempt, shared_seed_label)
                        distribution_diagnostics = []
                        selected_probabilities = []
                        if config.policy_id == "MF_P0":
                            structure, trace = legacy_one_draw(
                                cached, family, draw_seed,
                                float(item["row"]["dir_gap"]),
                            )
                            sampled_roles = None
                            if trace.get("sampled_A") and trace.get("sampled_B") and trace.get("sampled_X"):
                                sampled_roles = (trace["sampled_A"], trace["sampled_B"],
                                                 *([trace["sampled_X"]] * 3))
                            sampled = sampled_roles is not None
                            constructed = bool(trace["constructed"])
                            accepted = bool(trace["passed"])
                            reason = trace.get("rejection_stage")
                            cell_details = {"projected_cubic_length_angstrom": trace.get("projected_lattice_a"),
                                            "cell_source": "legacy_radius_derived"}
                        else:
                            proposal = policies.sample_roles(logits_batch[0], config, vocab,
                                                             draw_seed, declared)
                            sampled = bool(proposal["sampled"])
                            sampled_roles = tuple(proposal["roles"]) if sampled else None
                            distribution_diagnostics = proposal.get("distribution_diagnostics", [])
                            selected_probabilities = proposal.get("selected_probabilities", [])
                            if sampled:
                                structure, cell_details = policies.construct_candidate(
                                    sampled_roles, config, lat_out[0],
                                )
                            else:
                                structure = None
                                cell_details = {"constructed": False, "reason": proposal["reason"]}
                            constructed = structure is not None
                            reason = cell_details.get("reason")
                            accepted = False
                        minimal = policies.minimal_check(structure, sampled_roles or (), support)
                        if config.policy_id != "MF_P0":
                            accepted = constructed and bool(minimal["valid"])
                            if not accepted and reason is None:
                                reason = ";".join(minimal["failure_reasons"])
                        generated_legacy_family = (census.classify_family(sampled_roles[2:])[2]
                                                   if sampled_roles else None)
                        legacy_annotation = legacy_rescore(structure, sampled_roles or (),
                                                           generated_legacy_family, vocab)
                        source_composition = Counter(source)
                        sampled_exact = sampled_roles is not None and Counter(sampled_roles) == source_composition
                        scale_false, scale_true = (match_structure(reference, structure)
                                                   if accepted and sampled_exact else (False, False))
                        decoded_family = (census.classify_family(sampled_roles[2:])[0]
                                          if sampled_roles is not None else "NOT_SAMPLED")
                        anion_seen, composition_seen = sampled_training_presence(sampled_roles, training)
                        effective_diagnostics = (distribution_diagnostics if config.policy_id == "MF_P4"
                                                 else distribution_diagnostics[:3])
                        record = {
                            "policy_schema_version": policies.POLICY_SCHEMA_VERSION,
                            "population": population,
                            "material_id": material_id,
                            "composition_group": item["row"]["composition_group"],
                            "source_family": roles_info["family"],
                            "source_anion_key": roles_info["anion_key"],
                            "source_roles": source,
                            "parser_ab_aligned": roles_info["parser_ab_aligned"],
                            "latent_arm": arm,
                            "policy": config.policy_id,
                            "anion_setting": config.anion_setting,
                            "declared_domain": declared,
                            "source_in_policy_scope": source_scope["eligible"],
                            "source_scope_reasons": source_scope["reasons"],
                            "seed": replicate,
                            "attempt": attempt + 1,
                            "draw_seed": draw_seed,
                            "sampled": sampled,
                            "sampled_roles": sampled_roles,
                            "sampled_family": decoded_family,
                            "sampled_anion_key": census.anion_key(sampled_roles[2:]) if sampled_roles else None,
                            "sampled_full_composition": counter_key(sampled_roles) if sampled_roles else None,
                            "sampled_anion_multiset_seen_in_train": anion_seen,
                            "sampled_full_composition_seen_in_train": composition_seen,
                            "sampled_source_composition_exact": sampled_exact,
                            "sampled_source_A_B_exact": sampled_roles[:2] == source[:2] if sampled_roles else False,
                            "sampled_source_X_multiset_exact": Counter(sampled_roles[2:]) == Counter(source[2:]) if sampled_roles else False,
                            "selected_role_probabilities": selected_probabilities,
                            "numeric_underflow_support_count": sum(
                                diagnostic.get("numeric_underflow", 0)
                                for diagnostic in effective_diagnostics),
                            "masked_minus_infinity_support_count": sum(
                                diagnostic.get("masked_minus_infinity", 0)
                                for diagnostic in effective_diagnostics),
                            "constructed": constructed,
                            "minimal_valid": bool(minimal["valid"]),
                            "minimum_periodic_distance_angstrom": minimal["minimum_periodic_distance_angstrom"],
                            "minimal_failure_reasons": minimal["failure_reasons"],
                            "policy_accepted": accepted,
                            "rejection_reason": reason,
                            "raw_lengths_angstrom": previous.lattice_parameters(lat_out[0])[:3],
                            "raw_angles_degrees": previous.lattice_parameters(lat_out[0])[3:],
                            "projected_cubic_length_angstrom": cell_details.get("projected_cubic_length_angstrom"),
                            "cell_source": cell_details.get("cell_source"),
                            "source_structure_match_scale_false": scale_false,
                            "source_structure_match_scale_true": scale_true,
                            "legacy_rescore_status": legacy_annotation["status"],
                            "legacy_rescore_first_failure": legacy_annotation["first_failure"],
                            "legacy_tolerance_mu_status": legacy_annotation.get("tolerance_mu_status"),
                        }
                        ledger.append(record)
                        if accepted and structure is not None:
                            accepted_structures[len(ledger) - 1] = structure
    expected = sum(len(policy_variants(selected[mid]["roles"])) for mid in stochastic_ids)
    expected *= len(LATENT_ARMS) * len(SEEDS) * ATTEMPTS
    if len(ledger) != expected:
        raise AssertionError(f"Incomplete fixed attempt budget: {len(ledger)}/{expected}")
    return ledger, accepted_structures


def structural_cluster_count(indices: list[int], ledger: list[dict],
                             structures: dict[int, Structure]) -> int:
    """Composition-bucketed species-aware structural clustering."""
    buckets: dict[str, list[Structure]] = defaultdict(list)
    matcher = previous.matcher(False)
    clusters = 0
    for index in indices:
        candidate = structures.get(index)
        if candidate is None:
            continue
        key = ledger[index]["sampled_full_composition"]
        representatives = buckets[key]
        if not any(matcher.fit(reference, candidate) for reference in representatives):
            representatives.append(candidate)
            clusters += 1
    return clusters


def stochastic_summary(ledger: list[dict], structures: dict[int, Structure],
                       deterministic: list[dict]) -> list[dict]:
    raw_lookup = {(row["material_id"], row["latent_arm"]): row for row in deterministic}
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(ledger):
        key = (row["source_family"], row["policy"], row["anion_setting"], row["latent_arm"], row["seed"])
        groups[key].append(index)
    summary = []
    for (family, policy_id, setting, arm, seed), indices in sorted(groups.items()):
        subset = [ledger[index] for index in indices]
        material_ids = {row["material_id"] for row in subset}
        accepted = [index for index in indices if ledger[index]["policy_accepted"]]
        compositions = [ledger[index]["sampled_full_composition"] for index in accepted]
        recovered = {ledger[index]["material_id"] for index in accepted
                     if ledger[index]["source_structure_match_scale_false"]}
        eligible_ids = {row["material_id"] for row in subset if row["source_in_policy_scope"]}
        condition_concentration = []
        condition_unique = []
        for material_id in sorted(material_ids):
            counts = Counter(row["sampled_full_composition"] for row in subset
                             if row["material_id"] == material_id and row["policy_accepted"])
            if counts:
                condition_concentration.append(max(counts.values()) / sum(counts.values()))
                condition_unique.append(len(counts))
        failures = Counter(row["rejection_reason"] or "NONE" for row in subset if not row["policy_accepted"])
        result = {
            "population": subset[0]["population"],
            "source_family": family,
            "policy": policy_id,
            "anion_setting": setting,
            "latent_arm": arm,
            "seed": seed,
            "N_reference": len(material_ids),
            "N_source_scope_eligible": len(eligible_ids),
            "N_attempt": len(subset),
            "N_sampled": sum(row["sampled"] for row in subset),
            "N_constructed": sum(row["constructed"] for row in subset),
            "N_minimal_valid": sum(row["minimal_valid"] for row in subset),
            "N_policy_accepted": len(accepted),
            "N_accepted_anion_multiset_seen_in_train": sum(
                ledger[index]["sampled_anion_multiset_seen_in_train"] is True for index in accepted),
            "N_accepted_full_composition_seen_in_train": sum(
                ledger[index]["sampled_full_composition_seen_in_train"] is True for index in accepted),
            "N_unique_full_compositions": len(set(compositions)),
            "N_unique_structures": structural_cluster_count(accepted, ledger, structures),
            "N_raw_recovered": sum(raw_lookup[(mid, arm)]["raw_scale_false_match"] for mid in material_ids),
            "N_exported_recovered": sum(row["source_structure_match_scale_false"] for row in subset),
            "N_exported_recovered_eligible": sum(row["source_structure_match_scale_false"]
                                                  for row in subset if row["source_in_policy_scope"]),
            "source_recovered_per_eligible_draw": (
                sum(row["source_structure_match_scale_false"] for row in subset
                    if row["source_in_policy_scope"]) /
                sum(row["source_in_policy_scope"] for row in subset)
                if eligible_ids else None),
            "N_reference_recovered_within_budget": len(recovered),
            "N_reference_with_valid_output": len({ledger[index]["material_id"] for index in accepted}),
            "fraction_references_with_valid_output": (
                len({ledger[index]["material_id"] for index in accepted}) / len(material_ids)
                if material_ids else None),
            "N_sampled_source_composition": sum(row["sampled_source_composition_exact"] for row in subset),
            "accepted_per_attempt": len(accepted) / len(subset) if subset else None,
            "unique_compositions_per_attempt": len(set(compositions)) / len(subset) if subset else None,
            "recovered_references_per_eligible": len(recovered) / len(eligible_ids) if eligible_ids else None,
            "within_condition_max_composition_share_mean": (float(np.mean(condition_concentration))
                                                           if condition_concentration else None),
            "within_condition_unique_composition_mean": (float(np.mean(condition_unique))
                                                        if condition_unique else None),
            "failure_counts": dict(sorted(failures.items())),
        }
        summary.append(result)
    return summary


def family_confusion(deterministic: list[dict]) -> list[dict]:
    counts: Counter[tuple] = Counter()
    for row in deterministic:
        if "broad" not in row["panel_membership"]:
            continue
        for policy_id in ("MF_P3", "MF_P4"):
            counts[(row["latent_arm"], policy_id, row["source_family"],
                    row[f"{policy_id.lower()}_learned_family"])] += 1
    return [{"latent_arm": arm, "policy": policy_id, "source_family": source,
             "learned_output_family": predicted, "records": count}
            for (arm, policy_id, source, predicted), count in sorted(counts.items())]


def representative_cifs(ledger: list[dict], structures: dict[int, Structure],
                        report_dir: Path) -> list[dict]:
    from pymatgen.io.cif import CifWriter

    choices = []
    criteria = (
        lambda row: row["source_structure_match_scale_false"],
        lambda row: row["policy_accepted"] and row["source_family"] in
                    {"oxynitride", "oxyhalide", "oxychalcogenide", "other_mixed"},
        lambda row: row["policy_accepted"] and row["source_family"] == "oxide",
        lambda row: row["policy_accepted"] and row["source_family"] == "nitride",
    )
    used = set()
    for criterion in criteria:
        for index, row in enumerate(ledger):
            if index in structures and criterion(row) and row["material_id"] not in used:
                choices.append(index)
                used.add(row["material_id"])
                break
    for index, row in enumerate(ledger):
        if len(choices) >= 8:
            break
        if index in structures and index not in choices and row["material_id"] not in used:
            choices.append(index)
            used.add(row["material_id"])
    output = report_dir / "representative_cifs"
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for index in choices:
        row = ledger[index]
        filename = f"{row['material_id']}_{row['policy']}_{row['latent_arm']}_{row['seed']}_{row['attempt']}.cif"
        path = output / filename
        path.write_text(str(CifWriter(structures[index], symprec=None)), encoding="utf-8")
        inventory.append({"ledger_row": index, "material_id": row["material_id"],
                          "source_family": row["source_family"], "sampled_family": row["sampled_family"],
                          "policy": row["policy"], "relative_path": str(path.relative_to(report_dir)),
                          "sha256": file_hash(path)})
    return inventory


def deterministic_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        for membership in row["panel_membership"]:
            groups[(membership, row["source_family"], row["latent_arm"])].append(row)
    output = []
    for (membership, family, arm), subset in sorted(groups.items()):
        result = {
            "panel": membership, "source_family": family, "latent_arm": arm,
            "N_reference": len(subset),
            "N_composition_groups": len({row["composition_group"] for row in subset}),
            "N_parser_A_B_aligned": sum(row["parser_ab_aligned"] for row in subset),
            "N_parser_A_top1": sum(row["parser_A_top1"] for row in subset),
            "N_parser_B_top1": sum(row["parser_B_top1"] for row in subset),
            "N_parser_X_multiset_top1": sum(row["parser_X_multiset_top1"] for row in subset),
            "N_parser_full_composition_top1": sum(row["parser_full_composition_top1"] for row in subset),
            "N_raw_scale_false_match": sum(row["raw_scale_false_match"] for row in subset),
            "N_raw_geometry_minimal_only": sum(row["raw_geometry_minimal_only"] for row in subset),
            "N_parser_AB_NLL_all118_finite": sum(row["parser_A_B_nll_all118"] is not None for row in subset),
            "N_parser_X_NLL_all118_finite": sum(row["parser_X_nll_all118"] is not None for row in subset),
            "N_parser_AB_role_vocab_finite": sum(row["parser_A_B_nll_role_vocab"] is not None for row in subset),
            "N_parser_X_role_vocab_finite": sum(row["parser_X_nll_role_vocab"] is not None for row in subset),
            "mean_parser_AB_NLL_all118": (float(np.mean([row["parser_A_B_nll_all118"] for row in subset
                                                        if row["parser_A_B_nll_all118"] is not None]))
                                           if any(row["parser_A_B_nll_all118"] is not None for row in subset) else None),
            "mean_parser_X_NLL_all118": (float(np.mean([row["parser_X_nll_all118"] for row in subset
                                                       if row["parser_X_nll_all118"] is not None]))
                                          if any(row["parser_X_nll_all118"] is not None for row in subset) else None),
            "mean_raw_lattice_MAE_angstrom": float(np.mean([row["raw_lattice_length_mae_angstrom"] for row in subset])),
        }
        for policy_id in ("MF_P3", "MF_P4"):
            prefix = policy_id.lower()
            result[f"N_{policy_id}_learned_full_composition_top1"] = sum(
                row[f"{prefix}_learned_full_composition_exact"] for row in subset)
            result[f"N_{policy_id}_learned_X_multiset_top1"] = sum(
                row[f"{prefix}_learned_X_multiset_exact"] for row in subset)
            result[f"N_{policy_id}_learned_template_scale_false_match"] = sum(
                row[f"{prefix}_learned_scale_false_match"] for row in subset)
            result[f"N_{policy_id}_learned_minimal_valid"] = sum(
                row[f"{prefix}_learned_minimal_valid"] for row in subset)
            result[f"N_{policy_id}_learned_legacy_rescore_pass"] = sum(
                row[f"{prefix}_learned_legacy_rescore_status"] == "PASS" for row in subset)
        output.append(result)
    return output


def shared_declared_domain_table(ledger: list[dict], selected: dict[str, dict]) -> list[dict]:
    policies_needed = {"MF_P0", "MF_P1", "MF_P2", "MF_P3", "MF_P4"}
    ids = {row["material_id"] for row in ledger}
    for policy_id in policies_needed:
        ids &= {row["material_id"] for row in ledger
                if row["policy"] == policy_id and row["anion_setting"] == "DECLARED_DOMAIN"
                and row["source_in_policy_scope"]}
    table = []
    for family in ("ALL", "oxide", "nitride", "fluoride", "sulfide"):
        family_ids = ids if family == "ALL" else {
            mid for mid in ids if selected[mid]["roles"]["family"] == family
        }
        for policy_id in sorted(policies_needed):
            subset = [row for row in ledger if row["material_id"] in family_ids
                      and row["latent_arm"] == "joint" and row["policy"] == policy_id
                      and row["anion_setting"] == "DECLARED_DOMAIN"]
            table.append({"source_family": family, "policy": policy_id,
                          "anion_setting": "DECLARED_DOMAIN",
                          "shared_reference_count": len(family_ids), "N_attempt": len(subset),
                          "N_accepted": sum(row["policy_accepted"] for row in subset),
                          "N_source_structure_matches": sum(row["source_structure_match_scale_false"] for row in subset),
                          "N_recovered_references": len({row["material_id"] for row in subset
                                                         if row["source_structure_match_scale_false"]})})
    return table


def update_family_inference_counts(path: Path, selected: dict[str, dict],
                                   broad_ids: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Full CMR census has not written {path}")
    counts = Counter(selected[mid]["roles"]["family"] for mid in broad_ids)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["inference_count"] = counts[row["family"]]
        if int(row["val_count"]) == 0:
            row["assessment_status"] = "NOT_ASSESSABLE_NO_MATCHING_DATA"
        elif row["inference_count"] == 0:
            row["assessment_status"] = "NOT_ASSESSABLE_NOT_SELECTED"
        else:
            row["assessment_status"] = "ASSESSED_VALIDATION_DIAGNOSTIC"
    write_csv(path, rows)
    census_manifest_path = path.with_name("census_manifest.json")
    census_manifest = json.loads(census_manifest_path.read_text(encoding="utf-8"))
    census_manifest["output_sha256"]["family_support.csv"] = file_hash(path)
    addition = " The inference_count column was completed by the validation-only audit."
    if addition.strip() not in census_manifest["note"]:
        census_manifest["note"] += addition
    write_json(census_manifest_path, census_manifest)


def report_text(summary: dict) -> str:
    d = summary["deterministic_broad_joint"]
    s = summary["stochastic_joint"]
    census_all = summary["census_all"]
    lines = [
        "# Multifamily rule relaxation and decoder recovery",
        "",
        f"Policy schema: `{policies.POLICY_SCHEMA_VERSION}`. This informed follow-up used the frozen",
        "checkpoint, existing CMR splits, and the previously successful universal decoder input.",
        "The protocol was drafted before multifamily inference, clarified after an",
        "8-record implementation smoke run, and fixed before the full run. The earlier",
        "decoder audit had already been inspected.",
        "",
        "## Full CMR source conformity",
        "",
        f"The retained paired export contains **{census_all['records']:,}** source records: "
        f"{summary['split_counts']['train']:,} train, {summary['split_counts']['val']:,} validation, "
        f"and {summary['split_counts']['test']:,} final-test records. No final-test model inference ran.",
        f"All {census_all['source_minimal_valid_count']:,} source structures pass the frozen minimal",
        "site/cell/periodic-overlap checks; this is source conformity, not neural reconstruction.",
        "",
        "| Source policy scope | Retained records | Meaning |",
        "|---|---:|---|",
        f"| MF_P0 projected legacy oracle | {census_all['MF_P0_legacy_oracle_exportable_count']:,} | Current legacy source-species template passes all old filters |",
        f"| MF_P1/P2 legacy roles after veto removal | {census_all['MF_P1_source_eligible_count']:,} | Homogeneous source roles in old pools and source minimally valid |",
        f"| MF_P3 training-role single X | {census_all['MF_P3_source_eligible_count']:,} | Homogeneous X with train-observed A/B/X roles |",
        f"| MF_P4 training-role mixed-capable | {census_all['MF_P4_source_eligible_count']:,} | Five source roles in train-observed global role vocabularies |",
        "",
        "These are separately defined catalogue coverage counts, not accepted neural outputs",
        "or stability claims. Intersections and transition flags are counted in the census,",
        f"with {census_all['newly_eligible_after_legacy_veto_removal_count']:,} newly source-eligible",
        f"after legacy veto removal, {census_all['newly_eligible_after_global_vocabulary_count']:,}",
        f"after global role support, and {census_all['newly_eligible_after_mixed_X_count']:,}",
        "after heterogeneous X sites are allowed. These gains use the census's declared",
        "fixed-order waterfall and do not measure decoder recovery.",
        "The per-entry flags, overlapping reasons, fixed-order waterfall, exact anion inventory,",
        "and all split/family denominators are in the accompanying census CSVs.",
        "",
        "## Validation-only learned decoder",
        "",
        f"The deterministic panel has {summary['panel']['panel']['selected_rows']} records in "
        f"{summary['panel']['panel']['selected_composition_groups']} composition groups; the fixed-budget",
        f"proposal panel has {summary['panel']['stochastic_panel']['selected_rows']} records.",
        "The panel deliberately overrepresents rare positive-gap strata. The earlier 20 examples",
        "remain a separate reproducibility panel. The model sees the same universal coordinate",
        "and center tensors for every family; source/parser permutations are used offline for scoring.",
        "",
        "| Family | N | A top-1 | B top-1 | Parser X multiset | P4 learned X multiset | Raw composition | Raw structure match | P4 learned composition | P4 projected match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d:
        lines.append(
            f"| {row['source_family']} | {row['N_reference']} | {row['N_parser_A_top1']} | "
            f"{row['N_parser_B_top1']} | {row['N_parser_X_multiset_top1']} | "
            f"{row['N_MF_P4_learned_X_multiset_top1']} | "
            f"{row['N_parser_full_composition_top1']} | {row['N_raw_scale_false_match']} | "
            f"{row['N_MF_P4_learned_full_composition_top1']} | "
            f"{row['N_MF_P4_learned_template_scale_false_match']} |"
        )
    lines += [
        "",
        "A/B and X top-1 in this table use the verified source-to-parser mapping only for",
        "offline scoring. MF_P3/P4 generation always interprets fixed logits slots 0/1/2-4 as",
        "A/B/X1-X3; no per-reference permutation reaches the decoder or sampler. The raw",
        "structure metric uses the learned coordinates and full six-parameter raw cell.",
        "The projected metric uses decoded species and the learned cubic cell on the fixed",
        "five-site output template. `scale=False` requires species and cell agreement.",
        "",
        "## Fixed-budget output policies, joint latent",
        "",
        "Every applicable policy consumed 12 one-attempt draws for each of three seeds,",
        "with no early stopping. P1/P2 replay the same sampled roles. P0 uses the unchanged",
        "legacy branch at the prior audit's temperature 1.25, top-k 12, zero target-X-prior",
        "strength, and initially empty anti-repeat history. Therefore the P0-to-P1 package",
        "comparison does not identify a target-X-prior effect. P3/P4 learned-anion rows",
        "use one global training X vocabulary.",
        "",
        "| Source family | Policy | Anion setting | References eligible/attempted | Attempts | Accepted | Unique compositions | Eligible exported source matches |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in s:
        lines.append(
            f"| {row['source_family']} | {row['policy']} | {row['anion_setting']} | "
            f"{row['N_source_scope_eligible']}/{row['N_reference']} | {row['N_attempt']} | "
            f"{row['N_policy_accepted']} | {row['N_unique_full_compositions']} | "
            f"{row['N_exported_recovered_eligible'] if row['N_source_scope_eligible'] else 'N/A'} |"
        )
    joint_p4 = {row["source_family"]: row for row in s
                if row["policy"] == "MF_P4" and row["anion_setting"] == "LEARNED_ANION"}
    fixed_p4 = {row["source_family"]: row for row in summary["stochastic_all_seeds"]
                if row["policy"] == "MF_P4" and row["anion_setting"] == "LEARNED_ANION"
                and row["latent_arm"] == "fixed_real"}
    fixed_raw = {row["source_family"]: row for row in summary["deterministic_broad_fixed"]}
    lines += [
        "",
        "## Family conclusions on available paired data",
        "",
        "Joint and fixed-real columns expose latent dependence without treating the donor",
        "as a condition-matched control. Final recovery counts unique source references",
        "with at least one accepted species-aware `scale=False` match in the fixed budget.",
        "",
        "| Source family | Full CMR | Broad validation | Raw composition joint/fixed | Raw match joint | P4 final recovered joint/fixed (of stochastic N) | P4 accepted joint attempts | P4 generated legacy-rescore pass | Physical validation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in d:
        family = row["source_family"]
        fixed = fixed_raw[family]
        joint_final = joint_p4.get(family)
        fixed_final = fixed_p4.get(family)
        lines.append(
            f"| {family} | {summary['family_source_counts'][family]} | {row['N_reference']} | "
            f"{row['N_parser_full_composition_top1']}/{fixed['N_parser_full_composition_top1']} | "
            f"{row['N_raw_scale_false_match']} | "
            f"{joint_final['N_reference_recovered_within_budget'] if joint_final else 'N/A'}/"
            f"{fixed_final['N_reference_recovered_within_budget'] if fixed_final else 'N/A'} "
            f"(of {joint_final['N_reference'] if joint_final else 'N/A'}) | "
            f"{joint_final['N_policy_accepted'] if joint_final else 'N/A'}/"
            f"{joint_final['N_attempt'] if joint_final else 'N/A'} | "
            f"{summary['legacy_p4_joint_pass'].get(family, 0)} | not established |"
        )
    supported = [row for row in d if row["N_reference"] > 0]
    macro_raw = np.mean([row["N_raw_scale_false_match"] / row["N_reference"] for row in supported])
    lines += [
        "",
        "The P4 legacy rescore counts generated outputs that happened to meet the old",
        "homogeneous-family filters, even when the source family is mixed. It does not",
        "apply legacy t/mu chemistry to mixed-anion structures. The detailed deterministic",
        "table also records projected learned-cubic length and volume errors separately",
        "from raw six-parameter-cell errors and structure matches.",
        "",
        "Among accepted P4 learned-anion joint attempts in the stochastic panel,",
        f"{sum(row['N_accepted_anion_multiset_seen_in_train'] for row in joint_p4.values())}/"
        f"{sum(row['N_policy_accepted'] for row in joint_p4.values())} anion multisets and",
        f"{sum(row['N_accepted_full_composition_seen_in_train'] for row in joint_p4.values())}/"
        f"{sum(row['N_policy_accepted'] for row in joint_p4.values())} full compositions",
        "occurred in the training split. Absence from training is only a support/extrapolation",
        "flag, not verified chemical novelty. Both flags are recorded on every sampled attempt",
        "and aggregated by family, policy, latent arm, and seed.",
        "",
        f"The unweighted macro raw-match fraction across the {len(supported)} supported,",
        f"nonempty source families is {macro_raw:.4f}; included families are "
        + ", ".join(row["source_family"] for row in supported) + ".",
        "It is a diagnostic-panel average, not a full-CMR population estimate.",
        "",
        "The paired data and observed joint-latent raw/final matches support a small",
        "exploratory MF_P4 decoder-interface pilot for the six represented source strata:",
        "oxide, nitride, oxynitride, oxyhalide, oxychalcogenide, and mixed O/F/N.",
        "This is a scope recommendation for a later matched-budget latent-sampler study,",
        "not a claim that property-conditioned diffusion or target attainment works.",
        "Homogeneous fluoride/sulfide, selenide, telluride, iodide, mixed halide, and",
        "mixed chalcogenide paired families are absent locally and remain",
        "`NOT_ASSESSABLE_NO_MATCHING_DATA`.",
        "",
        "## Paired effects and scope attribution",
        "",
        "Joint-minus-fixed-real effects resample whole reduced-composition groups.",
        "",
    ]
    for name, effect in summary["paired_whole_group_effects"].items():
        lines.append(f"- `{name}`: mean {effect['mean_joint_minus_fixed']}, "
                     f"95% interval {effect['bootstrap_95']}, {effect['N_groups']} groups.")
    lines += [
        "",
        f"The shared old-role stochastic comparison has {summary['shared_legacy_scope'][0]['shared_reference_count']}",
        "source references. Its per-policy accepted and recovered counts are in",
        "`shared_legacy_scope_comparison.csv`; zero shared references make a policy effect",
        "unassessable, not negative. The earlier 20-example panel was evaluated",
        "separately after this zero-overlap finding; that extension is post-run and",
        "is never pooled into the balanced panel's estimates.",
        "",
        "| Earlier 20 panel family | Policy | Shared eligible sources | Attempts | Accepted | Recovered sources |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["old_20_shared_scope"]:
        if row["source_family"] in ("ALL", "oxide", "nitride"):
            lines.append(f"| {row['source_family']} | {row['policy']} | "
                         f"{row['shared_reference_count']} | {row['N_attempt']} | "
                         f"{row['N_accepted']} | {row['N_recovered_references']} |")
    lines += [
        "",
        "The old-panel one-attempt ledger, paired policy contrasts, replay assertions,",
        "and per-family/arm/seed statistics are stored separately. P1/P2 replay is",
        "exact and every policy condition has twelve draws in both panels.",
    ]
    lines += [
        "",
        "P0/P1/P2 and expanded declared-domain panels apply only to homogeneous source",
        "families and are reported separately from unrestricted learned-anion P3/P4.",
        "P3 proposals for mixed sources are exploratory outputs outside P3's source-recovery",
        "scope; their zero source recovery is undefined as a decoder metric.",
        "An X identity forced by a singleton declared-domain set receives no learned-X credit.",
        "The complete ledger preserves duplicates, sampled/constructed/minimal/accepted stages,",
        "failure reasons, emitted full composition, and original source correspondence.",
        "",
        "## Interpretation and limits",
        "",
        "The training-observed global role vocabularies and exact family support are in",
        "`training_support.json` and `family_support.csv`. A family with no matching local",
        "paired references is marked `NOT_ASSESSABLE_NO_MATCHING_DATA`, not zero recovery.",
        "Postprocessing acceptance only establishes compliance with the declared minimal",
        "rules. It does not establish charge stability, target properties, physical stability,",
        "novelty, SUN, synthesizability, or a successful paper-result reproduction.",
        "CMR labels remain provisional checkpoint inputs, and pretrained training overlap",
        "remains unknown. The fixed-real latent is a deliberately condition-mismatched control.",
        "A source direct gap of zero is retained as a reported CMR label and is not",
        "reclassified here as a confirmed metallic state.",
        "Repeated draws and seeds are not independent materials; the balanced panel does",
        "not estimate full-CMR population performance without weighting.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONDONTWRITEBYTECODE = '1'",
        "$env:PYTHONHASHSEED = '0'",
        "$env:MPLCONFIGDIR = Join-Path $env:TEMP 'meidnet-multifamily'",
        "& .\\.venv\\Scripts\\python.exe .\\scripts\\multifamily_census.py",
        "& .\\.venv\\Scripts\\python.exe .\\scripts\\audit_multifamily_decoder.py",
        "& .\\.venv\\Scripts\\python.exe -m pytest -q",
        "```",
        "",
        "The executed commands, revision/dirty state, split/checkpoint/input hashes, panel",
        "selection, policy settings, legacy parity checks, and output hashes are in",
        "`run_manifest.json` and `summary.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = arguments()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise EnvironmentError("Set PYTHONHASHSEED=0 for exact legacy set-order parity")
    if args.seed != 42 or args.max_panel > 384 or args.max_stochastic > 64:
        raise ValueError("This frozen protocol requires seed 42, <=384 broad, <=64 stochastic")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for path in (args.data_root / "train.csv", args.data_root / "val.csv",
                 args.data_root / "test.csv", args.checkpoint, INPUT_MANIFEST,
                 REPORT_ROOT / "protocol.md", REPORT_ROOT / "census_per_entry.csv"):
        if not path.exists():
            raise FileNotFoundError(path)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if report_dir != REPORT_ROOT.resolve():
        for name in ("census_per_entry.csv", "dataset_conformity_by_split_and_family.csv",
                     "exact_anion_inventory.csv", "training_support.json",
                     "family_support.csv", "census_manifest.json", "protocol.md"):
            shutil.copyfile(REPORT_ROOT / name, report_dir / name)
    protected = [
        args.checkpoint, INPUT_MANIFEST, OLD_AUDIT,
        *(args.data_root / f"{split}.csv" for split in ("train", "val", "test")),
        ROOT / "reports" / "template_reachability" / "summary.json",
        ROOT / "reports" / "template_reachability" / "report.md",
        ROOT / "reports" / "decoder_audit.md",
    ]
    protected_before = {str(path.relative_to(ROOT)): file_hash(path) for path in protected}
    rows_train = census.load_source_rows(args.data_root, ("train",))
    training = census.training_support(rows_train)
    vocab = {role: tuple(symbol for symbol in SPECIES_LIST if symbol in training[f"V_{role}"])
             for role in ("A", "B", "X")}
    validation = census.load_source_rows(args.data_root, ("val",))
    role_metadata = {row["material_id"]: census.source_roles(row) for row in validation}
    usable = [row | {"anion_key": role_metadata[row["material_id"]]["anion_key"]}
              for row in validation if role_metadata[row["material_id"]]["role_status"] == "VERIFIED"]
    panel_ids, stochastic_ids, selection = panel.select_validation_panels(
        usable, seed=args.seed, max_panel=args.max_panel,
        max_stochastic=args.max_stochastic,
    )
    old_frame = pd.read_csv(OLD_AUDIT, dtype={"material_id": "string"})
    old_ids = sorted(old_frame["material_id"].astype(str).tolist())
    validation_ids = {row["material_id"] for row in validation}
    if not set(old_ids) <= validation_ids:
        raise AssertionError("Old audit IDs are not all in validation")
    selected_ids = sorted(set(panel_ids) | set(old_ids))
    if not set(stochastic_ids) <= set(panel_ids) or not set(selected_ids) <= validation_ids:
        raise AssertionError("Panel selection crossed the validation boundary")
    selected = load_selected(args.data_root, selected_ids)
    for material_id, item in selected.items():
        item["panel_membership"] = tuple(
            label for label, ids in (("broad", panel_ids), ("old_20", old_ids))
            if material_id in ids
        )
    write_json(report_dir / "panel_manifest.json", selection | {
        "old_20_ids": old_ids,
        "old_20_overlap_with_broad": len(set(old_ids) & set(panel_ids)),
        "validation_ambiguous_role_exclusions": len(validation) - len(usable),
    })
    write_csv(report_dir / "old_20_panel.csv", [
        {"material_id": mid, "in_broad_panel": mid in panel_ids,
         "source_family": selected[mid]["roles"]["family"],
         "reproduces_prior_panel": True}
        for mid in old_ids
    ])
    write_csv(report_dir / "panel_membership.csv", [
        {"material_id": mid, "panel_membership": selected[mid]["panel_membership"],
         "composition_group": selected[mid]["row"]["composition_group"],
         "source_family": selected[mid]["roles"]["family"],
         "exact_anion_key": selected[mid]["roles"]["anion_key"],
         "source_direct_gap_eV": selected[mid]["row"]["dir_gap"],
         "parser_ab_aligned": selected[mid]["roles"]["parser_ab_aligned"]}
        for mid in selected_ids
    ])
    coordinates, center, template_manifest = template_input()
    checkpoint, checkpoint_metadata = previous.checkpoint_summary(args.checkpoint)
    model, model_metadata = previous.instantiate_model(checkpoint)
    generation_model, generation_metadata = previous.load_generation_model(checkpoint)
    model.eval().requires_grad_(False)
    generation_model.eval().requires_grad_(False)
    if model.training or generation_model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Checkpoint is not frozen in evaluation mode")
    donor_id, fixed_joint = donor_joint(model, args.data_root)
    cache = encode_and_decode(model, selected, coordinates, center, args.batch_size, fixed_joint)
    legacy_parity = verify_legacy_cache_parity(generation_model, selected, cache, coordinates, center)
    deterministic = deterministic_rows(selected, cache, vocab)
    census_entries = {
        row["material_id"]: row for row in csv.DictReader(
            (report_dir / "census_per_entry.csv").open(encoding="utf-8", newline="")
        )
    }
    for row in deterministic:
        source = census_entries[row["material_id"]]
        row["source_minimal_valid"] = source["source_minimal_valid"]
        row["source_legacy_oracle_status"] = source["legacy_oracle_status"]
        row["source_MF_P4_eligible"] = source["MF_P4_source_eligible"]
    write_csv(report_dir / "deterministic_per_reference.csv", deterministic)
    deterministic_groups = deterministic_summary(deterministic)
    write_csv(report_dir / "deterministic_by_family.csv", deterministic_groups)
    write_csv(report_dir / "family_confusion.csv", family_confusion(deterministic))
    scope_rows = []
    for material_id in selected_ids:
        roles = selected[material_id]["roles"]
        source = tuple(roles["source_symbols"])
        for config, declared in policy_variants(roles):
            scope = policies.source_scope(source, config, vocab, declared)
            scope_rows.append({"material_id": material_id,
                               "panel_membership": selected[material_id]["panel_membership"],
                               "source_family": roles["family"],
                               "policy": config.policy_id,
                               "anion_setting": config.anion_setting,
                               "source_scope_eligible": scope["eligible"],
                               "scope_reasons": scope["reasons"]})
    write_csv(report_dir / "policy_scope_panel.csv", scope_rows)
    ledger, accepted_structures = stochastic_rows(selected, cache, stochastic_ids, vocab, training)
    replay = assert_replay_and_budget(ledger)
    pair_comparisons = policy_pair_comparisons(ledger)
    write_csv(report_dir / "paired_policy_comparisons.csv", pair_comparisons)
    write_csv(report_dir / "attempts.csv", ledger)
    summaries = stochastic_summary(ledger, accepted_structures, deterministic)
    aggregate_ledger = [row | {"seed": "ALL"} for row in ledger]
    summaries.extend(stochastic_summary(aggregate_ledger, accepted_structures, deterministic))
    write_csv(report_dir / "stochastic_by_family_policy_arm_seed.csv", summaries)
    shared = shared_declared_domain_table(ledger, selected)
    write_csv(report_dir / "shared_legacy_scope_comparison.csv", shared)
    old_ledger, old_structures = stochastic_rows(
        selected, cache, old_ids, vocab, training,
        population="previous_20_reproducibility_panel"
    )
    old_replay = assert_replay_and_budget(old_ledger)
    old_pair_comparisons = policy_pair_comparisons(old_ledger)
    old_shared = shared_declared_domain_table(old_ledger, selected)
    write_csv(report_dir / "old_20_attempts.csv", old_ledger)
    old_summary = stochastic_summary(old_ledger, old_structures, deterministic)
    old_summary.extend(stochastic_summary(
        [row | {"seed": "ALL"} for row in old_ledger], old_structures, deterministic
    ))
    write_csv(report_dir / "old_20_stochastic_by_family_policy_arm_seed.csv", old_summary)
    write_csv(report_dir / "old_20_paired_policy_comparisons.csv", old_pair_comparisons)
    write_csv(report_dir / "old_20_shared_scope_comparison.csv", old_shared)
    uniqueness = Counter((row["source_family"], row["policy"], row["anion_setting"],
                          row["latent_arm"], row["sampled_full_composition"])
                         for row in ledger if row["policy_accepted"])
    write_csv(report_dir / "accepted_full_composition_counts.csv", [
        {"source_family": family, "policy": policy_id, "anion_setting": setting,
         "latent_arm": arm, "full_composition": composition, "accepted_attempts": count}
        for (family, policy_id, setting, arm, composition), count in sorted(uniqueness.items())
    ])
    failures = [row for row in ledger if not row["policy_accepted"]]
    chosen_failures = []
    for family in sorted({row["source_family"] for row in failures}):
        row = next((record for record in failures if record["source_family"] == family
                    and record["policy"] == "MF_P4"), None)
        if row:
            chosen_failures.append(row)
    write_csv(report_dir / "representative_failures.csv", chosen_failures)
    representative = representative_cifs(ledger, accepted_structures, report_dir)
    write_csv(report_dir / "representative_cifs.csv", representative)
    write_csv(report_dir / "matcher_failures.csv", MATCHER_FAILURES)
    update_family_inference_counts(report_dir / "family_support.csv", selected, panel_ids)
    census_aggregate = list(csv.DictReader(
        (report_dir / "dataset_conformity_by_split_and_family.csv").open(encoding="utf-8", newline="")
    ))
    census_all = next(row for row in census_aggregate if row["split"] == "total"
                      and row["stratum_type"] == "all")
    census_all = {key: int(value) if key.endswith("_count") or key == "records" else value
                  for key, value in census_all.items()}
    family_source_counts = {
        row["stratum"]: int(row["records"])
        for row in census_aggregate if row["split"] == "total" and row["stratum_type"] == "family"
    }
    final_recovery_rows = []
    for material_id in stochastic_ids:
        for arm in ("joint", "fixed_real"):
            recovered = any(row["source_structure_match_scale_false"]
                            for row in ledger if row["material_id"] == material_id
                            and row["latent_arm"] == arm and row["policy"] == "MF_P4"
                            and row["anion_setting"] == "LEARNED_ANION")
            final_recovery_rows.append({"material_id": material_id,
                                        "composition_group": selected[material_id]["row"]["composition_group"],
                                        "source_family": selected[material_id]["roles"]["family"],
                                        "latent_arm": arm, "panel_membership": ("broad",),
                                        "recovered": recovered})
    paired_effects = {
        "raw_P4_full_composition": paired_group_effect(
            deterministic, "mf_p4_learned_full_composition_exact"),
        "projected_P4_scale_false_match": paired_group_effect(
            deterministic, "mf_p4_learned_scale_false_match"),
        "fixed_budget_P4_reference_recovered": paired_group_effect(
            final_recovery_rows, "recovered"),
    }
    legacy_p4_joint_pass = Counter(
        row["source_family"] for row in ledger
        if row["latent_arm"] == "joint" and row["policy"] == "MF_P4"
        and row["anion_setting"] == "LEARNED_ANION" and row["policy_accepted"]
        and row["legacy_rescore_status"] == "PASS"
    )
    summary = {
        "policy_schema_version": policies.POLICY_SCHEMA_VERSION,
        "source_recovery_status": "CMR_RECONSTRUCTED_SPLITS_NOT_ORIGINAL",
        "checkpoint_label_compatibility": "UNRESOLVED_PROVISIONAL_NUMERICAL_INPUT",
        "pretrained_training_overlap": "UNKNOWN",
        "test_model_inference_count": 0,
        "split_counts": {"train": len(rows_train), "val": len(validation),
                         "test": len(census.load_source_rows(args.data_root, ("test",)))},
        "training_vocab_sizes": {role: len(vocab[role]) for role in vocab},
        "training_role_vocabularies": vocab,
        "panel": selection,
        "old_20_overlap_with_broad": len(set(old_ids) & set(panel_ids)),
        "old_20_panel_size": len(old_ids),
        "fixed_real_donor_id": donor_id,
        "census_all": census_all,
        "family_source_counts": family_source_counts,
        "deterministic_broad_joint": [row for row in deterministic_groups
                                      if row["panel"] == "broad" and row["latent_arm"] == "joint"],
        "deterministic_broad_fixed": [row for row in deterministic_groups
                                      if row["panel"] == "broad" and row["latent_arm"] == "fixed_real"],
        "stochastic_joint": [row for row in summaries
                             if row["seed"] == "ALL" and row["latent_arm"] == "joint"],
        "stochastic_all_seeds": [row for row in summaries if row["seed"] == "ALL"],
        "legacy_p4_joint_pass": dict(legacy_p4_joint_pass),
        "legacy_parity": legacy_parity,
        "proposal_replay_and_budget": replay,
        "shared_legacy_scope": shared,
        "old_20_shared_scope": old_shared,
        "old_20_paired_policy_comparisons": old_pair_comparisons,
        "old_20_replay_and_budget": old_replay,
        "old_20_joint": [row for row in old_summary
                         if row["seed"] == "ALL" and row["latent_arm"] == "joint"],
        "paired_policy_comparisons": pair_comparisons,
        "paired_whole_group_effects": paired_effects,
        "inference_reference_count": len(selected_ids),
        "forward_head_cache_entries": len(selected_ids) * len(LATENT_ARMS),
        "stochastic_attempt_count": len(ledger),
        "old_20_stochastic_attempt_count": len(old_ledger),
        "representative_cifs": representative,
        "matcher_failure_count": len(MATCHER_FAILURES),
    }
    (report_dir / "report.md").write_text(report_text(summary), encoding="utf-8")
    protected_after = {str(path.relative_to(ROOT)): file_hash(path) for path in protected}
    if protected_after != protected_before:
        raise AssertionError("Protected checkpoint, splits, template, or old reports changed")
    output_hashes = {
        str(path.relative_to(report_dir)): file_hash(path)
        for path in sorted(report_dir.rglob("*")) if path.is_file()
        and path.name not in {"summary.json", "run_manifest.json"}
    }
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actual_command": [sys.executable, *sys.argv],
        "git": current_git(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                        "device": "cpu"},
        "configuration": {"seed": args.seed, "max_panel": args.max_panel,
                          "max_stochastic": args.max_stochastic, "batch_size": args.batch_size,
                          "proposal_seeds": SEEDS, "attempts_per_seed": ATTEMPTS,
                          "temperature": TEMPERATURE},
        "protected_hashes_before_after_equal": protected_before == protected_after,
        "protected_hashes": protected_after,
        "input_hashes": {"raw_db": file_hash(args.data_root.parents[1] / "raw" / "cmr" / "cubic_perovskites.db"),
                         "checkpoint": file_hash(args.checkpoint),
                         "template_input_manifest": file_hash(INPUT_MANIFEST),
                         "policy_module": file_hash(ROOT / "scripts" / "multifamily_policies.py"),
                         "audit_module": file_hash(Path(__file__)),
                         "census_module": file_hash(ROOT / "scripts" / "multifamily_census.py"),
                         "panel_module": file_hash(ROOT / "scripts" / "multifamily_panel.py")},
        "model_loads": {"training": model_metadata, "generation": generation_metadata,
                        "checkpoint_sha256": checkpoint_metadata["sha256"]},
        "universal_input_hashes": {"coordinates": template_manifest["coordinates_sha256"],
                                   "center": template_manifest["center_sha256"]},
        "output_template_sha256": policies.OUTPUT_TEMPLATE_SHA256,
        "output_hashes": output_hashes,
        "no_final_test_model_inference": True,
    }
    write_json(report_dir / "run_manifest.json", manifest)
    summary["run_manifest_sha256"] = file_hash(report_dir / "run_manifest.json")
    summary["output_hashes"] = output_hashes
    write_json(report_dir / "summary.json", summary)
    print(json.dumps({"status": "passed", "broad_records": len(panel_ids),
                      "stochastic_records": len(stochastic_ids), "attempts": len(ledger),
                      "report": str(report_dir / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
