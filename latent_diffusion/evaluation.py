"""Small, asset-free helpers for the bounded CPU generation comparison."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import torch


def stable_seed(*parts: object) -> int:
    """Derive a repeatable seed without consuming a training or sampling stream."""
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def choose_training_targets(rows: list[dict], limit: int = 8) -> list[dict]:
    """Choose source-stratum zero gaps, then a positive gap, before generation.

    Only IDs and stored training properties decide membership. A repeated
    numerical property pair is never counted as a second target.
    """
    if limit < 1:
        raise ValueError("Target limit must be positive")
    ordered = sorted(rows, key=lambda row: row["material_id"])
    source_anion_keys = sorted({row["source_anion_key"] for row in ordered})
    selected: list[dict] = []
    used_property_pairs: set[tuple[float, float]] = set()

    def add_median_heat_candidate(candidates: list[dict]) -> bool:
        if not candidates:
            return False
        by_heat = sorted(candidates, key=lambda row: (float(row["heat_all"]), row["material_id"]))
        median_heat = float(by_heat[len(by_heat) // 2]["heat_all"])
        for row in sorted(by_heat, key=lambda item: (abs(float(item["heat_all"]) - median_heat),
                                                     item["material_id"])):
            pair = (float(row["heat_all"]), float(row["dir_gap"]))
            if pair not in used_property_pairs:
                selected.append(row)
                used_property_pairs.add(pair)
                return True
        return False

    for key in source_anion_keys:
        if len(selected) >= limit:
            break
        add_median_heat_candidate([row for row in ordered if row["source_anion_key"] == key
                                   and float(row["dir_gap"]) == 0.0])
    if len(selected) < limit:
        add_median_heat_candidate([row for row in ordered if float(row["dir_gap"]) > 0.0])
    for row in ordered:
        if len(selected) >= limit:
            break
        add_median_heat_candidate([row])
    return selected


def nearest_condition_indices(conditions: torch.Tensor, target: torch.Tensor,
                              std: torch.Tensor,
                              ids: list[str], k: int = 16) -> list[tuple[int, float]]:
    """Nearest cached TRAIN rows in train-standardized two-property space."""
    if conditions.ndim != 2 or conditions.shape[1] != 2 or target.shape != (2,):
        raise ValueError("Expected [N,2] conditions and a two-property target")
    if len(ids) != len(conditions) or k < 1 or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("Invalid IDs, k, or property scaler")
    distances = torch.linalg.vector_norm((conditions - target[None, :]) / std[None, :], dim=1)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("Nonfinite training-condition distance")
    return sorted(enumerate(distances.tolist()), key=lambda pair: (pair[1], ids[pair[0]]))[:k]


def full_composition_key(roles: tuple[str, ...] | list[str]) -> str:
    """Count sampled role symbols in a sorted composition key."""
    return "|".join(f"{element}:{count}" for element, count in sorted(Counter(roles).items()))


def summarize_attempts(rows: list[dict]) -> list[dict]:
    """Summarize the complete attempt ledger by requested target and method."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["target_id"], row["method"]), []).append(row)
    result = []
    for (target_id, method), attempts in sorted(groups.items()):
        accepted = [row for row in attempts if row["accepted"]]
        compositions = [row["full_composition"] for row in accepted]
        result.append({
            "target_id": target_id,
            "method": method,
            "attempts": len(attempts),
            "accepted": len(accepted),
            "unique_accepted_compositions": len(set(compositions)),
            "largest_composition_share": (max(Counter(compositions).values()) / len(compositions)
                                          if compositions else None),
            "rejection_reasons": dict(Counter(row["rejection_reason"] or "NONE"
                                              for row in attempts if not row["accepted"])),
        })
    return result
