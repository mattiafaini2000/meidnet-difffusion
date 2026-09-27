"""Deterministic, composition-group-aware validation panels for the multifamily audit.

Selection uses only source metadata and label availability. It does not inspect
decoder outputs, legacy acceptance, or test records.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _stratum(row: Mapping[str, Any]) -> str:
    anion_key = str(row.get("anion_key") or row.get("source_anion") or "").strip()
    if not anion_key:
        raise ValueError(f"Missing exact anion key for {row.get('material_id')}")
    gap = float(row["dir_gap"])
    if not math.isfinite(gap) or gap < 0:
        raise ValueError(f"Invalid direct-gap label for {row.get('material_id')}: {gap}")
    return f"{anion_key}|{'zero' if gap == 0 else 'positive'}"


def _prepare(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    prepared = []
    seen = set()
    for row in rows:
        material_id = str(row["material_id"])
        group = str(row["composition_group"])
        if not material_id or not group or material_id in seen:
            raise ValueError(f"Missing or repeated material ID/composition group: {material_id}")
        if "split" in row and str(row["split"]).lower() not in {"val", "validation"}:
            raise ValueError(f"Panel received a non-validation record: {material_id}")
        seen.add(material_id)
        prepared.append({"material_id": material_id, "group": group, "stratum": _stratum(row)})
    return sorted(prepared, key=lambda row: row["material_id"])


def _balanced_group_sample(
    rows: list[dict[str, str]], budget: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    population = Counter(row["stratum"] for row in rows)
    for row in rows:
        groups[row["group"]].append(row)
    for group, members in groups.items():
        for stratum in {member["stratum"] for member in members}:
            by_stratum[stratum].append(group)

    def group_order(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()

    for stratum in by_stratum:
        by_stratum[stratum].sort(key=lambda group: (group_order(group), group))

    selected_groups: set[str] = set()
    selected_counts: Counter[str] = Counter()
    selected_size = 0

    def add(group: str) -> None:
        nonlocal selected_size
        selected_groups.add(group)
        selected_size += len(groups[group])
        selected_counts.update(member["stratum"] for member in groups[group])

    # A stratum with no more rows than its equal-share quota is included first.
    sparse_cutoff = budget / len(population) if population else 0.0
    sparse = {stratum for stratum, count in population.items() if count <= sparse_cutoff}
    sparse_groups = {
        group for stratum in sparse for group in by_stratum[stratum]
    }
    for group in sorted(
        sparse_groups,
        key=lambda group: (
            min(population[row["stratum"]] for row in groups[group] if row["stratum"] in sparse),
            group_order(group),
            group,
        ),
    ):
        if selected_size + len(groups[group]) <= budget:
            add(group)

    # Round-robin over the least represented available stratum. A selected
    # composition group contributes all its records, including other strata.
    while selected_size < budget:
        chosen = None
        for stratum in sorted(population, key=lambda key: (selected_counts[key], key)):
            chosen = next(
                (
                    group
                    for group in by_stratum[stratum]
                    if group not in selected_groups
                    and selected_size + len(groups[group]) <= budget),
                None,
            )
            if chosen is not None:
                break
        if chosen is None:
            break
        add(chosen)

    selected = [row for row in rows if row["group"] in selected_groups]
    manifest = {
        "requested_maximum": budget,
        "selected_rows": len(selected),
        "selected_composition_groups": len(selected_groups),
        "sparse_strata": sorted(sparse),
        "sparse_strata_all_included": all(
            selected_counts[stratum] == population[stratum] for stratum in sparse
        ),
        "unfilled_capacity": budget - len(selected),
    }
    return selected, manifest


def select_validation_panels(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    max_panel: int = 384,
    max_stochastic: int = 64,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return sorted broad/stochastic IDs and their source-only selection manifest.

    ``rows`` is the validation population, with material_id, composition_group,
    dir_gap, and anion_key (or source_anion). Any supplied split must be val.
    Groups are indivisible, so a maximum may be slightly underfilled.
    """
    if max_panel <= 0 or max_stochastic <= 0:
        raise ValueError("Panel maxima must be positive")
    prepared = _prepare(rows)
    broad, broad_info = _balanced_group_sample(prepared, min(max_panel, len(prepared)), seed)
    stochastic, stochastic_info = _balanced_group_sample(
        broad, min(max_stochastic, len(broad)), seed + 1
    )
    population_counts = Counter(row["stratum"] for row in prepared)
    broad_counts = Counter(row["stratum"] for row in broad)
    stochastic_counts = Counter(row["stratum"] for row in stochastic)
    strata = {
        stratum: {
            "population": population_counts[stratum],
            "panel": broad_counts[stratum],
            "stochastic": stochastic_counts[stratum],
            "population_fraction": population_counts[stratum] / len(prepared),
            "panel_fraction": broad_counts[stratum] / len(broad) if broad else None,
            "panel_representation_ratio": (
                (broad_counts[stratum] / len(broad))
                / (population_counts[stratum] / len(prepared))
                if broad
                else None
            ),
            "panel_selection_fraction": broad_counts[stratum] / population_counts[stratum],
            "stochastic_selection_fraction": (
                stochastic_counts[stratum] / broad_counts[stratum]
                if broad_counts[stratum]
                else None
            ),
        }
        for stratum in sorted(population_counts)
    }
    panel_ids = [row["material_id"] for row in broad]
    stochastic_ids = [row["material_id"] for row in stochastic]
    manifest = {
        "selection_schema_version": "multifamily_panel_v1",
        "seed": seed,
        "max_panel": max_panel,
        "max_stochastic": max_stochastic,
        "population_rows": len(prepared),
        "population_composition_groups": len({row["group"] for row in prepared}),
        "panel": broad_info,
        "stochastic_panel": stochastic_info,
        "strata": strata,
        "panel_ids": panel_ids,
        "stochastic_ids": stochastic_ids,
        "sampling_bias": (
            "The two panels balance exact-anion and zero/positive-gap strata, with complete "
            "composition groups. Their unweighted rates do not estimate validation-population rates."
        ),
    }
    return panel_ids, stochastic_ids, manifest
