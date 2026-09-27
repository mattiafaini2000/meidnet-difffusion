"""Paired comparisons and fixed-draw accounting for multifamily proposals.

Each result keeps the existing material, seed, attempt, and composition-group
denominators. No model inference or file access occurs on import.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

LATENT_ARMS = ("joint", "fixed_real", "property")
ATTEMPTS = 12


def paired_group_effect(rows: list[dict], metric: str,
                        family: str | None = None) -> dict:
    """Whole-composition-group bootstrap of joint minus fixed-real rates."""
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if "broad" not in row["panel_membership"] or (family and row["source_family"] != family):
            continue
        if row["latent_arm"] in ("joint", "fixed_real"):
            pairs[row["material_id"]][row["latent_arm"]] = row
    group_differences: dict[str, list[float]] = defaultdict(list)
    for arms in pairs.values():
        if set(arms) == {"joint", "fixed_real"}:
            group = arms["joint"]["composition_group"]
            group_differences[group].append(
                float(arms["joint"][metric]) - float(arms["fixed_real"][metric])
            )
    values = np.asarray([np.mean(differences) for differences in group_differences.values()], dtype=float)
    if not len(values):
        return {"N_groups": 0, "mean_joint_minus_fixed": None, "bootstrap_95": None}
    rng = np.random.default_rng(20260924)
    samples = rng.choice(values, size=(3000, len(values)), replace=True).mean(axis=1)
    return {"N_groups": len(values), "mean_joint_minus_fixed": float(values.mean()),
            "bootstrap_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])]}


def assert_replay_and_budget(ledger: list[dict]) -> dict:
    proposal_pairs: dict[tuple, dict[str, tuple | None]] = defaultdict(dict)
    budgets: Counter[tuple] = Counter()
    for row in ledger:
        key = (row["material_id"], row["latent_arm"], row["seed"], row["attempt"])
        if row["policy"] in ("MF_P1", "MF_P2"):
            proposal_pairs[key][row["policy"]] = row["sampled_roles"]
        budget_key = (row["material_id"], row["latent_arm"], row["policy"],
                      row["anion_setting"], row["seed"])
        budgets[budget_key] += 1
    mismatched = [key for key, pair in proposal_pairs.items()
                  if set(pair) != {"MF_P1", "MF_P2"} or pair["MF_P1"] != pair["MF_P2"]]
    incorrect_budgets = [key for key, count in budgets.items() if count != ATTEMPTS]
    if mismatched or incorrect_budgets:
        raise AssertionError(f"Proposal replay/budget failed: {mismatched[:3]}, {incorrect_budgets[:3]}")
    return {"P1_P2_paired_proposals": len(proposal_pairs), "P1_P2_replay_exact": True,
            "policy_condition_seed_budgets": len(budgets), "every_budget_has_12_draws": True}


def policy_pair_comparisons(ledger: list[dict]) -> list[dict]:
    """Compare paired fixed budgets on each shared source panel and latent arm."""
    comparisons = [
        ("MF_P0", "DECLARED_DOMAIN", "MF_P1", "DECLARED_DOMAIN", True, "legacy_filter_sampling_package"),
        ("MF_P1", "DECLARED_DOMAIN", "MF_P2", "DECLARED_DOMAIN", True, "same_proposal_output_cell"),
        ("MF_P2", "DECLARED_DOMAIN", "MF_P3", "DECLARED_DOMAIN", True, "shared_source_vocabulary_expansion"),
        ("MF_P3", "LEARNED_ANION", "MF_P4", "LEARNED_ANION", True, "shared_homogeneous_X_projection"),
        ("MF_P3", "LEARNED_ANION", "MF_P4", "LEARNED_ANION", False, "all_broad_X_projection"),
    ]
    by_condition: dict[tuple, list[dict]] = defaultdict(list)
    for row in ledger:
        key = (row["material_id"], row["latent_arm"], row["seed"],
               row["policy"], row["anion_setting"])
        by_condition[key].append(row)
    output = []
    for left_id, left_setting, right_id, right_setting, require_scope, interpretation in comparisons:
        for arm in LATENT_ARMS:
            pairs = []
            keys = {(mid, latent_arm, seed) for mid, latent_arm, seed, _, _ in by_condition
                    if latent_arm == arm}
            for material_id, _, seed in sorted(keys):
                left = by_condition.get((material_id, arm, seed, left_id, left_setting))
                right = by_condition.get((material_id, arm, seed, right_id, right_setting))
                if left is None or right is None:
                    continue
                if require_scope and (not left[0]["source_in_policy_scope"]
                                      or not right[0]["source_in_policy_scope"]):
                    continue
                if len(left) != ATTEMPTS or len(right) != ATTEMPTS:
                    raise AssertionError("Unpaired fixed budget")
                pairs.append({
                    "material_id": material_id,
                    "composition_group": left[0]["composition_group"],
                    "accepted_left": sum(row["policy_accepted"] for row in left) / ATTEMPTS,
                    "accepted_right": sum(row["policy_accepted"] for row in right) / ATTEMPTS,
                    "recovered_left": any(row["source_structure_match_scale_false"] for row in left),
                    "recovered_right": any(row["source_structure_match_scale_false"] for row in right),
                })
            grouped: dict[str, list[float]] = defaultdict(list)
            for pair in pairs:
                grouped[pair["composition_group"]].append(pair["accepted_right"] - pair["accepted_left"])
            group_values = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
            if len(group_values):
                rng = np.random.default_rng(20260924)
                bootstrap = rng.choice(group_values, size=(3000, len(group_values)), replace=True).mean(1)
                interval = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
                effect = float(group_values.mean())
            else:
                interval = None
                effect = None
            output.append({
                "comparison": interpretation,
                "left_policy": left_id, "left_setting": left_setting,
                "right_policy": right_id, "right_setting": right_setting,
                "latent_arm": arm, "source_scope_intersection_required": require_scope,
                "N_references": len({pair["material_id"] for pair in pairs}),
                "N_groups": len(group_values), "N_paired_seed_conditions": len(pairs),
                "left_accepted_per_attempt": (float(np.mean([pair["accepted_left"] for pair in pairs]))
                                              if pairs else None),
                "right_accepted_per_attempt": (float(np.mean([pair["accepted_right"] for pair in pairs]))
                                               if pairs else None),
                "right_minus_left_accepted_per_attempt": effect,
                "bootstrap_95_group_interval": interval,
                "left_reference_seed_recoveries": sum(pair["recovered_left"] for pair in pairs),
                "right_reference_seed_recoveries": sum(pair["recovered_right"] for pair in pairs),
            })
    return output
