"""Explicit sampling and output construction for the multifamily decoder audit.

The frozen decoder and its universal *input* template live elsewhere. This module
only chooses five output species and constructs their output cell/structure.
MF_P0 must be run through the unchanged legacy decoder, not this sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from pymatgen.core import Lattice, Structure


MEIDNET_ROOT = Path(__file__).resolve().parents[1] / "MEIDNet-main"
if str(MEIDNET_ROOT) not in sys.path:
    sys.path.insert(0, str(MEIDNET_ROOT))

from meidnet import design as legacy  # noqa: E402


POLICY_SCHEMA_VERSION = "multifamily_v1"
POLICY_IDS = ("MF_P0", "MF_P1", "MF_P2", "MF_P3", "MF_P4")
LEGACY_FAMILY_X = {
    "oxide": ("O",),
    "nitride": ("N",),
    "halide": ("F", "Cl", "Br", "I"),
    "chalcogenide": ("S", "Se", "Te"),
}
# The decoder INPUT uses the 20-slot parser permutation recorded in the prior
# input manifest. Output construction keeps design.TEMPLATE's original order.
INPUT_TEMPLATE_20_SLOT_SHA256 = "2b92f48bde0008fe445fd81bc63a032b6121116e2080810acaf3ae6bd7315459"
TEMPLATE_FRACTIONAL = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
OUTPUT_TEMPLATE_SHA256 = hashlib.sha256(
    np.asarray(TEMPLATE_FRACTIONAL, dtype="<f4").tobytes(order="C")
).hexdigest()
MIN_SEPARATION_ANGSTROM = 0.8
MAX_PERIODIC_IMAGES = 50_000


@dataclass(frozen=True)
class PolicyConfig:
    policy_id: str
    family: str | None = None
    temperature: float = 1.25
    anion_setting: str | None = None

    def __post_init__(self) -> None:
        if self.policy_id not in POLICY_IDS:
            raise ValueError(f"Unknown policy: {self.policy_id}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        setting = self.anion_setting
        if setting is None:
            setting = "DECLARED_DOMAIN" if self.policy_id in POLICY_IDS[:3] else "LEARNED_ANION"
            object.__setattr__(self, "anion_setting", setting)
        if setting not in ("LEARNED_ANION", "DECLARED_DOMAIN"):
            raise ValueError(f"Unknown anion setting: {setting}")
        if self.policy_id in POLICY_IDS[:3]:
            if self.family not in LEGACY_FAMILY_X:
                raise ValueError("Legacy policies require an explicit supported family")
            if setting != "DECLARED_DOMAIN":
                raise ValueError("Legacy policies have declared-family X scope")


def proposal_seed(*parts: object) -> int:
    """Stable seed for one reference/latent/replicate/attempt tuple."""
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def build_training_vocabulary(
    train_roles: Sequence[Sequence[str]],
    species_list: Sequence[str] = legacy.SPECIES_LIST,
) -> dict[str, object]:
    """Role counts are computed solely from the supplied training role rows."""
    allowed = set(species_list)
    counts = {role: Counter() for role in ("A", "B", "X")}
    unsupported = []
    for row_number, symbols in enumerate(train_roles):
        if len(symbols) != 5:
            raise ValueError("Training role row must contain [A, B, X1, X2, X3]")
        for slot, role in enumerate(("A", "B", "X", "X", "X")):
            if symbols[slot] in allowed:
                counts[role][symbols[slot]] += 1
            else:
                unsupported.append({"row_number": row_number, "slot": slot, "symbol": symbols[slot]})
    return {
        role: tuple(symbol for symbol in species_list if counts[role][symbol])
        for role in counts
    } | {
        "counts": {role: dict(counts[role]) for role in counts},
        "unsupported_training_roles": unsupported,
    }


def allowed_roles(
    policy: PolicyConfig,
    vocab: Mapping[str, Sequence[str]],
    declared_anions: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return A/B/X supports without a source-dependent or oxide fallback."""
    if policy.policy_id in POLICY_IDS[:3]:
        assert policy.family is not None
        a_pool = legacy.A_CATIONS
        b_pool = legacy.VALENCE_B_BY_FAMILY[policy.family]
        x_pool = LEGACY_FAMILY_X[policy.family]
    else:
        a_pool, b_pool, x_pool = (vocab[role] for role in ("A", "B", "X"))
    if policy.anion_setting == "DECLARED_DOMAIN":
        if declared_anions is None:
            if policy.policy_id in POLICY_IDS[:3]:
                declared_anions = LEGACY_FAMILY_X[policy.family]
            else:
                raise ValueError("Expanded declared-domain policy needs explicit X support")
        x_pool = set(x_pool) & set(declared_anions)
    elif declared_anions is not None:
        raise ValueError("Learned-anion policy cannot receive reference/domain X support")
    return {
        "A": tuple(symbol for symbol in legacy.SPECIES_LIST if symbol in a_pool),
        "B": tuple(symbol for symbol in legacy.SPECIES_LIST if symbol in b_pool),
        "X": tuple(symbol for symbol in legacy.SPECIES_LIST if symbol in x_pool),
    }


def source_scope(
    roles: Sequence[str],
    policy: PolicyConfig,
    vocab: Mapping[str, Sequence[str]],
    declared_anions: Sequence[str] | None = None,
) -> dict[str, object]:
    """Membership of actual source-site roles; no geometry or chemistry claim."""
    if len(roles) != 5:
        return {"eligible": False, "reasons": ["INVALID_ROLE_COUNT"]}
    support = allowed_roles(policy, vocab, declared_anions)
    reasons = []
    if policy.policy_id in POLICY_IDS[:4] and len(set(roles[2:])) != 1:
        reasons.append("OUTSIDE_HOMOGENEOUS_X_SCOPE")
    for index, role in enumerate(("A", "B", "X", "X", "X")):
        if roles[index] not in support[role]:
            reasons.append(f"OUT_OF_SUPPORT_{role}{index - 1 if role == 'X' else ''}")
    return {"eligible": not reasons, "reasons": reasons}


def full_composition_key(roles: Sequence[str]) -> tuple[tuple[str, int], ...]:
    """All five species count; mixed X assignments cannot share an AB-only key."""
    return tuple(sorted(Counter(roles).items()))


def mixed_multiset_probability(
    x_slot_probabilities: Sequence[Mapping[str, float]],
    source_x: Sequence[str],
) -> float:
    """Sum independent X-slot mass over distinct source-compatible orders."""
    return math.exp(mixed_multiset_log_probability(x_slot_probabilities, source_x))


def mixed_multiset_log_probability(
    x_slot_probabilities: Sequence[Mapping[str, float]],
    source_x: Sequence[str],
) -> float:
    """Stable logsumexp over distinct compatible X orders, including repeated X."""
    if len(x_slot_probabilities) != 3 or len(source_x) != 3:
        raise ValueError("Three X slots are required")
    terms = []
    for order in sorted(set(permutations(source_x))):
        values = [x_slot_probabilities[slot].get(symbol, 0.0) for slot, symbol in enumerate(order)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("X probabilities must be finite and nonnegative")
        if all(value > 0 for value in values):
            terms.append(sum(math.log(value) for value in values))
    if not terms:
        return -math.inf
    largest = max(terms)
    return largest + math.log(sum(math.exp(value - largest) for value in terms))


def _array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _distribution(scores: np.ndarray) -> tuple[np.ndarray | None, str | None, dict[str, int]]:
    """Softmax over full declared support; -inf masks are explicit zero mass."""
    if np.isnan(scores).any() or np.isposinf(scores).any():
        return None, "INVALID_LOGITS", {}
    finite = np.isfinite(scores)
    if not finite.any():
        return None, "NO_FINITE_ALLOWED_LOGITS", {}
    shifted = scores - float(np.max(scores[finite]))
    probabilities = np.exp(shifted)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        return None, "INVALID_NORMALIZATION", {}
    probabilities /= total
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        return None, "INVALID_PROBABILITIES", {}
    zero_finite = int(np.count_nonzero(finite & (probabilities == 0)))
    return probabilities, None, {
        "masked_minus_infinity": int(np.count_nonzero(~finite)),
        "numeric_underflow": zero_finite,
    }


def sample_roles(
    logits: object,
    policy: PolicyConfig,
    vocab: Mapping[str, Sequence[str]],
    seed: int,
    declared_anions: Sequence[str] | None = None,
) -> dict[str, object]:
    """Sample one A,B,X1,X2,X3 proposal from cached, unmasked heads.

    P1/P2 replay exactly when called with the same heads, domain, and seed.
    P3 pools three X logits once; P4 draws each X slot independently.
    """
    if policy.policy_id == "MF_P0":
        raise NotImplementedError("MF_P0 uses the unchanged legacy decode_and_filter path")
    values = _array(logits).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[0] < 5 or values.shape[1] != len(legacy.SPECIES_LIST):
        return {"sampled": False, "roles": None, "reason": "INVALID_LOGIT_SHAPE"}
    support = allowed_roles(policy, vocab, declared_anions)
    if any(not support[role] for role in ("A", "B", "X")):
        return {"sampled": False, "roles": None, "reason": "EMPTY_ROLE_SUPPORT"}
    index = {symbol: position for position, symbol in enumerate(legacy.SPECIES_LIST)}
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    slot_probabilities: list[dict[str, float]] = []
    diagnostics: list[dict[str, int]] = []
    slot_roles = ("A", "B", "X", "X", "X")
    for slot, role in enumerate(slot_roles):
        if slot > 2 and policy.policy_id != "MF_P4":
            chosen.append(chosen[2])
            slot_probabilities.append(slot_probabilities[2])
            diagnostics.append(diagnostics[2])
            continue
        symbols = support[role]
        indices = [index[symbol] for symbol in symbols]
        head = values[2:5, indices].sum(axis=0) if slot == 2 and policy.policy_id != "MF_P4" else values[slot, indices]
        probabilities, failure, info = _distribution(head / policy.temperature)
        if failure:
            return {"sampled": False, "roles": None, "reason": f"{failure}_SLOT_{slot}"}
        assert probabilities is not None
        pick = int(rng.choice(len(symbols), p=probabilities))
        chosen.append(symbols[pick])
        slot_probabilities.append(dict(zip(symbols, map(float, probabilities))))
        diagnostics.append(info)
    return {
        "sampled": True,
        "roles": tuple(chosen),
        "reason": None,
        "slot_probabilities": slot_probabilities,
        "distribution_diagnostics": diagnostics,
        "selected_probabilities": [slot_probabilities[i][chosen[i]] for i in range(5)],
        "forced_x": policy.anion_setting == "DECLARED_DOMAIN" and len(support["X"]) == 1,
    }


def construct_candidate(
    roles: Sequence[str],
    policy: PolicyConfig,
    lat_out: object,
) -> tuple[Structure | None, dict[str, object]]:
    """Construct a decorated five-site output without symmetry refinement."""
    if policy.policy_id == "MF_P0":
        raise NotImplementedError("MF_P0 uses the unchanged legacy decode_and_filter path")
    if len(roles) != 5:
        return None, {"constructed": False, "reason": "INVALID_ROLE_COUNT"}
    if policy.policy_id in ("MF_P1", "MF_P2", "MF_P3") and len(set(roles[2:])) != 1:
        return None, {"constructed": False, "reason": "OUTSIDE_HOMOGENEOUS_X_SCOPE"}
    lattice_output = _array(lat_out).astype(np.float64, copy=False).reshape(-1)
    if lattice_output.size < 6:
        return None, {"constructed": False, "reason": "INVALID_LATTICE_HEAD_SHAPE"}
    raw_lengths = tuple(float(x) for x in 20.0 * lattice_output[:3])
    raw_angles = tuple(float(x) for x in 180.0 * lattice_output[3:6])
    details: dict[str, object] = {"raw_lengths_angstrom": raw_lengths, "raw_angles_degrees": raw_angles}
    if policy.policy_id == "MF_P1":
        # The legacy radius/fallback/clipping rule is retained here as a control.
        radius_b = legacy.get_ionic_radius(roles[1])
        radius_x = legacy.get_ionic_radius(roles[2])
        length = float(np.clip(2.0 * (radius_b + radius_x), 3.0, 8.0))
        details["cell_source"] = "legacy_radius_derived"
    else:
        if not np.isfinite(lattice_output[:6]).all() or any(x <= 0 for x in raw_lengths):
            return None, details | {"constructed": False, "reason": "INVALID_LEARNED_LATTICE"}
        length = float(np.mean(raw_lengths))
        details["cell_source"] = "mean_of_training_inverse_scaled_lengths"
    details["projected_cubic_length_angstrom"] = length
    try:
        structure = Structure(
            Lattice.cubic(length), list(roles), TEMPLATE_FRACTIONAL,
            coords_are_cartesian=False,
        )
    except (TypeError, ValueError) as error:
        return None, details | {"constructed": False, "reason": f"STRUCTURE_ERROR:{type(error).__name__}"}
    return structure, details | {"constructed": True, "reason": None}


def _shortest_nonzero_lattice_translation(lattice: np.ndarray) -> tuple[float | None, str | None]:
    singular_values = np.linalg.svd(lattice, compute_uv=False)
    smallest = float(np.min(singular_values))
    if not math.isfinite(smallest) or smallest <= 1e-8:
        return None, "SINGULAR_CELL"
    best = float(np.min(np.linalg.norm(lattice, axis=1)))
    bound = int(math.ceil(best / smallest)) + 1
    if (2 * bound + 1) ** 3 > MAX_PERIODIC_IMAGES:
        return None, "PERIODIC_SEARCH_LIMIT"
    for image in product(range(-bound, bound + 1), repeat=3):
        if image != (0, 0, 0):
            best = min(best, float(np.linalg.norm(np.asarray(image) @ lattice)))
    return best, None


def minimal_check(
    structure: Structure | None,
    roles: Sequence[str],
    vocab: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Finite five-site cell, declared role support, and periodic 0.8-A floor."""
    failures: list[str] = []
    if structure is None:
        return {"valid": False, "failure_reasons": ["NO_STRUCTURE"], "minimum_periodic_distance_angstrom": None}
    if len(roles) != 5 or len(structure) != 5:
        failures.append("SITE_COUNT_NOT_FIVE")
    support = {role: set(vocab[role]) for role in ("A", "B", "X")}
    if len(roles) == 5:
        for index, role in enumerate(("A", "B", "X", "X", "X")):
            if roles[index] not in support[role]:
                failures.append(f"OUT_OF_SUPPORT_{role}{index - 1 if role == 'X' else ''}")
    for index, site in enumerate(structure):
        if not site.is_ordered or not math.isclose(sum(site.species.values()), 1.0, abs_tol=1e-8):
            failures.append(f"NON_FULL_OR_DISORDERED_SITE_{index}")
        elif index < len(roles) and site.specie.symbol != roles[index]:
            failures.append(f"SITE_ROLE_MISMATCH_{index}")
    matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
    if (not np.isfinite(matrix).all() or not np.isfinite(structure.frac_coords).all()
            or not np.isfinite(structure.cart_coords).all()):
        failures.append("NONFINITE_GEOMETRY")
    if np.isfinite(matrix).all() and float(np.max(np.abs(matrix))) > 1e6:
        failures.append("CELL_NUMERICAL_GUARD")
    volume = float(abs(np.linalg.det(matrix))) if np.isfinite(matrix).all() else float("nan")
    if not math.isfinite(volume) or volume <= 1e-8:
        failures.append("NONPOSITIVE_OR_SINGULAR_VOLUME")
    if np.isfinite(matrix).all() and "CELL_NUMERICAL_GUARD" not in failures:
        metric_eigenvalues = np.linalg.eigvalsh(matrix @ matrix.T)
        if not np.isfinite(metric_eigenvalues).all() or float(metric_eigenvalues.min()) <= 1e-12:
            failures.append("INVALID_CELL_METRIC")
    minimum: float | None = None
    if len(structure) == 5 and not any(reason in failures for reason in (
        "NONFINITE_GEOMETRY", "NONPOSITIVE_OR_SINGULAR_VOLUME", "INVALID_CELL_METRIC",
        "CELL_NUMERICAL_GUARD",
    )):
        self_distance, search_failure = _shortest_nonzero_lattice_translation(matrix)
        if search_failure:
            failures.append(search_failure)
        else:
            assert self_distance is not None
            minimum = self_distance
            for i in range(5):
                for j in range(i + 1, 5):
                    minimum = min(minimum, float(structure.get_distance(i, j)))
            if minimum < MIN_SEPARATION_ANGSTROM:
                failures.append("PERIODIC_OVERLAP_BELOW_0P8_ANGSTROM")
    return {
        "valid": not failures,
        "failure_reasons": failures,
        "minimum_periodic_distance_angstrom": minimum,
        "cell_volume_angstrom3": volume,
    }
