"""Frozen MEIDNet encoding and reference-free MF_P4 decoding adapter.

This is the only diffusion module that imports the original model, CIF tools,
and the already audited output policy.  A generated proposal takes a latent and
an independent categorical seed; it has no reference-CIF/species input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MEIDNET_ROOT = ROOT / "MEIDNet-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MEIDNET_ROOT) not in sys.path:
    sys.path.insert(0, str(MEIDNET_ROOT))

from scripts import multifamily_policies as policies  # noqa: E402
from scripts import validate_setup  # noqa: E402
from meidnet.model import MAX_SITES, NUM_SPECIES  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from latent_diffusion.data import sha256_file  # noqa: E402


DEFAULT_CHECKPOINT = MEIDNET_ROOT / "checkpoints" / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
DEFAULT_TEMPLATE_MANIFEST = ROOT / "reports" / "template_reachability" / "input_manifest.json"
DEFAULT_TRAINING_SUPPORT = ROOT / "reports" / "multifamily_rule_relaxation" / "training_support.json"


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class FrozenMEIDNet:
    """Strict checkpoint wrapper with the original [heat_all, dir_gap] convention."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        template_manifest_path: Path = DEFAULT_TEMPLATE_MANIFEST,
        training_support_path: Path = DEFAULT_TRAINING_SUPPORT,
        device: str = "cpu",
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no fallback is allowed")
        self.device = requested
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.template_manifest_path = Path(template_manifest_path).resolve()
        self.training_support_path = Path(training_support_path).resolve()
        checkpoint, checkpoint_info = validate_setup.checkpoint_summary(self.checkpoint_path)
        model, load_info = validate_setup.instantiate_model(checkpoint)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.latent_dim = int(checkpoint["latent_dim_common"])
        self.checkpoint_sha256 = checkpoint_info["sha256"]
        self.model_state_sha256 = model_state_sha256(self.model)
        self.checkpoint_metadata = checkpoint_info["metadata"]
        self.strict_load = bool(load_info["strict_state_dict_load"])

        manifest = json.loads(self.template_manifest_path.read_text(encoding="utf-8"))
        coords = torch.tensor(manifest["coordinates"], dtype=torch.float32)
        center = torch.tensor(manifest["center_from_first_five"], dtype=torch.float32)
        if coords.shape != (MAX_SITES, 3) or center.shape != (3,):
            raise ValueError("Universal decoder input shape mismatch")
        if not (tensor_sha256(coords) == manifest["coordinates_sha256"] == policies.INPUT_TEMPLATE_20_SLOT_SHA256):
            raise ValueError("Universal decoder coordinate hash mismatch")
        if tensor_sha256(center) != manifest["center_sha256"]:
            raise ValueError("Universal decoder center hash mismatch")
        if not torch.equal(center, coords[:5].mean(dim=0)):
            raise ValueError("Universal decoder center is not occupied-site mean")
        self.coordinates = coords.to(self.device)
        self.center = center.to(self.device)
        self.template_sha256 = sha256_file(self.template_manifest_path)
        self.template_coordinates_sha256 = tensor_sha256(coords)

        support = json.loads(self.training_support_path.read_text(encoding="utf-8"))
        if support.get("policy_schema_version") != policies.POLICY_SCHEMA_VERSION:
            raise ValueError("Training vocabulary policy schema mismatch")
        self.training_support = support
        self.training_support_sha256 = sha256_file(self.training_support_path)
        self.vocab = {role: tuple(support[f"V_{role}"]) for role in ("A", "B", "X")}
        if any(not self.vocab[role] for role in self.vocab):
            raise ValueError("Empty training-derived role vocabulary")
        self.policy = policies.PolicyConfig("MF_P4", temperature=1.25, anion_setting="LEARNED_ANION")
        self.policy_sha256 = sha256_file(Path(policies.__file__))
        self.output_template_sha256 = policies.OUTPUT_TEMPLATE_SHA256
        from meidnet import design as original_generation  # noqa: E402
        if not np.array_equal(original_generation.TEMPLATE.detach().cpu().numpy().astype("<f4"),
                              np.asarray(policies.TEMPLATE_FRACTIONAL, dtype="<f4")):
            raise ValueError("Policy output template differs from original generation template")
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("MEIDNet must remain in eval mode with gradients disabled")
        if model_state_sha256(self.model) != self.model_state_sha256:
            raise AssertionError("MEIDNet state changed")

    @torch.no_grad()
    def encode_joint(self, crystal_vec: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Exact audited z_joint: average of two normalized projected latents."""
        crystal_vec = torch.as_tensor(crystal_vec, dtype=torch.float32, device=self.device)
        conditions = torch.as_tensor(conditions, dtype=torch.float32, device=self.device)
        if crystal_vec.ndim != 2 or conditions.shape != (len(crystal_vec), 2):
            raise ValueError("Expected batched dense crystals and raw [heat_all,dir_gap] pairs")
        _, _, joint, _, _, _ = self.model.encode_modalities(
            crystal_vec, conditions[:, 0], conditions[:, 1]
        )
        if joint.shape != (len(crystal_vec), self.latent_dim) or not bool(torch.isfinite(joint).all()):
            raise ValueError("Joint encoder returned invalid latent values")
        return joint.detach()

    @torch.no_grad()
    def property_only_latent(self, conditions: torch.Tensor) -> torch.Tensor:
        """Frozen property-common path with raw, *unstandardized* CMR conditions."""
        values = torch.as_tensor(conditions, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != 2 or not bool(torch.isfinite(values).all()):
            raise ValueError("Expected finite [heat_all,dir_gap] pairs")
        raw = self.model.property_encoder(values)
        common = torch.nn.functional.normalize(
            self.model.proj_prop(torch.nn.functional.normalize(raw, p=2, dim=1)), p=2, dim=1
        )
        return common.detach()

    @torch.no_grad()
    def decode_heads(self, latent: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.latent_dim or not bool(torch.isfinite(values).all()):
            raise ValueError("Expected finite common-space latent [N,D]")
        coords = self.coordinates.unsqueeze(0).expand(len(values), -1, -1)
        centers = self.center.unsqueeze(0).expand(len(values), -1)
        return self.model.crystal_decoder(values, input_coords=coords, center=centers)

    @torch.no_grad()
    def decode_mf_p4(self, latent: torch.Tensor, draw_seed: int) -> dict:
        """Decode one common latent using the fixed MF_P4 policy and draw seed.

        The returned record retains the raw lattice head and any construction
        or minimal-check rejection reason. CIF serialization errors are separate.
        """
        values = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape != (1, self.latent_dim):
            raise ValueError("decode_mf_p4 accepts exactly one latent, never a reference batch")
        heads = self.decode_heads(values)
        lat_out, _, logits, _ = heads
        raw_lat = lat_out[0].detach().cpu().float().tolist()
        raw_lengths = [20.0 * float(value) for value in raw_lat[:3]]
        raw_angles = [180.0 * float(value) for value in raw_lat[3:6]]
        result = {
            "policy": "MF_P4", "anion_setting": "LEARNED_ANION",
            "draw_seed": int(draw_seed), "raw_lattice_head": raw_lat,
            "raw_lengths_angstrom": raw_lengths, "raw_angles_degrees": raw_angles,
            "all_heads_finite": all(bool(torch.isfinite(head).all()) for head in heads),
            "sampled": False, "sampled_roles": None, "constructed": False,
            "minimal_valid": False, "accepted": False,
            "rejection_reason": None, "structure": None, "cif": None,
        }
        if not result["all_heads_finite"]:
            return result | {"rejection_reason": "NONFINITE_DECODER_HEAD"}
        proposal = policies.sample_roles(logits[0], self.policy, self.vocab, int(draw_seed))
        result["sampled"] = bool(proposal["sampled"])
        result["sampled_roles"] = tuple(proposal["roles"]) if proposal["sampled"] else None
        result["selected_role_probabilities"] = proposal.get("selected_probabilities", [])
        result["distribution_diagnostics"] = proposal.get("distribution_diagnostics", [])
        if not proposal["sampled"]:
            return result | {"rejection_reason": proposal["reason"]}
        roles = result["sampled_roles"]
        structure, cell = policies.construct_candidate(roles, self.policy, lat_out[0])
        result["constructed"] = structure is not None
        result["projected_cubic_length_angstrom"] = cell.get("projected_cubic_length_angstrom")
        result["cell_source"] = cell.get("cell_source")
        result["structure"] = structure
        if structure is None:
            return result | {"rejection_reason": cell["reason"]}
        minimal = policies.minimal_check(structure, roles, self.vocab)
        result["minimal_valid"] = bool(minimal["valid"])
        result["minimum_periodic_distance_angstrom"] = minimal["minimum_periodic_distance_angstrom"]
        result["minimal_failure_reasons"] = minimal["failure_reasons"]
        result["accepted"] = bool(minimal["valid"])
        if not result["accepted"]:
            return result | {"rejection_reason": ";".join(minimal["failure_reasons"])}
        try:
            result["cif"] = str(CifWriter(structure, symprec=None))
        except Exception as error:
            # Preserve the MF_P4 minimal-acceptance result and expose the
            # distinct downstream serialization failure without repair.
            result["cif_write_error"] = f"{type(error).__name__}:{error}"
        return result

    decode_from_latent = decode_mf_p4
