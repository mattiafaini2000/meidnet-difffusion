"""Run small, read-only import and platform checks for the setup environment."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = {
    "ase": "ase",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pymatgen": "pymatgen",
    "pytest": "pytest",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch": "torch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "MEIDNet-main",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root.resolve()))

    versions = {}
    for distribution, module in PACKAGES.items():
        importlib.import_module(module)
        versions[distribution] = metadata.version(distribution)

    import torch
    from meidnet.model import TripleModalityDataset, parse_cif_to_dense

    assert TripleModalityDataset is not None
    assert parse_cif_to_dense is not None

    report = {
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_root": str(args.source_root.resolve()),
        "packages": versions,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
