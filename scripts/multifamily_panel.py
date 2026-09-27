"""Compatibility imports for the multifamily validation-panel selector."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meidnet_audits.multifamily_panel import (  # noqa: E402
    _balanced_group_sample, _prepare, _stratum, select_validation_panels,
)
