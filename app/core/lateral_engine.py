from __future__ import annotations
from typing import Dict, Any

from .models import LateralInputs, LateralResults

import app.lateral as lateral_module

def run_lateral(inputs: LateralInputs) -> LateralResults:
  """
  Thin wrapper around existing lateral logic.
  UI should call this instead of contsructing + solving everything inside main_window.py.
  """
  payload: Dict[str, Any] = dict(input.params)

  out = lateral_module.run_lateral_analysis(payload)

  return LateralResults(results=out)