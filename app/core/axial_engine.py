from __future__ import annotations
from typing import Dict, Any

from .models import AxialInputs, AxialResults
import app.axial as axial_module

def run_axial(inputs: AxialInputs) -> AxialResults:
  """
  Core axial analysis entry point.
  """
  payload: Dict[str, Any] = dict(input.params)

  out = axial_module.run_axial_analysis(payload)

  return AxialResults(results=out)