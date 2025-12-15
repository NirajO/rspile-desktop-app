from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass(frozen=True)
class AxialInputs:
  params: Dict[str, Any]

@dataclass(frozen=True)
class AxialResults:
  results: Dict[str, Any]

@dataclass(frozen=True)
class LateralInputs:
  params: Dict[str, Any]

@dataclass(frozen=True)
class LateralResults:
  results: Dict[str, Any]