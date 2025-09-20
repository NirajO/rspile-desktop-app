"""
Axial load-settlement analysis for a single pile.

This routine slices the pile into segments and iterates on settlements using a simple Newton-Raphson scheme. It's intentionally compact for teaching; a production solver would add more guards, damping, and diagnostics.
"""

import numpy as np
import math
from .curves import get_tz_curve, get_qz_curve
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def axial_analysis(pile, loads, soil_profile, n_segments=50):
  """
  Compute head settlement vs. axial load and a few helper traces (e.g, shear)

  pile: dict with length_m, diameter_m, elastic_modulus_pa
  loads: dict with axial_kN (target load)
  soil_profile: list of soil layers (must be ordered by depth)
  n_segments: number of pile segments along the length

  Returns a dict with:
    -loads_kN: array of load steps
    -settlements_m: corresponding head settlements
    -plots: depthwise arrays that help plot internals (shear, toe resistance)
  """
  L= pile["length_m"]
  D = pile["diameter_m"]
  E = pile["elastic_modulus_pa"]
  A = math.pi * (D / 2) ** 2 
  EA = E * A

  dz = L / n_segments
  z = np.linspace (0, L, n_segments + 1)

  def get_layer_for_z(zz):
    for layer in soil_profile:
      if layer["from_m"] <= zz < layer["to_m"]:
        return layer
    return soil_profile[-1] if soil_profile else None
  
  # Precompute t_max per segment
  t_max_per_segment = []
  for i in range(n_segments):
    mid_z = z[i] + dz / 2
    layer = get_layer_for_z(mid_z)
    if layer:
      _, t = get_tz_curve(layer, D, mid_z)
      t_max_per_segment.append(max(t) if t else 0)
    else:
      t_max_per_segment.append(0)

  tip_layer = get_layer_for_z(L)
  q_max = 0
  if tip_layer:
    _, q = get_qz_curve(tip_layer, D, L)
    q_max = max(q) if q else 0

  # Newton-Raphson for settlement w (m)
  loads_kN = np.linspace(0, loads["axial_kN"], 10)
  settlements_m = []
  for P_kN in loads_kN:
    P = P_kN * 1000
    w = np.zeros(n_segments + 1)
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
      residual = np.zeros(n_segments + 1)
      Jacobian = np.zeros((n_segments + 1, n_segments + 1))

      # Top: strain = P / EA
      residual[0] = (w[1] - w[0] / dz - P / EA)
      Jacobian [0, 0] = -1 / dz
      Jacobian[0, 1] = 1 / dz

      # Internal
      perimeter = math.pi * D
      for i in range(1, n_segments):
        layer = get_layer_for_z(z[i])
        local_disp = abs(w[i] - w[i-1])
        tz_z, tz_t = get_tz_curve(layer, D, z[i])
        t_local = np.interp(local_disp, tz_z, tz_t) if len(tz_z) > 1 else 0
        shaft_force = t_local * 1000 * perimeter * dz
        residual[i] = EA * (w[i+1] - 2*w[i] + w[i-1]) / dz**2 + shaft_force / dz
        Jacobian[i, i-1] = EA / dz ** 2
        Jacobian[i, i] = -2 * EA / dz ** 2
        Jacobian[i, i+1] = EA / dz ** 2

      # Toe
      local_disp_toe = w[-1]
      qz_z, qz_t = get_qz_curve(tip_layer, D, L) if tip_layer else ([0], [0])
      q_toe = np.interp(local_disp_toe, qz_z, qz_t) if len(qz_z) > 1 else 0
      toe_force = q_toe * 1000 * A
      residual[-1] = (w[-1] - w[-2]) / dz + toe_force / EA
      Jacobian[-1, -2] = -1 / dz
      Jacobian[-1, -1] = 1 / dz

      dw = np.linalg.solve(Jacobian, -residual)
      w += dw
      if np.max(abs(dw)) < tol:
        break

    settlements_m.append(w[0])

  # Plots
  shear = np.cumsum([t_max_per_segment[i] * 1000 * math.pi * D * dz for i in range(n_segments)])
  toe_res = q_max * 1000 * A
  plots = {
    'z_m': z,
    'settlement_m': w,
    'shear_N': np.append(0, shear),
    'toe_res_N': toe_res
  }
  
  return {
    'loads_kN': loads_kN,
    'settlements_m': settlements_m,
    'plots': plots
  }

  
