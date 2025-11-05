"""
Axial load-settlement analysis for a single pile.

This routine slices the pile into segments and iterates on settlements using a simple Newton-Raphson scheme. It's intentionally compact for teaching; a production solver would add more guards, damping, and diagnostics.
"""

from __future__ import annotations
import numpy as np
from .curves import get_tz_curve, get_qz_curve
from typing import List, Dict, Tuple

def axial_analysis(pile: Dict, loads: Dict, soil_profile: List[Dict], n_segments: int = 50) -> Dict:
  # inputs and derived geometry
  L = float(pile["length_m"])
  D = float(pile["diameter_m"])
  E = float(pile["elastic_modulus_pa"])

  A = 0.25 * np.pi * D**2   # cross-sectional area
  Pm = np.pi * D            # perimeter
  EA = E * A                # Axial stiffness (N)

  target_kN = float(loads.get("axial_kN", 0.0))
  n_segments = int(max(10, n_segments))

  dz = L / n_segments
  z = np.linspace(0.0, L, n_segments + 1) # nodes, 0=head

  #--------------helpers--------------
  def get_layer_for_z(zz: float) -> Dict:
    for layer in soil_profile:
      if layer["from_m"] <= zz < layer["to_m"]:
        return layer
    return soil_profile[-1] if soil_profile else {}
  
  def interp_clamped(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    """
    Robust linear interpolation with clamping and monotonic fix:
    - removes NaNs
    - sorts by x ascending
    - nudges duplicates to ensure strict increase
    - clamps outside to endpoints
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # drop NaNs
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if x.size == 0:
      return 0.0
    if x.size == 1:
      return float(y[0])
    
    # sort by x
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    # enforce strictly increasing x
    eps = 1e-12
    for k in range(1, x.size):
      if x[k] <= x[k-1]:
        x[k] = x[k-1] + eps

    # interpolate with clamping
    return float(np.interp(xq, x, y, left=y[0], right=y[-1]))

  
  #-----------Load Stepping-------------
  if target_kN <= 0.0:
    return{
      "loads_kN": np.array([0.0]),
      "settlements_m": np.array([0.0]),
      "plots": {
        "z_m": z,
        "settlements_m": np.zeros_like(z),
        "shear_N": np.zeros_like(z),
        "toe_res_N": 0.0,
      },
    }
  
  n_steps = 10
  loads_kN = np.linspace(0.0, target_kN, n_steps)
  settlements = []
  w_prev = np.zeros(n_segments + 1)

  #-------------step over load levels----------
  for P_kN in loads_kN:
    P = P_kN * 1e3
    w = w_prev.copy() # warm start imporves stability

    tol_dw = 1e-6   # m(1 micron) for max component of dw
    max_iter = 80
    max_step = 5e-4  #m (0.5 mm) cap for Newton step

    for _ in range(max_iter):
      residual = np.zeros(n_segments + 1)
      J = np.zeros((n_segments + 1, n_segments + 1))

      #--------Head equilibrium: EA*(w1 - w0)/dz - P = 0\
      residual[0] = EA * (w[1] - w[0]) / dz - P
      J[0, 0] = -EA / dz
      J[0, 1] = EA / dz

      #-------Internal nodes: EA * w'' + q_side = 0
      for i in range(1, n_segments):
        mid_layer = get_layer_for_z(z[i])
        tz_z, tz_t = get_tz_curve(mid_layer, D, z[i]) if mid_layer else (np.array([0.0]), np.array([0.0]))

        slip = abs(w[i] - w[i - 1])  # relative movement along segment (m)

        # shear stress (kPa) from t-z curve at current slip
        t_kPa = interp_clamped(slip, tz_z, tz_t)
        t_Npm2 = t_kPa * 1e3
        q_side = t_Npm2 * Pm # N/m (distributed along the segment)

        # Residual via central difference for axial bar
        residual[i] = EA * (w[i + 1] - 2.0 * w[i] + w[i -1]) / dz**2 + q_side

        # Structural Jacobian part
        J[i, i - 1] += EA / dz**2
        J[i, i] += -2.0 * EA / dz**2
        J[i, i + 1] += EA / dz**2

        # Soil Jacobian part (secant derivative dt/dslip)
        if len(tz_z) > 1:
          ds = max(1e-8, 0.05 * (float(np.max(tz_z)) - float(np.min(tz_z))))
          t2 = interp_clamped(slip + ds, tz_z, tz_t)
          dt_dslip_kPa = (t2 - t_kPa) / ds
        else:
          dt_dslip_kPa = 0.0

        dqdw = (dt_dslip_kPa * 1e3) * Pm * (1.0 if (w[i] - w[i - 1]) >= 0.0 else -1.0)
        # q_side depends on (w[i] - w[i - 1]: +dq/dw[i], - dq/dw[i-1])
        J[i, i] += dqdw
        J[i, i - 1] -= dqdw

      #----------Toe equilibrium: EA*(w[-1] - w[-2])/dz + Q_toe = 0
      tip_layer = get_layer_for_z(L)
      qz_z, qz_q = get_qz_curve(tip_layer, D, L) if tip_layer else (np.array([0.0]), np.array([0.0]))

      slip_toe = abs(w[-1])
      q_kPa = interp_clamped(slip_toe, qz_z, qz_q)
      Q_toe = q_kPa * 1e3 * A

      residual[-1] = EA * (w[-1] - w[-2]) / dz + Q_toe
      J[-1, -2] = -EA / dz
      J[-1, -1] = EA / dz

      # dQ_toe/dw[-1]
      if np.size(qz_z) > 1:
        ds = max(1e-8, 0.05 * (float(np.max(qz_z)) - float(np.min(qz_z))))
        q2 = interp_clamped(slip_toe + ds, qz_z, qz_q)
        dq_dslip_kPa = (q2 - q_kPa) / ds
      else:
        dq_dslip_kPa = 0.0

      dQdw = (dq_dslip_kPa * 1e3) * A * (1.0 if w[-1] >= 0.0 else -1.0)
      J[-1, -1] += dQdw

      #-------solver for increment--------
      try:
        dw = np.linalg.solve(J, -residual)
      except np.linalg.LinAlgError:
        # Lightly regularize if near singular
        dw = np.linalg.lstsq(J + 1e-9 * np.eye(J.shape[0]), -residual, rcond=None)[0]

      # Cap the maximum component of the Newton Step (stability guard)
      max_comp = np.max(np.abs(dw))
      if np.isfinite(max_comp) and max_comp > max_step:
        dw *= (max_step / max_comp)
        
      w += dw
      if np.max(np.abs(dw)) < tol_dw:
        break

    settlements.append(w[0])
    w_final = w # Keep final converged profile of this load step
    w_prev = w.copy()

  #-----------Build plots from the final converged step------------
  shear_per_seg_N = np.zeros(n_segments)
  for i in range(n_segments):
    mid_z = z[i] + 0.5 * dz
    layer = get_layer_for_z(mid_z)
    tz_z, tz_t = get_tz_curve(layer, D, mid_z) if layer else (np.array([0.0]), np.array([0.0]))
    slip = abs(w_final[i + 1] - w_final[i])
    t_kPa = interp_clamped(slip, tz_z, tz_t)
    shear_per_seg_N[i] = (t_kPa * 1e3) * Pm * dz
  
  shear_cum = np.concatenate([[0.0], np.cumsum(shear_per_seg_N)])

  qz_z, qz_q = get_qz_curve(get_layer_for_z(L), D, L) if soil_profile else (np.array([0.0]), np.array([0.0]))
  toe_kPa = interp_clamped(abs(w_final[-1]), qz_z, qz_q)
  toe_res_N = float(toe_kPa * 1e3 * A)

  plots = {
    "z_m": z,
    "settlements_m": w_final,
    "shear_N": shear_cum,
    "toe_res_N": toe_res_N,
  }

  return {
    "loads_kN": np.asarray(loads_kN),
    "settlements_m": np.asarray(settlements),
    "plots": plots,
  }
  