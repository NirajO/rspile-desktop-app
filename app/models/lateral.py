from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Literal, Tuple
from enum import Enum

import numpy as np

"""
y(z): lateral deflection (m)
p(y, z): soil reaction per unit length (N/m)
Governing beam-on-nonlinear-Winkler foundation
"""
class BCType(str, Enum):
  FREE_HEAD = 'free_head'
  FIXED_HEAD = "fixed_head"

@dataclass
class PileProps:
  length_m: float
  EI_Nm2: float   #flexural rigidity
  d_m: float      # diameter (for reporting/future p-y scaling)
  n_nodes: int = 81 # odd number preffered (>=41)

@dataclass
class LateralLoadCase:
  H_N: float   #lateral load at head (z=0), positive to +x
  M_Nm: float = 0.0 #applied moment at head (optional)

@dataclass
class LateralConfig:
  bc: BCType = "free_head"
  max_iters: int = 40
  tol: float = 1e-6
  relax: float = 1.0    #0.5-1.0 helps convergence in very soft soils

@dataclass
class LateralResults:
  z_m: np.ndarray    #depth array (0 = head)
  y_m: np.ndarray    #deflection along depth
  theta_rad: np.ndarray #rotation dy/dz
  M_Nm: np.ndarray   # bending moment = EI * y''
  V_N: np.ndarray    #shear = dM/dz = EI * y'''
  p_N_per_m: np.ndarray # soil reaction along depth
  head_deflection_mm: float
  head_rotation_mrad: float

def finite_diff_mats(n: int, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Build 2nd- and 4th-derivative sparse-like tri/band matrices for unifrom grid.
  Returns (D2, D3, D4) as dense for simplicity (will switch to scipy.sparse later).
  """
  D2 = np.zeros((n, n))
  D3 = np.zeros((n, n))
  D4 = np.zeros((n, n))
  # central second derivative
  for i in range (1, n - 1):
    D2[i, i - 1] = 1.0
    D2[i, i] = -2.0
    D2[i, i + 1] = 1.0
  D2 /= dz**2
  # third derivative via D1(D2) approx (simple, adequate for baseline)
  D1 = np.zeros((n, n))
  for i in range(1, n - 1):
    D1[i, i + 1] = 0.5
    D1[i, i - 1] = -0.5
  D1 /= dz
  D3 = D1 @ D2
  D4 = D2 @ D2
  return D2, D3, D4

def apply_boundary_conditions(D2, D3, D4, y, pile: PileProps, cfg: LateralConfig, lc: LateralLoadCase, EI, dz):
  """
  Enforce head/tip BCs on matrices by replacing corresponding rows.
  Head (z=0): free_head -> M(0)=EI*y'' (0)=0 and V(0)=EI*y'''(0) = H
              fixed_head -> y(0)=0 and theta(0)=y'(0)=0; external H/M applied as restrictions
  Tip (z=L): free tip -> M(L)=0 and V(L)=0 (classical for long piles); can be adjusted if needed
  """
  n = y.size

  rhs_bc = np.zeros

  # Head (i = 0)
  if cfg.bc == BCType.FREE_HEAD:
    pass
  else:
    y[0] = 0.0
    D2[0, :] = 0.0; D3[0, :] = 0.0; D4[0, :] = 0.0
    D4[0, 0] = 1.0 #identify row, forces y(0)=0 via Newton matrix
    rhs_bc[0] = 0.0
    #theta(0)=0 -> y'(0)=0 ~ (-3y0 + 4y1 - y2)/(2dz) = 0 (use a ghost-free stencil)
    D2[1, :] = 0.0; D3[1, :] = 0.0; D4[1, :] = 0.0
    # Put a first-derivative row into D4 to reuse matrix; crude but effective
    if y.size >= 3:
      D4[1, 0] = -3.0/(2*dz)
      D4[1, 1] = 4.0/(2*dz)
      D4[1, 2] = -1.0/(2*dz)
    rhs_bc[1] = 0.0

  # Tip (i=n-1): free tip default => M(L)=0 and V(L)=0
  i = n - 1
  # Overwrite last two rows similarly to represent constraints:
  D2[i, :] = 0.0; D3[i, :]= 0.0; D4[i, :] = 0.0
  # Use D2 row to enforce M(L)=0 and D3 row to enforce V(L)=0 by copying stencils:
  # Simple approach: copy interior to last row indices
  if n >= 3:
    # M(L)=0: y''(L)=0
    D4[i, i-1] = 1.0/ (dz**2)
    D4[i, i] = -2.0 / (dz**2)
    D4[i, i-2] = 1.0 / (dz**2)
    rhs_bc[i] = 0.0
  
  if n >= 4:
    # V(L)=0: y'''(L)=0 -> add as row i-1 to avoid clobbering; keep it simple:
    D2[i-1, :] = 0.0; D3[i-1, :] = 0.0; D4[i-1, :] = 0.0
    D4[i-1, i] = 1.0/(dz**3)
    D4[i-1, i-1] = -3.0/(dz**3)
    D4[i-1, i-2] = 3.0/(dz**3)
    D4[i-1, i-3] = -1.0/(dz**3)
    rhs_bc[i-1] = 0.0

  return rhs_bc

def lateral_analysis(
    pile: PileProps,
    load_steps: List[LateralLoadCase],
    py_spring: Callable[[float, float], Tuple[float, float]],
    cfg: LateralConfig = LateralConfig()
) -> Dict[str, object]:
  """
  solver for lateral response under increasing head load/moment using Newton iterations.
  """
  n = int(pile.n_nodes)
  L = float(pile.length_m)
  z = np.linspace(0.0, L, n)
  dz = z[1] - z[0]
  EI = float(pile.EI_Nm2)

  D2, D3, D4 = finite_diff_mats(n, dz)

  # State across Load steps
  y_prev = np.zeros(n)

  results_per_step: List[LateralResults] = []
  head_curve = List[Tuple[float, float]] = []

  for _, lc in enumerate(load_steps, 1):
    y = y_prev.copy()
    for _ in range(cfg.max_iters):
      # Soil reaction and tangent at current y
      p = np.zeros(n)
      k = np.zeros(n)
      for i in range(n):
        p[i], k[i] = py_spring(y[i], z[i])

      # Newton system: EI*D4*y - p(y,z) = 0 -> (EI*D4 - diag(k)) dy = r
      A = EI * D4 - np.diag(k)
      r = p - (EI * (D4 @ y)) 

      # Boundary conditions
      rhs_bc = apply_boundary_conditions(D2.copy(), D3.copy(), A, y, pile, cfg, lc, EI, dz)
      
      # Inject head loads for free-head case
      if cfg.bc == BCType.FREE_HAND:
        #enforce M(0)=0: use D2 stencil row at 0
        A[0, :] = 0.0
        # approximate y''(0)=0 with [y0 - 2y1 + y2]/dz^2 = 0
        if n >= 3:
          A[0, 0] = 1.0/(dz**2)
          A[0, 1] = -2.0/(dz**2)
          A[0, 2] = 1.0/(dz**2)
        r[0] = 0.0
        # enforce V(0)=H: EI*y'''(0)=H -> add equation using D3-like stencil in A[1, :]
        A[1, :] = 0.0
        if n >= 4:
          # simple forward-biased third-derivative
          A[1, 3] = 1.0/(dz**3) * EI
          A[1, 2] = -3.0/(dz**3) * EI
          A[1, 1] = 3.0/(dz**3) * EI
          A[1, 0] = -1.0/(dz**3) * EI
        r[1] = lc.H_N

        # If moment is also applied at head, superimpose via curvature: EI*y''(0)=M
        if abs(lc.M_Nm) > 0.0:
          #Replace the M(0)=0 row (A[0, :]) with EI*y''(0)=M
          if n >= 3:
            A[0, :] = 0.0
            A[0, 0] = 1.0/(dz**2) * EI
            A[0, 1] = -2.0/(dz**2) * EI
            A[0, 2] = 1.0/(dz**2) * EI
            r[0] = lc.M_Nm

      # Merge rhs: residual + bc contributions (bc vector already matched rows)
      r = r + rhs_bc        

      # Solver for increment
      try:
        dy = np.linalg.solve(A, r)
      except np.linalg.LinAlgError:
        # fallback damped least squares if singular
        dy = np.linalg.lstsq(A + 1e-9*np.eye(n), r, rcond=None)[0]

      y += cfg.relax * dy

      if np.linalg.norm(dy, ord=np.inf) < cfg.tol:
        break
    
    # Post-processing
    theta = np.zeros(n)
    theta[1:-1] = (y[2:] - y[:-2]) / (2*dz)
    ypp = D2 @ y
    yppp = D3 @ y
    M = EI * ypp
    V = EI * yppp
    p = np.array([py_spring(y[i], z[i])[0] for i in range(n)])

    res = LateralResults(
      z_m=z, y_m=y, theta_rad=theta, M_Nm=M, V_N=V, p_N_per_m=p,
      head_deflection_mm=y[0]*1e3, head_rotation_mrad=theta[0]*1e3
    )
    results_per_step.append(res)
    head_curve.append((lc.H_N, float(y[0])))

    y_prev = y.copy()

  return{
    "steps": results_per_step,
    "head_curve": head_curve,
  }

