
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
  FREE_HEAD = "free_head"
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
  bc: BCType = BCType.FREE_HEAD
  max_iters: int = 40
  tol: float = 1e-6
  relax: float = 0.5    #0.5-1.0 helps convergence in very soft soils

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

def apply_boundary_conditions(D2: np.ndarray, D3: np.ndarray, A: np.ndarray, y: np.ndarray, pile: PileProps, cfg: LateralConfig, lc: LateralLoadCase, EI, dz) -> np.ndarray:
  """
  Enforce head/tip BCs on matrices by replacing corresponding rows.
  Head (z=0):
    - FREE_HEAD handled in the caller (latera_analysis) so we do nothing here for head.
    - FIXED_HEAD: y(0)=0 and y'(0)=0
  Tipe (z=L): free tip -> M(L)=0 and V(L)=0
  """
  n = y.size
  rhs_bc = np.zeros(n)

  #---Head BC if FIXED_HEAD-----
  if cfg.bc == BCType.FIXED_HEAD:
    # y(0) = 0 -> dy(0) = -y(0)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    rhs_bc[0] = -float(y[0])

    #y('(0)=0 -> (D1@dy)[0] = -(D1@y)[0]
    if n >= 2:
      A[1, :] = 0.0
      A[1, 0] = -1.0 / dz
      A[1, 1] = 1.0 / dz
      rhs_bc[1] = -float((y[1] - y[0]) / dz)

  #---- Tip (free): M(L)=0 and V(L)=0------
  i = n - 1

  # Enforce y''(L)=0 (curvature -> moment zero)
  if n >= 3:
    A[i, :] = EI * D2[i, :]
    rhs_bc[i] = -float(EI * (D2 @ y)[i])

  # Enforce y'''(L)=0 (shear zero) at the row i-1
  if n >= 4:
    A[i-1, :] = EI * D3[i-1, :]
    rhs_bc[i-1] = -float(EI * (D3 @ y)[i-1])

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
  y_prev = 1e-6 * np.exp(-z / max(1e-9, 0.2 * L))

  results_per_step: list[LateralResults] = []
  head_curve: list[tuple[float, float]] = []

  for _, lc in enumerate(load_steps, 1):
    y = y_prev.copy()
    converged = False

    for _ in range(cfg.max_iters):
      # Soil reaction and tangent at current y
      p = np.zeros(n)
      k = np.zeros(n)
      for i in range(n):
        p[i], k[i] = py_spring(y[i], z[i])
      p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
      k = np.nan_to_num(k, nan=1e5, posinf=1e9, neginf=1e5)
      k = np.clip(k, 1e3, 1e9)

      # Newton system: (EI*D4 - diag(k)) dy = p - EI*D4*y
      A = EI * D4 - np.diag(k)
      r = p - EI * (D4 @ y) 

      # Boundary conditions
      rhs_bc = apply_boundary_conditions(D2, D3, A, y, pile, cfg, lc, EI, dz)

      # Apply BC residual: where rhs_bc is non-zero, override the PDE residual
      bc_mask = rhs_bc != 0.0
      r[bc_mask] = rhs_bc[bc_mask] 
      
      
      # Inject head loads for free-head case
      if cfg.bc == BCType.FREE_HEAD:
        D2row0 = D2[0, :]
        D3row0 = D3[0, :]

        if abs(lc.H_N) < 1e-6 and abs(lc.M_Nm) < 1e-9:
          A[0, :] = 0.0
          A[0, 0] = 1.0
          r[0] = -float(y[0])
        else:
          A[0, :] = EI * D3row0
          r[0] = float(lc.H_N - EI * (D3row0 @ y))

          if abs(lc.M_Nm) > 0.0 and n >= 2:
            A[1, :] = EI * D2row0
            r[1] = float(lc.M_Nm - EI * (D2row0 @ y))

      A = np.nan_to_num (A, nan=0.0, posinf=0.0, neginf=0.0)
      r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
      A += (1e-9 * EI) * np.eye(n)      

      # Solver for increment
      try:
        dy = np.linalg.solve(A, r)
      except np.linalg.LinAlgError:
        # fallback damped least squares if singular
        dy = np.linalg.lstsq(A, r, rcond=None)[0]

      if not np.all(np.isfinite(dy)):
        break

      y_new = y + cfg.relax * dy

      if np.max(np.abs(y_new)) > 0.5 * pile.d_m:
        break

      y_new = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
      # y = np.clip(y, -0.5 * pile.d_m, 0.5 * pile.d_m)

      if np.linalg.norm(dy, ord=np.inf) < cfg.tol * max(1e-5, np.max(np.abs(y_new))):
        y = y_new
        converged = True
        break

      y = y_new

      if not converged:
        continue
    
    # Post-processing
    theta = np.zeros(n)
    if n >= 2:
      theta[0] = (y[1] - y[0]) / dz
      theta[-1] = (y[-1] - y[-2]) / dz
    if n >= 3:
      theta[1:-1] = (y[2:] - y[:-2]) / (2.0 * dz)

    ypp = D2 @ y
    yppp = D3 @ y
    M = EI * ypp
    V = EI * yppp
    p_out = np.array([py_spring(y[i], z[i])[0] for i in range(n)])

    res = LateralResults(
      z_m=z, y_m=y, theta_rad=theta, M_Nm=M, V_N=V, p_N_per_m=p_out,
      head_deflection_mm=float(y[0]*1e3), head_rotation_mrad=float(theta[0]*1e3)
    )
    results_per_step.append(res)
    head_curve.append((lc.H_N, float(y[0])))

    y_prev = y.copy()

  return{
    "steps": results_per_step,
    "head_curve": head_curve,
    "meta": {"EI_Nm2": EI, "length_m": L, "n_nodes": n}
  }


