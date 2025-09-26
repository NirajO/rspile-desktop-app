""""
Soil-pile interaction curve generators used by the app.

- t-z (shaft friction vs axial slip)
- q-z (tip resistance vs axial slip at the toe)
- p-y (lateral soil reaction vs lateral deflection)

These follow common engineering practice (API RP 2GEO 2011/2014; Matlock?Reese style formulations). Values  and shapes are typical approximations used for teaching and quick studies.
"""

import math
import numpy as np

def get_tz_curve(layer, pile_diameter, depth):
  """
  Build a shaft resistane (t-z) curve follwoing API RP 2GEO ideas.

  layer: soil properties (clay uses su; sand uses phi)
  pile_diameter: pile diameter in meters
  depth: depth below ground in meters (used to estimate vertical stress)

  Returns two lists: z (m displacement) and t (kPa friction).
  """
  gamma = layer["gamma_kNpm3"]
  sigma_v = gamma * depth      # kPa
  z = []
  t = []
  if layer["type"] == "clay":
    su = layer["undrained_shear_strength_kPa"]
    if sigma_v > 0 and su > 0:
      psi = su / sigma_v
      alpha = 0.5 * psi ** -0.5 if psi <= 1 else 0.5 * psi ** -0.25
      alpha = min(alpha, 1.0)
    else:
      alpha = 1.0
    t_max = alpha * su

    # Normalized z/D points (API RP 2GEO clay, same as RP 2A)
    residual = 0.8
    z_D = [0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, 0.1]
    t_tmax = [0, 0.3, 0.5, 0.75, 0.9, 1.0, residual, residual]
    z = [xd * pile_diameter for xd in z_D]
    t = [tt * t_max for tt in t_tmax]

  elif layer["type"] == "sand":
    phi_deg = layer["phi_deg"]
    phi_rad = math.radians(phi_deg)
    K = 0.8    #Earth pressure coeff for open-ended driven pile
    delta = math.radians(phi_deg -5)  # Interface friction angle
    t_max = K * sigma_v * math.tan(delta)  #kPa, limit to ~100 kPa if needed
    # Nonlinear per RP 2GEO 2011 (more points than bilinear RP 2A)
    z_abs = [0, 0.00025, 0.001, 0.0025, 0.01, 0.025]
    t_tmax = [0.0, 0.10, 0.30, 0.50, 0.80, 1.00]
    t = [tt * t_max for tt in t_tmax]
    z = z_abs
  return z, t

def get_qz_curve(layer, pile_diameter, tip_depth):
  """
  Build a tip resistance (q-z) curve for the pile too.

  layer: soil at the pile top
  pile_diameter: pile diameter in meters
  tip_depth: tip depth in meters

  Returns two lists: z (m displacement) and q (kPa tip resistance)
  """
  gamma = layer["gamma_kNpm3"]
  sigma_v = gamma * tip_depth
  if layer["type"] == "clay":
    su = layer["undrained_shear_strength_kPa"]
    q_max = 9 * su
  
  elif layer["type"] == "sand":
    phi_deg = layer["phi_deg"]
    phi_rad = math.radians(phi_deg)
    Nq = math.exp(math.pi * (phi_rad - 0.75) / (1.75 + phi_rad))
    q_max = Nq * sigma_v  # Limit per API if >50 MPa
    q_max = min(q_max, 50000) # 50 MPa cap typical

  # Normalized z/D points (same for clay/sand per RP 2GEO)
  z_D = [0.0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
  q_qmax = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00, 1.00]
  z = [xd * pile_diameter for xd in z_D]
  q = [qq * q_max for qq in q_qmax]
  return z, q

def get_py_curve(layer, pile_diameter, depth):
  """
  Build a lateral reaction (p-y) curve inspired by API RP 2GEO:
  Matlock-style in clay and Reese-style in sand.

  layer: soil properties
  pile_diameter: pile diameter (m)
  depth: depth below ground (m)

  Returns arrays y(m) and p(kN/m)
  """
  gamma = layer["gamma_kNpm3"]
  y = np.linspace(0, 0.05 * pile_diameter, 100)
  p = np.zeros_like(y)
  if layer["type"] == "clay":
    su = layer["undrained_shear_strength_kPa"]
    if su <= 0:
      print(f"Warning: Zero or negative su ({su}) in layer at depth {depth}, using default behavior")
      su = 1.0  # Minimal default to avoid divison by zero
    epsilon50 = 0.02 if su < 24 else 0.005

    y_c = 2.5 * epsilon50 * pile_diameter # Critical disp
    J = 0.5 # Factor for battered piles; 0.5 typical
    Pu = (3 + J + (gamma * depth / su)) * su * pile_diameter # kN/m
    for i, yi in enumerate(y):
      if yi <= 8 * y_c:
        p[i] = 0.5 * Pu * (yi / y_c) ** (1/3) # Cubic initial
      else:
        p[i] = Pu # Ultimate

  elif layer["type"] == "sand":
    phi_deg = layer["phi_deg"]
    phi = math.radians(phi_deg)
    D = pile_diameter
    z = depth
    # Factors per API RP 2GEO sand
    k_p = (1 + math.sin(phi) / (1 - math.sin(phi))) # Passive coeff
    k_a = 1 / k_p
    k0 = 0.4 # At-rest
    alpha = phi / 2
    beta = math.radians(45 + phi_deg / 2)
    C1 = math.tan(beta) * (k_p * math.tan(alpha) + k0 * (math.tan(phi) * math.sin(beta) * (1/math.cos(alpha) + 1) - math.tan(alpha)))
    C2 = k_p - k_a
    C3 = 3 * k_p * k_p * math.sqrt(k_p + k0 * math.tan(phi)) / math.sqrt(k_p)
    
    # Ultimate Pu (kN/m)
    gamma_prime = gamma - 9.81 if gamma > 9.81 else gamma 
    Pu_us = (C1 * z + C2 * D) * gamma_prime * z
    Pu_ud = C3 * D * gamma_prime * z
    Pu = min(Pu_us, Pu_ud)

    # Initial modulus k (MN/m^3) per bhi (from API tables/approx)
    k = 5.4 * phi_deg ** 1.5 / 1000  # MPa/m approx; refine with table
    A = max(3 - 0.8 * z / D, 0.9) # Reduction factorD
    for i, yi in enumerate(y):
      p[i] = A * Pu * math.tanh((k * z / Pu) * yi)
  return y, p
