#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 19:51:55 2025

@author: jaydenwang
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta, qmc
import time

# ---------------------------
# Reproducibility
# ---------------------------
np.random.seed(42)  # affects numpy RNG; Sobol is separately seeded below

# ---------------------------
# Model parameters
# ---------------------------
d = 10
p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
x_d = norm.ppf(1 - p)                             # thresholds so that P(X >= x_d) = p

# Allow distinct loadings for default & LGD (vectorized; constant here for simplicity)
rho_D = np.sqrt(0.5) * np.ones(d)
rho_L = np.sqrt(0.5) * np.ones(d)

# LGD marginal: Beta(alpha, beta), coupled to Z_L via Gaussian copula
alpha_beta = (2, 5)

# Systemic factor correlation Corr(Z_D, Z_L)
rho_S = 0.5

# VaR levels provided
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
a_VaR_values = [1.143161852]

# ---------------------------
# CE / Simulation parameters
# ---------------------------
K = 10_000_000           # total paths in final estimator (use chunking)
P = int(0.2 * K)         # pilot sample size per CE run
repetitions = 10
rho_quantile = 0.5       # elite fraction ρ (use 0.1 or 0.2 for rarer events)
delta = 0.005            # band half-width for conditioning near VaR
max_iterations = 5       # CE iterations
eps_qmc = 1e-12          # clipping for U to avoid 0/1
ce_smoothing = 1.0       # optional smoothing μ_{t+1} = α*μ̂ + (1-α)*μ_t

# Sobol sequence (global stream)
# total dimension = 2 systemic + 2*d idiosyncratic
sobol_engine = qmc.Sobol(d=2 + 2 * d, scramble=True, seed=0)

# Systemic covariance, inverse, and Cholesky
Sigma = np.array([[1.0, rho_S],
                  [rho_S, 1.0]], dtype=float)
Sigma_inv = np.linalg.inv(Sigma)
L_chol = np.linalg.cholesky(Sigma)  # lower-triangular


def cross_entropy_varc(alpha, var, P=P, K=K, repetitions=repetitions,
                       rho_quantile=rho_quantile, delta=delta,
                       max_iterations=max_iterations, ce_smoothing=ce_smoothing):
    """
    CE-based VaR contribution estimator aligned with the Simulation–Analytical description:
      • CE stage tilts the sampling distribution ONLY by shifting the mean of (Z_D, Z_L) (same covariance Σ)
      • CE update uses ELITE SAMPLES (indicator-only) to compute the MLE of the tilted mean (no LR in CE update)
      • Final estimator uses importance weights (likelihood ratio from base N(0,Σ) to tilted N(μ*,Σ))
      • Conditioning on L ∈ [VaR−δ, VaR+δ] to approximate E[LGD_i * D_i | L ≈ VaR]

    Returns:
      varc_mean: (d,) mean VaR contributions
      varc_se:   (d,) standard error over repetitions
      samples_mean: average #samples falling in the VaR band per repetition
    """
    start_total_time = time.time()

    # ---------- CE stage: optimize μ = (μ_D, μ_L) ----------
    mu = np.zeros(2, dtype=float)            # start at 0
    sL = np.sqrt(1.0 - rho_L**2)             # (d,)
    sD = np.sqrt(1.0 - rho_D**2)             # (d,)

    for t in range(max_iterations):
        # Pilot sampling under current μ
        U = sobol_engine.random(P)
        U = np.clip(U, eps_qmc, 1.0 - eps_qmc)
        Z_std = norm.ppf(U)                  # standard normals

        # First two columns -> independent normals, then correlate + shift
        gZ   = Z_std[:, :2]                  # (P, 2)
        Zsys = gZ @ L_chol.T + mu            # (P, 2), ~ N(μ, Σ)
        Z_D  = Zsys[:, 0]
        Z_L  = Zsys[:, 1]

        # Idiosyncratic parts (independent)
        eta_D = Z_std[:, 2:2 + d]            # (P, d)
        eta_L = Z_std[:, 2 + d:2 + 2*d]      # (P, d)

        # Defaults & LGDs
        X = rho_D[None, :] * Z_D[:, None] + sD[None, :] * eta_D   # (P, d)
        Y = rho_L[None, :] * Z_L[:, None] + sL[None, :] * eta_L   # (P, d)

        D = (X >= x_d[None, :]).astype(float)
        U_beta = norm.cdf(Y)
        U_beta = np.clip(U_beta, eps_qmc, 1.0 - eps_qmc)
        EPS = beta.ppf(U_beta, a=alpha_beta[0], b=alpha_beta[1])

        losses = np.sum(EPS * D, axis=1)                         # (P,)

        # Elite threshold γ_t = (1 - ρ) quantile under current μ
        gamma_t = np.quantile(losses, 1.0 - rho_quantile)

        # If elites already sit above target VaR, stop CE
        if gamma_t >= var:
            break

        # Elite indicator (NO LR inside CE update)
        elite_mask = (losses >= gamma_t)
        if not np.any(elite_mask):
            # no elites: keep μ unchanged and stop
            print(f"[CE] iter={t}: no elite samples; keep μ={mu}")
            break

        # MLE of mean for Gaussian with known Σ is just the (weighted) sample mean over elites
        mu_hat = Zsys[elite_mask].mean(axis=0)           # (2,)
        mu = ce_smoothing * mu_hat + (1.0 - ce_smoothing) * mu

    mu_star = mu.copy()

    # Precompute LR constants for final stage
    v_star = Sigma_inv @ mu_star                     # (2,)
    quad_star = 0.5 * (mu_star @ (Sigma_inv @ mu_star))

    # ---------- Final IS estimation (repetitions for SE) ----------
    varc_reps = np.zeros((repetitions, d))
    samples_counts = []

    # Chunking to control memory
    chunk_size = int(2_000_000)
    chunk_size = min(chunk_size, K)

    for rep in range(repetitions):
        t0 = time.time()
        denom_total = 0.0
        numerators = np.zeros(d, dtype=float)
        in_band_count = 0

        drawn = 0
        while drawn < K:
            n = min(chunk_size, K - drawn)
            U = sobol_engine.random(n)
            U = np.clip(U, eps_qmc, 1.0 - eps_qmc)
            Z_std = norm.ppf(U)

            # Systemic sampling under tilted μ*
            gZ   = Z_std[:, :2]
            Zsys = gZ @ L_chol.T + mu_star         # (n, 2)
            Z_D  = Zsys[:, 0]
            Z_L  = Zsys[:, 1]

            # Idiosyncratic parts remain standard normal
            eta_D = Z_std[:, 2:2 + d]
            eta_L = Z_std[:, 2 + d:2 + 2*d]

            # Build X, Y, defaults & LGDs
            X = rho_D[None, :] * Z_D[:, None] + sD[None, :] * eta_D
            Y = rho_L[None, :] * Z_L[:, None] + sL[None, :] * eta_L

            D = (X >= x_d[None, :]).astype(float)
            U_beta = norm.cdf(Y)
            U_beta = np.clip(U_beta, eps_qmc, 1.0 - 1e-12)
            EPS = beta.ppf(U_beta, a=alpha_beta[0], b=alpha_beta[1])

            losses = np.sum(EPS * D, axis=1)

            # LR from g=N(μ*,Σ) back to base f=N(0,Σ): exp(-μ*^T Σ^{-1} Z + 0.5 μ*^T Σ^{-1} μ*)
            LR = np.exp(-(Zsys @ v_star) + quad_star)   # (n,)

            # Select samples in VaR band
            mask = (losses >= var - delta) & (losses <= var + delta)
            if np.any(mask):
                w = LR[mask]
                denom_total += w.sum()
                numerators += (w[:, None] * (D[mask, :] * EPS[mask, :])).sum(axis=0)
                in_band_count += int(mask.sum())

            drawn += n

        samples_counts.append(in_band_count)
        if denom_total > 0:
            varc_reps[rep, :] = numerators / denom_total
        else:
            varc_reps[rep, :] = np.nan

        print(f"[Final] rep {rep+1}/{repetitions}: in-band={in_band_count}, time={time.time()-t0:.2f}s")

    # Aggregate stats
    varc_mean = np.nanmean(varc_reps, axis=0)
    varc_se   = np.nanstd(varc_reps, axis=0, ddof=1) / np.sqrt(repetitions)
    samples_mean = float(np.mean(samples_counts))
    print(f"Alpha {alpha}: μ*={mu_star}, avg in-band={samples_mean:.1f}, total time={time.time()-start_total_time:.2f}s")

    return varc_mean, varc_se, samples_mean


# ---------------------------
# Run across VaR levels
# ---------------------------
index_names = []
for alpha in alpha_values:
    pct = int(alpha * 100)
    index_names.extend([f'{pct}% VaRC', f'{pct}% VaRC SE', f'{pct}% Samples'])

col_names = [f'Obligor {i+1}' for i in range(d)]
results_df = pd.DataFrame(index=index_names, columns=col_names, dtype=float)

for alpha, var in zip(alpha_values, a_VaR_values):
    varc_mean, varc_se, samples = cross_entropy_varc(alpha, var)
    pct = int(alpha * 100)
    results_df.loc[f'{pct}% VaRC'] = varc_mean
    results_df.loc[f'{pct}% VaRC SE'] = varc_se
    results_df.loc[f'{pct}% Samples'] = samples

print("\nResults:")
print(results_df)

# Save to CSV
# results_df.to_csv('RhoS=0.5 VaRC CE.csv', index=True)
