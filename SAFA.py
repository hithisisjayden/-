import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import time

np.random.seed(42)

# -------------------------
# Parameters
# -------------------------
N = 10                                # Portfolio size
rho_S = 0.5                           # Corr(Z_D, Z_L)
rho_D = np.sqrt(0.5) * np.ones(N)     # Default factor loadings
rho_L = np.sqrt(0.5) * np.ones(N)     # LGD factor loadings
alpha_beta = 2                        # Beta(alpha, beta) for LGD
beta_beta  = 5
p = np.array([0.01 * (i+1) for i in range(N)])  # Unconditional PDs: 1%..10%
x_d = norm.ppf(1 - p)                              # Default thresholds

# VaR(a) for alpha in [0.95, 0.96, 0.97, 0.98, 0.99]
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
# a_VaR_values = [1.143161852]

# Simulation params
N_outer = 1000     # outer sims over (Z_D, Z_L)
N_inner = 1000     # inner sims for L^{-i} and ε_i | Z_L
reps    = 10       # repetitions for mean/SE


# -------------------------
# SAFA (Simulation-Analytical) — vectorized & corrected
# -------------------------
def compute_safa_varc(a_VaR_values, N_outer, N_inner, reps):
    """
    Vectorized, PDF-consistent SAFA:
    - Z sampled as (Z_D, Z_L)
    - For each outer draw, per-obligor ε_i|Z_L density/derivative computed correctly,
      but in a single (N, N_inner) vectorized pass
    - Boundary term vectorized across i
    - Only per-row searchsorted remains in a small Python loop
    """
    # Common-factor covariance; Z = (Z_D, Z_L) in this order
    cov_matrix = np.array([[1.0, rho_S],
                           [rho_S, 1.0]])

    # Precompute s_vec = sqrt(1 - rho_L^2) for broadcasting
    s_vec = np.sqrt(1.0 - rho_L**2)  # shape (N,)

    results = []

    for a in a_VaR_values:
        VaRC_reps = []
        rep_times = []
        total_start = time.time()

        for r in range(reps):
            rep_start = time.time()

            # Numerator A_i(a) accumulated over outer sims
            A = np.zeros(N)

            for _ in range(N_outer):
                # ----- Sample common factors Z = (Z_D, Z_L) -----
                Z = np.random.multivariate_normal([0.0, 0.0], cov_matrix)
                z_D, z_L = Z[0], Z[1]   # IMPORTANT: order consistent with model

                # ----- Simulate all obligors' losses to build L_total and L^{-i} -----
                # LGD shocks for all obligors and inner paths
                eta_L = np.random.randn(N, N_inner)
                # Default shocks for all obligors and inner paths
                eta_D = np.random.randn(N, N_inner)

                # ε_j | Z_L  (N, N_inner)
                Y = rho_L[:, None] * z_L + s_vec[:, None] * eta_L
                U = norm.cdf(Y)
                U = np.clip(U, 1e-16, 1 - 1e-16)
                epsilon = beta.ppf(U, alpha_beta, beta_beta)          # (N, N_inner)
                epsilon = np.clip(epsilon, 1e-12, 1 - 1e-12)

                # D_j | Z_D  (N, N_inner)
                X = rho_D[:, None] * z_D + np.sqrt(1 - rho_D[:, None]**2) * eta_D
                D = (X > x_d[:, None]).astype(float)

                # Loss matrix and totals
                losses  = epsilon * D                                 # (N, N_inner)
                L_total = np.sum(losses, axis=0)                      # (N_inner,)
                L_minus = L_total[None, :] - losses                   # (N, N_inner)
                sorted_L_minus = np.sort(L_minus, axis=1)             # sort each row for search

                # ----- Conditional PDs p_i(Z_D) -----
                arg    = (x_d - rho_D * z_D) / np.sqrt(1 - rho_D**2)
                p_cond = 1.0 - norm.cdf(arg)                          # shape (N,)
                prod_all = np.prod(1.0 - p_cond)                      # for boundary reuse

                # ----- Draw ε_i | Z_L for ALL i at once (vectorized) -----
                Eta_i = np.random.randn(N, N_inner)
                Y_i   = rho_L[:, None] * z_L + s_vec[:, None] * Eta_i
                U_i   = norm.cdf(Y_i)
                U_i   = np.clip(U_i, 1e-16, 1 - 1e-16)
                eps_i = beta.ppf(U_i, alpha_beta, beta_beta)          # (N, N_inner)
                eps_i = np.clip(eps_i, 1e-12, 1 - 1e-12)

                # ----- f_{ε_i|Z_L}(e) and f'_{ε_i|Z_L}(e) for all i (vectorized) -----
                f_beta_vals = beta.pdf(eps_i, alpha_beta, beta_beta)  # (N, N_inner)
                u_mat = beta.cdf(eps_i, alpha_beta, beta_beta)
                v_mat = norm.ppf(np.clip(u_mat, 1e-16, 1 - 1e-16))
                w_mat = (v_mat - rho_L[:, None] * z_L) / s_vec[:, None]
                phi_v = norm.pdf(v_mat)
                phi_w = norm.pdf(w_mat)

                # f_cond = f_beta * phi_w / (phi_v * s_i)
                denom = phi_v * s_vec[:, None]
                f_cond = np.where(phi_v > 1e-300, f_beta_vals * phi_w / denom, 0.0)

                # term1 = f_beta/phi_v * (v - w/s), term2 = (α-1)/e - (β-1)/(1-e)
                term1 = np.where(phi_v > 1e-300,
                                 f_beta_vals / phi_v * (v_mat - w_mat / s_vec[:, None]),
                                 0.0)
                term2 = (alpha_beta - 1.0) / eps_i - (beta_beta - 1.0) / (1.0 - eps_i)

                f_prime = f_cond * (term1 + term2)

                # g_i(e) = 1 + e * f'/f
                g_mat = np.ones_like(eps_i)
                good  = f_cond > 1e-300
                g_mat[good] = 1.0 + eps_i[good] * (f_prime[good] / f_cond[good])

                # ----- Boundary term for a < 1 (vectorized over i) -----
                boundary = np.zeros(N)
                if a < 1.0:
                    e_a  = np.clip(a, 1e-12, 1 - 1e-12)
                    f_ba = beta.pdf(e_a, alpha_beta, beta_beta)
                    u_a  = beta.cdf(e_a, alpha_beta, beta_beta)
                    v_a  = norm.ppf(np.clip(u_a, 1e-16, 1 - 1e-16))
                    # For each i:
                    w_a  = (v_a - rho_L * z_L) / s_vec
                    phi_v_a = norm.pdf(v_a)                # scalar
                    phi_w_a = norm.pdf(w_a)                # (N,)
                    f_cond_a_vec = np.where(phi_v_a > 1e-300,
                                            f_ba * phi_w_a / (phi_v_a * s_vec),
                                            0.0)
                    denom_vec     = np.maximum(1.0 - p_cond, 1e-300)   # (N,)
                    prod_except   = prod_all / denom_vec               # (N,)
                    boundary      = - a * f_cond_a_vec * prod_except   # (N,)

                # ----- Sort-Search CDF approx of L^{-i} at (a - e) -----
                inner_vals = np.zeros(N)
                thresholds_mat = a - eps_i                             # (N, N_inner)

                # Only this small loop remains (per-row searchsorted)
                for i in range(N):
                    F = np.searchsorted(sorted_L_minus[i], thresholds_mat[i], side='right') / N_inner
                    inner_vals[i] = np.mean(F * g_mat[i]) + boundary[i]

                # Accumulate: A_i(z) = p_i(z_D) * inner_vals[i]
                A += p_cond * inner_vals

            # Average over outer sims
            A /= N_outer

            # Denominator via full-allocation identity: f_L(a) = (Σ_i A_i(a)) / a
            sum_A = np.sum(A)
            f_L_a = sum_A / a if a != 0 else 0.0

            VaRC = A / f_L_a if f_L_a > 0 else np.zeros(N)
            VaRC_reps.append(VaRC)

            rep_times.append(time.time() - rep_start)
            print(f"Rep {r+1} for VaR={a}: Time = {rep_times[-1]:.2f}s")

        total_time = time.time() - total_start

        VaRC_array = np.array(VaRC_reps)                 # (reps, N)
        mean_VaRC  = np.mean(VaRC_array, axis=0)
        se_VaRC    = np.std(VaRC_array, axis=0, ddof=1) / np.sqrt(reps)

        results.append({
            'a': a,
            'mean_VaRC': mean_VaRC,
            'se_VaRC': se_VaRC,
            'rep_times': rep_times,
            'total_time': total_time
        })

    return results


# -------------------------
# Run & report
# -------------------------
safa_results = compute_safa_varc(a_VaR_values, N_outer, N_inner, reps)

# Collect results into a tidy table
VaRCs   = [r['mean_VaRC'] for r in safa_results]
VaRC_SE = [r['se_VaRC'] for r in safa_results]
CPUs    = [r['total_time'] for r in safa_results]

Risk_Contributions = pd.DataFrame({
    'VaRC 0.95'       : VaRCs[0],
    'VaRC S.E. 0.95'  : VaRC_SE[0],
    'VaRC 0.95 CPU'   : CPUs[0],

    'VaRC 0.96'       : VaRCs[1],
    'VaRC S.E. 0.96'  : VaRC_SE[1],
    'VaRC 0.96 CPU'   : CPUs[1],

    'VaRC 0.97'       : VaRCs[2],
    'VaRC S.E. 0.97'  : VaRC_SE[2],
    'VaRC 0.97 CPU'   : CPUs[2],

    'VaRC 0.98'       : VaRCs[3],
    'VaRC S.E. 0.98'  : VaRC_SE[3],
    'VaRC 0.98 CPU'   : CPUs[3],

    'VaRC 0.99'       : VaRCs[4],
    'VaRC S.E. 0.99'  : VaRC_SE[4],
    'VaRC 0.99 CPU'   : CPUs[4],
}, index=pd.Index([f'Obligor {i+1}' for i in range(N)])).T

# Risk_Contributions.to_csv('RhoS=0.5 VaRC SAFA.csv')
print(Risk_Contributions)
