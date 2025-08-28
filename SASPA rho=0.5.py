import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import time

np.random.seed(42)

# -------------------------
# Model parameters
# -------------------------
N = 10
rho_S = 0.5
rho_D = np.sqrt(0.5) * np.ones(N)
rho_L = np.sqrt(0.5) * np.ones(N)
alpha_beta = 2
beta_beta  = 5
p = np.array([0.01*(i+1) for i in range(N)])
x_d = norm.ppf(1 - p)

alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
# a_VaR_values = [1.143161852]

# SAFA sim params
N_outer = 1000
N_inner = 1000
reps = 1

# Quadrature params
N_gh = 24
N_gl = 64

# -------------------------
# Numerical quadrature helpers
# -------------------------
def gauss_legendre_on_01(n):
    x, w = leggauss(n)
    e = 0.5 * (x + 1.0)
    w = 0.5 * w
    return e, w

gh_x, gh_w = hermgauss(N_gh)

# -------------------------
# Conditional density f_{eps|ZL}(e; zL)
# -------------------------
def f_eps_cond(e, zL, rhoL, a_beta, b_beta):
    e = np.clip(e, 1e-14, 1 - 1e-14)
    f_b = beta.pdf(e, a_beta, b_beta)
    u = beta.cdf(e, a_beta, b_beta)
    u = np.clip(u, 1e-16, 1.0 - 1e-16)  # 防止 norm.ppf 溢出
    v = norm.ppf(u)
    s = np.sqrt(max(1.0 - rhoL**2, 1e-14))
    w = (v - rhoL * zL) / s
    num = norm.pdf(w)
    denom = norm.pdf(v) * s
    dens = np.where(denom > 1e-300, f_b * num / denom, 0.0)
    return dens

def M_eps_and_derivs(t, zL, rhoL, a_beta, b_beta, n_gl=N_gl):
    e_nodes, e_weights = gauss_legendre_on_01(n_gl)
    fvals = f_eps_cond(e_nodes, zL, rhoL, a_beta, b_beta)
    t_clipped = np.clip(t, -700, 700)
    et = np.exp(t_clipped * e_nodes)
    M0 = np.sum(e_weights * et * fvals)
    M1 = np.sum(e_weights * e_nodes * et * fvals)
    M2 = np.sum(e_weights * (e_nodes**2) * et * fvals)
    M3 = np.sum(e_weights * (e_nodes**3) * et * fvals)
    M4 = np.sum(e_weights * (e_nodes**4) * et * fvals)
    return M0, M1, M2, M3, M4

# -------------------------
# Ai and its derivatives
# -------------------------
def A_i_and_derivs(t, zD, zL, rhoDi, rhoLi, xdi, a_beta, b_beta):
    sD = np.sqrt(max(1.0 - rhoDi**2, 1e-14))
    arg = (xdi - rhoDi * zD) / sD
    pz = 1.0 - norm.cdf(arg)
    M0, M1, M2, M3, M4 = M_eps_and_derivs(t, zL, rhoLi, a_beta, b_beta)
    A0 = 1.0 - pz + pz * M0
    A1 = pz * M1
    A2 = pz * M2
    A3 = pz * M3
    A4 = pz * M4
    return A0, A1, A2, A3, A4

# -------------------------
# K and derivatives (Numerically Stabilized)
# -------------------------
def K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta):
    A0s = np.empty(len(xds))
    A1s = np.empty(len(xds))
    A2s = np.empty(len(xds))
    A3s = np.empty(len(xds))
    A4s = np.empty(len(xds))
    for i in range(len(xds)):
        A0, A1, A2, A3, A4 = A_i_and_derivs(t, zD, zL, rhoD_vec[i], rhoL_vec[i], xds[i], a_beta, b_beta)
        A0s[i] = max(A0, 1e-300)
        A1s[i] = A1
        A2s[i] = A2
        A3s[i] = A3
        A4s[i] = A4

    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.nan_to_num(A1s / A0s)
        r2 = np.nan_to_num(A2s / A0s)
        r3 = np.nan_to_num(A3s / A0s)
        r4 = np.nan_to_num(A4s / A0s)

    K0 = np.sum(np.log(A0s))
    K1 = np.sum(r1)
    K2 = np.sum(r2 - r1**2)
    K3 = np.sum(r3 - 3 * r1 * r2 + 2 * r1**3)
    K4 = np.sum(r4 - 4 * r1 * r3 - 3 * r2**2 + 12 * (r1**2) * r2 - 6 * r1**4)
    return K0, K1, K2, K3, K4

# -------------------------
# Solve saddlepoint K'(t)=a (Robust version)
# -------------------------
def solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, t0=0.1, max_iter=60):
    t = t0
    for _ in range(max_iter):
        try:
            _, K1, K2, _, _ = K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        except (OverflowError, ValueError):
            break
        diff = K1 - a_target
        if abs(diff) < 1e-11:
            return t
        if K2 <= 1e-12:
            break
        step = np.sign(diff) * min(abs(diff / K2), 1.0)
        t_new = t - step
        if not np.isfinite(t_new):
            break
        t = t_new

    lo, hi = -50.0, 50.0
    try:
        _, K1_lo, _, _, _ = K_and_derivs(lo, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        if K1_lo > a_target:
            return lo
        _, K1_hi, _, _, _ = K_and_derivs(hi, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        if K1_hi < a_target:
            return hi
    except (OverflowError, ValueError):
        return t0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        try:
            _, K1_mid, _, _, _ = K_and_derivs(mid, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
            if np.isnan(K1_mid):
                hi = mid if a_target < K1_mid else lo
                continue
            if K1_mid > a_target:
                hi = mid
            else:
                lo = mid
        except (OverflowError, ValueError):
            hi = mid
    return 0.5 * (lo + hi)

# -------------------------
# Conditional SPA density
# -------------------------
def spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta):
    t_hat = solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, t0=0.1)
    K0, _, K2, K3, K4 = K_and_derivs(t_hat, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
    if K2 <= 1e-12:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lam3 = np.nan_to_num(K3 / (K2**1.5))
        lam4 = np.nan_to_num(K4 / (K2**2))
    pref = np.exp(K0 - t_hat * a_target) / np.sqrt(2.0 * np.pi * K2)
    corr = 1.0 + 0.125 * (lam4 - (5.0 / 3.0) * (lam3**2))
    return max(0.0, pref * corr)

# -------------------------
# Unconditional f_L(a) via GH
# -------------------------
def f_L_via_SPA_correlated(a_target, rho_S, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, n_gh=N_gh):
    Sigma = np.array([[1.0, rho_S], [rho_S, 1.0]])
    L = np.linalg.cholesky(Sigma)
    x_nodes, w_nodes = hermgauss(n_gh)
    total = 0.0
    factor = 1.0 / np.pi  # 2D GH, 1/(√π)^2 = 1/π
    for i in range(n_gh):
        for j in range(n_gh):
            xvec = np.array([x_nodes[i], x_nodes[j]])
            z = np.sqrt(2.0) * (L @ xvec)
            zD, zL = z[0], z[1]
            w = w_nodes[i] * w_nodes[j] * factor
            dens = spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
            if np.isfinite(dens):
                total += w * dens
    return total

# -------------------------
# SAFA numerator (Corrected per-obligor LGD density/derivative)
# -------------------------
def compute_safa_numerator(N_outer, N_inner, rho_S, rhoD_vec, rhoL_vec, a_target, a_beta, b_beta, xds):
    cov_matrix = np.array([[1.0, rho_S], [rho_S, 1.0]])
    N = len(xds)
    A_accum = np.zeros(N)

    for _ in range(N_outer):
        Z = np.random.multivariate_normal([0.0, 0.0], cov_matrix)

        # ---- 统一顺序：zD 控 PD，zL 控 LGD（与分母一致）----
        zD, zL = Z[0], Z[1]

        # 核心模拟：对所有 obligor 生成 (ε_i, D_i)，用于 L_total 与 L^{-i}
        eta_L = np.random.randn(N, N_inner)
        eta_D = np.random.randn(N, N_inner)

        Y = rhoL_vec[:, None] * zL + np.sqrt(1 - rhoL_vec[:, None]**2) * eta_L
        U = norm.cdf(Y)
        eps = beta.ppf(np.clip(U, 1e-16, 1 - 1e-16), a_beta, b_beta)

        X = rhoD_vec[:, None] * zD + np.sqrt(1 - rhoD_vec[:, None]**2) * eta_D
        D = (X > xds[:, None]).astype(float)

        losses = eps * D
        L_total = np.sum(losses, axis=0)                 # shape: (N_inner,)
        L_minus = L_total[None, :] - losses              # shape: (N, N_inner)
        sorted_L_minus = np.sort(L_minus, axis=1)        # 每个 i 对应一行排序好的 L^{-i}

        # 条件违约概率 p_i(zD)
        arg = (xds - rhoD_vec * zD) / np.sqrt(1 - rhoD_vec**2)
        p_cond = 1.0 - norm.cdf(arg)
        prod_all = np.prod(1.0 - p_cond)                 # 供边界项复用

        inner_vals = np.zeros(N)

        # ---- 逐个 i 计算（密度、导数、g、边界）----
        for i in range(N):
            # 针对第 i 个 obligor，从 f_{ε_i|Z_L} 采样
            eta_L_i = np.random.randn(N_inner)
            Yi = rhoL_vec[i] * zL + np.sqrt(1 - rhoL_vec[i]**2) * eta_L_i
            Ui = norm.cdf(Yi)
            eps_smp = beta.ppf(np.clip(Ui, 1e-16, 1 - 1e-16), a_beta, b_beta)
            eps_smp = np.clip(eps_smp, 1e-12, 1 - 1e-12)

            # f_{ε_i|Z_L} 与其导数 f'
            f_beta_vals = beta.pdf(eps_smp, a_beta, b_beta)
            u = beta.cdf(eps_smp, a_beta, b_beta)
            v = norm.ppf(np.clip(u, 1e-16, 1 - 1e-16))
            s = np.sqrt(1 - rhoL_vec[i]**2)
            w = (v - rhoL_vec[i] * zL) / s
            phi_v = norm.pdf(v)
            phi_w = norm.pdf(w)

            f_cond = np.zeros_like(eps_smp)
            mask = phi_v > 1e-300
            f_cond[mask] = f_beta_vals[mask] * phi_w[mask] / (phi_v[mask] * s)

            term1 = np.zeros_like(eps_smp)
            term1[mask] = f_beta_vals[mask] / phi_v[mask] * (v[mask] - w[mask] / s)
            term2 = (a_beta - 1) / eps_smp - (b_beta - 1) / (1 - eps_smp)
            f_prime = f_cond * (term1 + term2)

            # g(e) = 1 + e * f'/f
            g = np.ones_like(eps_smp)
            good = f_cond > 1e-300
            g[good] = 1.0 + eps_smp[good] * (f_prime[good] / f_cond[good])

            # Sort-Search 逼近 F_{L^{-i}}(a - e)
            thresholds = a_target - eps_smp
            F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner

            # 边界项（a < 1 时）
            boundary_i = 0.0
            if a_target < 1.0:
                ei_a = np.clip(a_target, 1e-12, 1 - 1e-12)
                f_beta_a = beta.pdf(ei_a, a_beta, b_beta)
                u_a = beta.cdf(ei_a, a_beta, b_beta)
                v_a = norm.ppf(np.clip(u_a, 1e-16, 1 - 1e-16))
                sL = np.sqrt(1 - rhoL_vec[i]**2)
                w_a = (v_a - rhoL_vec[i] * zL) / sL
                phi_v_a = norm.pdf(v_a)
                phi_w_a = norm.pdf(w_a)
                f_cond_a = f_beta_a * phi_w_a / (phi_v_a * sL) if phi_v_a > 1e-300 else 0.0

                # ∏_{j≠i} (1 - p_j(zD))
                denom_i = max(1.0 - p_cond[i], 1e-300)
                prod_except_i = prod_all / denom_i

                boundary_i = - a_target * f_cond_a * prod_except_i

            inner_vals[i] = np.mean(F * g) + boundary_i

        # 汇总：A_i(z) = p_i(zD) * E[ F_{L^{-i}}(a - ε_i) * g_i(ε_i) ] + 边界
        A_accum += (p_cond * inner_vals)

    return A_accum / N_outer

# -------------------------
# Main SASPA runner 
# -------------------------
def compute_saspa(a_VaR_values, use_spa=True):
    results = []
    den_time_taken = 0.0  # 初始化
    for a in a_VaR_values:
        VaRC_reps = []
        rep_times = []
        print(f"\n--- Starting computations for a_VaR = {a} ---")

        # 分母：SPA（对每个 a 只算一次）
        f_den_spa = 0.0
        if use_spa:
            den_time_start = time.time()
            f_den_spa = f_L_via_SPA_correlated(a, rho_S, rho_D, rho_L, x_d, alpha_beta, beta_beta, n_gh=N_gh)
            den_time_end = time.time()
            den_time_taken = den_time_end - den_time_start
            print(f"SPA Denominator f_L(a) = {f_den_spa:.6g} (calculated once in {den_time_taken:.2f}s)")

        # Monte Carlo 分子
        for r in range(reps):
            t0 = time.time()
            numer = compute_safa_numerator(N_outer, N_inner, rho_S, rho_D, rho_L, a, alpha_beta, beta_beta, x_d)
            f_den = f_den_spa if use_spa else np.sum(numer) / max(a, 1e-16)
            VaRC = numer / max(f_den, 1e-300)
            VaRC_reps.append(VaRC)
            t1 = time.time()
            rep_times.append(t1 - t0)
            print(f"Rep {r+1}/{reps} for a_VaR={a}: Numerator calculated in {t1 - t0:.2f}s")

        VaRC_reps = np.array(VaRC_reps)
        results.append({
            'a': a,
            'mean_VaRC': VaRC_reps.mean(axis=0),
            'se_VaRC': VaRC_reps.std(axis=0, ddof=1) / np.sqrt(len(VaRC_reps)),
            'rep_times': rep_times,
            'total_time': np.sum(rep_times) + (den_time_taken if use_spa else 0)
        })
    return results

# -------------------------
# Run & Export
# -------------------------
sasp_results = compute_saspa(a_VaR_values, use_spa=True)

# --- ROBUST DataFrame Creation ---
all_data = {}
for i, result in enumerate(sasp_results):
    alpha = alpha_values[i]  # 若 a_VaR_values 与 alpha_values 长度不同，相应对齐
    all_data[f'VaRC {alpha}'] = result['mean_VaRC']
    all_data[f'VaRC S.E. {alpha}'] = result['se_VaRC']
    all_data[f'VaRC CPU {alpha}'] = result['total_time']

Risk_Contributions = pd.DataFrame(
    all_data,
    index=pd.Index([f'Obligor {i+1}' for i in range(N)])
).T

# Risk_Contributions.to_csv('RhoS=0.5 VaRC SASPA.csv')
print("\n--- Final Risk Contributions ---")
print(Risk_Contributions)
