import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import time

np.random.seed(42)

# Parameters
N = 10  # Portfolio size
rho_S = 0.5  # Correlation between Z_D and Z_L
rho_D = np.sqrt(0.5) * np.ones(N)  # Factor loadings for default, uniform
rho_L = np.sqrt(0.5) * np.ones(N)  # Factor loadings for LGD, uniform
alpha_beta = 2  # Shape parameter alpha for Beta
beta_beta = 5   # Shape parameter beta for Beta
p = np.array([0.01 * (i+1) for i in range(N)])  # Unconditional PDs: 0.01 to 0.10
x_d = norm.ppf(1 - p)  # Default thresholds

# Given VaR values for alphas [0.95,0.96,0.97,0.98,0.99]
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]

# # For testing with a single value
# a_VaR_values = [1.143161852]

# Simulation parameters (modifiable)
N_outer = 1000  # Number of outer simulations for Z (e.g., 1000 for testing, increase for accuracy)
N_inner = 1000  # Number of inner simulations for L_{-i} and epsilon_i (e.g., 10000)
reps = 10  # Number of repetitions for averaging and SE

def compute_safa_varc(a_VaR_values, N_outer, N_inner, reps):
    cov_matrix = np.array([[1, rho_S], [rho_S, 1]])
    
    results = []  # List to store for each a: mean_VaRC, se_VaRC, rep_times, total_time
    
    for a in a_VaR_values:
        VaRC_reps = []  # List of VaRC arrays for each rep
        rep_times = []  # List of times for each rep
        
        total_start = time.time()
        
        for r in range(reps):
            rep_start = time.time()
            
            A = np.zeros(N)  # Numerators A_i for each obligor
            
            for _ in range(N_outer):
                # Sample Z = (Z_L, Z_D)
                Z = np.random.multivariate_normal([0, 0], cov_matrix)
                z_L, z_D = Z[0], Z[1]
                
                # Sample inner paths for all obligors
                eta_L = np.random.randn(N, N_inner)
                eta_D = np.random.randn(N, N_inner)
                
                # Compute Y_j, epsilon_j
                Y = rho_L[:, np.newaxis] * z_L + np.sqrt(1 - rho_L[:, np.newaxis]**2) * eta_L
                U = norm.cdf(Y)
                epsilon = beta.ppf(U, alpha_beta, beta_beta)  # (N, N_inner)
                
                # Compute X_j, D_j
                X = rho_D[:, np.newaxis] * z_D + np.sqrt(1 - rho_D[:, np.newaxis]**2) * eta_D
                D = (X > x_d[:, np.newaxis]).astype(float)  # (N, N_inner)
                
                # Individual losses
                losses = epsilon * D  # (N, N_inner)
                
                # Total L
                L_total = np.sum(losses, axis=0)  # (N_inner,)
                
                # For each i, L_{-i} = L_total - losses_i
                L_minus = L_total[np.newaxis, :] - losses  # (N, N_inner)
                
                # Sort L_{-i} for each i for binary search
                sorted_L_minus = np.sort(L_minus, axis=1)  # (N, N_inner)
                
                # Sample M = N_inner independent epsilon_i | z_L (shared since rho_L same)
                eta_L_i = np.random.randn(N_inner)
                Y_i = rho_L[0] * z_L + np.sqrt(1 - rho_L[0]**2) * eta_L_i  # rho_L same for all
                U_i = norm.cdf(Y_i)
                epsilon_samples = beta.ppf(U_i, alpha_beta, beta_beta)  # (N_inner,)
                
                # Clip epsilon_samples to avoid extreme values
                epsilon_samples = np.clip(epsilon_samples, 1e-9, 1 - 1e-9)
                
                # Vectorized computation for f_cond, f_prime_cond, g
                f_beta = beta.pdf(epsilon_samples, alpha_beta, beta_beta)
                u = beta.cdf(epsilon_samples, alpha_beta, beta_beta)
                v = norm.ppf(u)
                w = (v - rho_L[0] * z_L) / np.sqrt(1 - rho_L[0]**2)
                phi_v = norm.pdf(v)
                phi_w = norm.pdf(w)
                s = np.sqrt(1 - rho_L[0]**2)
                f_cond = np.zeros(N_inner)
                mask_phi = phi_v > 0
                f_cond[mask_phi] = f_beta[mask_phi] * phi_w[mask_phi] / (phi_v[mask_phi] * s)
                
                term1 = np.zeros(N_inner)
                term1[mask_phi] = f_beta[mask_phi] / phi_v[mask_phi] * (v[mask_phi] - w[mask_phi] / s)
                term2 = (alpha_beta - 1) / epsilon_samples - (beta_beta - 1) / (1 - epsilon_samples)
                f_prime_cond = f_cond * (term1 + term2)
                
                g = np.zeros(N_inner)
                mask_f = f_cond > 0
                g[mask_f] = 1 + epsilon_samples[mask_f] * (f_prime_cond[mask_f] / f_cond[mask_f])
                
                # **BUG FIX:** Moved p_cond calculation here.
                # It must be calculated in every outer loop but before it is used.
                arg = (x_d - rho_D * z_D) / np.sqrt(1 - rho_D**2)
                p_cond = 1 - norm.cdf(arg)
                
                # Compute boundary
                boundary = np.zeros(N)
                if a < 1:
                    # Compute f_cond_a
                    ei_a = a
                    ei_a = np.clip(ei_a, 1e-9, 1 - 1e-9)
                    f_beta_a = beta.pdf(ei_a, alpha_beta, beta_beta)
                    u_a = beta.cdf(ei_a, alpha_beta, beta_beta)
                    v_a = norm.ppf(u_a)
                    w_a = (v_a - rho_L[0] * z_L) / s
                    phi_v_a = norm.pdf(v_a)
                    phi_w_a = norm.pdf(w_a)
                    f_cond_a = f_beta_a * phi_w_a / (phi_v_a * s) if phi_v_a > 0 else 0
                    
                    prod_all = np.prod(1 - p_cond)
                    for i in range(N):
                        denom = (1 - p_cond[i])
                        prod_i = prod_all / denom if denom != 0 else 0
                        boundary[i] = - a * f_cond_a * prod_i
                
                inner_values = np.zeros(N)
                
                for i in range(N):
                    # Compute F for all epsilon_samples
                    thresholds = a - epsilon_samples
                    F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner
                    
                    sum_term = np.sum(F * g)
                    inner_avg = sum_term / N_inner + boundary[i]
                    inner_values[i] = inner_avg
                
                # Accumulate to A
                A += (p_cond * inner_values) 
            
            # Divide by N_outer after the loop for efficiency
            A = A / N_outer

            # Denominator via full allocation: f_L(a) = sum A_i / a
            sum_A = np.sum(A)
            f_L_a = sum_A / a if a != 0 else 0
            print(f"f_L(a) = {f_L_a:.2f}")
            
            # VaRC_i = A_i / f_L_a
            VaRC = A / f_L_a if f_L_a != 0 else np.zeros(N)
            
            VaRC_reps.append(VaRC)
            
            rep_end = time.time()
            rep_duration = rep_end - rep_start
            rep_times.append(rep_duration)
            print(f"Rep {r+1} for VaR={a}: Time = {rep_duration:.2f} seconds")
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        # Compute mean and SE
        VaRC_array = np.array(VaRC_reps)  # (reps, N)
        mean_VaRC = np.mean(VaRC_array, axis=0)
        se_VaRC = np.std(VaRC_array, axis=0) / np.sqrt(reps)
        
        results.append({
            'a': a,
            'mean_VaRC': mean_VaRC,
            'se_VaRC': se_VaRC,
            'rep_times': rep_times,
            'total_time': total_duration
        })
    
    return results

# Run the computation
safa_results = compute_safa_varc(a_VaR_values, N_outer, N_inner, reps) 


# Print final results
VaRCs = [r['mean_VaRC'] for r in safa_results]      
VaRC_SEs = [r['se_VaRC'] for r in safa_results]     
CPUs = [r['total_time'] for r in safa_results] 

Risk_Contributions = pd.DataFrame({
    'VaRC 0.95' : VaRCs[0],
    'VaRC S.E. 0.95' : VaRC_SEs[0],
    'VaRC 0.95 CPU' : CPUs[0],
    
    'VaRC 0.96' : VaRCs[1],
    'VaRC S.E. 0.96' : VaRC_SEs[1],
    'VaRC 0.96 CPU' : CPUs[1],
    
    'VaRC 0.97' : VaRCs[2],
    'VaRC S.E. 0.97' : VaRC_SEs[2],
    'VaRC 0.97 CPU' : CPUs[2],
    
    'VaRC 0.98' : VaRCs[3],
    'VaRC S.E. 0.98' : VaRC_SEs[3],
    'VaRC 0.98 CPU' : CPUs[3],
    
    'VaRC 0.99' : VaRCs[4],
    'VaRC S.E. 0.99' : VaRC_SEs[4],
    'VaRC 0.99 CPU' : CPUs[4],

    }, index=pd.Index([f'Obligor {i+1}' for i in range(N)])).T

# Risk_Contributions.to_csv('RhoS=0.5 VaRC SAFA.csv')
print(Risk_Contributions)

