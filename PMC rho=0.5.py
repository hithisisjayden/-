#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 08:46:02 2025

@author: jaydenwang
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import time
import csv

np.random.seed(42)

# Portfolio size
d = 10

# Repetition for scenarios
n_repetitions = 10

# Simulation paths
simulation_runs = 10000000 # 10000000 一千万是极限了 再算restarting kernal了

# Bandwidth
bandwidth = 0.010 # 0.005

# Confidence level
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]

# LGD Shape parameters
LGD_a, LGD_b = 2, 5

# rho
rho = np.sqrt(0.5)

# rho_S, correlation of PD and LGD
rho_S = 0.5
covariance_matrix = np.array([[1, rho_S],
                              [rho_S, 1]])

# Z = np.random.multivariate_normal([0,0], covariance_matrix, simulation_runs)
# Z_L = Z[:, 0]
# Z_D = Z[:, 1]

# Default probability function
def default_probability(d):
    return np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
def loss_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def default_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs):
    
    Z = np.random.multivariate_normal([0,0], covariance_matrix, simulation_runs)
    Z_L = Z[:, 0]
    Z_D = Z[:, 1]
    
    # Z = np.random.normal(size=simulation_runs)
    # Z_L = Z
    # Z_D = Z
    
    eta_L = np.random.normal(size=(simulation_runs, d))
    eta_D = np.random.normal(size=(simulation_runs, d))
    Y = loss_driver(Z_L, eta_L)
    X = default_driver(Z_D, eta_D)
    epsilon = beta.ppf(norm.cdf(Y), LGD_a, LGD_b)
    p = default_probability(d)
    x_threshold = norm.ppf(1-p)
    D = (X > x_threshold).astype(int)
    L = np.sum(epsilon * D, axis = 1)
    return epsilon, D, L

def mean_se(array):
    mean = np.mean(array)
    se = np.std(array) / np.sqrt(len(array))
    return mean, se

def var_varc_es_esc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs, bandwidth=0.010):
    start_time = time.time()
    
    n_repetitions = 10
        
    VaRs, VaRCs, Samples_varc= [], [], []
    for _ in range(n_repetitions):
        epsilon, D, L = generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs)
        VaR = np.percentile(L, alpha * 100)
        
        LGDs = epsilon * D
        mask_varc = (L >= VaR - bandwidth) & (L <= VaR + bandwidth)
        
        VaRC = np.mean(LGDs[mask_varc], axis=0)

        Sample_varc = np.sum(mask_varc)
        
        VaRs.append(VaR)
        VaRCs.append(VaRC)
        Samples_varc.append(Sample_varc)
    
    VaRs = np.array(VaRs)
    VaRCs = np.array(VaRCs)
    Samples_varc = np.array(Samples_varc)  
    
    VaR_mean, VaR_se = mean_se(VaRs)
    VaRC_mean = np.mean(VaRCs, axis=0)
    VaRC_se = np.array([np.std(VaRCs[:,i]) / np.sqrt(len(VaRCs[:,i])) for i in range(d)])
    Sample_varc_mean = np.mean(Samples_varc)

    end_time = time.time()
    print(f"Time taken for PMC (VaRC_PMC): {end_time - start_time:.2f} seconds")
    
    return VaR_mean, VaR_se, VaRC_mean, VaRC_se, Sample_varc_mean


Result = [var_varc_es_esc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs) for alpha in alpha_values]
VaRs, VaR_SEs, VaRCs, VaRC_SEs, Sample_VaRCs = zip(*Result)

Risk_Measures = pd.DataFrame({
    'VaR' : VaRs,
    'VaR S.E.' : VaR_SEs,
    }, index=alpha_values).T

Risk_Contributions = pd.DataFrame({
    'VaRC 0.95' : VaRCs[0],
    'VaRC S.E. 0.95' : VaRC_SEs[0],
    'VaRC 0.95 Samples' : Sample_VaRCs[0],
    
    'VaRC 0.96' : VaRCs[1],
    'VaRC S.E. 0.96' : VaRC_SEs[1],
    'VaRC 0.96 Samples' : Sample_VaRCs[1],
    
    'VaRC 0.97' : VaRCs[2],
    'VaRC S.E. 0.97' : VaRC_SEs[2],
    'VaRC 0.97 Samples' : Sample_VaRCs[2],
    
    'VaRC 0.98' : VaRCs[3],
    'VaRC S.E. 0.98' : VaRC_SEs[3],
    'VaRC 0.98 Samples' : Sample_VaRCs[3],
    
    'VaRC 0.99' : VaRCs[4],
    'VaRC S.E. 0.99' : VaRC_SEs[4],
    'VaRC 0.99 Samples' : Sample_VaRCs[4],

    }, index=['Obligor 1','Obligor 2','Obligor 3','Obligor 4','Obligor 5','Obligor 6','Obligor 7','Obligor 8','Obligor 9','Obligor 10']).T

# Risk_Measures.to_csv('RhoS=0.5 VaR PMC.csv')
Risk_Contributions.to_csv('RhoS=0.5 VaRC PMC bandwidth=0.01.csv')

print(Risk_Contributions)
