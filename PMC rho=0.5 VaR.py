#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 10:43:17 2025

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
bandwidth = 0.005 # 0.005

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

def var_pmc(d, alpha, LGD_a, LGD_b, simulation_runs, bandwidth=0.005):
    start_time = time.time()
    
    VaRs = []
    for _ in range(n_repetitions):
        epsilon, D, L = generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs)
        VaR = np.percentile(L, alpha * 100)
        VaRs.append(VaR)
    
    VaR_mean, VaR_se = mean_se(VaRs)
    
    end_time = time.time()
    print(f"Time taken for PMC (VaRC_PMC): {end_time - start_time:.2f} seconds")
    
    return VaR_mean, VaR_se

Result = [var_pmc(d, alpha, LGD_a, LGD_b, simulation_runs) for alpha in alpha_values]
VaRs, VaR_SEs = zip(*Result)

Risk_Measures = pd.DataFrame({
    'VaR' : VaRs,
    'VaR S.E.' : VaR_SEs,
    }, index=alpha_values).T

# Risk_Measures.to_csv('RhoS=0.5 VaR PMC.csv')
