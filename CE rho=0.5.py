import numpy as np
import pandas as pd
from scipy.stats import norm, beta, qmc
import time

# Set random seed for reproducibility (only for numpy RNG; Sobol has its own seed)
np.random.seed(42)

# ---------------- 模型参数（可按需修改） ----------------
d = 10  # Number of obligors
p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])  # Default probabilities
rho = np.sqrt(0.5)  # factor loading for idiosyncratic structure (rho_i^D and rho_i^L)
alpha_beta = (2, 5)  # Beta distribution parameters for LGD
x_d = norm.ppf(1 - p)  # Default thresholds

# 系统因子之间的相关性（这就是你补充的 rho_S）
rho_S = 0.5  # 若想恢复无关情况设为0

# Given VaR values for specified confidence levels (assumed known)
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]

# Simulation parameters
K = 10_000_000  # Number of final simulation paths (large -> will use chunking)
P = int(0.2 * K)  # Number of pilot samples (pilot : final = 1 : 5)
repetitions = 10  # Number of repetitions
rho_quantile = 0.5  # Quantile for selecting elite samples (你的原文使用0.5)
delta = 0.005  # Bandwidth for VaRC estimation
max_iterations = 5  # Maximum iterations for the CE optimization
eps_qmc = 1e-12  # 在将U映射为正态前的裁剪数值（避免0或1）

# Sobol sequence generator (global, 与原代码风格一致)
# 总维度 = 2(systemic) + 2*d (idiosyncratic)
sobol_engine = qmc.Sobol(d=2 + 2 * d, scramble=True, seed=0)

# ------------- 预计算：协方差、Cholesky、逆矩阵 -------------
Sigma = np.array([[1.0, rho_S],
                  [rho_S, 1.0]], dtype=float)
Sigma_inv = np.linalg.inv(Sigma)
L_chol = np.linalg.cholesky(Sigma)  # 2x2 下三角，用于把独立标准正态变为具有 Sigma 的正态

# ------------- 主函数：cross-entropy VaRC（保持你的接口） -------------
def cross_entropy_varc(alpha, var, P=P, K=K, repetitions=repetitions,
                      rho_quantile=rho_quantile, delta=delta, max_iterations=max_iterations):
    """
    用交叉熵（CE）+ Sobol QMC 对给定 VaR 进行 VaR-contribution 估计（保持你原始代码风格）
    输入:
        alpha: 置信水平（仅用于打印）
        var: 已知 VaR 值 a
        其余参数同全局变量含义
    返回:
        varc_mean: d 维数组，各 obligor 的 VaRC 均值
        varc_se: d 维数组，各 obligor 的 VaRC 标准误
        samples_mean: 平均落入 VaR +/- delta 区间的样本数量（每次 repetition）
    """
    start_total_time = time.time()

    # CE 优化阶段：只对系统因子做均值移位 mu = (mu_D, mu_L)
    mu_D_t = 0.0
    mu_L_t = 0.0
    l = var  # 目标损失水平（给定 VaR）
    mu_vec = np.array([mu_D_t, mu_L_t], dtype=float)

    # 每轮 CE 使用 pilot 的 QMC（与原代码类似）
    for t in range(max_iterations):
        # 生成 P 条 pilot Sobol -> 标准正态
        U = sobol_engine.random(P)  # shape (P, 2+2d)
        U = np.clip(U, eps_qmc, 1.0 - eps_qmc)
        Z_std = norm.ppf(U)  # 标准正态样本

        # 前两维是系统因子的标准正态（独立），通过 Cholesky 转换为有协方差
        gZ = Z_std[:, 0:2]                    # (P,2) 独立标准正态
        Z_cor = gZ @ L_chol.T + mu_vec        # (P,2) 有协方差并加上均值偏移
        Z_D = Z_cor[:, 0]
        Z_L = Z_cor[:, 1]

        # 后 2d 维是个体因子（独立标准正态）
        eta_D = Z_std[:, 2:2 + d]
        eta_L = Z_std[:, 2 + d:2 + 2 * d]

        # 计算潜变量与损失 （与你原代码完全一致的结构，rho 为 factor loading）
        X = rho * Z_D[:, np.newaxis] + np.sqrt(1 - rho ** 2) * eta_D  # (P,d)
        Y = rho * Z_L[:, np.newaxis] + np.sqrt(1 - rho ** 2) * eta_L  # (P,d)
        D = (X >= x_d).astype(float)
        U_beta = norm.cdf(Y)
        U_beta = np.clip(U_beta, eps_qmc, 1.0 - eps_qmc)
        EPS = beta.ppf(U_beta, a=alpha_beta[0], b=alpha_beta[1])
        losses = np.sum(EPS * D, axis=1)  # (P,)

        # 取精英阈值 (1 - rho_quantile) 分位
        l_t = np.quantile(losses, 1.0 - rho_quantile)

        # 若已达到目标阈值即可停止（与你原代码逻辑一致）
        if l_t >= l:
            break

        # 计算似然比：注意系统因子现在相关，用矩阵式
        mu_vec = np.array([mu_D_t, mu_L_t], dtype=float)
        v = Sigma_inv @ mu_vec  # (2,)
        t1 = - (Z_cor @ v)      # (P,)
        t2 = 0.5 * (mu_vec @ Sigma_inv @ mu_vec)
        LR_p = np.exp(t1 + t2)  # (P,)

        # 计算权重：指示 × 似然比 -> 标准化
        indicator = (losses >= l_t).astype(float)
        weights = indicator * LR_p
        sum_w = weights.sum()
        if sum_w <= 1e-16:
            # 没有精英样本或权重极小 -> 不更新 mu（保持当前 mu）
            print(f"[CE] 迭代 {t}：没有有效精英样本，保持 mu 不变")
            break
        weights = weights / sum_w

        # 更新 mu 向量为加权样本均值（注意 Z_cor 是 (P,2)）
        hat_mu = (weights[:, None] * Z_cor).sum(axis=0)  # (2,)
        mu_D_t, mu_L_t = hat_mu[0], hat_mu[1]
        # 下一轮继续

    # CE 完成，得到 mu*
    mu_D_star = mu_D_t
    mu_L_star = mu_L_t
    mu_star = np.array([mu_D_star, mu_L_star], dtype=float)

    # 最终模拟阶段：重复 repetitions 次，每次用 K 路径（用 chunk 分批以节省内存）
    varc_results = np.zeros((repetitions, d))
    samples_results = []

    # chunk size：可根据机器内存调节（例如 2e6）。设为较大以降低 Sobol 调用次数。
    chunk_size = int(2_000_000)
    if chunk_size > K:
        chunk_size = K

    for rep in range(repetitions):
        start_rep = time.time()
        denom_total = 0.0
        numerators = np.zeros(d, dtype=float)
        count_in_range = 0

        drawn = 0
        while drawn < K:
            n_this = min(chunk_size, K - drawn)
            U = sobol_engine.random(n_this)
            U = np.clip(U, eps_qmc, 1.0 - eps_qmc)
            Z_std = norm.ppf(U)

            gZ = Z_std[:, 0:2]
            Z_cor = gZ @ L_chol.T + mu_star   # (n_this,2)
            Z_D = Z_cor[:, 0]
            Z_L = Z_cor[:, 1]

            eta_D = Z_std[:, 2:2 + d]
            eta_L = Z_std[:, 2 + d:2 + 2 * d]

            X = rho * Z_D[:, np.newaxis] + np.sqrt(1 - rho ** 2) * eta_D
            Y = rho * Z_L[:, np.newaxis] + np.sqrt(1 - rho ** 2) * eta_L
            D = (X >= x_d).astype(float)
            U_beta = norm.cdf(Y)
            U_beta = np.clip(U_beta, eps_qmc, 1.0 - eps_qmc)
            EPS = beta.ppf(U_beta, a=alpha_beta[0], b=alpha_beta[1])
            losses = np.sum(EPS * D, axis=1)

            # 计算似然比（带协方差）
            v = Sigma_inv @ mu_star
            t1 = - (Z_cor @ v)        # (n_this,)
            t2 = 0.5 * (mu_star @ Sigma_inv @ mu_star)
            LR_p = np.exp(t1 + t2)   # (n_this,)

            # 找到处于 var +/- delta 区间的样本
            in_range = (losses >= var - delta) & (losses <= var + delta)
            if np.any(in_range):
                # 累计分母与分子
                w_sel = LR_p[in_range]
                denom_total += w_sel.sum()
                # 对每个 obligor 累计 LR * D * EPS
                numerators += (w_sel[:, None] * (D[in_range, :] * EPS[in_range, :])).sum(axis=0)
                count_in_range += in_range.sum()

            drawn += n_this

        # 结束单次 repetition
        samples_results.append(count_in_range)
        if denom_total > 0:
            varc_results[rep, :] = numerators / denom_total
        else:
            varc_results[rep, :] = np.nan  # 未命中 -> NaN

        elapsed_rep = time.time() - start_rep
        print(f"[Final] rep={rep+1}/{repetitions}, found_in_range={count_in_range}, time={elapsed_rep:.1f}s")

    # 统计结果
    varc_mean = np.nanmean(varc_results, axis=0)
    varc_se = np.nanstd(varc_results, axis=0, ddof=0) / np.sqrt(repetitions)
    samples_mean = np.mean(samples_results)

    total_time = time.time() - start_total_time
    print(f"Alpha {alpha}: Total time {total_time:.2f} s, Samples mean {samples_mean:.2f}, mu*={mu_star}")

    return varc_mean, varc_se, samples_mean

# ----------------- 生成结果表格（与你原始代码风格一致） -----------------
index_names = []
for alpha in alpha_values:
    percent = int(alpha * 100)
    index_names.extend([f'{percent}% VaRC', f'{percent}% VaRC SE', f'{percent}% Samples'])
column_names = [f'Obligor {i+1}' for i in range(d)]
results_df = pd.DataFrame(index=index_names, columns=column_names, dtype=float)

# 主循环：对每个置信水平跑一次
for alpha, var in zip(alpha_values, a_VaR_values):
    varc_mean, varc_se, samples = cross_entropy_varc(alpha, var)
    percent = int(alpha * 100)
    results_df.loc[f'{percent}% VaRC'] = varc_mean
    results_df.loc[f'{percent}% VaRC SE'] = varc_se
    results_df.loc[f'{percent}% Samples'] = samples

# 显示结果
print("\nResults:")
print(results_df)

# (可选) 保存到 csv
results_df.to_csv('RhoS=0.5 VaRC CE.csv', index=True)
