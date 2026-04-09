import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hitran = pd.read_csv('hitran_values.csv')
quadrature = pd.read_csv('quadrature_points_weights.csv')

# Create frequency grid
nu = np.arange(7280, 7290, 0.01)

# Calculate k_nu for each frequency (Lorentz profile)
k_nu = np.zeros_like(nu)
for i, n in enumerate(nu):
    k_nu[i] = np.sum(hitran['S'] / np.pi * hitran['alpha'] / ((n - hitran['nu_0'])**2 + hitran['alpha']**2))

# Create 50 equal log intervals and compute cumulative distribution
k_min = k_nu[k_nu > 0].min()
k_max = k_nu.max()

# 50 equal intervals in log scale
log_intervals = np.logspace(np.log10(k_min), np.log10(k_max), 50)

# Compute cumulative number n(0,k) for each interval
n_cumulative = np.array([np.sum(k_nu <= k) for k in log_intervals])

# Convert to g-space (normalized cumulative distribution)
g = n_cumulative / len(k_nu)

# Plot all three: k_nu spectrum, cumulative distribution, and k(g)
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# plot: original k_nu spectrum
axs[0].plot(nu, k_nu, linewidth=1)
axs[0].set_yscale('log')
axs[0].set_xlabel('ν (cm⁻¹)', fontsize=14)
axs[0].set_ylabel('k_ν (atm cm)$^{-1}$', fontsize=14)
axs[0].set_title('H$_2$O 1.38-$\mu$m band')
axs[0].grid(True, alpha=0.3, which='both')

# plot: cumulative distribution
axs[1].plot(log_intervals, n_cumulative, linewidth=2, marker='o', markersize=3)
axs[1].set_xscale('log')
axs[1].set_xlabel('k_ν (atm cm)$^{-1}$', fontsize=14)
axs[1].set_ylabel('n(0,k)', fontsize=14)
axs[1].set_title('Cumulative Distribution of k_ν')
axs[1].grid(True, alpha=0.3, which='both')

# plot: k(g)
axs[2].plot(g, log_intervals, linewidth=2, marker='o', markersize=3)
axs[2].set_yscale('log')
axs[2].set_xlabel('g', fontsize=14)
axs[2].set_ylabel('k (atm cm)$^{-1}$', fontsize=14)
axs[2].set_title('k-distribution function k(g)')
axs[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

