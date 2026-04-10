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
    k_nu[i] = np.sum(
        hitran['S'] / np.pi * hitran['alpha'] /
        ((n - hitran['nu_0'])**2 + hitran['alpha']**2)
    )

n_bins = 50

# Avoid log(0)
k_safe = np.clip(k_nu, 1e-300, None)
logk = np.log10(k_safe)

# 50 equal intervals in log(k)
logk_edges = np.linspace(logk.min(), logk.max(), n_bins + 1)

# Count how many k_nu values fall in each interval
counts, _ = np.histogram(logk, bins=logk_edges)

# Progressive cumulative number n(0, k)
n_0_k = np.cumsum(counts)
N = len(k_nu)

# g(k) = n(0,k)/N
g_edges_upper = n_0_k / N

# Representative k for each bin (midpoint in log space)
logk_mid = 0.5 * (logk_edges[:-1] + logk_edges[1:])
k_bins = 10**logk_mid

# Representative g for each bin (midpoint of cumulative probability in each bin)
g_bins = (n_0_k - 0.5 * counts) / N



# Load quadrature points and weights (stripping whitespace from column names)
gq = quadrature['points'].to_numpy()
wq = quadrature[quadrature.columns[1]].to_numpy()  # 2nd column is weights with space

# Normalize weights in case they do not sum exactly to 1
wq = wq / np.sum(wq)

# Interpolate k(g) to quadrature points
# Add endpoints for interpolation stability
g_interp = np.concatenate(([0.0], g_bins, [1.0]))
k_interp = np.concatenate(([k_bins[0]], k_bins, [k_bins[-1]]))
kq = np.interp(gq, g_interp, k_interp)

# Path length u from 10^-5 to 10 g cm^-2
u = np.logspace(-5, 1, 200)

# Convert from (g cm^-2) to (atm cm)
# 1 (g cm^-2) = 2.24e4 / M (atm cm)
M = 18.01528  # molecular weight of H2O
u_atm_cm = u * (2.24e4 / M)

# Line-by-line transmittance
T_lbl = np.zeros_like(u)
for i, uu in enumerate(u_atm_cm):
    T_lbl[i] = np.mean(np.exp(-k_nu * uu))

# k-distribution transmittance using quadrature
T_kdist = np.zeros_like(u)
for i, uu in enumerate(u_atm_cm):
    T_kdist[i] = np.sum(wq * np.exp(-kq * uu))
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# (a) Original spectrum
axs[0, 0].plot(nu, k_nu, linewidth=1)
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('ν (cm⁻¹)', fontsize=14)
axs[0, 0].set_ylabel('k_ν (atm cm)$^{-1}$', fontsize=14)
axs[0, 0].set_title(r'H$_2$O 1.38-$\mu$m band', fontsize=14)
axs[0, 0].grid(True, alpha=0.3, which='both')

# (b) Cumulative distribution g(k) from 50 log-k intervals
axs[0, 1].step(k_bins, g_edges_upper, where='mid')
axs[0, 1].set_xscale('log')
axs[0, 1].set_xlabel('k (atm cm)$^{-1}$', fontsize=14)
axs[0, 1].set_ylabel('g(k)', fontsize=14)
axs[0, 1].set_title('Cumulative Distribution of k_ν', fontsize=14)
axs[0, 1].grid(True, alpha=0.3, which='both')

# (c) k(g) in g-domain
axs[1, 0].plot(g_bins, k_bins, linewidth=1, markersize=3)
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel('g', fontsize=14)
axs[1, 0].set_ylabel('k (atm cm)$^{-1}$', fontsize=14)
axs[1, 0].set_title('k-distribution function k(g)', fontsize=14)
axs[1, 0].grid(True, alpha=0.3, which='both')
# Plot comparison 
axs[1, 1].semilogx(u, T_lbl, label='Line-by-line', linewidth=2)
axs[1, 1].semilogx(u, T_kdist, '--', label='k-distribution', linewidth=2)
axs[1, 1].set_xlabel(r'Path length $u$ (g cm$^{-2}$)', fontsize=14)
axs[1, 1].set_ylabel('Transmittance', fontsize=14)
axs[1, 1].set_title('Comparison of spectral transmittance', fontsize=14)
axs[1, 1].grid(True, which='both', alpha=0.3)
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('plots.png', dpi=300)
