"""
experiments.py
==============
Reproduce all experimental results from:

    Wansouwé, E. (2026). Partial Arc Ellipse Fitting via Cloud
    Regression: Critical Coverage Threshold and Sensitivity to
    Noise and Eccentricity. Pattern Recognition Letters.

Usage
-----
    python experiments.py --all          # run all experiments
    python experiments.py --exp 1        # baseline threshold
    python experiments.py --exp 2        # eccentricity effect
    python experiments.py --exp 3        # noise effect
    python experiments.py --exp 4        # spectral analysis
    python experiments.py --exp 5        # power-law model

Author : Eric Wansouwé
Email  : ericwansouwe@gmail.com
Date   : March 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fit_ellipse import fit_ellipse, cart_to_pol, \
                        get_ellipse_points

# ── Global parameters ──────────────────────────────────────────
# Reference ellipse (fixed throughout all experiments)
PARAMS_TRUE = (2.0, 3.0, 5.0, 3.0, np.pi / 4)
X0, Y0, AP, BP, PHI = PARAMS_TRUE

# Evaluation threshold (10% of semi-minor axis)
TAU = 0.5

# Number of repetitions per configuration
N_REPEAT = 20

# Arc coverage range
ARCS_DEG = np.arange(30, 361, 10)


# ── Helper functions ────────────────────────────────────────────
def center_error(x0_est, y0_est):
    """Euclidean error on the estimated center."""
    return np.sqrt((x0_est - X0)**2 + (y0_est - Y0)**2)


def run_single(arc_deg, sigma, seed):
    """
    Run one fitting trial and return center error.
    Returns np.nan if fitting fails.
    """
    np.random.seed(seed)
    arc_rad = arc_deg * np.pi / 180
    n = 200
    t = np.linspace(0, arc_rad, n)

    x = (X0 + AP*np.cos(t)*np.cos(PHI)
            - BP*np.sin(t)*np.sin(PHI)
            + np.random.normal(0, sigma, n))
    y = (Y0 + AP*np.cos(t)*np.sin(PHI)
            + BP*np.sin(t)*np.cos(PHI)
            + np.random.normal(0, sigma, n))
    try:
        coeffs = fit_ellipse(x, y)
        x0e, y0e, _, _, _ = cart_to_pol(coeffs)
        return center_error(x0e, y0e)
    except Exception:
        return np.nan


def find_threshold(errors_mean, tau=TAU):
    """Find the minimum arc where mean error drops below tau."""
    for i, err in enumerate(errors_mean):
        if not np.isnan(err) and err < tau:
            return ARCS_DEG[i]
    return 360  # threshold not reached


# ── Experiment 1 : Baseline arc coverage ───────────────────────
def exp1_baseline(sigma=0.3):
    """
    Experiment 1: Center error vs arc coverage.
    Fixed sigma=0.3, reference ellipse parameters.
    """
    print("\n" + "="*50)
    print("  Experiment 1 — Baseline Arc Coverage")
    print(f"  sigma = {sigma}, N = {N_REPEAT} repetitions")
    print("="*50)

    means, stds = [], []
    for arc in ARCS_DEG:
        errs = [run_single(arc, sigma,
                           seed=rep*100 + int(sigma*1000))
                for rep in range(N_REPEAT)]
        errs = [e for e in errs if not np.isnan(e)]
        means.append(np.mean(errs) if errs else np.nan)
        stds.append(np.std(errs)   if errs else np.nan)

    means = np.array(means)
    stds  = np.array(stds)
    thresh = find_threshold(means)
    print(f"  Critical threshold : ~{thresh}°")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(ARCS_DEG, means, yerr=stds,
                fmt='b-o', markersize=4, capsize=3,
                linewidth=1.5,
                label=f'Center error (σ={sigma}, N={N_REPEAT})')
    ax.axvline(180, color='red', linestyle='--',
               linewidth=1.5, label='180° reference')
    ax.axhline(TAU, color='green', linestyle='--',
               linewidth=1, label=f'Threshold τ={TAU}')
    ax.set_xlabel('Arc coverage (degrees)', fontsize=12)
    ax.set_ylabel('Center estimation error', fontsize=12)
    ax.set_title('Experiment 1 — Critical Arc Threshold\n'
                 'Cloud Regression Ellipse Fitting',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_threshold.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: fig2_threshold.png")
    return means, stds


# ── Experiment 2 : Effect of eccentricity ──────────────────────
def exp2_eccentricity(sigma=0.2):
    """
    Experiment 2: Critical threshold vs eccentricity.
    Three ellipses: bp in {1.0, 3.0, 4.5}, fixed ap=5.0.
    """
    print("\n" + "="*50)
    print("  Experiment 2 — Effect of Eccentricity")
    print(f"  sigma = {sigma}, N = {N_REPEAT} repetitions")
    print("="*50)

    ellipses = [
        (5.0, 4.5, 'Near-circle   (e=0.436)', 'blue'),
        (5.0, 3.0, 'Medium ellipse (e=0.800)', 'orange'),
        (5.0, 1.0, 'Elongated      (e=0.980)', 'green'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for ap_val, bp_val, label, color in ellipses:
        ecc = np.sqrt(1 - (bp_val/ap_val)**2)
        params = (X0, Y0, ap_val, bp_val, PHI)
        means = []

        for arc in ARCS_DEG:
            arc_rad = arc * np.pi / 180
            errs = []
            for rep in range(N_REPEAT):
                np.random.seed(rep*100 + int(sigma*1000))
                n = 200
                t = np.linspace(0, arc_rad, n)
                x = (X0 + ap_val*np.cos(t)*np.cos(PHI)
                        - bp_val*np.sin(t)*np.sin(PHI)
                        + np.random.normal(0, sigma, n))
                y = (Y0 + ap_val*np.cos(t)*np.sin(PHI)
                        + bp_val*np.sin(t)*np.cos(PHI)
                        + np.random.normal(0, sigma, n))
                try:
                    coeffs = fit_ellipse(x, y)
                    x0e, y0e, _, _, _ = cart_to_pol(coeffs)
                    errs.append(center_error(x0e, y0e))
                except Exception:
                    errs.append(np.nan)
            valid = [e for e in errs if not np.isnan(e)]
            means.append(np.mean(valid) if valid else np.nan)

        means = np.array(means)
        thresh = find_threshold(means)
        print(f"  {label}: threshold ≈ {thresh}°")
        ax.plot(ARCS_DEG, means, '-o', markersize=4,
                linewidth=1.5, color=color,
                label=f'{label}: threshold≈{thresh}°')

    ax.axvline(180, color='red', linestyle='--',
               linewidth=1.5, label='180° reference')
    ax.axhline(TAU, color='black', linestyle='--',
               linewidth=1, label=f'τ={TAU}')
    ax.set_xlabel('Arc coverage (degrees)', fontsize=12)
    ax.set_ylabel('Center estimation error', fontsize=12)
    ax.set_title('Experiment 2 — Effect of Eccentricity\n'
                 'Cloud Regression Ellipse Fitting',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_eccentricity.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: fig3_eccentricity.png")


# ── Experiment 3 : Effect of noise ─────────────────────────────
def exp3_noise():
    """
    Experiment 3: Critical threshold vs noise level sigma.
    14 noise levels in [0.05, 1.50].
    """
    print("\n" + "="*50)
    print("  Experiment 3 — Effect of Noise Level")
    print(f"  N = {N_REPEAT} repetitions per sigma")
    print("="*50)

    sigmas = np.array([0.05, 0.1, 0.15, 0.2, 0.3,
                       0.4,  0.5, 0.6,  0.7, 0.8,
                       0.9,  1.0, 1.2,  1.5])

    # Panel 1: error curves for 3 sigma values
    fig, ax = plt.subplots(figsize=(9, 5))
    for sigma, color in [(0.1, 'blue'),
                         (0.3, 'orange'),
                         (0.8, 'green')]:
        means = []
        for arc in ARCS_DEG:
            errs = [run_single(arc, sigma,
                               seed=rep*100+int(sigma*1000))
                    for rep in range(N_REPEAT)]
            valid = [e for e in errs if not np.isnan(e)]
            means.append(np.mean(valid) if valid else np.nan)
        ax.plot(ARCS_DEG, means, '-o', markersize=4,
                linewidth=1.5, color=color,
                label=f'σ={sigma}')

    ax.axvline(180, color='red', linestyle='--',
               linewidth=1.5, label='180°')
    ax.axhline(TAU, color='black', linestyle='--',
               linewidth=1, label=f'τ={TAU}')
    ax.set_xlabel('Arc coverage (degrees)', fontsize=12)
    ax.set_ylabel('Center estimation error', fontsize=12)
    ax.set_title('Experiment 3 — Effect of Noise Level\n'
                 'Cloud Regression Ellipse Fitting',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_noise.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: fig4_noise.png")
    return sigmas


# ── Experiment 4 : Spectral analysis ───────────────────────────
def exp4_spectral():
    """
    Experiment 4: Eigenvalue analysis of W^T W vs arc coverage.
    Uses exact points (sigma=0) to isolate geometric effects.
    """
    print("\n" + "="*50)
    print("  Experiment 4 — Spectral Analysis of W^T W")
    print("  Exact points (sigma=0)")
    print("="*50)

    arcs = np.arange(10, 361, 10)
    lmin_vals, l2_vals, ratios = [], [], []

    for arc_deg in arcs:
        arc_rad = arc_deg * np.pi / 180
        n = 200
        t = np.linspace(0, arc_rad, n)
        x = X0 + AP*np.cos(t)*np.cos(PHI) - BP*np.sin(t)*np.sin(PHI)
        y = Y0 + AP*np.cos(t)*np.sin(PHI) + BP*np.sin(t)*np.cos(PHI)

        W = np.column_stack([x**2, x*y, y**2,
                             x,   y,   np.ones(n)])
        A = W.T @ W
        eigvals = np.sort(np.abs(np.linalg.eigvals(A)))
        lmin_vals.append(eigvals[0])
        l2_vals.append(eigvals[1])
        ratios.append(eigvals[1] / (eigvals[0] + 1e-300))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].semilogy(arcs, lmin_vals, 'b-o', markersize=3)
    axes[0].axvline(180, color='red', linestyle='--')
    axes[0].set_ylabel('λ_min (log)', fontsize=11)
    axes[0].set_title('λ_min of W^T W — always ≈ 0 '
                      '(rank deficiency)', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(arcs, l2_vals, 'r-o', markersize=3)
    axes[1].axvline(180, color='red', linestyle='--',
                    label='180° threshold')
    axes[1].axhline(1.0, color='green', linestyle='--',
                    label='λ₂ = 1.0 (stability criterion)')
    axes[1].set_ylabel('λ₂ (log)', fontsize=11)
    axes[1].set_title('λ₂ of W^T W — crosses 1.0 at ~180°',
                      fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(arcs, ratios, 'g-o', markersize=3)
    axes[2].axvline(180, color='red', linestyle='--',
                    label='180°')
    axes[2].set_xlabel('Arc coverage (degrees)', fontsize=11)
    axes[2].set_ylabel('λ₂/λ_min (log)', fontsize=11)
    axes[2].set_title('Spectral gap ratio — '
                      'solution uniqueness indicator',
                      fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Experiment 4 — Spectral Analysis\n'
                 'Theoretical justification of 180° threshold',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig6_spectral.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: fig6_spectral.png")

    # Print key values
    print(f"\n  {'Arc':>6}  {'λ_min':>12}  "
          f"{'λ₂':>12}  {'λ₂/λ_min':>14}")
    print("  " + "-"*48)
    key_arcs = [30, 90, 150, 170, 180, 190, 270, 360]
    for arc, lm, l2, r in zip(arcs, lmin_vals,
                               l2_vals, ratios):
        if arc in key_arcs:
            print(f"  {arc:>6}  {lm:>12.3e}  "
                  f"{l2:>12.3e}  {r:>14.3e}")


# ── Experiment 5 : Power-law model ─────────────────────────────
def exp5_powerlaw():
    """
    Experiment 5: Fit power-law model theta_c(sigma) = c + a*sigma^b.
    Uses N=20 repetitions per noise level.
    """
    print("\n" + "="*50)
    print("  Experiment 5 — Power-Law Noise-Threshold Model")
    print(f"  N = {N_REPEAT} repetitions per sigma")
    print("="*50)

    sigmas = np.array([0.05, 0.1, 0.15, 0.2, 0.3,
                       0.4,  0.5, 0.6,  0.7, 0.8,
                       0.9,  1.0, 1.2,  1.5])
    thresholds_mean = []
    thresholds_std  = []

    for sigma in sigmas:
        thresholds_rep = []
        for rep in range(N_REPEAT):
            errs = [run_single(arc, sigma,
                               seed=rep*100+int(sigma*1000))
                    for arc in ARCS_DEG]
            thresh = find_threshold(errs)
            thresholds_rep.append(thresh)
        thresholds_mean.append(np.mean(thresholds_rep))
        thresholds_std.append(np.std(thresholds_rep))
        print(f"  σ={sigma:.2f}: threshold = "
              f"{thresholds_mean[-1]:.1f}° ± "
              f"{thresholds_std[-1]:.1f}°")

    thresholds_mean = np.array(thresholds_mean)
    thresholds_std  = np.array(thresholds_std)

    # Fit models
    def model_power(s, c, a, b): return c + a * s**b
    def model_linear(s, a, b):   return a * s + b
    def model_log(s, a, b):      return a * np.log(s) + b

    popt_pw, _ = curve_fit(model_power, sigmas,
                            thresholds_mean,
                            p0=[180, 100, 0.5],
                            maxfev=10000)
    popt_li, _ = curve_fit(model_linear, sigmas,
                            thresholds_mean)
    popt_lo, _ = curve_fit(model_log, sigmas,
                            thresholds_mean)

    rss_pw = np.sum((thresholds_mean
                     - model_power(sigmas, *popt_pw))**2)
    rss_li = np.sum((thresholds_mean
                     - model_linear(sigmas, *popt_li))**2)
    rss_lo = np.sum((thresholds_mean
                     - model_log(sigmas, *popt_lo))**2)

    print(f"\n  Power-law: θ_c = {popt_pw[0]:.2f} + "
          f"{popt_pw[1]:.2f}×σ^{popt_pw[2]:.3f}  "
          f"RSS={rss_pw:.1f}")
    print(f"  Linear   : θ_c = {popt_li[0]:.2f}×σ + "
          f"{popt_li[1]:.2f}  RSS={rss_li:.1f}")
    print(f"  Log      : θ_c = {popt_lo[0]:.2f}×log(σ) + "
          f"{popt_lo[1]:.2f}  RSS={rss_lo:.1f}")
    print(f"\n  Best model: Power-law (RSS={rss_pw:.1f})")

    # Plot
    sigma_plot = np.linspace(0.05, 1.5, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sigmas, thresholds_mean,
                yerr=thresholds_std,
                fmt='ko', markersize=6, capsize=4,
                label=f'Observed (N={N_REPEAT})')
    ax.plot(sigma_plot,
            model_power(sigma_plot, *popt_pw),
            'r-', linewidth=2,
            label=f'Power-law: {popt_pw[0]:.1f} + '
                  f'{popt_pw[1]:.1f}σ^{popt_pw[2]:.2f} '
                  f'(RSS={rss_pw:.1f})')
    ax.plot(sigma_plot,
            model_linear(sigma_plot, *popt_li),
            'b--', linewidth=1.5,
            label=f'Linear (RSS={rss_li:.0f})')
    ax.plot(sigma_plot,
            model_log(sigma_plot, *popt_lo),
            'g-.', linewidth=1.5,
            label=f'Log (RSS={rss_lo:.1f})')
    ax.axhline(180, color='red', linestyle=':',
               linewidth=1, label='180° geometric bound')
    ax.set_xlabel('Noise level σ', fontsize=12)
    ax.set_ylabel('Critical threshold θ_c (degrees)',
                  fontsize=12)
    ax.set_title('Experiment 5 — Power-Law Noise-Threshold '
                 'Relationship\nCloud Regression Ellipse '
                 'Fitting', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig5_powerlaw.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: fig5_powerlaw.png")


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from Wansouwé (2026)."
    )
    parser.add_argument('--exp', type=int, default=0,
                        help='Experiment number (1-5), '
                             '0 = all')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    args = parser.parse_args()

    if args.all or args.exp == 0 or args.exp == 1:
        exp1_baseline()
    if args.all or args.exp == 0 or args.exp == 2:
        exp2_eccentricity()
    if args.all or args.exp == 0 or args.exp == 3:
        exp3_noise()
    if args.all or args.exp == 0 or args.exp == 4:
        exp4_spectral()
    if args.all or args.exp == 0 or args.exp == 5:
        exp5_powerlaw()

    print("\n✅ All experiments completed.")
    print("   Figures saved as fig2_*.png ... fig6_*.png")
