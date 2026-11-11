"""
Module 10 — Monte Carlo Analysis  (monte_carlo.py)
===================================================
Runs 1000 Monte Carlo simulations of the CMOS inverter to model
manufacturing variability.

Variability model
-----------------
In real silicon, threshold voltage (VT) has run-to-run variation due to
random dopant fluctuations, line-edge roughness, etc.

We model this by adding a Gaussian offset to VGS (and VSG) seen by each
transistor at every operating point.  This is equivalent to shifting VT:

    VGS_effective = VGS + ΔVT_n,    ΔVT_n ~ N(0, σ_VT)
    VSG_effective = VSG + ΔVT_p,    ΔVT_p ~ N(0, σ_VT)

σ_VT = 0.03 V  (≈ 5% of typical VT ≈ 0.5 V for 180 nm)

The switching voltage VM is recorded for each run.  The distribution of VM
across 1000 runs directly shows the yield sensitivity to process variation.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from current_predictor import predict_nmos_current, predict_pmos_current
from cmos_solver import VDD, extract_vtc_params

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Monte Carlo settings ───────────────────────────────────────────────────
N_RUNS   = 1000
SIGMA_VT = 0.03     # std-dev of VT variation (V)
N_POINTS = 91       # VIN grid points per run (coarser for speed)
Wn       = 0.5e-6;  Wp = 1.0e-6;  L = 180e-9;  T = 27.0
RNG      = np.random.default_rng(seed=42)


def _solve_vtc_mc(dvt_n, dvt_p):
    """Solve VTC with VT offsets applied to both transistors."""
    vin_arr  = np.linspace(0.0, VDD, N_POINTS)
    vout_arr = np.zeros(N_POINTS)

    for i, vin in enumerate(vin_arr):
        def residual(vout):
            idn = predict_nmos_current(
                VGS=max(vin + dvt_n, 0.0), VDS=max(vout, 0.0),
                Temperature=T, W=Wn, L=L)
            idp = predict_pmos_current(
                VSG=max(VDD - vin + dvt_p, 0.0), VSD=max(VDD - vout, 0.0),
                Temperature=T, W=Wp, L=L)
            return idn - idp

        try:
            vout_arr[i] = brentq(residual, 0.0, VDD,
                                  xtol=1e-5, rtol=1e-5, maxiter=80)
        except ValueError:
            vout_arr[i] = 0.0 if residual(0.0) > 0 else VDD

    return vin_arr, vout_arr


def run_monte_carlo(n_runs=N_RUNS, save=True):
    """
    Run Monte Carlo simulation.

    Returns
    -------
    all_vout : np.ndarray (n_runs × N_POINTS)
    vm_list  : np.ndarray (n_runs,)  switching voltages
    vin_arr  : np.ndarray (N_POINTS,)
    """
    print(f"[Monte Carlo] Running {n_runs} simulations (σ_VT={SIGMA_VT*1e3:.0f} mV) …")
    dvt_n_all = RNG.normal(0, SIGMA_VT, n_runs)
    dvt_p_all = RNG.normal(0, SIGMA_VT, n_runs)

    vin_arr  = np.linspace(0.0, VDD, N_POINTS)
    all_vout = np.zeros((n_runs, N_POINTS))
    vm_list  = np.zeros(n_runs)

    for i in range(n_runs):
        if (i + 1) % 100 == 0:
            print(f"  Run {i+1}/{n_runs} …")
        _, vout = _solve_vtc_mc(dvt_n_all[i], dvt_p_all[i])
        all_vout[i] = vout
        # VM: crossing of VIN = VOUT
        diff = vout - vin_arr
        idx  = np.argmin(np.abs(diff))
        vm_list[i] = vin_arr[idx]

    print(f"[Monte Carlo] Done.  VM: mean={vm_list.mean():.3f}V  "
          f"σ={vm_list.std():.4f}V  range=[{vm_list.min():.3f}, {vm_list.max():.3f}]V")

    if save:
        _plot_vtc_spread(vin_arr, all_vout)
        _plot_vm_histogram(vm_list)

    return all_vout, vm_list, vin_arr


def _plot_vtc_spread(vin_arr, all_vout):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # plot all runs in light grey
    for i in range(all_vout.shape[0]):
        ax.plot(vin_arr, all_vout[i], color="steelblue", alpha=0.04, lw=0.5)

    # median and ±3σ envelope
    med  = np.median(all_vout, axis=0)
    p1   = np.percentile(all_vout, 0.135, axis=0)   # -3σ ≈ 0.135 %ile
    p99  = np.percentile(all_vout, 99.865, axis=0)  # +3σ

    ax.plot(vin_arr, med, "r-",  lw=2,   label="Median VTC")
    ax.plot(vin_arr, p1,  "k--", lw=1.2, label="±3σ envelope")
    ax.plot(vin_arr, p99, "k--", lw=1.2)
    ax.fill_between(vin_arr, p1, p99, alpha=0.15, color="orange")

    ax.plot([0, VDD], [0, VDD], "k:", lw=1, alpha=0.4)
    ax.set_xlim(0, VDD);  ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("VOUT (V)")
    ax.set_title(f"Monte Carlo VTC Spread ({N_RUNS} runs, σ_VT={SIGMA_VT*1e3:.0f} mV)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mc_vtc_spread.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] MC VTC spread → {path}")


def _plot_vm_histogram(vm_list):
    mu  = vm_list.mean()
    sig = vm_list.std()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.hist(vm_list, bins=40, color="steelblue", edgecolor="k",
            alpha=0.75, density=True, label="MC histogram")

    # overlay Gaussian fit
    x = np.linspace(vm_list.min(), vm_list.max(), 300)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sig), "r-", lw=2,
            label=f"Gaussian fit\nμ={mu:.3f}V, σ={sig:.4f}V")

    ax.axvline(mu,          color="red",    lw=1.5, ls="--")
    ax.axvline(mu - 3*sig,  color="orange", lw=1.0, ls=":")
    ax.axvline(mu + 3*sig,  color="orange", lw=1.0, ls=":", label="±3σ limits")

    ax.set_xlabel("Switching Voltage VM (V)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Monte Carlo: Distribution of VM")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mc_vm_histogram.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] VM histogram → {path}")


if __name__ == "__main__":
    run_monte_carlo(n_runs=200)   # use fewer runs for quick test