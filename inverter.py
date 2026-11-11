"""
Module 5b — Inverter Characterization & Plots  (inverter.py)
=============================================================
Uses cmos_solver.py to generate the VTC and produce publication-quality
plots for the CMOS inverter.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmos_solver import solve_vtc, extract_vtc_params, VDD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def characterize_inverter(T=27.0, Wn=0.5e-6, Ln=180e-9,
                           Wp=2.5e-6, Lp=180e-9, label="INV", save=True):
    """
    Solve VTC, extract parameters, plot, and return results.

    Returns
    -------
    vin, vout : np.ndarray
    params    : dict  (VOH, VOL, VM, VIL, VIH, Gain, NMH, NML)
    """
    print(f"[Inverter] Solving VTC for {label} at T={T}°C …")
    vin, vout = solve_vtc(T=T, Wn=Wn, Ln=Ln, Wp=Wp, Lp=Lp)
    params    = extract_vtc_params(vin, vout)

    print(f"  VOH={params['VOH']:.3f}V  VOL={params['VOL']:.3f}V  "
          f"VM={params['VM']:.3f}V  Gain={params['Gain']:.1f}")
    print(f"  VIL={params['VIL']:.3f}V  VIH={params['VIH']:.3f}V  "
          f"NMH={params['NMH']:.3f}V  NML={params['NML']:.3f}V")

    if save:
        _plot_vtc(vin, vout, params, label, T)

    return vin, vout, params


def _plot_vtc(vin, vout, params, label="INV", T=27.0):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(vin, vout, "b-", lw=2, label="VTC")

    # diagonal reference (VIN = VOUT)
    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.5, label="VIN = VOUT")

    # Annotate the actual VTC points.  VIL and VIH are input thresholds,
    # so they must be located on the curve rather than at VOH/VOL.
    for name, xval, yval, clr in [
        ("VM",  params["VM"],  params["VM"],  "green"),
        ("VIL", params["VIL"], np.interp(params["VIL"], vin, vout), "orange"),
        ("VIH", params["VIH"], np.interp(params["VIH"], vin, vout), "red"),
    ]:
        ax.axvline(xval, color=clr, lw=0.8, ls=":")
        ax.scatter([xval], [yval], color=clr, zorder=5)
        ax.annotate(name, (xval, yval), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=clr)

    # noise margin shading
    ax.axhspan(params["VOL"], params["VIL"], alpha=0.08, color="blue",
               label=f"NML={params['NML']:.3f}V")
    ax.axhspan(params["VIH"], params["VOH"], alpha=0.08, color="red",
               label=f"NMH={params['NMH']:.3f}V")

    ax.set_xlim(0, VDD);  ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("VOUT (V)")
    ax.set_title(f"CMOS {label} — Voltage Transfer Characteristic  (T={27}°C)")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"vtc_{label.lower()}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Plot] VTC saved → {path}")


def plot_gain_curve(vin, vout, label="INV"):
    """Plot dVOUT/dVIN vs VIN."""
    gain = np.gradient(vout, vin)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.plot(vin, -gain, "r-", lw=2)
    ax.axhline(1, color="k", ls="--", lw=1, label="|Gain| = 1")
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("|dVOUT/dVIN|")
    ax.set_title(f"CMOS {label} — Gain Curve")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"gain_{label.lower()}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Plot] Gain curve saved → {path}")


if __name__ == "__main__":
    vin, vout, params = characterize_inverter()
    plot_gain_curve(vin, vout)
