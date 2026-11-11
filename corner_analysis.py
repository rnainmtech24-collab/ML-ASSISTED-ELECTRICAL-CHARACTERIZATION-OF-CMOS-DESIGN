"""
Module 8 & 9 — Temperature + Process Corner Analysis  (corner_analysis.py)
===========================================================================
Module 8 — Temperature Analysis
    Solves inverter VTC at: -40, 27, 75, 125 °C
    The ML model already learned temperature dependence from the dataset
    (Temperature is a direct input feature), so no physics approximations
    are needed — just sweep T.

Module 9 — Process Corner Analysis
    Five PVT corners: TT, SS, FF, FS, SF
    Implemented by scaling W/L around nominal values:

    Corner | NMOS   | PMOS   | Wn scale | Wp scale | L scale
    -------|--------|--------|----------|----------|--------
    TT     | Typ    | Typ    | 1.00     | 1.00     | 1.00
    SS     | Slow   | Slow   | 0.85     | 0.85     | 1.10  (longer, narrower)
    FF     | Fast   | Fast   | 1.15     | 1.15     | 0.90  (shorter, wider)
    FS     | Fast N | Slow P | 1.15 N   | 0.85 P   | nom
    SF     | Slow N | Fast P | 0.85 N   | 1.15 P   | nom
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

# ── base sizing ─────────────────────────────────────────────────────────────
Wn0 = 0.5e-6;  Wp0 = 1.0e-6;  L0 = 180e-9

# ── temperature sweep ──────────────────────────────────────────────────────
TEMPERATURES = [-40, 27, 75, 125]
TEMP_COLORS  = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]


def temperature_analysis(save=True):
    """Solve VTC at four temperatures, return dict and plot."""
    results = {}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for T, clr in zip(TEMPERATURES, TEMP_COLORS):
        print(f"[Temp Analysis] T = {T:+4d}°C …")
        vin, vout = solve_vtc(T=T, Wn=Wn0, Ln=L0, Wp=Wp0, Lp=L0)
        params    = extract_vtc_params(vin, vout)
        results[T] = {"vin": vin, "vout": vout, "params": params}
        ax.plot(vin, vout, lw=2, color=clr, label=f"T = {T:+d}°C  (VM={params['VM']:.3f}V)")
        print(f"  VM={params['VM']:.3f}V  Gain={params['Gain']:.1f}  "
              f"NMH={params['NMH']:.3f}V  NML={params['NML']:.3f}V")

    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, VDD);  ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("VOUT (V)")
    ax.set_title("CMOS Inverter VTC vs Temperature")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "vtc_temperature.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] Temperature VTC → {path}")
    else:
        return fig, results
    return results


# ── process corner analysis ────────────────────────────────────────────────
CORNERS = {
    "TT": {"wn_sc": 1.00, "wp_sc": 1.00, "l_sc": 1.00},
    "SS": {"wn_sc": 0.85, "wp_sc": 0.85, "l_sc": 1.10},
    "FF": {"wn_sc": 1.15, "wp_sc": 1.15, "l_sc": 0.90},
    "FS": {"wn_sc": 1.15, "wp_sc": 0.85, "l_sc": 1.00},
    "SF": {"wn_sc": 0.85, "wp_sc": 1.15, "l_sc": 1.00},
}
CORNER_COLORS = {
    "TT": "#2ca02c", "SS": "#d62728", "FF": "#1f77b4",
    "FS": "#ff7f0e", "SF": "#9467bd",
}


def corner_analysis(T=27.0, save=True):
    """Solve VTC for all five PVT corners."""
    results = {}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for corner, sc in CORNERS.items():
        Wn = Wn0 * sc["wn_sc"]
        Wp = Wp0 * sc["wp_sc"]
        Ln = Lp = L0 * sc["l_sc"]
        print(f"[Corner] {corner}: Wn={Wn*1e6:.3f}µm  Wp={Wp*1e6:.3f}µm  L={Ln*1e9:.1f}nm")
        vin, vout = solve_vtc(T=T, Wn=Wn, Ln=Ln, Wp=Wp, Lp=Lp)
        params    = extract_vtc_params(vin, vout)
        results[corner] = {"vin": vin, "vout": vout, "params": params}
        ax.plot(vin, vout, lw=2, color=CORNER_COLORS[corner],
                label=f"{corner}  (VM={params['VM']:.3f}V)")
        print(f"  VM={params['VM']:.3f}V  Gain={params['Gain']:.1f}  "
              f"NMH={params['NMH']:.3f}V  NML={params['NML']:.3f}V")

    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, VDD);  ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("VOUT (V)")
    ax.set_title(f"CMOS Inverter VTC — Process Corners (T={T}°C)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "vtc_corners.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] Corner VTC → {path}")
    return results


def plot_corner_bar(corner_results):
    """Bar chart comparing Gain, Delay proxy, across corners."""
    corners = list(corner_results.keys())
    gains   = [corner_results[c]["params"]["Gain"] for c in corners]
    vmths   = [corner_results[c]["params"]["VM"]   for c in corners]
    colors  = [CORNER_COLORS[c] for c in corners]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    for ax, vals, ylabel, title in zip(
        axes,
        [gains, vmths],
        ["Max Gain", "VM (V)"],
        ["Gain across Corners", "Switching Threshold VM across Corners"],
    ):
        bars = ax.bar(corners, vals, color=colors, edgecolor="k", width=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel);  ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "corner_bar.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] Corner bar chart → {path}")


if __name__ == "__main__":
    temperature_analysis()
    cr = corner_analysis()
    plot_corner_bar(cr)