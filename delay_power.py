"""
Module 7 — Delay and Power Analysis  (delay_power.py)
======================================================
Computes propagation delay, dynamic power, energy, and EDP for all gates.

Equations
---------
  tp   = CL × VDD / ION          [propagation delay, seconds]
  P    = α × CL × VDD² × f      [dynamic switching power, Watts]
  E    = P / f                   [energy per cycle = α × CL × VDD²]
  EDP  = E × tp                  [energy-delay product, J·s]

where:
  CL   = load capacitance (F)
  VDD  = 1.8 V
  α    = 0.5  (activity factor — toggles every other cycle on average)
  f    = clock frequency (Hz)
  ION  = on-state current at VGS = VDS = VDD (from the ML model)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from current_predictor import predict_nmos_current, predict_pmos_current
from cmos_solver import VDD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
ALPHA   = 0.5          # activity factor
T_NOM   = 27.0         # nominal temperature (°C)
Wn      = 0.5e-6
Wp      = 2.5e-6
L       = 180e-9

# First-order 180 nm switched parasitic capacitances (fF).  They represent
# output diffusion, interconnect, and internal nodes that toggle during a
# transition; replace them with extracted parasitics for sign-off work.
GATE_INTRINSIC_CAP_FF = {
    "INV":  0.75,
    "NAND": 1.30,
    "NOR":  0.95,
    "AND":  2.05,  # NAND stage plus output inverter
    "OR":   1.70,  # NOR stage plus output inverter
    "XOR":  2.80,
}

# Gate effective widths (from gates.py sizing)
GATE_WN = {
    "INV":  Wn,
    "NAND": Wn / 2,
    "NOR":  2 * Wn,
    "AND":  Wn / 2,
    "OR":   2 * Wn,
    "XOR":  2 * Wn,
}
GATE_WP = {
    "INV":  Wp,
    "NAND": 2 * Wp,
    "NOR":  Wp / 2,
    "AND":  2 * Wp,
    "OR":   Wp / 2,
    "XOR":  2 * Wp,
}


def get_ion(gate="INV", T=T_NOM, Wn_base=Wn, Wp_base=Wp, Ln=L, Lp=L):
    """
    ION = min(IDN_on, IDP_on) — the weaker transistor limits drive.
    VGS = VDS = VDD for NMOS, VSG = VSD = VDD for PMOS.
    """
    if gate not in GATE_WN:
        raise ValueError(f"Unknown gate: {gate}")
    if min(Wn_base, Wp_base, Ln, Lp) <= 0:
        raise ValueError("Transistor widths and lengths must be positive")
    wn = GATE_WN[gate] / Wn * Wn_base
    wp = GATE_WP[gate] / Wp * Wp_base
    idn = predict_nmos_current(VGS=VDD, VDS=VDD, Temperature=T, W=wn, L=Ln)
    idp = predict_pmos_current(VSG=VDD, VSD=VDD, Temperature=T, W=wp, L=Lp)
    return min(idn, idp)


def propagation_delay(
    CL, gate="INV", T=T_NOM, Wn_base=Wn, Wp_base=Wp, Ln=L, Lp=L,
):
    """tp = CL × VDD / ION  (seconds)"""
    ion = get_ion(gate, T, Wn_base, Wp_base, Ln, Lp)
    return (switched_capacitance(CL, gate) * VDD) / max(ion, 1e-15)


def switched_capacitance(CL, gate="INV"):
    """Total switched capacitance: user load plus gate intrinsic capacitance."""
    if gate not in GATE_INTRINSIC_CAP_FF:
        raise ValueError(f"Unknown gate: {gate}")
    if CL < 0:
        raise ValueError("CL must be non-negative")
    return CL + GATE_INTRINSIC_CAP_FF[gate] * 1e-15


def dynamic_power(CL, f, gate="INV", alpha=ALPHA):
    """P = α × CL × VDD² × f  (Watts)"""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * switched_capacitance(CL, gate) * (VDD ** 2) * f


def energy_per_cycle(CL, gate="INV", alpha=ALPHA):
    """E = α × CL × VDD²  (Joules) — independent of gate type"""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * switched_capacitance(CL, gate) * (VDD ** 2)


def edp(
    CL, gate="INV", f=1e9, T=T_NOM, alpha=ALPHA,
    Wn_base=Wn, Wp_base=Wp, Ln=L, Lp=L,
):
    """EDP = E × tp  (J·s)"""
    tp = propagation_delay(CL, gate, T, Wn_base, Wp_base, Ln, Lp)
    E  = energy_per_cycle(CL, gate, alpha)
    return E * tp


# ── sweep functions ────────────────────────────────────────────────────────

def sweep_delay_vs_CL(gates=None, CL_range=None, T=T_NOM):
    if gates    is None: gates    = ["INV", "NAND", "NOR", "AND", "OR", "XOR"]
    if CL_range is None: CL_range = np.logspace(-15, -12, 60)   # 1 fF → 1 pF

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    results = {}
    for gate, clr in zip(gates, colors):
        delays = [propagation_delay(cl, gate, T) * 1e12 for cl in CL_range]   # → ps
        ax.loglog(CL_range * 1e15, delays, lw=2, color=clr, label=gate)
        results[gate] = delays

    ax.set_xlabel("Load Capacitance CL (fF)")
    ax.set_ylabel("Propagation Delay tp (ps)")
    ax.set_title("Propagation Delay vs Load Capacitance")
    ax.legend(fontsize=9);  ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "delay_vs_CL.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] Delay vs CL → {path}")
    return CL_range, results


def sweep_power_vs_freq(CL=10e-15, gates=None):
    if gates is None: gates = ["INV", "NAND", "NOR", "AND", "OR", "XOR"]
    freqs  = np.logspace(6, 9, 60)   # 1 MHz → 1 GHz
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for gate, clr in zip(gates, colors):
        powers = [dynamic_power(CL, f, gate) * 1e6 for f in freqs]   # → µW
        ax.loglog(freqs / 1e6, powers, lw=2, color=clr, label=gate)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Dynamic Power (µW)")
    ax.set_title(f"Dynamic Power vs Frequency  (CL={CL*1e15:.0f} fF)")
    ax.legend(fontsize=9);  ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "power_vs_freq.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] Power vs Freq → {path}")


def plot_edp_comparison(CL=10e-15, f=1e9, gates=None, T=T_NOM):
    if gates is None: gates = ["INV", "NAND", "NOR", "AND", "OR", "XOR"]
    edps   = [edp(CL, g, f, T) * 1e27 for g in gates]   # → aJ·ns (×1e27)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(gates, edps, color=colors, edgecolor="k", width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("EDP (×10⁻²⁷ J·s)")
    ax.set_title(f"Energy-Delay Product Comparison  (CL={CL*1e15:.0f} fF, f={f/1e9:.0f} GHz)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "edp_comparison.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] EDP comparison → {path}")


def compute_performance_table(
    CL=10e-15, f=1e9, T=T_NOM, alpha=ALPHA,
    Wn_base=Wn, Wp_base=Wp, Ln=L, Lp=L,
):
    """
    Returns a list of dicts for all gates:
      {Circuit, Gain, Delay_ps, Power_uW, Energy_fJ}
    (Gain filled later from gates.py results)
    """
    gates = ["INV", "NAND", "NOR", "AND", "OR", "XOR"]
    table = []
    for g in gates:
        tp   = propagation_delay(CL, g, T, Wn_base, Wp_base, Ln, Lp)
        P    = dynamic_power(CL, f, g, alpha)
        E    = energy_per_cycle(CL, g, alpha)
        table.append({
            "Circuit":   g,
            "Csw_fF":    round(switched_capacitance(CL, g) * 1e15, 3),
            "Gain":      "—",           # filled by main.py
            "Delay_ps":  round(tp * 1e12, 2),
            "Power_uW":  round(P * 1e6,   4),
            "Energy_fJ": round(E * 1e15,  4),
        })
    return table


if __name__ == "__main__":
    sweep_delay_vs_CL()
    sweep_power_vs_freq()
    plot_edp_comparison()
    table = compute_performance_table()
    print("\nPerformance Table:")
    print(f"{'Circuit':<8} {'Delay(ps)':>10} {'Power(µW)':>10} {'Energy(fJ)':>12}")
    print("─" * 44)
    for row in table:
        print(f"{row['Circuit']:<8} {row['Delay_ps']:>10} "
              f"{row['Power_uW']:>10} {row['Energy_fJ']:>12}")
