"""
Module 6 — Logic Gate Characterization  (gates.py)
====================================================
Models INV, NAND, NOR, AND, OR, XOR by adjusting effective transistor
W/L ratios to mimic series/parallel stacking, then reuses the CMOS
inverter VTC solver for each configuration.

Transistor sizing rules (180 nm CMOS):
----------------------------------------
Gate   | NMOS config        | PMOS config        | Wn_eff | Wp_eff
-------|--------------------|--------------------|---------|---------
INV    | 1× single          | 1× single          | Wn      | Wp
NAND2  | 2 series           | 2 parallel         | Wn/2   | 2×Wp
NOR2   | 2 parallel         | 2 series           | 2×Wn   | Wp/2
AND    | NAND2 + INV        | (composite)        | —       | —
OR     | NOR2  + INV        | (composite)        | —       | —
XOR    | complex; approx 2× | complex            | 2×Wn   | 2×Wp

Series stacking halves drive strength (W_eff = W/N_series).
Parallel stacking doubles drive strength (W_eff = N_parallel × W).

The VTC of AND / OR / XOR is approximated by chaining two
single-variable VTCs (worst-case input sweep with one input toggling).
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
Wn_base = 0.5e-6   # NMOS width
Wp_base = 2.5e-6   # PMOS width
L_base  = 180e-9   # common channel length

# Gate sizing table: (Wn_eff, Wp_eff, Ln, Lp, description)
GATE_CONFIG = {
    "INV":  (Wn_base,       Wp_base,       L_base, L_base, "Single inverter"),
    "NAND": (Wn_base / 2,   2 * Wp_base,   L_base, L_base, "2-input NAND (series N, parallel P)"),
    "NOR":  (2 * Wn_base,   Wp_base / 2,   L_base, L_base, "2-input NOR  (parallel N, series P)"),
    "AND":  (Wn_base / 2,   2 * Wp_base,   L_base, L_base, "NAND+INV → AND"),
    "OR":   (2 * Wn_base,   Wp_base / 2,   L_base, L_base, "NOR+INV → OR"),
    "XOR":  (2 * Wn_base,   2 * Wp_base,   L_base, L_base, "XOR (approx 2× drive)"),
}

# Gates that require an extra inverter stage at output (AND, OR)
_DOUBLE_INVERSION = {"AND", "OR"}


def _invert_vtc(vout_stage1):
    """Apply a second ideal inversion to VOUT of stage 1 (for AND/OR)."""
    return VDD - vout_stage1


def characterize_gate(name, T=27.0):
    """
    Solve VTC for a named gate, extract parameters, return results.

    Returns
    -------
    vin, vout : np.ndarray
    params    : dict
    """
    Wn, Wp, Ln, Lp, desc = GATE_CONFIG[name]
    print(f"[Gate] {name}: {desc} @ T={T}°C")

    vin, vout = solve_vtc(T=T, Wn=Wn, Ln=Ln, Wp=Wp, Lp=Lp)

    if name in _DOUBLE_INVERSION:
        # AND = NAND → INV,  OR = NOR → INV
        # Model the second INV as an ideal inversion of the first stage VTC
        vout = _invert_vtc(vout)

    params = extract_vtc_params(vin, vout)
    print(f"  VM={params['VM']:.3f}V  Gain={params['Gain']:.1f}  "
          f"NMH={params['NMH']:.3f}V  NML={params['NML']:.3f}V")
    return vin, vout, params


def characterize_all_gates(T=27.0, save_plots=True):
    """
    Characterize all six gates and return a summary dict.

    Returns
    -------
    results : dict  {gate_name: {"vin": …, "vout": …, "params": …}}
    """
    results = {}
    for name in GATE_CONFIG:
        vin, vout, params = characterize_gate(name, T=T)
        results[name] = {"vin": vin, "vout": vout, "params": params}

    if save_plots:
        _plot_all_vtc(results)
        _plot_gain_comparison(results)

    return results


def _plot_all_vtc(results):
    """Overlay all gate VTCs on one figure."""
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for (name, data), clr in zip(results.items(), colors):
        ax.plot(data["vin"], data["vout"], lw=2, color=clr, label=name)

    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4, label="VIN=VOUT")
    ax.set_xlim(0, VDD);  ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("VIN (V)");  ax.set_ylabel("VOUT (V)")
    ax.set_title("CMOS Logic Gates — VTC Comparison")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "vtc_all_gates.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] All-gate VTC → {path}")


def _plot_gain_comparison(results):
    """Bar chart: max gain of each gate."""
    names  = list(results.keys())
    gains  = [results[n]["params"]["Gain"] for n in names]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(names, gains, color=colors, edgecolor="k", width=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Maximum Gain |dVOUT/dVIN|")
    ax.set_title("Logic Gate Gain Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gain_comparison.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] Gain comparison → {path}")


def print_gate_table(results):
    """Print a formatted performance table to console."""
    print("\n{'─'*62}")
    print(f"{'Gate':<8} {'VM(V)':>7} {'Gain':>7} {'NMH(V)':>8} {'NML(V)':>8}")
    print("─" * 42)
    for name, data in results.items():
        p = data["params"]
        print(f"{name:<8} {p['VM']:>7.3f} {p['Gain']:>7.1f} "
              f"{p['NMH']:>8.3f} {p['NML']:>8.3f}")
    print("─" * 42)


if __name__ == "__main__":
    res = characterize_all_gates()
    print_gate_table(res)