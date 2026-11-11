"""
Module 12 — D Flip-Flop Timing Model  (dff.py)
================================================
Models a master-slave edge-triggered D flip-flop (DFF) as a chain of
cascaded inverter delays, which is the standard first-order timing model
used in pre-layout VLSI analysis.

DFF internal structure (master-slave latch):
    D → [INV1] → [INV2] → [INV3] → [INV4] → Q
         (master latch)    (slave latch)

Timing definitions (JEDEC standard)
-------------------------------------
  t_setup  : data must be stable BEFORE clock edge
             ≈ 2 × tp  (master latch must capture D)

  t_hold   : data must remain stable AFTER clock edge
             ≈ 1 × tp  (feedback path in master latch)

  t_cq     : clock-to-Q propagation delay
             ≈ 4 × tp  (full master-slave pipeline)

These are analytical estimates; SPICE or post-layout extraction gives
exact values.  For a system clock:

    f_max = 1 / (t_cq + t_comb + t_setup)

where t_comb is the combinational logic delay between flip-flops.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delay_power import propagation_delay
from ring_oscillator import ring_oscillator_freq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def dff_timing(CL=10e-15, T=27.0, Wn=0.5e-6, Wp=2.5e-6,
               Ln=180e-9, Lp=180e-9):
    """
    Compute DFF timing parameters.

    Parameters
    ----------
    CL : float — load capacitance (F)
    T  : float — temperature (°C)

    Returns
    -------
    dict with keys: tp_ps, t_setup_ps, t_hold_ps, t_cq_ps, f_max_GHz
    """
    tp = propagation_delay(
        CL, gate="INV", T=T, Wn_base=Wn, Wp_base=Wp, Ln=Ln, Lp=Lp
    )

    t_setup = 2.0 * tp
    t_hold  = 1.0 * tp
    t_cq    = 4.0 * tp

    # Maximum clock frequency (no combinational logic between FFs)
    f_max = 1.0 / (t_cq + t_setup)

    results = {
        "tp_ps":       tp     * 1e12,
        "t_setup_ps":  t_setup * 1e12,
        "t_hold_ps":   t_hold  * 1e12,
        "t_cq_ps":     t_cq    * 1e12,
        "f_max_GHz":   f_max   / 1e9,
    }

    print("\n[DFF Timing]")
    print(f"  Inverter tp  : {results['tp_ps']:.2f} ps")
    print(f"  Setup time   : {results['t_setup_ps']:.2f} ps")
    print(f"  Hold  time   : {results['t_hold_ps']:.2f} ps")
    print(f"  Clock-to-Q   : {results['t_cq_ps']:.2f} ps")
    print(f"  Max frequency: {results['f_max_GHz']:.3f} GHz")

    return results


def plot_dff_timing(results, save=True):
    """Visualise DFF timing as a bar chart."""
    labels = ["tp\n(inverter)", "t_setup", "t_hold", "t_clk-Q"]
    values = [
        results["tp_ps"],
        results["t_setup_ps"],
        results["t_hold_ps"],
        results["t_cq_ps"],
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(labels, values, color=colors, edgecolor="k", width=0.5)
    ax.bar_label(bars, fmt="%.1f ps", padding=3, fontsize=9)
    ax.set_ylabel("Time (ps)")
    ax.set_title("D Flip-Flop Timing Parameters (180 nm CMOS)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "dff_timing.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] DFF timing → {path}")
    return fig


def dff_fmax_vs_CL(save=True):
    """Plot maximum DFF frequency vs load capacitance."""
    CL_range = np.logspace(-15, -12, 50)   # 1 fF → 1 pF
    f_max    = []
    for cl in CL_range:
        tp    = propagation_delay(cl, "INV")
        t_cq  = 4 * tp;  t_setup = 2 * tp
        f_max.append(1.0 / (t_cq + t_setup) / 1e9)   # GHz

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.semilogx(CL_range * 1e15, f_max, "b-", lw=2)
    ax.set_xlabel("Load Capacitance CL (fF)")
    ax.set_ylabel("Maximum Clock Frequency (GHz)")
    ax.set_title("DFF Maximum Frequency vs Load Capacitance")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "dff_fmax_vs_CL.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] DFF fmax vs CL → {path}")


if __name__ == "__main__":
    res = dff_timing()
    plot_dff_timing(res)
    dff_fmax_vs_CL()