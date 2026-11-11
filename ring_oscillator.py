"""
Module 11 — Ring Oscillator  (ring_oscillator.py)
===================================================
Models a CMOS ring oscillator built from N cascaded inverter stages
(N must be odd for guaranteed oscillation).

Physics
-------
Each inverter contributes a propagation delay tp.
Total delay for one half-period = N × tp.
Oscillation frequency:

    f_osc = 1 / (2 × N × tp)

tp is computed from the ML-predicted ION current:

    tp = CL × VDD / ION

where CL includes both intrinsic gate capacitance and wiring.

Fan-out-of-4 (FO4) delay: tp_FO4 = tp × (1 + 4 × Cg/CL)
is used as a technology figure of merit.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delay_power import propagation_delay, get_ion
from cmos_solver import VDD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CL_DEFAULT = 10e-15   # 10 fF intrinsic + wiring capacitance


def ring_oscillator_freq(
    N, CL=CL_DEFAULT, T=27.0, Wn=0.5e-6, Wp=2.5e-6,
    Ln=180e-9, Lp=180e-9,
):
    """
    Compute ring oscillator frequency for N stages.

    Parameters
    ----------
    N   : int   — number of inverter stages (odd)
    CL  : float — load capacitance per stage (F)
    T   : float — temperature (°C)

    Returns
    -------
    f_osc : float (Hz)
    tp    : float (s)
    """
    if N % 2 == 0:
        raise ValueError(f"N must be odd for a ring oscillator, got N={N}")
    tp    = propagation_delay(
        CL, gate="INV", T=T, Wn_base=Wn, Wp_base=Wp, Ln=Ln, Lp=Lp
    )
    f_osc = 1.0 / (2 * N * tp)
    return f_osc, tp


def sweep_ring_oscillator(stages=None, CL=CL_DEFAULT, T=27.0, save=True):
    """
    Sweep over odd number of stages and plot f vs N.

    Returns
    -------
    stages_arr : np.ndarray
    freqs_GHz  : np.ndarray
    """
    if stages is None:
        stages = [3, 5, 7, 9, 11, 13, 15, 21, 31, 51]

    freqs_GHz = []
    tps_ps    = []

    print(f"\n[Ring Oscillator] CL={CL*1e15:.0f} fF, T={T}°C")
    print(f"{'Stages':>8} {'tp (ps)':>10} {'f_osc (GHz)':>14}")
    print("─" * 36)

    for N in stages:
        f, tp = ring_oscillator_freq(N, CL, T)
        freqs_GHz.append(f / 1e9)
        tps_ps.append(tp * 1e12)
        print(f"{N:>8}   {tp*1e12:>9.3f}   {f/1e9:>12.4f}")

    stages_arr = np.array(stages)
    freqs_arr  = np.array(freqs_GHz)

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

        # f vs N
        axes[0].plot(stages_arr, freqs_arr, "bo-", ms=6, lw=2)
        axes[0].set_xlabel("Number of Stages (N)")
        axes[0].set_ylabel("Oscillation Frequency (GHz)")
        axes[0].set_title("Ring Oscillator: Frequency vs Stages")
        axes[0].grid(True, alpha=0.3)
        for i, (n, f) in enumerate(zip(stages_arr, freqs_arr)):
            axes[0].annotate(f"{f:.3f}", (n, f),
                             textcoords="offset points", xytext=(4, 5),
                             fontsize=7)

        # 1/f (period) vs N — linear for sanity check
        periods_ns = 1.0 / (freqs_arr * 1e9) * 1e9
        axes[1].plot(stages_arr, periods_ns, "rs-", ms=6, lw=2)
        axes[1].set_xlabel("Number of Stages (N)")
        axes[1].set_ylabel("Oscillation Period (ns)")
        axes[1].set_title("Ring Oscillator: Period vs Stages")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "ring_oscillator.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] Ring oscillator → {path}")

    return stages_arr, freqs_arr


def fo4_delay(CL=CL_DEFAULT, T=27.0):
    """
    Fan-Out-of-4 (FO4) inverter delay — standard technology benchmark.
    Approximated as tp × (1 + 4) since FO4 drives 4× the nominal CL.
    """
    tp_nom = propagation_delay(CL, "INV", T)
    tp_fo4 = propagation_delay(CL * 5, "INV", T)   # 4× extra load
    print(f"[RO] FO4 delay: {tp_fo4*1e12:.2f} ps  (nominal tp: {tp_nom*1e12:.2f} ps)")
    return tp_fo4


if __name__ == "__main__":
    sweep_ring_oscillator()
    fo4_delay()