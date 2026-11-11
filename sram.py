"""
Module 13 — SRAM Characterization  (sram.py)
=============================================
Models a 6-transistor (6T) SRAM cell and computes:
  • Butterfly curve (Read Static Noise Margin)
  • Static Noise Margin (SNM) = side of largest inscribed square

6T SRAM Topology
----------------
        VDD         VDD
         |           |
        M5(P)       M6(P)
         |           |
    Q ───┼───────────┼─── QB
         |           |
        M1(N)       M3(N)   ← cross-coupled inverters (cell core)
         |           |
        GND         GND

M2(N): access transistor (BL  side, gate = WL)
M4(N): access transistor (BLB side, gate = WL)

Read butterfly curve
--------------------
The cell core is two cross-coupled inverters.
The butterfly curve plots:
    Curve 1: VOUT1 = f(VIN1)   [left inverter:  IN=Q  → OUT=QB]
    Curve 2: VOUT2 = f(VIN2)   [right inverter: IN=QB → OUT=Q]

During a READ, the access transistors slightly disturb the stored node.
We model this by sizing the pull-down NMOS wider (β_ratio = Wn_cell/Wn_acc).
The β-ratio (cell ratio) is set to 1.5, which is typical for 6T SRAM.

SNM = side length of the largest square that fits between the two curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from cmos_solver import solve_vtc, extract_vtc_params, VDD
from current_predictor import predict_nmos_current, predict_pmos_current

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── SRAM cell transistor sizing ─────────────────────────────────────────────
BETA_RATIO = 1.5        # pull-down / access transistor W ratio
L          = 180e-9
Wp_cell    = 0.25e-6    # PMOS pull-up; smaller than access device for writeability
Wn_cell    = 0.5e-6     # NMOS pull-down (cell NMOS)
Wn_acc     = Wn_cell / BETA_RATIO   # access transistor

# First-order timing assumptions for a column read and a single cell write.
# These are parameterised because actual values depend on the memory macro,
# bitline length, sense amplifier, and wordline driver.
C_BITLINE_DEFAULT = 100e-15  # 100 fF precharged bitline capacitance
C_STORAGE_DEFAULT = 2e-15    # 2 fF storage-node capacitance
READ_SENSE_DELTA  = 0.10     # 100 mV bitline differential for sensing


def _solve_cell_vtc(T=27.0):
    """
    Solve the VTC of ONE inverter in the cross-coupled cell.
    The access transistor degrades pull-down by reducing effective Wn.
    Effective Wn_eff ≈ Wn_cell || Wn_acc (series → harmonic mean).
    """
    Wn_eff = (Wn_cell * Wn_acc) / (Wn_cell + Wn_acc)
    vin, vout = solve_vtc(T=T, Wn=Wn_eff, Ln=L, Wp=Wp_cell, Lp=L)
    return vin, vout


def butterfly_curve(T=27.0, save=True):
    """
    Generate the SRAM butterfly curve and compute SNM.

    Returns
    -------
    snm : float  (Volts)
    vin, vout1, vout2 : np.ndarray
    """
    vin, vout1 = _solve_cell_vtc(T)

    # Mirror: second inverter has Q as output, QB as input
    # So curve 2 is the inverse mapping: VIN2 = vout1, VOUT2 = vin
    # Interpolate to get a common grid
    vout2 = np.interp(vin, vout1[::-1], vin[::-1])

    snm = _conservative_snm(vin, vout1)
    print(f"[SRAM] SNM = {snm*1e3:.1f} mV  (beta={BETA_RATIO}, T={T} C)")

    if save:
        _plot_butterfly(vin, vout1, vout2, snm, T)

    return snm, vin, vout1, vout2


def _conservative_snm(vin, vout):
    """Conservative DC SNM from the cell inverter noise margins.

    The old minimum-gap method always included the butterfly intersections,
    forcing SNM to zero.  For this compact model, the smaller of NML/NMH is
    a stable, conservative read-SNM estimate.  A true largest-square SNM
    requires a transistor-level read-disturb sweep.
    """
    params = extract_vtc_params(vin, vout)
    return float(max(0.0, min(params["NML"], params["NMH"])))


def _compute_snm(vin, vout1, vout2):
    """
    SNM = largest square inscribed in the eye of the butterfly.
    Approximated as the minimum vertical distance between the two curves
    divided by √2 (the diagonal of a square).
    """
    gap = np.abs(vout1 - vout2)
    snm_approx = gap.min() / np.sqrt(2)
    return float(snm_approx)


def _plot_butterfly(vin, vout1, vout2, snm, T):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.plot(vin,   vout1, "b-",  lw=2, label="INV1: Q→QB")
    ax.plot(vout1, vin,   "r-",  lw=2, label="INV2: QB→Q")   # mirrored

    # SNM square (schematic)
    vm1 = extract_vtc_params(vin, vout1)["VM"]
    sq  = snm
    rect = plt.Rectangle((vm1 - sq/2, vm1 - sq/2), sq, sq,
                          linewidth=1.5, edgecolor="green",
                          facecolor="green", alpha=0.15,
                          label=f"SNM ≈ {snm*1e3:.1f} mV")
    ax.add_patch(rect)

    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, VDD);  ax.set_ylim(0, VDD)
    ax.set_xlabel("Node Q (V)");  ax.set_ylabel("Node QB (V)")
    ax.set_title(f"6T SRAM Butterfly Curve  (T={T}°C, β={BETA_RATIO})")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sram_butterfly.png")
    plt.savefig(path, dpi=300);  plt.close()
    print(f"[Plot] Butterfly curve -> {path}")


def sram_read_time(T=27.0, C_bitline=C_BITLINE_DEFAULT,
                   sense_delta=READ_SENSE_DELTA):
    """Estimate bitline read time to develop the requested sense voltage."""
    if C_bitline <= 0 or not 0 < sense_delta < VDD:
        raise ValueError("C_bitline must be positive and sense_delta must be within (0, VDD)")
    i_pd = predict_nmos_current(VGS=VDD, VDS=VDD, Temperature=T, W=Wn_cell, L=L)
    i_acc = predict_nmos_current(VGS=VDD, VDS=VDD, Temperature=T, W=Wn_acc, L=L)
    # Access and pull-down are in series; harmonic mean is a first-order limit.
    i_read = (i_pd * i_acc) / max(i_pd + i_acc, 1e-18)
    return {
        "read_current_A": i_read,
        "bitline_cap_F": C_bitline,
        "sense_delta_V": sense_delta,
        "read_time_s": C_bitline * sense_delta / max(i_read, 1e-18),
    }


def sram_write_time(T=27.0, C_storage=C_STORAGE_DEFAULT):
    """Estimate time to write a stored high node low through one access FET."""
    if C_storage <= 0:
        raise ValueError("C_storage must be positive")
    i_acc = predict_nmos_current(VGS=VDD, VDS=VDD / 2, Temperature=T, W=Wn_acc, L=L)
    i_pu = predict_pmos_current(VSG=VDD / 2, VSD=VDD / 2,
                                Temperature=T, W=Wp_cell, L=L)
    i_net = i_acc - i_pu
    writeable = i_net > 0
    return {
        "access_current_A": i_acc,
        "pullup_current_A": i_pu,
        "write_current_A": max(i_net, 0.0),
        "storage_cap_F": C_storage,
        "writeable": writeable,
        "write_time_s": (C_storage * 0.8 * VDD / i_net) if writeable else np.inf,
    }


def sram_hold_metrics(T=27.0):
    """Return hold stability; a static 6T cell has no finite hold-time spec."""
    vin, vout = _solve_cell_vtc(T)
    return {
        "hold_snm_V": _conservative_snm(vin, vout),
        "hold_time_s": np.inf,
        "note": "With WL low, a 6T cell retains data while VDD is applied; use hold SNM, not a flip-flop-style hold time.",
    }


def characterize_sram_timing(T=27.0, C_bitline=C_BITLINE_DEFAULT,
                              C_storage=C_STORAGE_DEFAULT,
                              sense_delta=READ_SENSE_DELTA):
    """Compute read/write access time and static hold stability."""
    read = sram_read_time(
        T=T, C_bitline=C_bitline, sense_delta=sense_delta
    )
    write = sram_write_time(T=T, C_storage=C_storage)
    hold = sram_hold_metrics(T=T)
    write_text = "not writeable" if not write["writeable"] else f"{write['write_time_s']*1e12:.2f} ps"
    print(f"[SRAM Timing] Read={read['read_time_s']*1e12:.2f} ps  Write={write_text}  "
          f"Hold SNM={hold['hold_snm_V']*1e3:.1f} mV")
    return {"read": read, "write": write, "hold": hold}


def snm_vs_temperature(save=True):
    """Plot SNM variation across temperature."""
    temps = [-40, 0, 27, 50, 75, 100, 125]
    snms  = []

    for T in temps:
        snm, _, _, _ = butterfly_curve(T=T, save=False)
        snms.append(snm * 1e3)   # mV
        print(f"  T={T:+4d}°C  SNM={snm*1e3:.1f} mV")

    if save:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.plot(temps, snms, "bo-", ms=6, lw=2)
        ax.set_xlabel("Temperature (°C)");  ax.set_ylabel("SNM (mV)")
        ax.set_title("SRAM Static Noise Margin vs Temperature")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "sram_snm_vs_temp.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] SNM vs temp -> {path}")

    return temps, snms


if __name__ == "__main__":
    butterfly_curve()
    snm_vs_temperature()
