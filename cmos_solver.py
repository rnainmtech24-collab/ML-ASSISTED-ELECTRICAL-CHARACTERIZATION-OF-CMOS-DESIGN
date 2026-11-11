"""
Module 5 — CMOS Inverter Solver  (cmos_solver.py)
===================================================
Solves the CMOS inverter voltage-transfer characteristic (VTC) using
Brent's root-finding method on:

    IDN(VIN, VOUT) = IDP(VIN, VOUT)

at each VIN point, and extracts all standard DC parameters.

Physics recap
-------------
•  NMOS pull-down: VGS_N = VIN,  VDS_N = VOUT
•  PMOS pull-up  : VSG_P = VDD − VIN, VSD_P = VDD − VOUT
•  At DC equilibrium, both currents must be equal.

Parameters extracted
--------------------
VOH   — output high voltage     (VOUT when VIN ≈ 0)
VOL   — output low voltage      (VOUT when VIN ≈ VDD)
VM    — switching threshold     (VIN where VOUT = VIN)
VIL   — input-low threshold     (where dVOUT/dVIN = −1 on low side)
VIH   — input-high threshold    (where dVOUT/dVIN = −1 on high side)
Gain  — maximum |dVOUT/dVIN|
NMH   — noise margin high = VOH − VIH
NML   — noise margin low  = VIL − VOL
"""

import numpy as np
from scipy.optimize import brentq
from current_predictor import predict_nmos_current, predict_pmos_current

VDD = 1.8     # supply voltage (V) — 180 nm CMOS
N_POINTS = 401  # VIN resolution


def _idn(vin, vout, T=27.0, W=0.5e-6, L=180e-9):
    """NMOS drain current for a given (VIN, VOUT) operating point."""
    return predict_nmos_current(VGS=vin, VDS=max(vout, 0.0),
                                Temperature=T, W=W, L=L)


def _idp(vin, vout, T=27.0, W=1.0e-6, L=180e-9):
    """
    PMOS drain current magnitude.
    VSG = VDD − VIN,  VSD = VDD − VOUT
    PMOS W is typically 2× NMOS to equalise drive strength.
    """
    vsg = VDD - vin
    vsd = VDD - vout
    return predict_pmos_current(VSG=max(vsg, 0.0), VSD=max(vsd, 0.0),
                                Temperature=T, W=W, L=L)


def solve_vtc(T=27.0, Wn=0.5e-6, Ln=180e-9,
              Wp=2.5e-6, Lp=180e-9, n_points=N_POINTS):
    """
    Solve VOUT(VIN) for the CMOS inverter.

    Returns
    -------
    vin_arr  : np.ndarray (V)
    vout_arr : np.ndarray (V)
    """
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    if min(Wn, Ln, Wp, Lp) <= 0:
        raise ValueError("Transistor widths and lengths must be positive")

    vin_arr  = np.linspace(0.0, VDD, n_points)
    vout_arr = np.zeros(n_points)

    for i, vin in enumerate(vin_arr):
        def residual(vout):
            return _idn(vin, vout, T, Wn, Ln) - _idp(vin, vout, T, Wp, Lp)

        try:
            # Brent's method: guaranteed convergence if root is bracketed
            vout_arr[i] = brentq(residual, 0.0, VDD,
                                  xtol=1e-6, rtol=1e-6, maxiter=100)
        except ValueError:
            # If no sign change → NMOS dominates or PMOS dominates
            f0 = residual(0.0)
            vout_arr[i] = 0.0 if f0 > 0 else VDD

    return vin_arr, vout_arr


def extract_vtc_params(vin_arr, vout_arr):
    """
    Extract all standard VTC metrics from (VIN, VOUT) arrays.

    Returns
    -------
    dict with keys: VOH, VOL, VM, VIL, VIH, Gain, NMH, NML
    """
    dv = np.gradient(vout_arr, vin_arr)   # dVOUT/dVIN

    # Rising transfer curves (e.g. AND/OR after the output inversion) use
    # the opposite endpoints from an inverter.
    rising = vout_arr[-1] > vout_arr[0]
    VOH = float(np.clip(max(vout_arr[0], vout_arr[-1]), 0.0, VDD))
    VOL = float(np.clip(min(vout_arr[0], vout_arr[-1]), 0.0, VDD))

    if rising:
        # A non-inverting gate's switching point is its mid-output crossing;
        # the diagonal also intersects at the logic-level endpoints.
        idx_vm = int(np.argmin(np.abs(vout_arr - VDD / 2)))
        VM = float(vin_arr[idx_vm])
        target_slope = 1.0
    else:
        # For an inverter, VM is the VIN = VOUT crossing.
        diff = vout_arr - vin_arr
        idx_vm = int(np.argmin(np.abs(diff)))
        VM = float(vin_arr[idx_vm])
        target_slope = -1.0

    # Gain: peak |dVOUT/dVIN|
    Gain = float(np.max(np.abs(dv)))

    # VIL/VIH are found where the transfer slope reaches the logic direction's
    # unity-gain threshold: -1 for inverting gates, +1 for non-inverting gates.
    left_mask = vin_arr < VM
    dv_left   = dv[left_mask]
    vin_left  = vin_arr[left_mask]
    cross_l   = np.where(np.diff(np.sign(dv_left - target_slope)))[0]
    VIL = float(vin_left[cross_l[-1]]) if len(cross_l) else vin_arr[0]

    # VIH: smallest VIN where the slope crosses the same threshold to the right.
    right_mask = vin_arr > VM
    dv_right  = dv[right_mask]
    vin_right = vin_arr[right_mask]
    cross_r   = np.where(np.diff(np.sign(dv_right - target_slope)))[0]
    VIH = float(vin_right[cross_r[0]]) if len(cross_r) else vin_arr[-1]

    # ensure VIL < VM < VIH
    VIL = min(VIL, VM)
    VIH = max(VIH, VM)
    
    NMH = float(VOH - VIH)
    NML = float(VIL - VOL)

    return {
        "VOH": VOH, "VOL": VOL, "VM": VM,
        "VIL": VIL, "VIH": VIH, "Gain": Gain,
        "NMH": NMH, "NML": NML,
    }