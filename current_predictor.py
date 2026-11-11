"""
Module 4 — Current Predictor  (current_predictor.py)
=====================================================
Wraps trained NMOS / PMOS models and their saved scalers into simple,
callable Python functions.  All downstream modules (inverter solver,
gate characterizer, …) import from here — never from the training scripts.

Pipeline (both functions):
    raw input → StandardScaler (X) → model → StandardScaler⁻¹ (y)
    → 10^(output) → IDS in Amperes (positive float)
"""

import os
import numpy as np
import torch
import joblib

from models import build_nmos_model, build_pmos_model

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")
DEVICE     = torch.device("cpu")   # inference always on CPU for portability

# ── lazy-loaded singletons ─────────────────────────────────────────────────
_nmos_model   = None
_pmos_model   = None
_nmos_scaler_X = None
_nmos_scaler_y = None
_pmos_scaler_X = None
_pmos_scaler_y = None


def _load_nmos():
    global _nmos_model, _nmos_scaler_X, _nmos_scaler_y
    if _nmos_model is None:
        _nmos_model = build_nmos_model().to(DEVICE)
        _nmos_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "nmos_model.pth"),
                       map_location=DEVICE))
        _nmos_model.eval()
        _nmos_scaler_X = joblib.load(os.path.join(SCALER_DIR, "nmos_scaler_X.pkl"))
        _nmos_scaler_y = joblib.load(os.path.join(SCALER_DIR, "nmos_scaler_y.pkl"))


def _load_pmos():
    global _pmos_model, _pmos_scaler_X, _pmos_scaler_y
    if _pmos_model is None:
        _pmos_model = build_pmos_model().to(DEVICE)
        _pmos_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "pmos_model.pth"),
                       map_location=DEVICE))
        _pmos_model.eval()
        _pmos_scaler_X = joblib.load(os.path.join(SCALER_DIR, "pmos_scaler_X.pkl"))
        _pmos_scaler_y = joblib.load(os.path.join(SCALER_DIR, "pmos_scaler_y.pkl"))


# ── public API ─────────────────────────────────────────────────────────────

def predict_nmos_current(VGS, VDS, Temperature=27.0,
                         W=0.5e-6, L=180e-9) -> float:
    """
    Predict NMOS drain current IDS.

    Parameters
    ----------
    VGS         : Gate-Source voltage (V)
    VDS         : Drain-Source voltage (V)
    Temperature : Junction temperature (°C), default 27
    W           : Channel width  (m), default 0.5 µm
    L           : Channel length (m), default 180 nm

    Returns
    -------
    IDS (A) — always ≥ 0
    """
    _load_nmos()
    x = np.array([[VGS, VDS, Temperature, W, L]], dtype=np.float32)
    x_sc = _nmos_scaler_X.transform(x).astype(np.float32)
    with torch.no_grad():
        y_norm = _nmos_model(torch.tensor(x_sc, device=DEVICE)).cpu().numpy()
    y_log = _nmos_scaler_y.inverse_transform(y_norm)
    ids = float(10 ** y_log[0, 0])
    return max(ids, 0.0)


def predict_pmos_current(VSG, VSD, Temperature=27.0,
                         W=1.0e-6, L=180e-9) -> float:
    """
    Predict |PMOS drain current| IDS (returned as positive value).

    Parameters
    ----------
    VSG         : Source-Gate voltage (= -VGS), V  ≥ 0
    VSD         : Source-Drain voltage (= -VDS), V  ≥ 0
    Temperature : Junction temperature (°C)
    W, L        : Width / Length (m)

    Returns
    -------
    |IDS| (A) — always ≥ 0
    """
    _load_pmos()
    x = np.array([[VSG, VSD, Temperature, W, L]], dtype=np.float32)
    x_sc = _pmos_scaler_X.transform(x).astype(np.float32)
    with torch.no_grad():
        y_norm = _pmos_model(torch.tensor(x_sc, device=DEVICE)).cpu().numpy()
    y_log = _pmos_scaler_y.inverse_transform(y_norm)
    ids = float(10 ** y_log[0, 0])
    return max(ids, 0.0)


# ── batch helpers (used by VTC solver) ────────────────────────────────────

def predict_nmos_batch(X: np.ndarray) -> np.ndarray:
    """X shape (N, 5): [VGS, VDS, Temp, W, L] → IDS (N,)"""
    _load_nmos()
    x_sc = _nmos_scaler_X.transform(X.astype(np.float32))
    with torch.no_grad():
        y_norm = _nmos_model(torch.tensor(x_sc, device=DEVICE)).cpu().numpy()
    y_log = _nmos_scaler_y.inverse_transform(y_norm)
    return np.maximum(10 ** y_log.ravel(), 0.0)


def predict_pmos_batch(X: np.ndarray) -> np.ndarray:
    """X shape (N, 5): [VSG, VSD, Temp, W, L] → |IDS| (N,)"""
    _load_pmos()
    x_sc = _pmos_scaler_X.transform(X.astype(np.float32))
    with torch.no_grad():
        y_norm = _pmos_model(torch.tensor(x_sc, device=DEVICE)).cpu().numpy()
    y_log = _pmos_scaler_y.inverse_transform(y_norm)
    return np.maximum(10 ** y_log.ravel(), 0.0)


# ── quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ids_n = predict_nmos_current(VGS=1.2, VDS=1.8)
    ids_p = predict_pmos_current(VSG=1.8, VSD=1.8)
    print(f"NMOS IDS = {ids_n:.4e} A")
    print(f"PMOS IDS = {ids_p:.4e} A")