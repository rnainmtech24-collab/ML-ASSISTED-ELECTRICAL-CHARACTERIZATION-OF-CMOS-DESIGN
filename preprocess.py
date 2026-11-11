"""
Module 1: Data Preprocessing
=============================
Loads NMOS and PMOS CSV datasets, cleans them, applies log10 transform
to IDS, normalizes features with StandardScaler, splits train/test,
and saves scalers for reproducible inference.

NMOS CSV columns : VGS, VDS, VG, Wval, Lval, IDS
  - Wval / Lval use suffix notation: "0.5µ", "180n" → converted to metres
  - No Temperature column → default 27 °C injected

PMOS CSV columns : VGS, VDS, Wval, Lval, Temp, IDS
  - VGS / VDS are already negative (PMOS convention)
  - We convert to VSG = -VGS, VSD = -VDS for physical intuition
  - Wval / Lval already in metres (scientific notation strings)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCALER_DIR  = os.path.join(BASE_DIR, "scalers")
os.makedirs(SCALER_DIR, exist_ok=True)

NMOS_CSV = os.path.join(BASE_DIR, "nmos.csv")
PMOS_CSV = os.path.join(BASE_DIR, "pmos.csv")
VDD = 1.8

# ── helpers ────────────────────────────────────────────────────────────────
_SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
           "m": 1e-3,  "k": 1e3,   "M": 1e6,  "G": 1e9}

def _parse_suffix(val):
    """Convert strings like '0.5µ', '180n', '1.8' to float (metres / SI)."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    # try plain float first
    try:
        return float(s)
    except ValueError:
        pass
    # strip suffix
    for sfx, mul in _SUFFIX.items():
        if s.endswith(sfx):
            try:
                return float(s[:-len(sfx)]) * mul
            except ValueError:
                pass
    return np.nan


# ── NMOS preprocessing ─────────────────────────────────────────────────────
def load_nmos(path=NMOS_CSV, test_size=0.2, random_state=42):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (float32)
    scaler_X, scaler_y               : fitted StandardScaler objects
    df_clean                         : cleaned DataFrame (for EDA)
    """
    print("[NMOS] Loading dataset …")
    df = pd.read_csv(path)

    # ── column normalisation ───────────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # parse W / L from suffix strings
    df["W"] = df["Wval"].apply(_parse_suffix)
    df["L"] = df["Lval"].apply(_parse_suffix)

    # VGS, VDS are already numeric
    df["VGS"] = pd.to_numeric(df["VGS"], errors="coerce")
    df["VDS"] = pd.to_numeric(df["VDS"], errors="coerce")
    df["IDS"] = pd.to_numeric(df["IDS"], errors="coerce")

    # No Temperature column → inject 27 °C (room temperature)
    if "Temperature" not in df.columns and "Temp" not in df.columns:
        df["Temperature"] = 27.0
    else:
        tcol = "Temperature" if "Temperature" in df.columns else "Temp"
        df["Temperature"] = pd.to_numeric(df[tcol], errors="coerce")

    # ── feature set ───────────────────────────────────────────────────────
    features = ["VGS", "VDS", "Temperature", "W", "L"]
    df = df[features + ["IDS"]].copy()

    # ── cleaning ──────────────────────────────────────────────────────────
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # Remove rows where IDS ≤ 0 (non-physical for NMOS forward bias)
    df = df[df["IDS"] > 0].copy()
    print(f"[NMOS] Rows: {before} → {len(df)} after cleaning")

    # ── log10 transform on IDS ────────────────────────────────────────────
    df["log_IDS"] = np.log10(df["IDS"])

    # ── normalise ─────────────────────────────────────────────────────────
    X = df[features].values.astype(np.float32)
    y = df["log_IDS"].values.reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_X.transform(X_train).astype(np.float32)
    X_test  = scaler_X.transform(X_test).astype(np.float32)
    y_train = scaler_y.transform(y_train).astype(np.float32)
    y_test  = scaler_y.transform(y_test).astype(np.float32)

    # ── save scalers ──────────────────────────────────────────────────────
    joblib.dump(scaler_X, os.path.join(SCALER_DIR, "nmos_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(SCALER_DIR, "nmos_scaler_y.pkl"))
    print("[NMOS] Scalers saved.")

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, df


# ── PMOS preprocessing ─────────────────────────────────────────────────────
def load_pmos(path=PMOS_CSV, test_size=0.2, random_state=42):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (float32)
    scaler_X, scaler_y               : fitted StandardScaler objects
    df_clean                         : cleaned DataFrame (for EDA)
    """
    print("[PMOS] Loading dataset …")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # ── parse columns ─────────────────────────────────────────────────────
    df["VGS"]  = pd.to_numeric(df["VGS"], errors="coerce")
    df["VDS"]  = pd.to_numeric(df["VDS"], errors="coerce")
    df["W"]    = df["Wval"].apply(_parse_suffix)
    df["L"]    = df["Lval"].apply(_parse_suffix)
    df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
    df["IDS"]  = pd.to_numeric(df["IDS"], errors="coerce")

    # PMOS: VGS ≤ 0, VDS ≤ 0 → convert to positive VSG, VSD
    # The CSV sweeps absolute gate/drain node voltages with the PMOS source
    # at VDD.  Train on physical terminal voltages, not their negation.
    df["VSG"] = VDD - df["VGS"]
    df["VSD"] = VDD - df["VDS"]

    # IDS is negative for PMOS → take absolute value before log
    df["IDS_abs"] = df["IDS"].abs()

    # ── feature set ───────────────────────────────────────────────────────
    features = ["VSG", "VSD", "Temp", "W", "L"]
    df = df[features + ["IDS", "IDS_abs"]].copy()

    # ── cleaning ──────────────────────────────────────────────────────────
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df["IDS_abs"] > 0].copy()
    print(f"[PMOS] Rows: {before} → {len(df)} after cleaning")

    # ── log10 transform ───────────────────────────────────────────────────
    df["log_IDS"] = np.log10(df["IDS_abs"])

    # ── normalise ─────────────────────────────────────────────────────────
    X = df[features].values.astype(np.float32)
    y = df["log_IDS"].values.reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_X.transform(X_train).astype(np.float32)
    X_test  = scaler_X.transform(X_test).astype(np.float32)
    y_train = scaler_y.transform(y_train).astype(np.float32)
    y_test  = scaler_y.transform(y_test).astype(np.float32)

    # ── save scalers ──────────────────────────────────────────────────────
    joblib.dump(scaler_X, os.path.join(SCALER_DIR, "pmos_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(SCALER_DIR, "pmos_scaler_y.pkl"))
    print("[PMOS] Scalers saved.")

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, df


# ── quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    Xtr, Xte, ytr, yte, sX, sy, df = load_nmos()
    print("NMOS train shape:", Xtr.shape, ytr.shape)
    Xtr, Xte, ytr, yte, sX, sy, df = load_pmos()
    print("PMOS train shape:", Xtr.shape, ytr.shape)
