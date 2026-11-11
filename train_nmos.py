"""
Module 2 — NMOS Neural Network Training  (train_nmos.py)
=========================================================
Trains the TransistorNet on the preprocessed NMOS dataset.

Hyper-parameters (as specified):
    Optimizer  : Adam
    LR         : 0.0005
    Loss       : MSE (in normalised log-space)
    Epochs     : 5000
    Batch mode : full-batch (dataset fits in memory)

Outputs
-------
models/nmos_model.pth          — saved model weights
plots/nmos_loss_curve.png      — training loss per epoch
plots/nmos_actual_vs_pred.png  — scatter: true vs predicted IDS
Console metrics: R², MAE, RMSE (back-transformed to linear IDS)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import load_nmos
from models import build_nmos_model

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models");  os.makedirs(MODEL_DIR, exist_ok=True)
PLOT_DIR   = os.path.join(BASE_DIR, "plots");   os.makedirs(PLOT_DIR,  exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "nmos_model.pth")

# ── training config ────────────────────────────────────────────────────────
LR     = 0.0005
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nmos():
    # ── data ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, _ = load_nmos()

    X_tr = torch.tensor(X_train, device=DEVICE)
    y_tr = torch.tensor(y_train, device=DEVICE)
    X_te = torch.tensor(X_test,  device=DEVICE)
    y_te = torch.tensor(y_test,  device=DEVICE)

    # ── model / optimiser / loss ───────────────────────────────────────────
    model     = build_nmos_model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────
    losses = []
    print(f"[NMOS] Training on {DEVICE} for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:5d} | Loss: {loss.item():.6f}")

    # ── save weights ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[NMOS] Model saved → {MODEL_PATH}")

    # ── evaluate on test set ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_te).cpu().numpy()

    # inverse-transform: normalised log-space → log10(IDS) → IDS
    y_true_log = scaler_y.inverse_transform(y_te.cpu().numpy())
    y_pred_log = scaler_y.inverse_transform(y_pred_norm)

    y_true_lin = 10 ** y_true_log.ravel()
    y_pred_lin = 10 ** y_pred_log.ravel()

    r2   = r2_score(y_true_lin, y_pred_lin)
    mae  = mean_absolute_error(y_true_lin, y_pred_lin)
    rmse = np.sqrt(mean_squared_error(y_true_lin, y_pred_lin))

    print(f"\n[NMOS Test Metrics]  R²={r2:.4f}  MAE={mae:.4e} A  RMSE={rmse:.4e} A")

    # ── loss curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.semilogy(losses, color="#1f77b4", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("NMOS Training Loss Curve")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nmos_loss_curve.png"), dpi=300)
    plt.close()

    # ── actual vs predicted scatter ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(y_true_lin, y_pred_lin, s=4, alpha=0.4, color="#1f77b4")
    lim = [min(y_true_lin.min(), y_pred_lin.min()),
           max(y_true_lin.max(), y_pred_lin.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual IDS (A)")
    ax.set_ylabel("Predicted IDS (A)")
    ax.set_title(f"NMOS Actual vs Predicted  (R²={r2:.4f})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nmos_actual_vs_pred.png"), dpi=300)
    plt.close()

    print("[NMOS] Plots saved.")
    return model, scaler_X, scaler_y, {"r2": r2, "mae": mae, "rmse": rmse}


if __name__ == "__main__":
    train_nmos()