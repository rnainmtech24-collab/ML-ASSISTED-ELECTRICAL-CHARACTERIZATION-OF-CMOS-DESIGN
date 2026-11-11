"""
Module 15 — Visualization  (visualization.py)
=============================================
Generates all publication-quality figures (dpi=300) from the raw datasets
and model predictions. Called after training is complete.

Figures generated:
  1.  nmos_ids_vgs.png      — IDS vs VGS (linear) for multiple VDS
  2.  nmos_ids_vds.png      — IDS vs VDS (linear) for multiple VGS
  3.  nmos_log_ids_vgs.png  — log10(IDS) vs VGS
  4.  pmos_ids_vsg.png      — |IDS| vs VSG (linear)
  5.  pmos_ids_vsd.png      — |IDS| vs VSD (linear)
  6.  pmos_log_ids_vsg.png  — log10|IDS| vs VSG
  (Actual vs Predicted and Loss curves saved by train_nmos/pmos.py)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from current_predictor import predict_nmos_current, predict_pmos_current
from cmos_solver import VDD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.alpha":     0.3,
    "figure.dpi":     300,
})

W  = 0.5e-6;  L = 180e-9;  T = 27.0
VDD_HALF = VDD / 2


# ── NMOS plots ─────────────────────────────────────────────────────────────
def plot_nmos_ids_vgs(save=True):
    """IDS vs VGS for three VDS values."""
    vgs_arr  = np.linspace(0, VDD, 181)
    vds_vals = [0.05, VDD_HALF, VDD]
    colors   = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    for vds, clr in zip(vds_vals, colors):
        ids = [predict_nmos_current(v, vds, T, W, L) for v in vgs_arr]
        axes[0].plot(vgs_arr, np.array(ids) * 1e6, color=clr,
                     label=f"VDS={vds:.2f}V")
        log_ids = [np.log10(max(i, 1e-15)) for i in ids]
        axes[1].plot(vgs_arr, log_ids, color=clr, label=f"VDS={vds:.2f}V")

    for ax, ylabel, title in zip(
        axes,
        ["IDS (µA)", "log₁₀(IDS)"],
        ["NMOS IDS vs VGS (Linear)", "NMOS log(IDS) vs VGS"],
    ):
        ax.set_xlabel("VGS (V)");  ax.set_ylabel(ylabel)
        ax.set_title(title);  ax.legend(fontsize=9);  ax.grid(True)

    plt.tight_layout()
    if save:
        for ax, name in zip(axes, ["nmos_ids_vgs.png", "nmos_log_ids_vgs.png"]):
            extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(PLOT_DIR, name), dpi=300, bbox_inches=extent)
        plt.close()
        print("[Plot] NMOS IDS-VGS plots saved.")
    return fig


def plot_nmos_ids_vds(save=True):
    """IDS vs VDS for multiple VGS (output characteristics)."""
    vds_arr  = np.linspace(0, VDD, 181)
    vgs_vals = [0.5, 0.9, 1.2, 1.5, 1.8]
    cmap = plt.colormaps["viridis"].resampled(len(vgs_vals))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for i, vgs in enumerate(vgs_vals):
        ids = [predict_nmos_current(vgs, vd, T, W, L) for vd in vds_arr]
        ax.plot(vds_arr, np.array(ids)*1e6, color=cmap(i), label=f"VGS={vgs:.1f}V")

    ax.set_xlabel("VDS (V)");  ax.set_ylabel("IDS (µA)")
    ax.set_title("NMOS Output Characteristics (IDS vs VDS)")
    ax.legend(fontsize=9, loc="upper left");  ax.grid(True)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "nmos_ids_vds.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] NMOS IDS-VDS → {path}")
    return fig


# ── PMOS plots ─────────────────────────────────────────────────────────────
def plot_pmos_ids_vsg(save=True):
    """|IDS| vs VSG for PMOS."""
    vsg_arr  = np.linspace(0, VDD, 181)
    vsd_vals = [0.05, VDD_HALF, VDD]
    Wp       = 1.0e-6
    colors   = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for vsd, clr in zip(vsd_vals, colors):
        ids = [predict_pmos_current(v, vsd, T, Wp, L) for v in vsg_arr]
        axes[0].plot(vsg_arr, np.array(ids)*1e6, color=clr, label=f"VSD={vsd:.2f}V")
        log_ids = [np.log10(max(i, 1e-15)) for i in ids]
        axes[1].plot(vsg_arr, log_ids, color=clr, label=f"VSD={vsd:.2f}V")

    for ax, ylabel, title in zip(
        axes,
        ["|IDS| (µA)", "log₁₀|IDS|"],
        ["PMOS |IDS| vs VSG (Linear)", "PMOS log|IDS| vs VSG"],
    ):
        ax.set_xlabel("VSG (V)");  ax.set_ylabel(ylabel)
        ax.set_title(title);  ax.legend(fontsize=9);  ax.grid(True)

    plt.tight_layout()
    if save:
        for ax, name in zip(axes, ["pmos_ids_vsg.png", "pmos_log_ids_vsg.png"]):
            extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(PLOT_DIR, name), dpi=300, bbox_inches=extent)
        plt.close()
        print("[Plot] PMOS IDS-VSG plots saved.")
    return fig


def plot_pmos_ids_vsd(save=True):
    """PMOS output characteristics."""
    vsd_arr  = np.linspace(0, VDD, 181)
    vsg_vals = [0.5, 0.9, 1.2, 1.5, 1.8]
    Wp       = 1.0e-6
    cmap = plt.colormaps["viridis"].resampled(len(vsg_vals))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for i, vsg in enumerate(vsg_vals):
        ids = [predict_pmos_current(vsg, vd, T, Wp, L) for vd in vsd_arr]
        ax.plot(vsd_arr, np.array(ids)*1e6, color=cmap(i), label=f"VSG={vsg:.1f}V")

    ax.set_xlabel("VSD (V)");  ax.set_ylabel("|IDS| (µA)")
    ax.set_title("PMOS Output Characteristics (|IDS| vs VSD)")
    ax.legend(fontsize=9, loc="upper left");  ax.grid(True)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "pmos_ids_vsd.png")
        plt.savefig(path, dpi=300);  plt.close()
        print(f"[Plot] PMOS IDS-VSD → {path}")
    return fig


def generate_all_iv_plots():
    """Generate all IV characteristic plots."""
    print("[Visualization] Generating IV plots …")
    plot_nmos_ids_vgs()
    plot_nmos_ids_vds()
    plot_pmos_ids_vsg()
    plot_pmos_ids_vsd()
    print("[Visualization] All IV plots done.")


if __name__ == "__main__":
    generate_all_iv_plots()