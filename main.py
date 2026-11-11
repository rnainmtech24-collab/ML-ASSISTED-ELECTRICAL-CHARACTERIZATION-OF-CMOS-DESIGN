"""
Module 18 — Main Pipeline  (main.py)
=====================================
Executes the complete AI-assisted CMOS characterization pipeline end-to-end.

Run order:
    python main.py           ← full pipeline
    python main.py --skip-train  ← skip NN training (use saved models)
    python main.py --skip-mc     ← skip Monte Carlo (slow)

After completion:
    streamlit run streamlit_app.py
"""

import os
import sys
import argparse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── CLI flags ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CMOS AI EDA — Full Pipeline")
parser.add_argument("--skip-train", action="store_true",
                    help="Skip neural network training (use saved models)")
parser.add_argument("--skip-mc",    action="store_true",
                    help="Skip Monte Carlo (speeds up the run significantly)")
parser.add_argument("--mc-runs",    type=int, default=1000,
                    help="Number of Monte Carlo runs (default: 1000)")
args = parser.parse_args()


def banner(title):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def main():
    t0 = time.time()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    banner("Step 1 — Data Preprocessing")
    from preprocess import load_nmos, load_pmos
    _, _, _, _, nmos_sX, nmos_sy, nmos_df = load_nmos()
    _, _, _, _, pmos_sX, pmos_sy, pmos_df = load_pmos()
    print(f"  NMOS dataset: {len(nmos_df)} clean samples")
    print(f"  PMOS dataset: {len(pmos_df)} clean samples")

    # ── Step 2: NMOS Training ─────────────────────────────────────────────
    nmos_metrics = {"r2": 0, "mae": 0, "rmse": 0}
    pmos_metrics = {"r2": 0, "mae": 0, "rmse": 0}

    if not args.skip_train:
        banner("Step 2 — NMOS Neural Network Training")
        from train_nmos import train_nmos
        _, _, _, nmos_metrics = train_nmos()

        banner("Step 3 — PMOS Neural Network Training")
        from train_pmos import train_pmos
        _, _, _, pmos_metrics = train_pmos()
    else:
        print("[Skipping training — using saved models]")
        # try to load metrics from a simple cache if available
        import json
        metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            nmos_metrics = m.get("nmos", nmos_metrics)
            pmos_metrics = m.get("pmos", pmos_metrics)

    # save metrics for report
    import json
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"nmos": nmos_metrics, "pmos": pmos_metrics}, f, indent=2)

    # ── Step 4: IV Characteristic Plots ───────────────────────────────────
    banner("Step 4 — IV Characteristic Visualization")
    from visualization import generate_all_iv_plots
    generate_all_iv_plots()

    # ── Step 5: CMOS Inverter ─────────────────────────────────────────────
    banner("Step 5 — CMOS Inverter Characterization")
    from inverter import characterize_inverter, plot_gain_curve
    vin_inv, vout_inv, inv_params = characterize_inverter()
    plot_gain_curve(vin_inv, vout_inv)

    # ── Step 6: Logic Gates ───────────────────────────────────────────────
    banner("Step 6 — Logic Gate Characterization")
    from gates import characterize_all_gates, print_gate_table
    gate_results = characterize_all_gates()
    print_gate_table(gate_results)
    gate_gains = {name: data["params"]["Gain"] for name, data in gate_results.items()}

    # ── Step 7: Delay and Power ────────────────────────────────────────────
    banner("Step 7 — Delay and Power Analysis")
    from delay_power import (sweep_delay_vs_CL, sweep_power_vs_freq,
                              plot_edp_comparison, compute_performance_table)
    sweep_delay_vs_CL()
    sweep_power_vs_freq()
    plot_edp_comparison()
    perf_table = compute_performance_table()

    # fill in gate gains
    for row in perf_table:
        row["Gain"] = f"{gate_gains.get(row['Circuit'], 0):.1f}"

    print("\nPerformance Table:")
    print(f"{'Circuit':<8} {'Gain':>6} {'Delay(ps)':>10} {'Power(µW)':>10} {'Energy(fJ)':>12}")
    print("─" * 50)
    for row in perf_table:
        print(f"{row['Circuit']:<8} {row['Gain']:>6} {row['Delay_ps']:>10} "
              f"{row['Power_uW']:>10} {row['Energy_fJ']:>12}")

    # ── Step 8: Temperature Analysis ──────────────────────────────────────
    banner("Step 8 — Temperature Analysis")
    from corner_analysis import temperature_analysis
    temperature_analysis()

    # ── Step 9: Process Corner Analysis ───────────────────────────────────
    banner("Step 9 — Process Corner Analysis")
    from corner_analysis import corner_analysis, plot_corner_bar
    corner_results = corner_analysis()
    plot_corner_bar(corner_results)

    # ── Step 10: Monte Carlo ───────────────────────────────────────────────
    if not args.skip_mc:
        banner(f"Step 10 — Monte Carlo Analysis ({args.mc_runs} runs)")
        from monte_carlo import run_monte_carlo
        run_monte_carlo(n_runs=args.mc_runs)
    else:
        print("[Skipping Monte Carlo]")

    # ── Step 11: Ring Oscillator ───────────────────────────────────────────
    banner("Step 11 — Ring Oscillator")
    from ring_oscillator import sweep_ring_oscillator, fo4_delay
    sweep_ring_oscillator()
    fo4_delay()

    # ── Step 12: DFF ──────────────────────────────────────────────────────
    banner("Step 12 — D Flip-Flop Timing")
    from dff import dff_timing, plot_dff_timing, dff_fmax_vs_CL
    dff_res = dff_timing()
    plot_dff_timing(dff_res)
    dff_fmax_vs_CL()

    # ── Step 13: SRAM ─────────────────────────────────────────────────────
    banner("Step 13 — SRAM Characterization")
    from sram import butterfly_curve, snm_vs_temperature, characterize_sram_timing
    butterfly_curve()
    characterize_sram_timing()
    snm_vs_temperature()

    # ── Step 14: PDF Report ───────────────────────────────────────────────
    banner("Step 14 — PDF Report Generation")
    from report_generator import generate_report
    generate_report(
        nmos_metrics=nmos_metrics,
        pmos_metrics=pmos_metrics,
        perf_table=perf_table,
        gate_gains=gate_gains,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"✅  Pipeline Complete  ({elapsed/60:.1f} min)")
    print(f"\n  Plots  → {os.path.join(BASE_DIR, 'plots')}/")
    print(f"  Report → {os.path.join(BASE_DIR, 'reports', 'CMOS_AI_EDA_Report.pdf')}")
    print(f"\n  Launch GUI:  streamlit run {os.path.join(BASE_DIR, 'streamlit_app.py')}\n")


if __name__ == "__main__":
    main()
