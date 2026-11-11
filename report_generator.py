"""
Module 17 — PDF Report Generator  (report_generator.py)
=========================================================
Auto-generates a professional PDF research report using fpdf2.
Contains:
  • Cover page
  • Model accuracy metrics (NMOS / PMOS)
  • All generated plots embedded as images
  • Performance metrics table
  • Process corner summary
  • SRAM SNM summary

Install fpdf2 if missing:  pip install fpdf2
"""

import os
import glob
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    # Keep this module importable so the pipeline can gracefully skip report
    # generation when the optional fpdf2 package is unavailable.
    FPDF = object
    print("[Report] fpdf2 not installed. Run: pip install fpdf2")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORT_DIR, "CMOS_AI_EDA_Report.pdf")


class CMOSReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "AI-Assisted Electrical Characterization of CMOS Digital Circuits", 0, 0, "C")
        self.ln(4)
        self.set_draw_color(100, 100, 200)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, "C")

    def cover_page(self):
        self.add_page()
        self.ln(30)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(30, 60, 120)
        self.multi_cell(0, 12, "AI-Assisted Electrical Characterization\nof CMOS Digital Circuits", 0, "C")
        self.ln(6)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, "Using Machine Learning (PyTorch + Scikit-Learn)", 0, 0, "C")
        self.ln(12)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "180 nm CMOS Technology  |  VDD = 1.8 V", 0, 0, "C")
        self.ln(8)
        self.cell(0, 8, datetime.now().strftime("Report Date: %B %d, %Y"), 0, 0, "C")
        self.ln(20)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        summary = (
            "This report documents the complete AI-based EDA characterization pipeline "
            "for CMOS digital circuits. Neural network models trained on SPICE simulation "
            "data are used to predict transistor currents, from which VTC curves, timing, "
            "power, Monte Carlo variation, and SRAM metrics are derived."
        )
        self.multi_cell(0, 6, summary, 0, "C")

    def section_title(self, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, title, 0, 1)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def metrics_table(self, nmos_metrics, pmos_metrics):
        self.section_title("1. Model Accuracy Metrics")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0)

        headers = ["Transistor", "R²", "MAE (A)", "RMSE (A)"]
        widths  = [40, 35, 55, 55]

        # header row
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        for h, w in zip(headers, widths):
            self.cell(w, 8, h, 1, 0, "C", fill=True)
        self.ln()

        # data rows
        self.set_text_color(0)
        rows = [
            ("NMOS", nmos_metrics.get("r2", 0), nmos_metrics.get("mae", 0), nmos_metrics.get("rmse", 0)),
            ("PMOS", pmos_metrics.get("r2", 0), pmos_metrics.get("mae", 0), pmos_metrics.get("rmse", 0)),
        ]
        fills = [False, True]
        self.set_fill_color(230, 235, 245)
        self.set_font("Helvetica", "", 10)
        for row, fill in zip(rows, fills):
            self.cell(widths[0], 7, str(row[0]), 1, 0, "C", fill)
            self.cell(widths[1], 7, f"{row[1]:.4f}", 1, 0, "C", fill)
            self.cell(widths[2], 7, f"{row[2]:.4e}", 1, 0, "C", fill)
            self.cell(widths[3], 7, f"{row[3]:.4e}", 1, 0, "C", fill)
            self.ln()
        self.ln(4)

    def performance_table(self, perf_table, gate_gains=None):
        self.section_title("2. Logic Gate Performance Summary")
        self.set_font("Helvetica", "", 10)

        headers = ["Circuit", "Gain", "Delay (ps)", "Power (µW)", "Energy (fJ)"]
        widths  = [30, 25, 35, 35, 35]

        self.set_fill_color(30, 60, 120)
        self.set_text_color(255)
        self.set_font("Helvetica", "B", 10)
        for h, w in zip(headers, widths):
            self.cell(w, 8, h, 1, 0, "C", fill=True)
        self.ln()

        self.set_fill_color(230, 235, 245)
        self.set_text_color(0)
        self.set_font("Helvetica", "", 10)
        for i, row in enumerate(perf_table):
            fill = (i % 2 == 1)
            gain_str = f"{gate_gains.get(row['Circuit'], 0):.1f}" if gate_gains else row.get("Gain", "—")
            self.cell(widths[0], 7, row["Circuit"],          1, 0, "C", fill)
            self.cell(widths[1], 7, str(gain_str),           1, 0, "C", fill)
            self.cell(widths[2], 7, str(row["Delay_ps"]),    1, 0, "C", fill)
            self.cell(widths[3], 7, str(row["Power_uW"]),    1, 0, "C", fill)
            self.cell(widths[4], 7, str(row["Energy_fJ"]),   1, 0, "C", fill)
            self.ln()
        self.ln(4)

    def add_plot(self, img_path, title="", w=170):
        if not os.path.exists(img_path):
            return
        self.ln(2)
        if title:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(50, 50, 50)
            self.cell(0, 6, title, 0, 1)
        x = (210 - w) / 2
        self.image(img_path, x=x, w=w)
        self.ln(4)

    def add_plots_section(self, section_title, img_paths):
        self.add_page()
        self.section_title(section_title)
        for path, caption in img_paths:
            self.add_plot(path, caption)
            if self.get_y() > 240:
                self.add_page()


def generate_report(nmos_metrics=None, pmos_metrics=None,
                    perf_table=None, gate_gains=None):
    if not FPDF_AVAILABLE:
        print("[Report] Cannot generate PDF: fpdf2 not installed.")
        return

    if nmos_metrics is None: nmos_metrics = {"r2": 0, "mae": 0, "rmse": 0}
    if pmos_metrics is None: pmos_metrics = {"r2": 0, "mae": 0, "rmse": 0}
    if perf_table   is None: perf_table   = []
    if gate_gains   is None: gate_gains   = {}

    pdf = CMOSReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 15, 10)

    # cover
    pdf.cover_page()

    # metrics table
    pdf.add_page()
    pdf.metrics_table(nmos_metrics, pmos_metrics)

    # performance table
    if perf_table:
        pdf.performance_table(perf_table, gate_gains)

    # ── IV characteristic plots ─────────────────────────────────────────
    iv_plots = [
        (os.path.join(PLOT_DIR, "nmos_ids_vgs.png"),      "NMOS IDS vs VGS"),
        (os.path.join(PLOT_DIR, "nmos_ids_vds.png"),      "NMOS IDS vs VDS"),
        (os.path.join(PLOT_DIR, "nmos_log_ids_vgs.png"),  "NMOS log(IDS) vs VGS"),
        (os.path.join(PLOT_DIR, "pmos_ids_vsg.png"),      "PMOS |IDS| vs VSG"),
        (os.path.join(PLOT_DIR, "pmos_ids_vsd.png"),      "PMOS |IDS| vs VSD"),
    ]
    pdf.add_plots_section("3. Transistor IV Characteristics", iv_plots)

    # ── model validation plots ──────────────────────────────────────────
    model_plots = [
        (os.path.join(PLOT_DIR, "nmos_loss_curve.png"),      "NMOS Training Loss"),
        (os.path.join(PLOT_DIR, "nmos_actual_vs_pred.png"),  "NMOS Actual vs Predicted"),
        (os.path.join(PLOT_DIR, "pmos_loss_curve.png"),      "PMOS Training Loss"),
        (os.path.join(PLOT_DIR, "pmos_actual_vs_pred.png"),  "PMOS Actual vs Predicted"),
    ]
    pdf.add_plots_section("4. Neural Network Training Results", model_plots)

    # ── VTC and gate plots ──────────────────────────────────────────────
    vtc_plots = [
        (os.path.join(PLOT_DIR, "vtc_inv.png"),       "Inverter VTC"),
        (os.path.join(PLOT_DIR, "vtc_all_gates.png"), "All Gate VTCs"),
        (os.path.join(PLOT_DIR, "gain_comparison.png"),"Gate Gain Comparison"),
    ]
    pdf.add_plots_section("5. Inverter and Logic Gate VTCs", vtc_plots)

    # ── timing and power ────────────────────────────────────────────────
    tp_plots = [
        (os.path.join(PLOT_DIR, "delay_vs_CL.png"),    "Delay vs CL"),
        (os.path.join(PLOT_DIR, "power_vs_freq.png"),  "Power vs Frequency"),
        (os.path.join(PLOT_DIR, "edp_comparison.png"), "EDP Comparison"),
    ]
    pdf.add_plots_section("6. Delay and Power Analysis", tp_plots)

    # ── corner and temperature ──────────────────────────────────────────
    corner_plots = [
        (os.path.join(PLOT_DIR, "vtc_temperature.png"), "VTC vs Temperature"),
        (os.path.join(PLOT_DIR, "vtc_corners.png"),     "VTC: Process Corners"),
        (os.path.join(PLOT_DIR, "corner_bar.png"),      "Corner Metrics Bar Chart"),
    ]
    pdf.add_plots_section("7. Temperature and Corner Analysis", corner_plots)

    # ── Monte Carlo ─────────────────────────────────────────────────────
    mc_plots = [
        (os.path.join(PLOT_DIR, "mc_vtc_spread.png"),  "Monte Carlo VTC Spread"),
        (os.path.join(PLOT_DIR, "mc_vm_histogram.png"),"VM Histogram"),
    ]
    pdf.add_plots_section("8. Monte Carlo Analysis", mc_plots)

    # ── ring oscillator / DFF / SRAM ────────────────────────────────────
    misc_plots = [
        (os.path.join(PLOT_DIR, "ring_oscillator.png"), "Ring Oscillator"),
        (os.path.join(PLOT_DIR, "dff_timing.png"),      "DFF Timing"),
        (os.path.join(PLOT_DIR, "sram_butterfly.png"),  "SRAM Butterfly Curve"),
        (os.path.join(PLOT_DIR, "sram_snm_vs_temp.png"),"SRAM SNM vs Temperature"),
    ]
    pdf.add_plots_section("9. Ring Oscillator, DFF, and SRAM", misc_plots)

    # save
    pdf.output(REPORT_PATH)
    print(f"\n[Report] PDF saved: {REPORT_PATH}")
    return REPORT_PATH
    print(f"\n[Report] PDF saved → {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    generate_report()
