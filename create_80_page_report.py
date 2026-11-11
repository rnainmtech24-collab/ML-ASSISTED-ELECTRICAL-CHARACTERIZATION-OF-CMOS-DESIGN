"""Generate the expanded, single-spaced project report from the thesis and code."""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from fpdf import FPDF


ROOT = Path(__file__).resolve().parent
SOURCE_DOCX = ROOT / "final" / "Ramandeep_6024620033.docx"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "CMOS_ML_180nm_80_Page_Project_Report.pdf"
PLOT_DIR = ROOT / "plots"
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FONT_REGULAR = FONT_DIR / "times.ttf"
FONT_BOLD = FONT_DIR / "timesbd.ttf"
FONT_ITALIC = FONT_DIR / "timesi.ttf"
FONT_FAMILY = "TimesNewRoman"
PAGE_TARGET = 80
BODY_PT = 12
BODY_LINE_MM = BODY_PT * 25.4 / 72

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W_VAL = f"{{{NS['w']}}}val"


def paragraph_text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.findall(".//w:t", NS)).strip()


def paragraph_style(element: ET.Element) -> str:
    node = element.find("w:pPr/w:pStyle", NS)
    return node.get(W_VAL, "") if node is not None else ""


def extract_document() -> tuple[list[tuple[str, str]], list[str], str]:
    with ZipFile(SOURCE_DOCX) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError(f"No document body found in {SOURCE_DOCX}")

    blocks: list[tuple[str, str]] = []
    references: list[str] = []
    abstract: list[str] = []
    active_section = ""
    reference_section = False

    for element in body:
        if element.tag.endswith("}p"):
            text = paragraph_text(element)
            style = paragraph_style(element)
            if not text:
                continue
            if style == "Heading1" and text.upper() == "ABSTRACT":
                active_section = "abstract"
                continue
            if style == "Heading1" and text.upper().startswith("CHAPTER 1:"):
                active_section = "chapters"
            if style == "Heading1" and text.upper() == "REFERENCES":
                reference_section = True
                active_section = "references"
                continue
            if style == "Heading1" and active_section == "abstract":
                active_section = ""

            if reference_section:
                if re.match(r"^\[\d+\]", text):
                    references.append(text)
            elif active_section == "abstract":
                abstract.append(text)
            elif active_section == "chapters":
                blocks.append((style, text))
        elif element.tag.endswith("}tbl") and active_section == "chapters":
            rows = []
            for row in element.findall(".//w:tr", NS):
                cells = [
                    paragraph_text(cell)
                    for cell in row.findall("./w:tc", NS)
                ]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(("Table", "\n".join(rows)))

    if not blocks:
        raise ValueError("The source thesis did not contain a Chapter 1 body.")
    if not abstract:
        raise ValueError("The source thesis did not contain an abstract.")
    if not references:
        raise ValueError("The source thesis did not contain numbered references.")
    return blocks, references, "\n\n".join(abstract)


class ProjectReport(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(13)
        self.set_font(FONT_FAMILY, "", 8)
        self.set_text_color(90, 100, 110)
        self.cell(0, 4, "AI-ASSISTED CMOS CHARACTERIZATION | PROJECT REPORT", align="C")
        self.set_draw_color(120, 160, 170)
        self.line(25.4, 19, 184.6, 19)

    def footer(self) -> None:
        self.set_y(-16)
        self.set_font(FONT_FAMILY, "", 8)
        self.set_text_color(90, 100, 110)
        self.cell(0, 5, f"Page {self.page_no()} of {PAGE_TARGET}", align="C")

    def body_paragraph(self, text: str, *, size: int = BODY_PT, indent: float = 0) -> None:
        self.set_font(FONT_FAMILY, "", size)
        self.set_text_color(30, 40, 48)
        self.set_x(25.4 + indent)
        line_height = size * 25.4 / 72
        self.multi_cell(159.2 - indent, line_height, text)
        self.ln(2)

    def heading(self, text: str, level: int = 1) -> None:
        sizes = {1: 18, 2: 15, 3: 13, 4: 12}
        size = sizes.get(level, 12)
        self.ln(4 if level == 1 else 2)
        self.set_font(FONT_FAMILY, "B", size)
        self.set_text_color(24, 67, 83)
        self.multi_cell(159.2, size * 25.4 / 72, text)
        self.set_draw_color(24, 126, 139)
        if level == 1:
            self.line(25.4, self.get_y() + 1, 184.6, self.get_y() + 1)
            self.ln(4)
        else:
            self.ln(2)

    def section_page(self, title: str, subtitle: str = "") -> None:
        self.add_page()
        self.set_y(72)
        self.set_font(FONT_FAMILY, "B", 22)
        self.set_text_color(24, 67, 83)
        self.multi_cell(159.2, 8, title, align="C")
        if subtitle:
            self.ln(6)
            self.set_font(FONT_FAMILY, "", 12)
            self.set_text_color(80, 95, 102)
            self.multi_cell(159.2, BODY_LINE_MM, subtitle, align="C")
        self.ln(14)


def add_cover(pdf: ProjectReport) -> None:
    pdf.add_page()
    pdf.set_fill_color(17, 116, 131)
    pdf.rect(0, 0, 210, 80, "F")
    pdf.set_y(27)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT_FAMILY, "B", 23)
    pdf.multi_cell(0, 10, "MACHINE LEARNING-ASSISTED\nELECTRICAL CHARACTERIZATION\nOF 180 nm CMOS DIGITAL CIRCUITS", align="C")
    pdf.set_y(100)
    pdf.set_text_color(24, 67, 83)
    pdf.set_font(FONT_FAMILY, "B", 16)
    pdf.cell(0, 9, "Expanded Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, "", 12)
    pdf.set_text_color(40, 50, 58)
    for line in (
        "Technology: 180 nm CMOS | Nominal supply: 1.8 V",
        "Author: Ramandeep Nain | Roll No.: 6024620033",
        "M.Tech, VLSI Design",
        "Thapar Institute of Engineering and Technology, Patiala",
        "Supervisor: Dr. Sumit Vyas",
        "Expanded technical report | IEEE numeric citations | 1.0 line spacing",
    ):
        pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(235)
    pdf.set_font(FONT_FAMILY, "I", 10)
    pdf.multi_cell(
        159.2,
        4.3,
        "All model outputs are simulator- or model-derived. They must not be interpreted as measured silicon data or sign-off results.",
        align="C",
    )


def add_flow_diagram(pdf: ProjectReport) -> None:
    pdf.section_page("PROJECT FLOW", "End-to-end data, model, circuit analysis, and reporting sequence")
    steps = [
        ("1  Source data", "NMOS/PMOS CSV or versioned SKY130 ngspice sweeps"),
        ("2  Preprocess", "Clean units and invalid rows; log-transform current; fit scalers on training data"),
        ("3  Train / load", "Train separate NMOS and PMOS regressors, or load saved weights and scalers"),
        ("4  Predict", "Map operating conditions and geometry to drain-current estimates"),
        ("5  Solve circuits", "Use current balance and numerical root-finding for inverter/gate transfer curves"),
        ("6  Analyze", "Estimate timing, power, PVT response, Monte Carlo variation, oscillator, DFF, SRAM"),
        ("7  Validate", "Compare held-out predictions, simulator references, assumptions, and limitations"),
        ("8  Deliver", "Plots, CSV tables, interactive Streamlit views, and PDF reports"),
    ]
    x, y, w, h, gap = 38, 42, 134, 20, 7
    for index, (title, detail) in enumerate(steps):
        y = 42 + index * (h + gap)
        pdf.set_fill_color(234, 245, 244)
        pdf.set_draw_color(24, 126, 139)
        pdf.rect(x, y, w, h, "DF")
        pdf.set_xy(x + 4, y + 2.2)
        pdf.set_font(FONT_FAMILY, "B", 11)
        pdf.set_text_color(24, 67, 83)
        pdf.cell(w - 8, 4.5, title)
        pdf.set_xy(x + 4, y + 8)
        pdf.set_font(FONT_FAMILY, "", 9)
        pdf.set_text_color(45, 55, 60)
        pdf.multi_cell(w - 8, 3.2, detail)
        if index < len(steps) - 1:
            cx = x + w / 2
            pdf.set_draw_color(24, 126, 139)
            pdf.line(cx, y + h, cx, y + h + gap - 1)
            pdf.line(cx, y + h + gap - 1, cx - 2, y + h + gap - 3)
            pdf.line(cx, y + h + gap - 1, cx + 2, y + h + gap - 3)
    pdf.set_y(42 + len(steps) * (h + gap) + 1)
    pdf.set_font(FONT_FAMILY, "I", 9)
    pdf.set_text_color(90, 100, 110)
    pdf.multi_cell(159.2, 3.5, "The original 180 nm CSV/model path and the SKY130 reference-data path are maintained separately.")


def add_project_context(pdf: ProjectReport, project_ref: int, sky130_ref: int) -> None:
    pdf.section_page("PROJECT CONTEXT AND REPRODUCIBILITY")
    pdf.heading("Report basis and interpretation", 1)
    pdf.body_paragraph(
        f"This expanded report consolidates the project thesis with an implementation-focused account of the repository [{project_ref}]. The thesis chapters and bibliography are retained as the technical foundation; repository-specific material explains how the code implements the workflow and how to reproduce its outputs. Numerical values are attributed to their source track and operating assumptions. Results from distinct datasets, modeling tracks, and test splits are not combined into a single accuracy ranking."
    )
    pdf.heading("Two related but distinct modeling tracks", 2)
    pdf.body_paragraph(
        "The device-centric track learns log10(|IDS|) for NMOS and PMOS devices from five inputs (bias, temperature, width, and length) and composes the predictions through a current-balance solver. The thesis also describes a direct circuit/gate surrogate track trained on gate-level sweeps. These tracks have different inputs, targets, metrics, and reference conditions; their scores should be reported separately."
    )
    pdf.heading("Reproducibility boundary", 2)
    pdf.body_paragraph(
        f"The repository's original datasets and trained models are the 180 nm project path. The SKY130 generator writes separately under data/sky130_reference and records the model-deck hash, PDK revision, simulator version, sweep conditions, and row counts [{sky130_ref}]. SKY130 outputs are SPICE-model references, not measured silicon observations."
    )
    pdf.heading("Reproduction checklist", 2)
    for item in [
        "Record the source dataset checksum, model files, scaler files, package versions, and run command.",
        "Run the full pipeline with python main.py; for saved checkpoints use python main.py --skip-train.",
        "Use --skip-mc for a faster run or set --mc-runs to document the exact Monte Carlo sample count.",
        "Retain the generated plots, metrics.json, console output, and report alongside the run configuration.",
        "For SKY130, set PDK_ROOT and the exact SKY130_PDK_REVISION, then preserve the generated manifest and netlists.",
    ]:
        pdf.body_paragraph(f"- {item}", size=11, indent=3)


def add_code_appendix(pdf: ProjectReport) -> None:
    code_sections = [
        ("preprocess.py", "Data cleaning, feature definitions, train/test split, and train-only scaling."),
        ("models.py", "Shared feed-forward regression architecture and initialization."),
        ("current_predictor.py", "Model/scaler loading and current-prediction interface."),
        ("cmos_solver.py", "Numerical inverter transfer-curve and parameter extraction."),
        ("delay_power.py", "First-order timing, dynamic-power, and energy estimation."),
        ("main.py", "Command-line controls and ordered pipeline orchestration."),
    ]
    for filename, description in code_sections:
        path = ROOT / filename
        if not path.exists():
            continue
        pdf.section_page(f"APPENDIX: {filename}", description)
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[:48]
        if len(lines) > 48:
            selected += ["", "# ... remaining implementation is in the project source file ..."]
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(35, 45, 50)
        listing_height = 8 * 25.4 / 72
        for line in selected:
            safe_line = line.expandtabs(4).encode("ascii", "replace").decode("ascii")[:94]
            if pdf.get_y() > 265:
                pdf.add_page()
            pdf.set_x(25.4)
            pdf.cell(159.2, listing_height, safe_line, new_x="LMARGIN", new_y="NEXT")


def add_plot_appendix(pdf: ProjectReport) -> None:
    plots = [
        ("nmos_actual_vs_pred.png", "NMOS held-out prediction comparison"),
        ("pmos_actual_vs_pred.png", "PMOS held-out prediction comparison"),
        ("nmos_ids_vgs.png", "NMOS transfer characteristics"),
        ("pmos_ids_vsg.png", "PMOS transfer characteristics"),
        ("vtc_inv.png", "Inverter voltage-transfer characteristic"),
        ("vtc_all_gates.png", "Logic-gate transfer characteristics"),
        ("delay_vs_CL.png", "Estimated propagation delay versus capacitive load"),
        ("power_vs_freq.png", "Dynamic-power estimate versus frequency"),
        ("vtc_corners.png", "Process-corner transfer curves"),
        ("mc_vm_histogram.png", "Monte Carlo switching-voltage distribution"),
        ("ring_oscillator.png", "Ring-oscillator estimate"),
        ("sram_butterfly.png", "SRAM butterfly curve"),
    ]
    existing = [(PLOT_DIR / name, caption) for name, caption in plots if (PLOT_DIR / name).exists()]
    for start in range(0, len(existing), 2):
        pdf.section_page("APPENDIX: PROJECT FIGURES", "Generated plots from the repository; see captions for the measured quantity.")
        for path, caption in existing[start:start + 2]:
            pdf.set_font(FONT_FAMILY, "B", 10)
            pdf.set_text_color(24, 67, 83)
            pdf.multi_cell(159.2, 4, caption)
            pdf.ln(1)
            pdf.image(str(path), x=37, w=136)
            pdf.ln(5)


def add_repository_summary(pdf: ProjectReport, project_ref: int, sky130_ref: int) -> None:
    pdf.section_page("IMPLEMENTATION AND LIMITATIONS")
    entries = [
        ("preprocess.py", "Loads device data, normalizes units and polarity, removes invalid rows, transforms current to log space, splits data, and saves scalers."),
        ("models.py / train_nmos.py / train_pmos.py", "Defines and fits independent feed-forward current regressors and persists checkpoints and metrics."),
        ("current_predictor.py", "Provides scaled inference using the saved model weights and feature/target scalers."),
        ("cmos_solver.py / inverter.py / gates.py", "Composes device predictions and numerically solves transfer curves for inverter and logic-gate configurations."),
        ("delay_power.py", "Computes first-order delay, switched-capacitance power, energy, and EDP under stated assumptions."),
        ("corner_analysis.py / monte_carlo.py", "Evaluates the configured temperature/corner sweeps and sampled variation assumptions."),
        ("ring_oscillator.py / dff.py / sram.py", "Builds higher-level illustrative estimates from inverter delay or DC transfer behavior."),
        ("streamlit_app.py", "Presents model-backed analysis pages, controls, plots, exports, and SKY130 reference browsing."),
        ("spice/generate_sky130_dataset.py", f"Runs recorded SKY130A ngspice sweeps; derived results and provenance are kept separate from the 180 nm input datasets [{sky130_ref}]."),
        ("tests/", "Contains focused checks for circuit controls and SKY130 reference-data schema/metrics."),
    ]
    for name, detail in entries:
        pdf.heading(name, 3)
        pdf.body_paragraph(detail, size=11)
    pdf.body_paragraph(f"Module responsibilities summarized above are based on the project source and README [{project_ref}].")
    pdf.heading("Data and model limitations", 2)
    pdf.body_paragraph(
        "The project's original device datasets contain DC terminal conditions, geometry, temperature, and drain current. They do not provide extracted terminal capacitances, validated leakage or breakdown data, noise data, mismatch characterization, or measured transient waveforms. Consequently, first-order timing/power and system-level quantities are estimates governed by explicit assumptions, not foundry sign-off results. Dedicated PDK analyses and, where required, silicon measurements are needed to validate those quantities."
    )
    pdf.body_paragraph(
        "A model may fit sampled points well and still behave poorly outside the sampled feature domain. A random train/test split can also overstate generalization when nearby bias points from a sweep appear in both partitions. Future validation should include grouped splits by geometry/corner, independent SPICE sweeps, uncertainty reporting, and bounds checks before design decisions."
    )


def add_metrics_snapshot(pdf: ProjectReport, project_ref: int) -> None:
    metric_path = ROOT / "models" / "metrics.json"
    if not metric_path.exists():
        return
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    pdf.section_page("REPOSITORY METRICS SNAPSHOT", "Values below are the saved metrics.json snapshot and are distinct from thesis benchmark tables.")
    pdf.set_font(FONT_FAMILY, "B", 11)
    pdf.set_fill_color(24, 67, 83)
    pdf.set_text_color(255, 255, 255)
    widths = [38, 34, 41, 46]
    for name, width in zip(["Model", "R-squared", "MAE (A)", "RMSE (A)"], widths):
        pdf.cell(width, 8, name, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font(FONT_FAMILY, "", 11)
    pdf.set_text_color(30, 40, 48)
    for device in ("nmos", "pmos"):
        row = metrics.get(device, {})
        values = [
            device.upper(),
            f"{row.get('r2', float('nan')):.6f}",
            f"{row.get('mae', float('nan')):.6e}",
            f"{row.get('rmse', float('nan')):.6e}",
        ]
        for value, width in zip(values, widths):
            pdf.cell(width, 8, value, border=1, align="C")
        pdf.ln()
    pdf.ln(8)
    pdf.body_paragraph(
        f"The repository snapshot is reported without inferring the test protocol from the metric file alone [{project_ref}]. The values are not interchangeable with thesis-reported log-current errors or direct gate-model scores."
    )


def add_references(pdf: ProjectReport, references: list[str]) -> None:
    pdf.section_page("REFERENCES")
    bibliography = list(references)
    first_extra = len(references) + 1
    bibliography.extend([
        f"[{first_extra}] R. Nain, CMOS ML Circuit Characterization project source tree, README, and recorded run artifacts, 2026.",
        f"[{first_extra + 1}] SkyWater Technology Foundry, SKY130 Process Design Kit documentation and model files, public project repository; model revision and SHA-256 are recorded in data/sky130_reference/manifest.json.",
        f"[{first_extra + 2}] ngspice developers, ngspice circuit-level simulation program and user documentation; simulator version is recorded in data/sky130_reference/manifest.json.",
    ])
    for reference in bibliography:
        pdf.body_paragraph(reference, size=10)


def add_glossary(pdf: ProjectReport) -> None:
    pdf.section_page("GLOSSARY AND SYMBOLS")
    terms = [
        ("ANN", "Artificial neural network; a trainable function approximator used here for regression."),
        ("BSIM", "Berkeley short-channel IGFET model family used as a compact-model reference in SPICE."),
        ("CMOS", "Complementary MOS technology using NMOS and PMOS devices."),
        ("EDP", "Energy-delay product, a combined metric derived from energy and propagation delay."),
        ("FO4", "Fan-out-of-four delay reference: inverter delay under a load comparable to four similar inverter inputs."),
        ("gds / gm", "Small-signal output conductance and transconductance, estimated from local current slopes."),
        ("IDS", "Drain current; the learned target is log10 of the positive current magnitude."),
        ("MAE / RMSE", "Mean absolute error and root mean squared error; report units and target transformation with each."),
        ("Monte Carlo", "Repeated sampling under specified variation assumptions; results depend on the assumed distributions."),
        ("NMH / NML", "High- and low-level noise margins derived from the voltage-transfer characteristic."),
        ("PVT", "Process, voltage, and temperature operating conditions."),
        ("SNM", "Static noise margin; a DC stability measure commonly estimated from an SRAM butterfly curve."),
        ("SPICE", "Circuit simulation framework solving electrical equations using device models."),
        ("VTC / VM", "Voltage-transfer characteristic and its switching voltage."),
        ("VOH / VOL", "High- and low-level output voltages of a logic transfer characteristic."),
    ]
    for term, definition in terms:
        pdf.set_font(FONT_FAMILY, "B", 10)
        pdf.set_text_color(24, 67, 83)
        pdf.write(4, f"{term}: ")
        pdf.set_font(FONT_FAMILY, "", 10)
        pdf.set_text_color(30, 40, 48)
        pdf.write(4, definition)
        pdf.ln(6)


def add_reproducibility_appendices(
    pdf: ProjectReport,
    project_ref: int,
    spice_ref: int,
) -> None:
    pdf.section_page("APPENDIX: DATA AND PROVENANCE")
    pdf.heading("Record the training domain", 2)
    pdf.body_paragraph(
        f"For each dataset, preserve the column names, units, polarity convention, valid ranges, row count before and after cleaning, and checksum. NMOS and PMOS use different voltage conventions; the preprocessing path converts inputs to the model's declared feature convention before scaling. Width and length suffixes must be parsed into SI units consistently. A missing temperature field is an assumption, not a measured value, and should be stated with the dataset record [{project_ref}]."
    )
    pdf.heading("Keep the reference tracks separate", 2)
    pdf.body_paragraph(
        f"The original project data and models correspond to the 180 nm track. The SKY130 path has its own output directory and manifest with model-deck SHA-256, PDK revision, simulator version, sweep conditions, and row counts. Reproduction requires preserving this manifest with the generated netlists and CSV outputs; a model revision or simulator change can alter the reference data [{spice_ref}]."
    )
    pdf.heading("Minimum run record", 2)
    pdf.body_paragraph(
        "A useful run record includes the code revision, Python and package versions, random seeds, dataset checksum, model/scaler checkpoint hashes, command-line flags, Monte Carlo count and distribution assumptions, circuit controls, and output filenames. It also distinguishes generated simulation points from measured device data and identifies any points outside the training domain."
    )

    pdf.section_page("APPENDIX: MODEL EVALUATION")
    pdf.heading("Metrics must retain their target definitions", 2)
    pdf.body_paragraph(
        "R-squared is a normalized measure of explained variance; it does not establish small absolute error or physical validity. MAE and RMSE depend on the target's scale and units. For a log-current model, errors in log10(|IDS|) describe multiplicative current differences and should not be presented as ampere-valued errors without an explicit inverse transform. Gate-voltage, delay, power, and energy targets require their own units and error summaries."
    )
    pdf.heading("Test design and leakage checks", 2)
    pdf.body_paragraph(
        f"A random split can place neighboring points from the same voltage sweep in both train and test partitions. To test interpolation robustness, use grouped holdouts by bias sweep or geometry; to test transfer, reserve complete geometries, temperatures, or process corners. Fit all preprocessing transforms on training data only, then freeze them for validation and test inference [{project_ref}]."
    )
    pdf.heading("Circuit-level validation", 2)
    pdf.body_paragraph(
        "Compare predicted transfer curves with independent SPICE results at operating points not used for training. Check current balance, monotonicity, output rails, switching-point movement, residual shape, and behavior near the edges of the training domain. Report both aggregate errors and worst-case errors; inspect failure cases instead of relying on a single summary score."
    )

    pdf.section_page("APPENDIX: CIRCUIT ESTIMATE ASSUMPTIONS")
    pdf.heading("First-order performance equations", 2)
    pdf.body_paragraph(
        "The project uses the familiar dynamic-power relationship P = alpha C_L V_DD^2 f and forms energy per switching event as E = P/f under the same activity and load assumptions. Its first-order delay estimate scales with C_L V_DD / I_ON. These relations are useful for design-space comparisons, but actual delay and power depend on waveform shape, parasitic capacitances, input slew, short-circuit current, leakage, and load topology."
    )
    pdf.heading("Composed circuit quantities", 2)
    pdf.body_paragraph(
        "The ring-oscillator estimate uses f = 1/(2 N t_p) for an odd number of stages and assumed stage delay. DFF setup, hold, and clock-to-Q estimates in the code are derived from inverter delay multipliers; they are illustrative relationships, not transistor-level sequential-cell characterization. SRAM butterfly curves use DC transfer behavior and do not replace transient read/write, mismatch, or array-level verification."
    )
    pdf.heading("Use boundary labels", 2)
    pdf.body_paragraph(
        "Every reported circuit estimate should state its supply, geometry, load, activity/frequency, temperature/corner, and model source. Where terminal capacitance or transient data are not available, label dependent quantities as assumed or first-order. Do not present these estimates as silicon measurements, sign-off timing, yield, or foundry-qualified limits."
    )


def make_report() -> Path:
    blocks, references, abstract = extract_document()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = ProjectReport(format="A4", unit="mm")
    pdf.set_margins(25.4, 25.4, 25.4)
    pdf.set_auto_page_break(auto=True, margin=25.4)
    pdf.add_font(FONT_FAMILY, "", str(FONT_REGULAR))
    pdf.add_font(FONT_FAMILY, "B", str(FONT_BOLD))
    pdf.add_font(FONT_FAMILY, "I", str(FONT_ITALIC))
    add_cover(pdf)
    pdf.section_page("ABSTRACT")
    pdf.body_paragraph(abstract)
    project_ref = len(references) + 1
    sky130_ref = project_ref + 1
    add_project_context(pdf, project_ref, sky130_ref)
    add_flow_diagram(pdf)

    for style, text in blocks:
        if style == "Heading1":
            pdf.heading(text, 1)
        elif style == "Heading2":
            pdf.heading(text, 2)
        elif style == "Heading3":
            pdf.heading(text, 3)
        elif style == "Table":
            pdf.set_font(FONT_FAMILY, "I", 10)
            pdf.set_text_color(55, 65, 72)
            for row in text.splitlines():
                pdf.multi_cell(159.2, 3.6, row)
            pdf.ln(3)
        elif text.startswith("[") and re.match(r"^\[\d+\]", text):
            continue
        else:
            pdf.body_paragraph(text)

    add_repository_summary(pdf, project_ref, sky130_ref)
    add_metrics_snapshot(pdf, project_ref)
    add_plot_appendix(pdf)
    add_code_appendix(pdf)
    add_glossary(pdf)
    add_reproducibility_appendices(pdf, project_ref, sky130_ref + 2)
    add_references(pdf, references)

    page_count = pdf.page_no()
    if page_count != PAGE_TARGET:
        raise RuntimeError(
            f"Report currently renders to {page_count} pages, expected exactly {PAGE_TARGET}. "
            "Adjust substantive appendix content or layout; do not insert blank pages."
        )
    pdf.output(str(REPORT_PATH))
    return REPORT_PATH


if __name__ == "__main__":
    print(f"Created {make_report()}")
