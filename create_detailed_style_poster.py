from pathlib import Path
from zipfile import ZipFile
import tempfile
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(r"C:\Users\iitdr\OneDrive - randeepnain\Desktop\fnal")
THESIS = ROOT / "final" / "Ramandeep_6024620033.docx"
OUT_A0_PDF = ROOT / "final" / "CMOS_ML_180nm_Project_Poster_Detailed_A0.pdf"
OUT_A0_PNG = ROOT / "final" / "CMOS_ML_180nm_Project_Poster_Detailed_A0.png"
OUT_A1_PDF = ROOT / "final" / "CMOS_ML_180nm_Project_Poster_Detailed_A1.pdf"
OUT_A1_PNG = ROOT / "final" / "CMOS_ML_180nm_Project_Poster_Detailed_A1.png"
CONTACT_EMAIL = "rnain_mtech24@thapar.edu"

# A0 portrait, matching the reference's three-column academic-poster structure.
PAGE_W, PAGE_H = 33.11, 46.81
CYAN = "#18A5BF"
CYAN_DARK = "#11869B"
NAVY = "#18364A"
INK = "#26343B"
MUTED = "#53656D"
PAPER = "#FAF9F3"
PALE = "#F2F7F7"
WHITE = "#FFFFFF"
LINE = "#B8D5D9"
GOLD = "#E8A93B"
BLUE = "#2876A6"
TEAL = "#159C91"
FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=PAPER)


def section(x, y, w, h, title, size=13):
    """Create one bordered poster section; coordinates are inches from bottom."""
    ax = fig.add_axes([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=PAPER,
                           edgecolor=LINE, linewidth=1.0,
                           transform=ax.transAxes))
    bar_h = 0.095
    ax.add_patch(Rectangle((0, 1 - bar_h), 1, bar_h, facecolor=CYAN,
                           edgecolor=CYAN, transform=ax.transAxes))
    ax.text(0.5, 1 - bar_h / 2, title, fontsize=size, color=WHITE,
            fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    return ax


def add_text(ax, x, y, content, size=9.4, color=INK, bold=False,
             va="top", align="left", linespacing=1.28):
    ax.text(x, y, content, fontsize=size, color=color, fontweight="bold" if bold else "normal",
            ha=align, va=va, transform=ax.transAxes,
            wrap=True, linespacing=linespacing)


def paragraph(ax, y, content, size=9.2, x=0.055, width=0.89,
              color=INK, bold=False, leading=1.26):
    wrapped = textwrap.fill(content, width=max(25, int(width * 112)))
    add_text(ax, x, y, wrapped, size, color, bold, linespacing=leading)
    lines = wrapped.count("\n") + 1
    return y - lines * (size / 72) * leading / (ax.get_position().height * PAGE_H) - 0.025


def bullet_list(ax, y, items, size=9.0, color=INK, gap=0.035):
    for item in items:
        wrapped = textwrap.fill(item, width=96, subsequent_indent="   ")
        ax.text(0.065, y, "•", fontsize=size + 1, color=CYAN_DARK,
                fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.105, y, wrapped, fontsize=size, color=color,
                transform=ax.transAxes, va="top", wrap=True, linespacing=1.25)
        lines = wrapped.count("\n") + 1
        y -= lines * 0.033 + gap
    return y


def image(ax, path, x, y, w, h):
    with Image.open(path) as source:
        source = source.convert("RGB")
    inset = ax.inset_axes([x, y, w, h])
    inset.imshow(source)
    inset.set_axis_off()
    inset.set_aspect("equal", adjustable="box")
    return inset


def caption(ax, x, y, content, size=7.6, align="center"):
    ax.text(x, y, content, fontsize=size, color=NAVY, fontweight="bold",
            ha=align, va="top", transform=ax.transAxes, wrap=True,
            linespacing=1.15)


def small_plot_pair(ax, left, right, cap_left, cap_right,
                    y=0.17, image_h=0.29, image_top=0.52):
    image(ax, left, 0.035, image_top, 0.45, image_h)
    image(ax, right, 0.515, image_top, 0.45, image_h)
    caption(ax, 0.26, y, cap_left, 7.5)
    caption(ax, 0.74, y, cap_right, 7.5)


with tempfile.TemporaryDirectory(prefix="cmos_a0_media_") as temporary:
    media_dir = Path(temporary)
    with ZipFile(THESIS) as archive:
        for name in archive.namelist():
            if name.startswith("word/media/image"):
                (media_dir / Path(name).name).write_bytes(archive.read(name))

    def img(number):
        png = media_dir / f"image{number}.png"
        return png if png.exists() else media_dir / f"image{number}.jpg"

    qr_payload = f"mailto:{CONTACT_EMAIL}"
    qr_code = cv2.QRCodeEncoder_create().encode(qr_payload)
    qr_code = cv2.copyMakeBorder(
        qr_code, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
    )
    qr_code = cv2.resize(
        qr_code, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST
    )
    qr_path = media_dir / "contact_qr.png"
    if not cv2.imwrite(str(qr_path), qr_code):
        raise OSError(f"Could not write contact QR code to {qr_path}")

    # Header panel.
    header_y, header_h = 42.12, 4.45
    header = fig.add_axes([0.0, header_y / PAGE_H, 1.0, header_h / PAGE_H])
    header.set_axis_off()
    header.add_patch(Rectangle((0, 0), 1, 1, facecolor=CYAN,
                               transform=header.transAxes))
    header.add_patch(Rectangle((0, 0), 1, 0.055, facecolor=CYAN_DARK,
                               transform=header.transAxes))
    # Institute mark from the thesis cover.
    image(header, img(1), 0.018, 0.22, 0.12, 0.68)
    header.text(0.16, 0.91, "M.TECH VLSI DESIGN • FINAL VIVA PROJECT",
                fontsize=10, color=WHITE, fontweight="bold",
                transform=header.transAxes, va="top")
    header.text(
        0.16, 0.77,
        "Machine Learning-Assisted Electrical Characterization\n"
        "of 180 nm CMOS Digital Circuits",
        fontsize=26, color=WHITE, fontweight="bold",
        transform=header.transAxes, va="top", linespacing=1.13
    )
    header.text(
        0.16, 0.49,
        "TAKEAWAY | Device-current ANN surrogates reached R² = 0.9985 (NMOS) and 0.9967 (PMOS); "
        "a separate gate benchmark reported 90×–110× speed-up.",
        fontsize=11, color=WHITE, fontweight="bold",
        transform=header.transAxes, va="center"
    )
    header.text(
        0.16, 0.37,
        f"Ramandeep Nain  |  Roll No. 6024620033  |  M.Tech, VLSI Design  |  {CONTACT_EMAIL}",
        fontsize=11.5, color=WHITE, fontweight="bold",
        transform=header.transAxes, va="center"
    )
    header.text(
        0.16, 0.23,
        "Supervisor: Dr. Sumit Vyas, Assistant Professor  •  Electronics and Communication Engineering",
        fontsize=9.5, color=WHITE, transform=header.transAxes, va="center"
    )
    header.text(
        0.16, 0.12,
        "Thapar Institute of Engineering and Technology, Patiala  |  June 2026",
        fontsize=9.5, color=WHITE, transform=header.transAxes, va="center"
    )
    image(header, qr_path, 0.87, 0.19, 0.10, 0.74)
    header.text(
        0.92, 0.12, "SCAN TO EMAIL", fontsize=7.5, color=WHITE,
        fontweight="bold", transform=header.transAxes,
        ha="center", va="center"
    )

    # Three-column layout.
    margin, gutter = 0.38, 0.20
    col_w = (PAGE_W - 2 * margin - 2 * gutter) / 3
    cols = [margin, margin + col_w + gutter, margin + 2 * (col_w + gutter)]
    bottom = 1.45
    available = 40.34
    gap = 0.19

    # LEFT COLUMN
    y = 36.97
    h = 5.15
    ax = section(cols[0], y, col_w, h, "Introduction")
    yy = 0.84
    yy = paragraph(ax, yy,
        "CMOS circuit design relies on repeated electrical characterization of transistors and logic cells. "
        "Designers sweep bias, geometry, supply voltage, load, frequency, temperature and process conditions "
        "to understand current, switching behavior, timing and power.",
        9.5)
    yy = paragraph(ax, yy,
        "Physics-based compact models such as BSIM3v3 provide the SPICE reference, but repeated nonlinear "
        "circuit solves can make broad characterization and optimization expensive.",
        9.5)
    yy = paragraph(ax, yy,
        "This thesis investigates supervised machine-learning surrogates that learn selected input-output "
        "relationships from SPICE data and reuse them for faster design-space exploration at the mature "
        "180 nm node.",
        9.5)
    yy = paragraph(ax, yy,
        "Research question: Can learned models accelerate device and circuit characterization while "
        "preserving sufficient agreement with the simulator over the conditions tested?",
        9.2, color=NAVY, bold=True)

    y = 31.43
    h = 5.35
    ax = section(cols[0], y, col_w, h, "Literature Review")
    yy = 0.84
    yy = paragraph(ax, yy,
        "SPICE, introduced by Nagel and Pederson, established the numerical framework used by modern "
        "circuit simulation. Compact models such as BSIM capture short-channel transistor behavior and "
        "provide a practical device-level reference [1, 2].",
        9.2)
    yy = paragraph(ax, yy,
        "Simulation acceleration has been explored through FastSPICE algorithms, model-order reduction, "
        "parallel computation and data-driven surrogates. Neural networks, Gaussian processes, support "
        "vector regression and ensemble methods offer different trade-offs between speed, uncertainty "
        "and model flexibility.",
        9.2)
    yy = paragraph(ax, yy,
        "A key challenge for learned models is generalization: good fit on sampled data does not guarantee "
        "physical behavior outside the training domain. For this reason, the thesis combines held-out "
        "testing with SPICE comparison, circuit-level checks and explicit discussion of model scope.",
        9.2)
    yy = paragraph(ax, yy,
        "Research gap addressed: a practical workflow linking simulator-generated data, learned device "
        "or gate response, numerical circuit solving and system-level examples.",
        9.2, color=NAVY, bold=True)
    image(ax, img(7), 0.18, 0.055, 0.64, 0.19)
    caption(ax, 0.5, 0.04, "Machine-learning approaches reviewed in the thesis", 7.2)

    y = 25.99
    h = 5.25
    ax = section(cols[0], y, col_w, h, "Research Objectives & Contributions")
    yy = 0.84
    yy = bullet_list(ax, yy, [
        "Develop ANN-based models for 180 nm NMOS/PMOS current and CMOS gate performance.",
        "Represent operating conditions, device geometry and relevant process or temperature inputs.",
        "Evaluate inverter, AND, OR, NAND, NOR and XNOR behavior against SPICE reference results.",
        "Integrate learned behavior with numerical solving and optimization workflows.",
        "Demonstrate selected system-level analyses: SRAM, D flip-flop, ring oscillator, multiplexers and adders."
    ], 9.1, gap=0.03)
    yy = paragraph(ax, yy - 0.02,
        "Contribution: two complementary surrogate tracks—direct prediction of gate metrics and "
        "device-current models composed through current balance.",
        9.0, color=NAVY, bold=True)

    y = 18.90
    h = 6.9
    ax = section(cols[0], y, col_w, h, "CMOS Device Foundation")
    image(ax, img(8), 0.14, 0.56, 0.72, 0.30)
    caption(ax, 0.5, 0.535, "Four-terminal MOSFET structure (thesis Figure 5)", 7.5)
    yy = 0.48
    yy = paragraph(ax, yy,
        "A MOSFET comprises gate, drain, source and body terminals. The gate-source and drain-source "
        "biases determine channel charge and drain current; geometry and temperature also affect "
        "the electrical response.",
        9.0)
    yy = paragraph(ax, yy,
        "Across operating regions, current transitions from weak-inversion conduction below threshold "
        "to approximately linear behavior at small drain bias and saturation-like behavior at larger "
        "drain bias. Short-channel effects and process variation complicate simple square-law models.",
        9.0)
    yy = paragraph(ax, yy,
        "Small-signal quantities such as transconductance gm = dID/dVGS and output conductance "
        "gds = dID/dVDS describe local sensitivity. In the thesis, automatic differentiation is used "
        "with the differentiable neural-network model.",
        8.8)
    yy = paragraph(ax, yy,
        "Voltage-transfer curves, switching thresholds and noise margins connect device response "
        "to logic-level behavior.",
        8.8)

    y = 12.41
    h = 6.3
    ax = section(cols[0], y, col_w, h, "Inputs, Targets & Evaluation")
    bullet_list(ax, 0.84, [
        "Device inputs: VGS, VDS, geometry and operating conditions; target: drain current ID.",
        "Circuit inputs include sizing and operating conditions such as VDD, load, frequency and temperature.",
        "Circuit targets include VOH, VOL, tPLH, tPHL, average/peak/static power and leakage-related metrics.",
        "Reported metrics: R-squared, MAE and RMSE; device-current networks are assessed in log-current space.",
        "For the circuit track, the thesis reports an 80/20 split (85/15 for XNOR); preprocessing is fitted using training data."
    ], 9.0, gap=0.035)
    image(ax, img(12), 0.24, 0.035, 0.52, 0.23)
    caption(ax, 0.5, 0.02, "Feedforward network: inputs → hidden layers → predicted outputs", 7.2)

    y = 7.12
    h = 5.10
    ax = section(cols[0], y, col_w, h, "Technology Context & Variability")
    image(ax, img(4), 0.05, 0.50, 0.43, 0.31)
    image(ax, img(5), 0.52, 0.50, 0.43, 0.31)
    caption(ax, 0.265, 0.475, "180 nm in scaling history", 7.0)
    caption(ax, 0.735, 0.475, "Sources of device variation", 7.0)
    paragraph(ax, 0.41,
        "Mature-node design remains important in mixed-signal and industrial applications. "
        "The thesis identifies random dopant fluctuation, line-edge roughness, oxide-thickness "
        "variation and metal-gate granularity as contributors to device variability.",
        8.7)
    paragraph(ax, 0.21,
        "Process and temperature changes can shift threshold, switching point, delay and power. "
        "Predictions are meaningful only within conditions adequately represented in reference data.",
        8.7, color=NAVY, bold=True)

    y = 1.65
    h = 5.28
    ax = section(cols[0], y, col_w, h, "Model Training & Evaluation")
    image(ax, img(13), 0.04, 0.42, 0.43, 0.32)
    image(ax, img(20), 0.53, 0.42, 0.43, 0.32)
    caption(ax, 0.255, 0.395, "Training loss", 7.1)
    caption(ax, 0.745, 0.395, "Validation loss", 7.1)
    bullet_list(ax, 0.32, [
        "R-squared measures explained variance; MAE and RMSE quantify error magnitude.",
        "Device-current metrics are calculated in log-current space; circuit targets use their own physical units.",
        "A held-out test set checks generalization; fit scores are not equally strong for every target."
    ], 8.1, gap=0.012)

    # MIDDLE COLUMN
    y = 36.02
    h = 6.1
    ax = section(cols[1], y, col_w, h, "Methods: Two Modeling Tracks")
    ax.add_patch(FancyBboxPatch((0.055, 0.54), 0.89, 0.29,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PALE, edgecolor=LINE, transform=ax.transAxes))
    add_text(ax, 0.08, 0.79, "PART I • DIRECT GATE-METRIC SURROGATES",
             9.4, BLUE, True)
    add_text(ax, 0.08, 0.70,
        "LTspice gate sweeps provide labels; ANN models predict voltage, delay, power and leakage-related metrics for six logic-gate types.",
        8.7, INK)
    ax.add_patch(FancyBboxPatch((0.055, 0.18), 0.89, 0.29,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PALE, edgecolor=LINE, transform=ax.transAxes))
    add_text(ax, 0.08, 0.43, "PART II • DEVICE-CENTRIC SURROGATES",
             9.4, TEAL, True)
    add_text(ax, 0.08, 0.34,
        "Separate NMOS and PMOS networks learn drain current from BSIM-SPICE data; device predictions are composed using current balance and gate-scaling rules.",
        8.7, INK)
    add_text(ax, 0.07, 0.09,
        "The tracks answer related but different questions and use different labels and evaluation scales.",
        8.5, NAVY, True)

    y = 29.23
    h = 6.6
    ax = section(cols[1], y, col_w, h, "SPICE-to-ML Methodology")
    image(ax, img(11), 0.06, 0.50, 0.88, 0.34)
    caption(ax, 0.5, 0.475, "Thesis methodology: characterize, train, validate, optimize", 7.5)
    steps = [
        ("1  Define design space", "Bias, device sizing, supply, load and relevant PVT conditions."),
        ("2  Generate reference data", "Run SPICE using the selected 180 nm compact-model setup."),
        ("3  Prepare datasets", "Filter implausible points; transform targets and scale inputs."),
        ("4  Train regressors", "Use supervised learning with validation and held-out testing."),
        ("5  Validate and apply", "Compare against reference; use surrogates in circuit solving or design search.")
    ]
    yy = 0.42
    for title, detail in steps:
        add_text(ax, 0.075, yy, title, 8.8, NAVY, True)
        add_text(ax, 0.12, yy - 0.035, detail, 8.1, INK)
        yy -= 0.074

    y = 23.04
    h = 6.0
    ax = section(cols[1], y, col_w, h, "Training & Validation")
    image(ax, img(13), 0.03, 0.50, 0.45, 0.32)
    image(ax, img(20), 0.52, 0.50, 0.45, 0.32)
    caption(ax, 0.255, 0.475, "Training loss", 7.3)
    caption(ax, 0.745, 0.475, "Validation loss", 7.3)
    yy = 0.40
    yy = bullet_list(ax, yy, [
        "ANN regression: ReLU activation, mean-squared error and Adam optimizer.",
        "Train-only normalization statistics are used to avoid leaking test-set information.",
        "Test metrics are target-specific; R-squared alone does not describe absolute error.",
        "Circuit-level references and held-out device points provide complementary checks."
    ], 8.7, gap=0.02)
    add_text(ax, 0.07, 0.065,
        "A low training loss is not sufficient: generalization must be checked on unseen samples.",
        8.4, NAVY, True)

    y = 14.85
    h = 8.0
    ax = section(cols[1], y, col_w, h, "Results 1 • Device-Level Characterization")
    image(ax, img(25), 0.03, 0.70, 0.45, 0.21)
    image(ax, img(26), 0.52, 0.70, 0.45, 0.21)
    caption(ax, 0.255, 0.68, "NMOS transfer curve", 7.4)
    caption(ax, 0.745, 0.68, "PMOS transfer curve", 7.4)
    image(ax, img(27), 0.03, 0.43, 0.45, 0.21)
    image(ax, img(28), 0.52, 0.43, 0.45, 0.21)
    caption(ax, 0.255, 0.41, "NMOS output characteristic", 7.2)
    caption(ax, 0.745, 0.41, "PMOS output characteristic", 7.2)
    add_text(ax, 0.06, 0.355,
        "The learned current surfaces reproduce transfer and output behavior across bias conditions. "
        "Log-current plots expose the weak-inversion region that is difficult to see on a linear scale.",
        8.7, INK)
    image(ax, img(29), 0.04, 0.105, 0.43, 0.20)
    image(ax, img(30), 0.53, 0.105, 0.43, 0.20)
    caption(ax, 0.255, 0.087, "NMOS log-current", 7.2)
    caption(ax, 0.745, 0.087, "PMOS log-current", 7.2)

    y = 1.65
    h = 13.01
    ax = section(cols[1], y, col_w, h, "Quantitative Results")
    rows = [
        ("Part II NMOS", "R² 0.9985", "RMSE(log) 0.0985"),
        ("Part II PMOS", "R² 0.9967", "RMSE(log) 0.1603"),
        ("Part I NMOS ID", "R² 0.9983", "MAE 4.494 × 10⁻⁶ A"),
        ("Part I inverter VOUT", "R² 0.9571", "MAE 49.9 mV"),
        ("Part I XNOR PDP", "R² 0.9981*", "Figure caption"),
    ]
    yy = 0.84
    for i, (model, score, detail) in enumerate(rows):
        fill = PALE if i % 2 == 0 else PAPER
        ax.add_patch(Rectangle((0.045, yy - 0.065), 0.91, 0.09,
                               facecolor=fill, edgecolor=LINE, linewidth=0.45,
                               transform=ax.transAxes))
        add_text(ax, 0.06, yy, model, 8.8, INK, i == 0)
        add_text(ax, 0.46, yy, score, 8.8, BLUE, True)
        add_text(ax, 0.72, yy, detail, 7.8, MUTED)
        yy -= 0.10
    add_text(ax, 0.06, yy - 0.005,
        "*Reported in thesis Figure 19 caption. Values above come from different tracks and targets; do not compare as one ranking.",
        8.2, NAVY, True)
    add_text(ax, 0.06, yy - 0.075,
        "Part I gate results vary by metric. Examples from Table 11: OR VOH R² = 0.9964 but VOL = 0.9201; XNOR PDP R² = 0.9983 but VOL = 0.9063.",
        8.0, INK)
    add_text(ax, 0.06, 0.22,
        "BSIM-SPICE reference validation (Part II): the thesis reports correlation above 0.999 across approximately 10⁻¹²–10⁻³ A.",
        8.5, NAVY, True)
    image(ax, img(45), 0.05, 0.025, 0.43, 0.15)
    image(ax, img(46), 0.52, 0.025, 0.43, 0.15)
    caption(ax, 0.265, 0.012, "NMOS: ML vs BSIM-SPICE", 7.1)
    caption(ax, 0.735, 0.012, "PMOS: ML vs BSIM-SPICE", 7.1)

    # RIGHT COLUMN
    y = 34.82
    h = 7.3
    ax = section(cols[2], y, col_w, h, "Results 2 • Gate-Level Characterization")
    image(ax, img(31), 0.04, 0.54, 0.44, 0.34)
    image(ax, img(32), 0.52, 0.54, 0.44, 0.34)
    caption(ax, 0.26, 0.515, "Inverter VTC", 7.4)
    caption(ax, 0.74, 0.515, "Voltage gain", 7.4)
    add_text(ax, 0.065, 0.455,
        "At VDD = 1.8 V, the thesis reports VOH = 1.800 V, VOL = 0 V, "
        "VIL = 0.703 V, VIH = 0.985 V and switching threshold VM = 0.855 V.",
        8.8, INK)
    add_text(ax, 0.065, 0.35,
        "Noise margins are NML = 0.703 V and NMH = 0.815 V. Peak voltage-gain magnitude is 20.04. "
        "These values describe model-derived inverter behavior at the stated conditions.",
        8.8, INK)
    add_text(ax, 0.065, 0.205,
        "The VTC transition and its derivative explain the inverter's restoring behavior and noise tolerance.",
        8.5, NAVY, True)

    y = 27.83
    h = 6.8
    ax = section(cols[2], y, col_w, h, "Gate Delay, Power & Energy")
    image(ax, img(34), 0.03, 0.50, 0.45, 0.34)
    image(ax, img(35), 0.52, 0.50, 0.45, 0.34)
    caption(ax, 0.255, 0.475, "Propagation delay (ps)", 7.2)
    caption(ax, 0.745, 0.475, "Dynamic power (µW)", 7.2)
    image(ax, img(36), 0.26, 0.19, 0.48, 0.24)
    caption(ax, 0.5, 0.165, "Energy comparison (thesis Figure 29)", 7.2)
    add_text(ax, 0.06, 0.10,
        "At the reported operating point: INV = 1.487 ps, 1.62 µW, 3.24 fJ; "
        "XOR = 23.796 ps, 6.48 µW, 12.96 fJ.",
        8.3, NAVY, True)

    y = 20.54
    h = 7.1
    ax = section(cols[2], y, col_w, h, "Results 3 • System-Level Examples")
    image(ax, img(37), 0.03, 0.54, 0.45, 0.32)
    image(ax, img(38), 0.52, 0.54, 0.45, 0.32)
    caption(ax, 0.255, 0.515, "6T SRAM butterfly", 7.2)
    caption(ax, 0.745, 0.515, "D flip-flop timing", 7.2)
    add_text(ax, 0.06, 0.445,
        "SRAM: the butterfly curve illustrates bistability; reported static noise margin is approximately 54.3 mV for the stated model configuration.",
        8.3, INK)
    image(ax, img(39), 0.03, 0.16, 0.45, 0.22)
    image(ax, img(40), 0.52, 0.16, 0.45, 0.22)
    caption(ax, 0.255, 0.14, "Five-stage ring-oscillator output", 7.0)
    caption(ax, 0.745, 0.14, "Frequency vs stage count", 7.0)
    add_text(ax, 0.06, 0.08,
        "Ring-oscillator estimate: N = 5, stage delay = 1.487 ps, frequency = 67.24 GHz.",
        8.1, NAVY, True)

    y = 13.85
    h = 6.5
    ax = section(cols[2], y, col_w, h, "Additional Circuits & Validation")
    image(ax, img(41), 0.035, 0.58, 0.44, 0.25)
    image(ax, img(42), 0.52, 0.58, 0.44, 0.25)
    caption(ax, 0.255, 0.555, "MUX family (2:1–16:1)", 7.0)
    caption(ax, 0.74, 0.555, "Full-adder outputs", 7.0)
    add_text(ax, 0.06, 0.48,
        "The device-centric framework is also applied to a half adder and full adder, extending the examples to arithmetic behavior.",
        8.2, INK)
    add_text(ax, 0.06, 0.365,
        "Robustness checks include 20 Monte Carlo runs with ±5% variation (reported VM spread about ±0.09 V) and a qualitative 2.2 V extrapolation beyond 1.8 V training.",
        8.2, INK)
    add_text(ax, 0.06, 0.22,
        "Thesis reports >0.999 correlation against BSIM-SPICE across approximately 10⁻¹²–10⁻³ A and mean relative current-balance error of 0.065%.",
        8.2, NAVY, True)
    add_text(ax, 0.06, 0.09,
        "Monte Carlo and extrapolation are exploratory—not yield certification or guaranteed out-of-domain accuracy.",
        7.9, MUTED)

    y = 8.06
    h = 5.6
    ax = section(cols[2], y, col_w, h, "Discussion, Conclusion & Next Steps")
    yy = 0.84
    yy = paragraph(ax, yy,
        "The thesis demonstrates that neural surrogates can approximate selected SPICE-generated device and gate responses and can be composed into representative circuit analyses. The reported Part I benchmark shows 90×–110× speed-up versus SPICE for its measurement setup.",
        8.5)
    yy = paragraph(ax, yy,
        "Accuracy is target-dependent. Strong R-squared scores for device current or PDP do not imply equal accuracy for every voltage, timing or power metric. Results from Part I and Part II use different targets and scales.",
        8.5)
    yy = paragraph(ax, yy,
        "Limitations: results are model/simulator-derived, not silicon measurements, and training covers a finite 180 nm domain. A separate, rough SKY130 first-order cross-check at one condition (27°C, 1.8 V, 10 fF, 1 ns; five corners) showed mean absolute errors of 48.1% for fall delay, 47.1% for rise delay, and 23.0% for dynamic power/energy. It does not validate the 180 nm models.",
        8.3)
    yy = paragraph(ax, yy,
        "Conclusion: ML models are a fast complement for early design-space exploration. Final sign-off requires validated PDK/SPICE analysis and, where appropriate, silicon measurement.",
        8.5, color=NAVY, bold=True)
    yy = paragraph(ax, yy,
        "Next steps: validate against the target 180 nm PDK using transient SPICE; sweep load capacitance and input slew across process corners and temperatures; then report error distributions before using the models for design decisions.",
        8.5, color=NAVY, bold=True)

    y = 1.65
    h = 6.22
    ax = section(cols[2], y, col_w, h, "References")
    references = [
        "[1] L. W. Nagel and D. O. Pederson, “SPICE,” University of California, Berkeley, Memo UCB/ERL M382, 1973.",
        "[2] W. Liu et al., BSIM3v3.3 MOSFET Model User’s Manual, University of California, Berkeley, 2005.",
        "[3] Y. Taur and T. H. Ning, Fundamentals of Modern VLSI Devices, 2nd ed., Cambridge University Press, 2009.",
        "[4] G. Gielen and R. Rutenbar, “Computer-aided design of analog and mixed-signal integrated circuits,” Proceedings of the IEEE, vol. 88, no. 12, pp. 1825–1854, 2000.",
        "[5] A. Asenov, “Random dopant induced threshold voltage lowering and fluctuations in sub-0.1 μm MOSFETs,” IEEE Transactions on Electron Devices, vol. 45, no. 12, pp. 2505–2513, 1998.",
        "[6] R. Nain, “Machine Learning-Assisted Electrical Characterization of 180 nm CMOS Digital Circuits,” M.Tech thesis, Thapar Institute of Engineering and Technology, Patiala, 2026."
    ]
    yy = 0.87
    for i, ref in enumerate(references):
        wrapped = textwrap.fill(ref, width=104)
        ax.text(0.055, yy, wrapped, fontsize=8.15, color=INK,
                transform=ax.transAxes, va="top", wrap=True, linespacing=1.25)
        yy -= (wrapped.count("\n") + 1) * 0.045 + 0.035
    add_text(ax, 0.055, 0.07,
        "Numerical values and project figures are drawn from the thesis; simulator/model-derived results are not measured silicon data.",
        8.0, NAVY, True)

    # Full-width poster footer.
    footer = fig.add_axes([0.012, 0.008, 0.976, 1.15 / PAGE_H])
    footer.set_axis_off()
    footer.add_patch(Rectangle((0, 0), 1, 1, facecolor=CYAN,
                               transform=footer.transAxes))
    footer.text(0.02, 0.63, "TAKE-HOME MESSAGE", fontsize=9,
                color=WHITE, fontweight="bold",
                transform=footer.transAxes, va="center")
    footer.text(
        0.18, 0.63,
        "Use ANN surrogates for rapid exploration; confirm candidate designs with validated target-process SPICE before sign-off.",
        fontsize=10, color=WHITE, fontweight="bold",
        transform=footer.transAxes, va="center"
    )
    footer.text(
        0.02, 0.23,
        "Poster prepared from the M.Tech thesis. All performance figures are simulator/model-derived unless stated otherwise.",
        fontsize=8, color=WHITE, transform=footer.transAxes, va="center"
    )

    fig.savefig(OUT_A0_PDF, facecolor=PAPER, bbox_inches=None, metadata={
        "Title": "Machine Learning-Assisted Electrical Characterization of 180 nm CMOS Digital Circuits",
        "Author": "Ramandeep Nain",
        "Subject": "Detailed A0 portrait research poster with introduction, literature, methods, results, discussion, conclusion and references",
    })
    fig.savefig(OUT_A0_PNG, dpi=120, facecolor=PAPER, bbox_inches=None)
    plt.close(fig)

with Image.open(OUT_A0_PNG) as source:
    width = 3509
    height = round(width * (46.81 / 33.11))
    a1 = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    a1.save(OUT_A1_PNG, dpi=(150, 150), optimize=True)
    a1.save(OUT_A1_PDF, "PDF", resolution=150.0)

for output in (OUT_A0_PDF, OUT_A0_PNG, OUT_A1_PDF, OUT_A1_PNG):
    print(f"Created {output} ({output.stat().st_size:,} bytes)")
