"""Create editable A0 and A1 PowerPoint versions of the project poster."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
THESIS = ROOT / "final" / "Ramandeep_6024620033.docx"
OUT_DIR = ROOT / "final"
A0_W, A0_H = 33.11, 46.81
A1_W, A1_H = 23.39, 33.11
COLORS = {
    "cyan": "18A5BF",
    "cyan_dark": "11869B",
    "navy": "18364A",
    "ink": "26343B",
    "muted": "53656D",
    "paper": "FAF9F3",
    "pale": "F2F7F7",
    "white": "FFFFFF",
    "line": "B8D5D9",
    "blue": "2876A6",
    "teal": "159C91",
}


def rgb(name: str) -> RGBColor:
    value = COLORS.get(name, name).lstrip("#")
    return RGBColor.from_string(value)


def load_media() -> dict[int, bytes]:
    result: dict[int, bytes] = {}
    with ZipFile(THESIS) as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/image"):
                continue
            stem = Path(name).stem
            number_text = stem.removeprefix("image")
            if not number_text.isdigit():
                continue
            try:
                with Image.open(BytesIO(archive.read(name))) as source:
                    image = source.convert("RGB")
                    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=88, optimize=True)
                    result[int(number_text)] = buffer.getvalue()
            except OSError:
                continue
    return result


class EditablePoster:
    def __init__(self, width: float, height: float, media: dict[int, bytes]):
        self.width = width
        self.height = height
        self.scale = width / A0_W
        self.media = media
        self.presentation = Presentation()
        self.presentation.slide_width = Inches(width)
        self.presentation.slide_height = Inches(height)
        self.presentation.core_properties.title = (
            "Machine Learning-Assisted Electrical Characterization of 180 nm CMOS Digital Circuits"
        )
        self.presentation.core_properties.author = "Ramandeep Nain"
        self.presentation.core_properties.subject = (
            "Editable academic poster; text and layout are PowerPoint objects"
        )
        self.slide = self.presentation.slides.add_slide(
            self.presentation.slide_layouts[6]
        )
        self.rect(0, 0, width, height, "paper", line="paper")

    def rect(
        self,
        x: float,
        y_top: float,
        width: float,
        height: float,
        fill: str,
        *,
        line: str | None = None,
        line_pt: float = 0.75,
        rounded: bool = False,
    ):
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
        shape = self.slide.shapes.add_shape(
            shape_type,
            Inches(x * self.scale),
            Inches(y_top * self.scale),
            Inches(width * self.scale),
            Inches(height * self.scale),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.color.rgb = rgb(line or fill)
        shape.line.width = Pt(line_pt * self.scale)
        return shape

    def text(
        self,
        x: float,
        y_top: float,
        width: float,
        height: float,
        value: str,
        *,
        size: float = 9,
        color: str = "ink",
        bold: bool = False,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
        margin: float = 0,
        font: str = "Arial",
        fit: bool = True,
        line_spacing: float = 1.1,
    ):
        box = self.slide.shapes.add_textbox(
            Inches(x * self.scale),
            Inches(y_top * self.scale),
            Inches(width * self.scale),
            Inches(height * self.scale),
        )
        frame = box.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_left = Inches(margin * self.scale)
        frame.margin_right = Inches(margin * self.scale)
        frame.margin_top = Inches(margin * self.scale)
        frame.margin_bottom = Inches(margin * self.scale)
        frame.vertical_anchor = valign
        if fit:
            frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        for index, line in enumerate(value.split("\n")):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.alignment = align
            paragraph.space_before = Pt(0)
            paragraph.space_after = Pt(0)
            paragraph.line_spacing = line_spacing
            for run in paragraph.runs:
                run.font.name = font
                run.font.size = Pt(size * self.scale)
                run.font.bold = bold
                run.font.color.rgb = rgb(color)
        return box

    def panel(self, x: float, y_bottom: float, width: float, height: float, title: str):
        y_top = A0_H - y_bottom - height
        self.rect(x, y_top, width, height, "paper", line="line", line_pt=1)
        header_height = 0.095 * height
        self.rect(x, y_top, width, header_height, "cyan", line="cyan")
        self.text(
            x + 0.12,
            y_top + 0.015,
            width - 0.24,
            header_height - 0.02,
            title,
            size=13,
            color="white",
            bold=True,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        return {"x": x, "y": y_top, "w": width, "h": height}

    def panel_text(
        self,
        panel: dict[str, float],
        x_rel: float,
        y_rel: float,
        width_rel: float,
        height_rel: float,
        value: str,
        *,
        size: float = 9,
        color: str = "ink",
        bold: bool = False,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        fit: bool = True,
    ):
        x = panel["x"] + x_rel * panel["w"]
        y = panel["y"] + (1 - y_rel) * panel["h"]
        return self.text(
            x,
            y,
            width_rel * panel["w"],
            height_rel * panel["h"],
            value,
            size=size,
            color=color,
            bold=bold,
            align=align,
            fit=fit,
        )

    def panel_image(
        self,
        panel: dict[str, float],
        number: int,
        x_rel: float,
        y_rel: float,
        width_rel: float,
        height_rel: float,
    ) -> None:
        payload = self.media.get(number)
        if payload is None:
            self.panel_text(
                panel,
                x_rel,
                y_rel,
                width_rel,
                min(height_rel, 0.08),
                f"[Image {number} unavailable]",
                size=7,
                color="muted",
            )
            return
        box_w = width_rel * panel["w"]
        box_h = height_rel * panel["h"]
        box_x = panel["x"] + x_rel * panel["w"]
        box_y = panel["y"] + (1 - y_rel) * panel["h"]
        with Image.open(BytesIO(payload)) as source:
            image_ratio = source.width / source.height
        box_ratio = box_w / box_h
        if image_ratio > box_ratio:
            picture_w = box_w
            picture_h = box_w / image_ratio
        else:
            picture_h = box_h
            picture_w = box_h * image_ratio
        x = box_x + (box_w - picture_w) / 2
        y = box_y + (box_h - picture_h) / 2
        self.slide.shapes.add_picture(
            BytesIO(payload),
            Inches(x * self.scale),
            Inches(y * self.scale),
            width=Inches(picture_w * self.scale),
            height=Inches(picture_h * self.scale),
        )

    def caption(
        self,
        panel: dict[str, float],
        x_rel: float,
        y_rel: float,
        value: str,
        width_rel: float = 0.44,
        size: float = 7.4,
    ) -> None:
        self.panel_text(
            panel,
            x_rel - width_rel / 2,
            y_rel,
            width_rel,
            0.055,
            value,
            size=size,
            color="navy",
            bold=True,
            align=PP_ALIGN.CENTER,
        )

    def add_header(self) -> None:
        height = 4.45
        y = A0_H - 42.12 - height
        self.rect(0, y, A0_W, height, "cyan", line="cyan")
        self.rect(0, y + height - 0.24, A0_W, 0.24, "cyan_dark", line="cyan_dark")
        self.add_image(1, 0.6, y + 0.45, 3.8, 3.1)
        self.text(
            5.3, y + 0.32, 27.1, 0.42,
            "M.TECH VLSI DESIGN • FINAL VIVA PROJECT",
            size=10, color="white", bold=True,
        )
        self.text(
            5.3, y + 0.88, 27.0, 1.55,
            "Machine Learning-Assisted Electrical Characterization\nof 180 nm CMOS Digital Circuits",
            size=26, color="white", bold=True, line_spacing=1.1,
        )
        self.text(
            5.3, y + 2.45, 27.0, 0.42,
            "Ramandeep Nain  |  Roll No. 6024620033  |  M.Tech, VLSI Design",
            size=12.5, color="white", bold=True,
        )
        self.text(
            5.3, y + 3.05, 27.0, 0.35,
            "Supervisor: Dr. Sumit Vyas, Assistant Professor  •  Electronics and Communication Engineering",
            size=9.5, color="white",
        )
        self.text(
            5.3, y + 3.53, 27.0, 0.35,
            "Thapar Institute of Engineering and Technology, Patiala  |  June 2026",
            size=9.5, color="white",
        )

    def add_image(
        self, number: int, x: float, y_top: float, width: float, height: float
    ) -> None:
        payload = self.media.get(number)
        if payload is None:
            return
        with Image.open(BytesIO(payload)) as source:
            ratio = source.width / source.height
        box_ratio = width / height
        if ratio > box_ratio:
            actual_w, actual_h = width, width / ratio
        else:
            actual_h, actual_w = height, height * ratio
        x += (width - actual_w) / 2
        y_top += (height - actual_h) / 2
        self.slide.shapes.add_picture(
            BytesIO(payload),
            Inches(x * self.scale),
            Inches(y_top * self.scale),
            width=Inches(actual_w * self.scale),
            height=Inches(actual_h * self.scale),
        )

    def build_panels(self) -> None:
        margin, gutter = 0.38, 0.20
        col_w = (A0_W - 2 * margin - 2 * gutter) / 3
        cols = [margin, margin + col_w + gutter, margin + 2 * (col_w + gutter)]
        left, middle, right = cols

        p = self.panel(left, 36.97, col_w, 5.15, "Introduction")
        self.panel_text(p, .055, .84, .89, .67,
            "CMOS circuit design relies on repeated electrical characterization of transistors and logic cells. Designers sweep bias, geometry, supply voltage, load, frequency, temperature and process conditions to understand current, switching behavior, timing and power.\n\n"
            "Physics-based compact models such as BSIM3v3 provide the SPICE reference, but repeated nonlinear circuit solves can make broad characterization and optimization expensive.\n\n"
            "This thesis investigates supervised machine-learning surrogates that learn selected input-output relationships from SPICE data and reuse them for faster design-space exploration at the mature 180 nm node.\n\n"
            "Research question: Can learned models accelerate device and circuit characterization while preserving sufficient agreement with the simulator over the conditions tested?",
            size=9.2, color="ink")

        p = self.panel(left, 31.43, col_w, 5.35, "Literature Review")
        self.panel_text(p, .055, .84, .89, .55,
            "SPICE, introduced by Nagel and Pederson, established the numerical framework used by modern circuit simulation. Compact models such as BSIM capture short-channel transistor behavior and provide a practical device-level reference [1, 2].\n\n"
            "Simulation acceleration includes FastSPICE, model-order reduction, parallel computation and data-driven surrogates. Neural networks, Gaussian processes, support vector regression and ensembles offer different trade-offs.\n\n"
            "A key challenge is generalization: a good fit on sampled data does not guarantee physical behavior outside the training domain. Held-out testing and SPICE comparison are therefore important.\n\n"
            "Research gap: a practical workflow linking simulator-generated data, learned device or gate response, numerical circuit solving and system-level examples.",
            size=8.9)
        self.panel_image(p, 7, .18, .24, .64, .19)
        self.caption(p, .50, .22, "Machine-learning approaches reviewed in the thesis", .70, 7.2)

        p = self.panel(left, 25.99, col_w, 5.25, "Research Objectives & Contributions")
        self.panel_text(p, .065, .84, .87, .64,
            "• Develop ANN-based models for 180 nm NMOS/PMOS current and CMOS gate performance.\n"
            "• Represent operating conditions, device geometry and relevant process or temperature inputs.\n"
            "• Evaluate inverter, AND, OR, NAND, NOR and XNOR behavior against SPICE reference results.\n"
            "• Integrate learned behavior with numerical solving and optimization workflows.\n"
            "• Demonstrate SRAM, D flip-flop, ring oscillator, multiplexers and adders.\n\n"
            "Contribution: two complementary surrogate tracks—direct prediction of gate metrics and device-current models composed through current balance.",
            size=9.0, color="ink")

        p = self.panel(left, 18.90, col_w, 6.90, "CMOS Device Foundation")
        self.panel_image(p, 8, .14, .56, .72, .30)
        self.caption(p, .50, .535, "Four-terminal MOSFET structure (thesis Figure 5)", .85, 7.5)
        self.panel_text(p, .055, .48, .89, .40,
            "A MOSFET comprises gate, drain, source and body terminals. Gate-source and drain-source biases determine channel charge and drain current; geometry and temperature also affect its response.\n\n"
            "Current transitions from weak inversion below threshold to approximately linear behavior at small drain bias and saturation-like behavior at larger bias. Short-channel effects and process variation complicate simple square-law models.\n\n"
            "Small-signal gm = dID/dVGS and gds = dID/dVDS describe local sensitivity. Voltage-transfer curves, switching thresholds and noise margins connect device response to logic-level behavior.",
            size=8.8)

        p = self.panel(left, 12.41, col_w, 6.30, "Inputs, Targets & Evaluation")
        self.panel_text(p, .065, .84, .87, .43,
            "• Device inputs: VGS, VDS, geometry and operating conditions; target: drain current ID.\n"
            "• Circuit inputs include sizing, VDD, load, frequency and temperature.\n"
            "• Targets include VOH, VOL, tPLH, tPHL, power and leakage-related metrics.\n"
            "• Reported metrics: R-squared, MAE and RMSE; device-current networks use log-current space.\n"
            "• Circuit track reports an 80/20 split (85/15 for XNOR); fit preprocessing on training data.",
            size=8.8)
        self.panel_image(p, 12, .24, .035, .52, .23)
        self.caption(p, .50, .025, "Feedforward network: inputs to predicted outputs", .85, 7.2)

        p = self.panel(left, 7.12, col_w, 5.10, "Technology Context & Variability")
        self.panel_image(p, 4, .05, .50, .43, .31)
        self.panel_image(p, 5, .52, .50, .43, .31)
        self.caption(p, .265, .475, "180 nm in scaling history", .43, 7.0)
        self.caption(p, .735, .475, "Sources of device variation", .43, 7.0)
        self.panel_text(p, .055, .41, .89, .20,
            "Mature-node design remains important in mixed-signal and industrial applications. Variability sources include random dopant fluctuation, line-edge roughness, oxide-thickness variation and metal-gate granularity.",
            size=8.5)
        self.panel_text(p, .055, .205, .89, .15,
            "Process and temperature can shift threshold, switching point, delay and power. Predictions are meaningful only within conditions represented in reference data.",
            size=8.5, color="navy", bold=True)

        p = self.panel(left, 1.65, col_w, 5.28, "Model Training & Evaluation")
        self.panel_image(p, 13, .04, .42, .43, .32)
        self.panel_image(p, 20, .53, .42, .43, .32)
        self.caption(p, .255, .395, "Training loss", .43, 7.1)
        self.caption(p, .745, .395, "Validation loss", .43, 7.1)
        self.panel_text(p, .06, .32, .88, .27,
            "• R-squared measures explained variance; MAE and RMSE quantify error.\n"
            "• Device metrics use log-current; circuit targets retain their physical units.\n"
            "• A held-out test set checks generalization; scores vary across targets.",
            size=8.0)

        p = self.panel(middle, 36.02, col_w, 6.10, "Methods: Two Modeling Tracks")
        self.rect(p["x"] + .055*col_w, p["y"] + .17*p["h"], .89*col_w, .29*p["h"], "pale", line="line", rounded=True)
        self.panel_text(p, .08, .79, .84, .09, "PART I • DIRECT GATE-METRIC SURROGATES", size=9.4, color="blue", bold=True)
        self.panel_text(p, .08, .70, .84, .15,
            "LTspice gate sweeps provide labels; ANN models predict voltage, delay, power and leakage-related metrics for six logic-gate types.",
            size=8.7)
        self.rect(p["x"] + .055*col_w, p["y"] + .53*p["h"], .89*col_w, .29*p["h"], "pale", line="line", rounded=True)
        self.panel_text(p, .08, .43, .84, .09, "PART II • DEVICE-CENTRIC SURROGATES", size=9.4, color="teal", bold=True)
        self.panel_text(p, .08, .34, .84, .15,
            "Separate NMOS and PMOS networks learn drain current from BSIM-SPICE data; device predictions are composed using current balance and gate-scaling rules.",
            size=8.7)
        self.panel_text(p, .07, .09, .86, .08,
            "The tracks answer related but different questions and use different labels and evaluation scales.",
            size=8.5, color="navy", bold=True)

        p = self.panel(middle, 29.23, col_w, 6.60, "SPICE-to-ML Methodology")
        self.panel_image(p, 11, .06, .50, .88, .34)
        self.caption(p, .50, .475, "Thesis methodology: characterize, train, validate, optimize", .90, 7.5)
        steps = [
            ("1  Define design space", "Bias, device sizing, supply, load and relevant PVT conditions."),
            ("2  Generate reference data", "Run SPICE using the selected 180 nm compact-model setup."),
            ("3  Prepare datasets", "Filter implausible points; transform targets and scale inputs."),
            ("4  Train regressors", "Use supervised learning with validation and held-out testing."),
            ("5  Validate and apply", "Compare against reference; use surrogates in circuit solving or design search."),
        ]
        y = .42
        for title, detail in steps:
            self.panel_text(p, .075, y, .85, .045, title, size=8.8, color="navy", bold=True)
            self.panel_text(p, .12, y-.040, .81, .045, detail, size=8.1)
            y -= .074

        p = self.panel(middle, 23.04, col_w, 6.00, "Training & Validation")
        self.panel_image(p, 13, .03, .50, .45, .32)
        self.panel_image(p, 20, .52, .50, .45, .32)
        self.caption(p, .255, .475, "Training loss", .45, 7.3)
        self.caption(p, .745, .475, "Validation loss", .45, 7.3)
        self.panel_text(p, .07, .40, .87, .27,
            "• ANN regression uses mean-squared error and Adam.\n"
            "• Fit normalization statistics on training data only.\n"
            "• Test metrics are target-specific; R-squared does not describe absolute error.\n"
            "• Held-out points and SPICE references provide complementary checks.\n\n"
            "A low training loss alone is not evidence of generalization.",
            size=8.5)

        p = self.panel(middle, 14.85, col_w, 8.00, "Results 1 • Device-Level Characterization")
        for number, x, y, cap in [
            (25, .03, .70, "NMOS transfer curve"), (26, .52, .70, "PMOS transfer curve"),
            (27, .03, .43, "NMOS output characteristic"), (28, .52, .43, "PMOS output characteristic"),
            (29, .04, .105, "NMOS log-current"), (30, .53, .105, "PMOS log-current"),
        ]:
            self.panel_image(p, number, x, y, .43, .21 if y != .105 else .20)
            self.caption(p, x + .215, y - .02, cap, .43, 7.2)
        self.panel_text(p, .06, .355, .88, .08,
            "Learned current surfaces reproduce transfer and output behavior across bias conditions. Log-current plots expose weak inversion.",
            size=8.6)

        p = self.panel(middle, 1.65, col_w, 13.01, "Quantitative Results")
        rows = [
            ("Part II NMOS", "R2 0.9985", "RMSE(log) 0.0985"),
            ("Part II PMOS", "R2 0.9967", "RMSE(log) 0.1603"),
            ("Part I NMOS ID", "R2 0.9983", "MAE 4.494 × 10⁻⁶ A"),
            ("Part I inverter VOUT", "R2 0.9571", "MAE 49.9 mV"),
            ("Part I XNOR PDP", "R2 0.9981*", "Figure caption"),
        ]
        y = .84
        for index, (model, score, detail) in enumerate(rows):
            self.rect(p["x"] + .045*col_w, p["y"] + (1-y)*p["h"] - .065*p["h"], .91*col_w, .09*p["h"], "pale" if index % 2 == 0 else "paper", line="line", line_pt=.45)
            self.panel_text(p, .06, y, .39, .07, model, size=8.6, bold=index == 0)
            self.panel_text(p, .46, y, .25, .07, score, size=8.5, color="blue", bold=True)
            self.panel_text(p, .72, y, .23, .07, detail, size=7.6, color="muted")
            y -= .10
        self.panel_text(p, .06, y-.015, .88, .09,
            "*Reported in thesis Figure 19 caption. Values above come from different tracks and targets; do not compare as one ranking.",
            size=8.0, color="navy", bold=True)
        self.panel_text(p, .06, y-.12, .88, .09,
            "Part I gate results vary by metric. OR VOH R2 = 0.9964 but VOL = 0.9201; XNOR PDP R2 = 0.9983 but VOL = 0.9063.",
            size=7.9)
        self.panel_text(p, .06, .22, .88, .12,
            "BSIM-SPICE reference validation (Part II): the thesis reports correlation above 0.999 across approximately 10⁻¹²–10⁻³ A.",
            size=8.3, color="navy", bold=True)
        self.panel_image(p, 45, .05, .025, .43, .15)
        self.panel_image(p, 46, .52, .025, .43, .15)
        self.caption(p, .265, .012, "NMOS: ML vs BSIM-SPICE", .43, 7.1)
        self.caption(p, .735, .012, "PMOS: ML vs BSIM-SPICE", .43, 7.1)

        p = self.panel(right, 34.82, col_w, 7.30, "Results 2 • Gate-Level Characterization")
        self.panel_image(p, 31, .04, .54, .44, .34)
        self.panel_image(p, 32, .52, .54, .44, .34)
        self.caption(p, .26, .515, "Inverter VTC", .44, 7.4)
        self.caption(p, .74, .515, "Voltage gain", .44, 7.4)
        self.panel_text(p, .065, .455, .87, .11,
            "At VDD = 1.8 V: VOH = 1.800 V, VOL = 0 V, VIL = 0.703 V, VIH = 0.985 V and switching threshold VM = 0.855 V.",
            size=8.6)
        self.panel_text(p, .065, .35, .87, .12,
            "Noise margins are NML = 0.703 V and NMH = 0.815 V. Peak voltage-gain magnitude is 20.04. These values describe model-derived behavior at the stated conditions.",
            size=8.5)
        self.panel_text(p, .065, .205, .87, .10,
            "The VTC transition and its derivative explain inverter restoring behavior and noise tolerance.",
            size=8.3, color="navy", bold=True)

        p = self.panel(right, 27.83, col_w, 6.80, "Gate Delay, Power & Energy")
        self.panel_image(p, 34, .03, .50, .45, .34)
        self.panel_image(p, 35, .52, .50, .45, .34)
        self.caption(p, .255, .475, "Propagation delay (ps)", .45, 7.2)
        self.caption(p, .745, .475, "Dynamic power (µW)", .45, 7.2)
        self.panel_image(p, 36, .26, .19, .48, .24)
        self.caption(p, .50, .165, "Energy comparison (thesis Figure 29)", .62, 7.2)
        self.panel_text(p, .06, .10, .88, .08,
            "At the reported operating point: INV = 1.487 ps, 1.62 µW, 3.24 fJ; XOR = 23.796 ps, 6.48 µW, 12.96 fJ.",
            size=8.1, color="navy", bold=True)

        p = self.panel(right, 20.54, col_w, 7.10, "Results 3 • System-Level Examples")
        self.panel_image(p, 37, .03, .54, .45, .32)
        self.panel_image(p, 38, .52, .54, .45, .32)
        self.caption(p, .255, .515, "6T SRAM butterfly", .45, 7.2)
        self.caption(p, .745, .515, "D flip-flop timing", .45, 7.2)
        self.panel_text(p, .06, .445, .88, .09,
            "SRAM: the butterfly curve illustrates bistability; reported static noise margin is approximately 54.3 mV for the stated model configuration.",
            size=8.1)
        self.panel_image(p, 39, .03, .16, .45, .22)
        self.panel_image(p, 40, .52, .16, .45, .22)
        self.caption(p, .255, .14, "Five-stage ring-oscillator output", .45, 7.0)
        self.caption(p, .745, .14, "Frequency vs stage count", .45, 7.0)
        self.panel_text(p, .06, .08, .88, .06,
            "Ring-oscillator estimate: N = 5, stage delay = 1.487 ps, frequency = 67.24 GHz.",
            size=7.9, color="navy", bold=True)

        p = self.panel(right, 13.85, col_w, 6.50, "Additional Circuits & Validation")
        self.panel_image(p, 41, .035, .58, .44, .25)
        self.panel_image(p, 42, .52, .58, .44, .25)
        self.caption(p, .255, .555, "MUX family (2:1–16:1)", .44, 7.0)
        self.caption(p, .74, .555, "Full-adder outputs", .44, 7.0)
        self.panel_text(p, .06, .48, .88, .10,
            "The device-centric framework is also applied to half adders and full adders, extending examples to arithmetic behavior.",
            size=8.0)
        self.panel_text(p, .06, .365, .88, .12,
            "Robustness checks include 20 Monte Carlo runs with ±5% variation (reported VM spread about ±0.09 V) and qualitative 2.2 V extrapolation beyond 1.8 V training.",
            size=7.9)
        self.panel_text(p, .06, .22, .88, .12,
            "Thesis reports >0.999 correlation against BSIM-SPICE across approximately 10⁻¹²–10⁻³ A and mean relative current-balance error of 0.065%.",
            size=7.9, color="navy", bold=True)
        self.panel_text(p, .06, .09, .88, .08,
            "Monte Carlo and extrapolation are exploratory—not yield certification or guaranteed out-of-domain accuracy.",
            size=7.6, color="muted")

        p = self.panel(right, 8.06, col_w, 5.60, "Discussion & Conclusion")
        self.panel_text(p, .055, .84, .89, .72,
            "The thesis demonstrates that neural surrogates approximate selected SPICE-generated device and gate responses and can be composed into representative circuit analyses. The reported Part I benchmark shows 90×–110× speed-up for its measurement setup.\n\n"
            "Accuracy is target-dependent. Strong R-squared scores for device current or PDP do not imply equal accuracy for every voltage, timing or power metric. Part I and Part II use different targets and scales.\n\n"
            "Limitations include one 180 nm process focus, finite training coverage, model-based rather than measured-silicon results, limited Monte Carlo sampling and qualitative extrapolation.\n\n"
            "Conclusion: ML models complement early design-space exploration. Final sign-off requires validated PDK/SPICE analysis and, where appropriate, silicon measurement.\n\n"
            "Future work: physics-informed training, broader process data, advanced nodes, graph-based full-chip models and integration with EDA flows.",
            size=8.4)

        p = self.panel(right, 1.65, col_w, 6.22, "References")
        refs = [
            "[1] L. W. Nagel and D. O. Pederson, “SPICE,” University of California, Berkeley, Memo UCB/ERL M382, 1973.",
            "[2] W. Liu et al., BSIM3v3.3 MOSFET Model User’s Manual, University of California, Berkeley, 2005.",
            "[3] Y. Taur and T. H. Ning, Fundamentals of Modern VLSI Devices, 2nd ed., Cambridge University Press, 2009.",
            "[4] G. Gielen and R. Rutenbar, “Computer-aided design of analog and mixed-signal integrated circuits,” Proc. IEEE, vol. 88, no. 12, pp. 1825–1854, 2000.",
            "[5] A. Asenov, “Random dopant induced threshold voltage lowering and fluctuations in sub-0.1 µm MOSFETs,” IEEE Trans. Electron Devices, vol. 45, no. 12, pp. 2505–2513, 1998.",
            "[6] R. Nain, “Machine Learning-Assisted Electrical Characterization of 180 nm CMOS Digital Circuits,” M.Tech thesis, Thapar Institute of Engineering and Technology, 2026.",
        ]
        self.panel_text(p, .055, .87, .89, .72, "\n\n".join(refs), size=7.4)
        self.panel_text(p, .055, .07, .89, .06,
            "Numerical values and project figures are thesis-derived; simulator/model results are not measured silicon data.",
            size=7.5, color="navy", bold=True)

    def add_footer(self) -> None:
        self.rect(.012, A0_H - .008 - 1.15, .976*A0_W, 1.15, "cyan", line="cyan")
        self.text(.66, A0_H - 1.08, 4.4, .38, "TAKE-HOME MESSAGE", size=9, color="white", bold=True)
        self.text(
            5.9, A0_H - 1.08, 26.0, .38,
            "SPICE-trained surrogates enable faster 180 nm design exploration while preserving a reference-based validation path.",
            size=10, color="white", bold=True,
        )
        self.text(
            .66, A0_H - .56, 31.0, .30,
            "Poster prepared from the M.Tech thesis. All performance figures are simulator/model-derived unless stated otherwise.",
            size=8, color="white",
        )

    def save(self, path: Path) -> None:
        self.add_header()
        self.build_panels()
        self.add_footer()
        self.presentation.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    media = load_media()
    targets = [
        (A0_W, A0_H, OUT_DIR / "CMOS_ML_180nm_Project_Poster_Editable_A0.pptx"),
        (A1_W, A1_H, OUT_DIR / "CMOS_ML_180nm_Project_Poster_Editable_A1.pptx"),
    ]
    for width, height, path in targets:
        EditablePoster(width, height, media).save(path)
        print(f"Created {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
