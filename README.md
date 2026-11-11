# AI-Assisted Electrical Characterization of CMOS Digital Circuits

### Using Machine Learning (PyTorch + Scikit-Learn)

**Technology: 180 nm CMOS | VDD = 1.8 V**

---

## Project Structure

```
cmos_ai_eda/
├── nmos.csv                  ← NMOS SPICE dataset
├── pmos.csv                  ← PMOS SPICE dataset
├── spice/                    ← SKY130 reference-data generator + schema
│   ├── generate_sky130_dataset.py
│   └── analyze_reference.py   ← Extract Ion/Ioff, gm/gds, VTC/noise-margin metrics
├── data/sky130_reference/    ← Generated, provenance-tracked SKY130 results
├── requirements.txt
│
├── preprocess.py             ← Module 1:  Data loading, cleaning, scaling
├── models.py                 ← Module 2/3: PyTorch TransistorNet definition
├── train_nmos.py             ← Module 2:  NMOS training (5000 epochs)
├── train_pmos.py             ← Module 3:  PMOS training (5000 epochs)
├── current_predictor.py      ← Module 4:  Inference wrapper functions
├── cmos_solver.py            ← Module 5:  Brent VTC solver + parameter extraction
├── inverter.py               ← Module 5b: Inverter plotting
├── gates.py                  ← Module 6:  INV/NAND/NOR/AND/OR/XOR
├── delay_power.py            ← Module 7:  tp, Power, EDP
├── corner_analysis.py        ← Module 8/9: Temperature + PVT corners
├── monte_carlo.py            ← Module 10: 1000-run Monte Carlo
├── ring_oscillator.py        ← Module 11: Ring oscillator f = 1/(2Ntp)
├── dff.py                    ← Module 12: DFF setup/hold/clk-Q timing
├── sram.py                   ← Module 13: 6T SRAM butterfly + SNM
├── visualization.py          ← Module 15: IV characteristic plots
├── report_generator.py       ← Module 17: Auto PDF report (fpdf2)
├── streamlit_app.py          ← Module 16: Streamlit GUI with ordered circuit navigation
├── main.py                   ← Module 18: Full pipeline orchestrator
│
├── models/                   ← Saved .pth weights + metrics.json
├── scalers/                  ← Saved StandardScaler .pkl files
├── plots/                    ← All generated PNG figures (dpi=300)
└── reports/                  ← Generated PDF report
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSV files in the project root
#    nmos.csv, pmos.csv  ← already there
```

---

## Usage

### Full Pipeline (recommended first run)

```bash
python main.py
```

This runs all 14 steps sequentially (~20-40 min with 1000 MC runs).

### Skip Training (use saved models)

```bash
python main.py --skip-train
```

### Skip Monte Carlo (faster run)

```bash
python main.py --skip-mc
```

### Fewer Monte Carlo runs (for testing)

```bash
python main.py --mc-runs 100
```

### Launch GUI

```bash
streamlit run streamlit_app.py
```

### Project flow, expanded report, and posters

The project flow is:

1. Load and clean the NMOS/PMOS device datasets; convert units and polarity,
   log-transform drain current, then split and scale the data.
2. Train the device-current regressors or load the saved model weights and
   scalers.
3. Predict device currents and solve inverter/gate transfer curves using
   current balance.
4. Run the configured timing, power, temperature, process-corner, Monte Carlo,
   ring-oscillator, DFF, SRAM, and Boolean analyses.
5. Review plots and metrics in the Streamlit app and generate the report.

The separate SKY130 reference-data flow generates and analyzes ngspice sweeps
under `data/sky130_reference/`; it does not replace the original 180 nm inputs.

Generate the expanded, exactly 80-page report from the thesis and project
source:

```powershell
python .\create_80_page_report.py
python .\create_report_docx.py
```

The report is A4 with Times New Roman 12 pt, 1-inch margins, 1.0 line spacing,
and IEEE-style numeric citations. The PDF and editable Word outputs are
`reports/CMOS_ML_180nm_80_Page_Project_Report.pdf` and
`reports/CMOS_ML_180nm_80_Page_Project_Report.docx`.

Generate the standardized academic posters in both portrait sizes:

```powershell
python .\create_detailed_style_poster.py
```

The A0/A1 PDFs and preview PNGs are written to `final/` with the
`CMOS_ML_180nm_Project_Poster_Standardized_` filename prefix.

Generate editable PowerPoint poster files (text and layout are editable;
plots/illustrations are embedded image objects):

```powershell
python .\create_editable_poster.py
```

Editable A0 and A1 files are written to `final/` with the
`CMOS_ML_180nm_Project_Poster_Editable_` filename prefix.

### Grouped validation and model benchmarks

Run reproducible grouped holdouts using the existing CSVs and saved SKY130
reference data:

```powershell
python .\validation_experiments.py
```

By default, the experiment holds out 85 °C and the 1.0 µm / 0.25 µm device
geometry for both NMOS and PMOS. It also leaves out each available SKY130
process corner in turn, compares a square-law baseline, ridge regression,
ExtraTrees, histogram gradient boosting, and an MLP, and produces error
metrics in log10-current and amperes. It saves fold metrics, per-sample
predictions, uncertainty-proxy bins, transient/power cross-checks, source
hashes, and run configuration in a UTC-named directory under
`reports/validation_experiments/`.

Use `--all-holdouts` to evaluate every 180 nm temperature and geometry group.
Use `--max-train-samples 0 --max-test-samples 0` to use every row in each fold.
The default sample caps reduce run time and are recorded in the manifest.

The uncertainty output is a model-disagreement heuristic, not a calibrated
confidence interval. SKY130 corner holdouts are exploratory because corner
labels are not model inputs and the existing reference data covers only
27 °C, 1.8 V, and 0.42 µm / 0.15 µm device geometry. Timing/power checks use
the existing single load/slew transient condition; they do not replace fresh
SPICE sweeps or sign-off validation.

### All-Circuits Characterization

The **All-circuits overview** runs the available model-backed analyses together and
provides a CSV export of the logic-gate characterization table. It reports
device IV curves, finite-difference `gm`/`gds` at stated bias points, gate DC
VTC parameters (`VOH`, `VOL`, `VM`, `VIL`, `VIH`, gain, `NMH`, `NML`), and
first-order delay, switched-capacitance, dynamic-power, energy, and EDP
estimates. It also includes process/temperature sweeps, Monte Carlo switching
voltage, ring-oscillator estimates, DFF timing estimates, SRAM stability and
access-time estimates, and Boolean-expression results.

VTCs use a higher-resolution DC solve (401 VIN samples by default), with
shape-preserving interpolation used only to render the sampled curves;
reported metrics are computed from the model samples. The Streamlit inverter
and logic-gate pages let you adjust transistor widths, lengths, temperature,
and VTC resolution. Timing/power, Monte Carlo, ring oscillator, and SRAM pages
also expose their relevant load, frequency/activity, variation, stage-count,
and capacitance/sense-voltage assumptions. The all-circuits overview provides
shared geometry, resolution, switching-activity, ring-stage, SRAM read, and
Monte Carlo controls for a single run.

The current datasets contain DC terminal voltages, geometry, temperature, and
drain current. They do not provide extracted terminal capacitances, validated
leakage/breakdown/noise data, or transient waveforms. Consequently, those
current datasets contain DC terminal voltages, geometry, temperature, and
drain current. They do not provide extracted terminal capacitances, validated
leakage/breakdown/noise data, or transient waveforms. Consequently, those
parameters and sign-off-grade rise/fall delays require a foundry PDK/SPICE
deck and dedicated DC/AC/transient analyses; they are not fabricated by the
dashboard.

### SKY130 ngspice reference data

The original `nmos.csv`, `pmos.csv`, trained models, and existing circuit
analyses are kept separate and remain labeled as the project's current
180 nm work. The SKY130 flow writes new files under
`data/sky130_reference/`; it never overwrites those existing inputs.

The generator uses the public SKY130A ngspice model deck and creates:

- `device_dc.csv`: NMOS/PMOS DC sweeps across TT/SS/FF/FS/SF, temperature,
  geometry, and the requested body-bias values.
- `inverter_dc.csv`: inverter transfer curves and static supply power across
  process, temperature, and supply voltage.
- `inverter_transient.csv`: `tpHL`, `tpLH`, rise/fall time, average power, and
  energy per input period for the recorded load and input slew.
- `device_dc_metrics.csv`: `ION`, `IOFF`, finite-difference `gm` and `gds` per
  device, geometry, and process/temperature condition.
- `inverter_dc_metrics.csv`: `VOH`, `VOL`, `VM`, `VIL`, `VIH`, gain, `NMH`,
  and `NML` per process/temperature/supply condition.
- `manifest.json` and saved netlists: model-deck SHA-256, PDK revision,
  simulator version, sweep conditions, and row counts.

The model deck and ngspice executable are external prerequisites. Install the
public [SkyWater SKY130 PDK](https://github.com/google/skywater-pdk) with
[Ciel](https://github.com/fossi-foundation/ciel) or
[open_pdks](https://github.com/fossi-foundation/open-pdks), then install
ngspice.
Then run from the project directory (or pass the absolute model-file path):

```powershell
$env:PDK_ROOT = "C:\path\to\pdk-root"
$env:SKY130_PDK_REVISION = "<exact-open-pdks-or-ciel-revision>"
python .\spice\generate_sky130_dataset.py --corners tt --temperatures 27 --vdds 1.8 --widths-um 0.42 --lengths-um 0.15
```

For ngspice available only inside WSL, run the script from WSL and set
`PDK_ROOT` to the Linux-visible PDK root:

```bash
export PDK_ROOT="$HOME/.ciel"
export SKY130_PDK_REVISION="<exact-open-pdks-or-ciel-revision>"
export NGSPICE_BIN="$HOME/.local/opt/ngspice/root/usr/bin/ngspice"
export LD_LIBRARY_PATH="$HOME/.local/opt/ngspice/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "/mnt/c/Users/iitdr/OneDrive - randeepnain/Desktop/fnal"
python3 spice/generate_sky130_dataset.py
```

The generator automatically creates both derived DC metric files. To
recalculate them after manually editing/importing CSV data, run
`python3 spice/analyze_reference.py`.

Use `python spice/generate_sky130_dataset.py --help` to select smaller smoke
tests or expand the geometry, body-bias, and PVT sweeps. The PDK revision is
required so generated rows identify the exact PDK build. Generated rows are
SPICE-model references, not measured silicon data. The generator does not
invent capacitance, noise, leakage, mismatch, or silicon-validation data; those
require dedicated analyses and/or measurement sources.

---

## Module Summary

| Module | File                   | Description                                        |
| ------ | ---------------------- | -------------------------------------------------- |
| 1      | `preprocess.py`        | Load, clean, log-transform, normalize, split       |
| 2      | `train_nmos.py`        | Train NMOS net: Adam, LR=0.0005, 5000 epochs       |
| 3      | `train_pmos.py`        | Train PMOS net (same architecture)                 |
| 4      | `current_predictor.py` | `predict_nmos_current()`, `predict_pmos_current()` |
| 5      | `cmos_solver.py`       | Brentq VTC solver, VOH/VOL/VM/VIL/VIH/Gain/NM      |
| 5b     | `inverter.py`          | Inverter VTC + gain curve plots                    |
| 6      | `gates.py`             | INV/NAND/NOR/AND/OR/XOR VTC + gain bar chart       |
| 7      | `delay_power.py`       | tp = CL·VDD/ION, P = α·CL·VDD²·f, EDP              |
| 8      | `corner_analysis.py`   | VTC at −40/27/75/125°C                             |
| 9      | `corner_analysis.py`   | PVT corners: TT/SS/FF/FS/SF                        |
| 10     | `monte_carlo.py`       | 1000 runs, VM histogram, VTC spread                |
| 11     | `ring_oscillator.py`   | f = 1/(2Ntp) vs N, FO4 delay                       |
| 12     | `dff.py`               | Setup/hold/clk-Q from inverter delay               |
| 13     | `sram.py`              | 6T SRAM butterfly curve + SNM                      |
| 14     | `delay_power.py`       | Performance table (all gates)                      |
| 15     | `visualization.py`     | IDS-VGS, IDS-VDS, log plots (dpi=300)              |
| 16     | `streamlit_app.py`     | 13-page interactive GUI, including SKY130 data      |
| 16b    | `spice/generate_sky130_dataset.py` | SKY130 device/inverter SPICE dataset generator |
| 16c    | `spice/analyze_reference.py` | Extracted device and inverter metrics from SPICE sweeps |
| 17     | `report_generator.py`  | Auto PDF with metrics + all plots                  |
| 18     | `main.py`              | Orchestrates complete pipeline                     |

---

## Neural Network Architecture

```
Input(5) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(32) → Tanh → Linear(1)

NMOS inputs: [VGS, VDS, Temperature, W, L]
PMOS inputs: [VSG, VSD, Temperature, W, L]
Output:      log10(IDS)  [normalized via StandardScaler]

Optimizer:   Adam  |  LR = 0.0005  |  Loss = MSE  |  Epochs = 5000
```

---

## Key Physics Equations

```
VTC balance:    IDN(VIN, VOUT) = IDP(VIN, VOUT)   [Brent's method]
Delay:          tp = CL × VDD / ION
Dynamic Power:  P  = α × CL × VDD² × f
EDP:            EDP = (α × CL × VDD²) × tp
Ring Osc:       f  = 1 / (2 × N × tp)
DFF timing:     tsetup = 2tp, thold = tp, tclk-Q = 4tp
```
