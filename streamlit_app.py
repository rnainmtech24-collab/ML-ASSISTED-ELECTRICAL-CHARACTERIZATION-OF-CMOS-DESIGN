"""
Module 16 — Streamlit EDA Tool GUI  (streamlit_app.py)
=======================================================
Run with:  streamlit run streamlit_app.py

Navigation pages (shown in order):
  1  NMOS Characterization
  2  PMOS Characterization
  3  Inverter
  4  Logic Gates
  5  Delay and Power
  6  Corner Analysis
  7  Monte Carlo
  8  Ring Oscillator
  9  SRAM
  10 Boolean Characterizer
  11 Report Generator
  12 All Circuits — run every analysis in one overview
  13  SKY130 — ngspice reference datasets
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMOS AI EDA Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.edgecolor": "#C7D8D6",
    "axes.linewidth": 0.9,
    "axes.facecolor": "#FBFDFC",
    "figure.facecolor": "#FBFDFC",
    "grid.color": "#DDE9E7",
    "grid.alpha": 0.85,
    "text.color": "#18343A",
    "axes.labelcolor": "#38565B",
    "xtick.color": "#526D70",
    "ytick.color": "#526D70",
    "axes.prop_cycle": plt.cycler(color=[
        "#087E8B", "#F28C45", "#6657C8", "#D14D72",
        "#4B9B72", "#4385C6", "#B88724",
    ]),
    "lines.linewidth": 2.4,
    "lines.antialiased": True,
    "savefig.facecolor": "#FBFDFC",
})

st.markdown(
    """
    <style>
    :root {
        --ink: #17343A;
        --muted: #60777A;
        --teal: #087E8B;
        --teal-dark: #075D67;
        --mint: #E8F4F1;
        --gold: #E4A84B;
        --line: #DCE8E5;
        --surface: #FFFFFF;
        --page: #F3F7F5;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 92% 3%, rgba(8,126,139,.08), transparent 28rem),
            var(--page);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(243,247,245,.94);
        backdrop-filter: blur(12px);
    }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1540px;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h2 { padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
    p, label { color: #405C60; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102D35 0%, #173D44 58%, #174A4D 100%);
        border-right: 1px solid rgba(255,255,255,.09);
    }
    [data-testid="stSidebar"] * { color: #E9F4F1; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #C0D9D4;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFFFF, #F8FCFA);
        border: 1px solid var(--line);
        border-left: 3px solid var(--teal);
        padding: .95rem 1.05rem; border-radius: 12px;
        box-shadow: 0 5px 16px rgba(21,62,63,.055);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: .3rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .58rem .7rem;
        margin: 0;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #D8E9E5;
        transition: background .16s ease, border-color .16s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(222,248,237,.09);
        border-color: rgba(222,248,237,.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(100deg, rgba(91,196,170,.22), rgba(91,196,170,.08));
        border-color: rgba(130,219,193,.34);
        color: #FFFFFF;
        box-shadow: inset 3px 0 #72D4B4;
    }
    div.stButton > button[kind="primary"] {
        min-height: 2.8rem; border: 0; border-radius: 10px;
        color: #FFFFFF; font-weight: 750;
        background: linear-gradient(110deg, #087E8B, #075D67);
        box-shadow: 0 5px 14px rgba(8,126,139,.22);
        transition: transform .16s ease, box-shadow .16s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        color: #FFFFFF; transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(8,126,139,.30);
    }
    div.stButton > button:not([kind="primary"]),
    [data-testid="stDownloadButton"] button {
        border-radius: 9px; border-color: #BCD4D0; color: var(--teal-dark);
        background: #FFFFFF; font-weight: 600;
    }
    div.stButton > button:not([kind="primary"]):hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--teal); color: var(--teal-dark);
        background: var(--mint);
    }
    input, textarea, [data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        background: var(--teal); border-color: var(--teal);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line); border-radius: 11px;
        box-shadow: 0 4px 12px rgba(21,62,63,.035);
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.86);
        border: 1px solid var(--line); border-radius: 11px;
    }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stProgressBar"] > div > div > div > div {
        background: linear-gradient(90deg, var(--teal), #55B7A2);
    }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    .parameter-icon {
        width: 2.85rem; height: 2.85rem; margin: .55rem auto 0;
        display: flex; align-items: center; justify-content: center;
        border-radius: 50%;
        border: 1px solid rgba(8,126,139,.18);
        background: radial-gradient(circle at 34% 28%, #FFFFFF 0%, #DFF2EC 58%, #C8E6DF 100%);
        box-shadow: 0 4px 11px rgba(8,92,98,.13), inset 0 1px 2px #FFFFFF;
        font-size: 1.25rem;
    }
    .parameter-list-title {
        color: #17343A; font-weight: 750; font-size: 1.1rem;
        letter-spacing: -.01em; margin: .1rem 0 .2rem;
    }
    .parameter-list-caption { color: #60777A; font-size: .88rem; }
    @media (max-width: 720px) {
        .block-container { padding: .8rem .75rem 2rem; }
        .parameter-icon { width: 2.35rem; height: 2.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="position:relative; overflow:hidden; padding:1.55rem 1.8rem;
                margin-bottom:1rem; border-radius:18px; color:#F4FBF9;
                border:1px solid rgba(255,255,255,.13);
                background:linear-gradient(112deg,#102D35 0%,#13535A 54%,#087E8B 100%);
                box-shadow:0 12px 30px rgba(16,65,68,.17);">
      <div style="position:absolute; width:240px; height:240px; border-radius:50%;
                  right:8%; top:-155px; border:1px solid rgba(210,247,232,.23);
                  box-shadow:0 0 0 28px rgba(210,247,232,.035),
                             0 0 0 58px rgba(210,247,232,.025);"></div>
      <div style="position:relative; z-index:1;">
        <div style="display:inline-block; padding:.3rem .6rem; border-radius:999px;
                    background:rgba(222,248,237,.12); border:1px solid rgba(222,248,237,.2);
                    font-size:.72rem; letter-spacing:.15em; color:#D6F1E6; font-weight:750;">
          RESEARCH CHARACTERIZATION WORKBENCH
        </div>
        <div style="font-size:2.15rem; line-height:1.18; font-weight:780; margin-top:.7rem;">
          CMOS Circuit Analysis
        </div>
        <div style="font-size:.98rem; color:#D0E8E3; margin-top:.4rem;">
          ML-backed device models <span style="color:#E4A84B;">·</span>
          180 nm CMOS <span style="color:#E4A84B;">·</span> VDD = 1.8 V
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── lazy imports (avoid loading torch until needed) ─────────────────────────
@st.cache_resource
def load_predictor():
    from current_predictor import predict_nmos_current, predict_pmos_current
    return predict_nmos_current, predict_pmos_current

@st.cache_data
def cached_vtc(T, Wn, Ln, Wp, Lp):
    from cmos_solver import solve_vtc, extract_vtc_params
    vin, vout = solve_vtc(T=T, Wn=Wn, Ln=Ln, Wp=Wp, Lp=Lp)
    params    = extract_vtc_params(vin, vout)
    return vin, vout, params


@st.cache_data
def cached_vtc_with_resolution(T, Wn, Ln, Wp, Lp, n_points):
    from cmos_solver import solve_vtc, extract_vtc_params
    vin, vout = solve_vtc(
        T=T, Wn=Wn, Ln=Ln, Wp=Wp, Lp=Lp, n_points=n_points
    )
    params = extract_vtc_params(vin, vout)
    return vin, vout, params


def plot_smooth_curve(ax, x, y, points=400, **kwargs):
    """Render a shape-preserving smooth line without changing model data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return ax.plot(x, y, antialiased=True, solid_capstyle="round", **kwargs)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    if len(x_unique) != len(x_sorted):
        return ax.plot(x, y, antialiased=True, solid_capstyle="round", **kwargs)
    y_unique = y_sorted[unique_indices]
    if len(x_unique) < 3:
        return ax.plot(x, y, antialiased=True, solid_capstyle="round", **kwargs)

    from scipy.interpolate import PchipInterpolator
    points = max(int(points), len(x_unique))
    x_dense = np.linspace(x_unique[0], x_unique[-1], points)
    y_dense = PchipInterpolator(x_unique, y_unique)(x_dense)
    return ax.plot(
        x_dense, y_dense, antialiased=True, solid_capstyle="round", **kwargs
    )


VDD = 1.8

# ── ordered circuit navigation ──────────────────────────────────────────────
NAV_PAGES = [
    "🔵  NMOS transistor",
    "🔴  PMOS transistor",
    "⚡  CMOS inverter",
    "🔗  Logic gates",
    "⏱️  Delay & power",
    "🌡️  Process & temperature corners",
    "🎲  Monte Carlo variation",
    "🔄  Ring oscillator",
    "💾  6T SRAM",
    "🔣  Boolean expression",
    "📄  PDF report",
    "📊  All-circuits overview",
    "🧪  SKY130 SPICE reference data",
]

st.sidebar.markdown(
    """
    <div style="padding:.45rem .25rem .75rem">
      <div style="font-size:.72rem; letter-spacing:.13em; color:#A8CDC4;
                  font-weight:750;">CMOS LAB</div>
      <div style="font-size:1.28rem; font-weight:760; color:#F1FAF7;
                  margin-top:.25rem;">Circuit explorer</div>
      <div style="font-size:.82rem; color:#BBD4CD; margin-top:.25rem;">
        Select an analysis to inspect
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_page = st.sidebar.radio(
    "Circuit analysis",
    NAV_PAGES,
    index=0,
    label_visibility="collapsed",
    key="circuit_navigation",
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — NMOS
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[0]:
    st.header("NMOS Transistor Characterization")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        vgs  = st.slider("VGS (V)",  0.0, VDD, 1.2, 0.05)
        vds  = st.slider("VDS (V)",  0.0, VDD, 1.8, 0.05)
        temp = st.slider("Temperature (°C)", -40, 125, 27, 5)
        W_n  = st.number_input("Width W (µm)", 0.1, 10.0, 0.5, 0.1) * 1e-6
        L_n  = st.number_input("Length L (nm)", 90, 1000, 180, 10) * 1e-9

        if st.button("Predict IDS", key="nmos_pred"):
            try:
                pn, _ = load_predictor()
                ids = pn(VGS=vgs, VDS=vds, Temperature=temp, W=W_n, L=L_n)
                st.success(f"**IDS = {ids:.4e} A  ({ids*1e6:.4f} µA)**")
            except Exception as e:
                st.error(f"Model not loaded: {e}")

    with col2:
        st.subheader("IDS vs VGS Sweep")
        vds_fixed = st.slider("Fixed VDS for sweep (V)", 0.05, VDD, 1.8, 0.05, key="nmos_vds_sw")
        if st.button("Generate IV Curve", key="nmos_iv"):
            try:
                pn, _ = load_predictor()
                vgs_arr = np.linspace(0, VDD, 181)
                ids_arr = [pn(v, vds_fixed, temp, W_n, L_n) for v in vgs_arr]
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
                plot_smooth_curve(axes[0], vgs_arr, np.array(ids_arr)*1e6,
                                  color="blue")
                axes[0].set_xlabel("VGS (V)"); axes[0].set_ylabel("IDS (µA)")
                axes[0].set_title("Linear"); axes[0].grid(True, alpha=0.3)
                log_ids = [np.log10(max(i, 1e-15)) for i in ids_arr]
                plot_smooth_curve(axes[1], vgs_arr, log_ids, color="red")
                axes[1].set_xlabel("VGS (V)"); axes[1].set_ylabel("log₁₀(IDS)")
                axes[1].set_title("Log Scale"); axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PMOS
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[1]:
    st.header("PMOS Transistor Characterization")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        vsg  = st.slider("VSG (V)",  0.0, VDD, 1.2, 0.05)
        vsd  = st.slider("VSD (V)",  0.0, VDD, 1.8, 0.05)
        temp_p = st.slider("Temperature (°C)", -40, 125, 27, 5, key="ptemp")
        W_p  = st.number_input("Width W (µm)", 0.1, 10.0, 1.0, 0.1, key="wp") * 1e-6
        L_p  = st.number_input("Length L (nm)", 90, 1000, 180, 10, key="lp") * 1e-9

        if st.button("Predict IDS", key="pmos_pred"):
            try:
                _, pp = load_predictor()
                ids = pp(VSG=vsg, VSD=vsd, Temperature=temp_p, W=W_p, L=L_p)
                st.success(f"**|IDS| = {ids:.4e} A  ({ids*1e6:.4f} µA)**")
            except Exception as e:
                st.error(f"Model not loaded: {e}")

    with col2:
        st.subheader("|IDS| vs VSG Sweep")
        vsd_fixed = st.slider("Fixed VSD for sweep (V)", 0.05, VDD, 1.8, 0.05, key="p_vsd_sw")
        if st.button("Generate IV Curve", key="pmos_iv"):
            try:
                _, pp = load_predictor()
                vsg_arr = np.linspace(0, VDD, 181)
                ids_arr = [pp(v, vsd_fixed, temp_p, W_p, L_p) for v in vsg_arr]
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
                plot_smooth_curve(axes[0], vsg_arr, np.array(ids_arr)*1e6,
                                  color="red")
                axes[0].set_xlabel("VSG (V)"); axes[0].set_ylabel("|IDS| (µA)")
                axes[0].set_title("Linear"); axes[0].grid(True, alpha=0.3)
                log_ids = [np.log10(max(i, 1e-15)) for i in ids_arr]
                plot_smooth_curve(axes[1], vsg_arr, log_ids, color="blue")
                axes[1].set_xlabel("VSG (V)"); axes[1].set_ylabel("log₁₀|IDS|")
                axes[1].set_title("Log Scale"); axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — INVERTER
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[2]:
    st.header("CMOS Inverter VTC")
    col1, col2 = st.columns([1, 2])
    with col1:
        temp_inv = st.slider("Temperature (°C)", -40, 125, 27, 5, key="inv_t")
        Wn_inv   = st.number_input("NMOS W (µm)", 0.1, 5.0, 0.5, 0.1, key="wn_inv") * 1e-6
        Wp_inv   = st.number_input("PMOS W (µm)", 0.1, 5.0, 1.0, 0.1, key="wp_inv") * 1e-6
        Ln_inv   = st.number_input("NMOS L (nm)", 90, 1000, 180, 10, key="ln_inv") * 1e-9
        Lp_inv   = st.number_input("PMOS L (nm)", 90, 1000, 180, 10, key="lp_inv") * 1e-9
        n_inv    = st.slider("VTC samples", 101, 1001, 501, 100, key="n_inv")
        run_inv  = st.button("Solve VTC", key="inv_run")

    with col2:
        if run_inv:
            with st.spinner("Solving VTC …"):
                try:
                    vin, vout, params = cached_vtc_with_resolution(
                        temp_inv, Wn_inv, Ln_inv, Wp_inv, Lp_inv, n_inv
                    )
                    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
                    plot_smooth_curve(ax, vin, vout, color="blue",
                                      lw=2.5, label="VTC")
                    ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.5)
                    ax.axvline(params["VM"],  color="green",  ls=":", lw=1.2)
                    ax.axvline(params["VIL"], color="orange", ls=":", lw=1.2)
                    ax.axvline(params["VIH"], color="red",    ls=":", lw=1.2)
                    ax.set_xlim(0, VDD); ax.set_ylim(-0.05, VDD+0.05)
                    ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
                    ax.set_title(f"CMOS Inverter VTC  (T={temp_inv}°C)")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig); plt.close()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("VM",   f"{params['VM']:.3f} V")
                    c2.metric("Gain", f"{params['Gain']:.1f}")
                    c3.metric("NMH",  f"{params['NMH']:.3f} V")
                    c4.metric("NML",  f"{params['NML']:.3f} V")
                    st.json(params)
                except Exception as e:
                    st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — LOGIC GATES
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[3]:
    st.header("Logic Gate Characterization")
    gate_sel = st.selectbox("Select Gate", ["INV", "NAND", "NOR", "AND", "OR", "XOR"])
    temp_gate = st.slider("Temperature (°C)", -40, 125, 27, 5, key="gate_t")

    gate_col1, gate_col2 = st.columns(2)
    with gate_col1:
        gate_wn_base = st.number_input(
            "Base NMOS W (µm)", 0.1, 5.0, 0.5, 0.1, key="gate_wn"
        ) * 1e-6
        gate_ln = st.number_input(
            "NMOS L (nm)", 90, 1000, 180, 10, key="gate_ln"
        ) * 1e-9
    with gate_col2:
        gate_wp_base = st.number_input(
            "Base PMOS W (µm)", 0.1, 5.0, 1.0, 0.1, key="gate_wp"
        ) * 1e-6
        gate_lp = st.number_input(
            "PMOS L (nm)", 90, 1000, 180, 10, key="gate_lp"
        ) * 1e-9
    gate_points = st.slider("VTC samples", 101, 1001, 501, 100, key="gate_points")
    gate_wn = {"INV": gate_wn_base, "NAND": gate_wn_base / 2,
               "NOR": 2 * gate_wn_base, "AND": gate_wn_base / 2,
               "OR": 2 * gate_wn_base, "XOR": 2 * gate_wn_base}
    gate_wp = {"INV": gate_wp_base, "NAND": 2 * gate_wp_base,
               "NOR": gate_wp_base / 2, "AND": 2 * gate_wp_base,
               "OR": gate_wp_base / 2, "XOR": 2 * gate_wp_base}

    if st.button("Solve Gate VTC", key="gate_run"):
        with st.spinner(f"Solving {gate_sel} VTC …"):
            try:
                vin, vout, params = cached_vtc_with_resolution(
                    temp_gate, gate_wn[gate_sel], gate_ln,
                    gate_wp[gate_sel], gate_lp, gate_points,
                )
                if gate_sel in ["AND", "OR"]:
                    vout = VDD - vout
                    params = __import__("cmos_solver").extract_vtc_params(vin, vout)

                fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
                plot_smooth_curve(ax, vin, vout, lw=2.5, label=gate_sel)
                ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4)
                ax.set_xlim(0, VDD); ax.set_ylim(-0.05, VDD+0.05)
                ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
                ax.set_title(f"CMOS {gate_sel} Gate VTC")
                ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig); plt.close()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("VM",   f"{params['VM']:.3f} V")
                c2.metric("Gain", f"{params['Gain']:.1f}")
                c3.metric("NMH",  f"{params['NMH']:.3f} V")
                c4.metric("NML",  f"{params['NML']:.3f} V")
            except Exception as e:
                st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — DELAY & POWER
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[4]:
    st.header("Delay and Power Analysis")
    col1, col2 = st.columns(2)
    with col1:
        CL_fF = st.slider("Load Capacitance CL (fF)", 1, 1000, 10)
        freq_MHz = st.slider("Frequency (MHz)", 1, 1000, 500)
        gate_dp = st.selectbox("Gate", ["INV","NAND","NOR","AND","OR","XOR"], key="dp_gate")
        temp_dp = st.slider("Temperature (°C)", -40, 125, 27, 5, key="dp_temp")
        alpha_dp = st.slider("Activity factor α", 0.0, 1.0, 0.5, 0.05, key="dp_alpha")
        wn_dp = st.slider("Base NMOS W (µm)", 0.1, 5.0, 0.5, 0.1, key="dp_wn") * 1e-6
        wp_dp = st.slider("Base PMOS W (µm)", 0.1, 5.0, 2.5, 0.1, key="dp_wp") * 1e-6
        ln_dp = st.slider("NMOS L (nm)", 90, 1000, 180, 10, key="dp_ln") * 1e-9
        lp_dp = st.slider("PMOS L (nm)", 90, 1000, 180, 10, key="dp_lp") * 1e-9
    with col2:
        if st.button("Compute", key="dp_run"):
            from delay_power import propagation_delay, dynamic_power, energy_per_cycle, edp
            CL  = CL_fF * 1e-15
            f   = freq_MHz * 1e6
            tp  = propagation_delay(
                CL, gate_dp, temp_dp, wn_dp, wp_dp, ln_dp, lp_dp
            )
            P   = dynamic_power(CL, f, gate_dp, alpha_dp)
            E   = energy_per_cycle(CL, gate_dp, alpha_dp)
            EDP = edp(
                CL, gate_dp, f, temp_dp, alpha_dp, wn_dp, wp_dp, ln_dp, lp_dp
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("tp",     f"{tp*1e12:.2f} ps")
            c2.metric("Power",  f"{P*1e6:.3f} µW")
            c3.metric("Energy", f"{E*1e15:.3f} fJ")
            c4.metric("EDP",    f"{EDP*1e27:.3f} ×10⁻²⁷")

    st.subheader("Delay vs CL (log-log)")
    if st.button("Plot Delay Sweep", key="delay_sw"):
        from delay_power import propagation_delay
        CL_arr = np.logspace(-15, -12, 60)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        for g, clr in zip(["INV","NAND","NOR"], ["b","r","g"]):
            delays = [propagation_delay(cl, g)*1e12 for cl in CL_arr]
            ax.loglog(CL_arr*1e15, delays, color=clr, lw=2, label=g)
        ax.set_xlabel("CL (fF)"); ax.set_ylabel("tp (ps)")
        ax.set_title("Delay vs CL"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — CORNERS
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[5]:
    st.header("Process Corner & Temperature Analysis")
    analysis_type = st.radio("Analysis Type", ["Temperature", "Process Corners"])

    if analysis_type == "Temperature":
        temps_sel = st.multiselect("Temperatures (°C)", [-40, 0, 27, 75, 125], default=[-40, 27, 125])
        if st.button("Run Temperature Analysis"):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps_sel)))
            for T, clr in zip(sorted(temps_sel), colors):
                with st.spinner(f"T={T}°C …"):
                    vin, vout, params = cached_vtc(T, 0.5e-6, 180e-9, 1.0e-6, 180e-9)
                    plot_smooth_curve(
                        ax, vin, vout, color=clr, lw=2,
                        label=f"T={T:+d}°C  VM={params['VM']:.3f}V",
                    )
            ax.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.4)
            ax.set_xlim(0, VDD); ax.set_ylim(-0.05, VDD+0.05)
            ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
            ax.set_title("Inverter VTC vs Temperature")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()

    else:
        CORNERS = {
            "TT": (1.00, 1.00, 1.00),
            "SS": (0.85, 0.85, 1.10),
            "FF": (1.15, 1.15, 0.90),
            "FS": (1.15, 0.85, 1.00),
            "SF": (0.85, 1.15, 1.00),
        }
        corners_sel = st.multiselect("Corners", list(CORNERS.keys()), default=["TT","SS","FF"])
        if st.button("Run Corner Analysis"):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            colors = {"TT":"green","SS":"red","FF":"blue","FS":"orange","SF":"purple"}
            for c in corners_sel:
                wn_sc, wp_sc, l_sc = CORNERS[c]
                vin, vout, params = cached_vtc(27, 0.5e-6*wn_sc, 180e-9*l_sc, 1.0e-6*wp_sc, 180e-9*l_sc)
                plot_smooth_curve(
                    ax, vin, vout, color=colors.get(c, "gray"), lw=2,
                    label=f"{c}  VM={params['VM']:.3f}V",
                )
            ax.plot([0,VDD],[0,VDD],"k--",lw=1,alpha=0.4)
            ax.set_xlim(0,VDD); ax.set_ylim(-0.05,VDD+0.05)
            ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
            ax.set_title("VTC: Process Corners")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[6]:
    st.header("Monte Carlo Analysis")
    n_mc    = st.slider("Number of Runs", 50, 1000, 200, 50)
    sigma   = st.slider("σ_VT (mV)", 5, 80, 30, 5) / 1000
    temp_mc = st.slider("Temperature (°C)", -40, 125, 27, 5, key="mc_temp")
    mc_points = st.slider("VIN samples per run", 31, 201, 61, 10, key="mc_points")
    st.info(f"Running {n_mc} simulations with σ_VT = {sigma*1e3:.0f} mV. More runs = slower but more accurate.")

    if st.button("Run Monte Carlo", key="mc_run"):
        from scipy.optimize import brentq as _brentq
        from current_predictor import predict_nmos_current as _pn, predict_pmos_current as _pp
        rng     = np.random.default_rng(42)
        dvt_n   = rng.normal(0, sigma, n_mc)
        dvt_p   = rng.normal(0, sigma, n_mc)
        vin_arr = np.linspace(0, VDD, mc_points)
        vm_list = []
        progress = st.progress(0)

        for i in range(n_mc):
            vout_run = np.zeros(len(vin_arr))
            for j, vin in enumerate(vin_arr):
                def res(vo):
                    return (_pn(max(vin+dvt_n[i],0), max(vo,0), temp_mc, 0.5e-6, 180e-9)
                          - _pp(max(VDD-vin+dvt_p[i],0), max(VDD-vo,0), temp_mc, 1e-6, 180e-9))
                try:
                    vout_run[j] = _brentq(res, 0, VDD, xtol=1e-5, maxiter=60)
                except:
                    vout_run[j] = 0 if res(0) > 0 else VDD
            diff = vout_run - vin_arr
            vm_list.append(float(vin_arr[np.argmin(np.abs(diff))]))
            progress.progress((i+1)/n_mc)

        vm_arr = np.array(vm_list)
        st.success(f"VM: mean={vm_arr.mean():.3f}V  σ={vm_arr.std():.4f}V")

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.hist(vm_arr, bins=30, color="steelblue", edgecolor="k", alpha=0.75, density=True)
        from scipy.stats import norm as _norm
        x = np.linspace(vm_arr.min(), vm_arr.max(), 200)
        ax.plot(x, _norm.pdf(x, vm_arr.mean(), vm_arr.std()), "r-", lw=2)
        ax.set_xlabel("VM (V)"); ax.set_ylabel("Density")
        ax.set_title("Monte Carlo VM Distribution")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — RING OSCILLATOR
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[7]:
    st.header("Ring Oscillator")
    CL_ro  = st.slider("CL per stage (fF)", 1, 100, 10) * 1e-15
    temp_ro = st.slider("Temperature (°C)", -40, 125, 27, 5, key="ro_temp")
    wn_ro = st.slider("Base NMOS W (µm)", 0.1, 5.0, 0.5, 0.1, key="ro_wn") * 1e-6
    wp_ro = st.slider("Base PMOS W (µm)", 0.1, 5.0, 2.5, 0.1, key="ro_wp") * 1e-6
    ln_ro = st.slider("NMOS L (nm)", 90, 1000, 180, 10, key="ro_ln") * 1e-9
    lp_ro = st.slider("PMOS L (nm)", 90, 1000, 180, 10, key="ro_lp") * 1e-9
    stages = st.multiselect("Stages (odd only)", [3,5,7,9,11,13,15,21,31], default=[3,5,7,9,11])

    if st.button("Compute", key="ro_run"):
        from delay_power import propagation_delay
        freqs = []
        data  = []
        for N in sorted(stages):
            tp = propagation_delay(
                CL_ro, "INV", temp_ro, wn_ro, wp_ro, ln_ro, lp_ro
            )
            f  = 1 / (2 * N * tp)
            freqs.append(f/1e9)
            data.append({"Stages": N, "tp (ps)": f"{tp*1e12:.2f}", "f (GHz)": f"{f/1e9:.4f}"})

        import pandas as pd
        st.dataframe(pd.DataFrame(data))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(sorted(stages), freqs, "bo-", ms=8, lw=2)
        ax.set_xlabel("Number of Stages"); ax.set_ylabel("Frequency (GHz)")
        ax.set_title("Ring Oscillator Frequency vs Stages")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 9 — SRAM
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[8]:
    st.header("6T SRAM Characterization")
    temp_sram = st.slider("Temperature (°C)", -40, 125, 27, 5, key="sram_t")
    bitline_ff = st.slider("Bitline capacitance (fF)", 10, 500, 100, 10)
    sense_mv = st.slider("Read sense differential (mV)", 10, 500, 100, 10)
    storage_ff = st.slider("Storage-node capacitance (fF)", 0.5, 20.0, 2.0, 0.5)

    if st.button("Generate Butterfly Curve", key="sram_run"):
        with st.spinner("Computing …"):
            try:
                from sram import butterfly_curve, characterize_sram_timing
                snm, vin, vout1, vout2 = butterfly_curve(T=temp_sram, save=False)
                timing = characterize_sram_timing(
                    T=temp_sram, C_bitline=bitline_ff * 1e-15,
                    C_storage=storage_ff * 1e-15,
                    sense_delta=sense_mv / 1000,
                )

                fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                plot_smooth_curve(ax, vin, vout1, color="blue", lw=2,
                                  label="INV1 Q→QB")
                plot_smooth_curve(ax, vout1, vin, color="red", lw=2,
                                  label="INV2 QB→Q")
                ax.plot([0,VDD],[0,VDD],"k--",lw=1,alpha=0.4)
                ax.set_xlim(0, VDD); ax.set_ylim(0, VDD)
                ax.set_xlabel("Q (V)"); ax.set_ylabel("QB (V)")
                ax.set_title(f"SRAM Butterfly Curve  (T={temp_sram}°C)")
                ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
                st.pyplot(fig); plt.close()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Read time", f"{timing['read']['read_time_s']*1e12:.2f} ps")
                c2.metric("Write time", "Not writeable" if not timing['write']['writeable']
                          else f"{timing['write']['write_time_s']*1e12:.2f} ps")
                c3.metric("Read SNM", f"{snm*1e3:.1f} mV")
                c4.metric("Hold SNM", f"{timing['hold']['hold_snm_V']*1e3:.1f} mV")
                st.caption("A static 6T SRAM cell has no finite flip-flop-style hold time while WL is low and VDD is applied.")
            except Exception as e:
                st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 10 — BOOLEAN CHARACTERIZER
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[9]:
    from boolean_characterizer import streamlit_boolean_tab
    streamlit_boolean_tab()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 11 — REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
if selected_page == NAV_PAGES[10]:
    st.header("Automatic PDF Report Generator")
    st.info("Ensure main.py has been run at least once to generate all plots before generating the report.")

    REPORT_PATH = os.path.join(BASE_DIR, "reports", "CMOS_AI_EDA_Report.pdf")

    if st.button("Generate PDF Report", key="report_run"):
        with st.spinner("Building PDF …"):
            try:
                from report_generator import generate_report
                path = generate_report()
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(
                            "📥 Download PDF Report",
                            data=f,
                            file_name="CMOS_AI_EDA_Report.pdf",
                            mime="application/pdf",
                        )
                    st.success(f"Report generated: {path}")
                else:
                    st.warning("PDF generation requires fpdf2: pip install fpdf2")
            except Exception as e:
                st.error(str(e))

    # show existing plots
    st.subheader("Generated Plots")
    plot_dir = os.path.join(BASE_DIR, "plots")
    if os.path.exists(plot_dir):
        png_files = sorted(
            [f for f in os.listdir(plot_dir) if f.endswith(".png")]
        )
        if png_files:
            cols = st.columns(3)
            for i, fname in enumerate(png_files):
                with cols[i % 3]:
                    st.image(os.path.join(plot_dir, fname),
                             caption=fname.replace("_", " ").replace(".png", ""),
                             width="stretch")
        else:
            st.warning("No plots found. Run main.py first.")
    else:
        st.warning("Plots directory not found. Run main.py first.")


def _compute_all_characterization(
    T, CL, freq, mc_runs, sigma_vt, expression, progress,
    vtc_points, wn_um, wp_um, ln_nm, lp_nm, alpha, ro_stages,
    bitline_ff, sense_mv,
):
    """Run the same model-backed analyses exposed by the individual tabs."""
    import pandas as pd
    from scipy.optimize import brentq
    from cmos_solver import extract_vtc_params
    from delay_power import (
        compute_performance_table, dynamic_power, edp, get_ion,
        propagation_delay,
    )
    from ring_oscillator import ring_oscillator_freq
    from dff import dff_timing
    from sram import butterfly_curve, characterize_sram_timing
    from boolean_characterizer import characterize_boolean

    pn, pp = load_predictor()
    wn = wn_um * 1e-6
    wp = wp_um * 1e-6
    ln = ln_nm * 1e-9
    lp = lp_nm * 1e-9
    vgs = np.linspace(0, VDD, vtc_points)
    nmos_ids = np.asarray([pn(v, VDD, T, wn, ln) for v in vgs])
    pmos_ids = np.asarray([pp(v, VDD, T, wp, lp) for v in vgs])
    vds = np.linspace(0, VDD, 181)
    nmos_output_ids = np.asarray(
        [pn(VDD, v, T, wn, ln) for v in vds]
    )
    pmos_output_ids = np.asarray(
        [pp(VDD, v, T, wp, lp) for v in vds]
    )
    gm_nmos = float(np.gradient(nmos_ids, vgs)[np.argmin(np.abs(vgs - 1.2))])
    gm_pmos = float(np.gradient(pmos_ids, vgs)[np.argmin(np.abs(vgs - 1.2))])
    gds_index = int(np.argmin(np.abs(vds - VDD / 2)))
    gds_nmos = float(np.gradient(nmos_output_ids, vds)[gds_index])
    gds_pmos = float(np.gradient(pmos_output_ids, vds)[gds_index])
    progress.progress(0.08, text="Transistor sweeps complete")

    gate_wn = {"INV": wn, "NAND": wn / 2, "NOR": 2 * wn,
               "AND": wn / 2, "OR": 2 * wn, "XOR": 2 * wn}
    gate_wp = {"INV": wp, "NAND": 2 * wp, "NOR": wp / 2,
               "AND": 2 * wp, "OR": wp / 2, "XOR": 2 * wp}
    gates = list(gate_wn)
    vtcs = {}
    for gate in gates:
        vin, vout, _ = cached_vtc_with_resolution(
            T, gate_wn[gate], ln, gate_wp[gate], lp, vtc_points
        )
        if gate in ("AND", "OR"):
            vout = VDD - vout
        vtcs[gate] = {
            "vin": vin,
            "vout": vout,
            "params": extract_vtc_params(vin, vout),
        }
    progress.progress(0.2, text="Inverter and logic-gate VTCs complete")

    performance = pd.DataFrame(
        compute_performance_table(
            CL=CL, f=freq, T=T, alpha=alpha,
            Wn_base=wn, Wp_base=wp, Ln=ln, Lp=lp,
        )
    )
    performance["Gain"] = [
        vtcs[gate]["params"]["Gain"] for gate in performance["Circuit"]
    ]
    performance["VM_V"] = [
        vtcs[gate]["params"]["VM"] for gate in performance["Circuit"]
    ]
    performance["NMH_V"] = [
        vtcs[gate]["params"]["NMH"] for gate in performance["Circuit"]
    ]
    performance["NML_V"] = [
        vtcs[gate]["params"]["NML"] for gate in performance["Circuit"]
    ]
    performance["VOH_V"] = [
        vtcs[gate]["params"]["VOH"] for gate in performance["Circuit"]
    ]
    performance["VOL_V"] = [
        vtcs[gate]["params"]["VOL"] for gate in performance["Circuit"]
    ]
    performance["VIL_V"] = [
        vtcs[gate]["params"]["VIL"] for gate in performance["Circuit"]
    ]
    performance["VIH_V"] = [
        vtcs[gate]["params"]["VIH"] for gate in performance["Circuit"]
    ]
    performance["ION_uA"] = [
        get_ion(gate, T, wn, wp, ln, lp) * 1e6
        for gate in performance["Circuit"]
    ]
    performance["EDP_x1e27_Js"] = [
        edp(CL, gate, freq, T, alpha, wn, wp, ln, lp) * 1e27
        for gate in performance["Circuit"]
    ]

    cl_sweep = np.logspace(-15, -12, 60)
    delay_sweep = {
        gate: np.asarray([
            propagation_delay(cl, gate, T, wn, wp, ln, lp) * 1e12
                          for cl in cl_sweep])
        for gate in gates
    }
    freq_sweep = np.logspace(6, 9, 60)
    power_sweep = {
        gate: np.asarray([dynamic_power(CL, f, gate, alpha) * 1e6
                          for f in freq_sweep])
        for gate in gates
    }
    edp_values = {
        gate: edp(CL, gate, freq, T, alpha, wn, wp, ln, lp) * 1e27
        for gate in gates
    }
    progress.progress(0.35, text="Delay, power, energy, and EDP complete")

    corners = {
        "TT": (1.00, 1.00, 1.00),
        "SS": (0.85, 0.85, 1.10),
        "FF": (1.15, 1.15, 0.90),
        "FS": (1.15, 0.85, 1.00),
        "SF": (0.85, 1.15, 1.00),
    }
    corner_vtcs = {}
    for name, (wn_scale, wp_scale, l_scale) in corners.items():
        vin, vout, params = cached_vtc_with_resolution(
            T, wn * wn_scale, ln * l_scale,
            wp * wp_scale, lp * l_scale, vtc_points,
        )
        corner_vtcs[name] = (vin, vout, params)
    temperature_vtcs = {
        temp: cached_vtc_with_resolution(
            temp, wn, ln, wp, lp, vtc_points
        )
        for temp in (-40, 27, 75, 125)
    }
    progress.progress(0.48, text="Temperature and process-corner analyses complete")

    mc_rng = np.random.default_rng(42)
    dvt_n = mc_rng.normal(0, sigma_vt, mc_runs)
    dvt_p = mc_rng.normal(0, sigma_vt, mc_runs)
    mc_vin = np.linspace(0, VDD, min(vtc_points, 201))
    mc_vouts = np.empty((mc_runs, len(mc_vin)))
    mc_vm = np.empty(mc_runs)
    for i in range(mc_runs):
        vout_run = np.zeros(len(mc_vin))
        for j, vin in enumerate(mc_vin):
            def residual(vout):
                return (
                    pn(max(vin + dvt_n[i], 0), max(vout, 0), T, wn, ln)
                    - pp(max(VDD - vin + dvt_p[i], 0),
                         max(VDD - vout, 0), T, wp, lp)
                )

            try:
                vout_run[j] = brentq(residual, 0, VDD, xtol=1e-5, maxiter=60)
            except ValueError:
                vout_run[j] = 0 if residual(0) > 0 else VDD
        mc_vouts[i] = vout_run
        mc_vm[i] = mc_vin[np.argmin(np.abs(vout_run - mc_vin))]
        if i % max(1, mc_runs // 20) == 0 or i == mc_runs - 1:
            progress.progress(
                0.48 + 0.2 * (i + 1) / mc_runs,
                text=f"Monte Carlo: {i + 1}/{mc_runs} runs",
            )

    stages = sorted(set((3, ro_stages)))
    ro_rows = []
    for count in stages:
        ro_freq, ro_tp = ring_oscillator_freq(
            count, CL, T, wn, wp, ln, lp
        )
        ro_rows.append({
            "Stages": count,
            "Delay per stage (ps)": ro_tp * 1e12,
            "Frequency (GHz)": ro_freq / 1e9,
        })
    dff = dff_timing(CL=CL, T=T, Wn=wn, Wp=wp, Ln=ln, Lp=lp)
    snm, sram_vin, sram_vout1, sram_vout2 = butterfly_curve(T=T, save=False)
    sram_timing = characterize_sram_timing(
        T=T, C_bitline=bitline_ff * 1e-15, sense_delta=sense_mv / 1000
    )
    progress.progress(0.78, text="Oscillator, D flip-flop, and SRAM analyses complete")

    boolean = characterize_boolean(
        expression, T=T, CL=CL, f=freq, verbose=False, save_plots=False,
    )
    if boolean is None:
        raise ValueError("Boolean expression could not be characterized.")
    progress.progress(1.0, text="Full characterization complete")

    return {
        "temperature": T,
        "load_capacitance": CL,
        "frequency": freq,
        "vtc_points": vtc_points,
        "transistor_widths_um": (wn_um, wp_um),
        "transistor_lengths_nm": (ln_nm, lp_nm),
        "activity_factor": alpha,
        "ring_stages": ro_stages,
        "vgs": vgs,
        "nmos_ids": nmos_ids,
        "pmos_ids": pmos_ids,
        "vds": vds,
        "nmos_output_ids": nmos_output_ids,
        "pmos_output_ids": pmos_output_ids,
        "transistor_small_signal": {
            "NMOS gm at VGS=1.2 V, VDS=VDD (S)": gm_nmos,
            "NMOS gds at VGS=VDD, VDS=VDD/2 (S)": gds_nmos,
            "PMOS gm at VSG=1.2 V, VSD=VDD (S)": gm_pmos,
            "PMOS gds at VSG=VDD, VSD=VDD/2 (S)": gds_pmos,
        },
        "vtcs": vtcs,
        "performance": performance,
        "cl_sweep": cl_sweep,
        "delay_sweep": delay_sweep,
        "freq_sweep": freq_sweep,
        "power_sweep": power_sweep,
        "edp_values": edp_values,
        "corner_vtcs": corner_vtcs,
        "temperature_vtcs": temperature_vtcs,
        "mc_vin": mc_vin,
        "mc_vouts": mc_vouts,
        "mc_vm": mc_vm,
        "ro_table": pd.DataFrame(ro_rows),
        "dff": dff,
        "snm": snm,
        "sram_vin": sram_vin,
        "sram_vout1": sram_vout1,
        "sram_vout2": sram_vout2,
        "sram_timing": sram_timing,
        "boolean": boolean,
    }


if selected_page == NAV_PAGES[11]:
    st.header("All-Circuits Characterization")
    st.markdown(
        "Run the transistor, logic, timing, variability, oscillator, SRAM, "
        "and Boolean analyses together. Results remain visible while you "
        "explore other settings."
    )
    st.subheader("Analysis parameters")

    def parameter_row(icon, title, description, widget):
        with st.container(border=True):
            icon_col, detail_col, input_col = st.columns([0.09, 0.34, 0.57])
            with icon_col:
                st.markdown(
                    f'<div class="parameter-icon">{icon}</div>',
                    unsafe_allow_html=True,
                )
            with detail_col:
                st.markdown(
                    f'<div class="parameter-list-title">{title}</div>'
                    f'<div class="parameter-list-caption">{description}</div>',
                    unsafe_allow_html=True,
                )
            with input_col:
                return widget()

    all_temp = parameter_row(
        "🌡️", "Temperature", "Device and circuit temperature in °C.",
        lambda: st.slider(
            "Analysis temperature (°C)", -40, 125, 27, 5, key="all_temp"
        ),
    )
    all_cl_ff = parameter_row(
        "🔋", "Load capacitance", "Output load used for timing and power.",
        lambda: st.slider(
            "Load capacitance (fF)", 1, 1000, 10, key="all_cl"
        ),
    )
    all_freq_mhz = parameter_row(
        "⏱️", "Clock frequency", "Operating frequency used for power estimates.",
        lambda: st.slider(
            "Clock frequency (MHz)", 1, 1000, 500, key="all_freq"
        ),
    )
    all_mc_runs = parameter_row(
        "🎲", "Monte Carlo samples", "Number of threshold-variation runs.",
        lambda: st.slider(
            "Monte Carlo runs", 20, 200, 50, 10, key="all_mc_runs"
        ),
    )
    all_sigma_vt = parameter_row(
        "📈", "Threshold variation", "Gaussian threshold sigma in millivolts.",
        lambda: st.slider(
            "Monte Carlo σVT (mV)", 5, 80, 30, 5, key="all_sigma"
        ),
    ) / 1000
    all_vtc_points = parameter_row(
        "〰️", "VTC resolution",
        "Number of model samples used by each transfer-curve solve.",
        lambda: st.slider(
            "VTC points", 101, 1001, 501, 100, key="all_vtc_points"
        ),
    )
    all_wn_um = parameter_row(
        "↔️", "NMOS width", "Base NMOS channel width in micrometres.",
        lambda: st.slider(
            "NMOS W (µm)", 0.1, 5.0, 0.5, 0.1, key="all_wn"
        ),
    )
    all_wp_um = parameter_row(
        "↔️", "PMOS width", "Base PMOS channel width in micrometres.",
        lambda: st.slider(
            "PMOS W (µm)", 0.1, 5.0, 1.0, 0.1, key="all_wp"
        ),
    )
    all_ln_nm = parameter_row(
        "↕️", "NMOS length", "NMOS channel length in nanometres.",
        lambda: st.slider(
            "NMOS L (nm)", 90, 1000, 180, 10, key="all_ln"
        ),
    )
    all_lp_nm = parameter_row(
        "↕️", "PMOS length", "PMOS channel length in nanometres.",
        lambda: st.slider(
            "PMOS L (nm)", 90, 1000, 180, 10, key="all_lp"
        ),
    )
    all_alpha = parameter_row(
        "⚡", "Switching activity",
        "Activity factor used for dynamic power and energy estimates.",
        lambda: st.slider(
            "Activity factor α", 0.0, 1.0, 0.5, 0.05, key="all_alpha"
        ),
    )
    all_ro_stages = parameter_row(
        "🔄", "Ring oscillator stages", "Odd inverter-stage count for the oscillator.",
        lambda: st.select_slider(
            "Odd number of stages", options=[3, 5, 7, 9, 11, 15, 21, 31],
            value=7, key="all_ro_stages",
        ),
    )
    all_bitline_ff = parameter_row(
        "💾", "SRAM bitline capacitance",
        "First-order bitline capacitance used for SRAM read-time estimates.",
        lambda: st.slider(
            "Bitline capacitance (fF)", 10, 500, 100, 10,
            key="all_bitline_cap",
        ),
    )
    all_sense_mv = parameter_row(
        "📉", "SRAM sense voltage",
        "Bitline differential required by the read-time estimate.",
        lambda: st.slider(
            "Sense differential (mV)", 10, 500, 100, 10,
            key="all_sense_delta",
        ),
    )
    all_expression = parameter_row(
        "🔣", "Boolean expression",
        "Supported operators: AND, OR, NOT, NAND, NOR, XOR, XNOR.",
        lambda: st.text_input(
            "Boolean expression",
            value="A AND B OR NOT C",
            key="all_boolean_expression",
            help="Parentheses are supported for grouping.",
        ),
    )

    if st.button("⚡ Run Full Characterization", type="primary", key="all_run"):
        progress = st.progress(0, text="Starting all circuit analyses…")
        try:
            with st.spinner("Running the complete characterization…"):
                st.session_state["all_circuit_results"] = _compute_all_characterization(
                    all_temp, all_cl_ff * 1e-15, all_freq_mhz * 1e6,
                    all_mc_runs, all_sigma_vt, all_expression, progress,
                    all_vtc_points, all_wn_um, all_wp_um, all_ln_nm, all_lp_nm,
                    all_alpha, all_ro_stages, all_bitline_ff, all_sense_mv,
                )
            st.success("All circuit calculations and plots are ready.")
        except Exception as error:
            st.error(f"Full characterization failed: {error}")
            st.exception(error)

    results = st.session_state.get("all_circuit_results")
    if results:
        import pandas as pd

        st.caption(
            f"Showing the last completed run: T={results['temperature']}°C, "
            f"CL={results['load_capacitance'] * 1e15:g} fF, "
            f"f={results['frequency'] / 1e6:g} MHz, "
            f"VTC resolution={results['vtc_points']} points, "
            f"α={results['activity_factor']:g}."
        )
        with st.expander("Calculation models and assumptions", expanded=False):
            st.markdown(
                "- Transistor currents and VTCs come from the saved ML models.\n"
                "- The device table reports model-derived DC current and "
                "finite-difference small-signal `gm`/`gds` at the stated "
                "bias points; these are estimates from the trained model.\n"
                "- `tp = Csw × VDD / ION`, `P = α × Csw × VDD² × f`, "
                "and `E = α × Csw × VDD²`; `Csw` includes the modeled "
                "first-order intrinsic-capacitance assumption and external "
                "load. Delay is a single first-order estimate, not separate "
                "rise/fall propagation delay.\n"
                "- `EDP = E × tp`; ring oscillator frequency is "
                "`1 / (2 × N × tp)`.\n"
                "- DFF estimates use `tsetup = 2tp`, `thold = tp`, "
                "`tcq = 4tp`, and `fmax = 1 / (tcq + tsetup)`.\n"
                "- SRAM read/write times are first-order estimates; its "
                "butterfly SNM is estimated from inverter noise margins.\n"
                "- Monte Carlo applies independent Gaussian NMOS/PMOS "
                "threshold offsets at each run.\n"
                "- These inputs do not provide extracted `Cgs/Cgd/Cdb`, "
                "reverse leakage, breakdown, noise, or measured transient "
                "waveforms. Those parameters require validated SPICE models "
                "and dedicated DC/AC/transient analyses; they are not inferred "
                "from smoothed plots."
            )

        st.subheader("Transistor characteristics")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        plot_smooth_curve(axes[0, 0], results["vgs"],
                          results["nmos_ids"] * 1e6, color="blue")
        axes[0, 0].set(title="NMOS IDS–VGS", xlabel="VGS (V)", ylabel="IDS (µA)")
        plot_smooth_curve(axes[0, 1], results["vgs"],
                          results["pmos_ids"] * 1e6, color="red")
        axes[0, 1].set(title="PMOS |IDS|–VSG", xlabel="VSG (V)", ylabel="|IDS| (µA)")
        plot_smooth_curve(axes[1, 0], results["vds"],
                          results["nmos_output_ids"] * 1e6, color="blue")
        axes[1, 0].set(title="NMOS IDS–VDS at VGS=VDD",
                       xlabel="VDS (V)", ylabel="IDS (µA)")
        plot_smooth_curve(axes[1, 1], results["vds"],
                          results["pmos_output_ids"] * 1e6, color="red")
        axes[1, 1].set(title="PMOS |IDS|–VSD at VSG=VDD",
                       xlabel="VSD (V)", ylabel="|IDS| (µA)")
        for ax in axes:
            for axis in ax:
                axis.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        st.caption(
            "The following small-signal slopes are finite differences of the "
            "model-predicted IV curves at the stated bias points."
        )
        st.dataframe(
            pd.DataFrame(
                [{"Parameter": name, "Estimated value (S)": value}
                 for name, value in results["transistor_small_signal"].items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Inverter and logic-gate DC characterization")
        st.caption(
            "Values below are from the model-backed DC VTC and the "
            "first-order delay/power model at the selected operating point."
        )
        circuit_columns = [
            "Circuit", "VOH_V", "VOL_V", "VM_V", "VIL_V", "VIH_V",
            "Gain", "NMH_V", "NML_V", "ION_uA", "Csw_fF", "Delay_ps",
            "Power_uW", "Energy_fJ", "EDP_x1e27_Js",
        ]
        circuit_table = results["performance"][circuit_columns].rename(
            columns={
                "VOH_V": "VOH (V)", "VOL_V": "VOL (V)", "VM_V": "VM (V)",
                "VIL_V": "VIL (V)", "VIH_V": "VIH (V)",
                "Gain": "Peak |dVOUT/dVIN|", "NMH_V": "NMH (V)",
                "NML_V": "NML (V)", "ION_uA": "ION (µA)",
                "Csw_fF": "Csw estimate (fF)", "Delay_ps": "Delay estimate (ps)",
                "Power_uW": "Power (µW)", "Energy_fJ": "Energy/cycle (fJ)",
                "EDP_x1e27_Js": "EDP (×10⁻²⁷ J·s)",
            }
        )
        st.dataframe(circuit_table, hide_index=True, use_container_width=True)
        st.download_button(
            "Download circuit characterization table (CSV)",
            data=circuit_table.to_csv(index=False).encode("utf-8"),
            file_name="cmos_circuit_characterization.csv",
            mime="text/csv",
            key="all_download_circuits",
        )
        with st.expander("Parameters requiring transistor-level SPICE data"):
            st.markdown(
                "Not calculated in this model: `Cgs`, `Cgd`, `Cdb`, leakage "
                "current, breakdown voltage, short-circuit current, "
                "rise/fall-specific delay, propagation delay under a "
                "specified input slew, dynamic node waveforms, noise "
                "spectrum, and process-qualified statistical limits. "
                "Use a foundry PDK/SPICE deck and DC, AC, and transient "
                "simulations before reporting these as circuit results."
            )

        st.subheader("Inverter and logic-gate VTCs")
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=150)
        for ax, (gate, curve) in zip(axes.flat, results["vtcs"].items()):
            plot_smooth_curve(ax, curve["vin"], curve["vout"], lw=2, label=gate)
            ax.plot([0, VDD], [0, VDD], "k--", lw=0.8, alpha=0.4)
            ax.axvline(curve["params"]["VM"], color="green", ls=":", lw=1)
            ax.set(title=gate, xlabel="VIN (V)", ylabel="VOUT (V)",
                   xlim=(0, VDD), ylim=(-0.05, VDD + 0.05))
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        st.subheader("Delay, power, energy, and EDP")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=150)
        for gate, values in results["delay_sweep"].items():
            plot_smooth_curve(axes[0], results["cl_sweep"] * 1e15,
                              values, label=gate)
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
        axes[0].set(title="Delay vs load", xlabel="CL (fF)", ylabel="Delay (ps)")
        for gate, values in results["power_sweep"].items():
            plot_smooth_curve(axes[1], results["freq_sweep"] / 1e6,
                              values, label=gate)
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
        axes[1].set(title="Dynamic power vs frequency", xlabel="Frequency (MHz)",
                    ylabel="Power (µW)")
        axes[2].bar(list(results["edp_values"]), list(results["edp_values"].values()))
        axes[2].set(title="Energy-delay product", ylabel="EDP (×10⁻²⁷ J·s)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            if ax is not axes[2]:
                ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"At the selected load and frequency, energy per cycle is "
            f"{results['performance']['Energy_fJ'].iloc[0]:.4g} fJ for the "
            "inverter; the table above reports each gate."
        )

        st.subheader("Temperature and process corners")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        for temp, (vin, vout, params) in results["temperature_vtcs"].items():
            plot_smooth_curve(
                axes[0], vin, vout,
                label=f"{temp}°C, VM={params['VM']:.3f} V",
            )
        for name, (vin, vout, params) in results["corner_vtcs"].items():
            plot_smooth_curve(
                axes[1], vin, vout,
                label=f"{name}, VM={params['VM']:.3f} V",
            )
        for ax, title in zip(axes, ("Temperature sweep", "Process corners")):
            ax.plot([0, VDD], [0, VDD], "k--", lw=0.8, alpha=0.4)
            ax.set(title=title, xlabel="VIN (V)", ylabel="VOUT (V)",
                   xlim=(0, VDD), ylim=(-0.05, VDD + 0.05))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        corner_rows = [
            {"Corner": name, "VM (V)": params["VM"], "Gain": params["Gain"],
             "NMH (V)": params["NMH"], "NML (V)": params["NML"]}
            for name, (_, _, params) in results["corner_vtcs"].items()
        ]
        st.dataframe(pd.DataFrame(corner_rows), hide_index=True,
                     use_container_width=True)

        st.subheader("Monte Carlo variation")
        mc_vm = results["mc_vm"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean VM", f"{mc_vm.mean():.4f} V")
        c2.metric("VM standard deviation", f"{mc_vm.std():.4f} V")
        c3.metric("VM range", f"{mc_vm.min():.3f}–{mc_vm.max():.3f} V")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
        axes[0].hist(mc_vm, bins=25, color="steelblue", edgecolor="black")
        axes[0].set(title="Switching-voltage distribution", xlabel="VM (V)",
                    ylabel="Count")
        for mc_curve in results["mc_vouts"]:
            plot_smooth_curve(
                axes[1], results["mc_vin"], mc_curve, points=200,
                color="steelblue", alpha=0.08, lw=0.7,
            )
        plot_smooth_curve(
            axes[1], results["mc_vin"],
            np.median(results["mc_vouts"], axis=0),
            color="red", lw=2, label="Median VTC",
        )
        axes[1].set(title="Monte Carlo VTC spread", xlabel="VIN (V)",
                    ylabel="VOUT (V)", xlim=(0, VDD), ylim=(-0.05, VDD + 0.05))
        axes[1].legend()
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Curves are rendered with shape-preserving interpolation where "
            "the sampled axis is unique. Interpolation is visual only; all "
            "reported values use the underlying model samples."
        )

        st.subheader("Ring oscillator")
        st.dataframe(results["ro_table"], hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(results["ro_table"]["Stages"],
                results["ro_table"]["Frequency (GHz)"], "bo-", lw=2)
        ax.set(title="Oscillation frequency vs stage count",
               xlabel="Number of inverter stages", ylabel="Frequency (GHz)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("D flip-flop timing")
        dff = results["dff"]
        dff_labels = ["tp", "Setup", "Hold", "Clock-to-Q"]
        dff_values = [dff["tp_ps"], dff["t_setup_ps"],
                      dff["t_hold_ps"], dff["t_cq_ps"]]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Inverter tp", f"{dff['tp_ps']:.2f} ps")
        c2.metric("Setup", f"{dff['t_setup_ps']:.2f} ps")
        c3.metric("Hold", f"{dff['t_hold_ps']:.2f} ps")
        c4.metric("Clock-to-Q", f"{dff['t_cq_ps']:.2f} ps")
        c5.metric("Maximum clock", f"{dff['f_max_GHz']:.3f} GHz")
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.bar(dff_labels, dff_values, color=["#1f77b4", "#ff7f0e",
                                              "#2ca02c", "#d62728"])
        ax.set(title="DFF timing estimates", ylabel="Time (ps)")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("6T SRAM")
        timing = results["sram_timing"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Read time", f"{timing['read']['read_time_s'] * 1e12:.2f} ps")
        c2.metric("Write time", "Not writeable" if not timing["write"]["writeable"]
                  else f"{timing['write']['write_time_s'] * 1e12:.2f} ps")
        c3.metric("Read SNM", f"{results['snm'] * 1e3:.1f} mV")
        c4.metric("Hold SNM",
                  f"{timing['hold']['hold_snm_V'] * 1e3:.1f} mV")
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        plot_smooth_curve(ax, results["sram_vin"], results["sram_vout1"],
                          label="INV1 Q→QB")
        plot_smooth_curve(ax, results["sram_vout1"], results["sram_vin"],
                          label="INV2 QB→Q")
        ax.plot([0, VDD], [0, VDD], "k--", lw=0.8, alpha=0.4)
        ax.set(title="6T SRAM butterfly curve", xlabel="Q (V)", ylabel="QB (V)",
               xlim=(0, VDD), ylim=(0, VDD))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Boolean expression circuit")
        boolean = results["boolean"]
        metrics = boolean["metrics"]
        params = metrics["output_params"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Logic stages", str(metrics["num_stages"]))
        c2.metric("Total delay", f"{metrics['total_delay_ps']:.2f} ps")
        c3.metric("Total power", f"{metrics['total_power_uW']:.4f} µW")
        c4.metric("EDP", f"{metrics['EDP'] * 1e27:.4f} ×10⁻²⁷ J·s")
        st.write(f"Expression: `{boolean['expression']}`")
        st.dataframe(pd.DataFrame(metrics["stage_metrics"]),
                     hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        plot_smooth_curve(
            ax, metrics["output_vin"], metrics["output_vout"], lw=2,
            label="Expression VTC",
        )
        ax.plot([0, VDD], [0, VDD], "k--", alpha=0.4, label="VIN = VOUT")
        ax.axvline(params["VM"], color="green", ls=":", label="VM")
        ax.set(title="Boolean expression VTC", xlabel="Input voltage (V)",
               ylabel="Output voltage (V)", xlim=(0, VDD),
               ylim=(-0.05, VDD + 0.05))
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


if selected_page == NAV_PAGES[12]:
    import json
    import pandas as pd

    reference_dir = os.path.join(BASE_DIR, "data", "sky130_reference")
    manifest_path = os.path.join(reference_dir, "manifest.json")
    st.header("SKY130 ngspice reference data")
    st.markdown(
        "A separate 130 nm reference dataset generated from the recorded "
        "SKY130 PDK model deck. It does not replace or relabel the existing "
        "180 nm model and CSV files."
    )
    if not os.path.isfile(manifest_path):
        st.warning(
            "No SKY130 reference dataset has been generated yet. This project "
            "contains the generator and schema, but real data requires an "
            "installed SKY130 PDK model deck and ngspice."
        )
        st.code(
            "python spice/generate_sky130_dataset.py "
            "--pdk-root <PDK_ROOT> --pdk-revision <exact-PDK-revision>"
        )
        schema_path = os.path.join(BASE_DIR, "spice", "data_schema.json")
        with open(schema_path, "rb") as schema_file:
            st.download_button(
                "Download SKY130 data schema",
                data=schema_file.read(),
                file_name="sky130_data_schema.json",
                mime="application/json",
            )
    else:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        st.success(
            "Reference data is present. These are ngspice model results, "
            "not measured-silicon data."
        )
        st.json(manifest)
        from spice.analyze_reference import extract_reference_metrics

        device_path = os.path.join(reference_dir, "device_dc.csv")
        inverter_path = os.path.join(reference_dir, "inverter_dc.csv")
        if os.path.isfile(device_path) and os.path.isfile(inverter_path):
            device_metrics, inverter_metrics = extract_reference_metrics(
                pd.read_csv(device_path), pd.read_csv(inverter_path)
            )
            st.subheader("Extracted device and inverter DC metrics")
            device_display = device_metrics[
                [
                    "device_type", "corner", "temperature_c", "w_um", "l_um",
                    "ion_a", "ioff_a", "gm_max_s", "gds_max_s",
                ]
            ].copy()
            device_display["ion_uA"] = device_display.pop("ion_a") * 1e6
            device_display["ioff_nA"] = device_display.pop("ioff_a") * 1e9
            device_display["gm_max_uS"] = device_display.pop("gm_max_s") * 1e6
            device_display["gds_max_uS"] = device_display.pop("gds_max_s") * 1e6
            st.dataframe(
                device_display.rename(
                    columns={
                        "device_type": "Device",
                        "corner": "Corner",
                        "temperature_c": "T (°C)",
                        "w_um": "W (µm)",
                        "l_um": "L (µm)",
                        "ion_uA": "ION (µA)",
                        "ioff_nA": "IOFF (nA)",
                        "gm_max_uS": "gm,max (µS)",
                        "gds_max_uS": "gds,max (µS)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            inverter_columns = [
                "corner", "temperature_c", "vdd_v", "voh_v", "vol_v", "vm_v",
                "vil_v", "vih_v", "gain_v_per_v", "nmh_v", "nml_v",
            ]
            st.dataframe(
                inverter_metrics[inverter_columns].rename(
                    columns={
                        "corner": "Corner",
                        "temperature_c": "T (°C)",
                        "vdd_v": "VDD (V)",
                        "voh_v": "VOH (V)",
                        "vol_v": "VOL (V)",
                        "vm_v": "VM (V)",
                        "vil_v": "VIL (V)",
                        "vih_v": "VIH (V)",
                        "gain_v_per_v": "Peak |gain|",
                        "nmh_v": "NMH (V)",
                        "nml_v": "NML (V)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            metric_downloads = st.columns(2)
            metric_downloads[0].download_button(
                "Download device DC metrics",
                data=device_metrics.to_csv(index=False).encode("utf-8"),
                file_name="device_dc_metrics.csv",
                mime="text/csv",
            )
            metric_downloads[1].download_button(
                "Download inverter DC metrics",
                data=inverter_metrics.to_csv(index=False).encode("utf-8"),
                file_name="inverter_dc_metrics.csv",
                mime="text/csv",
            )
        for filename, title in (
            ("device_dc.csv", "Device DC sweeps"),
            ("inverter_dc.csv", "Inverter DC sweeps"),
            ("inverter_transient.csv", "Inverter transient results"),
        ):
            table_path = os.path.join(reference_dir, filename)
            if not os.path.isfile(table_path):
                st.error(f"Manifest-listed dataset is missing: {filename}")
                continue
            frame = pd.read_csv(table_path)
            st.subheader(title)
            st.caption(f"{len(frame):,} rows")
            if filename == "device_dc.csv":
                device_types = sorted(frame["device_type"].dropna().unique())
                device = st.selectbox(
                    "Device", device_types, key="sky130_device_type"
                )
                device_frame = frame[frame["device_type"] == device]
                corners = sorted(device_frame["corner"].dropna().unique())
                corner = st.selectbox(
                    "Process corner",
                    corners,
                    index=corners.index("TT") if "TT" in corners else 0,
                    key="sky130_device_corner",
                )
                corner_frame = device_frame[device_frame["corner"] == corner]
                temperatures = sorted(
                    corner_frame["temperature_c"].dropna().unique()
                )
                temperature = st.selectbox(
                    "Temperature (°C)", temperatures, key="sky130_device_temperature"
                )
                condition_frame = corner_frame[
                    corner_frame["temperature_c"] == temperature
                ]
                widths = sorted(condition_frame["w_um"].dropna().unique())
                width = st.selectbox(
                    "Width (µm)", widths, key="sky130_device_width"
                )
                geometry_frame = condition_frame[condition_frame["w_um"] == width]
                lengths = sorted(geometry_frame["l_um"].dropna().unique())
                length = st.selectbox(
                    "Length (µm)", lengths, key="sky130_device_length"
                )
                bias_frame = geometry_frame[geometry_frame["l_um"] == length]
                drain_biases = sorted(bias_frame["vds_mag_v"].dropna().unique())
                drain_bias = st.selectbox(
                    "VDS / VSD (V)",
                    drain_biases,
                    index=len(drain_biases) - 1,
                    key="sky130_device_vds",
                )
                curve = bias_frame[bias_frame["vds_mag_v"] == drain_bias].sort_values(
                    "vgs_mag_v"
                )
                fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
                ax.semilogy(
                    curve["vgs_mag_v"],
                    curve["ids_a"].clip(lower=1e-30),
                    color="#087E8B",
                    label=f"{device} · {corner} · {temperature:g}°C",
                )
                ax.set(
                    title="SKY130 device DC characteristic",
                    xlabel="VGS / VSG (V)",
                    ylabel="|IDS| (A)",
                )
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            elif filename == "inverter_dc.csv":
                corners = sorted(frame["corner"].dropna().unique())
                corner = st.selectbox(
                    "Process corner",
                    corners,
                    index=corners.index("TT") if "TT" in corners else 0,
                    key="sky130_inverter_corner",
                )
                corner_frame = frame[frame["corner"] == corner]
                temperatures = sorted(
                    corner_frame["temperature_c"].dropna().unique()
                )
                temperature = st.selectbox(
                    "Temperature (°C)", temperatures,
                    key="sky130_inverter_temperature",
                )
                temp_frame = corner_frame[
                    corner_frame["temperature_c"] == temperature
                ]
                supplies = sorted(temp_frame["vdd_v"].dropna().unique())
                supply = st.selectbox(
                    "VDD (V)", supplies, key="sky130_inverter_vdd"
                )
                curve = temp_frame[temp_frame["vdd_v"] == supply].sort_values(
                    "vin_v"
                )
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
                axes[0].plot(curve["vin_v"], curve["vout_v"], color="#087E8B")
                axes[0].plot([0, supply], [supply, 0], "k--", alpha=0.35)
                axes[0].set(
                    title=f"SKY130 inverter VTC · {corner}",
                    xlabel="VIN (V)",
                    ylabel="VOUT (V)",
                )
                axes[1].semilogy(
                    curve["vin_v"],
                    curve["static_power_w"].clip(lower=1e-30),
                    color="#D14D72",
                )
                axes[1].set(
                    title="Static supply power",
                    xlabel="VIN (V)",
                    ylabel="Power (W)",
                )
                for axis in axes:
                    axis.grid(True, which="both", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif filename == "inverter_transient.csv":
                time_columns = {
                    "tphl_s": "tpHL (ps)",
                    "tplh_s": "tpLH (ps)",
                    "trise_s": "Rise time (ps)",
                    "tfall_s": "Fall time (ps)",
                    "average_power_w": "Average power (µW)",
                    "energy_per_period_j": "Energy/period (fJ)",
                }
                display_frame = frame.copy()
                for source_column, label in time_columns.items():
                    scale = 1e6 if source_column == "average_power_w" else (
                        1e15 if source_column == "energy_per_period_j" else 1e12
                    )
                    display_frame[label] = display_frame[source_column] * scale
                st.dataframe(
                    display_frame[list(time_columns.values())],
                    hide_index=True,
                    use_container_width=True,
                )
            st.dataframe(frame.head(500), hide_index=True, use_container_width=True)
            with open(table_path, "rb") as data_file:
                st.download_button(
                    f"Download {filename}",
                    data=data_file.read(),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                )
