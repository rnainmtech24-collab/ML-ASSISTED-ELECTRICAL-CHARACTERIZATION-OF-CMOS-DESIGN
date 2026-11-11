"""
Module 19 — Boolean Expression CMOS Characterizer  (boolean_characterizer.py)
==============================================================================
Takes any Boolean expression and characterizes the CMOS implementation:

  Inputs accepted:
    "A AND B"
    "NOT (A OR B)"
    "(A AND B) OR (C AND D)"
    "A NAND B"
    "A XOR B"
    "NOT A AND (B OR C)"

Pipeline:
  1. Parse Boolean expression → expression tree
  2. Map to CMOS gate primitives (INV, NAND, NOR, AND, OR, XOR)
  3. Solve VTC for each gate stage using ML models
  4. Propagate signal through gate chain
  5. Compute total delay, power, noise margins
  6. Generate complete visualization

Supported operators (case-insensitive):
  AND, OR, NOT, NAND, NOR, XOR, XNOR
  Parentheses supported for grouping
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

VDD = 1.8

# ── Gate configs (Wn, Wp, L) ───────────────────────────────────────────────
GATE_CFG = {
    "INV":  (0.5e-6, 1.0e-6, 180e-9),
    "NAND": (0.25e-6, 2.0e-6, 180e-9),
    "NOR":  (1.0e-6, 0.5e-6, 180e-9),
    "AND":  (0.25e-6, 2.0e-6, 180e-9),
    "OR":   (1.0e-6, 0.5e-6, 180e-9),
    "XOR":  (1.0e-6, 2.0e-6, 180e-9),
    "XNOR": (1.0e-6, 2.0e-6, 180e-9),
}

# ── AST Node ───────────────────────────────────────────────────────────────
@dataclass
class Node:
    op:    str                        # "AND","OR","NOT","NAND","NOR","XOR","XNOR","VAR"
    left:  Optional["Node"] = None
    right: Optional["Node"] = None
    name:  str = ""                   # variable name if op=="VAR"

    def __repr__(self):
        if self.op == "VAR":
            return self.name
        if self.op == "NOT":
            return f"NOT({self.left})"
        return f"({self.left} {self.op} {self.right})"


# ══════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════
class BooleanParser:
    """
    Recursive-descent parser for Boolean expressions.

    Grammar (precedence low → high):
        expr   := xor_expr  (OR  xor_expr)*
        xor    := and_expr  (XOR and_expr)*
        and    := not_expr  (AND not_expr)*
        not    := NOT not_expr | atom
        atom   := VAR | '(' expr ')'
        VAR    := identifier (NAND / NOR also handled as binary atoms)
    """

    KEYWORDS = {"AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"}

    def __init__(self, text: str):
        # tokenize: keywords, identifiers, parentheses
        token_re = r"XNOR|NAND|NOR|XOR|AND|OR|NOT|[A-Za-z_][A-Za-z0-9_]*|[()]"
        self.tokens = re.findall(token_re, text.upper())
        self.pos    = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise SyntaxError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_or()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"Unexpected token: {self.tokens[self.pos]}")
        return node

    def parse_or(self) -> Node:
        left = self.parse_xor()
        while self.peek() == "OR":
            self.consume("OR")
            right = self.parse_xor()
            left  = Node("OR", left, right)
        return left

    def parse_xor(self) -> Node:
        left = self.parse_and()
        while self.peek() in ("XOR", "XNOR"):
            op    = self.consume()
            right = self.parse_and()
            left  = Node(op, left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume("AND")
            right = self.parse_not()
            left  = Node("AND", left, right)
        return left

    def parse_not(self) -> Node:
        if self.peek() == "NOT":
            self.consume("NOT")
            operand = self.parse_not()
            return Node("NOT", operand)
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            node = self.parse_or()
            self.consume(")")
            return node
        if tok in ("NAND", "NOR"):
            # binary operator used in atom position: NAND A B
            op = self.consume()
            left  = self.parse_atom()
            right = self.parse_atom()
            return Node(op, left, right)
        if tok and tok not in self.KEYWORDS:
            self.consume()
            return Node("VAR", name=tok)
        raise SyntaxError(f"Unexpected token: {tok!r}")


# ══════════════════════════════════════════════════════════════════════════
#  GATE CHAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class GateStage:
    gate:  str          # "INV","NAND","NOR","AND","OR","XOR","XNOR"
    label: str          # human-readable label e.g. "AND(A,B)"
    depth: int = 0      # logic depth from inputs


def ast_to_gates(node: Node, depth: int = 0) -> list[GateStage]:
    """
    Walk AST and collect the ordered list of gate stages needed.
    Returns gates in topological order (inputs first, output last).
    """
    stages = []

    if node.op == "VAR":
        return []

    # recurse into children first
    if node.left:
        stages += ast_to_gates(node.left,  depth + 1)
    if node.right:
        stages += ast_to_gates(node.right, depth + 1)

    # map AST op → CMOS gate
    op_map = {
        "AND":  "AND",
        "OR":   "OR",
        "NOT":  "INV",
        "NAND": "NAND",
        "NOR":  "NOR",
        "XOR":  "XOR",
        "XNOR": "XNOR",
    }
    gate = op_map[node.op]

    # XNOR = XOR + INV
    if gate == "XNOR":
        stages.append(GateStage("XOR", f"XOR({node.left},{node.right})", depth))
        stages.append(GateStage("INV", f"INV(XOR result)", depth + 1))
    else:
        label = (f"{gate}({node.left},{node.right})"
                 if node.right else f"{gate}({node.left})")
        stages.append(GateStage(gate, label, depth))

    return stages


def get_variables(node: Node) -> list[str]:
    """Extract all unique variable names from AST."""
    if node.op == "VAR":
        return [node.name]
    vars_ = []
    if node.left:  vars_ += get_variables(node.left)
    if node.right: vars_ += get_variables(node.right)
    return list(dict.fromkeys(vars_))   # unique, order preserved


# ══════════════════════════════════════════════════════════════════════════
#  VTC CHAIN SOLVER
# ══════════════════════════════════════════════════════════════════════════
def solve_gate_vtc(gate: str, T=27.0):
    """Solve VTC for a single gate, return (vin, vout)."""
    from cmos_solver import solve_vtc
    
    if gate == "AND":
        # AND = NAND + INV
        # Step 1: solve NAND
        Wn, Wp, L = GATE_CFG["NAND"]
        vin, vout_nand = solve_vtc(T=T, Wn=Wn, Ln=L, Wp=Wp, Lp=L)
        # Step 2: solve INV
        Wn2, Wp2, L2 = GATE_CFG["INV"]
        _, vout_inv = solve_vtc(T=T, Wn=Wn2, Ln=L2, Wp=Wp2, Lp=L2)
        # Step 3: chain them — output of NAND feeds INV
        vout = np.interp(vout_nand, vin, vout_inv)
        return vin, vout

    elif gate == "OR":
        # OR = NOR + INV
        Wn, Wp, L = GATE_CFG["NOR"]
        vin, vout_nor = solve_vtc(T=T, Wn=Wn, Ln=L, Wp=Wp, Lp=L)
        Wn2, Wp2, L2 = GATE_CFG["INV"]
        _, vout_inv = solve_vtc(T=T, Wn=Wn2, Ln=L2, Wp=Wp2, Lp=L2)
        vout = np.interp(vout_nor, vin, vout_inv)
        return vin, vout

    elif gate == "XNOR":
        # XNOR = XOR + INV
        Wn, Wp, L = GATE_CFG["XOR"]
        vin, vout_xor = solve_vtc(T=T, Wn=Wn, Ln=L, Wp=Wp, Lp=L)
        Wn2, Wp2, L2 = GATE_CFG["INV"]
        _, vout_inv = solve_vtc(T=T, Wn=Wn2, Ln=L2, Wp=Wp2, Lp=L2)
        vout = np.interp(vout_xor, vin, vout_inv)
        return vin, vout

    else:
        # INV, NAND, NOR, XOR — solve directly
        Wn, Wp, L = GATE_CFG.get(gate, GATE_CFG["INV"])
        vin, vout = solve_vtc(T=T, Wn=Wn, Ln=L, Wp=Wp, Lp=L)
        return vin, vout

def propagate_signal(stages: list[GateStage], vin_input=None, T=27.0):
    """
    Propagate an input ramp through the gate chain.
    Returns list of (vin, vout) per stage.
    """
    from cmos_solver import extract_vtc_params
    vin_arr = np.linspace(0, VDD, 181) if vin_input is None else vin_input

    results = []
    current_vin = vin_arr.copy()

    for stage in stages:
        # solve this gate's VTC on its own grid
        vin_stg, vout_stg = solve_gate_vtc(stage.gate, T)

        # extract params from the gate's own clean VTC
        params = extract_vtc_params(vin_stg, vout_stg)

        # interpolate: map current signal through this gate
        vout_interp = np.interp(current_vin, vin_stg, vout_stg)

        results.append({
            "stage":  stage,
            "vin":    current_vin.copy(),
            "vout":   vout_interp.copy(),
            "params": params,
        })
        current_vin = vout_interp.copy()

    return vin_arr, results

# ══════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════
def compute_circuit_metrics(stages: list[GateStage], stage_results: list,
                             CL=10e-15, f=1e9, T=27.0):
    """
    Compute total delay, power, EDP for the full gate chain.
    """
    from delay_power import propagation_delay, dynamic_power, energy_per_cycle
    from cmos_solver import extract_vtc_params

    total_delay  = 0.0
    total_power  = 0.0
    stage_metrics = []

    for stage, res in zip(stages, stage_results):
        tp = propagation_delay(CL, stage.gate, T)
        P  = dynamic_power(CL, f, stage.gate)
        E  = energy_per_cycle(CL, stage.gate)
        total_delay += tp
        total_power += P
        stage_metrics.append({
            "gate":     stage.gate,
            "label":    stage.label,
            "tp_ps":    tp * 1e12,
            "power_uW": P  * 1e6,
            "VM":       res["params"]["VM"],
            "Gain":     res["params"]["Gain"],
            "NMH":      res["params"]["NMH"],
            "NML":      res["params"]["NML"],
        })

    E_total  = sum(energy_per_cycle(CL, stage.gate) for stage in stages)
    EDP      = E_total * total_delay
    # Pair the final output with the original input sweep. Later-stage input
    # arrays are propagated voltages and can contain duplicates, which are
    # not a valid x-grid for the numerical VTC derivative.
    output_vin = stage_results[0]["vin"]
    output_vout = stage_results[-1]["vout"]
    output_params = extract_vtc_params(output_vin, output_vout)

    return {
        "total_delay_ps":  total_delay * 1e12,
        "total_power_uW":  total_power * 1e6,
        "total_energy_fJ": E_total * 1e15,
        "EDP":             EDP,
        "num_stages":      len(stages),
        "output_vin":      output_vin,
        "output_vout":     output_vout,
        "output_params":   output_params,
        "stage_metrics":   stage_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
def plot_boolean_characterization(expr_str: str, vin_arr, stage_results,
                                   metrics: dict, save=True):
    """Generate complete characterization plot for the Boolean expression."""
    n_stages = len(stage_results)
    if n_stages == 0:
        print("[Plot] No gate stages to plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_stages, 1)))

    # ── Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.suptitle(f"CMOS Characterization: {expr_str}",
                 fontsize=13, fontweight="bold", y=0.98)

    # Grid: top row = VTC per stage, bottom = signal propagation + metrics
    n_cols   = min(n_stages, 4)
    n_rows   = 2 + (1 if n_stages > 4 else 0)
    gs       = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    # ── Row 0: VTC per stage ────────────────────────────────────────────
    for i, res in enumerate(stage_results[:4]):
        ax = fig.add_subplot(gs[0, i % n_cols])
        ax.plot(res["vin"], res["vout"], color=colors[i], lw=2)
        ax.plot([0, VDD], [0, VDD], "k--", lw=0.8, alpha=0.4)
        ax.axvline(res["params"]["VM"], color="green", ls=":", lw=1)
        ax.set_xlim(0, VDD); ax.set_ylim(-0.05, VDD + 0.05)
        ax.set_xlabel("VIN (V)", fontsize=8)
        ax.set_ylabel("VOUT (V)", fontsize=8)
        ax.set_title(f"Stage {i+1}: {res['stage'].gate}\n{res['stage'].label[:20]}",
                     fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # ── Row 1: Signal propagation through chain ─────────────────────────
    ax_sig = fig.add_subplot(gs[1, :])
    ax_sig.plot(vin_arr, vin_arr, "k--", lw=1, alpha=0.4, label="Input ramp")
    for i, res in enumerate(stage_results):
        ax_sig.plot(vin_arr, res["vout"], color=colors[i], lw=1.8,
                    label=f"After {res['stage'].gate} (Stage {i+1})")
    ax_sig.set_xlim(0, VDD); ax_sig.set_ylim(-0.05, VDD + 0.05)
    ax_sig.set_xlabel("Input Voltage VIN (V)")
    ax_sig.set_ylabel("Output Voltage (V)")
    ax_sig.set_title("Signal Propagation Through Gate Chain")
    ax_sig.legend(fontsize=7, loc="upper right", ncol=min(n_stages+1, 4))
    ax_sig.grid(True, alpha=0.3)

    # ── Row 2 (if > 4 stages): remaining VTC plots ──────────────────────
    if n_stages > 4:
        ax_extra = fig.add_subplot(gs[2, :])
        for i, res in enumerate(stage_results[4:], start=4):
            ax_extra.plot(res["vin"], res["vout"], color=colors[i % 10],
                          lw=2, label=f"Stage {i+1}: {res['stage'].gate}")
        ax_extra.set_xlim(0, VDD); ax_extra.set_ylim(-0.05, VDD + 0.05)
        ax_extra.set_xlabel("VIN (V)"); ax_extra.set_ylabel("VOUT (V)")
        ax_extra.set_title("Additional Gate Stages (5+)")
        ax_extra.legend(fontsize=8); ax_extra.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", expr_str)[:40]
        path = os.path.join(PLOT_DIR, f"boolean_{safe_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Boolean characterization → {path}")
        return path
    return fig


def plot_expression_summary(expr_str: str, metrics: dict, save=True):
    """Plot one composite VTC and one set of expression-level parameters."""
    vin = metrics["output_vin"]
    vout = metrics["output_vout"]
    p = metrics["output_params"]
    fig, (ax_vtc, ax_text) = plt.subplots(1, 2, figsize=(12, 5), dpi=300,
                                           gridspec_kw={"width_ratios": [3, 2]})
    ax_vtc.plot(vin, vout, color="#1f77b4", lw=2.5, label="Expression VTC")
    ax_vtc.plot([0, VDD], [0, VDD], "k--", lw=1, alpha=0.5, label="VIN = VOUT")
    for name, color in [("VM", "green"), ("VIL", "orange"), ("VIH", "red")]:
        ax_vtc.axvline(p[name], color=color, ls=":", lw=1.2, label=name)
    ax_vtc.set_xlim(0, VDD); ax_vtc.set_ylim(-0.05, VDD + 0.05)
    ax_vtc.set_xlabel("Expression input sweep (V)"); ax_vtc.set_ylabel("Expression output (V)")
    ax_vtc.set_title("Combined Boolean-Expression VTC")
    ax_vtc.legend(fontsize=8); ax_vtc.grid(True, alpha=0.3)

    ax_text.axis("off")
    rows = [
        ("Logic depth", str(metrics["num_stages"])),
        ("VM", f"{p['VM']:.3f} V"),
        ("Gain", f"{p['Gain']:.2f}"),
        ("NMH / NML", f"{p['NMH']:.3f} / {p['NML']:.3f} V"),
        ("Total delay", f"{metrics['total_delay_ps']:.2f} ps"),
        ("Total power", f"{metrics['total_power_uW']:.3f} µW"),
        ("Total energy", f"{metrics['total_energy_fJ']:.3f} fJ"),
        ("EDP", f"{metrics['EDP']*1e27:.3f} ×10⁻²⁷ J·s"),
    ]
    ax_text.set_title("Expression Parameters", loc="left")
    ax_text.table(cellText=[[a, b] for a, b in rows], colLabels=["Parameter", "Value"],
                  cellLoc="left", colLoc="left", loc="center")
    fig.suptitle(f"CMOS Characterization: {expr_str}", fontweight="bold")
    plt.tight_layout()
    if save:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", expr_str)[:40]
        path = os.path.join(PLOT_DIR, f"boolean_{safe_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Expression characterization -> {path}")
        return path
    return fig


def plot_metrics_summary(expr_str: str, metrics: dict, save=True):
    """Bar charts for per-stage delay and power."""
    sm     = metrics["stage_metrics"]
    labels = [f"S{i+1}:{s['gate']}" for i, s in enumerate(sm)]
    delays = [s["tp_ps"]    for s in sm]
    powers = [s["power_uW"] for s in sm]
    gains  = [s["Gain"]     for s in sm]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sm)))

    for ax, vals, ylabel, title in zip(
        axes,
        [delays, powers, gains],
        ["Delay (ps)", "Power (µW)", "Gain"],
        ["Stage Propagation Delay", "Stage Dynamic Power", "Stage Gain"],
    ):
        bars = ax.bar(labels, vals, color=colors, edgecolor="k", width=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Stage Metrics: {expr_str}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", expr_str)[:40]
        path = os.path.join(PLOT_DIR, f"boolean_metrics_{safe_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Metrics summary → {path}")
        return path
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  MAIN API FUNCTION
# ══════════════════════════════════════════════════════════════════════════
def characterize_boolean(expression: str, T=27.0, CL=10e-15,
                          f=1e9, verbose=True, save_plots=True):
    """
    Full characterization of a Boolean expression in CMOS.

    Parameters
    ----------
    expression : str   e.g. "A AND B OR NOT C"
    T          : float temperature (°C)
    CL         : float load capacitance (F)
    f          : float clock frequency (Hz)
    verbose    : bool  print detailed output
    save_plots : bool  save PNG files

    Returns
    -------
    dict with keys: ast, stages, stage_results, metrics, plot_paths
    """
    print(f"\n{'='*60}")
    print(f"  Boolean Expression: {expression}")
    print(f"{'='*60}")

    # ── 1. Parse ──────────────────────────────────────────────────────────
    try:
        parser = BooleanParser(expression)
        ast    = parser.parse()
    except SyntaxError as e:
        print(f"[ERROR] Parse error: {e}")
        return None

    variables = get_variables(ast)
    print(f"\n  Variables : {', '.join(variables)}")
    print(f"  AST       : {ast}")

    # ── 2. Build gate chain ───────────────────────────────────────────────
    stages = ast_to_gates(ast)
    if not stages:
        print("[ERROR] No gate stages found — check expression.")
        return None

    print(f"\n  Gate Chain ({len(stages)} stages):")
    for i, s in enumerate(stages):
        print(f"    Stage {i+1}: {s.gate:6s}  <- {s.label}")

    # ── 3. Solve VTCs and propagate ───────────────────────────────────────
    print("\n  Solving VTCs ...")
    vin_arr, stage_results = propagate_signal(stages, T=T)

    # ── 4. Compute metrics ────────────────────────────────────────────────
    metrics = compute_circuit_metrics(stages, stage_results, CL=CL, f=f, T=T)

    # ── 5. Print summary ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  {'Stage':<6} {'Gate':<6} {'VM(V)':>7} {'Gain':>7} "
              f"{'NMH(V)':>8} {'NML(V)':>8} {'tp(ps)':>8}")
        print(f"  {'─'*55}")
        for i, sm in enumerate(metrics["stage_metrics"]):
            print(f"  {i+1:<6} {sm['gate']:<6} {sm['VM']:>7.3f} "
                  f"{sm['Gain']:>7.1f} {sm['NMH']:>8.3f} "
                  f"{sm['NML']:>8.3f} {sm['tp_ps']:>8.2f}")
        print(f"  {'─'*55}")
        print(f"\n  ┌─ Circuit Summary ────────────────────────────────┐")
        print(f"  │  Total Stages   : {metrics['num_stages']}")
        print(f"  │  Total Delay    : {metrics['total_delay_ps']:.2f} ps")
        print(f"  │  Total Power    : {metrics['total_power_uW']:.4f} µW")
        print(f"  │  Total Energy   : {metrics['total_energy_fJ']:.4f} fJ")
        print(f"  │  EDP            : {metrics['EDP']*1e27:.4f} ×10⁻²⁷ J·s")
        print(f"  └──────────────────────────────────────────────────┘\n")

    # ── 6. Plots ──────────────────────────────────────────────────────────
    plot_paths = []
    if save_plots:
        p1 = plot_expression_summary(expression, metrics)
        if p1: plot_paths.append(p1)

    return {
        "expression":    expression,
        "ast":           ast,
        "variables":     variables,
        "stages":        stages,
        "stage_results": stage_results,
        "metrics":       metrics,
        "plot_paths":    plot_paths,
    }


# ══════════════════════════════════════════════════════════════════════════
#  STREAMLIT TAB  (call this from streamlit_app.py)
# ══════════════════════════════════════════════════════════════════════════
def streamlit_boolean_tab():
    """Render the Boolean Characterizer tab in Streamlit."""
    import streamlit as st

    st.header("🔣 Boolean Expression CMOS Characterizer")
    st.markdown("""
    Enter any Boolean expression to get full CMOS characterization.
    
    **Supported operators:** `AND`, `OR`, `NOT`, `NAND`, `NOR`, `XOR`, `XNOR`  
    **Parentheses** supported for grouping.
    """)

    # ── Example buttons ──────────────────────────────────────────────────
    st.subheader("Quick Examples")
    ex_cols = st.columns(4)
    examples = [
        "A AND B",
        "NOT (A OR B)",
        "(A AND B) OR C",
        "A XOR B",
        "NOT A AND NOT B",
        "(A AND B) OR (C AND D)",
        "A NAND B",
        "NOT (A AND B AND C)",
    ]
    selected_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i % 4]:
            if st.button(ex, key=f"ex_{i}"):
                selected_example = ex

    st.divider()

    # ── Input ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        default_expr = selected_example if selected_example else "A AND B OR NOT C"
        expr = st.text_input(
            "Boolean Expression",
            value=default_expr,
            placeholder="e.g.  (A AND B) OR (NOT C AND D)",
            help="Use AND, OR, NOT, NAND, NOR, XOR, XNOR and parentheses"
        )
    with col2:
        T_b   = st.number_input("Temp (°C)", -40, 125, 27, key="bool_T")
        CL_b  = st.number_input("CL (fF)",   1, 1000, 10,  key="bool_CL") * 1e-15
        f_b   = st.number_input("f (MHz)",   1, 1000, 500, key="bool_f")  * 1e6

    if st.button("⚡ Characterize", type="primary", key="bool_run"):
        if not expr.strip():
            st.error("Please enter a Boolean expression.")
            return

        with st.spinner("Parsing expression and solving VTCs …"):
            try:
                result = characterize_boolean(
                    expr, T=T_b, CL=CL_b, f=f_b,
                    verbose=False, save_plots=True
                )
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

        if result is None:
            st.error("Could not parse expression. Check syntax.")
            return

        # ── Results ───────────────────────────────────────────────────
        st.success(f"✅ Characterized {result['metrics']['num_stages']} gate stages")

        # Summary metrics
        m = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stages",       str(m["num_stages"]))
        c2.metric("Total Delay",  f"{m['total_delay_ps']:.1f} ps")
        c3.metric("Total Power",  f"{m['total_power_uW']:.3f} µW")
        c4.metric("Total Energy", f"{m['total_energy_fJ']:.3f} fJ")

        p = m["output_params"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expression VM", f"{p['VM']:.3f} V")
        c2.metric("Expression Gain", f"{p['Gain']:.2f}")
        c3.metric("Expression NMH", f"{p['NMH']:.3f} V")
        c4.metric("Expression NML", f"{p['NML']:.3f} V")

        # Gate chain
        st.subheader("Implementation breakdown (reference only)")
        import pandas as pd
        rows = []
        for i, sm in enumerate(m["stage_metrics"]):
            rows.append({
                "Stage":    i + 1,
                "Gate":     sm["gate"],
                "Operation": sm["label"],
                "VM (V)":   round(sm["VM"],  3),
                "Gain":     round(sm["Gain"], 1),
                "NMH (V)":  round(sm["NMH"], 3),
                "NML (V)":  round(sm["NML"], 3),
                "tp (ps)":  round(sm["tp_ps"], 2),
                "Power (µW)": round(sm["power_uW"], 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # AST
        with st.expander("Expression Tree (AST)"):
            st.code(str(result["ast"]))
            st.write(f"**Variables:** {', '.join(result['variables'])}")

        # Plots
        st.subheader("Characterization Plots")
        for path in result["plot_paths"]:
            if os.path.exists(path):
                st.image(path, width="stretch",
                         caption=os.path.basename(path))


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Command-line usage:
    #   python boolean_characterizer.py "A AND B OR NOT C"
    #   python boolean_characterizer.py "(A AND B) OR (C AND D)"

    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
    else:
        # demo expressions
        demo_exprs = [
            "A AND B",
            "NOT (A OR B)",
            "(A AND B) OR (C AND D)",
            "A XOR B",
        ]
        print("No expression provided. Running demos …\n")
        for expr in demo_exprs:
            characterize_boolean(expr)
        sys.exit(0)

    characterize_boolean(expr)
