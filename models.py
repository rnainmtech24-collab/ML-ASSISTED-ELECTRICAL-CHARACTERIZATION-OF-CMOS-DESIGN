"""
Module 2 / 3 — Model Definitions  (models.py)
==============================================
Defines the shared PyTorch regression architecture used for both
NMOS and PMOS current prediction.

Architecture (as specified):
    Linear(5→64) → Tanh → Linear(64→64) → Tanh
    → Linear(64→32) → Tanh → Linear(32→1)

Tanh is chosen over ReLU because:
  • Transistor IDS curves are smooth and bounded — Tanh mirrors
    that saturation characteristic.
  • Tanh output is centred at 0, which suits StandardScaler-normalised
    targets well.
  • No "dying neuron" problem present with ReLU on negative inputs.
"""

import torch
import torch.nn as nn


class TransistorNet(nn.Module):
    """
    General-purpose regression network for log10(IDS) prediction.

    Parameters
    ----------
    in_features : int
        Number of input features (5 for both NMOS and PMOS).
    """

    def __init__(self, in_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation — good default for Tanh networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── factory helpers ────────────────────────────────────────────────────────
def build_nmos_model() -> TransistorNet:
    """Return a fresh (untrained) NMOS model."""
    return TransistorNet(in_features=5)


def build_pmos_model() -> TransistorNet:
    """Return a fresh (untrained) PMOS model."""
    return TransistorNet(in_features=5)


# ── quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_nmos_model()
    print(m)
    x = torch.randn(8, 5)
    y = m(x)
    print("Output shape:", y.shape)   # (8, 1)