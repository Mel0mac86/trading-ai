"""
Grafici dei report (Modulo 9).

Genera e salva su file PNG: curva di equity, drawdown (underwater plot) e
distribuzione dei profitti per-trade. Usiamo il backend 'Agg' di matplotlib,
non interattivo: funziona su server/Kaggle/CI senza display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # backend non interattivo (nessuna finestra richiesta)
import matplotlib.pyplot as plt  # noqa: E402  (import dopo use())
import numpy as np
import pandas as pd


def plot_equity(equity: pd.Series, path: str | Path, title: str = "Equity") -> Path:
    """Disegna la curva di equity e la salva come PNG."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)               # liberiamo memoria (importante in loop lunghi)
    return path


def plot_drawdown(equity: pd.Series, path: str | Path) -> Path:
    """Disegna il drawdown relativo nel tempo (area sotto lo zero)."""
    path = Path(path)
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max     # frazione sotto il picco (<=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_profit_distribution(trade_returns: np.ndarray, path: str | Path) -> Path:
    """Istogramma della distribuzione dei rendimenti per-trade."""
    path = Path(path)
    tr = np.asarray(trade_returns, dtype=float)
    tr = tr[np.isfinite(tr)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(tr) > 0:
        ax.hist(tr, bins=min(50, max(10, len(tr) // 5)), color="#2ca02c", alpha=0.7)
        ax.axvline(0, color="black", linewidth=1)     # separa vincenti/perdenti
    ax.set_title("Distribuzione dei profitti per trade")
    ax.set_xlabel("Rendimento per trade")
    ax.set_ylabel("Frequenza")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
