"""
Ottimizzazione automatica dei parametri di rischio (Modulo 7).

Cerca, su una griglia di parametri SL/TP/BE/trailing, la combinazione che
massimizza un OBIETTIVO ROBUSTO. Per ridurre l'overfitting:
  - l'obiettivo penalizza pesantemente il drawdown,
  - le combinazioni con troppi pochi trade sono escluse,
  - la griglia e' volutamente grossolana (niente fine-tuning su rumore).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from trading_ai.strategy_generator import summarize_backtest


def robust_objective(summary: dict, min_trades: int = 30) -> float:
    """
    Obiettivo da massimizzare: premia il rendimento, punisce il drawdown,
    azzera le configurazioni con campione insufficiente.
    """
    if summary["n_trades"] < min_trades:
        return float("-inf")                          # campione troppo piccolo: escluso
    total_return = summary["total_return"]
    dd = abs(summary["max_drawdown"])
    # Rendimento penalizzato dal rischio estremo (simile a un Calmar grezzo).
    return total_return - 3.0 * dd


def optimize_risk(
    strategy,
    features: pd.DataFrame,
    sl_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    tp_grid: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0),
    be_grid: tuple[float, ...] = (0.0, 1.0),
    trail_grid: tuple[float, ...] = (0.0, 1.5),
    min_trades: int = 30,
) -> dict:
    """
    Esplora la griglia di parametri e ritorna la configurazione migliore.

    Ritorna
    -------
    dict con 'best_risk' (RiskParams), 'best_score', 'best_summary' e la
    'table' completa di tutte le combinazioni valutate.
    """
    rows = []
    best_score = float("-inf")
    best_risk = strategy.risk
    best_summary = None

    # product() genera tutte le combinazioni dei valori di griglia.
    for sl, tp, be, tr in product(sl_grid, tp_grid, be_grid, trail_grid):
        if tp <= 0 or sl <= 0:
            continue
        new_risk = replace(strategy.risk, sl_atr=sl, tp_atr=tp, be_atr=be, trail_atr=tr)
        variant = replace(strategy, risk=new_risk)
        summary = summarize_backtest(variant.run(features))
        score = robust_objective(summary, min_trades)
        rows.append({"sl_atr": sl, "tp_atr": tp, "be_atr": be, "trail_atr": tr,
                     "score": score, "total_return": summary["total_return"],
                     "max_drawdown": summary["max_drawdown"],
                     "n_trades": summary["n_trades"]})
        if score > best_score:
            best_score, best_risk, best_summary = score, new_risk, summary

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return {"best_risk": best_risk, "best_score": best_score,
            "best_summary": best_summary, "table": table}
