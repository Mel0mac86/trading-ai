"""
Metriche di performance per i report (Modulo 9).

Calcola gli indicatori richiesti dal progetto a partire dal risultato di un
backtest (curva equity + trade): Sharpe, Sortino, Calmar, Win Rate, Profit
Factor, Expectancy, Recovery Factor, numero di trade e distribuzione dei
profitti. Le metriche basate sul tempo (Sharpe/Sortino/Calmar) vengono
annualizzate stimando la frequenza delle barre dall'indice temporale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """Stima quante barre ci sono in un anno dalla spaziatura mediana dell'indice."""
    if len(index) < 3:
        return 252.0                                   # fallback: giorni di borsa
    # Differenza temporale mediana tra barre, in secondi.
    deltas = np.diff(index.view("int64")) / 1e9        # nanosecondi -> secondi
    median_sec = float(np.median(deltas))
    if median_sec <= 0:
        return 252.0
    seconds_per_year = 365.0 * 24 * 3600
    return seconds_per_year / median_sec


def _to_periodic_returns(equity: pd.Series) -> pd.Series:
    """Rendimenti periodici (barra su barra) della curva equity."""
    return equity.pct_change().dropna()


def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    """
    Sharpe annualizzato: (rendimento medio - risk free) / volatilita', scalato
    per la radice del numero di periodi annui.
    """
    r = _to_periodic_returns(equity)
    if r.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    ppy = _periods_per_year(equity.index)
    excess = r.mean() - rf / ppy
    return float(excess / r.std(ddof=1) * np.sqrt(ppy))


def sortino_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    """Come lo Sharpe ma penalizza solo la volatilita' NEGATIVA (downside)."""
    r = _to_periodic_returns(equity)
    if len(r) < 2:
        return 0.0
    ppy = _periods_per_year(equity.index)
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    excess = r.mean() - rf / ppy
    return float(excess / dd * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    """Massimo drawdown relativo (frazione del picco, valore <= 0)."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    """Tasso di crescita annuo composto, dato lo span temporale della equity."""
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.0 * 24 * 3600)
    if years <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar = CAGR / |max drawdown|: rendimento per unita' di rischio estremo."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / mdd)


def compute_report_metrics(equity: pd.Series, trade_returns: np.ndarray) -> dict:
    """
    Aggrega tutte le metriche di performance in un dizionario.

    Parametri
    ---------
    equity : pd.Series
        Curva di equity indicizzata per data (output del backtester).
    trade_returns : np.ndarray
        Rendimenti per-trade (per win rate, profit factor, distribuzione).
    """
    tr = np.asarray(trade_returns, dtype=float)
    tr = tr[np.isfinite(tr)]
    n = len(tr)

    gains = tr[tr > 0].sum()
    losses = -tr[tr < 0].sum()
    profit_factor = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    mdd = max_drawdown(equity) if len(equity) else 0.0
    recovery_factor = (total_return / abs(mdd)) if mdd != 0 else 0.0

    return {
        "n_trades": n,
        "total_return": total_return,
        "cagr": cagr(equity) if len(equity) else 0.0,
        "sharpe": sharpe_ratio(equity) if len(equity) else 0.0,
        "sortino": sortino_ratio(equity) if len(equity) else 0.0,
        "calmar": calmar_ratio(equity) if len(equity) else 0.0,
        "max_drawdown": mdd,
        "recovery_factor": float(recovery_factor),
        "win_rate": float((tr > 0).mean()) if n else 0.0,
        "profit_factor": float(profit_factor),
        "expectancy": float(tr.mean()) if n else 0.0,
        "avg_win": float(tr[tr > 0].mean()) if (tr > 0).any() else 0.0,
        "avg_loss": float(tr[tr < 0].mean()) if (tr < 0).any() else 0.0,
        "best_trade": float(tr.max()) if n else 0.0,
        "worst_trade": float(tr.min()) if n else 0.0,
    }
