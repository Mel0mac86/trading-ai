"""
Metriche statistiche per singolo pattern (Modulo 3).

Dato l'insieme dei rendimenti futuri delle barre che appartengono a un pattern
(eventualmente moltiplicati per la direzione long/short), calcoliamo le metriche
richieste dal progetto: frequenza, rendimento medio, prob. rialzo/ribasso,
drawdown, profit factor, expectancy, durata media del movimento.

Le metriche sono pure funzioni su array: niente stato, facili da testare.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PatternStats:
    """Contenitore tipizzato delle statistiche di un pattern."""

    count: int                 # numero di occorrenze (trade)
    frequency: float           # occorrenze / totale barre valutabili
    mean_return: float         # rendimento medio per trade (= expectancy)
    median_return: float       # rendimento mediano (robusto agli outlier)
    prob_up: float             # P(rendimento > 0)
    prob_down: float           # P(rendimento < 0)
    profit_factor: float       # somma guadagni / somma |perdite|
    expectancy: float          # valore atteso per trade
    max_drawdown: float        # max drawdown dell'equity dei trade (<=0)
    sharpe_like: float         # media/devstd dei rendimenti (per-trade, non annualizzato)
    avg_duration: float        # durata media del movimento (barre fino al picco)

    def as_dict(self) -> dict:
        return asdict(self)


def equity_max_drawdown(returns: np.ndarray) -> float:
    """
    Max drawdown dell'equity costruita sommando i rendimenti dei trade in ordine.

    Usiamo la somma cumulata (equity additiva in r): semplice e stabile. Il
    drawdown e' la massima distanza sotto il picco precedente (valore <= 0).
    """
    if len(returns) == 0:
        return 0.0
    equity = np.cumsum(returns)                        # curva di equity additiva
    running_max = np.maximum.accumulate(equity)        # massimo corrente (picco)
    drawdown = equity - running_max                    # distanza dal picco (<=0)
    return float(drawdown.min())                       # il piu' profondo


def compute_stats(
    returns: np.ndarray,
    total_bars: int,
    durations: np.ndarray | None = None,
) -> PatternStats:
    """
    Calcola tutte le metriche per un pattern.

    Parametri
    ---------
    returns : np.ndarray
        Rendimenti futuri (gia' orientati secondo la direzione del pattern:
        per uno short si passa -forward_return).
    total_bars : int
        Numero totale di barre valutabili (per la frequenza).
    durations : np.ndarray | None
        Barre fino al picco per ogni trade (per la durata media).
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]              # scartiamo eventuali NaN
    count = len(returns)
    if count == 0:
        # Pattern vuoto: ritorniamo statistiche neutre per non rompere il flusso.
        return PatternStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    gains = returns[returns > 0].sum()                 # somma dei guadagni
    losses = -returns[returns < 0].sum()               # somma dei |perdite| (positiva)
    # profit_factor: inf se non ci sono perdite -> lo cappiamo per evitare inf nei report.
    if losses == 0:
        profit_factor = float("inf") if gains > 0 else 0.0
    else:
        profit_factor = gains / losses

    std = returns.std(ddof=1) if count > 1 else 0.0    # dev.std campionaria
    sharpe_like = returns.mean() / std if std > 0 else 0.0

    avg_duration = float(np.nanmean(durations)) if durations is not None and len(durations) else float("nan")

    return PatternStats(
        count=count,
        frequency=count / total_bars if total_bars else 0.0,
        mean_return=float(returns.mean()),
        median_return=float(np.median(returns)),
        prob_up=float((returns > 0).mean()),
        prob_down=float((returns < 0).mean()),
        profit_factor=float(profit_factor),
        expectancy=float(returns.mean()),
        max_drawdown=equity_max_drawdown(returns),
        sharpe_like=float(sharpe_like),
        avg_duration=avg_duration,
    )
