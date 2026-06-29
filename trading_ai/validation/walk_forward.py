"""
Walk-Forward & Out-of-Sample (Modulo 5).

Dividiamo la storia in finestre sequenziali e valutiamo la strategia su ciascuna
porzione FUORI CAMPIONE (i pattern sono stati scoperti su dati precedenti). Una
strategia robusta deve restare profittevole in MOLTE finestre diverse, non solo
nel periodo in cui e' stata trovata. Misuriamo quindi la CONSISTENZA temporale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.strategy_generator import summarize_backtest


def walk_forward(strategy, features: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Backtesta la strategia su `n_splits` finestre temporali consecutive.

    Parametri
    ---------
    strategy : Strategy
        Strategia gia' costruita (Modulo 4), col suo clusterer addestrato.
    features : pd.DataFrame
        Matrice feature completa (Modulo 2), con 'atr' e 'close'.
    n_splits : int
        Numero di finestre out-of-sample in cui dividere la storia.

    Ritorna
    -------
    dict con metriche per-finestra e indici di consistenza.
    """
    n = len(features)
    if n < n_splits * 50:
        raise ValueError("Dati insufficienti per il numero di finestre richiesto.")

    # Tagliamo l'indice in n_splits blocchi contigui di uguale ampiezza.
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    per_window = []
    for w in range(n_splits):
        chunk = features.iloc[bounds[w]:bounds[w + 1]]
        res = strategy.run(chunk)                       # backtest sul blocco
        summ = summarize_backtest(res)
        summ["window"] = w
        per_window.append(summ)

    table = pd.DataFrame(per_window)
    # Consistenza: frazione di finestre (con almeno un trade) chiuse in profitto.
    traded = table[table["n_trades"] > 0]
    if len(traded) == 0:
        consistency = 0.0
        mean_return = 0.0
    else:
        consistency = float((traded["total_return"] > 0).mean())
        mean_return = float(traded["total_return"].mean())

    return {
        "n_splits": n_splits,
        "windows_traded": int(len(traded)),
        "consistency": consistency,          # quanto spesso e' profittevole nel tempo
        "mean_window_return": mean_return,
        "per_window": table,
    }
