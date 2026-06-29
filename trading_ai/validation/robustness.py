"""
Robustness & Sensitivity (Modulo 5).

Una strategia robusta NON deve dipendere da valori esatti dei parametri: se
funziona solo con SL=2.0*ATR ma crolla con 1.8 o 2.2, e' overfittata sui dati.
Perturbiamo i parametri principali di rischio (SL e TP) su una griglia e
verifichiamo:
  - robustness: quale frazione delle varianti resta profittevole;
  - sensitivity: quanto e' variabile la performance al variare dei parametri
    (deviazione standard relativa: piu' bassa = piu' stabile).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from trading_ai.strategy_generator import summarize_backtest


def parameter_robustness(
    strategy,
    features: pd.DataFrame,
    sl_factors: tuple[float, ...] = (0.75, 1.0, 1.25),
    tp_factors: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> dict:
    """
    Esplora una griglia di perturbazioni di SL/TP attorno ai valori base.

    Ritorna metriche aggregate sulla griglia (frazione profittevole, dispersione
    dei rendimenti) e la tabella completa per ispezione/sensitivity.
    """
    base_sl = strategy.risk.sl_atr
    base_tp = strategy.risk.tp_atr
    rows = []
    for sf in sl_factors:
        for tf in tp_factors:
            # Cloniamo la strategia con SL/TP perturbati (dataclasses.replace).
            new_risk = replace(strategy.risk, sl_atr=base_sl * sf, tp_atr=base_tp * tf)
            variant = replace(strategy, risk=new_risk)
            res = variant.run(features)
            summ = summarize_backtest(res)
            rows.append({"sl_factor": sf, "tp_factor": tf,
                         "total_return": summ["total_return"],
                         "profit_factor": summ["profit_factor"],
                         "n_trades": summ["n_trades"]})

    table = pd.DataFrame(rows)
    returns = table["total_return"].to_numpy()
    # Robustezza: frazione di varianti con rendimento positivo.
    profitable_fraction = float((returns > 0).mean())
    # Sensitivity: deviazione standard dei rendimenti normalizzata (coefficiente
    # di variazione). Valore basso -> performance poco sensibile ai parametri.
    mean_abs = np.abs(returns).mean()
    sensitivity = float(returns.std() / mean_abs) if mean_abs > 0 else float("inf")

    return {
        "profitable_fraction": profitable_fraction,
        "sensitivity": sensitivity,
        "mean_return": float(returns.mean()),
        "grid": table,
    }
