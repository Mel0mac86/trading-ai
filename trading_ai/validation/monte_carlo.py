"""
Monte Carlo sui trade (Modulo 5).

Un singolo backtest e' UNA sola realizzazione della storia: potrebbe essere
stato fortunato (ordine favorevole dei trade). La simulazione Monte Carlo
ricampiona i trade migliaia di volte per stimare la DISTRIBUZIONE di rendimento
finale e drawdown: cosi' vediamo lo scenario tipico, quello sfortunato (coda) e
la probabilita' di restare in profitto.

Ricampioniamo gli R-multiple (profitto in unita' di rischio) e ricostruiamo
l'equity a rischio fisso frazionario, coerentemente col backtester del Modulo 4.
"""

from __future__ import annotations

import numpy as np


def monte_carlo_trades(
    r_multiples: np.ndarray,
    risk_per_trade: float = 0.01,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Esegue il Monte Carlo bootstrap sugli R-multiple dei trade.

    Parametri
    ---------
    r_multiples : np.ndarray
        R-multiple di ogni trade (return / rischio iniziale).
    risk_per_trade : float
        Frazione di equity rischiata per trade (come nel backtester).
    n_sims : int
        Numero di simulazioni (ognuna rimescola/ricampiona i trade).
    seed : int
        Seme per riproducibilita'.

    Ritorna
    -------
    dict con percentili di rendimento finale e drawdown, prob. di profitto.
    """
    r = np.asarray(r_multiples, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        # Nessun trade: ritorniamo un esito neutro/negativo per non passare il filtro.
        return {"n_trades": 0, "prob_profit": 0.0, "final_p05": 0.0,
                "final_p50": 0.0, "final_p95": 0.0, "maxdd_p50": 0.0,
                "maxdd_p95": 0.0}

    rng = np.random.default_rng(seed)
    final_returns = np.empty(n_sims)
    max_drawdowns = np.empty(n_sims)

    for s in range(n_sims):
        # Bootstrap: ricampioniamo n trade CON reinserimento (ordine casuale).
        sample = rng.choice(r, size=n, replace=True)
        # Equity compounded a rischio fisso: e_{k+1} = e_k * (1 + risk% * R_k).
        steps = 1.0 + risk_per_trade * sample
        # Evitiamo equity negativa/zero (un singolo R molto negativo): cappiamo a piccolo positivo.
        steps = np.clip(steps, 1e-6, None)
        equity = np.cumprod(steps)
        final_returns[s] = equity[-1] - 1.0            # rendimento finale relativo
        running_max = np.maximum.accumulate(equity)    # picco corrente
        dd = ((equity - running_max) / running_max).min()  # max drawdown (<=0)
        max_drawdowns[s] = dd

    return {
        "n_trades": n,
        "prob_profit": float((final_returns > 0).mean()),     # quanto spesso si chiude in utile
        "final_p05": float(np.percentile(final_returns, 5)),   # scenario sfortunato
        "final_p50": float(np.percentile(final_returns, 50)),  # scenario tipico (mediano)
        "final_p95": float(np.percentile(final_returns, 95)),  # scenario fortunato
        "maxdd_p50": float(np.percentile(max_drawdowns, 50)),  # drawdown tipico
        "maxdd_p95": float(np.percentile(max_drawdowns, 5)),   # drawdown di coda (peggiore)
    }
