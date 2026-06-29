"""
Labeling degli outcome (Modulo 3).

Per misurare se un pattern "funziona" servono le ETICHETTE: cosa e' successo al
prezzo DOPO l'ingresso. Usiamo l'approccio fixed-horizon (orizzonte fisso di N
barre), robusto e standard. La triple-barrier (TP/SL/tempo) e' prevista come
opzione avanzata nel Modulo 5.

IMPORTANTISSIMO (anti-leakage): le etichette guardano il FUTURO per costruzione
(rendimento a N barre in avanti). Sono usate SOLO per valutare i pattern, MAI
come feature di input al clustering. Le ultime N barre non hanno futuro completo
e vengono marcate NaN (poi scartate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    Rendimento semplice a `horizon` barre in avanti: close[t+h]/close[t] - 1.

    Le ultime `horizon` righe diventano NaN (manca il futuro completo).
    """
    fwd = close.shift(-horizon) / close - 1.0
    return fwd


def forward_excursions(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Per ogni barra t calcola, sulla finestra futura (t+1 ... t+horizon):
      - mfe  : Maximum Favorable Excursion (max rialzo relativo) -> potenziale TP long
      - mae  : Maximum Adverse Excursion (max ribasso relativo)  -> potenziale SL long
      - tt_peak : barre necessarie a toccare il massimo (durata del movimento)

    Utile per stimare drawdown intra-trade e durata media del movimento, e in
    seguito (Modulo 4) per dimensionare SL/TP.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    tt_peak = np.full(n, np.nan)

    # Scorriamo solo le barre che hanno un futuro completo di `horizon` barre.
    for t in range(n - horizon):
        entry = close[t]                               # prezzo d'ingresso
        win_high = high[t + 1: t + 1 + horizon]         # massimi futuri
        win_low = low[t + 1: t + 1 + horizon]           # minimi futuri
        mfe[t] = win_high.max() / entry - 1.0           # massimo guadagno potenziale
        mae[t] = win_low.min() / entry - 1.0            # massima perdita potenziale (<=0 di solito)
        tt_peak[t] = int(np.argmax(win_high)) + 1       # barre fino al picco (>=1)

    return pd.DataFrame(
        {"mfe": mfe, "mae": mae, "tt_peak": tt_peak},
        index=df.index,
    )
