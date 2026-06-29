"""
Market structure & feature avanzate (Modulo 2).

Implementa concetti di "Smart Money": Swing High/Low, Break of Structure (BOS),
Change of Character (CHoCH), Fair Value Gap (FVG), Supporti/Resistenze,
Liquidita', Volatilita' e Trend Strength.

NOTA ANTI-LEAKAGE (cruciale): uno swing point e' un massimo/minimo locale che
si "conferma" solo k barre DOPO il pivot. Usare il pivot all'istante del pivot
introdurrebbe look-ahead. Per questo i livelli di struttura usati come feature
sono resi CAUSALI con uno shift di k barre (vengono "conosciuti" solo quando
realmente confermati).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.feature_engineering.indicators import atr
from trading_ai.feature_engineering.registry import feature


# --- Swing points ------------------------------------------------------------
def swing_points(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Identifica swing high/low con la logica del "frattale" a finestra 2k+1.

    Uno swing high e' una barra il cui massimo e' il piu' alto nelle k barre a
    sinistra E nelle k a destra. Le colonne 'swing_high'/'swing_low' marcano il
    pivot (0/1) NELLA SUA POSIZIONE (non causale: serve solo per analisi/plot).
    """
    high, low = df["high"], df["low"]
    win = 2 * k + 1                                     # ampiezza finestra centrata
    # center=True confronta con k barre passate e k future: definizione di pivot.
    is_high = high == high.rolling(win, center=True, min_periods=win).max()
    is_low = low == low.rolling(win, center=True, min_periods=win).min()
    return pd.DataFrame({
        "swing_high": is_high.fillna(False).astype("int8"),
        "swing_low": is_low.fillna(False).astype("int8"),
    })


def causal_levels(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Livelli di struttura CAUSALI: l'ultimo swing high/low gia' CONFERMATO.

    Spostiamo il prezzo del pivot avanti di k barre (istante di conferma) e poi
    ffill: ad ogni barra abbiamo l'ultimo livello realmente noto, senza leakage.
    Ritorna 'last_swing_high' (resistenza) e 'last_swing_low' (supporto).
    """
    sw = swing_points(df, k)
    sh_price = df["high"].where(sw["swing_high"] == 1)   # prezzo solo sui pivot high
    sl_price = df["low"].where(sw["swing_low"] == 1)      # prezzo solo sui pivot low
    # shift(k): il pivot diventa noto k barre dopo; ffill: propaga fino al prossimo.
    last_sh = sh_price.shift(k).ffill()
    last_sl = sl_price.shift(k).ffill()
    return pd.DataFrame({"last_swing_high": last_sh, "last_swing_low": last_sl})


# --- BOS / CHoCH (state machine causale) ------------------------------------
def structure_events(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Calcola Break of Structure (BOS) e Change of Character (CHoCH) in un'unica
    passata O(n) sui livelli causali.

    - BOS  (+1/-1): continuazione. La close rompe l'ultima resistenza (+1) o
      l'ultimo supporto (-1) confermati.
    - CHoCH(+1/-1): primo break in direzione OPPOSTA al trend corrente
      (cambio di carattere del mercato).
    - trend (+1/-1/0): stato della struttura aggiornato dagli eventi.
    """
    lv = causal_levels(df, k)
    close = df["close"].to_numpy()
    sh = lv["last_swing_high"].to_numpy()              # resistenza nota (causale)
    sl = lv["last_swing_low"].to_numpy()               # supporto noto (causale)
    n = len(df)

    bos = np.zeros(n, dtype="int8")
    choch = np.zeros(n, dtype="int8")
    trend = np.zeros(n, dtype="int8")

    cur = 0  # stato trend corrente: 0 neutro, +1 rialzista, -1 ribassista
    for i in range(n):
        broke_up = not np.isnan(sh[i]) and close[i] > sh[i]    # rottura resistenza
        broke_dn = not np.isnan(sl[i]) and close[i] < sl[i]    # rottura supporto
        if broke_up:
            bos[i] = 1
            if cur == -1:               # eravamo ribassisti -> cambio di carattere
                choch[i] = 1
            cur = 1                     # passiamo/confermiamo trend rialzista
        elif broke_dn:
            bos[i] = -1
            if cur == 1:                # eravamo rialzisti -> cambio di carattere
                choch[i] = -1
            cur = -1                    # passiamo/confermiamo trend ribassista
        trend[i] = cur                  # registriamo lo stato corrente
    idx = df.index
    return pd.DataFrame({"bos": bos, "choch": choch, "trend": trend}, index=idx)


# --- Fair Value Gap ----------------------------------------------------------
def fair_value_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap (imbalance) su 3 candele:
      - bullish (+1): low[i] > high[i-2]  (gap lasciato salendo)
      - bearish (-1): high[i] < low[i-2]  (gap lasciato scendendo)
    'fvg' = direzione, 'fvg_size' = ampiezza del gap normalizzata sulla close.
    """
    high, low, close = df["high"], df["low"], df["close"]
    h2, l2 = high.shift(2), low.shift(2)               # candela due barre prima
    bull = low > h2                                     # gap rialzista
    bear = high < l2                                    # gap ribassista
    fvg = pd.Series(0, index=df.index, dtype="int8").mask(bull, 1).mask(bear, -1)
    # Dimensione del gap (solo dove esiste), normalizzata per confrontare strumenti.
    size = pd.Series(0.0, index=df.index)
    size = size.mask(bull, (low - h2) / close).mask(bear, (l2 - high) / close)
    return pd.DataFrame({"fvg": fvg, "fvg_size": size.astype("float32")})


# --- Volatilita' -------------------------------------------------------------
def volatility(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Volatilita' realizzata (deviazione standard dei rendimenti log) e ATR%
    (ATR rapportato al prezzo), entrambe scale-invariant.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))   # rendimenti log
    realized = log_ret.rolling(n, min_periods=n).std(ddof=0)  # vol realizzata
    atr_pct = atr(df["high"], df["low"], df["close"], 14) / df["close"]  # ATR normalizzato
    return pd.DataFrame({"volatility": realized.astype("float32"),
                         "atr_pct": atr_pct.astype("float32")})


# --- Trend strength ----------------------------------------------------------
def trend_strength(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Forza/direzione del trend come pendenza della regressione lineare della
    close sulle ultime n barre, normalizzata per il prezzo (scale-invariant).

    Usiamo la forma chiusa dei minimi quadrati: slope = sum((x-xm)*y) / Sxx,
    con x = posizioni 0..n-1 (Sxx costante). Cosi' e' veloce anche su milioni
    di barre (apply con raw=True su array numpy).
    """
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()                        # varianza non normalizzata di x (costante)
    w = (x - xm) / sxx                                  # pesi della regressione (precalcolati)
    # Per ogni finestra: slope = dot(w, y). raw=True passa array numpy -> veloce.
    slope = df["close"].rolling(n, min_periods=n).apply(lambda y: float(np.dot(w, y)), raw=True)
    return (slope / df["close"]).astype("float32")     # pendenza relativa al prezzo


# --- Supporti/Resistenze & Liquidita' ---------------------------------------
def support_resistance(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Distanza (relativa) dal supporto e dalla resistenza causali piu' vicini,
    piu' flag di 'liquidita'' su massimi/minimi uguali (equal highs/lows), zone
    dove tipicamente si accumulano stop.
    """
    lv = causal_levels(df, k)
    close = df["close"]
    # Distanza percentuale: >0 = quanto manca alla resistenza sopra / supporto sotto.
    dist_res = (lv["last_swing_high"] - close) / close
    dist_sup = (close - lv["last_swing_low"]) / close

    # Equal highs/lows: due swing consecutivi sullo stesso livello entro tolleranza
    # (proporzionale all'ATR) -> pool di liquidita'.
    sw = swing_points(df, k)
    atr_ = atr(df["high"], df["low"], df["close"], 14)
    tol = (0.1 * atr_)                                  # tolleranza dinamica
    sh_price = df["high"].where(sw["swing_high"] == 1).shift(k).ffill()
    prev_sh = sh_price.shift(1)
    sl_price = df["low"].where(sw["swing_low"] == 1).shift(k).ffill()
    prev_sl = sl_price.shift(1)
    eq_highs = ((sh_price - prev_sh).abs() < tol).fillna(False).astype("int8")
    eq_lows = ((sl_price - prev_sl).abs() < tol).fillna(False).astype("int8")

    return pd.DataFrame({
        "dist_resistance": dist_res.astype("float32"),
        "dist_support": dist_sup.astype("float32"),
        "liq_equal_highs": eq_highs,
        "liq_equal_lows": eq_lows,
    })


# --- Registrazione nel registry ---------------------------------------------
@feature("swings", group="structure")
def _f_swings(df: pd.DataFrame) -> pd.DataFrame:
    return swing_points(df, k=2)


@feature("structure", group="structure")
def _f_structure(df: pd.DataFrame) -> pd.DataFrame:
    return structure_events(df, k=2)


@feature("fvg", group="structure")
def _f_fvg(df: pd.DataFrame) -> pd.DataFrame:
    return fair_value_gap(df)


@feature("sr", group="structure")
def _f_sr(df: pd.DataFrame) -> pd.DataFrame:
    return support_resistance(df, k=2)


@feature("volatility", group="volatility")
def _f_vol(df: pd.DataFrame) -> pd.DataFrame:
    return volatility(df)


@feature("trend_strength", group="volatility")
def _f_trend(df: pd.DataFrame) -> pd.Series:
    return trend_strength(df)
