"""
Pattern candlestick (Modulo 2).

Riconoscimento VETTORIALE (niente loop) dei pattern piu' noti. Ogni pattern e'
una colonna 0/1 (assente/presente) o -1/0/+1 quando ha una direzione.
Calcolati solo su dati passati/correnti: nessun look-ahead.

I corpi/ombre sono misurati in modo relativo al range della candela, cosi' i
pattern sono indipendenti dal livello assoluto di prezzo dello strumento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.feature_engineering.registry import feature


def _anatomy(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Scompone ogni candela in: corpo, ombra alta, ombra bassa, range totale."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0.0, np.nan)                 # range totale (evita /0 su candele piatte)
    body = (c - o)                                      # corpo con segno (+ rialzista)
    upper = h - np.maximum(o, c)                        # ombra superiore
    lower = np.minimum(o, c) - l                        # ombra inferiore
    return {"o": o, "h": h, "l": l, "c": c, "rng": rng,
            "body": body, "abs_body": body.abs(),
            "upper": upper, "lower": lower}


def doji(df: pd.DataFrame, body_frac: float = 0.1) -> pd.Series:
    """Doji: corpo minuscolo rispetto al range (indecisione del mercato)."""
    a = _anatomy(df)
    return ((a["abs_body"] / a["rng"]) < body_frac).astype("int8")


def hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: piccolo corpo in alto, lunga ombra inferiore (>=2x corpo), poca
    ombra superiore. Segnale di possibile inversione rialzista.
    """
    a = _anatomy(df)
    cond = (
        (a["lower"] >= 2.0 * a["abs_body"]) &          # ombra inferiore lunga
        (a["upper"] <= a["abs_body"]) &                 # ombra superiore corta
        (a["abs_body"] / a["rng"] < 0.4)                # corpo non troppo grande
    )
    return cond.fillna(False).astype("int8")


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: speculare all'hammer, lunga ombra superiore (ribassista)."""
    a = _anatomy(df)
    cond = (
        (a["upper"] >= 2.0 * a["abs_body"]) &
        (a["lower"] <= a["abs_body"]) &
        (a["abs_body"] / a["rng"] < 0.4)
    )
    return cond.fillna(False).astype("int8")


def marubozu(df: pd.DataFrame, body_frac: float = 0.9) -> pd.Series:
    """Marubozu: corpo che occupa quasi tutto il range. +1 rialzista, -1 ribassista."""
    a = _anatomy(df)
    big = (a["abs_body"] / a["rng"]) > body_frac
    direction = np.sign(a["body"])                      # +1 / -1 a seconda del colore
    return (big.fillna(False).astype("int8") * direction).astype("int8")


def engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Engulfing: il corpo corrente "ingloba" quello precedente con colore opposto.
    +1 = bullish engulfing, -1 = bearish engulfing, 0 = nessuno.
    """
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)  # verde ingloba rosso
    bear = (c < o) & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)  # rosso ingloba verde
    out = pd.Series(0, index=df.index, dtype="int8")
    out = out.mask(bull, 1).mask(bear, -1)
    return out


# --- Registrazione -----------------------------------------------------------
@feature("cdl_doji", group="candlestick")
def _f_doji(df: pd.DataFrame) -> pd.Series:
    return doji(df)


@feature("cdl_hammer", group="candlestick")
def _f_hammer(df: pd.DataFrame) -> pd.Series:
    return hammer(df)


@feature("cdl_shooting_star", group="candlestick")
def _f_star(df: pd.DataFrame) -> pd.Series:
    return shooting_star(df)


@feature("cdl_marubozu", group="candlestick")
def _f_marubozu(df: pd.DataFrame) -> pd.Series:
    return marubozu(df)


@feature("cdl_engulfing", group="candlestick")
def _f_engulfing(df: pd.DataFrame) -> pd.Series:
    return engulfing(df)
