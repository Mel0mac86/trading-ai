"""
Indicatori tecnici classici (Modulo 2).

Implementati in pandas/numpy puro con le formule STANDARD di letteratura
(in particolare lo smoothing di Wilder per RSI/ATR/ADX). Sono validati dai
test automatici. Tutti gli indicatori usano solo dati passati o correnti:
nessun look-ahead, quindi niente data leakage.

Ogni funzione e' anche registrata nel registry tramite @feature, cosi' il
FeatureEngine puo' comporle automaticamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.feature_engineering.registry import feature


# --- Medie mobili ------------------------------------------------------------
def sma(close: pd.Series, n: int = 20) -> pd.Series:
    """Simple Moving Average: media aritmetica delle ultime n chiusure."""
    return close.rolling(window=n, min_periods=n).mean()


def ema(close: pd.Series, n: int = 20) -> pd.Series:
    """
    Exponential Moving Average: media pesata che da' piu' peso ai dati recenti.
    adjust=False replica la formula ricorsiva classica usata dai trader/MT.
    """
    return close.ewm(span=n, adjust=False).mean()


# --- RSI (Relative Strength Index) ------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI di Wilder: oscillatore 0-100 della forza relativa dei rialzi sui ribassi.

    Passi: variazioni -> separa guadagni/perdite -> media esponenziale di Wilder
    (alpha = 1/n) -> RS = avg_gain/avg_loss -> RSI = 100 - 100/(1+RS).
    """
    delta = close.diff()                              # variazione rispetto alla barra prima
    gain = delta.clip(lower=0.0)                       # solo i movimenti positivi
    loss = -delta.clip(upper=0.0)                      # valore assoluto dei movimenti negativi
    # Smoothing di Wilder = EMA con alpha=1/n (non span). min_periods=n per stabilita'.
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)      # evitiamo divisione per zero
    out = 100.0 - (100.0 / (1.0 + rs))                 # formula RSI
    # Se avg_loss=0 (solo rialzi) RSI->100; se avg_gain=0 (solo ribassi) RSI->0.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, 0.0)
    return out


# --- ATR (Average True Range) -----------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: massima escursione considerando anche il gap dalla barra prima."""
    prev_close = close.shift(1)                        # chiusura precedente
    # TR = max delle tre ampiezze: range barra, gap up, gap down.
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR di Wilder: media esponenziale (alpha=1/n) del True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# --- MACD --------------------------------------------------------------------
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD: differenza tra EMA veloce e lenta, piu' linea segnale e istogramma.
    Ritorna un DataFrame con colonne 'line', 'signal', 'hist'.
    """
    macd_line = ema(close, fast) - ema(close, slow)    # momentum di trend
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # EMA del MACD
    hist = macd_line - signal_line                      # istogramma (accelerazione)
    return pd.DataFrame({"line": macd_line, "signal": signal_line, "hist": hist})


# --- ADX (Average Directional Index) ----------------------------------------
def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """
    ADX di Wilder: misura la FORZA del trend (non la direzione).
    Ritorna DataFrame con 'plus_di', 'minus_di', 'adx'.
    """
    up_move = high.diff()                               # variazione dei massimi
    down_move = -low.diff()                             # variazione dei minimi (segno invertito)
    # +DM attivo solo se il movimento verso l'alto domina ed e' positivo.
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)                   # True Range come base
    atr_ = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()  # ATR smoothed
    # Directional Indicators in percentuale dell'ATR.
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    # DX = quanto i due DI divergono; ADX = sua media smoothed.
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_})


# --- Bollinger Bands ---------------------------------------------------------
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    Bande di Bollinger: media mobile +/- k deviazioni standard.
    Ritorna 'mid', 'upper', 'lower', 'width' (ampiezza normalizzata) e
    'pctb' (%B: posizione del prezzo dentro le bande, util per ML).
    """
    mid = sma(close, n)                                 # banda centrale = SMA
    std = close.rolling(window=n, min_periods=n).std(ddof=0)  # deviazione standard
    upper = mid + k * std                               # banda superiore
    lower = mid - k * std                               # banda inferiore
    width = (upper - lower) / mid                        # ampiezza relativa (volatilita')
    pctb = (close - lower) / (upper - lower)             # %B in [0,1] tipicamente
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower,
                         "width": width, "pctb": pctb})


# --- VWAP (Volume Weighted Average Price) -----------------------------------
def vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP con reset giornaliero (convenzione intraday standard).

    Prezzo tipico = (H+L+C)/3, pesato per il volume e accumulato dentro ogni
    giornata. Se il volume manca, ripieghiamo su un VWAP non pesato (= media
    cumulata del prezzo tipico) per non rompere la pipeline.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    # Pesi = volume; ma se il volume manca O e' tutto zero (tipico dei dati
    # HistData/Forex), ripieghiamo su pesi uniformi per non produrre tutti NaN.
    if "volume" in df.columns and float(df["volume"].fillna(0.0).abs().sum()) > 0.0:
        vol = df["volume"].fillna(0.0)
    else:
        vol = pd.Series(1.0, index=df.index)            # peso uniforme se niente volume
    day = df.index.normalize()                          # chiave di raggruppamento per giorno
    # Somme cumulate resettate ogni giorno tramite groupby.cumsum().
    cum_pv = (typical * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum().replace(0.0, np.nan)
    return cum_pv / cum_v


# ---------------------------------------------------------------------------
# Registrazione nel registry: ogni feature riceve il DataFrame OHLCV completo.
# ---------------------------------------------------------------------------
@feature("sma", group="indicator")
def _f_sma(df: pd.DataFrame) -> pd.DataFrame:
    # Piu' periodi comuni in un colpo solo -> colonne sma_10, sma_20, sma_50.
    return pd.DataFrame({f"sma_{n}": sma(df["close"], n) for n in (10, 20, 50)})


@feature("ema", group="indicator")
def _f_ema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"ema_{n}": ema(df["close"], n) for n in (10, 20, 50)})


@feature("rsi", group="indicator")
def _f_rsi(df: pd.DataFrame) -> pd.Series:
    return rsi(df["close"], 14)


@feature("atr", group="indicator")
def _f_atr(df: pd.DataFrame) -> pd.Series:
    return atr(df["high"], df["low"], df["close"], 14)


@feature("macd", group="indicator")
def _f_macd(df: pd.DataFrame) -> pd.DataFrame:
    return macd(df["close"])


@feature("adx", group="indicator")
def _f_adx(df: pd.DataFrame) -> pd.DataFrame:
    return adx(df["high"], df["low"], df["close"], 14)


@feature("bollinger", group="indicator")
def _f_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    return bollinger(df["close"])


@feature("vwap", group="indicator")
def _f_vwap(df: pd.DataFrame) -> pd.Series:
    return vwap(df)
