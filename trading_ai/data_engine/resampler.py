"""
Resampling multi-timeframe (Modulo 1 - parte "Timeframe multipli").

Dato un dataset in un timeframe base (di solito M1), aggreghiamo correttamente
le candele verso timeframe superiori (M5, M15, M30, H1, H4, D1).

L'aggregazione OHLCV segue le regole standard del trading:
  open   -> primo valore del periodo
  high   -> massimo del periodo
  low    -> minimo del periodo
  close  -> ultimo valore del periodo
  volume -> somma del periodo
"""

from __future__ import annotations

import pandas as pd

from trading_ai.config import TIMEFRAME_MINUTES
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

# Regole di aggregazione passate a pandas .agg(). Definite una volta sola.
_AGG_RULES: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggrega un DataFrame OHLCV al timeframe richiesto.

    Parametri
    ---------
    df : pd.DataFrame
        Dati nel timeframe base, con DatetimeIndex ordinato.
    timeframe : str
        Uno tra le chiavi di TIMEFRAME_MINUTES (M1, M5, M15, M30, H1, H4, D1).

    Ritorna
    -------
    pd.DataFrame
        Candele aggregate al nuovo timeframe, senza periodi vuoti.
    """
    if timeframe not in TIMEFRAME_MINUTES:
        # Errore esplicito: meglio fallire subito che produrre dati sbagliati.
        raise ValueError(
            f"Timeframe '{timeframe}' non supportato. "
            f"Validi: {list(TIMEFRAME_MINUTES)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Il resampling richiede un DatetimeIndex.")

    minutes = TIMEFRAME_MINUTES[timeframe]   # minuti per barra del target
    rule = f"{minutes}min"                    # stringa frequenza per pandas (es. '60min')

    # Usiamo solo le regole per le colonne effettivamente presenti
    # (il volume potrebbe mancare su alcuni feed).
    agg = {c: r for c, r in _AGG_RULES.items() if c in df.columns}

    # label='left'/closed='left': la barra e' etichettata con l'inizio del
    # periodo e include [inizio, fine) -> convenzione standard MetaTrader.
    out = df.resample(rule, label="left", closed="left").agg(agg)

    # Il resample crea righe anche per i periodi senza scambi (es. weekend):
    # le rimuoviamo eliminando le barre dove la close e' NaN.
    out = out.dropna(subset=["close"])

    logger.info("Resample -> %s: %d barre", timeframe, len(out))
    return out


def resample_many(df: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """
    Comodita': resampla verso piu' timeframe e ritorna un dizionario
    {timeframe: DataFrame}. Pratico per preparare in un colpo solo i dati
    multi-timeframe usati dal Modulo 2 (Feature Engineering).
    """
    return {tf: resample(df, tf) for tf in timeframes}
