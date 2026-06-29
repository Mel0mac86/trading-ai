"""
Acquisizione automatica dei dati (Modulo 1 - parte "Import autonomo").

L'autopilota deve girare SENZA che l'utente fornisca nulla. Questa funzione
prova in cascata le fonti disponibili e ritorna sempre un dataset:

  1) CSV locali in /datasets (es. export MetaTrader gia' presenti);
  2) download da una fonte online opzionale (yfinance), se installata e la rete
     e' disponibile;
  3) fallback SINTETICO deterministico (sempre disponibile, anche offline).

Ogni dataset e' accompagnato dalla sua "provenienza" (source) per tracciabilita'.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ai.config import PATHS
from trading_ai.data_engine.loader import load_csv
from trading_ai.data_engine.synthetic import generate_ohlcv
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)


def _find_local_csv(symbol: str, datasets_dir: Path) -> Path | None:
    """Cerca in /datasets un CSV il cui nome contiene il simbolo (case-insensitive)."""
    if not datasets_dir.exists():
        return None
    sym = symbol.lower().replace("/", "").replace("=x", "")
    for p in sorted(datasets_dir.glob("*.csv")):
        if sym in p.stem.lower().replace("_", "").replace("-", ""):
            return p
    return None


def _try_download(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """
    Tenta un download via yfinance, se disponibile. Ritorna None in caso di
    libreria assente, errore di rete o dati vuoti (mai solleva: la cascata
    deve poter ripiegare sul sintetico).
    """
    try:
        import yfinance as yf  # import locale: dipendenza OPZIONALE, non richiesta
    except Exception:
        return None
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # yfinance puo' restituire colonne multi-livello: le appiattiamo.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:                              # rete assente, simbolo errato, ecc.
        logger.warning("Download di %s fallito (%s): uso il fallback.", symbol, e)
        return None


def acquire(
    symbol: str = "EURUSD",
    *,
    datasets_dir: Path | None = None,
    allow_download: bool = True,
    download_period: str = "2y",
    download_interval: str = "1h",
    synthetic_bars: int = 200_000,
    synthetic_freq: str = "1min",
) -> tuple[pd.DataFrame, str]:
    """
    Acquisisce i dati per `symbol` in modo autonomo, con fallback garantito.

    Ritorna
    -------
    (df, source) : il DataFrame grezzo e una stringa di provenienza
                   ('csv:<file>', 'yfinance', 'synthetic').
    """
    datasets_dir = datasets_dir or PATHS.datasets

    # 1) CSV locale.
    csv = _find_local_csv(symbol, datasets_dir)
    if csv is not None:
        logger.info("Dati per %s da CSV locale: %s", symbol, csv.name)
        return load_csv(csv), f"csv:{csv.name}"

    # 2) Download opzionale.
    if allow_download:
        df = _try_download(symbol, download_period, download_interval)
        if df is not None:
            logger.info("Dati per %s scaricati (yfinance), %d righe", symbol, len(df))
            return df, "yfinance"

    # 3) Fallback sintetico (deterministico per simbolo, sempre disponibile).
    seed = abs(hash(symbol)) % (2**32 - 1)             # seme stabile-per-sessione dal simbolo
    logger.info("Dati per %s generati sinteticamente (%d barre)", symbol, synthetic_bars)
    df = generate_ohlcv(n=synthetic_bars, freq=synthetic_freq, seed=seed)
    return df, "synthetic"
