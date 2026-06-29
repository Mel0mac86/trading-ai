"""
Caricamento e normalizzazione dei dati grezzi (Modulo 1 - parte "Import").

Supporta i formati piu' comuni con cui arrivano i dati di Forex/CFD/Metalli/
Indici/Commodities:
  - CSV esportati da MetaTrader 4/5 (con o senza header)
  - CSV generici con colonne nominate
  - DataFrame gia' in memoria

Indipendentemente dalla sorgente, l'output e' SEMPRE lo stesso schema canonico:
indice DatetimeIndex ordinato + colonne [open, high, low, close, volume].
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ai.config import OHLCV_COLUMNS
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)  # logger dedicato a questo modulo

# Mappa di sinonimi: nomi di colonna che possiamo incontrare -> nome canonico.
# Cosi' accettiamo "Open"/"O"/"<OPEN>" e li riconduciamo tutti a "open".
_COLUMN_ALIASES: dict[str, str] = {
    "o": "open", "open": "open", "<open>": "open",
    "h": "high", "high": "high", "<high>": "high",
    "l": "low", "low": "low", "<low>": "low",
    "c": "close", "close": "close", "<close>": "close",
    "v": "volume", "vol": "volume", "volume": "volume",
    "tickvol": "volume", "<tickvol>": "volume", "<vol>": "volume",
}

# Possibili nomi per le colonne data/ora.
_DATE_ALIASES = {"date", "<date>", "time", "<time>", "datetime", "timestamp"}
_TIME_ALIASES = {"time", "<time>"}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rinomina le colonne ai nomi canonici usando la mappa dei sinonimi."""
    # Abbassiamo a minuscolo e togliamo spazi per un confronto robusto.
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()         # normalizziamo il nome
        if key in _COLUMN_ALIASES:             # se e' un sinonimo noto...
            renamed[col] = _COLUMN_ALIASES[key]  # ...lo mappiamo al canonico
    return df.rename(columns=renamed)          # rename non distrugge le colonne non mappate


def _build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un DatetimeIndex unendo eventuali colonne date+time separate
    (tipico dei CSV MetaTrader) oppure usando un'unica colonna datetime.
    """
    cols_lower = {str(c).strip().lower(): c for c in df.columns}  # mappa minuscolo->originale

    # Caso 1: colonne separate "date" e "time" (MT4/MT5 le esporta cosi').
    if "date" in cols_lower and "time" in cols_lower:
        date_col = cols_lower["date"]
        time_col = cols_lower["time"]
        # Concateniamo "YYYY.MM.DD" + " " + "HH:MM" e lasciamo che pandas inferisca.
        dt = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",  # valori non parsabili diventano NaT (li gestira' il cleaner)
        )
        df = df.drop(columns=[date_col, time_col])  # rimuoviamo le colonne ormai fuse

    else:
        # Caso 2: un'unica colonna data/ora. Cerchiamo il primo alias disponibile.
        dt_col = next((cols_lower[a] for a in _DATE_ALIASES if a in cols_lower), None)
        if dt_col is None:
            raise ValueError(
                "Nessuna colonna data/ora riconosciuta. "
                "Attese una tra: date, time, datetime, timestamp."
            )
        dt = pd.to_datetime(df[dt_col], errors="coerce")  # parsing robusto
        df = df.drop(columns=[dt_col])                    # rimuoviamo la colonna originale

    df.index = pd.DatetimeIndex(dt)  # impostiamo l'indice temporale
    df.index.name = "time"           # nome coerente
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riduce l'uso di memoria convertendo i float a 32 bit.

    Con milioni di candele questo dimezza la RAM richiesta; la precisione a
    32 bit (~7 cifre significative) e' piu' che sufficiente per i prezzi.
    """
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):     # solo colonne float
            df[col] = pd.to_numeric(df[col], downcast="float")  # float64 -> float32
    return df


def load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza un DataFrame gia' in memoria allo schema canonico.

    Utile quando i dati arrivano da un'API o da un altro notebook.
    """
    df = df.copy()                       # non modifichiamo l'input del chiamante
    df = _standardize_columns(df)        # nomi colonna canonici

    # Se l'indice non e' gia' temporale, proviamo a costruirlo dalle colonne.
    if not isinstance(df.index, pd.DatetimeIndex):
        df = _build_datetime_index(df)

    # Teniamo solo le colonne OHLCV presenti (volume puo' mancare).
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    if "close" not in keep:              # senza close non possiamo fare nulla
        raise ValueError("Colonna 'close' mancante: dati non utilizzabili.")
    df = df[keep]                        # selezione finale delle colonne

    df = df.sort_index()                 # garantiamo ordine cronologico crescente
    df = _downcast(df)                   # ottimizzazione memoria
    return df


def load_csv(
    path: str | Path,
    sep: str | None = None,        # separatore: None => pandas lo inferisce
    has_header: bool = True,       # True se il CSV ha una riga di intestazione
) -> pd.DataFrame:
    """
    Carica un file CSV e lo normalizza allo schema canonico.

    Parametri
    ---------
    path : str | Path
        Percorso del CSV (es. esportazione MetaTrader).
    sep : str | None
        Separatore di campo. Se None, pandas prova a inferirlo (engine python).
    has_header : bool
        Se False, assume il formato MT senza header:
        date,time,open,high,low,close,volume.

    Ritorna
    -------
    pd.DataFrame
        Dati normalizzati [open, high, low, close, volume] con indice temporale.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV non trovato: {path}")

    if has_header:
        # sep=None + engine='python' lascia che pandas deduca virgola/tab/;.
        raw = pd.read_csv(path, sep=sep, engine="python")
    else:
        # Formato MetaTrader classico senza intestazione: assegniamo noi i nomi.
        names = ["date", "time", "open", "high", "low", "close", "volume"]
        raw = pd.read_csv(path, sep=sep, engine="python", header=None, names=names)

    logger.info("Caricato CSV %s: %d righe grezze", path.name, len(raw))
    return load_dataframe(raw)  # riusiamo la normalizzazione comune
