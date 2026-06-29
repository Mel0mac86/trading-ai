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

import numpy as np
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
    # Spread reale (in PUNTI) esportato da MetaTrader: lo conserviamo per i costi.
    "spread": "spread", "<spread>": "spread",
}

# Formati datetime piu' comuni negli export (MetaTrader usa il punto come
# separatore di data). Specificarli esplicitamente e' ANCHE una questione di
# performance: senza format, pandas ricade sul parser per-elemento (dateutil),
# lentissimo su milioni di candele.
_DATETIME_FORMATS = [
    "%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%Y%m%d %H%M%S", "%Y%m%d %H%M",   # formato ASCII HistData: 'YYYYMMDD HHMMSS'
]


def _parse_datetime(s: pd.Series) -> pd.Series:
    """
    Converte una serie di stringhe in datetime provando prima i formati noti
    (veloce e robusto), con fallback al parser generico solo se necessario.
    """
    s = s.astype(str).str.strip()
    for fmt in _DATETIME_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.99:       # formato indovinato (quasi tutto valido)
            return parsed
    # Ultima spiaggia: parser generico (lento ma flessibile).
    return pd.to_datetime(s, errors="coerce")

# Possibili nomi per le colonne data/ora.
_DATE_ALIASES = {"date", "<date>", "time", "<time>", "datetime", "timestamp"}
_TIME_ALIASES = {"time", "<time>"}


# Token tipici di una riga d'intestazione: se compaiono nella prima riga,
# il file HA l'header; altrimenti e' un export "grezzo" (HistData/MT senza header).
_HEADER_TOKENS = {
    "date", "time", "datetime", "timestamp", "open", "high", "low", "close",
    "volume", "vol", "tickvol", "spread", "o", "h", "l", "c", "v",
}


def _detect_header(path: Path) -> bool:
    """
    Determina se la prima riga del CSV e' un'intestazione, esaminandone i token.
    Robusto sia agli export con header sia a quelli senza (HistData, MT grezzo).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    if not first:
        return False
    # Rileviamo il separatore piu' frequente tra quelli comuni.
    counts = {d: first.count(d) for d in [",", ";", "\t", "|"]}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    tokens = [t.strip().strip('"<>').lower() for t in first.split(sep)]
    return any(t in _HEADER_TOKENS for t in tokens)


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
        # Concateniamo "YYYY.MM.DD" + " " + "HH:MM" e usiamo i formati noti.
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        dt = _parse_datetime(combined)
        df = df.drop(columns=[date_col, time_col])  # rimuoviamo le colonne ormai fuse

    else:
        # Caso 2: un'unica colonna data/ora. Cerchiamo il primo alias disponibile.
        dt_col = next((cols_lower[a] for a in _DATE_ALIASES if a in cols_lower), None)
        if dt_col is None:
            raise ValueError(
                "Nessuna colonna data/ora riconosciuta. "
                "Attese una tra: date, time, datetime, timestamp."
            )
        dt = _parse_datetime(df[dt_col])                  # parsing robusto e veloce
        df = df.drop(columns=[dt_col])                    # rimuoviamo la colonna originale

    df.index = pd.DatetimeIndex(dt)  # impostiamo l'indice temporale
    df.index.name = "time"           # nome coerente
    return df


def _infer_point_value(close: pd.Series) -> float:
    """
    Inferisce il "valore del punto" (granularita' di prezzo) dal numero di
    decimali del prezzo: 0.001 per 3 decimali (oro), 0.00001 per 5 (FX), ecc.

    Calcolato su float64 PRIMA del downcast (a float32 introdurrebbe rumore).
    Cerca il minor numero di decimali che rappresenta fedelmente i prezzi.
    """
    arr = close.dropna().astype(float).to_numpy()
    if arr.size == 0:
        return 1e-5                                    # default prudente (FX 5 decimali)
    sample = arr[: min(20000, arr.size)]               # un campione basta
    scale = np.maximum(1.0, np.abs(sample))            # tolleranza relativa alla magnitudine
    # Tolleranza 1e-6 relativa: assorbe il rumore del float32 (~1e-7 relativo)
    # senza confondere decimali realmente distinti.
    for d in range(0, 9):                              # da 0 a 8 decimali
        if np.all(np.abs(sample - np.round(sample, d)) <= 1e-6 * scale):
            return float(10.0 ** (-d))
    return 1e-5


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

    # Teniamo le colonne OHLCV presenti (volume puo' mancare) piu' lo 'spread'
    # reale se esportato: ci serve per modellare i costi di transazione veri.
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    if "close" not in keep:              # senza close non possiamo fare nulla
        raise ValueError("Colonna 'close' mancante: dati non utilizzabili.")
    if "spread" in df.columns:
        keep = keep + ["spread"]
    df = df[keep]                        # selezione finale delle colonne

    df = df.sort_index()                 # garantiamo ordine cronologico crescente
    # Inferiamo il valore del punto su float64 (prima del downcast). Se il df
    # era gia' stato caricato (attrs presente), riusiamo quel valore: evita di
    # re-inferire su float32 (rumoroso) in caso di doppia normalizzazione.
    point_value = df.attrs.get("point_value") or _infer_point_value(df["close"])
    df = _downcast(df)                   # ottimizzazione memoria
    df.attrs["point_value"] = point_value  # propaghiamo il valore del punto (per i costi reali)
    return df


# Nomi di colonna per gli export SENZA intestazione, in base al numero di campi.
# 7: date,time,O,H,L,C,V (MT)  |  6: datetime,O,H,L,C,V (ASCII)  |  5: datetime,O,H,L,C
_NO_HEADER_NAMES = {
    8: ["date", "time", "open", "high", "low", "close", "volume", "spread"],
    7: ["date", "time", "open", "high", "low", "close", "volume"],
    6: ["datetime", "open", "high", "low", "close", "volume"],
    5: ["datetime", "open", "high", "low", "close"],
}


def load_csv(
    path: str | Path,
    sep: str | None = None,        # separatore: None => pandas lo inferisce
    has_header: bool | None = None,  # None => rileva automaticamente l'intestazione
) -> pd.DataFrame:
    """
    Carica un file CSV e lo normalizza allo schema canonico.

    Riconosce automaticamente:
      - presenza/assenza dell'intestazione (auto se has_header=None);
      - separatore (virgola/tab/';');
      - export MT (date,time,...) e ASCII HistData (datetime unico 'YYYYMMDD HHMMSS').

    Ritorna un DataFrame [open, high, low, close, volume(, spread)] con indice
    temporale.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV non trovato: {path}")

    if has_header is None:                 # auto-rilevamento robusto
        has_header = _detect_header(path)

    if has_header:
        # sep=None + engine='python' lascia che pandas deduca virgola/tab/;.
        raw = pd.read_csv(path, sep=sep, engine="python")
    else:
        # Senza intestazione: leggiamo, contiamo le colonne e assegniamo i nomi.
        raw = pd.read_csv(path, sep=sep, engine="python", header=None)
        names = _NO_HEADER_NAMES.get(raw.shape[1])
        if names is None:                  # numero di colonne inatteso: fallback ai primi 5+
            base = ["datetime", "open", "high", "low", "close", "volume", "spread"]
            names = base[: raw.shape[1]]
        raw.columns = names

    logger.info("Caricato CSV %s: %d righe grezze (header=%s)",
                path.name, len(raw), has_header)
    return load_dataframe(raw)  # riusiamo la normalizzazione comune
