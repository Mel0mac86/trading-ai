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


def _find_local_files(symbol: str, datasets_dir: Path) -> list[Path]:
    """
    Cerca in /datasets TUTTI i file (CSV/TXT) il cui nome contiene il simbolo.
    Ritorna la lista ordinata: cosi' piu' file annuali dello stesso strumento
    vengono trovati e poi fusi in un'unica serie continua.
    """
    if not datasets_dir.exists():
        return []
    sym = symbol.lower().replace("/", "").replace("=x", "")
    found = []
    for pattern in ("*.csv", "*.txt"):
        for p in datasets_dir.glob(pattern):
            stem = p.stem.lower().replace("_", "").replace("-", "")
            if sym in stem:
                found.append(p)
    return sorted(found)


def _load_and_merge(files: list[Path]) -> pd.DataFrame:
    """Carica e fonde piu' file in un'unica serie ordinata e deduplicata."""
    frames = [load_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]         # rimuoviamo eventuali sovrapposizioni
    # Conserviamo il point_value (le operazioni pandas possono perdere attrs).
    pv = next((f.attrs.get("point_value") for f in frames if f.attrs.get("point_value")), None)
    if pv:
        df.attrs["point_value"] = pv
    return df


def _try_kaggle_dataset(slug: str, dest: Path) -> bool:
    """
    Scarica (una volta) un dataset Kaggle nella cartella `dest`, se la libreria
    kaggle e' installata e le credenziali sono disponibili (KAGGLE_API_TOKEN o
    ~/.kaggle). Ritorna True se i file sono pronti, False in caso di problemi
    (la cascata ripieghera' su altre fonti). Non scarica due volte: se `dest`
    contiene gia' dei CSV, li riusa.
    """
    if any(dest.glob("*.csv")) if dest.exists() else False:
        return True                                    # gia' scaricato in precedenza (cache)
    try:
        import kaggle  # dipendenza OPZIONALE; l'import autentica leggendo le credenziali
    except Exception:
        return False
    try:
        dest.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=True)
        return any(dest.glob("*.csv"))
    except Exception as e:                              # creds assenti, slug errato, rete...
        logger.warning("Download Kaggle '%s' fallito (%s).", slug, e)
        return False


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
    kaggle_dataset: str | None = None,   # slug 'owner/dataset' da cui scaricare i dati
    download_period: str = "2y",
    download_interval: str = "1h",
    synthetic_bars: int = 200_000,
    synthetic_freq: str = "1min",
) -> tuple[pd.DataFrame, str]:
    """
    Acquisisce i dati per `symbol` in modo autonomo, con fallback garantito.

    Ordine: file locali -> dataset Kaggle (se configurato) -> download yfinance
    -> sintetico. Ritorna (df, source) dove source indica la provenienza.
    """
    datasets_dir = datasets_dir or PATHS.datasets

    # 1) File locali (uno o piu' anni dello stesso strumento).
    files = _find_local_files(symbol, datasets_dir)
    src_parts = ["locale"] if files else []

    # 1b) Se e' configurato un dataset Kaggle, lo scarichiamo (cache sotto
    # datasets/_kaggle/<slug>) e UNIAMO i suoi file a quelli locali: cosi' la
    # storia e' completa anche se un anno e' solo nel repo e gli altri su Kaggle.
    if kaggle_dataset:
        cache = datasets_dir / "_kaggle" / kaggle_dataset.replace("/", "_")
        if _try_kaggle_dataset(kaggle_dataset, cache):
            kfiles = _find_local_files(symbol, cache)
            if kfiles:
                files = files + kfiles
                src_parts.append("kaggle")

    if files:
        names = ", ".join(p.name for p in files)
        src = "+".join(src_parts) or "csv"
        logger.info("Dati per %s da %d file (%s): %s", symbol, len(files), src, names)
        df = _load_and_merge(files) if len(files) > 1 else load_csv(files[0])
        suffix = files[0].name if len(files) == 1 else f"{len(files)} file"
        return df, f"{src}:{suffix}"

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
