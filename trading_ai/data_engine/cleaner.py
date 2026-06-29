"""
Pulizia automatica dei dati (Modulo 1 - parte "Pulisci/Rimuovi corrotti").

Regole applicate, nell'ordine:
  1) Rimozione righe con timestamp non valido (NaT).
  2) Rimozione duplicati di timestamp (teniamo l'ultima occorrenza).
  3) Rimozione righe con OHLC mancanti.
  4) Rimozione candele con prezzi non positivi (impossibili su un mercato reale).
  5) Rimozione candele con OHLC incoerenti (es. high < low, close fuori range).
  6) Rimozione outlier estremi via salto di prezzo improbabile (dati corrotti).

Ogni passo viene loggato con il conteggio delle righe scartate, per piena
tracciabilita' (nessuna pulizia "silenziosa").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Riepilogo di quante righe sono state rimosse e perche'."""

    rows_in: int = 0          # righe in ingresso
    bad_timestamp: int = 0    # righe con timestamp NaT
    duplicates: int = 0       # timestamp duplicati rimossi
    missing_ohlc: int = 0     # righe con valori OHLC mancanti
    non_positive: int = 0     # prezzi <= 0
    inconsistent: int = 0     # relazioni OHLC violate
    outliers: int = 0         # salti di prezzo anomali
    rows_out: int = 0         # righe finali

    def as_dict(self) -> dict:
        """Versione dizionario, comoda per salvarla in un report JSON."""
        return self.__dict__.copy()


def clean(
    df: pd.DataFrame,
    max_return: float = 0.20,   # salto massimo plausibile tra due candele (20%)
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Pulisce un DataFrame OHLCV e restituisce (dati_puliti, report).

    Il parametro `max_return` definisce la soglia oltre la quale un movimento
    di prezzo tra due candele consecutive e' considerato dato corrotto. 0.20
    (20%) e' molto permissivo per timeframe bassi; alzalo/abbassalo secondo
    lo strumento (le crypto sono piu' volatili del Forex).
    """
    report = CleaningReport(rows_in=len(df))  # registriamo le righe iniziali
    df = df.copy()                            # non mutiamo l'input

    # --- 1) Timestamp non validi (NaT) --------------------------------------
    mask_nat = df.index.isna()                # True dove l'indice e' NaT
    report.bad_timestamp = int(mask_nat.sum())
    df = df[~mask_nat]                         # teniamo solo i timestamp validi

    # --- 2) Duplicati di timestamp ------------------------------------------
    # keep='last': in caso di doppione teniamo l'ultimo (di solito il piu' aggiornato).
    dup_mask = df.index.duplicated(keep="last")
    report.duplicates = int(dup_mask.sum())
    df = df[~dup_mask]

    # Ordiniamo per sicurezza dopo le rimozioni.
    df = df.sort_index()

    # --- 3) OHLC mancanti ----------------------------------------------------
    ohlc = ["open", "high", "low", "close"]
    present = [c for c in ohlc if c in df.columns]  # colonne effettivamente presenti
    before = len(df)
    df = df.dropna(subset=present)                   # togliamo righe con NaN in OHLC
    report.missing_ohlc = before - len(df)

    # --- 4) Prezzi non positivi ---------------------------------------------
    # Un prezzo <= 0 e' fisicamente impossibile: dato corrotto.
    before = len(df)
    positive_mask = (df[present] > 0).all(axis=1)    # True se TUTTI gli OHLC > 0
    df = df[positive_mask]
    report.non_positive = before - len(df)

    # --- 5) Coerenza OHLC ----------------------------------------------------
    # Relazioni che DEVONO valere su una candela reale:
    #   high >= max(open, close)  e  low <= min(open, close)  e  high >= low.
    before = len(df)
    valid = (
        (df["high"] >= df["low"]) &
        (df["high"] >= df[["open", "close"]].max(axis=1)) &
        (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    df = df[valid]
    report.inconsistent = before - len(df)

    # --- 6) Outlier: salti di prezzo improbabili ----------------------------
    # Calcoliamo il rendimento semplice della close; se supera max_return in
    # valore assoluto e' quasi certamente un errore di feed (tick "fantasma").
    before = len(df)
    if len(df) > 1:
        ret = df["close"].pct_change()               # variazione % rispetto alla barra prima
        # La prima riga ha NaN (nessun precedente): la consideriamo valida.
        outlier_mask = ret.abs() > max_return
        outlier_mask.iloc[0] = False
        df = df[~outlier_mask]
    report.outliers = before - len(df)

    report.rows_out = len(df)  # righe sopravvissute alla pulizia

    # Log riassuntivo: utile su Kaggle per vedere subito la qualita' del feed.
    logger.info(
        "Pulizia: %d -> %d righe | NaT=%d dup=%d miss=%d neg=%d incoer=%d outlier=%d",
        report.rows_in, report.rows_out, report.bad_timestamp, report.duplicates,
        report.missing_ohlc, report.non_positive, report.inconsistent, report.outliers,
    )
    return df, report
