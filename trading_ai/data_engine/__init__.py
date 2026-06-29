"""
Modulo 1 - Data Engine
======================

Pipeline completa e INDIPENDENTE per:
  - importare dati storici (Forex, CFD, Metalli, Indici, Commodities)
  - pulire automaticamente e rimuovere dati corrotti
  - aggregare su timeframe multipli (M1...D1)
  - normalizzare il dataset

Uso rapido
----------
    from trading_ai.data_engine import DataEngine

    eng = DataEngine()
    df = eng.load_csv("datasets/EURUSD_M1.csv")   # importa + pulisce
    h1 = eng.to_timeframe(df, "H1")               # resample a H1
    norm = eng.normalize(h1, method="returns")    # rendimenti log

Le singole funzioni restano accessibili per usi avanzati:
    from trading_ai.data_engine import load_csv, clean, resample, Normalizer
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Re-export delle API a basso livello dei sotto-moduli.
from trading_ai.data_engine.cleaner import CleaningReport, clean
from trading_ai.data_engine.loader import load_csv, load_dataframe
from trading_ai.data_engine.normalizer import Normalizer
from trading_ai.data_engine.resampler import resample, resample_many
from trading_ai.data_engine.synthetic import generate_ohlcv
from trading_ai.utils.io import load_parquet, save_parquet
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DataEngine",
    "load_csv", "load_dataframe", "clean", "CleaningReport",
    "resample", "resample_many", "Normalizer", "generate_ohlcv",
    "save_parquet", "load_parquet",
]


class DataEngine:
    """
    Facciata di alto livello che orchestra import -> pulizia -> (resample) ->
    (normalizzazione). Mantiene l'ultimo report di pulizia per ispezione.
    """

    def __init__(self, max_return: float = 0.20):
        # Soglia per il rilevamento outlier passata al cleaner.
        self.max_return = max_return
        # Ultimo report di pulizia generato, utile da loggare nei /reports.
        self.last_report: CleaningReport | None = None

    # --- Import + pulizia in un colpo solo ----------------------------------
    def load_csv(self, path: str | Path, has_header: bool = True,
                 sep: str | None = None, do_clean: bool = True) -> pd.DataFrame:
        """Carica un CSV e (di default) lo pulisce automaticamente."""
        df = load_csv(path, sep=sep, has_header=has_header)
        return self._maybe_clean(df, do_clean)

    def load_dataframe(self, df: pd.DataFrame, do_clean: bool = True) -> pd.DataFrame:
        """Normalizza allo schema canonico un DataFrame in memoria e lo pulisce."""
        df = load_dataframe(df)
        return self._maybe_clean(df, do_clean)

    def _maybe_clean(self, df: pd.DataFrame, do_clean: bool) -> pd.DataFrame:
        """Applica la pulizia se richiesta, memorizzando il report."""
        point_value = df.attrs.get("point_value")      # da preservare oltre la pulizia
        if do_clean:
            df, self.last_report = clean(df, max_return=self.max_return)
        if point_value is not None:                    # le operazioni pandas possono perdere attrs
            df.attrs["point_value"] = point_value
        return df

    # --- Trasformazioni ------------------------------------------------------
    def to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggrega al timeframe richiesto (delega al resampler)."""
        return resample(df, timeframe)

    def to_timeframes(self, df: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
        """Aggrega verso piu' timeframe contemporaneamente."""
        return resample_many(df, timeframes)

    def normalize(self, df: pd.DataFrame, method: str = "returns",
                  columns: list[str] | None = None) -> pd.DataFrame:
        """Normalizza il dataset (default 'returns', il piu' adatto al ML)."""
        return Normalizer(method=method, columns=columns).fit_transform(df)

    # --- Persistenza ---------------------------------------------------------
    def save(self, df: pd.DataFrame, path: str | Path) -> Path:
        """Salva un DataFrame in Parquet (formato consigliato per i dataset)."""
        return save_parquet(df, path)

    def load(self, path: str | Path) -> pd.DataFrame:
        """Carica un DataFrame da Parquet."""
        return load_parquet(path)
