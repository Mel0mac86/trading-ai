"""
Modulo 2 - Feature Engineering
==============================

Estrae automaticamente un set ricco di feature dai dati OHLCV prodotti dal
Modulo 1: indicatori tecnici, pattern candlestick e market structure.

Tutte le feature sono registrate in un REGISTRY estensibile: aggiungerne una
nuova richiede solo il decoratore @feature, senza modificare il FeatureEngine.

Uso rapido
----------
    from trading_ai.feature_engineering import FeatureEngine

    fe = FeatureEngine()
    feats = fe.transform(df)          # tutte le feature registrate
    feats = fe.transform(df, groups=["indicator"])  # solo un gruppo

Anti-leakage: ogni feature usa solo dati passati/correnti (i livelli di
struttura sono resi causali con uno shift). Le righe iniziali con valori NaN
(periodo di warm-up degli indicatori) possono essere rimosse con dropna=True.
"""

from __future__ import annotations

import pandas as pd

from trading_ai.config import OHLCV_COLUMNS
from trading_ai.feature_engineering.registry import (
    FEATURE_REGISTRY, feature, list_features,
)
from trading_ai.utils.logging import get_logger

# Import dei moduli per ATTIVARE le registrazioni @feature (side-effect import).
from trading_ai.feature_engineering import indicators  # noqa: F401,E402
from trading_ai.feature_engineering import candlestick  # noqa: F401,E402
from trading_ai.feature_engineering import market_structure  # noqa: F401,E402

logger = get_logger(__name__)

__all__ = ["FeatureEngine", "feature", "list_features", "FEATURE_REGISTRY"]


class FeatureEngine:
    """Compositore di feature: applica le funzioni del registry a un DataFrame."""

    def __init__(self, keep_ohlcv: bool = True):
        # keep_ohlcv: se True mantiene le colonne OHLCV originali accanto alle feature.
        self.keep_ohlcv = keep_ohlcv

    def transform(
        self,
        df: pd.DataFrame,
        features: list[str] | None = None,
        groups: list[str] | None = None,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """
        Calcola le feature richieste e le concatena al DataFrame.

        Parametri
        ---------
        df : pd.DataFrame
            OHLCV pulito (output del Modulo 1), con DatetimeIndex.
        features : list[str] | None
            Nomi specifici da calcolare. Se None -> tutte (eventualmente filtrate per gruppo).
        groups : list[str] | None
            Filtra per gruppo ('indicator', 'candlestick', 'structure', 'volatility').
        dropna : bool
            Se True rimuove le righe iniziali di warm-up con NaN.

        Ritorna
        -------
        pd.DataFrame
            OHLCV (se keep_ohlcv) + tutte le colonne-feature.
        """
        self._validate(df)

        # Selezione delle feature da calcolare.
        names = features if features is not None else list(FEATURE_REGISTRY)
        if groups is not None:
            names = [n for n in names if FEATURE_REGISTRY[n][1] in groups]

        # Partiamo dalle colonne OHLCV (o da un frame vuoto con lo stesso indice).
        out = df.copy() if self.keep_ohlcv else pd.DataFrame(index=df.index)

        for name in names:
            fn, _group = FEATURE_REGISTRY[name]
            result = fn(df)                            # ogni funzione riceve l'OHLCV completo
            if isinstance(result, pd.Series):
                out[name] = result                      # Series -> una colonna
            elif isinstance(result, pd.DataFrame):
                # DataFrame -> piu' colonne, prefissate col nome della feature.
                renamed = result.add_prefix(f"{name}_")
                out = pd.concat([out, renamed], axis=1)
            else:
                raise TypeError(
                    f"La feature '{name}' deve ritornare Series o DataFrame, "
                    f"ottenuto {type(result)}."
                )

        n_before = len(out)
        if dropna:
            out = out.dropna()                          # togliamo il warm-up degli indicatori
            logger.info("Feature: %d -> %d righe dopo dropna (%d feature)",
                        n_before, len(out), out.shape[1])
        else:
            logger.info("Feature calcolate: %d colonne, %d righe",
                        out.shape[1], len(out))
        return out

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Verifica che l'input abbia lo schema atteso dal Modulo 1."""
        required = [c for c in OHLCV_COLUMNS if c != "volume"]  # volume opzionale
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonne OHLC mancanti: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Il FeatureEngine richiede un DatetimeIndex.")
