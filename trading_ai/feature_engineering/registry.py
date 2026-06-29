"""
Registry estensibile delle feature (Modulo 2).

Obiettivo della regola di progetto: "permetti di aggiungere facilmente nuove
feature". Lo otteniamo con un semplice pattern a registro + decoratore:

    from trading_ai.feature_engineering.registry import feature

    @feature("mia_feature", group="custom")
    def mia_feature(df):
        # df ha colonne open/high/low/close/volume e DatetimeIndex
        return (df["close"] - df["open"])   # ritorna una Series o un DataFrame

Da quel momento "mia_feature" e' disponibile nel FeatureEngine senza toccare
altro codice. Ogni funzione riceve il DataFrame OHLCV e ritorna:
  - una pd.Series  -> diventa UNA colonna (nome = nome della feature)
  - un pd.DataFrame -> diventa PIU' colonne (prefissate col nome della feature)
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Tipo di una funzione-feature: prende un DataFrame e ritorna Series o DataFrame.
FeatureFn = Callable[..., "pd.Series | pd.DataFrame"]

# Registro globale: nome -> (funzione, gruppo). Popolato dal decoratore @feature.
FEATURE_REGISTRY: dict[str, tuple[FeatureFn, str]] = {}


def feature(name: str, group: str = "custom") -> Callable[[FeatureFn], FeatureFn]:
    """
    Decoratore che registra una funzione come feature riutilizzabile.

    Parametri
    ---------
    name : str
        Nome univoco della feature (usato come nome/prefisso di colonna).
    group : str
        Categoria logica (es. 'indicator', 'structure', 'candlestick').
        Utile per attivare/disattivare interi gruppi nel FeatureEngine.
    """
    def decorator(fn: FeatureFn) -> FeatureFn:
        if name in FEATURE_REGISTRY:
            # Evitiamo registrazioni doppie silenziose (tipico bug nei notebook).
            raise ValueError(f"Feature '{name}' gia' registrata.")
        FEATURE_REGISTRY[name] = (fn, group)
        return fn
    return decorator


def list_features(group: str | None = None) -> list[str]:
    """Elenca i nomi delle feature registrate, opzionalmente filtrate per gruppo."""
    if group is None:
        return sorted(FEATURE_REGISTRY)
    return sorted(n for n, (_, g) in FEATURE_REGISTRY.items() if g == group)
