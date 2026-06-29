"""
Normalizzazione del dataset (Modulo 1 - parte "Normalizza").

ATTENZIONE al data leakage (regola di progetto): la normalizzazione deve
imparare i suoi parametri (media, std, min, max) SOLO dai dati di training e
applicarli poi a validation/test. Per questo offriamo un'interfaccia in stile
scikit-learn: fit() impara, transform() applica.

Metodi disponibili:
  - 'zscore'  : (x - media) / std            -> distribuzione centrata, std 1
  - 'minmax'  : (x - min) / (max - min)      -> valori in [0, 1]
  - 'returns' : rendimenti log delle close   -> serie stazionaria per il ML

Per il trading la trasformazione piu' utile e' spesso 'returns': rende la
serie stazionaria (i livelli di prezzo assoluti non sono comparabili nel tempo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Normalizer:
    """Normalizzatore con interfaccia fit/transform anti-leakage."""

    def __init__(self, method: str = "zscore", columns: list[str] | None = None):
        """
        Parametri
        ---------
        method : str
            'zscore', 'minmax' o 'returns'.
        columns : list[str] | None
            Colonne da normalizzare. Se None, tutte le numeriche.
        """
        if method not in {"zscore", "minmax", "returns"}:
            raise ValueError(f"Metodo '{method}' non valido.")
        self.method = method
        self.columns = columns
        self.params_: dict[str, dict[str, float]] = {}  # parametri appresi nel fit
        self._fitted = False                            # flag di stato

    def _target_columns(self, df: pd.DataFrame) -> list[str]:
        """Determina su quali colonne operare."""
        if self.columns is not None:
            return self.columns
        # Tutte le colonne numeriche se non specificato.
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """
        Apprende i parametri di normalizzazione SOLO da questi dati (training).
        """
        cols = self._target_columns(df)
        self.params_ = {}
        for col in cols:
            series = df[col]
            if self.method == "zscore":
                # Salviamo media e std; std=0 viene sostituita con 1 per non dividere per zero.
                std = float(series.std(ddof=0))
                self.params_[col] = {"mean": float(series.mean()), "std": std or 1.0}
            elif self.method == "minmax":
                mn, mx = float(series.min()), float(series.max())
                # range nullo (colonna costante) -> usiamo 1 per evitare /0.
                self.params_[col] = {"min": mn, "range": (mx - mn) or 1.0}
            # 'returns' non richiede parametri appresi: e' una trasformazione locale.
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica la normalizzazione usando i parametri appresi nel fit."""
        if not self._fitted:
            raise RuntimeError("Chiama prima fit() (o fit_transform()).")
        out = df.copy()
        cols = self._target_columns(df)
        for col in cols:
            if self.method == "zscore":
                p = self.params_[col]
                out[col] = (df[col] - p["mean"]) / p["std"]
            elif self.method == "minmax":
                p = self.params_[col]
                out[col] = (df[col] - p["min"]) / p["range"]
            elif self.method == "returns":
                # Rendimento log: ln(x_t / x_{t-1}). Prima riga -> NaN, la riempiamo con 0.
                out[col] = np.log(df[col] / df[col].shift(1)).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scorciatoia: fit() sui dati e subito transform()."""
        return self.fit(df).transform(df)
