"""
Clustering delle feature per scoprire pattern (Modulo 3).

L'AI NON usa pattern predefiniti: lascia che sia un algoritmo NON supervisionato
(KMeans) a raggruppare le configurazioni di mercato ricorrenti. Ogni cluster
diventa un "pattern" candidato, di cui poi misuriamo le statistiche.

Anti-leakage: lo scaler e il KMeans vengono ADDESTRATI SOLO sui dati di
training; su validation/test si applica solo .transform()/.predict().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class FeatureClusterer:
    """Wrapper anti-leakage attorno a StandardScaler + KMeans."""

    def __init__(self, n_clusters: int = 20, random_state: int = 42,
                 feature_columns: list[str] | None = None):
        """
        Parametri
        ---------
        n_clusters : int
            Numero di pattern candidati da cercare.
        random_state : int
            Seme per risultati riproducibili.
        feature_columns : list[str] | None
            Colonne da usare come feature. Se None, tutte le numeriche tranne OHLCV.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()                 # standardizzazione (media 0, std 1)
        # n_init='auto' lascia a sklearn la scelta del numero di inizializzazioni.
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self._fitted = False

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleziona e ordina le colonne feature, escludendo OHLCV grezzo."""
        if self.feature_columns is not None:
            cols = self.feature_columns
        else:
            exclude = {"open", "high", "low", "close", "volume"}
            cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
        # Memorizziamo l'ordine al fit per garantire coerenza al transform.
        return df[cols]

    def fit(self, df: pd.DataFrame) -> "FeatureClusterer":
        """Addestra scaler + KMeans SOLO su questi dati (training)."""
        x = self._select(df)
        self.feature_columns = list(x.columns)         # congeliamo lo schema feature
        x = x.to_numpy()
        x = self.scaler.fit_transform(x)               # impara media/std dal training
        self.kmeans.fit(x)                             # trova i centroidi dei cluster
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Assegna a ogni barra l'etichetta di cluster (usando i parametri appresi)."""
        if not self._fitted:
            raise RuntimeError("Chiama prima fit().")
        x = df[self.feature_columns].to_numpy()        # stesse colonne, stesso ordine
        x = self.scaler.transform(x)                   # scala coi parametri del training
        labels = self.kmeans.predict(x)               # cluster piu' vicino per ogni barra
        return pd.Series(labels, index=df.index, name="cluster")

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """Comodita': fit sui dati e subito predict sugli stessi."""
        return self.fit(df).predict(df)
