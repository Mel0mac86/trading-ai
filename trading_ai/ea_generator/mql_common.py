"""
Utility condivise per la generazione MQL4/MQL5 (Modulo 6).

Serializza il modello (StandardScaler + centroidi KMeans) in dichiarazioni di
array MQL e costruisce i frammenti di codice comuni ai due linguaggi (calcolo
del vettore feature, standardizzazione, cluster piu' vicino).

Il modello e' embeddato direttamente nel sorgente: l'EA e' autosufficiente e non
richiede file esterni.
"""

from __future__ import annotations

import numpy as np


def _fmt(x: float) -> str:
    """Formatta un float per MQL con precisione sufficiente e suffisso decimale."""
    # repr garantisce abbastanza cifre; assicuriamo il punto decimale per i double.
    s = repr(float(x))
    if "e" in s or "E" in s:      # notazione esponenziale accettata da MQL
        return s
    if "." not in s:
        s += ".0"
    return s


def array_1d(name: str, values: np.ndarray) -> str:
    """Genera la dichiarazione di un array MQL 1D di double inizializzato."""
    body = ", ".join(_fmt(v) for v in np.asarray(values, dtype=float).ravel())
    return f"double {name}[] = {{{body}}};"


def centroids_block(name: str, centroids: np.ndarray) -> str:
    """
    Genera un array MQL 1D che appiattisce i centroidi (n_clusters x n_features)
    in row-major. L'accesso e' cent[c*n_features + j].
    """
    flat = np.asarray(centroids, dtype=float).ravel()
    return array_1d(name, flat)


def model_constants(n_features: int, n_clusters: int, cluster_id: int,
                    direction: int) -> str:
    """Costanti #define del modello embeddato."""
    return (
        f"#define FEAT_COUNT {n_features}\n"
        f"#define N_CLUSTERS {n_clusters}\n"
        f"#define TARGET_CLUSTER {cluster_id}\n"
        f"#define DIRECTION {direction}   // +1 = BUY, -1 = SELL\n"
    )


def extract_model(strategy) -> dict:
    """
    Estrae dal clusterer della strategia i pezzi necessari all'EA:
    schema feature, medie/scale dello scaler, centroidi KMeans.
    """
    clu = strategy.clusterer
    return {
        "feature_columns": list(clu.feature_columns),
        "mean": clu.scaler.mean_.astype(float),
        "scale": clu.scaler.scale_.astype(float),
        "centroids": clu.kmeans.cluster_centers_.astype(float),
        "n_features": len(clu.feature_columns),
        "n_clusters": clu.kmeans.n_clusters,
    }
