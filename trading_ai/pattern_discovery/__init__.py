"""
Modulo 3 - Pattern Discovery
============================

Scopre AUTONOMAMENTE configurazioni di mercato ricorrenti (nessun pattern
predefinito) tramite clustering non supervisionato sulle feature del Modulo 2,
e ne valuta la solidita' statistica con metriche complete e validazione
OUT-OF-SAMPLE. I pattern instabili vengono scartati automaticamente.

Pipeline:
  1) Split temporale train/test (niente shuffle: e' una serie storica).
  2) Clustering (scaler+KMeans) addestrato SOLO sul train.
  3) Etichettatura outcome (forward return) e statistiche per cluster su train.
  4) Determinazione direzione (long/short) dal train.
  5) Ri-valutazione degli stessi cluster sul test (OOS).
  6) Filtro di stabilita': si tengono solo i pattern coerenti train vs test.

Anti-leakage garantito: feature causali (Modulo 2), scaler/KMeans fittati solo
sul train, etichette (futuro) mai usate come input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from trading_ai.pattern_discovery.clustering import FeatureClusterer
from trading_ai.pattern_discovery.labeling import forward_excursions, forward_return
from trading_ai.pattern_discovery.metrics import PatternStats, compute_stats
from trading_ai.utils.logging import get_logger

# NB: deflated_sharpe_ratio e' importato LAZY dentro discover() per evitare un
# import circolare (validation -> strategy_generator -> pattern_discovery).

logger = get_logger(__name__)

__all__ = ["PatternDiscovery", "DiscoveredPattern", "FeatureClusterer",
           "compute_stats", "forward_return"]


@dataclass
class DiscoveredPattern:
    """Un pattern scoperto e validato, con statistiche train e test."""

    cluster_id: int            # id del cluster KMeans
    direction: int             # +1 long, -1 short (deciso sul train)
    train: PatternStats        # statistiche in-sample
    test: PatternStats         # statistiche out-of-sample
    stable: bool               # ha superato il filtro di stabilita'?
    dsr: float = float("nan")  # Deflated Sharpe Ratio (prob. di edge reale dopo multiple testing)

    def as_dict(self) -> dict:
        """Riga piatta comoda per costruire una tabella riassuntiva."""
        row = {"cluster_id": self.cluster_id, "direction": self.direction,
               "stable": self.stable, "dsr": self.dsr}
        for prefix, st in (("train", self.train), ("test", self.test)):
            for k, v in st.as_dict().items():
                row[f"{prefix}_{k}"] = v
        return row


class PatternDiscovery:
    """Orchestratore della scoperta e validazione dei pattern."""

    def __init__(
        self,
        n_clusters: int = 20,
        horizon: int = 10,            # orizzonte (barre) per il forward return
        train_frac: float = 0.7,      # quota di dati per il training
        min_frequency: float = 0.005, # frequenza minima (0.5%) per essere rilevante
        min_profit_factor: float = 1.1,  # profit factor minimo OOS
        min_count_test: int = 20,     # numero minimo di trade OOS per significativita'
        random_state: int = 42,
        feature_columns: list[str] | None = None,  # restringe le feature usate (es. set esportabile EA)
        use_dsr: bool = True,         # applica la correzione per multiple testing (Deflated Sharpe)
        min_dsr: float = 0.90,        # DSR minimo per accettare un pattern (prob. di edge reale)
    ):
        self.n_clusters = n_clusters
        self.horizon = horizon
        self.train_frac = train_frac
        self.min_frequency = min_frequency
        self.min_profit_factor = min_profit_factor
        self.min_count_test = min_count_test
        self.random_state = random_state
        self.feature_columns = feature_columns
        self.use_dsr = use_dsr
        self.min_dsr = min_dsr
        self.clusterer: FeatureClusterer | None = None
        self.patterns_: list[DiscoveredPattern] = []

    # --- API principale ------------------------------------------------------
    def discover(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Esegue l'intera pipeline e ritorna una tabella di pattern.

        `features` e' l'output del Modulo 2 (OHLCV + feature). Deve contenere
        almeno la colonna 'close' per calcolare gli outcome.
        """
        if "close" not in features.columns:
            raise ValueError("Serve la colonna 'close' per etichettare gli outcome.")

        df = features.dropna().copy()                  # rimuoviamo warm-up/NaN residui
        if len(df) < 100:
            raise ValueError("Troppe poche righe dopo dropna per una scoperta affidabile.")

        # --- 1) Etichettatura outcome (forward return + excursioni) ---------
        fwd = forward_return(df["close"], self.horizon)
        exc = forward_excursions(df, self.horizon)
        # Teniamo solo le barre con futuro completo (le ultime `horizon` sono NaN).
        valid = fwd.notna()
        df, fwd, exc = df[valid], fwd[valid], exc[valid]

        # --- 2) Split temporale (NO shuffle: serie storica) -----------------
        split = int(len(df) * self.train_frac)
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        fwd_tr, fwd_te = fwd.iloc[:split], fwd.iloc[split:]
        dur_tr = exc["tt_peak"].iloc[:split]
        dur_te = exc["tt_peak"].iloc[split:]

        # --- 3) Clustering addestrato SOLO sul train ------------------------
        self.clusterer = FeatureClusterer(self.n_clusters, self.random_state,
                                          feature_columns=self.feature_columns)
        labels_tr = self.clusterer.fit_predict(train_df)   # fit+predict sul train
        labels_te = self.clusterer.predict(test_df)        # solo predict sul test

        # --- 4-5-6) Statistiche per cluster, direzione, OOS, filtro ---------
        # Pass 1: statistiche e stabilita' OOS preliminare per ogni cluster.
        self.patterns_ = []
        oos_returns: list[np.ndarray] = []   # rendimenti OOS orientati, per il DSR
        n_train, n_test = len(train_df), len(test_df)
        for cid in range(self.n_clusters):
            mask_tr = (labels_tr == cid).to_numpy()
            mask_te = (labels_te == cid).to_numpy()

            r_tr = fwd_tr.to_numpy()[mask_tr]               # rendimenti del cluster nel train
            if len(r_tr) == 0:
                continue                                    # cluster vuoto nel train: ignora

            # Direzione decisa dal train: long se rendimento medio positivo, altrimenti short.
            direction = 1 if np.nanmean(r_tr) >= 0 else -1

            # Orientiamo i rendimenti secondo la direzione (short -> -r).
            stats_tr = compute_stats(direction * r_tr, n_train,
                                     dur_tr.to_numpy()[mask_tr])
            r_te = direction * fwd_te.to_numpy()[mask_te]
            stats_te = compute_stats(r_te, n_test, dur_te.to_numpy()[mask_te])

            stable = self._is_stable(stats_tr, stats_te)
            self.patterns_.append(
                DiscoveredPattern(cid, direction, stats_tr, stats_te, stable)
            )
            oos_returns.append(r_te[np.isfinite(r_te)])

        # Pass 2: Deflated Sharpe Ratio (correzione per multiple testing).
        # n_trials = numero di cluster candidati esaminati; la varianza delle
        # stime di Sharpe tra i cluster alimenta l'Expected Maximum Sharpe.
        self._apply_dsr(oos_returns)

        n_stable = sum(p.stable for p in self.patterns_)
        logger.info("Pattern scoperti: %d | stabili (OOS%s): %d",
                    len(self.patterns_), "+DSR" if self.use_dsr else "", n_stable)

        # Tabella ordinata: prima i stabili, poi per expectancy OOS decrescente.
        table = pd.DataFrame([p.as_dict() for p in self.patterns_])
        if not table.empty:
            table = table.sort_values(
                ["stable", "test_expectancy"], ascending=[False, False]
            ).reset_index(drop=True)
        return table

    def stable_patterns(self) -> list[DiscoveredPattern]:
        """Ritorna solo i pattern che hanno superato il filtro di stabilita'."""
        return [p for p in self.patterns_ if p.stable]

    # --- Correzione per multiple testing (Deflated Sharpe Ratio) ------------
    def _apply_dsr(self, oos_returns: list[np.ndarray]) -> None:
        """
        Calcola il Deflated Sharpe Ratio per ogni pattern e, se use_dsr e'
        attivo, lo rende un requisito aggiuntivo per la stabilita'.

        `oos_returns[i]` sono i rendimenti OOS (gia' orientati) del pattern i.
        Lo Sharpe per-trade di ciascun pattern e' un "trial"; la varianza di
        questi Sharpe alimenta la stima dello Sharpe massimo atteso sotto nulla.
        """
        # Import lazy per evitare l'import circolare con il package validation.
        from trading_ai.validation.multiple_testing import deflated_sharpe_ratio

        n_trials = len(self.patterns_)                  # quanti pattern abbiamo esaminato
        # Sharpe per-trade di ogni pattern (gia' calcolato in stats_te).
        sharpes = np.array([p.test.sharpe_like for p in self.patterns_], dtype=float)
        var_sharpes = float(np.var(sharpes)) if len(sharpes) > 1 else 0.0

        for p, r in zip(self.patterns_, oos_returns):
            if len(r) >= 3:
                # Curtosi NON in eccesso (Pearson): fisher=False -> normale = 3.
                sk = float(_skew(r))
                ku = float(_kurtosis(r, fisher=False))
                p.dsr = deflated_sharpe_ratio(
                    p.test.sharpe_like, len(r), n_trials, var_sharpes, sk, ku)
            else:
                p.dsr = 0.0                              # troppi pochi trade: non valutabile
            # Se richiesto, il DSR diventa un requisito aggiuntivo della stabilita'.
            if self.use_dsr:
                p.stable = p.stable and (p.dsr >= self.min_dsr)

    # --- Filtro di stabilita' -----------------------------------------------
    def _is_stable(self, tr: PatternStats, te: PatternStats) -> bool:
        """
        Un pattern e' STABILE se i suoi numeri reggono fuori campione:
          - frequenza sufficiente sia in train sia in test,
          - abbastanza trade OOS da essere significativo,
          - expectancy positiva in ENTRAMBI i periodi (coerenza di segno),
          - profit factor OOS sopra la soglia minima.
        Questo scarta automaticamente i pattern overfittati.
        """
        return (
            tr.frequency >= self.min_frequency and
            te.frequency >= self.min_frequency and
            te.count >= self.min_count_test and
            tr.expectancy > 0 and
            te.expectancy > 0 and
            te.profit_factor >= self.min_profit_factor
        )
