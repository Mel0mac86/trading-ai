"""
Pipeline end-to-end della piattaforma.

Orchestratore che collega TUTTI i moduli in un'unica chiamata operativa:

    dati grezzi
      -> Modulo 1  Data Engine        (pulizia + timeframe)
      -> Modulo 2  Feature Engineering (estrazione feature)
      -> Modulo 3  Pattern Discovery   (scoperta + validazione OOS)
      -> Modulo 4  Strategy Generator  (strategie dai pattern stabili)
      -> Modulo 5  Validation          (robustezza, scarto dei non robusti)
      -> Modulo 9  Report              (metriche + grafici + razionale)
      -> Modulo 6  EA Generator        (export MQL4/MQL5)
      -> Modulo 7  AI Feedback         (ottimizzazione + versioning) [opzionale]

E' il punto d'ingresso che rende la piattaforma utilizzabile "in un colpo solo",
sia da script sia da notebook Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from trading_ai.data_engine import DataEngine
from trading_ai.ea_generator import EAGenerator, EXPORTABLE_FEATURES
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.reporting import ReportGenerator
from trading_ai.strategy_generator import CostModel, Filters, RiskParams, StrategyGenerator
from trading_ai.validation import Validator
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]


def _is_zero_cost(costs: CostModel) -> bool:
    """True se il CostModel e' a costo nullo (nessun costo impostato dall'utente)."""
    return costs.spread == 0 and costs.slippage == 0 and costs.commission == 0


@dataclass
class PipelineConfig:
    """Tutti i parametri della pipeline in un solo posto."""

    timeframe: str = "H1"              # timeframe operativo (Modulo 1)
    horizon: int = 10                  # orizzonte outcome (Modulo 3)
    n_clusters: int = 25               # numero pattern candidati (Modulo 3)
    min_profit_factor: float = 1.1     # soglia OOS (Modulo 3)
    min_count_test: int = 20           # trade OOS minimi (Modulo 3)
    use_dsr: bool = True               # correzione multiple testing (Deflated Sharpe)
    min_dsr: float = 0.90              # DSR minimo per accettare un pattern
    exportable_only: bool = True       # usa solo feature esportabili in MQL (per il Modulo 6)
    risk: RiskParams = field(default_factory=RiskParams)
    filters: Filters = field(default_factory=Filters)
    costs: CostModel = field(default_factory=CostModel)  # costi di transazione (Modulo 4)
    point_value: "float | None" = None  # valore del punto per i costi da spread (None = inferito dai dati)
    validate: bool = True              # esegui il Modulo 5
    make_reports: bool = True          # esegui il Modulo 9
    export_ea: bool = True             # esegui il Modulo 6
    instrument: str = ""               # prefisso per i nomi strategia (evita collisioni multi-strumento)
    reports_dir: "Path | None" = None  # cartella report (default: /reports)
    mql4_dir: "Path | None" = None     # cartella export MQL4 (default: /mql4)
    mql5_dir: "Path | None" = None     # cartella export MQL5 (default: /mql5)


@dataclass
class PipelineResult:
    """Artefatti prodotti dalla pipeline."""

    features: pd.DataFrame
    patterns: pd.DataFrame
    strategies: list
    validation_table: pd.DataFrame | None
    robust_strategies: list
    reports: list
    ea_files: list


def run_pipeline(df: pd.DataFrame, config: PipelineConfig | None = None) -> PipelineResult:
    """
    Esegue la pipeline completa su un DataFrame OHLCV grezzo.

    Parametri
    ---------
    df : pd.DataFrame
        Dati grezzi (qualsiasi schema riconosciuto dal Modulo 1).
    config : PipelineConfig | None
        Configurazione; se None usa i default.
    """
    cfg = config or PipelineConfig()

    # --- Modulo 1: pulizia + timeframe --------------------------------------
    eng = DataEngine()
    clean = eng.load_dataframe(df)                     # normalizza + pulisce
    bars = eng.to_timeframe(clean, cfg.timeframe)      # resample al timeframe operativo

    # --- Costi reali dallo spread MetaTrader (se presente) ------------------
    # Lo spread esportato (in punti) viene convertito in costi reali e poi
    # RIMOSSO, cosi' non finisce per errore tra le feature del clustering.
    costs = cfg.costs
    if "spread" in bars.columns:
        point_value = cfg.point_value or clean.attrs.get("point_value", 1e-5)
        median_pts = float(bars["spread"].median())
        bars = bars.drop(columns=["spread"])
        if _is_zero_cost(cfg.costs):                   # solo se l'utente non ha gia' fissato i costi
            costs = CostModel.from_spread_points(median_pts, point_value)
            logger.info("Costi dedotti dallo spread reale: spread=%.6f prezzo "
                        "(mediana %.1f punti, point_value=%.5g)",
                        costs.spread, median_pts, point_value)

    # --- Modulo 2: feature ---------------------------------------------------
    # Se vogliamo esportare EA fedeli, limitiamo ai gruppi calcolabili in MQL.
    if cfg.exportable_only:
        feats = FeatureEngine().transform(bars, groups=["indicator", "volatility"], dropna=True)
        feature_cols = [c for c in EXPORTABLE_FEATURES if c in feats.columns]
    else:
        feats = FeatureEngine().transform(bars, dropna=True)
        feature_cols = None

    # --- Modulo 3: scoperta pattern + validazione OOS -----------------------
    disc = PatternDiscovery(
        n_clusters=cfg.n_clusters, horizon=cfg.horizon,
        min_profit_factor=cfg.min_profit_factor, min_count_test=cfg.min_count_test,
        feature_columns=feature_cols, use_dsr=cfg.use_dsr, min_dsr=cfg.min_dsr,
    )
    patterns = disc.discover(feats)

    # --- Modulo 4: strategie dai pattern stabili ----------------------------
    gen = StrategyGenerator(disc, risk=cfg.risk, filters=cfg.filters, costs=costs)
    strategies = gen.build()
    # Prefisso strumento sui nomi (evita collisioni di file tra strumenti diversi).
    if cfg.instrument:
        strategies = [replace(s, name=f"{cfg.instrument}_{s.name}") for s in strategies]

    # --- Modulo 5: validazione robustezza -----------------------------------
    validation_table = None
    robust_strategies = list(strategies)               # default: tutte (se non validiamo)
    if cfg.validate and strategies:
        validator = Validator()
        reports = [validator.validate(s, feats) for s in strategies]
        validation_table = pd.DataFrame([r.summary() for r in reports])
        # Teniamo solo le strategie dichiarate robuste.
        robust_names = {r.name for r in reports if r.robust}
        robust_strategies = [s for s in strategies if s.name in robust_names]
        # Indicizziamo i report per nome per riusarli nel Modulo 9.
        report_by_name = {r.name: r for r in reports}
    else:
        report_by_name = {}

    # --- Modulo 9: report ----------------------------------------------------
    reports_out = []
    if cfg.make_reports:
        rep = ReportGenerator(reports_dir=cfg.reports_dir)
        for s in robust_strategies:
            res = s.run(feats)
            reports_out.append(
                rep.generate(s, res, validation=report_by_name.get(s.name))
            )

    # --- Modulo 6: export EA -------------------------------------------------
    ea_files = []
    if cfg.export_ea and robust_strategies:
        eag = EAGenerator(mql4_dir=cfg.mql4_dir, mql5_dir=cfg.mql5_dir)
        ea_files = eag.export_many(robust_strategies)  # ignora con log le non esportabili

    logger.info(
        "Pipeline completata: %d pattern, %d strategie, %d robuste, %d EA.",
        len(patterns), len(strategies), len(robust_strategies), len(ea_files),
    )
    return PipelineResult(feats, patterns, strategies, validation_table,
                          robust_strategies, reports_out, ea_files)
