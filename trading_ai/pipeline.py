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

from dataclasses import dataclass, field

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


@dataclass
class PipelineConfig:
    """Tutti i parametri della pipeline in un solo posto."""

    timeframe: str = "H1"              # timeframe operativo (Modulo 1)
    horizon: int = 10                  # orizzonte outcome (Modulo 3)
    n_clusters: int = 25               # numero pattern candidati (Modulo 3)
    min_profit_factor: float = 1.1     # soglia OOS (Modulo 3)
    min_count_test: int = 20           # trade OOS minimi (Modulo 3)
    exportable_only: bool = True       # usa solo feature esportabili in MQL (per il Modulo 6)
    risk: RiskParams = field(default_factory=RiskParams)
    filters: Filters = field(default_factory=Filters)
    costs: CostModel = field(default_factory=CostModel)  # costi di transazione (Modulo 4)
    validate: bool = True              # esegui il Modulo 5
    make_reports: bool = True          # esegui il Modulo 9
    export_ea: bool = True             # esegui il Modulo 6


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
        feature_columns=feature_cols,
    )
    patterns = disc.discover(feats)

    # --- Modulo 4: strategie dai pattern stabili ----------------------------
    gen = StrategyGenerator(disc, risk=cfg.risk, filters=cfg.filters, costs=cfg.costs)
    strategies = gen.build()

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
        rep = ReportGenerator()
        for s in robust_strategies:
            res = s.run(feats)
            reports_out.append(
                rep.generate(s, res, validation=report_by_name.get(s.name))
            )

    # --- Modulo 6: export EA -------------------------------------------------
    ea_files = []
    if cfg.export_ea and robust_strategies:
        eag = EAGenerator()
        ea_files = eag.export_many(robust_strategies)  # ignora con log le non esportabili

    logger.info(
        "Pipeline completata: %d pattern, %d strategie, %d robuste, %d EA.",
        len(patterns), len(strategies), len(robust_strategies), len(ea_files),
    )
    return PipelineResult(feats, patterns, strategies, validation_table,
                          robust_strategies, reports_out, ea_files)
