"""
Test d'integrazione end-to-end della piattaforma (tutti i moduli insieme).

Verifica che la pipeline completa giri senza errori su dati grezzi e produca
artefatti coerenti, anche nel caso (atteso su dati sintetici) in cui nessuna
strategia risulti robusta.
"""

from __future__ import annotations

import pandas as pd

from trading_ai.data_engine import generate_ohlcv
from trading_ai.pipeline import PipelineConfig, PipelineResult, run_pipeline
from trading_ai.strategy_generator import RiskParams


def test_pipeline_runs_end_to_end(tmp_path):
    """La pipeline completa deve girare e ritornare un risultato strutturato."""
    raw = generate_ohlcv(n=60_000, freq="1min", seed=4)
    cfg = PipelineConfig(
        timeframe="H1", horizon=8, n_clusters=10,
        min_profit_factor=1.0, min_count_test=10,
        risk=RiskParams(max_bars=20),
        validate=True, make_reports=False, export_ea=False,  # snelliamo per la CI
    )
    result = run_pipeline(raw, cfg)

    assert isinstance(result, PipelineResult)
    assert isinstance(result.features, pd.DataFrame) and len(result.features) > 0
    assert isinstance(result.patterns, pd.DataFrame)
    # robust_strategies e' sempre una lista (eventualmente vuota su random walk).
    assert isinstance(result.robust_strategies, list)
    # Se ci sono strategie, la tabella di validazione esiste.
    if result.strategies:
        assert result.validation_table is not None


def test_pipeline_reports_and_ea(tmp_path):
    """Con report+EA attivi la pipeline non deve sollevare errori."""
    raw = generate_ohlcv(n=50_000, freq="1min", seed=11)
    cfg = PipelineConfig(
        timeframe="H1", horizon=8, n_clusters=8,
        min_profit_factor=1.0, min_count_test=8,
        risk=RiskParams(max_bars=20),
        validate=True, make_reports=True, export_ea=True,
    )
    result = run_pipeline(raw, cfg)
    # Gli artefatti sono liste (vuote se nessuna strategia robusta): nessun crash.
    assert isinstance(result.reports, list)
    assert isinstance(result.ea_files, list)
