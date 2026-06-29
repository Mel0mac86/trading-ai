"""
Test dell'autopilota, dell'acquisizione dati e della persistenza.

Verifichiamo che la piattaforma giri in modo completamente autonomo (fallback
sintetico garantito), che i modelli si salvino/ricarichino e che la run produca
manifest e summary.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from trading_ai.autopilot import AutopilotConfig, run_autopilot
from trading_ai.data_engine import generate_ohlcv
from trading_ai.data_engine.sources import acquire
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.persistence import (
    load_object, load_strategy, save_object, save_strategy,
)
from trading_ai.pattern_discovery.clustering import FeatureClusterer
from trading_ai.pipeline import PipelineConfig
from trading_ai.strategy_generator import RiskParams, Strategy


# --- Acquisizione dati -------------------------------------------------------
def test_acquire_synthetic_fallback(tmp_path):
    """Senza CSV locali e senza rete, deve ripiegare sul sintetico."""
    df, source = acquire("EURUSD", datasets_dir=tmp_path,
                         allow_download=False, synthetic_bars=5000)
    assert source == "synthetic"
    assert len(df) == 5000
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_acquire_prefers_local_csv(tmp_path):
    """Se esiste un CSV col simbolo nel nome, viene usato quello."""
    csv = tmp_path / "EURUSD_H1.csv"
    csv.write_text(
        "date,open,high,low,close,volume\n"
        "2020-01-01 00:00,1.1,1.11,1.09,1.105,100\n"
        "2020-01-01 01:00,1.105,1.112,1.10,1.108,120\n"
    )
    df, source = acquire("EURUSD", datasets_dir=tmp_path, allow_download=False)
    assert "locale" in source and "EURUSD_H1.csv" in source
    assert len(df) == 2


def test_acquire_merges_multiple_yearly_files(tmp_path):
    """Piu' file annuali dello stesso strumento vengono fusi in una serie unica."""
    (tmp_path / "DAT_MT_XAUUSD_M1_2023.csv").write_text(
        "2023.12.31,23:58,2062.0,2063.0,2061.0,2062.5,0\n"
        "2023.12.31,23:59,2062.5,2063.5,2062.0,2063.0,0\n"
    )
    (tmp_path / "DAT_MT_XAUUSD_M1_2024.csv").write_text(
        "2024.01.01,00:00,2063.0,2064.0,2062.5,2063.5,0\n"
        "2024.01.01,00:01,2063.5,2064.5,2063.0,2064.0,0\n"
    )
    df, source = acquire("XAUUSD", datasets_dir=tmp_path, allow_download=False)
    assert "2 file" in source            # due file fusi
    assert len(df) == 4
    assert df.index.is_monotonic_increasing   # ordinati cronologicamente
    assert not df.index.has_duplicates


# --- Persistenza -------------------------------------------------------------
def test_save_load_object_roundtrip(tmp_path):
    """save_object/load_object devono restituire un oggetto equivalente."""
    obj = {"a": [1, 2, 3], "b": "x"}
    path = save_object(obj, tmp_path / "obj.joblib")
    assert path.exists()
    assert load_object(path) == obj


def test_load_object_missing(tmp_path):
    """Caricare un file inesistente solleva FileNotFoundError chiaro."""
    with pytest.raises(FileNotFoundError):
        load_object(tmp_path / "nope.joblib")


def test_save_load_strategy_roundtrip(tmp_path):
    """Una strategia (clusterer incluso) deve sopravvivere al round-trip."""
    df = generate_ohlcv(n=2000, seed=4)
    feats = FeatureEngine().transform(df, groups=["indicator"], dropna=True)
    cols = [c for c in feats.columns if c not in ("open", "high", "low", "close", "volume")]
    clu = FeatureClusterer(n_clusters=4, feature_columns=cols).fit(feats)
    strat = Strategy(name="PAT00_LONG", cluster_id=1, direction=1, clusterer=clu,
                     risk=RiskParams(sl_atr=2, tp_atr=3))
    path = save_strategy(strat, tmp_path)
    loaded = load_strategy(path)
    assert loaded.name == strat.name
    # Il clusterer ricaricato deve produrre le stesse predizioni.
    pred_a = strat.clusterer.predict(feats)
    pred_b = loaded.clusterer.predict(feats)
    assert (pred_a.to_numpy() == pred_b.to_numpy()).all()


# --- Autopilota end-to-end ---------------------------------------------------
def test_autopilot_runs_autonomously(tmp_path):
    """Senza alcun input, l'autopilota completa la run e scrive gli artefatti."""
    cfg = AutopilotConfig(
        instruments=["EURUSD"],
        allow_download=False,
        synthetic_bars=40_000,
        persist_models=False,                 # niente scrittura in /models nel test
        output_root=tmp_path,
        pipeline=PipelineConfig(timeframe="H1", horizon=8, n_clusters=8,
                                min_profit_factor=1.0, min_count_test=10),
    )
    manifest = run_autopilot(cfg)

    assert manifest["run_id"]
    assert len(manifest["instruments"]) == 1
    entry = manifest["instruments"][0]
    assert entry["instrument"] == "EURUSD"
    assert entry["status"] == "ok"
    assert entry["source"] == "synthetic"

    run_dir = tmp_path / manifest["run_id"]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.md").exists()
    # Il manifest su file deve essere JSON valido e coerente.
    saved = json.loads((run_dir / "manifest.json").read_text())
    assert saved["run_id"] == manifest["run_id"]


def test_autopilot_multi_instrument_continues_on_error(tmp_path):
    """Un errore su uno strumento non deve fermare gli altri."""
    cfg = AutopilotConfig(
        instruments=["EURUSD", "GBPUSD"],
        allow_download=False,
        synthetic_bars=12_000,
        persist_models=False,
        output_root=tmp_path,
        pipeline=PipelineConfig(timeframe="H1", horizon=6, n_clusters=6,
                                min_count_test=8, validate=False,
                                make_reports=False, export_ea=False),
    )
    manifest = run_autopilot(cfg)
    assert len(manifest["instruments"]) == 2
    # Entrambi gli strumenti sono stati elaborati (status presente per ciascuno).
    assert all("status" in e for e in manifest["instruments"])


# --- CLI ---------------------------------------------------------------------
def test_cli_main_runs_from_yaml(tmp_path):
    """La CLI deve eseguire una run da file YAML e ritornare codice 0."""
    from trading_ai.__main__ import main

    yaml_cfg = tmp_path / "cfg.yaml"
    yaml_cfg.write_text(
        "instruments: [EURUSD]\n"
        "allow_download: false\n"
        "synthetic_bars: 12000\n"
        "persist_models: false\n"
        f"output_root: {tmp_path / 'out'}\n"
        "pipeline:\n"
        "  timeframe: H1\n"
        "  horizon: 6\n"
        "  n_clusters: 6\n"
        "  min_count_test: 8\n"
        "  validate: false\n"
        "  make_reports: false\n"
        "  export_ea: false\n"
    )
    rc = main(["--config", str(yaml_cfg)])
    assert rc == 0
    # La run deve aver creato almeno una cartella con manifest.
    runs = list((tmp_path / "out").glob("*/manifest.json"))
    assert len(runs) == 1


def test_cli_build_config_overrides():
    """Gli argomenti CLI hanno priorita' sui default."""
    from trading_ai.__main__ import _build_config
    import argparse
    ns = argparse.Namespace(config=None, instruments=["XAUUSD"], no_download=True,
                            bars=5000, output=None, kaggle_dataset="melomac/histdata")
    cfg = _build_config(ns)
    assert cfg.instruments == ["XAUUSD"]
    assert cfg.allow_download is False
    assert cfg.synthetic_bars == 5000
    assert cfg.kaggle_dataset == "melomac/histdata"
