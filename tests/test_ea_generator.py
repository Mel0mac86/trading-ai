"""
Test automatici del Modulo 6 - EA Generator.

Non potendo compilare MQL in CI, verifichiamo che il codice generato sia
STRUTTURALMENTE corretto e fedele al modello: presenza delle sezioni chiave,
dimensioni esatte degli array embeddati (scaler e centroidi), parametri di
rischio, estensioni dei file e il controllo sulle feature esportabili.
"""

from __future__ import annotations

import re

import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.ea_generator import EAGenerator, EXPORTABLE_FEATURES
from trading_ai.ea_generator.features_map import check_exportable
from trading_ai.pattern_discovery.clustering import FeatureClusterer
from trading_ai.strategy_generator import RiskParams, Strategy


@pytest.fixture
def strategy():
    """Costruisce una strategia con clusterer addestrato su feature ESPORTABILI."""
    df = generate_ohlcv(n=4000, seed=8)
    feats = FeatureEngine().transform(df, groups=["indicator", "volatility"], dropna=True)
    # Restringiamo alle feature esportabili presenti nel frame.
    cols = [c for c in EXPORTABLE_FEATURES if c in feats.columns]
    clu = FeatureClusterer(n_clusters=6, feature_columns=cols).fit(feats)
    return Strategy(name="PAT00_LONG", cluster_id=2, direction=1,
                    clusterer=clu, risk=RiskParams(sl_atr=2, tp_atr=3))


def test_check_exportable_rejects_unknown():
    """Feature non native (es. fvg, structure) devono essere rifiutate."""
    with pytest.raises(ValueError):
        check_exportable(["rsi", "fvg_fvg", "structure_bos"])


def test_check_exportable_accepts_known():
    """Le feature esportabili non sollevano errori."""
    check_exportable(["rsi", "atr", "macd_hist", "bollinger_pctb"])


def test_mql4_contains_key_sections(strategy):
    """Il sorgente MQL4 deve contenere le sezioni fondamentali."""
    code = EAGenerator().to_mql4(strategy)
    for token in ["#property strict", "void OnTick()", "int OnInit()",
                  "NearestCluster", "Standardize", "OrderSend",
                  "#define TARGET_CLUSTER 2", "#define DIRECTION 1"]:
        assert token in code


def test_mql5_contains_key_sections(strategy):
    """Il sorgente MQL5 deve usare l'API a handle e CTrade."""
    code = EAGenerator().to_mql5(strategy)
    for token in ["#include <Trade/Trade.mqh>", "CopyBuffer", "CTrade",
                  "NearestCluster", "#define TARGET_CLUSTER 2", "gTrade.Buy"]:
        assert token in code


def test_embedded_arrays_have_correct_size(strategy):
    """gMean/gScale devono avere FEAT_COUNT valori; gCent N_CLUSTERS*FEAT_COUNT."""
    code = EAGenerator().to_mql4(strategy)
    n_feat = len(strategy.clusterer.feature_columns)
    n_clu = strategy.clusterer.kmeans.n_clusters

    def count_values(arr_name: str) -> int:
        m = re.search(arr_name + r"\[\] = \{([^}]*)\}", code)
        assert m, f"array {arr_name} non trovato"
        return len([x for x in m.group(1).split(",") if x.strip()])

    assert count_values("gMean") == n_feat
    assert count_values("gScale") == n_feat
    assert count_values("gCent") == n_feat * n_clu


def test_export_writes_files(strategy, tmp_path):
    """L'export deve creare i file .mq4 e .mq5 nelle cartelle indicate."""
    eag = EAGenerator(mql4_dir=tmp_path / "mql4", mql5_dir=tmp_path / "mql5")
    paths = eag.export(strategy)
    assert paths["mql4"].exists() and paths["mql4"].suffix == ".mq4"
    assert paths["mql5"].exists() and paths["mql5"].suffix == ".mq5"
    assert "OnTick" in paths["mql4"].read_text()


def test_magic_number_is_deterministic(strategy):
    """Lo stesso nome strategia deve produrre SEMPRE lo stesso magic number."""
    from trading_ai.ea_generator.mql_common import stable_magic
    # Determinismo tra chiamate (a differenza di hash() randomizzato).
    assert stable_magic("PAT00_LONG") == stable_magic("PAT00_LONG")
    assert 10000 <= stable_magic("qualsiasi") <= 99999
    # Il valore nel sorgente coincide con quello calcolato dall'helper.
    code = EAGenerator().to_mql4(strategy)
    expected = stable_magic(strategy.name)
    assert f"InpMagic       = {expected}" in code


def test_export_rejects_non_exportable(strategy):
    """Se il clusterer usa feature non native, l'export deve fallire chiaramente."""
    strategy.clusterer.feature_columns = ["rsi", "fvg_fvg"]   # forziamo una non esportabile
    with pytest.raises(ValueError):
        EAGenerator().to_mql4(strategy)
