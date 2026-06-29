"""
Autopilota della piattaforma.

Esegue l'INTERA piattaforma in modo completamente autonomo, senza alcun input
manuale: acquisisce i dati (con fallback garantito), gira la pipeline su piu'
strumenti, valida, genera report ed EA, PERSISTE i modelli e scrive un report
CONSOLIDATO della run. Pensato per essere lanciato da CLI (`python -m trading_ai`)
o da scheduler (cron/CI), con output organizzato e tracciabile.

Layout di output per ogni run (sotto reports/runs/<timestamp>/):
    manifest.json     -> configurazione + esiti macchina-leggibili
    summary.md        -> sintesi leggibile (tabella per strumento + strategie robuste)
    <Strategia>/      -> report di dettaglio (equity, drawdown, metriche)
e in parallelo:
    models/<run>/     -> clusterer e strategie robuste serializzati (joblib)
    mql4|mql5/        -> Expert Advisor esportati
    logs/<run>.log    -> log completo dell'esecuzione
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_ai.config import PATHS
from trading_ai.data_engine.sources import acquire
from trading_ai.persistence import save_object, save_strategy
from trading_ai.pipeline import PipelineConfig, run_pipeline
from trading_ai.utils.logging import add_file_handler, get_logger

logger = get_logger(__name__)

__all__ = ["AutopilotConfig", "run_autopilot"]

# Strumenti di default: un paniere rappresentativo (Forex, metallo, indice).
# Senza CSV locali ne' rete, l'autopilota usa il fallback sintetico per ognuno.
_DEFAULT_INSTRUMENTS = ["EURUSD", "XAUUSD", "US500"]


@dataclass
class AutopilotConfig:
    """Configurazione completa dell'autopilota (tutto ha un default sensato)."""

    instruments: list[str] = field(default_factory=lambda: list(_DEFAULT_INSTRUMENTS))
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    allow_download: bool = True        # prova a scaricare dati reali se possibile
    kaggle_dataset: str | None = None  # slug 'owner/dataset' Kaggle da cui prendere i dati
    synthetic_bars: int = 200_000      # barre del fallback sintetico
    persist_models: bool = True        # salva clusterer e strategie robuste
    output_root: Path | None = None    # radice degli output (default: /reports/runs)


def _to_plain(obj):
    """Rende serializzabile in JSON un oggetto (dataclass/Path/np types)."""
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def run_autopilot(config: AutopilotConfig | None = None) -> dict:
    """
    Esegue l'autopilota end-to-end e ritorna il manifest della run.

    Non richiede alcun input: con la configurazione di default acquisisce i dati
    (fallback sintetico se serve), processa ogni strumento, persiste i modelli e
    scrive i report. Robusto agli errori del singolo strumento (li registra e
    prosegue).
    """
    cfg = config or AutopilotConfig()

    # --- Cartella della run + logging su file -------------------------------
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.output_root) if cfg.output_root else (PATHS.reports / "runs")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(PATHS.logs / f"autopilot_{run_id}.log")   # traccia completa su file
    models_dir = PATHS.models / run_id

    logger.info("=== AUTOPILOTA avviato (run %s) — strumenti: %s ===",
                run_id, ", ".join(cfg.instruments))

    per_instrument: list[dict] = []
    all_robust_rows: list[dict] = []

    for symbol in cfg.instruments:
        entry = {"instrument": symbol}
        try:
            # 1) Acquisizione dati (con fallback garantito).
            raw, source = acquire(symbol, allow_download=cfg.allow_download,
                                  kaggle_dataset=cfg.kaggle_dataset,
                                  synthetic_bars=cfg.synthetic_bars)
            entry["source"] = source

            # 2) Pipeline completa per questo strumento, con output namespaced.
            pcfg = _instrument_pipeline_config(cfg.pipeline, symbol, run_dir)
            result = run_pipeline(raw, pcfg)

            entry.update({
                "rows": int(len(result.features)),
                "patterns": int(len(result.patterns)),
                "strategies": int(len(result.strategies)),
                "robust": int(len(result.robust_strategies)),
                "ea_files": len(result.ea_files),
            })

            # 3) Persistenza modelli (clusterer + strategie robuste).
            if cfg.persist_models and result.robust_strategies:
                inst_models = models_dir / symbol
                save_object(result.robust_strategies[0].clusterer,
                            inst_models / "clusterer.joblib")
                for s in result.robust_strategies:
                    save_strategy(s, inst_models)
                entry["models_dir"] = str(inst_models)

            # 4) Righe di sintesi delle strategie robuste (per il report globale).
            if result.validation_table is not None and not result.validation_table.empty:
                vt = result.validation_table
                for _, row in vt[vt["robust"]].iterrows():
                    r = {"instrument": symbol}
                    r.update(row.to_dict())
                    all_robust_rows.append(r)

            entry["status"] = "ok"
            logger.info("[%s] OK — %d robuste su %d strategie (fonte: %s)",
                        symbol, entry["robust"], entry["strategies"], source)
        except Exception as e:                          # un errore non deve fermare l'intera run
            entry["status"] = "error"
            entry["error"] = str(e)
            logger.exception("[%s] errore durante l'elaborazione: %s", symbol, e)

        per_instrument.append(entry)

    # --- Manifest + report consolidato --------------------------------------
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": _to_plain(cfg),
        "instruments": per_instrument,
        "total_robust": int(len(all_robust_rows)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(run_dir, manifest, all_robust_rows)

    logger.info("=== AUTOPILOTA completato — %d strategie robuste totali. Output: %s ===",
                manifest["total_robust"], run_dir)
    manifest["run_dir"] = str(run_dir)
    return manifest


def _instrument_pipeline_config(base: PipelineConfig, symbol: str,
                                run_dir: Path) -> PipelineConfig:
    """Clona la PipelineConfig impostando strumento e cartelle di output dedicate."""
    from dataclasses import replace
    return replace(
        base,
        instrument=symbol,
        reports_dir=run_dir,                # i report di dettaglio finiscono nella run
        mql4_dir=PATHS.mql4, mql5_dir=PATHS.mql5,
    )


def _write_summary_md(run_dir: Path, manifest: dict, robust_rows: list[dict]) -> Path:
    """Scrive un riepilogo Markdown leggibile della run."""
    lines = [
        f"# Autopilota — run {manifest['run_id']}",
        "",
        f"_Generato: {manifest['created_utc']}_",
        "",
        "## Esito per strumento",
        "",
        "| Strumento | Fonte dati | Barre | Pattern | Strategie | Robuste | EA | Stato |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in manifest["instruments"]:
        lines.append(
            f"| {e['instrument']} | {e.get('source','-')} | {e.get('rows','-')} | "
            f"{e.get('patterns','-')} | {e.get('strategies','-')} | "
            f"{e.get('robust','-')} | {e.get('ea_files','-')} | {e.get('status','-')} |"
        )

    lines += ["", f"**Strategie robuste totali: {manifest['total_robust']}**", ""]

    if robust_rows:
        lines += ["## Strategie robuste", "",
                  "| Strumento | Nome | Rendimento | Profit Factor | DSR (MC prob) |",
                  "|---|---|---|---|---|"]
        for r in robust_rows:
            ret = r.get("bt_total_return")
            pf = r.get("bt_profit_factor")
            mc = r.get("mc_prob_profit")
            ret_s = f"{ret*100:.2f}%" if isinstance(ret, (int, float)) else "-"
            pf_s = f"{pf:.2f}" if isinstance(pf, (int, float)) else "-"
            mc_s = f"{mc:.2f}" if isinstance(mc, (int, float)) else "-"
            lines.append(f"| {r.get('instrument','-')} | {r.get('name','-')} | "
                         f"{ret_s} | {pf_s} | {mc_s} |")
    else:
        lines += ["_Nessuna strategia robusta in questa run._",
                  "",
                  "> Su dati sintetici (random walk) e' il risultato CORRETTO: "
                  "la piattaforma non lascia passare pattern illusori. Collega "
                  "dati di mercato reali (CSV in /datasets) per risultati significativi."]

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
