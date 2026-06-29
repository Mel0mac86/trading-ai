"""
Entry point a riga di comando della piattaforma.

Uso (tutto opzionale: senza argomenti esegue una run completa con i default):

    python -m trading_ai                      # autopilota, zero input
    python -m trading_ai --instruments EURUSD GBPUSD
    python -m trading_ai --config config/default.yaml
    python -m trading_ai --no-download        # solo dati locali/sintetici

Pensato anche per scheduler/CI: codice di uscita 0 se la run completa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading_ai import __version__
from trading_ai.autopilot import AutopilotConfig, run_autopilot
from trading_ai.pipeline import PipelineConfig


def _load_yaml_config(path: Path) -> AutopilotConfig:
    """Costruisce una AutopilotConfig da un file YAML (campi tutti opzionali)."""
    import yaml  # dipendenza gia' presente (PyYAML)

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pipe_data = data.pop("pipeline", {}) or {}
    pipeline = PipelineConfig(**pipe_data)             # i campi non presenti restano ai default
    return AutopilotConfig(pipeline=pipeline, **data)


def _build_config(args: argparse.Namespace) -> AutopilotConfig:
    """Determina la configurazione da file YAML e/o argomenti CLI."""
    if args.config:
        cfg = _load_yaml_config(Path(args.config))
    else:
        cfg = AutopilotConfig()

    # Gli argomenti CLI hanno priorita' sui default/sul file.
    if args.instruments:
        cfg.instruments = args.instruments
    if args.no_download:
        cfg.allow_download = False
    if args.kaggle_dataset:
        cfg.kaggle_dataset = args.kaggle_dataset
    if args.bars:
        cfg.synthetic_bars = args.bars
    if args.output:
        cfg.output_root = Path(args.output)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Funzione principale della CLI. Ritorna il codice di uscita."""
    parser = argparse.ArgumentParser(
        prog="trading_ai",
        description="Piattaforma AI Trading — autopilota end-to-end (zero input).",
    )
    parser.add_argument("--version", action="version", version=f"trading_ai {__version__}")
    parser.add_argument("--config", help="Percorso di un file YAML di configurazione.")
    parser.add_argument("--instruments", nargs="+",
                        help="Lista di strumenti (es. EURUSD XAUUSD US500).")
    parser.add_argument("--no-download", action="store_true",
                        help="Non scaricare dati online: usa solo CSV locali o sintetici.")
    parser.add_argument("--kaggle-dataset",
                        help="Slug Kaggle 'owner/dataset' da cui scaricare i dati "
                             "(richiede KAGGLE_API_TOKEN o ~/.kaggle).")
    parser.add_argument("--bars", type=int,
                        help="Numero di barre del fallback sintetico.")
    parser.add_argument("--output", help="Cartella radice degli output della run.")
    args = parser.parse_args(argv)

    cfg = _build_config(args)
    manifest = run_autopilot(cfg)

    # Riepilogo finale su stdout (oltre ai file generati).
    print(f"\nRun {manifest['run_id']} completata.")
    print(f"Strategie robuste totali: {manifest['total_robust']}")
    print(f"Output: {manifest['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
