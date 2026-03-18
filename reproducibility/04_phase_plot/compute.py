#!/usr/bin/env python3
"""Extract phase plot data (VUN vs loss) from AutoGraph training logs."""

import json

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_jsonl,
)

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "04_phase_plot" / "results"
)


@hydra.main(
    config_path="../configs", config_name="04_phase_plot", version_base=None
)
def main(cfg: DictConfig) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_files = {
        "sbm_small": DATA_DIR
        / "AUTOGRAPH"
        / "logs"
        / "sbm_proc_small_metrics.csv",
        "sbm_large": DATA_DIR
        / "AUTOGRAPH"
        / "logs"
        / "sbm_proc_large_metrics.csv",
    }

    results = {}
    for name, log_path in log_files.items():
        if not log_path.exists():
            logger.warning("Log file not found: {}", log_path)
            continue
        df = pd.read_csv(log_path)
        loss_col = "val/loss_epoch"
        vun_col = "val/valid_unique_novel_mle"
        if loss_col not in df.columns or vun_col not in df.columns:
            logger.warning("Expected columns not found in {}", log_path)
            continue
        x = df[loss_col].dropna()
        y = df[vun_col].dropna()
        n = min(len(x), len(y))
        results[name] = {
            "val_loss": x.iloc[:n].tolist(),
            "vun": y.iloc[:n].tolist(),
            "steps": list(range(n)),
        }
        logger.info("Extracted {} points for {}", n, name)

    if results:
        out_path = RESULTS_DIR / "phase_plot.json"
        out_path.write_text(json.dumps(results, indent=2))
        logger.success("Saved to {}", out_path)
        maybe_append_jsonl(
            {
                "experiment": "04_phase_plot",
                "script": "compute.py",
                "status": "ok",
                "output_path": str(out_path),
                "series": list(results.keys()),
            }
        )
    else:
        logger.warning("No data. Download AutoGraph training logs first.")
        maybe_append_jsonl(
            {
                "experiment": "04_phase_plot",
                "script": "compute.py",
                "status": "skipped",
                "reason": "no_phase_plot_data",
            }
        )


if __name__ == "__main__":
    main()
