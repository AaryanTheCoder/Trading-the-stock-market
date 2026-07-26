"""Freeze the best pre-2025 validation model into a small JSON artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from research_engine import FEATURE_NAMES, MODEL_ARTIFACT_DIR, TRAINING_DATA_DIR


def main() -> None:
    sweep_path = TRAINING_DATA_DIR / "factor_sweep_validation.csv"
    if not sweep_path.exists():
        raise FileNotFoundError("Run sweep_factors.py before freezing the model.")
    sweep = pd.read_csv(sweep_path).sort_values("objective", ascending=False)
    winner = sweep.iloc[0]
    weights = json.loads(winner["weights"])
    unknown = set(weights) - set(FEATURE_NAMES)
    if unknown:
        raise RuntimeError(f"Unknown feature names in winning model: {sorted(unknown)}")

    model = {
        "name": "July26 pre-2025 cross-sectional momentum model",
        "model_family": "validated linear factor ranker",
        "universe": "RL Codex 2 fixed 100-stock universe",
        "feature_information_cutoff": "signal date close",
        "selection_data_start": "2021-01-01",
        "selection_data_end": "2024-12-31",
        "test_data_start": "2025-01-01",
        "weights": weights,
        "holding_days": int(winner["holding_days"]),
        "top_k": int(winner["top_k"]),
        "keep_rank": int(winner["top_k"]) * 2,
        "fee": 0.001,
        "validation": {
            "return_2021": float(winner["return_2021"]),
            "return_2022": float(winner["return_2022"]),
            "return_2023": float(winner["return_2023"]),
            "return_2024": float(winner["return_2024"]),
            "compounded_return": float(winner["compounded_return"]),
            "mean_sharpe": float(winner["mean_sharpe"]),
            "worst_drawdown": float(winner["worst_drawdown"]),
            "objective": float(winner["objective"]),
        },
        "selection_rule": (
            "Highest pre-2025 objective: compounded four-year return + "
            "0.5*worst calendar-year return + 0.1*mean Sharpe + "
            "0.25*mean calendar-year drawdown"
        ),
        "known_limitations": [
            "Fixed present-day large-cap universe creates survivorship bias.",
            "Final 2025+ result is a development backtest, not live evidence.",
            "Close execution matches RL2 but is optimistic; next-open stress is reported.",
        ],
    }
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    model["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = MODEL_ARTIFACT_DIR / "july26_model.json"
    output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen model: {output.name}")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
