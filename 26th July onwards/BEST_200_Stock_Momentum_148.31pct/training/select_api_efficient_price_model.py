"""Freeze a 40-session/top-10 foundation compatible with 20 API calls/day."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))
sys.path.insert(0, str(MODEL_ROOT / "training"))

from price_engine import (
    cross_sectional_ranks,
    load_market_data,
    score_from_weights,
    simulate_scores,
)
from select_price_model import candidate_factors


YEARS = (2021, 2022, 2023, 2024)
HOLDING_DAYS = 40
TOP_K = 10


def main() -> None:
    universe = pd.read_csv(MODEL_ROOT / "data" / "training" / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    rows = []
    for label, weights in candidate_factors():
        scores = score_from_weights(ranked, weights)
        yearly = []
        drawdowns = []
        sharpes = []
        for year in YEARS:
            result = simulate_scores(
                data,
                scores,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                holding_days=HOLDING_DAYS,
                top_k=TOP_K,
                keep_rank=TOP_K,
            )
            yearly.append(result["return"])
            drawdowns.append(result["max_drawdown"])
            sharpes.append(result["sharpe"])
        compounded = float(np.prod(np.asarray(yearly) + 1.0) - 1.0)
        objective = (
            compounded
            + 0.50 * min(yearly)
            + 0.10 * float(np.mean(sharpes))
            + 0.25 * float(np.mean(drawdowns))
        )
        rows.append(
            {
                "label": label,
                "weights": json.dumps(
                    {
                        name: float(value)
                        for name, value in zip(
                            __import__("price_engine").FEATURE_NAMES, weights
                        )
                        if value
                    },
                    sort_keys=True,
                ),
                "return_2021": yearly[0],
                "return_2022": yearly[1],
                "return_2023": yearly[2],
                "return_2024": yearly[3],
                "compounded_return": compounded,
                "mean_sharpe": float(np.mean(sharpes)),
                "worst_drawdown": float(min(drawdowns)),
                "objective": objective,
            }
        )
    sweep = pd.DataFrame(rows).sort_values("objective", ascending=False)
    sweep.to_csv(
        MODEL_ROOT
        / "data"
        / "training"
        / "api_efficient_price_sweep_validation.csv",
        index=False,
    )
    winner = sweep.iloc[0]
    model = {
        "name": "API-efficient 200-stock price foundation",
        "purpose": (
            "40-session cadence and 10 candidates require 17 grounded calls "
            "for 2024 validation plus 2025-current holdout"
        ),
        "universe_size": 200,
        "selection_data_start": "2021-01-01",
        "selection_data_end": "2024-12-31",
        "test_data_start": "2025-01-01",
        "weights": json.loads(winner["weights"]),
        "holding_days": HOLDING_DAYS,
        "top_k": TOP_K,
        "keep_rank": TOP_K,
        "gemini_candidate_count": TOP_K,
        "fee": 0.001,
        "validation": {
            key: float(winner[key])
            for key in (
                "return_2021",
                "return_2022",
                "return_2023",
                "return_2024",
                "compounded_return",
                "mean_sharpe",
                "worst_drawdown",
                "objective",
            )
        },
        "survivorship_bias_warning": "Universe selected from constituents known in 2026.",
    }
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    model["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    path = MODEL_ROOT / "model" / "api_efficient_price_model_200.json"
    path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(sweep.head(15).to_string(index=False))
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()

