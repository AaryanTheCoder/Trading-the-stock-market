"""Exploratory only: apply Gemini on fully cached 2024 dates and nowhere else.

This script deliberately does NOT freeze a model. It exists to inspect response
behavior while API quota prevents complete validation coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from news_overlay import load_news_scores, tie_aware_rank
from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    rebalance_positions,
    score_from_weights,
    simulate_scores,
)


ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)
OUTPUT = MODEL_ROOT / "data" / "training" / "partial_gemini_weight_exploration.csv"
METADATA = (
    MODEL_ROOT / "data" / "training" / "partial_gemini_weight_exploration.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.1-flash-lite",
        ),
        default="gemini-2.5-flash",
    )
    args = parser.parse_args()
    cache_root = (
        MODEL_ROOT
        / "data"
        / "cache"
        / args.model.replace(".", "_").replace("-", "_")
    )
    universe = pd.read_csv(MODEL_ROOT / "data" / "training" / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    model = json.loads(
        (MODEL_ROOT / "model" / "price_model_200.json").read_text(encoding="utf-8")
    )
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    weights = np.asarray(
        [model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=float,
    )
    price_scores = score_from_weights(ranked, weights)
    rebalances = rebalance_positions(
        data, "2024-01-01", "2024-12-31", int(model["holding_days"])
    )
    complete: dict[int, tuple[list[int], np.ndarray]] = {}
    skipped: dict[str, str] = {}
    for position in rebalances:
        price_row = price_scores[position]
        ranking = np.argsort(np.nan_to_num(price_row, nan=-np.inf))[::-1]
        candidates = [
            int(number)
            for number in ranking
            if np.isfinite(price_row[number])
        ][: int(model["keep_rank"])]
        candidate_tickers = [tickers[number] for number in candidates]
        as_of = data.dates[position].date()
        try:
            news, _ = load_news_scores(
                cache_root, as_of, tickers, candidate_tickers
            )
        except (FileNotFoundError, ValueError) as error:
            skipped[as_of.isoformat()] = str(error)
            continue
        complete[int(position)] = (
            candidates,
            tie_aware_rank(news[candidates]),
        )

    rows = []
    for alpha in ALPHAS:
        scores = price_scores.copy()
        for position, (candidates, news_rank) in complete.items():
            combined = np.full(len(tickers), np.nan)
            combined[candidates] = (
                price_scores[position, candidates] + alpha * news_rank
            )
            scores[position] = combined
        for execution in ("close", "next_open"):
            result = simulate_scores(
                data,
                scores,
                start="2024-01-01",
                end="2024-12-31",
                holding_days=int(model["holding_days"]),
                top_k=int(model["top_k"]),
                keep_rank=int(model["keep_rank"]),
                fee=float(model["fee"]),
                execution=execution,
            )
            rows.append(
                {
                    "news_alpha": alpha,
                    "execution": execution,
                    "return": result["return"],
                    "max_drawdown": result["max_drawdown"],
                    "sharpe": result["sharpe"],
                    "turnover": result["turnover"],
                    "gemini_rebalances": len(complete),
                    "total_rebalances": len(rebalances),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT, index=False)
    metadata = {
        "status": "PROVISIONAL_NOT_FOR_MODEL_SELECTION",
        "fully_covered_dates": [
            data.dates[position].date().isoformat() for position in complete
        ],
        "covered_rebalances": len(complete),
        "required_rebalances": len(rebalances),
        "skipped_dates": skipped,
        "warning": (
            "Gemini was applied only on fully cached dates; all other dates use "
            "price-only scores. This is a response experiment, not validation."
        ),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
