"""Collect grounded Gemini news scores for all 200 stocks at rebalances."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from gemini_news import StockDescriptor, score_batch, write_cache
from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    rebalance_positions,
    score_from_weights,
)


UNIVERSE_PATH = MODEL_ROOT / "data" / "training" / "universe_200.csv"
PERIODS = {
    "validation": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-12-31"),
}


def chunks(items: list[StockDescriptor], size: int):
    for start in range(0, len(items), size):
        yield start // size, items[start : start + size]


def cache_path(cache_root: Path, as_of: date, batch_number: int) -> Path:
    return cache_root / as_of.isoformat() / f"batch_{batch_number:02d}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--period", choices=("validation", "test", "both"), default="both"
    )
    parser.add_argument(
        "--model",
        choices=(
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.1-flash-lite",
        ),
        default="gemini-2.5-flash",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Stop after this many uncached API calls (useful for a smoke test).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace matching cached responses."
    )
    parser.add_argument(
        "--price-model",
        default="price_model_200.json",
        help="Filename under model/ that controls cadence and candidates.",
    )
    parser.add_argument(
        "--cache-tag",
        help="Optional cache folder name; defaults to the Gemini model slug.",
    )
    parser.add_argument(
        "--pause", type=float, default=0.25, help="Seconds between successful calls."
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 25:
        parser.error("--batch-size must be between 1 and 25")
    cache_slug = args.cache_tag or args.model.replace(".", "_").replace("-", "_")
    if "/" in cache_slug or "\\" in cache_slug or cache_slug in {".", ".."}:
        parser.error("--cache-tag must be one safe folder name")
    cache_root = MODEL_ROOT / "data" / "cache" / cache_slug

    universe = pd.read_csv(UNIVERSE_PATH)
    stocks = [
        StockDescriptor(
            ticker=str(row.ticker),
            company=str(row.company),
            sector=str(row.sector),
        )
        for row in universe.itertuples(index=False)
    ]
    if len(stocks) != 200:
        raise RuntimeError(f"Expected 200 stocks, found {len(stocks)}.")
    model_path = MODEL_ROOT / "model" / args.price_model
    if model_path.parent != MODEL_ROOT / "model":
        parser.error("--price-model must be a filename under model/")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    data = load_market_data([stock.ticker for stock in stocks])
    ranked = cross_sectional_ranks(data.features)
    feature_weights = np.asarray(
        [model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    price_scores = score_from_weights(ranked, feature_weights)
    candidate_count = int(model.get("gemini_candidate_count", model["keep_rank"]))
    selected_periods = (
        tuple(PERIODS) if args.period == "both" else (args.period,)
    )
    calls = 0
    cached = 0
    for period_name in selected_periods:
        start, requested_end = PERIODS[period_name]
        end = min(pd.Timestamp(requested_end), data.dates[-1]).date().isoformat()
        positions = rebalance_positions(
            data, start, end, int(model["holding_days"])
        )
        print(
            f"{period_name}: {len(positions)} rebalances, "
            f"{data.dates[positions[0]].date()} to {data.dates[positions[-1]].date()}"
        )
        for position in positions:
            as_of = data.dates[position].date()
            ranking = np.argsort(
                np.nan_to_num(price_scores[position], nan=-np.inf)
            )[::-1]
            candidate_indices = [
                int(number)
                for number in ranking
                if np.isfinite(price_scores[position, number])
            ][:candidate_count]
            candidates = [stocks[number] for number in candidate_indices]
            if len(candidates) != candidate_count:
                raise RuntimeError(
                    f"Only {len(candidates)} eligible candidates on {as_of}."
                )
            for batch_number, stock_batch in chunks(candidates, args.batch_size):
                destination = cache_path(cache_root, as_of, batch_number)
                if destination.exists() and not args.force:
                    cached += 1
                    continue
                if args.max_batches is not None and calls >= args.max_batches:
                    print(f"Stopped after {calls} new calls; {cached} caches reused.")
                    return
                result = score_batch(stock_batch, as_of, args.model)
                write_cache(destination, result)
                calls += 1
                passed = result["metadata"]["audit_pass_count"]
                print(
                    f"{as_of} batch {batch_number:02d}: "
                    f"{passed}/{len(stock_batch)} passed audit"
                )
                time.sleep(max(0.0, args.pause))
    print(f"Complete: {calls} new calls, {cached} caches reused.")


if __name__ == "__main__":
    main()
