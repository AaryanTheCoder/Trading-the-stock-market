"""Stress tests for the frozen 200-stock price foundation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    score_from_weights,
    simulate_scores,
)


def main() -> None:
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
    scores = score_from_weights(ranked, weights)
    start = "2025-01-01"
    end = data.dates[-1].date().isoformat()
    holding = int(model["holding_days"])
    rows = []

    def add(kind: str, setting: str, **parameters) -> None:
        result = simulate_scores(
            data,
            scores,
            start=start,
            end=end,
            holding_days=holding,
            top_k=parameters.pop("top_k", int(model["top_k"])),
            keep_rank=parameters.pop("keep_rank", int(model["keep_rank"])),
            fee=parameters.pop("fee", float(model["fee"])),
            execution=parameters.pop("execution", "next_open"),
            rebalance_offset=parameters.pop("rebalance_offset", 0),
        )
        rows.append(
            {
                "test": kind,
                "setting": setting,
                "return": result["return"],
                "max_drawdown": result["max_drawdown"],
                "sharpe": result["sharpe"],
                "turnover": result["turnover"],
            }
        )

    for offset in range(holding):
        add("rebalance_offset", str(offset), rebalance_offset=offset)
    for top_k in (5, 8, 10, 15, 20, 30):
        add(
            "concentration",
            f"top_{top_k}",
            top_k=top_k,
            keep_rank=top_k * 2,
        )
    for fee in (0.001, 0.002, 0.005, 0.010):
        add("trading_cost", f"{fee:.3%}", fee=fee)
    add("execution", "close", execution="close")
    add("execution", "next_open", execution="next_open")

    frame = pd.DataFrame(rows)
    path = MODEL_ROOT / "data" / "simulation" / "price_robustness.csv"
    frame.to_csv(path, index=False)
    offsets = frame.loc[frame["test"] == "rebalance_offset", "return"]
    summary = {
        "offset_count": len(offsets),
        "offset_minimum_return": float(offsets.min()),
        "offset_median_return": float(offsets.median()),
        "offset_maximum_return": float(offsets.max()),
        "offset_fraction_above_69_30_percent": float((offsets > 0.693).mean()),
    }
    summary_path = (
        MODEL_ROOT / "data" / "simulation" / "price_robustness_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

