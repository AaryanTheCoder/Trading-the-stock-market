"""Reproduce the frozen 200-stock price-only holdout control."""

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

from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    score_from_weights,
    simulate_scores,
)


def serialize(result: dict) -> dict:
    output = {}
    for key, value in result.items():
        if key in {"equity", "trades"}:
            continue
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-model", default="price_model_200.json")
    parser.add_argument("--output-prefix", default="price_only")
    args = parser.parse_args()
    if "/" in args.price_model or "\\" in args.price_model:
        parser.error("--price-model must be a filename under model/")
    if "/" in args.output_prefix or "\\" in args.output_prefix:
        parser.error("--output-prefix must be one safe filename prefix")
    universe = pd.read_csv(MODEL_ROOT / "data" / "training" / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    model = json.loads(
        (MODEL_ROOT / "model" / args.price_model).read_text(encoding="utf-8")
    )
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    weights = np.asarray(
        [model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=float,
    )
    scores = score_from_weights(ranked, weights)
    output_dir = MODEL_ROOT / "data" / "simulation"
    trade_dir = output_dir / "trades"
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete_price_only_control",
        "model_fingerprint": model["fingerprint"],
        "survivorship_bias_warning": model["survivorship_bias_warning"],
    }
    for execution in ("close", "next_open"):
        result = simulate_scores(
            data,
            scores,
            start="2025-01-01",
            end=data.dates[-1].date().isoformat(),
            holding_days=int(model["holding_days"]),
            top_k=int(model["top_k"]),
            keep_rank=int(model["keep_rank"]),
            fee=float(model["fee"]),
            execution=execution,
        )
        summary[execution] = serialize(result)
        result["equity"].to_csv(
            output_dir / f"{args.output_prefix}_equity_{execution}.csv",
            index_label="date",
        )
        result["trades"].to_csv(
            trade_dir / f"{args.output_prefix}_trades_{execution}.csv", index=False
        )
    (output_dir / f"{args.output_prefix}_baseline.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
