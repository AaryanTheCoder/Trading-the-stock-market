"""Select Gemini position-size influence on complete 2024 caches only."""

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

from news_overlay import news_tilt_targets
from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    rebalance_positions,
    score_from_weights,
    simulate_target_weights,
)


PRICE_MODEL_PATH = MODEL_ROOT / "model" / "api_efficient_price_model_200.json"
CACHE_TAG = "gemini_2_5_flash_api_efficient"
CACHE_ROOT = MODEL_ROOT / "data" / "cache" / CACHE_TAG
OUTPUT_PATH = MODEL_ROOT / "model" / "gemini_news_model_200.json"
ALPHAS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)


def main() -> None:
    universe = pd.read_csv(MODEL_ROOT / "data" / "training" / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    model = json.loads(PRICE_MODEL_PATH.read_text(encoding="utf-8"))
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    weights = np.asarray(
        [model["weights"].get(name, 0.0) for name in FEATURE_NAMES], dtype=float
    )
    price_scores = score_from_weights(ranked, weights)
    rebalances = rebalance_positions(
        data, "2024-01-01", "2024-12-31", int(model["holding_days"])
    )
    rows = []
    audit_rows = None
    for alpha in ALPHAS:
        targets, audit = news_tilt_targets(
            price_scores,
            data.dates,
            rebalances,
            CACHE_ROOT,
            tickers,
            alpha,
            int(model["gemini_candidate_count"]),
        )
        if audit_rows is None:
            audit_rows = audit
        results = {}
        for execution in ("close", "next_open"):
            results[execution] = simulate_target_weights(
                data,
                targets,
                start="2024-01-01",
                end="2024-12-31",
                holding_days=int(model["holding_days"]),
                fee=float(model["fee"]),
                execution=execution,
            )
        close = results["close"]
        next_open = results["next_open"]
        objective = (
            0.60 * close["return"]
            + 0.40 * next_open["return"]
            + 0.10 * min(close["sharpe"], next_open["sharpe"])
            + 0.20 * min(close["max_drawdown"], next_open["max_drawdown"])
        )
        rows.append(
            {
                "news_alpha": alpha,
                "close_return": close["return"],
                "next_open_return": next_open["return"],
                "close_sharpe": close["sharpe"],
                "next_open_sharpe": next_open["sharpe"],
                "worst_drawdown": min(
                    close["max_drawdown"], next_open["max_drawdown"]
                ),
                "objective": objective,
            }
        )
    sweep = pd.DataFrame(rows).sort_values("objective", ascending=False)
    sweep.to_csv(
        MODEL_ROOT
        / "data"
        / "training"
        / "gemini_allocation_weight_sweep_2024.csv",
        index=False,
    )
    pd.DataFrame(audit_rows).to_csv(
        MODEL_ROOT
        / "data"
        / "training"
        / "gemini_api_efficient_cache_audit_2024.csv",
        index=False,
    )
    winner = sweep.iloc[0]
    artifact = {
        "name": "Gemini 2.5 Flash grounded-news allocation overlay",
        "price_model": PRICE_MODEL_PATH.name,
        "price_model_fingerprint": model["fingerprint"],
        "gemini_model": "gemini-2.5-flash",
        "cache_tag": CACHE_TAG,
        "prompt_version": "news-score-v1",
        "universe_size": 200,
        "news_candidates": int(model["gemini_candidate_count"]),
        "combination": (
            "select top-10 price candidates; weights are softmax("
            "news_alpha * audited_news_rank)"
        ),
        "news_alpha": float(winner["news_alpha"]),
        "selection_start": "2024-01-01",
        "selection_end": "2024-12-31",
        "holdout_start": "2025-01-01",
        "validation": {
            key: float(winner[key])
            for key in (
                "close_return",
                "next_open_return",
                "close_sharpe",
                "next_open_sharpe",
                "worst_drawdown",
                "objective",
            )
        },
        "retrospective_search_warning": (
            "Historical search is date-filtered and source-audited, but "
            "retrospective ranking can still leak hindsight."
        ),
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    artifact["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(sweep.to_string(index=False))
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

