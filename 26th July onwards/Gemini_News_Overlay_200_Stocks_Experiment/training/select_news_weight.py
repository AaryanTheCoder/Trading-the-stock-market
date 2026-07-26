"""Select the Gemini overlay weight on 2024, leaving 2025+ untouched."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from news_overlay import combined_scores
from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    load_market_data,
    rebalance_positions,
    score_from_weights,
    simulate_scores,
)


UNIVERSE_PATH = MODEL_ROOT / "data" / "training" / "universe_200.csv"
PRICE_MODEL_PATH = MODEL_ROOT / "model" / "price_model_200.json"
OUTPUT_MODEL_PATH = MODEL_ROOT / "model" / "gemini_news_model_200.json"
SWEEP_PATH = MODEL_ROOT / "data" / "training" / "gemini_weight_sweep_2024.csv"
AUDIT_PATH = MODEL_ROOT / "data" / "training" / "gemini_cache_audit_2024.csv"
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)


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
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].astype(str).tolist()
    price_model = json.loads(PRICE_MODEL_PATH.read_text(encoding="utf-8"))
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    weights = np.asarray(
        [price_model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    price_scores = score_from_weights(ranked, weights)
    holding_days = int(price_model["holding_days"])
    rebalances = rebalance_positions(data, "2024-01-01", "2024-12-31", holding_days)

    rows = []
    audit_rows = None
    for alpha in ALPHAS:
        if alpha == 0:
            scores = price_scores
            current_audit = []
        else:
            scores, current_audit = combined_scores(
                price_scores,
                data.dates,
                rebalances,
                cache_root,
                tickers,
                alpha,
                int(price_model["keep_rank"]),
            )
            if audit_rows is None:
                audit_rows = current_audit
        close = simulate_scores(
            data,
            scores,
            start="2024-01-01",
            end="2024-12-31",
            holding_days=holding_days,
            top_k=int(price_model["top_k"]),
            keep_rank=int(price_model["keep_rank"]),
            fee=float(price_model["fee"]),
            execution="close",
        )
        next_open = simulate_scores(
            data,
            scores,
            start="2024-01-01",
            end="2024-12-31",
            holding_days=holding_days,
            top_k=int(price_model["top_k"]),
            keep_rank=int(price_model["keep_rank"]),
            fee=float(price_model["fee"]),
            execution="next_open",
        )
        # Favor return, but require that the result survives next-open execution
        # and penalize drawdown. Selection never sees 2025+.
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
                "close_turnover": close["turnover"],
                "next_open_turnover": next_open["turnover"],
                "objective": objective,
            }
        )

    sweep = pd.DataFrame(rows).sort_values("objective", ascending=False)
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(SWEEP_PATH, index=False)
    if audit_rows is not None:
        pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
    winner = sweep.iloc[0]
    model = {
        "name": "Gemini 2.5 Flash grounded-news overlay, 200 stocks",
        "price_model": "price_model_200.json",
        "price_model_fingerprint": price_model["fingerprint"],
        "gemini_model": args.model,
        "prompt_version": "news-score-v1",
        "news_score": "audited raw score / 100 * confidence, then cross-sectional rank",
        "combination": "price_rank + news_alpha * news_rank",
        "news_alpha": float(winner["news_alpha"]),
        "candidate_universe": 200,
        "weight_selection_start": "2024-01-01",
        "weight_selection_end": "2024-12-31",
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
            "Historical Google Search is constrained by a date filter and "
            "source-date audit, but retrospective search ranking can still "
            "introduce look-ahead bias. Treat as research, not live evidence."
        ),
    }
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    model["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUTPUT_MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(sweep.to_string(index=False))
    print(f"\nFrozen news model: {OUTPUT_MODEL_PATH}")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
