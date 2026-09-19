"""Evaluate the frozen 2024-selected Gemini allocation alpha on 2025+."""

from __future__ import annotations

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


ALPHAS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)


def main() -> None:
    universe = pd.read_csv(MODEL_ROOT / "data" / "training" / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    news_model = json.loads(
        (MODEL_ROOT / "model" / "gemini_news_model_200.json").read_text(
            encoding="utf-8"
        )
    )
    price_model = json.loads(
        (MODEL_ROOT / "model" / news_model["price_model"]).read_text(
            encoding="utf-8"
        )
    )
    if news_model["price_model_fingerprint"] != price_model["fingerprint"]:
        raise RuntimeError("Price/news model fingerprint mismatch.")
    cache_root = MODEL_ROOT / "data" / "cache" / news_model["cache_tag"]
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    weights = np.asarray(
        [price_model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=float,
    )
    price_scores = score_from_weights(ranked, weights)
    start = "2025-01-01"
    end = data.dates[-1].date().isoformat()
    rebalances = rebalance_positions(
        data, start, end, int(price_model["holding_days"])
    )
    output_dir = MODEL_ROOT / "data" / "simulation"
    trades_dir = output_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_results = {}
    for alpha in ALPHAS:
        targets, _ = news_tilt_targets(
            price_scores,
            data.dates,
            rebalances,
            cache_root,
            tickers,
            alpha,
            int(price_model["gemini_candidate_count"]),
        )
        for execution in ("close", "next_open"):
            result = simulate_target_weights(
                data,
                targets,
                start=start,
                end=end,
                holding_days=int(price_model["holding_days"]),
                fee=float(price_model["fee"]),
                execution=execution,
            )
            selected = abs(alpha - news_model["news_alpha"]) < 1e-12
            rows.append(
                {
                    "news_alpha": alpha,
                    "execution": execution,
                    "selected_on_2024": selected,
                    "return": result["return"],
                    "final_balance": result["final_balance"],
                    "max_drawdown": result["max_drawdown"],
                    "sharpe": result["sharpe"],
                    "turnover": result["turnover"],
                }
            )
            if selected:
                selected_results[execution] = {
                    key: (
                        value.isoformat()
                        if hasattr(value, "isoformat")
                        else value
                    )
                    for key, value in result.items()
                    if key not in {"equity", "trades"}
                }
                result["equity"].to_csv(
                    output_dir / f"gemini_frozen_equity_{execution}.csv",
                    index_label="date",
                )
                result["trades"].to_csv(
                    trades_dir / f"gemini_frozen_trades_{execution}.csv",
                    index=False,
                )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(
        output_dir / "gemini_holdout_alpha_diagnostics.csv", index=False
    )
    summary = {
        "status": "complete",
        "news_model_fingerprint": news_model["fingerprint"],
        "selected_news_alpha": news_model["news_alpha"],
        "selected_results": selected_results,
        "post_holdout_alpha_diagnostics_warning": (
            "All alphas are shown for transparency; only the 2024-selected "
            "alpha is the frozen model."
        ),
    }
    (output_dir / "gemini_holdout_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(diagnostics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

