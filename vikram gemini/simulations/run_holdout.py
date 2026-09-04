"""Run the untouched 2025-current holdout for the frozen Gemini model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from news_overlay import combined_scores
from price_engine import (
    FEATURE_NAMES,
    cross_sectional_ranks,
    equal_weight_benchmark,
    load_market_data,
    rebalance_positions,
    score_from_weights,
    simulate_scores,
)


UNIVERSE_PATH = MODEL_ROOT / "data" / "training" / "universe_200.csv"
PRICE_MODEL_PATH = MODEL_ROOT / "model" / "price_model_200.json"
NEWS_MODEL_PATH = MODEL_ROOT / "model" / "gemini_news_model_200.json"
OUTPUT_DIR = MODEL_ROOT / "data" / "simulation"
TRADES_DIR = OUTPUT_DIR / "trades"
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)


def scalar_summary(result: dict) -> dict:
    return {
        key: (
            value.isoformat()
            if hasattr(value, "isoformat")
            else value
        )
        for key, value in result.items()
        if key not in {"equity", "trades", "final_weights"}
    } | {"final_weights": result["final_weights"]}


def main() -> None:
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].astype(str).tolist()
    price_model = json.loads(PRICE_MODEL_PATH.read_text(encoding="utf-8"))
    news_model = json.loads(NEWS_MODEL_PATH.read_text(encoding="utf-8"))
    cache_root = (
        MODEL_ROOT
        / "data"
        / "cache"
        / news_model["gemini_model"].replace(".", "_").replace("-", "_")
    )
    if news_model["price_model_fingerprint"] != price_model["fingerprint"]:
        raise RuntimeError("Frozen news and price models have mismatched fingerprints.")
    data = load_market_data(tickers)
    end = data.dates[-1].date().isoformat()
    start = "2025-01-01"
    ranked = cross_sectional_ranks(data.features)
    feature_weights = np.asarray(
        [price_model["weights"].get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    price_scores = score_from_weights(ranked, feature_weights)
    holding_days = int(price_model["holding_days"])
    rebalances = rebalance_positions(data, start, end, holding_days)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    full_results: dict[str, dict] = {}
    audit_rows = None
    for alpha in ALPHAS:
        if alpha == 0:
            scores = price_scores
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
        for execution in ("close", "next_open"):
            result = simulate_scores(
                data,
                scores,
                start=start,
                end=end,
                holding_days=holding_days,
                top_k=int(price_model["top_k"]),
                keep_rank=int(price_model["keep_rank"]),
                fee=float(price_model["fee"]),
                execution=execution,
            )
            selected = abs(alpha - float(news_model["news_alpha"])) < 1e-12
            label = f"alpha_{alpha:g}_{execution}"
            rows.append(
                {
                    "model": "frozen_gemini" if selected else "weight_diagnostic",
                    "news_alpha": alpha,
                    "execution": execution,
                    "selected_on_2024": selected,
                    "first_date": result["first_date"].date().isoformat(),
                    "last_date": result["last_date"].date().isoformat(),
                    "return": result["return"],
                    "final_balance": result["final_balance"],
                    "max_drawdown": result["max_drawdown"],
                    "annual_volatility": result["annual_volatility"],
                    "sharpe": result["sharpe"],
                    "turnover": result["turnover"],
                }
            )
            full_results[label] = scalar_summary(result)
            result["equity"].rename("equity").to_csv(
                OUTPUT_DIR / f"equity_{label}.csv", index_label="date"
            )
            result["trades"].to_csv(
                TRADES_DIR / f"trades_{label}.csv", index=False
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "holdout_weight_diagnostics.csv", index=False)
    if audit_rows is not None:
        pd.DataFrame(audit_rows).to_csv(
            OUTPUT_DIR / "gemini_cache_audit_holdout.csv", index=False
        )
    benchmark = equal_weight_benchmark(data, start, end)

    prior_dir = (
        REPO_ROOT
        / "26th July onwards"
        / "BEST_July26_Momentum_69.30pct"
        / "data"
        / "simulation"
    )
    comparison = {
        "test_protocol": {
            "start": start,
            "end": end,
            "news_alpha_selected_only_on": "2024-01-01 through 2024-12-31",
            "diagnostic_warning": (
                "Other alpha results are reported transparently but were not "
                "eligible to replace the frozen 2024-selected alpha."
            ),
        },
        "price_model_fingerprint": price_model["fingerprint"],
        "news_model_fingerprint": news_model["fingerprint"],
        "selected_news_alpha": news_model["news_alpha"],
        "results": full_results,
        "equal_weight_200_stock_benchmark": benchmark,
        "prior_july26_model": json.loads(
            (prior_dir / "july26_summary.json").read_text(encoding="utf-8")
        ),
        "rl_codex_2": json.loads(
            (prior_dir / "rl2_benchmark.json").read_text(encoding="utf-8")
        ),
    }
    (OUTPUT_DIR / "holdout_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\n200-stock equal-weight benchmark: {benchmark['return']:.2%}")
    print(f"Selected news alpha: {news_model['news_alpha']}")
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
