"""Evaluate the frozen model once on 2025+ and run robustness checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from research_engine import (
    FEATURE_NAMES,
    MODEL_ARTIFACT_DIR,
    SIMULATION_DATA_DIR,
    TRADES_DATA_DIR,
    cross_sectional_ranks,
    equal_weight_benchmark,
    load_market_data,
    score_from_weights,
    simulate_scores,
)


def result_metrics(result: dict) -> dict:
    return {
        "first_date": result["first_date"].date().isoformat(),
        "last_date": result["last_date"].date().isoformat(),
        "final_balance": result["final_balance"],
        "return": result["return"],
        "max_drawdown": result["max_drawdown"],
        "annual_volatility": result["annual_volatility"],
        "sharpe": result["sharpe"],
        "turnover": result["turnover"],
        "average_holdings": result["average_holdings"],
        "final_weights": result["final_weights"],
    }


def main() -> None:
    model_path = MODEL_ARTIFACT_DIR / "july26_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if pd.Timestamp(model["selection_data_end"]) >= pd.Timestamp("2025-01-01"):
        raise AssertionError("Selection/training data leaked into 2025.")
    if pd.Timestamp(model["test_data_start"]) != pd.Timestamp("2025-01-01"):
        raise AssertionError("Unexpected test boundary.")

    data = load_market_data()
    ranked = cross_sectional_ranks(data.features)
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for name, value in model["weights"].items():
        weights[FEATURE_NAMES.index(name)] = float(value)
    scores = score_from_weights(ranked, weights)
    common = {
        "start": "2025-01-01",
        "end": "2026-12-31",
        "holding_days": int(model["holding_days"]),
        "top_k": int(model["top_k"]),
        "keep_rank": int(model["keep_rank"]),
    }

    close_result = simulate_scores(data, scores, **common, execution="close")
    next_open_result = simulate_scores(data, scores, **common, execution="next_open")
    benchmark = equal_weight_benchmark(data, common["start"], common["end"])

    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    close_result["trades"].to_csv(
        TRADES_DATA_DIR / "july26_close_trades.csv", index=False
    )
    next_open_result["trades"].to_csv(
        TRADES_DATA_DIR / "july26_next_open_trades.csv", index=False
    )
    pd.concat(
        {
            "close_execution": close_result["equity"],
            "next_open_execution": next_open_result["equity"],
        },
        axis=1,
    ).to_csv(SIMULATION_DATA_DIR / "july26_equity.csv", index_label="date")

    stress_rows = []
    for execution in ("close", "next_open"):
        for fee in (0.001, 0.002, 0.005):
            result = simulate_scores(data, scores, **common, execution=execution, fee=fee)
            stress_rows.append(
                {
                    "test": "execution_cost",
                    "execution": execution,
                    "fee": fee,
                    "rebalance_offset": 0,
                    "top_k": common["top_k"],
                    "return": result["return"],
                    "max_drawdown": result["max_drawdown"],
                    "sharpe": result["sharpe"],
                    "turnover": result["turnover"],
                }
            )
    for offset in range(common["holding_days"]):
        result = simulate_scores(
            data,
            scores,
            **common,
            execution="next_open",
            rebalance_offset=offset,
        )
        stress_rows.append(
            {
                "test": "schedule_offset",
                "execution": "next_open",
                "fee": 0.001,
                "rebalance_offset": offset,
                "top_k": common["top_k"],
                "return": result["return"],
                "max_drawdown": result["max_drawdown"],
                "sharpe": result["sharpe"],
                "turnover": result["turnover"],
            }
        )
    for top_k in (5, 8, 10, 12, 15, 20):
        result = simulate_scores(
            data,
            scores,
            start=common["start"],
            end=common["end"],
            holding_days=common["holding_days"],
            top_k=top_k,
            keep_rank=top_k * 2,
            execution="next_open",
        )
        stress_rows.append(
            {
                "test": "concentration",
                "execution": "next_open",
                "fee": 0.001,
                "rebalance_offset": 0,
                "top_k": top_k,
                "return": result["return"],
                "max_drawdown": result["max_drawdown"],
                "sharpe": result["sharpe"],
                "turnover": result["turnover"],
            }
        )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(SIMULATION_DATA_DIR / "robustness_checks.csv", index=False)
    offsets = stress.loc[stress["test"] == "schedule_offset", "return"]

    summary = {
        "model_fingerprint": model["fingerprint"],
        "data_last_session": data.dates[-1].date().isoformat(),
        "close_execution": result_metrics(close_result),
        "next_open_execution": result_metrics(next_open_result),
        "equal_weight_100_stock_benchmark": benchmark,
        "schedule_offset_next_open": {
            "count": int(len(offsets)),
            "minimum_return": float(offsets.min()),
            "median_return": float(offsets.median()),
            "maximum_return": float(offsets.max()),
            "fraction_above_45_percent": float((offsets > 0.45).mean()),
        },
    }
    path = SIMULATION_DATA_DIR / "july26_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path.name}, robustness_checks.csv, trades, and equity.")


if __name__ == "__main__":
    main()
