"""Re-run both saved RL Codex 2 policies on the exact local comparison panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from research_engine import STOCKS, WORK_DIR, _read_one


REFERENCE_DIR = WORK_DIR.parent / "CODEX NOT ME"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REFERENCE_DIR))

from RL_CODEX_2 import (  # noqa: E402
    add_features,
    simulate_portfolio,
)


def load_reference_universe() -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in STOCKS:
        frame = _read_one(ticker)
        featured = add_features(frame)
        result[ticker] = featured.loc["2025-01-01":"2026-12-31"].copy()
    return result


def serializable_metrics(name: str, result: dict) -> dict:
    return {
        "policy": name,
        "first_date": result["first_date"].date().isoformat(),
        "last_date": result["last_date"].date().isoformat(),
        "trading_days": result["trading_days"],
        "starting_balance": result["starting_balance"],
        "final_balance": result["final_balance"],
        "return": result["return"],
        "benchmark_balance": result["benchmark_balance"],
        "benchmark_return": result["benchmark_return"],
        "max_drawdown": result["max_drawdown"],
        "rebalances": result["rebalances"],
        "turnover": result["total_turnover"],
        "average_holdings": result["average_holdings"],
    }


def main() -> None:
    universe = load_reference_universe()
    policies = (
        ("RL Codex 2 long-term", REFERENCE_DIR / "rl_codex_2_long_term_policy"),
        ("RL Codex 2 original", REFERENCE_DIR / "rl_codex_2_policy"),
    )
    metrics = []
    for name, path in policies:
        model = PPO.load(path, device="cpu")
        result = simulate_portfolio(model, universe)
        metrics.append(serializable_metrics(name, result))
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        pd.DataFrame(result["trade_log"]).to_csv(
            WORK_DIR / f"{safe_name}_trades.csv", index=False
        )
        print(
            f"{name}: {result['return']:+.2%}, "
            f"drawdown {result['max_drawdown']:.2%}"
        )
    path = WORK_DIR / "rl2_benchmark.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
