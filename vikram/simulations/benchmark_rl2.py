"""Re-run both saved RL Codex 2 policies on the exact local comparison panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

import pandas as pd
from stable_baselines3 import PPO

from research_engine import SIMULATION_DATA_DIR, STOCKS, TRADES_DATA_DIR, _read_one


REFERENCE_ROOT = (
    REPO_ROOT
    / "Before 26th July"
    / "Models"
    / "BEST_RL_Codex_2_45.95pct"
)
REFERENCE_SOURCE_DIR = REFERENCE_ROOT / "source"
REFERENCE_MODEL_DIR = REFERENCE_ROOT / "model"
sys.path.insert(0, str(REFERENCE_SOURCE_DIR))

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
        (
            "RL Codex 2 long-term",
            REFERENCE_MODEL_DIR / "rl_codex_2_long_term_policy",
        ),
        ("RL Codex 2 original", REFERENCE_MODEL_DIR / "rl_codex_2_policy"),
    )
    metrics = []
    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in policies:
        model = PPO.load(path, device="cpu")
        result = simulate_portfolio(model, universe)
        metrics.append(serializable_metrics(name, result))
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        pd.DataFrame(result["trade_log"]).to_csv(
            TRADES_DATA_DIR / f"{safe_name}_trades.csv", index=False
        )
        print(
            f"{name}: {result['return']:+.2%}, "
            f"drawdown {result['max_drawdown']:.2%}"
        )
    path = SIMULATION_DATA_DIR / "rl2_benchmark.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
