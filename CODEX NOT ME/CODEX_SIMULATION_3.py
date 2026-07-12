"""CODEX SIMULATION 3: evaluate RL3 from July 11, 2025 to July 11, 2026.

Run RL_CODEX_3.py first.  This program never trains or selects a model.  It
loads the frozen through-July-10-2025 policy and compares it with:

1. SPY buy-and-hold.
2. A monthly equal-weight portfolio from the same current-member universe.
3. A simple 60-day momentum portfolio using RL3's same execution/risk engine.

The AI sees the closing information on a signal date and trades at the next
market open.  Every simulated trade includes spread/slippage/market-impact
costs.  Use --paper-signal to save a dated, non-executing plan for the next
available market open.  This code deliberately has no real-broker order API.

Because RL3 was designed after earlier experiments, this historical report is
still development evidence. Only signals recorded before their future prices
exist are genuine forward paper tests.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from RL_CODEX_3 import (
    BASE_EXECUTION_COST,
    DATA_QUALITY_PATH as TRAINING_DATA_QUALITY_PATH,
    FACTOR_COLUMNS,
    FINAL_TRAIN_END,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_STOCKS_EXAMINED,
    METADATA_PATH,
    MODEL_PATH,
    SCRIPT_DIRECTORY,
    TARGET_ANNUAL_VOLATILITY,
    TOP_STOCKS_TO_HOLD,
    apply_trade_controls,
    build_market_bundle,
    buy_and_hold_spy,
    construct_target_weights,
    simulate_fixed_factor,
    simulate_model,
    simulate_strategy,
)


STARTING_CASH = 100_000.0
DOWNLOAD_START = "2024-01-01"  # Indicator warm-up only.
SIMULATION_START = "2025-07-11"
SIMULATION_END = "2026-07-11"  # Saturday; last market session is July 10.

TRADE_LOG_PATH = SCRIPT_DIRECTORY / "CODEX_SIMULATION_3_TRADES.csv"
EQUITY_PATH = SCRIPT_DIRECTORY / "CODEX_SIMULATION_3_EQUITY.csv"
HOLDINGS_PATH = SCRIPT_DIRECTORY / "CODEX_SIMULATION_3_FINAL_HOLDINGS.csv"
SUMMARY_PATH = SCRIPT_DIRECTORY / "CODEX_SIMULATION_3_SUMMARY.json"
SIMULATION_DATA_QUALITY_PATH = (
    SCRIPT_DIRECTORY / "CODEX_SIMULATION_3_DATA_QUALITY.json"
)
PAPER_DIRECTORY = SCRIPT_DIRECTORY / "RL3_PAPER_SIGNALS"


def _json_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "starting_balance",
        "final_balance",
        "return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "turnover",
        "estimated_execution_costs",
        "average_holdings",
        "trading_days",
    ]
    output: dict[str, Any] = {}
    for key in keys:
        if key not in result:
            continue
        value = result[key]
        output[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    for key in ("first_date", "last_date"):
        if key in result:
            output[key] = pd.Timestamp(result[key]).date().isoformat()
    return output


def _print_result(label: str, result: dict[str, Any]) -> None:
    print(
        f"{label:<27} "
        f"balance=${result['final_balance']:>11,.2f} | "
        f"return={result['return']:>+8.2%} | "
        f"CAGR={result['cagr']:>+7.2%} | "
        f"Sharpe={result['sharpe']:>5.2f} | "
        f"drawdown={result['max_drawdown']:>7.2%}"
    )


def _equity_frame(
    ai: dict[str, Any],
    spy: dict[str, Any],
    equal_weight: dict[str, Any],
    momentum: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.concat(
        {
            "rl3_ai": ai["equity_curve"],
            "spy_buy_hold": spy["equity_curve"],
            "point_in_time_equal_weight": equal_weight["equity_curve"],
            "simple_60d_momentum": momentum["equity_curve"],
        },
        axis=1,
    ).sort_index()
    frame.index.name = "date"
    return frame.ffill()


def write_forward_paper_signal(
    model: PPO,
    bundle: Any,
    metadata: dict[str, Any],
    ai_result: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    """Write a plan; never send an order or pretend simulated holdings are real."""
    signal_date = pd.Timestamp(ai_result["last_date"])
    current_weights = dict(ai_result["final_weights"])
    equity = ai_result["equity_curve"]
    peak = float(equity.cummax().iloc[-1])
    current_value = float(equity.iloc[-1])
    drawdown = current_value / peak - 1
    recent_return = (
        float(equity.iloc[-1] / equity.iloc[max(0, len(equity) - 21)] - 1)
        if len(equity) > 1
        else 0.0
    )
    observation = bundle.observation(
        signal_date,
        current_weights,
        drawdown=drawdown,
        recent_return=recent_return,
        previous_turnover=0.0,
    )
    action, _state = model.predict(observation, deterministic=True)
    desired, scores, desired_exposure = construct_target_weights(
        bundle, signal_date, current_weights, np.asarray(action)
    )
    controlled, planned_turnover = apply_trade_controls(current_weights, desired)

    rows = []
    for ticker in sorted(set(current_weights) | set(controlled)):
        before = float(current_weights.get(ticker, 0.0))
        after = float(controlled.get(ticker, 0.0))
        change = after - before
        if abs(change) < 1e-8:
            continue
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "planned_execution": "NEXT_AVAILABLE_MARKET_OPEN",
                "ticker": ticker,
                "sector": bundle.sectors.get(ticker, "Unknown"),
                "action": "BUY" if change > 0 else "SELL",
                "weight_before_simulated": before,
                "weight_after_target": after,
                "weight_change": change,
                "factor_score": scores.get(ticker, np.nan),
                "desired_total_exposure": desired_exposure,
                "planned_total_weight_change": planned_turnover,
                "model_fingerprint": metadata["fingerprint"],
                "warning": "PAPER PLAN ONLY - DOES NOT READ A BROKER ACCOUNT OR PLACE ORDERS",
            }
        )

    PAPER_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = PAPER_DIRECTORY / f"RL3_PAPER_SIGNAL_{signal_date.date().isoformat()}.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Refusing to rewrite forward evidence. "
            "Use --overwrite-paper-signal only to correct a known file problem."
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--allow-incomplete-data", action="store_true")
    parser.add_argument("--allow-quick-model", action="store_true")
    parser.add_argument("--paper-signal", action="store_true")
    parser.add_argument("--overwrite-paper-signal", action="store_true")
    parser.add_argument(
        "--price-data-dir",
        type=Path,
        default=(Path(os.environ["RL3_PRICE_DATA_DIR"]) if os.getenv("RL3_PRICE_DATA_DIR") else None),
    )
    args = parser.parse_args()

    model_file = Path(f"{MODEL_PATH}.zip")
    if not model_file.exists():
        raise FileNotFoundError(f"{model_file} is missing. Run RL_CODEX_3.py first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} is missing. Retrain RL Codex 3.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("training_end") != FINAL_TRAIN_END:
        raise RuntimeError("RL3 metadata does not contain the expected training cutoff.")
    if metadata.get("quick_model") and not args.allow_quick_model:
        raise RuntimeError(
            "This was trained with --quick and is only a plumbing test. "
            "Train the full model or explicitly pass --allow-quick-model."
        )

    # yfinance's end is exclusive, so add one day to include SIMULATION_END.
    download_end = (
        date.fromisoformat(SIMULATION_END) + timedelta(days=1)
    ).isoformat()
    print("=" * 96)
    print("CODEX SIMULATION 3 - CURRENT S&P 500 RL PORTFOLIO, 2025-07-11 TO 2026-07-11")
    print("=" * 96)
    print(f"Model fingerprint: {metadata['fingerprint']}")
    print(f"Model learned only through: {metadata['training_end']}")
    print(f"Maximum stocks examined: {MAX_STOCKS_EXAMINED}")
    print(f"Maximum stocks held: {TOP_STOCKS_TO_HOLD}")
    print(f"Per-stock cap: {MAX_POSITION_WEIGHT:.1%}")
    print(f"Known-sector cap: {MAX_SECTOR_WEIGHT:.1%}")
    print(f"Target annual volatility: {TARGET_ANNUAL_VOLATILITY:.1%}")
    print(f"Base execution cost: {BASE_EXECUTION_COST:.2%} plus market impact")
    print("Signal at close -> execution at next open")

    bundle = build_market_bundle(
        DOWNLOAD_START,
        download_end,
        refresh=args.refresh_data,
        allow_incomplete=args.allow_incomplete_data,
        price_data_directory=args.price_data_dir,
        quality_report_path=SIMULATION_DATA_QUALITY_PATH,
    )
    available = bundle.dates[
        (bundle.dates >= pd.Timestamp(SIMULATION_START))
        & (bundle.dates <= pd.Timestamp(SIMULATION_END))
    ]
    if len(available) < 2:
        raise RuntimeError("No prices were available in the simulation window.")
    simulation_end = available[-1].date().isoformat()

    print("Loading PPO policy. Learning and model selection are disabled...")
    model = PPO.load(MODEL_PATH, device="cpu")
    ai = simulate_model(
        model,
        bundle,
        SIMULATION_START,
        simulation_end,
        starting_cash=STARTING_CASH,
        record_trades=True,
    )
    spy = buy_and_hold_spy(bundle, SIMULATION_START, simulation_end, STARTING_CASH)
    equal_weight = simulate_strategy(
        bundle,
        SIMULATION_START,
        simulation_end,
        starting_cash=STARTING_CASH,
        equal_weight_all=True,
    )
    momentum = simulate_fixed_factor(
        bundle,
        SIMULATION_START,
        simulation_end,
        "factor_return_60d",
        starting_cash=STARTING_CASH,
    )

    trades = pd.DataFrame(ai["trade_log"])
    trades.to_csv(TRADE_LOG_PATH, index=False)
    equity = _equity_frame(ai, spy, equal_weight, momentum)
    equity.to_csv(EQUITY_PATH)
    holdings = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "sector": bundle.sectors.get(ticker, "Unknown"),
                "final_weight": weight,
                "estimated_value": weight * ai["final_balance"],
            }
            for ticker, weight in sorted(
                ai["final_weights"].items(), key=lambda item: item[1], reverse=True
            )
        ]
    )
    holdings.to_csv(HOLDINGS_PATH, index=False)

    summary = {
        "created_on": date.today().isoformat(),
        "model_fingerprint": metadata["fingerprint"],
        "model_training_end": metadata["training_end"],
        "simulation_start": SIMULATION_START,
        "simulation_end": simulation_end,
        "development_backtest_warning": (
            "The design followed earlier 2025 experiments. Only future signals "
            "recorded before outcomes exist are untouched paper evidence."
        ),
        "rl3_ai": _json_metrics(ai),
        "spy_buy_hold": _json_metrics(spy),
        "current_universe_equal_weight": _json_metrics(equal_weight),
        "simple_60d_momentum": _json_metrics(momentum),
        "training_data_quality_report": str(TRAINING_DATA_QUALITY_PATH),
        "simulation_data_quality_report": str(SIMULATION_DATA_QUALITY_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print(f"DEVELOPMENT BACKTEST: {ai['first_date'].date()} to {ai['last_date'].date()}")
    print("=" * 96)
    _print_result("RL3 PPO portfolio", ai)
    _print_result("SPY buy-and-hold", spy)
    _print_result("Current-universe eligible EW", equal_weight)
    _print_result("Simple 60-day momentum", momentum)
    print("-" * 96)
    print(f"RL3 total absolute weight changes : {ai['turnover']:.2f}x")
    print(f"RL3 estimated execution costs     : ${ai['estimated_execution_costs']:,.2f}")
    print(f"RL3 average stocks held           : {ai['average_holdings']:.1f}")
    print(f"RL3 individual trades             : {len(trades)}")
    print(f"Trade log                         : {TRADE_LOG_PATH}")
    print(f"Daily equity curves               : {EQUITY_PATH}")
    print(f"Final holdings                    : {HOLDINGS_PATH}")
    print(f"Machine-readable summary          : {SUMMARY_PATH}")
    print(f"Training data audit               : {TRAINING_DATA_QUALITY_PATH}")
    print(f"Simulation data audit             : {SIMULATION_DATA_QUALITY_PATH}")
    print("=" * 96)
    print(
        "Do not tune RL3 from this result. Freeze the fingerprint and collect "
        "new dated paper signals. Taxes remain account/jurisdiction dependent "
        "and are not included."
    )

    if args.paper_signal:
        paper_path = write_forward_paper_signal(
            model,
            bundle,
            metadata,
            ai,
            overwrite=args.overwrite_paper_signal,
        )
        print(f"Forward paper plan saved: {paper_path}")
        print("It placed zero broker orders.")


if __name__ == "__main__":
    main()
