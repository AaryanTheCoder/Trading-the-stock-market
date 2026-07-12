"""CODEX SIMULATION 1: evaluate RL CODEX 1 on untouched AAPL data from 2025.

This program does not train or improve the agent. It loads the saved policy,
gives it $100,000 in cash, charges a 0.1% fee whenever stock is bought or sold,
and walks through 2025 once in chronological order.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from RL_CODEX_1 import (
    MODEL_FILE,
    TRADING_FEE,
    add_features,
    download_aapl,
    evaluate_policy,
    select_dates,
)


STARTING_CASH = 100_000

# Most of 2024 is downloaded only to warm up the longest moving average.
# It is removed before simulation, so the agent trades only in 2025.
DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END = "2026-01-05"  # Exclusive and safely includes 2025-12-31.
SIMULATION_START = "2025-01-01"
SIMULATION_END = "2025-12-31"


def main() -> None:
    print("=" * 72)
    print("CODEX SIMULATION 1 - UNTOUCHED AAPL 2025")
    print("=" * 72)

    model_path = Path(f"{MODEL_FILE}.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Run RL_CODEX_1.py first."
        )

    print("Downloading AAPL data and calculating the same training features...")
    downloaded = download_aapl(DOWNLOAD_START, DOWNLOAD_END)
    featured = add_features(downloaded)
    simulation_data = select_dates(
        featured, SIMULATION_START, SIMULATION_END
    )

    if not (simulation_data.index.year == 2025).all():
        raise AssertionError("Simulation data contains dates outside 2025.")

    expected_sessions = int((downloaded.index.year == 2025).sum())
    if len(simulation_data) != expected_sessions:
        raise RuntimeError(
            "The feature warm-up was insufficient to simulate every 2025 "
            f"session: expected {expected_sessions}, got {len(simulation_data)}."
        )

    print(
        f"Simulation period: {simulation_data.index.min().date()} to "
        f"{simulation_data.index.max().date()} "
        f"({len(simulation_data)} trading sessions)"
    )
    print(f"Starting cash: ${STARTING_CASH:,.2f}")
    print(f"Trading fee: {TRADING_FEE:.2%} per buy or sell")
    print("Loading the saved PPO policy; no further learning will occur...")

    model = PPO.load(MODEL_FILE, device="cpu")
    result = evaluate_policy(
        model, simulation_data, starting_cash=STARTING_CASH
    )

    buy_and_hold_balance = STARTING_CASH * (1 + result["market_return"])
    excess_return = result["return"] - result["market_return"]

    print("\n" + "=" * 72)
    print("FINAL 2025 RESULTS")
    print("=" * 72)
    print(f"Starting balance         : ${result['starting_balance']:,.2f}")
    print(
        f"AI final balance         : ${result['final_balance']:,.2f} "
        f"({result['return']:+.2%})"
    )
    print(
        f"Buy-and-hold balance     : ${buy_and_hold_balance:,.2f} "
        f"({result['market_return']:+.2%})"
    )
    print(f"AI excess return         : {excess_return:+.2%}")
    print(f"Maximum AI drawdown      : {result['max_drawdown']:.2%}")
    print(f"Position changes         : {result['position_changes']}")
    print(f"Time long                : {result['long_fraction']:.1%}")
    print(f"Time in cash             : {result['cash_fraction']:.1%}")
    print(f"Time short               : {result['short_fraction']:.1%}")
    print("=" * 72)

    if result["return"] > result["market_return"]:
        print("The AI outperformed buy-and-hold during this one test year.")
    else:
        print("Buy-and-hold outperformed the AI during this one test year.")
    print(
        "One year is not proof of a durable edge; consistency across future "
        "untouched periods matters."
    )


if __name__ == "__main__":
    main()
