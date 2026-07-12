"""CODEX SIMULATION 2: trade 100 large US stocks through 2025 and 2026 so far.

The PPO policy examines all 100 stocks approximately once per month. It invests
one shared $100,000 account equally among at most ten sufficiently confident
long choices. Existing holdings receive a ranking buffer to prevent needless
turnover. Every dollar bought or sold costs 0.1%.

No learning occurs inside this program. However, the slower strategy was
designed after inspecting an earlier 2025 result, so this run is now a
development backtest rather than a genuinely untouched test.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from RL_CODEX_2 import (
    HOLDING_PERIOD_DAYS,
    KEEP_HOLDING_WITHIN_TOP_RANK,
    MIN_LONG_PROBABILITY,
    MODEL_PATH,
    SCRIPT_DIRECTORY,
    STOCKS,
    TOP_STOCKS_TO_HOLD,
    TRADING_FEE,
    download_stocks,
    prepare_universe,
    select_period,
    simulate_portfolio,
)


STARTING_CASH = 100_000
DOWNLOAD_START = "2024-01-01"  # Feature warm-up only.

# yfinance's end date is exclusive. Tomorrow asks for the newest available
# prices, while 2027-01-01 prevents this experiment from drifting beyond 2026.
DOWNLOAD_END = min(date.today() + timedelta(days=1), date(2027, 1, 1)).isoformat()
SIMULATION_START = "2025-01-01"
SIMULATION_END = "2026-12-31"
TRADE_LOG_PATH = SCRIPT_DIRECTORY / "CODEX_SIMULATION_2_TRADES.csv"


def main() -> None:
    print("=" * 76)
    print("CODEX SIMULATION 2 - $100,000 ACROSS 100 STOCKS IN 2025-2026")
    print("=" * 76)

    model_file = Path(f"{MODEL_PATH}.zip")
    if not model_file.exists():
        raise FileNotFoundError(
            f"{model_file} does not exist. Run RL_CODEX_2.py first."
        )

    downloaded = download_stocks(DOWNLOAD_START, DOWNLOAD_END)
    prepared = prepare_universe(downloaded)
    simulation_universe = select_period(
        prepared, SIMULATION_START, SIMULATION_END
    )

    if len(simulation_universe) != 100:
        raise AssertionError(
            f"Expected 100 stocks, received {len(simulation_universe)}."
        )
    if any(
        not data.index.year.isin([2025, 2026]).all()
        for data in simulation_universe.values()
    ):
        raise AssertionError("Simulation contains dates outside 2025-2026.")

    print(f"Stocks examined each day: {len(STOCKS)}")
    print(f"Maximum stocks held: {TOP_STOCKS_TO_HOLD}")
    print(f"Decision interval: every {HOLDING_PERIOD_DAYS} trading sessions")
    print(f"Minimum long confidence: {MIN_LONG_PROBABILITY:.0%}")
    print(f"Existing-holding keep zone: top {KEEP_HOLDING_WITHIN_TOP_RANK}")
    print(f"Starting cash: ${STARTING_CASH:,.2f}")
    print(f"Trading fee: {TRADING_FEE:.2%} of every purchase or sale")
    print("Loading the final PPO policy. Learning is disabled...")

    model = PPO.load(MODEL_PATH, device="cpu")
    result = simulate_portfolio(
        model,
        simulation_universe,
        starting_cash=STARTING_CASH,
        top_k=TOP_STOCKS_TO_HOLD,
    )
    excess_return = result["return"] - result["benchmark_return"]

    trade_columns = [
        "date",
        "ticker",
        "action",
        "weight_before",
        "weight_after",
        "weight_change",
        "estimated_dollars_traded",
        "estimated_fee",
        "long_probability",
        "portfolio_before_trades",
    ]
    trades = pd.DataFrame(result["trade_log"], columns=trade_columns)
    trades.to_csv(TRADE_LOG_PATH, index=False)

    print("\n" + "=" * 76)
    print("2025-2026 LONG-TERM STRATEGY DEVELOPMENT RESULTS")
    print("=" * 76)
    print(
        f"Market period              : {result['first_date'].date()} to "
        f"{result['last_date'].date()} ({result['trading_days']} sessions)"
    )
    print(f"Starting balance           : ${result['starting_balance']:,.2f}")
    print(
        f"AI final balance           : ${result['final_balance']:,.2f} "
        f"({result['return']:+.2%})"
    )
    print(
        f"100-stock buy/hold balance : ${result['benchmark_balance']:,.2f} "
        f"({result['benchmark_return']:+.2%})"
    )
    print(f"AI excess return           : {excess_return:+.2%}")
    print(f"AI maximum drawdown        : {result['max_drawdown']:.2%}")
    print(f"Days with portfolio changes: {result['rebalances']}")
    print(f"Total one-way turnover     : {result['total_turnover']:.2f}x")
    print(f"Average stocks held        : {result['average_holdings']:.1f}")
    print(f"Individual trades recorded : {len(trades)}")
    print(f"Trade document             : {TRADE_LOG_PATH}")
    print("=" * 76)

    if result["return"] > result["benchmark_return"]:
        print("The PPO portfolio beat equal-weight buy-and-hold in 2025.")
    else:
        print("Equal-weight buy-and-hold beat the PPO portfolio in 2025.")
    print(
        "This is one survivorship-biased educational experiment, not proof "
        "of a repeatable trading advantage."
    )


if __name__ == "__main__":
    main()
