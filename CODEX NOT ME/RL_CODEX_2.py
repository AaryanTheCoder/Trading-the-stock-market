"""RL CODEX 2: train one PPO policy across 100 large US stocks.

What changed from RL CODEX 1?
--------------------------------
RL1 learned from only Apple. RL2 learns the same trading lesson from many
different companies and market regimes. At the beginning of every training
episode, the environment randomly chooses one of 100 stocks and a random
one-year period. This encourages the neural network to learn general patterns
instead of memorizing one AAPL chart.

Training and validation:
    Candidate learning data : 2018-2023
    Candidate validation    : 2024
    Final learning data     : 2018-2024
    Simulation period       : 2025 (downloaded only by Simulation 2)

The saved PPO model is a *shared stock-selection policy*. It makes one decision
every 20 trading sessions, roughly once per month. During portfolio simulation,
it ranks the stocks it wants to own and puts the single $100,000 account into at
most ten of its strongest long choices.

Important research limitations:
--------------------------------
* The universe is a fixed collection of companies known to be large today.
  Backtesting that list in 2018 creates survivorship bias: failed or diminished
  former companies are absent.
* Indicators calculated using today's close are assumed tradable at that close.
  A production system needs next-open execution, spreads, and slippage.
* This is educational code, not evidence that PPO can predict future markets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


# ---------------------------------------------------------------------------
# Reproducible experiment settings
# ---------------------------------------------------------------------------

# This is a 100-security, S&P-100-inspired US large-cap universe. It keeps only
# one Alphabet share class and substitutes four long-history large companies
# for recent constituents that did not trade throughout the training period.
STOCKS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE",
    "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GILD", "GM", "GOOGL",
    "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO",
    "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE", "NFLX", "NKE", "NOW",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "QCOM", "RTX", "SBUX", "SCHW",
    "SO", "SPG", "T", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
    "USB", "V", "VZ", "WFC", "WMT", "XOM", "ADP", "TGT", "PYPL", "O",
]

assert len(STOCKS) == 100
assert len(set(STOCKS)) == 100

DOWNLOAD_START = "2017-01-01"  # Warm-up for the 200-session moving average.
DOWNLOAD_END = "2025-01-01"  # Exclusive, so training cannot inspect 2025.
TRAIN_START = "2018-01-01"
CANDIDATE_TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
FINAL_TRAIN_END = "2024-12-31"

TRADING_FEE = 0.001  # 0.1% of value traded.
POSITIONS = np.array([-1, 0, 1], dtype=np.int8)  # Short, cash, long.
TOP_STOCKS_TO_HOLD = 10
TRAINING_EPISODE_DAYS = 252
HOLDING_PERIOD_DAYS = 20  # Approximately one month between decisions.
MIN_LONG_PROBABILITY = 0.45
KEEP_HOLDING_WITHIN_TOP_RANK = 20

CANDIDATE_SEEDS = (11, 22, 33, 44)
CANDIDATE_TRAINING_STEPS = 200_000
FINAL_TRAINING_STEPS = 400_000

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIRECTORY / "rl_codex_2_long_term_policy"

FEATURE_COLUMNS = [
    "feature_return_1d",
    "feature_return_5d",
    "feature_return_20d",
    "feature_sma_10",
    "feature_sma_20",
    "feature_sma_50",
    "feature_sma_200",
    "feature_volatility_20",
    "feature_rsi_14",
    "feature_volume_20",
    "feature_daily_range",
]


# ---------------------------------------------------------------------------
# Downloading and feature engineering
# ---------------------------------------------------------------------------

def download_stocks(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download all stocks together and return one clean DataFrame per ticker."""
    print(f"Downloading {len(STOCKS)} stocks from {start} to {end}...")
    downloaded = yf.download(
        STOCKS,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no stock data.")
    if not isinstance(downloaded.columns, pd.MultiIndex):
        raise RuntimeError("Expected multi-stock columns from Yahoo Finance.")

    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in STOCKS:
        try:
            stock = downloaded[ticker].copy()
        except KeyError:
            missing.append(ticker)
            continue

        stock.columns = [str(column).lower() for column in stock.columns]
        stock = stock.dropna(subset=["open", "high", "low", "close", "volume"])
        stock = stock.loc[~stock.index.duplicated(keep="first")].sort_index()

        if stock.index.tz is not None:
            stock.index = stock.index.tz_localize(None)

        if stock.empty:
            missing.append(ticker)
        else:
            result[ticker] = stock

    if missing:
        raise RuntimeError(
            "Complete data was not returned for these stocks: "
            + ", ".join(missing)
            + ". Run the program again in case Yahoo temporarily rate-limited "
            "part of the request."
        )

    return result


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using only current and previous closing prices."""
    change = close.diff()
    average_gain = change.clip(lower=0).rolling(window).mean()
    average_loss = (-change.clip(upper=0)).rolling(window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    return rsi


def add_features(stock: pd.DataFrame) -> pd.DataFrame:
    """Convert changing price levels into comparable, scale-free indicators."""
    featured = stock.copy()
    close = featured["close"]
    daily_return = close.pct_change(fill_method=None)

    featured["feature_return_1d"] = daily_return.clip(-0.20, 0.20) * 5
    featured["feature_return_5d"] = (
        close.pct_change(5, fill_method=None).clip(-0.40, 0.40) * 2.5
    )
    featured["feature_return_20d"] = close.pct_change(
        20, fill_method=None
    ).clip(-0.60, 0.60)

    for window in (10, 20, 50, 200):
        average = close.rolling(window).mean()
        featured[f"feature_sma_{window}"] = (
            (close / average - 1).clip(-0.50, 0.50) * 2
        )

    featured["feature_volatility_20"] = (
        daily_return.rolling(20).std().clip(0, 0.10) * 10
    )
    featured["feature_rsi_14"] = (calculate_rsi(close) - 50) / 50
    featured["feature_volume_20"] = (
        (featured["volume"] / featured["volume"].rolling(20).mean() - 1)
        .clip(-2, 2)
        / 2
    )
    featured["feature_daily_range"] = (
        ((featured["high"] - featured["low"]) / close).clip(0, 0.20) * 5
    )

    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.dropna(subset=FEATURE_COLUMNS).copy()


def prepare_universe(
    downloaded: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply exactly the same feature logic to every stock."""
    prepared = {ticker: add_features(data) for ticker, data in downloaded.items()}
    bad = [
        ticker
        for ticker, data in prepared.items()
        if data.empty
        or data[FEATURE_COLUMNS].isna().any().any()
        or not np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all()
    ]
    if bad:
        raise RuntimeError(f"Invalid features for: {', '.join(bad)}")
    return prepared


def select_period(
    universe: dict[str, pd.DataFrame], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Select one inclusive date range for every stock."""
    selected = {
        ticker: data.loc[start:end].copy()
        for ticker, data in universe.items()
    }
    empty = [ticker for ticker, data in selected.items() if len(data) < 2]
    if empty:
        raise RuntimeError(f"Insufficient period data for: {', '.join(empty)}")
    return selected


# ---------------------------------------------------------------------------
# RL environment: one randomly chosen stock per episode
# ---------------------------------------------------------------------------

class RandomStockTradingEnvironment(gym.Env):
    """Teach one shared policy by sampling stocks and historical periods."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        universe: dict[str, pd.DataFrame],
        episode_days: int = TRAINING_EPISODE_DAYS,
    ) -> None:
        super().__init__()
        self.universe = universe
        self.tickers = sorted(universe)
        self.episode_days = episode_days

        too_short = [
            ticker
            for ticker, data in universe.items()
            if len(data) < episode_days + 1
        ]
        if too_short:
            raise RuntimeError(
                "These stocks lack a complete training episode: "
                + ", ".join(too_short)
            )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(len(FEATURE_COLUMNS) + 1,),
            dtype=np.float32,
        )

        self.stock = universe[self.tickers[0]]
        self.ticker = self.tickers[0]
        self.index = 0
        self.last_index = episode_days
        self.position = 0
        self.portfolio_value = 1.0

    def _observation(self) -> np.ndarray:
        features = self.stock.iloc[self.index][FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )
        return np.append(features, np.float32(self.position)).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.ticker = str(self.np_random.choice(self.tickers))
        self.stock = self.universe[self.ticker]

        maximum_start = len(self.stock) - self.episode_days - 1
        self.index = int(self.np_random.integers(0, maximum_start + 1))
        self.last_index = self.index + self.episode_days
        self.position = 0
        self.portfolio_value = 1.0

        return self._observation(), {"ticker": self.ticker}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        old_position = self.position
        self.position = int(POSITIONS[int(action)])

        # Moving cash -> long costs 0.1%. Flipping short -> long trades twice
        # the portfolio and therefore costs 0.2%.
        turnover = abs(self.position - old_position)
        fee_multiplier = 1 - TRADING_FEE * turnover

        # The selected position remains unchanged for about one month. PPO is
        # therefore rewarded for identifying medium-term regimes instead of
        # reacting to every small daily movement.
        next_index = min(
            self.index + HOLDING_PERIOD_DAYS,
            self.last_index,
        )
        current_close = float(self.stock.iloc[self.index]["close"])
        next_close = float(self.stock.iloc[next_index]["close"])
        stock_return = next_close / current_close - 1
        market_multiplier = 1 + self.position * stock_return

        old_value = self.portfolio_value
        self.portfolio_value *= fee_multiplier * market_multiplier
        reward = float(np.log(max(self.portfolio_value, 1e-12) / old_value))

        self.index = next_index
        terminated = self.portfolio_value <= 0
        truncated = self.index >= self.last_index
        info = {
            "ticker": self.ticker,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "stock_return": stock_return,
        }
        return self._observation(), reward, terminated, truncated, info


def make_model(environment: gym.Env, seed: int) -> PPO:
    """Construct PPO with enough capacity for a general cross-stock policy."""
    set_random_seed(seed)
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device="cpu",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# One-account portfolio simulation used for validation and the final test
# ---------------------------------------------------------------------------

def action_probabilities(
    model: PPO, observations: np.ndarray
) -> np.ndarray:
    """Return PPO's short/cash/long probabilities for a batch of stocks."""
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observations)
        distribution = model.policy.get_distribution(tensor)
        return distribution.distribution.probs.cpu().numpy()


def common_trading_dates(
    universe: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Find sessions available for every security in the portfolio."""
    dates: pd.DatetimeIndex | None = None
    for data in universe.values():
        dates = data.index if dates is None else dates.intersection(data.index)
    if dates is None or len(dates) < 2:
        raise RuntimeError("Stocks do not share enough trading sessions.")
    return dates.sort_values()


def simulate_portfolio(
    model: PPO,
    universe: dict[str, pd.DataFrame],
    *,
    starting_cash: float = 100_000,
    top_k: int = TOP_STOCKS_TO_HOLD,
) -> dict[str, Any]:
    """Trade one slower long-only account using PPO's strongest signals.

    Rebalancing occurs approximately monthly. A stock already owned remains in
    the portfolio while it stays within the top-20 eligible signals. This
    buffer prevents tiny changes in ranking from causing unnecessary trades.
    """
    tickers = sorted(universe)
    dates = common_trading_dates(universe)
    count = len(tickers)

    current_weights = np.zeros(count, dtype=float)
    portfolio_value = float(starting_cash)
    values = [portfolio_value]
    total_turnover = 0.0
    rebalances = 0
    holdings_each_day: list[int] = []
    trade_log: list[dict[str, Any]] = []

    for day_number in range(len(dates) - 1):
        today = dates[day_number]
        tomorrow = dates[day_number + 1]

        next_returns = []
        for ticker in tickers:
            data = universe[ticker]
            next_returns.append(
                float(data.loc[tomorrow, "close"] / data.loc[today, "close"] - 1)
            )

        rebalance_today = day_number % HOLDING_PERIOD_DAYS == 0
        probabilities: np.ndarray | None = None

        if rebalance_today:
            observations = []
            for ticker_number, ticker in enumerate(tickers):
                features = universe[ticker].loc[
                    today, FEATURE_COLUMNS
                ].to_numpy(dtype=np.float32)
                currently_held = np.float32(
                    current_weights[ticker_number] > 0.0001
                )
                observations.append(np.append(features, currently_held))

            probabilities = action_probabilities(
                model, np.asarray(observations, dtype=np.float32)
            )
            preferred_actions = np.argmax(probabilities, axis=1)
            long_candidates = np.flatnonzero(
                (preferred_actions == 2)
                & (probabilities[:, 2] >= MIN_LONG_PROBABILITY)
            )

            ranked = long_candidates[
                np.argsort(probabilities[long_candidates, 2])[::-1]
            ]
            keep_zone = set(ranked[:KEEP_HOLDING_WITHIN_TOP_RANK].tolist())
            currently_held = np.flatnonzero(current_weights > 0.0001)

            # Preserve qualifying existing positions first, then fill empty
            # slots with the strongest new opportunities.
            chosen_list = [
                number for number in currently_held if number in keep_zone
            ][:top_k]
            for number in ranked:
                if len(chosen_list) >= top_k:
                    break
                stock_number = int(number)
                if stock_number not in chosen_list:
                    chosen_list.append(stock_number)

            target_weights = np.zeros(count, dtype=float)
            if chosen_list:
                target_weights[chosen_list] = 1 / len(chosen_list)
        else:
            # Between decision dates, allow weights to drift naturally and do
            # not buy or sell anything.
            target_weights = current_weights.copy()

        turnover = float(np.abs(target_weights - current_weights).sum())
        if turnover > 1e-8:
            rebalances += 1
        total_turnover += turnover

        if probabilities is not None:
            portfolio_before_trades = portfolio_value
            weight_changes = target_weights - current_weights
            for ticker_number in np.flatnonzero(np.abs(weight_changes) > 1e-6):
                weight_change = float(weight_changes[ticker_number])
                dollars_traded = abs(weight_change) * portfolio_before_trades
                trade_log.append(
                    {
                        "date": today.date().isoformat(),
                        "ticker": tickers[ticker_number],
                        "action": "BUY" if weight_change > 0 else "SELL",
                        "weight_before": float(current_weights[ticker_number]),
                        "weight_after": float(target_weights[ticker_number]),
                        "weight_change": weight_change,
                        "estimated_dollars_traded": dollars_traded,
                        "estimated_fee": dollars_traded * TRADING_FEE,
                        "long_probability": float(
                            probabilities[ticker_number, 2]
                        ),
                        "portfolio_before_trades": portfolio_before_trades,
                    }
                )

        # Fee is charged on every dollar bought or sold.
        portfolio_value *= max(0, 1 - TRADING_FEE * turnover)

        returns = np.asarray(next_returns)
        cash_weight = 1 - target_weights.sum()
        gross_multiplier = cash_weight + np.sum(
            target_weights * (1 + returns)
        )
        portfolio_value *= gross_multiplier
        values.append(portfolio_value)
        holdings_each_day.append(
            int(np.count_nonzero(target_weights > 0.0001))
        )

        # Price movements cause weights to drift before the next rebalance.
        if gross_multiplier > 0:
            current_weights = (
                target_weights * (1 + returns) / gross_multiplier
            )
        else:
            current_weights = np.zeros(count)

    values_array = np.asarray(values)
    running_high = np.maximum.accumulate(values_array)
    maximum_drawdown = float((values_array / running_high - 1).min())

    # Equal-weight buy-and-hold provides a transparent 100-stock benchmark.
    price_multipliers = np.array(
        [
            universe[ticker].loc[dates[-1], "close"]
            / universe[ticker].loc[dates[0], "close"]
            for ticker in tickers
        ],
        dtype=float,
    )
    benchmark_final = (
        starting_cash
        * (1 - TRADING_FEE)
        * float(price_multipliers.mean())
    )

    return {
        "first_date": dates[0],
        "last_date": dates[-1],
        "trading_days": len(dates),
        "starting_balance": float(starting_cash),
        "final_balance": float(portfolio_value),
        "return": float(portfolio_value / starting_cash - 1),
        "benchmark_balance": float(benchmark_final),
        "benchmark_return": float(benchmark_final / starting_cash - 1),
        "max_drawdown": maximum_drawdown,
        "rebalances": rebalances,
        "total_turnover": total_turnover,
        "average_holdings": float(np.mean(holdings_each_day)),
        "trade_log": trade_log,
    }


def print_result(label: str, result: dict[str, Any]) -> None:
    print(
        f"{label}: return={result['return']:+.2%}, "
        f"balance=${result['final_balance']:,.2f}, "
        f"drawdown={result['max_drawdown']:.2%}, "
        f"average holdings={result['average_holdings']:.1f}"
    )


# ---------------------------------------------------------------------------
# Candidate selection and final training
# ---------------------------------------------------------------------------

def train_candidate(
    seed: int,
    training_universe: dict[str, pd.DataFrame],
    validation_universe: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    print(f"\nTraining candidate seed {seed} for {CANDIDATE_TRAINING_STEPS:,} steps...")
    environment = RandomStockTradingEnvironment(training_universe)
    model = make_model(environment, seed)
    model.learn(total_timesteps=CANDIDATE_TRAINING_STEPS)
    environment.close()

    result = simulate_portfolio(model, validation_universe)
    print_result(f"Seed {seed} on 2024", result)
    return result


def main() -> None:
    print("=" * 76)
    print("RL CODEX 2 - SHARED PPO POLICY ACROSS 100 LARGE US STOCKS")
    print("=" * 76)
    downloaded = download_stocks(DOWNLOAD_START, DOWNLOAD_END)
    prepared = prepare_universe(downloaded)

    candidate_training = select_period(
        prepared, TRAIN_START, CANDIDATE_TRAIN_END
    )
    validation = select_period(
        prepared, VALIDATION_START, FINAL_TRAIN_END
    )
    final_training = select_period(prepared, TRAIN_START, FINAL_TRAIN_END)

    if any((data.index.year >= 2025).any() for data in final_training.values()):
        raise AssertionError("2025 data leaked into training.")

    print(f"Usable stocks: {len(final_training)}")
    print(f"Trading fee: {TRADING_FEE:.2%} of value traded")
    print(f"Maximum portfolio holdings: {TOP_STOCKS_TO_HOLD}")
    print(f"Minimum holding period: {HOLDING_PERIOD_DAYS} trading sessions")
    print(
        "Candidate period: 2018-2023 | Validation: 2024 | "
        "Final learning: 2018-2024"
    )

    results: list[tuple[int, dict[str, Any]]] = []
    for seed in CANDIDATE_SEEDS:
        result = train_candidate(seed, candidate_training, validation)
        results.append((seed, result))

    winning_seed, winning_result = max(
        results, key=lambda item: item[1]["return"]
    )
    print("\n" + "-" * 76)
    print(f"Winning seed selected without examining 2025: {winning_seed}")
    print_result("Winning 2024 validation", winning_result)
    print("-" * 76)

    print(
        f"\nRetraining seed {winning_seed} on all 100 stocks from "
        f"2018-2024 for {FINAL_TRAINING_STEPS:,} steps..."
    )
    final_environment = RandomStockTradingEnvironment(final_training)
    final_model = make_model(final_environment, winning_seed)
    final_model.learn(total_timesteps=FINAL_TRAINING_STEPS)
    final_environment.close()
    final_model.save(MODEL_PATH)

    print(f"\nSaved final model: {MODEL_PATH}.zip")
    print("This trainer never downloads 2025. Run CODEX_SIMULATION_2.py next.")
    print(
        "Because the long-term design followed an earlier 2025 result, that "
        "simulation is now a development backtest, not a fresh holdout."
    )


if __name__ == "__main__":
    main()
