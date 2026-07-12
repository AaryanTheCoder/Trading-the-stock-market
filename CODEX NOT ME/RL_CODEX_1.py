"""RL CODEX 1: train a simple PPO agent to trade Apple stock.

Learning design
---------------
1. Market observations contain only AAPL information known by today's close.
2. The action is the position held from today's close to the next close:
      0 = short, 1 = cash, 2 = long
3. The reward is the portfolio's daily log return after a 0.1% trading fee.
4. Several PPO agents train on 2018-2023.
5. Their policies are compared on 2024, which acts as validation data.
6. The winning configuration is retrained on all data from 2018-2024.

The year 2025 is never downloaded or inspected by this training program. It is
reserved for CODEX_SIMULATION_1.py.

Important simplification
------------------------
The environment assumes that indicators calculated at the close can be traded
at that closing price. A real system would need explicit order timing, slippage,
spread, and possibly next-open execution. The 0.1% fee makes the assumption less
optimistic, but does not make this a production trading system.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gym_trading_env  # Registers "TradingEnv" with Gymnasium.
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

TICKER = "AAPL"
DOWNLOAD_START = "2017-01-01"  # Warm-up period for the 200-day moving average.
DOWNLOAD_END = "2025-01-01"  # Exclusive: includes 2024-12-31, never includes 2025.

TRAIN_START = "2018-01-01"
CANDIDATE_TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
FINAL_TRAIN_END = "2024-12-31"

TRADING_FEE = 0.001  # 0.1% whenever stock is bought or sold.
POSITIONS = [-1, 0, 1]  # Short, cash, long.
MODEL_FILE = "rl_codex_1_policy"

# Multiple seeds produce independently initialized neural networks.
CANDIDATE_SEEDS = (11, 22, 33, 44)
CANDIDATE_TRAINING_STEPS = 100_000
FINAL_TRAINING_STEPS = 200_000

# Random one-year episodes expose PPO to many different market regimes instead
# of always beginning its experience in January 2018.
TRAINING_EPISODE_DAYS = 252


# Gym Trading Env automatically includes every column containing "feature" in
# the neural network's observation. These are all scale-free indicators.
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


def download_aapl(start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV data and check that it is usable."""
    data = yf.download(
        TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            "No AAPL data was downloaded. Check the internet connection."
        )

    data = data.copy()
    data.columns = [str(column).lower() for column in data.columns]
    data = data.loc[~data.index.duplicated(keep="first")].sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {sorted(missing)}")

    return data


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index using past price changes."""
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)

    # Define sensible values for windows containing no losses or no movement.
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    return rsi


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create stationary features from AAPL prices and volume.

    Raw prices continually change scale over the years. Ratios and percentage
    changes are easier for a neural network to compare across market regimes.
    Multipliers keep most observations approximately between -1 and +1.
    """
    featured = data.copy()
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
        moving_average = close.rolling(window).mean()
        featured[f"feature_sma_{window}"] = (
            (close / moving_average - 1).clip(-0.50, 0.50) * 2
        )

    featured["feature_volatility_20"] = (
        daily_return.rolling(20).std().clip(0, 0.10) * 10
    )
    featured["feature_rsi_14"] = (calculate_rsi(close) - 50) / 50

    average_volume = featured["volume"].rolling(20).mean()
    featured["feature_volume_20"] = (
        (featured["volume"] / average_volume - 1).clip(-2, 2) / 2
    )
    featured["feature_daily_range"] = (
        ((featured["high"] - featured["low"]) / close).clip(0, 0.20) * 5
    )

    featured = featured.replace([np.inf, -np.inf], np.nan)
    featured = featured.dropna(subset=FEATURE_COLUMNS).copy()

    if featured[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Feature preparation produced missing values.")
    if not np.isfinite(featured[FEATURE_COLUMNS].to_numpy()).all():
        raise RuntimeError("Feature preparation produced infinite values.")

    return featured


def select_dates(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return an inclusive date range and fail clearly if it is empty."""
    selected = data.loc[start:end].copy()
    if selected.empty:
        raise RuntimeError(f"No observations exist between {start} and {end}.")
    return selected


def make_environment(
    data: pd.DataFrame,
    *,
    training: bool,
    starting_cash: float = 100_000,
) -> gym.Env:
    """Create identical market mechanics for training and evaluation."""
    episode_duration: int | str
    episode_duration = TRAINING_EPISODE_DAYS if training else "max"

    return gym.make(
        "TradingEnv",
        df=data,
        positions=POSITIONS,
        trading_fees=TRADING_FEE,
        initial_position=0,  # Every episode starts in cash.
        portfolio_initial_value=starting_cash,
        max_episode_duration=episode_duration,
        verbose=0,
    )


def make_ppo(environment: gym.Env, seed: int) -> PPO:
    """Create a small PPO network suitable for the tabular observation."""
    set_random_seed(seed)
    np.random.seed(seed)  # The installed trading environment uses NumPy directly.

    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.005,  # Maintains some exploration during training.
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def evaluate_policy(
    model: PPO,
    data: pd.DataFrame,
    *,
    starting_cash: float = 100_000,
) -> dict[str, Any]:
    """Run one deterministic episode and calculate understandable metrics."""
    environment = make_environment(
        data, training=False, starting_cash=starting_cash
    )
    observation, _ = environment.reset(seed=0)
    finished = False

    while not finished:
        action, _ = model.predict(observation, deterministic=True)
        action_number = int(np.asarray(action).item())
        observation, _, terminated, truncated, _ = environment.step(action_number)
        finished = terminated or truncated

    history = environment.unwrapped.historical_info
    portfolio_values = np.asarray(history["portfolio_valuation"], dtype=float)
    positions = np.asarray(history["position"], dtype=float)
    market_prices = np.asarray(history["data_close"], dtype=float)

    running_high = np.maximum.accumulate(portfolio_values)
    drawdowns = portfolio_values / running_high - 1
    position_changes = int(np.count_nonzero(np.diff(positions)))

    result = {
        "starting_balance": float(portfolio_values[0]),
        "final_balance": float(portfolio_values[-1]),
        "return": float(portfolio_values[-1] / portfolio_values[0] - 1),
        "market_return": float(market_prices[-1] / market_prices[0] - 1),
        "max_drawdown": float(drawdowns.min()),
        "position_changes": position_changes,
        "long_fraction": float(np.mean(positions == 1)),
        "cash_fraction": float(np.mean(positions == 0)),
        "short_fraction": float(np.mean(positions == -1)),
    }
    environment.close()
    return result


def print_policy_result(label: str, result: dict[str, Any]) -> None:
    """Print the metrics used to understand and compare a policy."""
    print(
        f"{label}: return={result['return']:+.2%}, "
        f"balance=${result['final_balance']:,.2f}, "
        f"drawdown={result['max_drawdown']:.2%}, "
        f"trades={result['position_changes']}"
    )


def train_candidate(
    seed: int,
    candidate_training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> tuple[PPO, dict[str, Any]]:
    """Train one independently initialized policy and validate it on 2024."""
    print(f"\nTraining candidate seed {seed}...")
    environment = make_environment(candidate_training_data, training=True)
    model = make_ppo(environment, seed)
    model.learn(total_timesteps=CANDIDATE_TRAINING_STEPS)
    environment.close()

    validation_result = evaluate_policy(model, validation_data)
    print_policy_result(f"Seed {seed} on 2024", validation_result)
    return model, validation_result


def main() -> None:
    print("=" * 72)
    print("RL CODEX 1 - AAPL PPO TRAINING")
    print("=" * 72)
    print("Downloading AAPL data through the end of 2024...")

    downloaded = download_aapl(DOWNLOAD_START, DOWNLOAD_END)
    featured = add_features(downloaded)

    candidate_training_data = select_dates(
        featured, TRAIN_START, CANDIDATE_TRAIN_END
    )
    validation_data = select_dates(
        featured, VALIDATION_START, FINAL_TRAIN_END
    )
    final_training_data = select_dates(featured, TRAIN_START, FINAL_TRAIN_END)

    if (final_training_data.index.year >= 2025).any():
        raise AssertionError("2025 data leaked into RL training.")

    print(
        f"Candidate training: {candidate_training_data.index.min().date()} to "
        f"{candidate_training_data.index.max().date()} "
        f"({len(candidate_training_data)} sessions)"
    )
    print(
        f"Policy validation: {validation_data.index.min().date()} to "
        f"{validation_data.index.max().date()} ({len(validation_data)} sessions)"
    )
    print(f"Trading fee: {TRADING_FEE:.2%} per buy or sell")

    validation_results: list[tuple[int, dict[str, Any]]] = []
    for seed in CANDIDATE_SEEDS:
        candidate, result = train_candidate(
            seed, candidate_training_data, validation_data
        )
        validation_results.append((seed, result))
        # The candidate is no longer required after its validation result.
        del candidate

    # Maximize validation return. In a larger research project, selection could
    # also penalize drawdown, but here the objective is deliberately simple.
    winning_seed, winning_result = max(
        validation_results, key=lambda item: item[1]["return"]
    )

    print("\n" + "-" * 72)
    print(f"Winning candidate seed: {winning_seed}")
    print_policy_result("Winning 2024 validation", winning_result)
    print("-" * 72)

    print(
        "\nRetraining the winning configuration on every eligible "
        "2018-2024 session..."
    )
    final_environment = make_environment(final_training_data, training=True)
    final_model = make_ppo(final_environment, winning_seed)
    final_model.learn(total_timesteps=FINAL_TRAINING_STEPS)
    final_environment.close()

    final_model.save(MODEL_FILE)
    print(f"\nSaved final policy as {MODEL_FILE}.zip")
    print("2025 has remained untouched. Run CODEX_SIMULATION_1.py exactly once")
    print("when you are ready to perform the final out-of-sample experiment.")


if __name__ == "__main__":
    main()
