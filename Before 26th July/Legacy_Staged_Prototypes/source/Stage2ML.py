"""Leakage-safe AAPL next-session direction experiment.

The model sees data available at the end of a trading session and predicts
whether the next trading session will close higher. Model selection uses only
2018-2024 data. The entire 2025 calendar year is reserved for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TICKER = "AAPL"

# Warm-up data is required before 2018 for rolling features. Data from early
# 2026 is required to score the prediction made after the final 2025 session.
DOWNLOAD_START = "2017-09-01"
DOWNLOAD_END = "2026-01-10"  # yfinance treats the end date as exclusive.
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")

VALIDATION_YEARS = (2020, 2021, 2022, 2023, 2024)
RANDOM_STATE = 42
NEAR_TIE_TOLERANCE = 0.002  # 0.2 percentage points.

FEATURES = [
    "Return_1d",
    "Return_2d",
    "Return_5d",
    "Return_10d",
    "Return_20d",
    "Close_to_MA5",
    "Close_to_MA10",
    "Close_to_MA20",
    "Close_to_MA50",
    "Volatility_5d",
    "Volatility_10d",
    "Volatility_20d",
    "Volume_to_MA20",
    "Overnight_Gap",
    "Intraday_Return",
    "High_Low_Range",
    "Close_Location",
    "RSI_14",
]


@dataclass
class CandidateResult:
    """Cross-validation result for one estimator configuration."""

    family: str
    parameters: dict[str, Any]
    estimator: BaseEstimator
    yearly_scores: dict[int, float]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(list(self.yearly_scores.values())))

    @property
    def standard_deviation(self) -> float:
        return float(np.std(list(self.yearly_scores.values())))


def download_aapl_data() -> pd.DataFrame:
    """Download and validate adjusted AAPL OHLCV data."""
    print(f"Downloading adjusted {TICKER} data...")
    raw = yf.download(
        TICKER,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data. Check the internet connection and ticker."
        )

    # Newer yfinance versions return a MultiIndex even for one ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.copy()
    raw.columns = [str(column).title() for column in raw.columns]
    raw = raw.loc[~raw.index.duplicated(keep="first")].sort_index()

    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_columns.difference(raw.columns)
    if missing:
        raise RuntimeError(
            f"Downloaded data is missing required columns: {sorted(missing)}"
        )

    if raw.index.min() >= TRAIN_START:
        raise RuntimeError("Downloaded data does not contain the required warm-up period.")
    if not (raw.index.year == 2025).any():
        raise RuntimeError("Downloaded data does not contain any 2025 trading sessions.")
    if not (raw.index.year == 2026).any():
        raise RuntimeError(
            "Early-2026 data is required to score the final 2025 prediction."
        )

    return raw


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI while handling one-sided and completely flat windows."""
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    return rsi


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Create features and targets using current and historical data only."""
    data = raw.copy()
    close = data["Close"]

    for days in (1, 2, 5, 10, 20):
        data[f"Return_{days}d"] = close.pct_change(days, fill_method=None)

    for days in (5, 10, 20, 50):
        moving_average = close.rolling(days, min_periods=days).mean()
        data[f"Close_to_MA{days}"] = close / moving_average - 1

    daily_return = close.pct_change(fill_method=None)
    for days in (5, 10, 20):
        data[f"Volatility_{days}d"] = daily_return.rolling(
            days, min_periods=days
        ).std()

    data["Volume_to_MA20"] = (
        data["Volume"] / data["Volume"].rolling(20, min_periods=20).mean() - 1
    )
    data["Overnight_Gap"] = data["Open"] / close.shift(1) - 1
    data["Intraday_Return"] = close / data["Open"] - 1
    data["High_Low_Range"] = (data["High"] - data["Low"]) / close

    daily_range = (data["High"] - data["Low"]).replace(0, np.nan)
    data["Close_Location"] = (close - data["Low"]) / daily_range
    data["RSI_14"] = calculate_rsi(close) / 100

    # Preserve the actual date of the outcome. This prevents a row dated in
    # 2024 from entering training when its next-session label belongs to 2025.
    data["Target_Date"] = pd.Series(data.index, index=data.index).shift(-1)
    data["Next_Close"] = close.shift(-1)
    data["Target"] = np.where(
        data["Next_Close"].notna(),
        (data["Next_Close"] > close).astype(int),
        np.nan,
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=FEATURES + ["Target", "Target_Date"]).copy()


def split_dataset(
    dataset: pd.DataFrame, raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a strict pre-2025 training set and a 2025 holdout set."""
    train = dataset.loc[
        (dataset.index >= TRAIN_START)
        & (dataset.index <= TRAIN_END)
        & (dataset["Target_Date"] <= TRAIN_END)
    ].copy()

    test = dataset.loc[
        (dataset.index >= TEST_START) & (dataset.index <= TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise RuntimeError("Training or test data is empty after feature preparation.")
    if len(train) < 500:
        raise RuntimeError(f"Too few training samples were produced: {len(train)}")
    if (train.index.year >= 2025).any() or (
        pd.to_datetime(train["Target_Date"]).dt.year >= 2025
    ).any():
        raise AssertionError("2025 data leaked into the training set.")
    if not (test.index.year == 2025).all():
        raise AssertionError("The holdout set contains feature dates outside 2025.")

    expected_2025_sessions = int((raw.index.year == 2025).sum())
    if len(test) != expected_2025_sessions:
        raise RuntimeError(
            "Not every 2025 session can be scored: "
            f"expected {expected_2025_sessions}, produced {len(test)}."
        )

    final_row = test.iloc[-1]
    if pd.Timestamp(final_row["Target_Date"]).year != 2026:
        raise AssertionError(
            "The final 2025 session is not scored against a 2026 session."
        )

    validate_feature_matrix(train[FEATURES], "training")
    validate_feature_matrix(test[FEATURES], "test")
    return train, test


def validate_feature_matrix(features: pd.DataFrame, name: str) -> None:
    """Reject malformed model inputs before fitting."""
    if features.empty:
        raise RuntimeError(f"The {name} feature matrix is empty.")
    if features.isna().any().any():
        raise RuntimeError(f"The {name} feature matrix contains missing values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise RuntimeError(f"The {name} feature matrix contains infinite values.")


def candidate_models() -> list[tuple[str, dict[str, Any], BaseEstimator]]:
    """Return a compact set of regularized linear, forest, and boosting models."""
    candidates: list[tuple[str, dict[str, Any], BaseEstimator]] = []

    for c_value in (0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            parameters = {"C": c_value, "class_weight": class_weight}
            estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c_value,
                            class_weight=class_weight,
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            candidates.append(("LogisticRegression", parameters, estimator))

    for max_depth in (4, 8, None):
        for min_samples_leaf in (5, 20):
            for class_weight in (None, "balanced"):
                parameters = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": class_weight,
                }
                estimator = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
                candidates.append(("RandomForest", parameters, estimator))

    for learning_rate in (0.03, 0.07):
        for max_leaf_nodes in (7, 15):
            for l2_regularization in (0.0, 1.0):
                parameters = {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                }
                estimator = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    max_iter=200,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                )
                candidates.append(("HistGradientBoosting", parameters, estimator))

    return candidates


def evaluate_candidate(
    family: str,
    parameters: dict[str, Any],
    estimator: BaseEstimator,
    training_data: pd.DataFrame,
) -> CandidateResult:
    """Evaluate one configuration with expanding annual validation."""
    yearly_scores: dict[int, float] = {}

    for year in VALIDATION_YEARS:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        fold_train = training_data.loc[
            (training_data.index < year_start)
            & (training_data["Target_Date"] < year_start)
        ]
        fold_validation = training_data.loc[
            (training_data.index >= year_start)
            & (training_data.index <= year_end)
            & (training_data["Target_Date"] <= year_end)
        ]

        if fold_train.empty or fold_validation.empty:
            raise RuntimeError(f"Unable to construct the {year} validation fold.")

        fold_model = clone(estimator)
        fold_model.fit(
            fold_train[FEATURES], fold_train["Target"].astype(int)
        )
        predictions = fold_model.predict(fold_validation[FEATURES])
        yearly_scores[year] = accuracy_score(
            fold_validation["Target"].astype(int), predictions
        )

    return CandidateResult(
        family=family,
        parameters=parameters,
        estimator=estimator,
        yearly_scores=yearly_scores,
    )


def select_model(training_data: pd.DataFrame) -> CandidateResult:
    """Select a model without reading any 2025 feature or outcome."""
    configurations = candidate_models()
    print(
        f"Evaluating {len(configurations)} configurations with expanding "
        f"{VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]} validation..."
    )

    results = [
        evaluate_candidate(family, parameters, estimator, training_data)
        for family, parameters, estimator in configurations
    ]

    best_mean = max(result.mean_accuracy for result in results)
    near_ties = [
        result
        for result in results
        if result.mean_accuracy >= best_mean - NEAR_TIE_TOLERANCE
    ]
    selected = min(
        near_ties,
        key=lambda result: (
            result.standard_deviation,
            -result.mean_accuracy,
            result.family,
        ),
    )

    ranked = sorted(
        results,
        key=lambda result: (-result.mean_accuracy, result.standard_deviation),
    )
    print("\nTop validation configurations:")
    for result in ranked[:5]:
        print(
            f"  {result.family:<22} mean={result.mean_accuracy:.2%}, "
            f"std={result.standard_deviation:.2%}, params={result.parameters}"
        )

    return selected


def wilson_confidence_interval(
    successes: int, observations: int, z_score: float = 1.96
) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial accuracy estimate."""
    if observations <= 0:
        raise ValueError("At least one observation is required.")

    proportion = successes / observations
    denominator = 1 + z_score**2 / observations
    center = (proportion + z_score**2 / (2 * observations)) / denominator
    margin = (
        z_score
        * sqrt(
            proportion * (1 - proportion) / observations
            + z_score**2 / (4 * observations**2)
        )
        / denominator
    )
    return center - margin, center + margin


def print_feature_importance(model: BaseEstimator) -> None:
    """Print comparable importances when the selected model exposes them."""
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline):
        final_model = model.named_steps.get("model")
        if final_model is not None and hasattr(final_model, "coef_"):
            importances = np.abs(np.asarray(final_model.coef_[0], dtype=float))

    if importances is None:
        print("\nFeature importance is not exposed by the selected estimator.")
        return

    ranked = sorted(
        zip(FEATURES, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nTop feature importances:")
    for feature, importance in ranked[:10]:
        print(f"  {feature:<20} {importance:.6f}")


def report_results(
    selected: CandidateResult,
    fitted_model: BaseEstimator,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Evaluate the final model and honest baselines on untouched 2025 data."""
    actual = test_data["Target"].astype(int)
    predictions = fitted_model.predict(test_data[FEATURES]).astype(int)

    correct = int((predictions == actual.to_numpy()).sum())
    total = len(actual)
    confidence_low, confidence_high = wilson_confidence_interval(correct, total)

    # Baseline 1 predicts the class that was most common in pre-2025 training.
    majority_class = int(training_data["Target"].mode().iloc[0])
    majority_predictions = np.full(total, majority_class, dtype=int)

    # Baseline 2 assumes tomorrow's direction will repeat today's direction.
    previous_direction_predictions = (
        test_data["Return_1d"].to_numpy() > 0
    ).astype(int)
    model_accuracy = accuracy_score(actual, predictions)
    majority_accuracy = accuracy_score(actual, majority_predictions)
    previous_direction_accuracy = accuracy_score(
        actual, previous_direction_predictions
    )

    matrix = confusion_matrix(actual, predictions, labels=[0, 1])

    print("\n" + "=" * 68)
    print("AAPL NEXT-SESSION DIRECTION: UNTOUCHED 2025 RESULTS")
    print("=" * 68)
    print(f"Selected model             : {selected.family}")
    print(f"Selected parameters        : {selected.parameters}")
    print(f"Training samples           : {len(training_data)}")
    print(f"2025 predictions           : {total}")
    print(f"Correct predictions        : {correct}")
    print(f"Accuracy                   : {model_accuracy:.2%}")
    print(
        "95% accuracy interval      : "
        f"{confidence_low:.2%} to {confidence_high:.2%}"
    )
    print(
        f"Balanced accuracy          : "
        f"{balanced_accuracy_score(actual, predictions):.2%}"
    )
    print(
        f"Precision (up)             : "
        f"{precision_score(actual, predictions, zero_division=0):.2%}"
    )
    print(
        f"Recall (up)                : "
        f"{recall_score(actual, predictions, zero_division=0):.2%}"
    )
    print(
        f"F1 (up)                    : "
        f"{f1_score(actual, predictions, zero_division=0):.2%}"
    )
    print(
        f"Majority-class baseline    : "
        f"{majority_accuracy:.2%}"
    )
    print(
        f"Previous-direction baseline: "
        f"{previous_direction_accuracy:.2%}"
    )
    best_baseline = max(majority_accuracy, previous_direction_accuracy)
    if model_accuracy > best_baseline:
        print("Holdout conclusion         : Model beat both simple baselines.")
    else:
        print("Holdout conclusion         : No directional edge over the baselines.")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted down  Predicted up")
    print(f"Actual down      {matrix[0, 0]:>14}  {matrix[0, 1]:>12}")
    print(f"Actual up        {matrix[1, 0]:>14}  {matrix[1, 1]:>12}")

    print("\nSelected-model yearly validation accuracy:")
    for year, score in selected.yearly_scores.items():
        print(f"  {year}: {score:.2%}")
    print(
        f"  Mean: {selected.mean_accuracy:.2%} "
        f"(std {selected.standard_deviation:.2%})"
    )
    print("=" * 68)

    print_feature_importance(fitted_model)


def main() -> None:
    raw = download_aapl_data()
    dataset = build_dataset(raw)
    training_data, test_data = split_dataset(dataset, raw)

    print(
        f"Training period: {training_data.index.min().date()} through "
        f"{training_data.index.max().date()} ({len(training_data)} samples)"
    )
    print(
        f"Test period:     {test_data.index.min().date()} through "
        f"{test_data.index.max().date()} ({len(test_data)} samples)"
    )

    selected = select_model(training_data)
    final_model = clone(selected.estimator)
    final_model.fit(
        training_data[FEATURES], training_data["Target"].astype(int)
    )
    report_results(
        selected=selected,
        fitted_model=final_model,
        training_data=training_data,
        test_data=test_data,
    )


if __name__ == "__main__":
    main()
