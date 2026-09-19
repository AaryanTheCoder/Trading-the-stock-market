from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / "source"))

from price_engine import MarketData, simulate_target_weights


class TargetWeightSimulatorTests(unittest.TestCase):
    def market(self):
        return MarketData(
            dates=pd.date_range("2024-01-02", periods=3, freq="B"),
            tickers=("AAA", "BBB"),
            close=np.asarray([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0]]),
            open=np.asarray([[100.0, 100.0], [100.0, 100.0], [110.0, 100.0]]),
            features=np.empty((3, 2, 0)),
        )

    def test_close_equal_weight_accounting(self):
        targets = np.zeros((3, 2))
        targets[0] = [0.5, 0.5]
        result = simulate_target_weights(
            self.market(),
            targets,
            start="2024-01-02",
            end="2024-01-04",
            holding_days=10,
            fee=0.0,
            execution="close",
        )
        self.assertAlmostEqual(result["final_balance"], 110_000.0)

    def test_invalid_target_fails(self):
        targets = np.zeros((3, 2))
        targets[0] = [0.8, 0.8]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            simulate_target_weights(
                self.market(),
                targets,
                start="2024-01-02",
                end="2024-01-04",
                holding_days=10,
            )


if __name__ == "__main__":
    unittest.main()

