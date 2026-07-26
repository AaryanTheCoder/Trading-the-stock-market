"""Small deterministic tests for ranking, accounting, and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from research_engine import (
    FEATURE_NAMES,
    MarketData,
    CACHE_DATA_DIR,
    MODEL_ARTIFACT_DIR,
    SIMULATION_DATA_DIR,
    cross_sectional_ranks,
    simulate_scores,
)


class ResearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2025-01-02", periods=3)
        self.data = MarketData(
            dates=self.dates,
            tickers=("A", "B"),
            close=np.array(
                [
                    [100.0, 100.0],
                    [110.0, 90.0],
                    [121.0, 81.0],
                ]
            ),
            open=np.array(
                [
                    [100.0, 100.0],
                    [100.0, 100.0],
                    [110.0, 90.0],
                ]
            ),
            features=np.zeros((3, 2, len(FEATURE_NAMES))),
        )
        self.scores = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    def test_cross_sectional_ranks(self) -> None:
        values = np.array([[[3.0], [1.0], [2.0]]])
        actual = cross_sectional_ranks(values)
        np.testing.assert_allclose(actual[0, :, 0], [1.0, -1.0, 0.0])

    def test_close_accounting_without_fee(self) -> None:
        result = simulate_scores(
            self.data,
            self.scores,
            start="2025-01-01",
            end="2025-12-31",
            holding_days=5,
            top_k=1,
            keep_rank=1,
            fee=0.0,
            execution="close",
        )
        self.assertAlmostEqual(result["return"], 0.21, places=12)

    def test_next_open_accounting_without_fee(self) -> None:
        result = simulate_scores(
            self.data,
            self.scores,
            start="2025-01-01",
            end="2025-12-31",
            holding_days=5,
            top_k=1,
            keep_rank=1,
            fee=0.0,
            execution="next_open",
        )
        # Day-one entry is at 100 open and the two closes are 110 and 121.
        self.assertAlmostEqual(result["return"], 0.21, places=12)

    def test_rebalance_offset_stays_in_cash_until_first_signal(self) -> None:
        result = simulate_scores(
            self.data,
            self.scores,
            start="2025-01-01",
            end="2025-12-31",
            holding_days=2,
            top_k=1,
            keep_rank=1,
            fee=0.0,
            execution="next_open",
            rebalance_offset=1,
        )
        # Cash on day one, then buy A at the second next-open (110 -> 121).
        self.assertAlmostEqual(result["return"], 0.10, places=12)

    def test_frozen_model_has_pre_2025_selection_cutoff(self) -> None:
        model = json.loads(
            (MODEL_ARTIFACT_DIR / "july26_model.json").read_text(encoding="utf-8")
        )
        self.assertLess(
            pd.Timestamp(model["selection_data_end"]),
            pd.Timestamp(model["test_data_start"]),
        )

    def test_saved_candidate_beats_identical_panel_rl2_results(self) -> None:
        summary = json.loads(
            (SIMULATION_DATA_DIR / "july26_summary.json").read_text(encoding="utf-8")
        )
        rl2 = json.loads(
            (SIMULATION_DATA_DIR / "rl2_benchmark.json").read_text(encoding="utf-8")
        )
        best_rl2 = max(item["return"] for item in rl2)
        self.assertGreater(summary["close_execution"]["return"], best_rl2)
        self.assertGreater(summary["next_open_execution"]["return"], best_rl2)
        self.assertEqual(
            summary["close_execution"]["last_date"],
            rl2[0]["last_date"],
        )

    def test_latest_overlay_is_complete(self) -> None:
        manifest = json.loads(
            (CACHE_DATA_DIR / "refresh_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["successful_tickers"], 100)
        self.assertEqual(manifest["missing_tickers"], [])
        self.assertEqual(manifest["observed_last_dates"], ["2026-07-24"])


if __name__ == "__main__":
    unittest.main()
