from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / "source"))

from news_overlay import (
    combined_scores,
    load_news_scores,
    news_tilt_targets,
    tie_aware_rank,
)


class NewsOverlayTests(unittest.TestCase):
    def test_tie_aware_rank(self):
        actual = tie_aware_rank(np.asarray([-1.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(actual, [-1.0, 0.0, 0.0, 1.0])

    def test_cache_coverage_and_combination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            day_dir = root / "2024-01-02"
            day_dir.mkdir()
            document = {
                "as_of": "2024-01-02",
                "prompt_version": "news-score-v1",
                "model": "gemini-2.5-flash",
                "records": [
                    {
                        "ticker": "AAA",
                        "effective_news_score": -0.5,
                        "audit_passed": True,
                    },
                    {
                        "ticker": "BBB",
                        "effective_news_score": 0.5,
                        "audit_passed": True,
                    },
                ],
                "metadata": {"usage_metadata": {"totalTokenCount": 10}},
            }
            (day_dir / "batch_00.json").write_text(json.dumps(document))
            values, audit = load_news_scores(
                root,
                pd.Timestamp("2024-01-02").date(),
                ["AAA", "BBB", "CCC"],
                ["AAA", "BBB"],
            )
            np.testing.assert_allclose(values, [-0.5, 0.5, 0.0])
            self.assertEqual(audit["stocks"], 2)
            price = np.asarray([[0.4, -0.2, -0.8]])
            combined, _ = combined_scores(
                price,
                pd.DatetimeIndex(["2024-01-02"]),
                np.asarray([0]),
                root,
                ["AAA", "BBB", "CCC"],
                0.5,
                2,
            )
            np.testing.assert_allclose(combined, [[-0.1, 0.3, np.nan]])

    def test_missing_stock_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            day_dir = root / "2024-01-02"
            day_dir.mkdir()
            document = {
                "as_of": "2024-01-02",
                "records": [
                    {
                        "ticker": "AAA",
                        "effective_news_score": 0.0,
                        "audit_passed": False,
                    }
                ],
            }
            (day_dir / "batch_00.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "1 missing"):
                load_news_scores(
                    root, pd.Timestamp("2024-01-02").date(), ["AAA", "BBB"]
                )

    def test_news_tilt_targets_normalize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            day_dir = root / "2024-01-02"
            day_dir.mkdir()
            (day_dir / "batch_00.json").write_text(
                json.dumps(
                    {
                        "as_of": "2024-01-02",
                        "records": [
                            {
                                "ticker": "AAA",
                                "effective_news_score": -0.5,
                                "audit_passed": True,
                            },
                            {
                                "ticker": "BBB",
                                "effective_news_score": 0.5,
                                "audit_passed": True,
                            },
                        ],
                    }
                )
            )
            targets, _ = news_tilt_targets(
                np.asarray([[0.9, 0.8, 0.1]]),
                pd.DatetimeIndex(["2024-01-02"]),
                np.asarray([0]),
                root,
                ["AAA", "BBB", "CCC"],
                1.0,
                2,
            )
            self.assertAlmostEqual(targets[0].sum(), 1.0)
            self.assertGreater(targets[0, 1], targets[0, 0])
            self.assertEqual(targets[0, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
