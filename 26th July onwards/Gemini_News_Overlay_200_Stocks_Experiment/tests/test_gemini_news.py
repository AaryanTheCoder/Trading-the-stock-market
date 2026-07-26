from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import unittest


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / "source"))

from gemini_news import (
    StockDescriptor,
    build_prompt,
    make_request_body,
    parse_and_audit,
)


STOCKS = [
    StockDescriptor("AAA", "Alpha Corp", "Technology"),
    StockDescriptor("BBB", "Beta Inc", "Financials"),
]


def response_for(rows):
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps({"stocks": rows})}]
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "webSearchQueries": ["Alpha Corp news"],
                    "groundingChunks": [
                        {"web": {"title": "Publisher", "uri": "https://example.com"}}
                    ],
                },
            }
        ],
        "usageMetadata": {"totalTokenCount": 123},
    }


class GeminiNewsTests(unittest.TestCase):
    def test_prompt_and_search_window_end_at_cutoff(self):
        as_of = date(2024, 6, 28)
        prompt = build_prompt(STOCKS, as_of)
        body = make_request_body(prompt, as_of)
        window = body["tools"][0]["googleSearch"]["timeRangeFilter"]
        self.assertIn("2024-06-28", prompt)
        self.assertEqual(window["startTime"], "2024-03-30T00:00:00Z")
        self.assertEqual(window["endTime"], "2024-06-29T00:00:00Z")

    def test_valid_source_produces_confidence_scaled_score(self):
        response = response_for(
            [
                {
                    "ticker": "AAA",
                    "news_score": 80,
                    "confidence": 0.5,
                    "insufficient_information": False,
                    "summary": "New product.",
                    "sources": [
                        {
                            "title": "Product launch",
                            "url": "https://example.com/launch",
                            "published_date": "2024-06-20",
                        }
                    ],
                },
                {
                    "ticker": "BBB",
                    "news_score": -10,
                    "confidence": 0.8,
                    "insufficient_information": False,
                    "sources": [
                        {
                            "title": "Loan update",
                            "url": "https://example.com/loan",
                            "published_date": "2024-05-20",
                        }
                    ],
                },
            ]
        )
        rows, metadata = parse_and_audit(response, STOCKS, date(2024, 6, 28))
        self.assertAlmostEqual(rows[0]["effective_news_score"], 0.4)
        self.assertAlmostEqual(rows[1]["effective_news_score"], -0.08)
        self.assertEqual(metadata["audit_pass_count"], 2)

    def test_future_or_missing_source_is_zeroed(self):
        response = response_for(
            [
                {
                    "ticker": "AAA",
                    "news_score": 100,
                    "confidence": 1,
                    "insufficient_information": False,
                    "sources": [
                        {
                            "title": "Future leak",
                            "url": "https://example.com/future",
                            "published_date": "2024-07-01",
                        }
                    ],
                }
            ]
        )
        rows, metadata = parse_and_audit(response, STOCKS, date(2024, 6, 28))
        self.assertEqual(rows[0]["effective_news_score"], 0.0)
        self.assertFalse(rows[0]["audit_passed"])
        self.assertEqual(rows[1]["effective_news_score"], 0.0)
        self.assertIn("missing_ticker", rows[1]["audit_reasons"])
        self.assertEqual(metadata["audit_pass_count"], 0)


if __name__ == "__main__":
    unittest.main()

