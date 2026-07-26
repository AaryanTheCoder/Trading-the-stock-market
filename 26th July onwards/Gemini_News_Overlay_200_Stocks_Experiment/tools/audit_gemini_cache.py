"""Summarize cached Gemini responses without requiring full date coverage."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.1-flash-lite",
        ),
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--cache-tag",
        help="Audit this cache folder instead of the model-name folder.",
    )
    args = parser.parse_args()
    slug = args.cache_tag or args.model.replace(".", "_").replace("-", "_")
    if "/" in slug or "\\" in slug or slug in {".", ".."}:
        parser.error("--cache-tag must be one safe folder name")
    cache_root = MODEL_ROOT / "data" / "cache" / slug
    output_csv = (
        MODEL_ROOT / "data" / "training" / f"{slug}_response_audit.csv"
    )
    output_json = (
        MODEL_ROOT / "data" / "training" / f"{slug}_response_audit_summary.json"
    )
    rows = []
    total_tokens = 0
    query_count = 0
    reason_counts: Counter[str] = Counter()
    models: Counter[str] = Counter()
    for path in sorted(cache_root.glob("*/batch_*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        models[str(document.get("model", ""))] += 1
        metadata = document.get("metadata", {})
        usage = metadata.get("usage_metadata", {})
        total_tokens += int(usage.get("totalTokenCount", 0) or 0)
        query_count += len(metadata.get("web_search_queries", []))
        for record in document.get("records", []):
            reasons = list(record.get("audit_reasons", []))
            reason_counts.update(reasons)
            rows.append(
                {
                    "as_of": document.get("as_of"),
                    "batch_file": str(path.relative_to(MODEL_ROOT)),
                    "ticker": record.get("ticker"),
                    "raw_news_score": record.get("raw_news_score"),
                    "confidence": record.get("confidence"),
                    "effective_news_score": record.get("effective_news_score"),
                    "audit_passed": bool(record.get("audit_passed")),
                    "accepted_source_count": len(
                        record.get("accepted_sources", [])
                    ),
                    "rejected_source_count": len(
                        record.get("rejected_sources", [])
                    ),
                    "audit_reasons": "|".join(reasons),
                    "summary": record.get("summary", ""),
                    "catalyst": record.get("catalyst", ""),
                    "risk": record.get("risk", ""),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No Gemini batch caches under {cache_root}.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    accepted = frame.loc[frame["audit_passed"]]
    summary = {
        "batch_files": int(frame["batch_file"].nunique()),
        "rebalance_dates_touched": int(frame["as_of"].nunique()),
        "stock_date_responses": len(frame),
        "audit_pass_count": int(frame["audit_passed"].sum()),
        "audit_pass_rate": float(frame["audit_passed"].mean()),
        "nonzero_effective_scores": int(
            (frame["effective_news_score"].abs() > 1e-12).sum()
        ),
        "mean_confidence_all": float(frame["confidence"].mean()),
        "mean_confidence_accepted": (
            float(accepted["confidence"].mean()) if len(accepted) else 0.0
        ),
        "mean_effective_score_accepted": (
            float(accepted["effective_news_score"].mean())
            if len(accepted)
            else 0.0
        ),
        "minimum_effective_score": float(frame["effective_news_score"].min()),
        "maximum_effective_score": float(frame["effective_news_score"].max()),
        "grounded_search_queries": query_count,
        "reported_total_tokens": total_tokens,
        "models_by_batch": dict(models),
        "audit_failure_reasons": dict(reason_counts.most_common()),
        "coverage_warning": (
            "Partial cache only. Do not select news weight or claim a holdout "
            "return until every required rebalance candidate is cached."
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
