"""Grounded Gemini news scoring with strict point-in-time auditing.

The API key is read only from GEMINI_API_KEY (or GOOGLE_API_KEY) and is never
written to a cache, prompt, error message, or model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
PROMPT_VERSION = "news-score-v1"
DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 10


@dataclass(frozen=True)
class StockDescriptor:
    ticker: str
    company: str
    sector: str


class GeminiError(RuntimeError):
    """A sanitized Gemini error that cannot contain an API key."""


def _parse_iso_day(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _clip_number(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(lower, min(upper, number))


def build_prompt(stocks: Iterable[StockDescriptor], as_of: date) -> str:
    """Build a deterministic prompt that bans information after ``as_of``."""
    stock_list = list(stocks)
    window_start = as_of - timedelta(days=90)
    rows = "\n".join(
        f"- {item.ticker} | {item.company} | sector: {item.sector}"
        for item in stock_list
    )
    return f"""You are a point-in-time US equity news analyst.

AS-OF CUTOFF: {as_of.isoformat()} 23:59:59 UTC.
SEARCH WINDOW: {window_start.isoformat()} through {as_of.isoformat()}.
FORECAST HORIZON: the next 20 US trading sessions after the cutoff.

Use Google Search for EACH stock below. Use only sources actually published
inside the search window. Never use an article, earnings result, price move,
retrospective summary, or any other fact published after the as-of cutoff.
Pretend you are operating at the cutoff and do not reveal later outcomes.

Score sector-relative expected NEWS impact, not past price momentum:
  +100 = exceptionally strong, credible positive catalyst
     0 = balanced, stale, immaterial, or insufficient evidence
  -100 = exceptionally strong, credible negative catalyst

Confidence is 0.0 to 1.0 and should be low for thin, contradictory, duplicated,
or speculative reporting. A stock with no qualifying source MUST receive
news_score 0, confidence 0, and insufficient_information true.

Stocks:
{rows}

Return JSON only, with exactly one object per requested ticker:
{{
  "as_of": "{as_of.isoformat()}",
  "forecast_trading_days": 20,
  "stocks": [
    {{
      "ticker": "AAPL",
      "news_score": 0,
      "confidence": 0.0,
      "insufficient_information": true,
      "summary": "One short factual sentence.",
      "catalyst": "Main catalyst or none.",
      "risk": "Main risk or none.",
      "sources": [
        {{
          "title": "Article title",
          "url": "https://publisher.example/article",
          "published_date": "YYYY-MM-DD"
        }}
      ]
    }}
  ]
}}

Do not omit tickers. Do not add markdown or commentary outside the JSON."""


def make_request_body(prompt: str, as_of: date) -> dict[str, Any]:
    start = datetime.combine(
        as_of - timedelta(days=90), datetime.min.time(), tzinfo=timezone.utc
    )
    end = datetime.combine(
        as_of + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [
            {
                "googleSearch": {
                    "timeRangeFilter": {
                        "startTime": start.isoformat().replace("+00:00", "Z"),
                        "endTime": end.isoformat().replace("+00:00", "Z"),
                    }
                }
            }
        ],
        "generationConfig": {
            # Google Search grounding currently rejects responseMimeType on
            # this endpoint. The prompt still requires JSON and the parser
            # accepts either plain JSON or a fenced JSON response.
            "temperature": 0.0,
            "maxOutputTokens": 8192,
        },
    }


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiError(
            "Gemini credential missing. Export GEMINI_API_KEY in the shell; "
            "do not put it in the repository or command arguments."
        )
    return key


def _safe_http_error(error: HTTPError) -> str:
    try:
        raw = error.read(4096).decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        message = str(parsed.get("error", {}).get("message", ""))
    except Exception:
        message = ""
    # Google errors occasionally echo malformed authentication material.
    message = re.sub(r"AQ\.[A-Za-z0-9_-]+", "[REDACTED]", message)
    return f"Gemini HTTP {error.code}" + (f": {message}" if message else "")


def call_gemini(
    body: dict[str, Any], model_name: str = DEFAULT_MODEL_NAME
) -> dict[str, Any]:
    """Call GenerateContent with bounded exponential retry and sanitized errors."""
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    last_error = "unknown error"
    configured_retries = int(os.environ.get("GEMINI_MAX_RETRIES", MAX_RETRIES))
    configured_retries = max(1, min(MAX_RETRIES, configured_retries))
    transport_failures = 0
    for attempt in range(configured_retries):
        request = Request(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent"
            ),
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": _api_key(),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = _safe_http_error(error)
            retryable = error.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt == configured_retries - 1:
                raise GeminiError(last_error) from None
            retry_after = error.headers.get("Retry-After")
            try:
                server_delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                server_delay = 0.0
            body_delay = re.search(
                r"retry in ([0-9]+(?:\.[0-9]+)?)s", last_error, re.IGNORECASE
            )
            if body_delay:
                server_delay = max(server_delay, float(body_delay.group(1)) + 1.0)
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            transport_failures += 1
            last_error = f"Gemini transport/JSON failure: {type(error).__name__}"
            if transport_failures >= 3 or attempt == configured_retries - 1:
                raise GeminiError(last_error) from None
            server_delay = 0.0
        delay = max(
            server_delay,
            min(60.0, 2.0**attempt + random.random()),
        )
        print(
            f"{last_error.split(':', 1)[0]}; retry "
            f"{attempt + 2}/{configured_retries} in {delay:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise GeminiError(last_error)


def extract_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(
        str(part.get("text", "")) for part in parts if isinstance(part, dict)
    ).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text:
        raise GeminiError("Gemini candidate contained no text.")
    return text


def grounding_summary(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates", [])
    metadata = candidates[0].get("groundingMetadata", {}) if candidates else {}
    chunks = metadata.get("groundingChunks", [])
    sources = []
    for chunk in chunks if isinstance(chunks, list) else []:
        web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
        uri = web.get("uri")
        if uri:
            sources.append({"title": str(web.get("title", "")), "uri": str(uri)})
    return {
        "web_search_queries": metadata.get("webSearchQueries", []),
        "grounding_sources": sources,
        "grounding_supports": metadata.get("groundingSupports", []),
    }


def parse_and_audit(
    response: dict[str, Any],
    requested: Iterable[StockDescriptor],
    as_of: date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse scores and zero records that fail point-in-time source checks."""
    requested_list = list(requested)
    requested_map = {item.ticker.upper(): item for item in requested_list}
    try:
        document = json.loads(extract_text(response))
    except json.JSONDecodeError as error:
        raise GeminiError(f"Gemini returned invalid JSON at byte {error.pos}.") from None
    rows = document.get("stocks", []) if isinstance(document, dict) else []
    by_ticker = {
        str(row.get("ticker", "")).upper(): row
        for row in rows
        if isinstance(row, dict)
    }
    window_start = as_of - timedelta(days=90)
    normalized: list[dict[str, Any]] = []
    for ticker, descriptor in requested_map.items():
        row = by_ticker.get(ticker, {})
        raw_score = _clip_number(row.get("news_score"), -100.0, 100.0, 0.0)
        confidence = _clip_number(row.get("confidence"), 0.0, 1.0, 0.0)
        insufficient = bool(row.get("insufficient_information", not row))
        accepted_sources: list[dict[str, str]] = []
        rejected_sources: list[dict[str, str]] = []
        sources = row.get("sources", [])
        for source in sources if isinstance(sources, list) else []:
            if not isinstance(source, dict):
                continue
            source_day = _parse_iso_day(source.get("published_date"))
            normalized_source = {
                "title": str(source.get("title", ""))[:500],
                "url": str(source.get("url", ""))[:2000],
                "published_date": source_day.isoformat() if source_day else "",
            }
            valid = (
                source_day is not None
                and window_start <= source_day <= as_of
                and normalized_source["url"].startswith(("https://", "http://"))
            )
            (accepted_sources if valid else rejected_sources).append(normalized_source)

        audit_reasons = []
        if not row:
            audit_reasons.append("missing_ticker")
        if insufficient:
            audit_reasons.append("model_reported_insufficient_information")
        if not accepted_sources:
            audit_reasons.append("no_source_with_valid_point_in_time_date")
        audit_passed = not audit_reasons
        effective_score = raw_score / 100.0 * confidence if audit_passed else 0.0
        normalized.append(
            {
                "ticker": ticker,
                "company": descriptor.company,
                "sector": descriptor.sector,
                "as_of": as_of.isoformat(),
                "raw_news_score": raw_score / 100.0,
                "confidence": confidence,
                "effective_news_score": effective_score,
                "insufficient_information": insufficient,
                "audit_passed": audit_passed,
                "audit_reasons": audit_reasons,
                "summary": str(row.get("summary", ""))[:1000],
                "catalyst": str(row.get("catalyst", ""))[:500],
                "risk": str(row.get("risk", ""))[:500],
                "accepted_sources": accepted_sources,
                "rejected_sources": rejected_sources,
            }
        )
    meta = grounding_summary(response)
    meta.update(
        {
            "requested_count": len(requested_list),
            "returned_count": len(by_ticker),
            "audit_pass_count": sum(row["audit_passed"] for row in normalized),
            "usage_metadata": response.get("usageMetadata", {}),
            "finish_reason": (
                response.get("candidates", [{}])[0].get("finishReason", "")
                if response.get("candidates")
                else ""
            ),
        }
    )
    return normalized, meta


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def score_batch(
    stocks: Iterable[StockDescriptor],
    as_of: date,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    stock_list = list(stocks)
    prompt = build_prompt(stock_list, as_of)
    response: dict[str, Any] = {}
    parse_errors: list[str] = []
    for _ in range(3):
        response = call_gemini(make_request_body(prompt, as_of), model_name)
        try:
            records, metadata = parse_and_audit(response, stock_list, as_of)
            break
        except GeminiError as error:
            parse_errors.append(str(error))
    else:
        # Fail closed after repeated malformed generations. Keeping a neutral
        # record lets a long cache collection continue while making the loss
        # of usable evidence explicit in the audit.
        records = [
            {
                "ticker": stock.ticker,
                "company": stock.company,
                "sector": stock.sector,
                "as_of": as_of.isoformat(),
                "raw_news_score": 0.0,
                "confidence": 0.0,
                "effective_news_score": 0.0,
                "insufficient_information": True,
                "audit_passed": False,
                "audit_reasons": ["invalid_json_after_three_generations"],
                "summary": "",
                "catalyst": "",
                "risk": "",
                "accepted_sources": [],
                "rejected_sources": [],
            }
            for stock in stock_list
        ]
        metadata = grounding_summary(response)
        metadata.update(
            {
                "requested_count": len(stock_list),
                "returned_count": 0,
                "audit_pass_count": 0,
                "usage_metadata": response.get("usageMetadata", {}),
                "finish_reason": (
                    response.get("candidates", [{}])[0].get("finishReason", "")
                    if response.get("candidates")
                    else ""
                ),
                "parse_errors": parse_errors,
            }
        )
    if parse_errors:
        metadata["prior_parse_errors"] = parse_errors
    return {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
        "as_of": as_of.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": prompt_hash(prompt),
        "requested_tickers": [stock.ticker for stock in stock_list],
        "records": records,
        "metadata": metadata,
        # Retain the provider response for reproducibility and qualitative
        # review. Authentication is supplied only in the request header and
        # therefore cannot appear here under normal API behavior.
        "raw_response": response,
    }


def write_cache(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
