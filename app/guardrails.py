"""Guardrails — input validation, output validation, anomaly detection."""

from __future__ import annotations

import re

import yfinance as yf

from app.config import config
from app.models import FinalRecommendation, MarketDataReport, SentimentReport


# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------


def validate_ticker(raw_input: str) -> tuple[bool, str, str]:
    """Validate and sanitize ticker input.

    Returns:
        (is_valid, cleaned_ticker, error_message)
    """
    # Strip and uppercase
    cleaned = raw_input.strip().upper()

    # Check length
    if not cleaned:
        return False, "", "Ticker cannot be empty."

    if len(cleaned) > config.MAX_TICKER_LENGTH:
        return False, "", f"Ticker too long (max {config.MAX_TICKER_LENGTH} characters)."

    # Allow only alphanumeric + dots (for BRK.A) + hyphens
    if not re.match(r"^[A-Z0-9.\-]+$", cleaned):
        return False, "", "Ticker contains invalid characters. Use letters, numbers, dots, or hyphens only."

    # Normalize: Yahoo Finance uses hyphens for share classes (BRK.B → BRK-B)
    cleaned = cleaned.replace(".", "-")

    # Check if ticker exists in yfinance
    try:
        stock = yf.Ticker(cleaned)
        # Try fetching recent history first — most reliable check
        hist = stock.history(period="5d")
        if not hist.empty:
            return True, cleaned, ""
        # Fallback: check info dict for price fields
        info = stock.info or {}
        has_price = (
            info.get("regularMarketPrice") is not None
            or info.get("currentPrice") is not None
            or info.get("previousClose") is not None
            or info.get("regularMarketPreviousClose") is not None
        )
        if has_price:
            return True, cleaned, ""
        return False, "", f"Ticker '{cleaned}' not found or has no market data."
    except Exception as e:
        return False, "", f"Unable to validate ticker: {str(e)[:100]}"

    return True, cleaned, ""


# ---------------------------------------------------------------------------
# Anomaly Detection (Hybrid orchestrator trigger)
# ---------------------------------------------------------------------------


def detect_anomalies(
    market_data: MarketDataReport | None,
    sentiment: SentimentReport | None,
) -> list[str]:
    """Detect data anomalies that require extra caution.

    Returns a list of anomaly descriptions. Empty list = normal conditions.
    """
    anomalies = []

    if market_data:
        tech = market_data.technical
        fund = market_data.fundamental

        # Extreme RSI
        if tech.rsi > 85:
            anomalies.append(f"Extreme overbought RSI ({tech.rsi})")
        elif tech.rsi < 15:
            anomalies.append(f"Extreme oversold RSI ({tech.rsi})")

        # Negative PE (unprofitable)
        if fund.pe_ratio is not None and fund.pe_ratio < 0:
            anomalies.append("Negative P/E — company is currently unprofitable")

        # Absurdly high PE
        if fund.pe_ratio is not None and fund.pe_ratio > 200:
            anomalies.append(f"Extreme P/E ratio ({fund.pe_ratio}) — possible distortion")

        # High leverage
        if fund.debt_to_equity is not None and fund.debt_to_equity > 5.0:
            anomalies.append(f"High leverage (Debt/Equity: {fund.debt_to_equity})")

        # Extreme volatility
        if tech.volatility == "extreme":
            anomalies.append(f"Extreme volatility (ATR: {tech.atr_percent}%)")

    if market_data is None:
        anomalies.append("Market data unavailable — partial analysis only")

    if sentiment is None:
        anomalies.append("Sentiment data unavailable — partial analysis only")

    return anomalies


# ---------------------------------------------------------------------------
# Output Guardrails
# ---------------------------------------------------------------------------


def validate_output(recommendation: FinalRecommendation) -> tuple[bool, list[str]]:
    """Validate the final recommendation output.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check recommendation is valid enum
    if recommendation.recommendation not in ("Buy", "Sell", "Hold"):
        issues.append(f"Invalid recommendation: {recommendation.recommendation}")

    # Check confidence bounds
    if recommendation.confidence < config.CONFIDENCE_MIN:
        issues.append(f"Confidence below minimum: {recommendation.confidence}")
    if recommendation.confidence > config.CONFIDENCE_MAX:
        issues.append(f"Confidence above maximum: {recommendation.confidence}")

    # Check for prohibited phrases
    full_text = f"{recommendation.rationale} {recommendation.sentiment}".lower()
    for phrase in config.PROHIBITED_PHRASES:
        if phrase in full_text:
            issues.append(f"Prohibited phrase detected: '{phrase}'")

    # Check rationale is not empty
    if not recommendation.rationale or len(recommendation.rationale) < 10:
        issues.append("Rationale is missing or too short")

    return len(issues) == 0, issues


def sanitize_output(recommendation: FinalRecommendation) -> FinalRecommendation:
    """Remove prohibited phrases from output text."""
    rationale = recommendation.rationale
    sentiment = recommendation.sentiment

    for phrase in config.PROHIBITED_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        rationale = pattern.sub("", rationale)
        sentiment = pattern.sub("", sentiment)

    return recommendation.model_copy(update={
        "rationale": rationale.strip(),
        "sentiment": sentiment.strip(),
        "confidence": max(0, min(100, recommendation.confidence)),
    })
