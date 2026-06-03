"""Correctness evals — schema compliance, data accuracy, output format."""

import pytest
from pydantic import ValidationError

from app.models import (
    FinalRecommendation,
    FundamentalData,
    MarketDataReport,
    SentimentReport,
    TechnicalData,
)
from app.tools import compute_technical_indicators, get_fundamentals, get_price_history


# ---------------------------------------------------------------------------
# Schema Compliance
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    """Verify that all models enforce their contracts."""

    def test_technical_data_rsi_bounds(self):
        """RSI must be between 0 and 100."""
        with pytest.raises(ValidationError):
            TechnicalData(
                current_price=100.0, rsi=150.0, rsi_signal="neutral",
                macd_line=0, macd_signal_line=0, macd_histogram=0,
                macd_crossover="none", sma_50=100, sma_200=100,
                price_vs_sma50="above", price_vs_sma200="above",
                trend="sideways", bollinger_upper=110, bollinger_lower=90,
                bollinger_position="upper_half", atr=1.0, atr_percent=1.0,
                volatility="moderate",
            )

    def test_recommendation_enum_enforced(self):
        """Recommendation must be exactly Buy, Sell, or Hold."""
        with pytest.raises(ValidationError):
            FinalRecommendation(
                ticker="TEST", company_name="Test",
                recommendation="Strong Buy",  # Invalid
                confidence=80, sentiment="positive", rationale="test",
            )

    def test_confidence_bounds_enforced(self):
        """Confidence must be 0-100."""
        with pytest.raises(ValidationError):
            FinalRecommendation(
                ticker="TEST", company_name="Test",
                recommendation="Buy",
                confidence=150,  # Invalid
                sentiment="positive", rationale="test",
            )

    def test_sentiment_score_bounds(self):
        """Sentiment scores must be -1 to +1."""
        with pytest.raises(ValidationError):
            SentimentReport(
                ticker="TEST", analyst_consensus="buy",
                news_score=2.0,  # Invalid
                news_sentiment="positive", overall_score=0.5,
                summary="test",
            )

    def test_valid_recommendation_passes(self):
        """A properly formed recommendation passes validation."""
        rec = FinalRecommendation(
            ticker="AAPL", company_name="Apple Inc.",
            recommendation="Hold", confidence=65,
            sentiment="Mixed signals", rationale="Technical momentum positive but valuation stretched.",
        )
        assert rec.recommendation == "Hold"
        assert rec.confidence == 65


# ---------------------------------------------------------------------------
# Data Accuracy (requires network — mark as integration test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDataAccuracy:
    """Verify that tools return accurate, well-formed data."""

    def test_price_history_returns_dataframe(self):
        """get_price_history returns non-empty DataFrame for valid ticker."""
        df = get_price_history("AAPL", period="1mo")
        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) >= 15  # At least 15 trading days in a month

    def test_technical_indicators_valid_ranges(self):
        """Computed indicators fall within expected ranges."""
        df = get_price_history("MSFT", period="6mo")
        indicators = compute_technical_indicators(df)

        assert 0 <= indicators["rsi"] <= 100
        assert indicators["rsi_signal"] in ("overbought", "neutral", "oversold")
        assert indicators["macd_crossover"] in ("bullish", "bearish", "none")
        assert indicators["trend"] in ("uptrend", "downtrend", "sideways")
        assert indicators["volatility"] in ("low", "moderate", "high", "extreme")
        assert indicators["current_price"] > 0

    def test_fundamentals_returns_data(self):
        """get_fundamentals returns at least some metrics for a major stock."""
        data = get_fundamentals("AAPL")
        # At least PE should be available for Apple
        assert data["pe_ratio"] is not None or data["market_cap"] is not None
