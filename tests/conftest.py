"""Test fixtures and mock data for evals."""

import pytest
from app.models import (
    FinalRecommendation,
    FundamentalData,
    MarketDataReport,
    SentimentReport,
    TechnicalData,
)


@pytest.fixture
def bullish_market_data() -> MarketDataReport:
    """Strongly bullish market data."""
    return MarketDataReport(
        ticker="TEST",
        company_name="Test Corp",
        technical=TechnicalData(
            current_price=150.0,
            rsi=45.0,
            rsi_signal="neutral",
            macd_line=2.5,
            macd_signal_line=1.0,
            macd_histogram=1.5,
            macd_crossover="bullish",
            sma_50=145.0,
            sma_200=130.0,
            price_vs_sma50="above",
            price_vs_sma200="above",
            trend="uptrend",
            bollinger_upper=160.0,
            bollinger_lower=140.0,
            bollinger_position="upper_half",
            atr=2.5,
            atr_percent=1.7,
            volatility="moderate",
        ),
        fundamental=FundamentalData(
            pe_ratio=18.0,
            peg_ratio=1.0,
            revenue_growth=0.20,
            profit_margin=0.25,
            debt_to_equity=0.5,
            free_cash_flow=5000000000,
            market_cap=200000000000,
            sector="Technology",
        ),
        technical_summary="Strong bullish momentum with MACD crossover and uptrend.",
        fundamental_summary="Fairly valued with strong growth metrics.",
    )


@pytest.fixture
def bearish_market_data() -> MarketDataReport:
    """Strongly bearish market data."""
    return MarketDataReport(
        ticker="TEST",
        company_name="Test Corp",
        technical=TechnicalData(
            current_price=50.0,
            rsi=75.0,
            rsi_signal="overbought",
            macd_line=-1.5,
            macd_signal_line=0.5,
            macd_histogram=-2.0,
            macd_crossover="bearish",
            sma_50=55.0,
            sma_200=65.0,
            price_vs_sma50="below",
            price_vs_sma200="below",
            trend="downtrend",
            bollinger_upper=58.0,
            bollinger_lower=45.0,
            bollinger_position="lower_half",
            atr=3.5,
            atr_percent=7.0,
            volatility="extreme",
        ),
        fundamental=FundamentalData(
            pe_ratio=45.0,
            peg_ratio=3.5,
            revenue_growth=-0.05,
            profit_margin=0.05,
            debt_to_equity=3.0,
            free_cash_flow=-1000000000,
            market_cap=10000000000,
            sector="Technology",
        ),
        technical_summary="Bearish MACD crossover with price in downtrend.",
        fundamental_summary="Overvalued with declining revenue and high debt.",
    )


@pytest.fixture
def bullish_sentiment() -> SentimentReport:
    """Strongly bullish sentiment."""
    return SentimentReport(
        ticker="TEST",
        analyst_buy=25,
        analyst_hold=5,
        analyst_sell=1,
        analyst_consensus="strong_buy",
        news_score=0.7,
        news_sentiment="positive",
        headline_count=10,
        overall_score=0.7,
        summary="Strong analyst consensus with mostly positive news coverage.",
    )


@pytest.fixture
def bearish_sentiment() -> SentimentReport:
    """Strongly bearish sentiment."""
    return SentimentReport(
        ticker="TEST",
        analyst_buy=2,
        analyst_hold=5,
        analyst_sell=20,
        analyst_consensus="sell",
        news_score=-0.6,
        news_sentiment="negative",
        headline_count=8,
        overall_score=-0.6,
        summary="Analyst downgrades and negative news dominate.",
    )
