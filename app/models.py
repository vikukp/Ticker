"""Pydantic models — data contracts for all agents and the orchestrator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Market Data Agent output
# ---------------------------------------------------------------------------


class TechnicalData(BaseModel):
    """Technical indicator values computed from price history."""

    current_price: float
    rsi: float = Field(..., ge=0, le=100)
    rsi_signal: Literal["overbought", "neutral", "oversold"]
    macd_line: float
    macd_signal_line: float
    macd_histogram: float
    macd_crossover: Literal["bullish", "bearish", "none"]
    sma_50: float
    sma_200: float
    price_vs_sma50: Literal["above", "below"]
    price_vs_sma200: Literal["above", "below"]
    trend: Literal["uptrend", "downtrend", "sideways"]
    bollinger_upper: float
    bollinger_lower: float
    bollinger_position: Literal["above_upper", "upper_half", "lower_half", "below_lower"]
    atr: float
    atr_percent: float
    volatility: Literal["low", "moderate", "high", "extreme"]


class FundamentalData(BaseModel):
    """Key fundamental metrics for valuation."""

    pe_ratio: float | None = None
    peg_ratio: float | None = None
    revenue_growth: float | None = None
    profit_margin: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None


class MarketDataReport(BaseModel):
    """Combined output of the Market Data Agent."""

    ticker: str
    company_name: str
    technical: TechnicalData
    fundamental: FundamentalData
    technical_summary: str
    fundamental_summary: str


# ---------------------------------------------------------------------------
# Sentiment Agent output
# ---------------------------------------------------------------------------


class SentimentReport(BaseModel):
    """Output of the Sentiment Agent."""

    ticker: str
    analyst_buy: int = 0
    analyst_hold: int = 0
    analyst_sell: int = 0
    analyst_consensus: Literal["strong_buy", "buy", "hold", "sell", "strong_sell", "unavailable"]
    news_score: float = Field(0.0, ge=-1.0, le=1.0)
    news_sentiment: Literal["very_positive", "positive", "neutral", "negative", "very_negative"]
    headline_count: int = 0
    overall_score: float = Field(0.0, ge=-1.0, le=1.0)
    summary: str


# ---------------------------------------------------------------------------
# Synthesizer Agent output (Final Recommendation)
# ---------------------------------------------------------------------------


class FinalRecommendation(BaseModel):
    """The final investment recommendation — output of the Synthesizer Agent."""

    ticker: str
    company_name: str
    recommendation: Literal["Buy", "Sell", "Hold"]
    confidence: int = Field(..., ge=0, le=100)
    sentiment: str
    rationale: str
    caveats: list[str] = Field(default_factory=list)
    key_metrics: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator state
# ---------------------------------------------------------------------------


class ResearchState(BaseModel):
    """Internal state passed through the LangGraph orchestrator."""

    ticker: str
    stock_info: dict = Field(default_factory=dict)
    market_data: MarketDataReport | None = None
    sentiment: SentimentReport | None = None
    recommendation: FinalRecommendation | None = None
    anomalies: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    error: str | None = None
