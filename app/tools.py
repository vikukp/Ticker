"""Data tools — wrappers around yfinance and the ta library.

These are pure data-fetching functions. No LLM, no opinions.
Each function returns raw data that agents consume.
"""

from __future__ import annotations

import os
import ssl

# Fix SSL certificate issues on corporate networks — must be set before importing yfinance
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

import yfinance as yf
import pandas as pd
import ta
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch yfinance to disable SSL verification
try:
    from yfinance.utils import session as _yf_session
    _yf_session.verify = False
except (ImportError, AttributeError):
    pass

# Also try patching via the data module
try:
    yf.set_tz_cache_location = getattr(yf, "set_tz_cache_location", None)
    # Force disable SSL for curl_cffi sessions used by yfinance
    from curl_cffi.requests import Session as CurlSession
    _original_init = CurlSession.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _original_init(self, *args, **kwargs)

    CurlSession.__init__ = _patched_init
except (ImportError, AttributeError):
    pass

from app.config import config


# ---------------------------------------------------------------------------
# Stock info
# ---------------------------------------------------------------------------


def get_stock_info(ticker: str) -> dict:
    """Fetch basic stock info: name, sector, price, market cap."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
    }


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------


def get_price_history(ticker: str, period: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV daily data. Returns a DataFrame with Date index."""
    period = period or config.PRICE_HISTORY_PERIOD
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No price history available for {ticker}")
    return df


# ---------------------------------------------------------------------------
# Market conditions
# ---------------------------------------------------------------------------


def get_market_conditions(period: str = "3mo") -> dict:
    """Estimate broad market conditions from SPY price action."""
    df = get_price_history("SPY", period=period)
    close = df["Close"]
    returns = close.pct_change().dropna()

    current_price = close.iloc[-1]
    start_price = close.iloc[0]
    return_percent = ((current_price / start_price) - 1) * 100 if start_price else 0

    sma_50 = close.rolling(window=min(50, len(close))).mean().iloc[-1]
    annualized_volatility = returns.std() * (252 ** 0.5) * 100 if not returns.empty else 0

    if current_price > sma_50 and return_percent > 0:
        regime = "supportive"
    elif current_price < sma_50 and return_percent < 0:
        regime = "risk_off"
    else:
        regime = "mixed"

    if annualized_volatility > 25 or return_percent < -10:
        risk_level = "high"
    elif annualized_volatility > 18 or return_percent < -5:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "benchmark": "SPY",
        "regime": regime,
        "risk_level": risk_level,
        "return_percent": round(return_percent, 2),
        "volatility_percent": round(annualized_volatility, 2),
        "price_vs_sma50": "above" if current_price > sma_50 else "below",
    }


# ---------------------------------------------------------------------------
# Technical indicators (computed from price history)
# ---------------------------------------------------------------------------


def compute_technical_indicators(df: pd.DataFrame) -> dict:
    """Compute all technical indicators from a price DataFrame.

    Returns a flat dict of indicator values.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    rsi = rsi_indicator.rsi().iloc[-1]

    # MACD
    macd_indicator = ta.trend.MACD(close=close)
    macd_line = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_hist = macd_indicator.macd_diff().iloc[-1]

    # Determine MACD crossover (check last 3 days)
    macd_series = macd_indicator.macd()
    signal_series = macd_indicator.macd_signal()
    if len(macd_series) >= 3:
        prev_diff = macd_series.iloc[-3] - signal_series.iloc[-3]
        curr_diff = macd_series.iloc[-1] - signal_series.iloc[-1]
        if prev_diff < 0 and curr_diff > 0:
            macd_crossover = "bullish"
        elif prev_diff > 0 and curr_diff < 0:
            macd_crossover = "bearish"
        else:
            macd_crossover = "none"
    else:
        macd_crossover = "none"

    # SMA
    sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1]
    sma_200_series = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    sma_200 = sma_200_series.iloc[-1] if not sma_200_series.isna().iloc[-1] else sma_50 * 0.95

    current_price = close.iloc[-1]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_middle = bb.bollinger_mavg().iloc[-1]

    # Bollinger position
    if current_price > bb_upper:
        bb_position = "above_upper"
    elif current_price > bb_middle:
        bb_position = "upper_half"
    elif current_price > bb_lower:
        bb_position = "lower_half"
    else:
        bb_position = "below_lower"

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    atr = atr_indicator.average_true_range().iloc[-1]
    atr_percent = (atr / current_price) * 100 if current_price > 0 else 0

    # Volatility regime
    if atr_percent < 1.0:
        volatility = "low"
    elif atr_percent < 2.5:
        volatility = "moderate"
    elif atr_percent < 5.0:
        volatility = "high"
    else:
        volatility = "extreme"

    # Trend determination
    if current_price > sma_50 > sma_200:
        trend = "uptrend"
    elif current_price < sma_50 < sma_200:
        trend = "downtrend"
    else:
        trend = "sideways"

    # RSI signal
    if rsi > 70:
        rsi_signal = "overbought"
    elif rsi < 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    return {
        "current_price": round(current_price, 2),
        "rsi": round(rsi, 2),
        "rsi_signal": rsi_signal,
        "macd_line": round(macd_line, 4),
        "macd_signal_line": round(macd_signal, 4),
        "macd_histogram": round(macd_hist, 4),
        "macd_crossover": macd_crossover,
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "price_vs_sma50": "above" if current_price > sma_50 else "below",
        "price_vs_sma200": "above" if current_price > sma_200 else "below",
        "trend": trend,
        "bollinger_upper": round(bb_upper, 2),
        "bollinger_lower": round(bb_lower, 2),
        "bollinger_position": bb_position,
        "atr": round(atr, 4),
        "atr_percent": round(atr_percent, 2),
        "volatility": volatility,
    }


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------


def get_fundamentals(ticker: str) -> dict:
    """Fetch fundamental metrics from yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "pe_ratio": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cash_flow": info.get("freeCashflow"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


# ---------------------------------------------------------------------------
# Sentiment data (news + analyst ratings)
# ---------------------------------------------------------------------------


def get_sentiment_data(ticker: str) -> dict:
    """Fetch news headlines and analyst recommendations from yfinance."""
    stock = yf.Ticker(ticker)

    # News headlines
    news = stock.news or []
    headlines = []
    for item in news[:15]:  # Limit to 15 most recent
        title = item.get("title") or item.get("content", {}).get("title", "")
        if title:
            headlines.append(title)

    # Analyst recommendations
    try:
        rec = stock.recommendations
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            analyst_data = {
                "buy": int(latest.get("strongBuy", 0)) + int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)) + int(latest.get("strongSell", 0)),
            }
        else:
            analyst_data = {"buy": 0, "hold": 0, "sell": 0}
    except Exception:
        analyst_data = {"buy": 0, "hold": 0, "sell": 0}

    return {
        "headlines": headlines,
        "headline_count": len(headlines),
        "analyst_buy": analyst_data["buy"],
        "analyst_hold": analyst_data["hold"],
        "analyst_sell": analyst_data["sell"],
    }
