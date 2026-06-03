"""Agent definitions — Market Data Agent, Sentiment Agent, Synthesizer Agent.

Each agent is a function that takes inputs and returns a structured Pydantic model.
LLM is used only where judgment is needed (sentiment scoring, synthesis reasoning).
"""

from __future__ import annotations

import logging
import re

from app.config import config
from app.models import (
    FundamentalData,
    MarketDataReport,
    SentimentReport,
    FinalRecommendation,
    TechnicalData,
)
from app.tools import (
    compute_technical_indicators,
    get_fundamentals,
    get_price_history,
    get_sentiment_data,
    get_stock_info,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def _parse_labeled_response(text: str) -> dict[str, str]:
    """Parse simple LABEL: value responses, keeping multiline values together."""
    values: dict[str, str] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^([A-Za-z_ ]+)\s*:\s*(.*)$", line)
        if match:
            current_key = match.group(1).strip().upper().replace(" ", "_")
            values[current_key] = match.group(2).strip()
        elif current_key:
            values[current_key] = f"{values[current_key]} {line}".strip()

    return values


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _get_llm():
    """Create LLM instance (lazy import to speed up app startup).
    
    Supports OpenAI, Azure OpenAI, and local Ollama based on LLM_PROVIDER config.
    """
    if config.LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaLLM
            logger.info(f"Initializing Ollama LLM: {config.OLLAMA_MODEL} at {config.OLLAMA_BASE_URL}")
            
            # Test connection
            import requests
            try:
                response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.debug("Ollama server is reachable")
                else:
                    logger.warning(f"Ollama server returned status {response.status_code}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}: {e}")
                raise ConnectionError(
                    f"Ollama not running at {config.OLLAMA_BASE_URL}. "
                    "Please start Ollama with: ollama serve"
                )
            
            return OllamaLLM(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_MODEL,
                temperature=config.TEMPERATURE,
                num_predict=500,  # Limit response length for faster generation
            )
        except ImportError as e:
            logger.error(f"langchain_ollama not installed: {e}")
            logger.warning("Falling back to OpenAI")
            return _get_openai_llm()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
    elif config.LLM_PROVIDER == "azure":
        logger.info("Using Azure OpenAI LLM")
        return _get_azure_llm()
    else:
        logger.info("Using OpenAI LLM")
        return _get_openai_llm()


def _get_openai_llm():
    """Create OpenAI LLM instance."""
    import httpx
    from langchain_openai import ChatOpenAI

    logger.info(f"Initializing OpenAI LLM: {config.MODEL_NAME}")
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not configured. "
            "Set OPENAI_API_KEY in your .env file or use LLM_PROVIDER=ollama"
        )
    
    # Disable SSL verification for corporate networks with custom certificates
    http_client = httpx.Client(verify=False)
    
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=config.TEMPERATURE,
        http_client=http_client,
    )


def _get_azure_llm():
    """Create Azure OpenAI LLM instance."""
    from langchain_openai import AzureChatOpenAI

    logger.info("Initializing Azure OpenAI LLM")
    if not config.AZURE_OPENAI_API_KEY:
        raise ValueError(
            "Azure OpenAI API key not configured. "
            "Set AZURE_OPENAI_API_KEY or use LLM_PROVIDER=ollama"
        )
    
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=config.TEMPERATURE,
        openai_api_key=config.AZURE_OPENAI_API_KEY,
    )


def _invoke_llm(system_prompt: str, user_content: str) -> str:
    """Invoke the LLM with a system + user message pair (with retry)."""
    import time
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = None
    last_error = None

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            if attempt > 0:
                time.sleep(1 * attempt)  # Brief backoff between retries
                logger.info(f"LLM retry attempt {attempt + 1}")

            if llm is None:
                logger.debug(f"LLM Provider: {config.LLM_PROVIDER}")
                llm = _get_llm()
                logger.debug(f"LLM instance created: {type(llm).__name__}")

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            logger.debug(f"LLM response received: {len(str(response))} chars")

            # Handle both message objects and raw strings
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            last_error = e
            logger.warning(f"LLM attempt {attempt + 1} failed: {str(e)[:100]}")

    logger.error(f"LLM invocation failed after {config.MAX_RETRIES + 1} attempts", exc_info=True)
    raise last_error


# ---------------------------------------------------------------------------
# Agent 1: Market Data Agent
# ---------------------------------------------------------------------------

MARKET_DATA_SYSTEM_PROMPT = """You are a quantitative market analyst. You are given computed technical 
indicators and fundamental metrics for a stock.

Write TWO brief summaries (2-3 sentences each):
1. technical_summary — summarize the technical picture (momentum, trend, volatility)
2. fundamental_summary — summarize the valuation picture (expensive/cheap, growth, health)

Rules:
- State facts only. Reference specific numbers.
- Never use "I" or give advice.
- Never say "you should" or "guaranteed".

Respond in this exact format:
TECHNICAL: <your technical summary>
FUNDAMENTAL: <your fundamental summary>
"""


def _fallback_market_summaries(
    ticker: str,
    stock_info: dict,
    tech_indicators: dict,
    fundamentals: dict,
) -> tuple[str, str]:
    """Generate fact-only summaries when the LLM is unavailable."""
    company_name = stock_info.get("company_name", ticker)
    technical_summary = (
        f"{company_name} trades at ${tech_indicators['current_price']} with "
        f"RSI {tech_indicators['rsi']} ({tech_indicators['rsi_signal']}) and a "
        f"{tech_indicators['trend']} trend. MACD crossover is "
        f"{tech_indicators['macd_crossover']}, price is "
        f"{tech_indicators['price_vs_sma50']} SMA50 and "
        f"{tech_indicators['price_vs_sma200']} SMA200, with "
        f"{tech_indicators['volatility']} volatility."
    )

    pe = fundamentals.get("pe_ratio")
    peg = fundamentals.get("peg_ratio")
    revenue_growth = fundamentals.get("revenue_growth")
    margin = fundamentals.get("profit_margin")
    debt = fundamentals.get("debt_to_equity")
    fundamental_summary = (
        f"Valuation metrics show P/E {pe if pe is not None else 'unavailable'} "
        f"and PEG {peg if peg is not None else 'unavailable'}. Revenue growth is "
        f"{revenue_growth if revenue_growth is not None else 'unavailable'}, "
        f"profit margin is {margin if margin is not None else 'unavailable'}, "
        f"and debt/equity is {debt if debt is not None else 'unavailable'}."
    )
    return technical_summary, fundamental_summary


def run_market_data_agent(ticker: str) -> MarketDataReport:
    """Market Data Agent — fetches price data, computes indicators, summarizes."""
    # Fetch data
    stock_info = get_stock_info(ticker)
    df = get_price_history(ticker)
    tech_indicators = compute_technical_indicators(df)
    fundamentals = get_fundamentals(ticker)

    # Build structured data
    technical = TechnicalData(**tech_indicators)
    fundamental = FundamentalData(**fundamentals)

    # LLM generates summaries
    data_context = (
        f"Ticker: {ticker} ({stock_info['company_name']})\n"
        f"Sector: {fundamentals.get('sector', 'Unknown')}\n\n"
        f"Technical Indicators:\n"
        f"- Price: ${tech_indicators['current_price']}\n"
        f"- RSI(14): {tech_indicators['rsi']} ({tech_indicators['rsi_signal']})\n"
        f"- MACD Crossover: {tech_indicators['macd_crossover']}\n"
        f"- Trend: {tech_indicators['trend']}\n"
        f"- Price vs SMA50: {tech_indicators['price_vs_sma50']}\n"
        f"- Price vs SMA200: {tech_indicators['price_vs_sma200']}\n"
        f"- Bollinger Position: {tech_indicators['bollinger_position']}\n"
        f"- Volatility: {tech_indicators['volatility']} (ATR%: {tech_indicators['atr_percent']}%)\n\n"
        f"Fundamentals:\n"
        f"- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}\n"
        f"- PEG Ratio: {fundamentals.get('peg_ratio', 'N/A')}\n"
        f"- Revenue Growth: {fundamentals.get('revenue_growth', 'N/A')}\n"
        f"- Profit Margin: {fundamentals.get('profit_margin', 'N/A')}\n"
        f"- Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}\n"
    )

    try:
        text = _invoke_llm(MARKET_DATA_SYSTEM_PROMPT, data_context)
        fields = _parse_labeled_response(text)
        technical_summary = fields.get("TECHNICAL", "").strip()
        fundamental_summary = fields.get("FUNDAMENTAL", "").strip()
        if not technical_summary or not fundamental_summary:
            raise ValueError("LLM response missing market summary labels")
    except Exception as e:
        logger.warning("Using fallback market summaries for %s: %s", ticker, e)
        technical_summary, fundamental_summary = _fallback_market_summaries(
            ticker=ticker,
            stock_info=stock_info,
            tech_indicators=tech_indicators,
            fundamentals=fundamentals,
        )

    return MarketDataReport(
        ticker=ticker,
        company_name=stock_info["company_name"],
        technical=technical,
        fundamental=fundamental,
        technical_summary=technical_summary,
        fundamental_summary=fundamental_summary,
    )


# ---------------------------------------------------------------------------
# Agent 2: Sentiment Agent
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = """You are a market sentiment analyst. You are given news headlines and 
analyst ratings for a stock.

Your job:
1. Score each headline as positive (+1), neutral (0), or negative (-1).
2. Compute an average news sentiment score (-1.0 to +1.0).
3. Assess the overall analyst consensus.
4. Produce a 1-2 sentence summary of market sentiment.

Respond in this exact format:
NEWS_SCORE: <float between -1.0 and 1.0>
NEWS_SENTIMENT: <very_positive|positive|neutral|negative|very_negative>
OVERALL_SCORE: <float between -1.0 and 1.0>
SUMMARY: <your summary>

Rules:
- If no headlines are available, NEWS_SCORE is 0.0 and NEWS_SENTIMENT is neutral.
- Base analyst consensus on the buy/hold/sell ratio.
- Never fabricate headlines or data.
"""


def _news_sentiment_label(score: float) -> str:
    if score >= 0.6:
        return "very_positive"
    if score >= 0.2:
        return "positive"
    if score <= -0.6:
        return "very_negative"
    if score <= -0.2:
        return "negative"
    return "neutral"


def _fallback_sentiment_values(sentiment_data: dict, analyst_consensus: str) -> tuple[float, str, float, str]:
    """Estimate sentiment from analyst distribution when LLM scoring is unavailable."""
    buy = sentiment_data["analyst_buy"]
    hold = sentiment_data["analyst_hold"]
    sell = sentiment_data["analyst_sell"]
    total = buy + hold + sell

    analyst_score = ((buy - sell) / total) if total else 0.0
    analyst_score = _clamp_float(analyst_score, -1.0, 1.0)
    news_score = 0.0
    overall_score = round(_clamp_float((analyst_score + news_score) / 2, -1.0, 1.0), 2)
    news_sentiment = _news_sentiment_label(news_score)

    if total:
        summary = (
            f"News sentiment was unavailable, so sentiment is based on analyst "
            f"coverage: {buy} buy, {hold} hold, and {sell} sell ratings "
            f"({analyst_consensus})."
        )
    else:
        summary = "News headlines and analyst coverage are unavailable."

    return news_score, news_sentiment, overall_score, summary


def run_sentiment_agent(ticker: str) -> SentimentReport:
    """Sentiment Agent — scores news headlines and analyst consensus."""
    sentiment_data = get_sentiment_data(ticker)

    # Determine analyst consensus from numbers
    buy = sentiment_data["analyst_buy"]
    hold = sentiment_data["analyst_hold"]
    sell = sentiment_data["analyst_sell"]
    total = buy + hold + sell

    if total == 0:
        analyst_consensus = "unavailable"
    elif buy / total > 0.7:
        analyst_consensus = "strong_buy"
    elif buy / total > 0.5:
        analyst_consensus = "buy"
    elif sell / total > 0.7:
        analyst_consensus = "strong_sell"
    elif sell / total > 0.5:
        analyst_consensus = "sell"
    else:
        analyst_consensus = "hold"

    # LLM scores headlines
    headlines_text = "\n".join(
        f"- {h}" for h in sentiment_data["headlines"]
    ) if sentiment_data["headlines"] else "No recent headlines available."

    context = (
        f"Ticker: {ticker}\n\n"
        f"Analyst Ratings: {buy} Buy, {hold} Hold, {sell} Sell\n\n"
        f"Recent Headlines:\n{headlines_text}"
    )

    try:
        text = _invoke_llm(SENTIMENT_SYSTEM_PROMPT, context)
        fields = _parse_labeled_response(text)

        news_score = _clamp_float(float(fields.get("NEWS_SCORE", "0.0")), -1.0, 1.0)
        news_sentiment = fields.get("NEWS_SENTIMENT", "neutral").strip().lower()
        if news_sentiment not in ("very_positive", "positive", "neutral", "negative", "very_negative"):
            news_sentiment = _news_sentiment_label(news_score)
        overall_score = _clamp_float(float(fields.get("OVERALL_SCORE", str(news_score))), -1.0, 1.0)
        summary = fields.get("SUMMARY", "").strip()
        if not summary:
            raise ValueError("LLM response missing sentiment summary")
    except Exception as e:
        logger.warning("Using fallback sentiment scoring for %s: %s", ticker, e)
        news_score, news_sentiment, overall_score, summary = _fallback_sentiment_values(
            sentiment_data=sentiment_data,
            analyst_consensus=analyst_consensus,
        )

    return SentimentReport(
        ticker=ticker,
        analyst_buy=buy,
        analyst_hold=hold,
        analyst_sell=sell,
        analyst_consensus=analyst_consensus,
        news_score=news_score,
        news_sentiment=news_sentiment,
        headline_count=sentiment_data["headline_count"],
        overall_score=overall_score,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Agent 3: Synthesizer Agent
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM_PROMPT = """You are a senior portfolio manager. A quantitative model has already produced \
a recommendation and confidence score. Your job is to write a clear, concise explanation of WHY the \
recommendation makes sense given the data.

You will receive:
1. The DECISION (recommendation + confidence) already made by the scoring model.
2. The market data, sentiment data, and market conditions that fed into it.
3. The caveats already identified by the model.

YOUR TASK: Write a 2-4 sentence rationale explaining the decision, and a one-line sentiment summary.

Rules:
- Reference specific numbers from the data (RSI, P/E, trend, analyst %).
- Do NOT override or contradict the recommendation or confidence — they are final.
- Do NOT say "you should", "guaranteed", or give personal advice.
- Do NOT claim data is missing if it appears in the input.
- Do NOT repeat the caveats in your rationale.

Respond in this EXACT format:
SENTIMENT: <one line summary of overall market mood>
RATIONALE: <2-4 sentences explaining why the recommendation is justified>
"""


def _score_market_technical(market_data: MarketDataReport | None) -> int:
    if not market_data:
        return 0

    tech = market_data.technical
    score = 0
    if tech.trend == "uptrend":
        score += 1
    elif tech.trend == "downtrend":
        score -= 1

    if tech.macd_crossover == "bullish":
        score += 1
    elif tech.macd_crossover == "bearish":
        score -= 1

    if tech.price_vs_sma200 == "above":
        score += 1
    else:
        score -= 1

    if tech.rsi_signal == "overbought":
        score -= 1
    elif tech.rsi_signal == "oversold":
        score += 1

    return score


def _score_market_fundamental(market_data: MarketDataReport | None) -> int:
    if not market_data:
        return 0

    fund = market_data.fundamental
    score = 0

    if fund.pe_ratio is not None:
        if fund.pe_ratio < 0:
            score -= 2
        elif fund.pe_ratio < 25:
            score += 1
        elif fund.pe_ratio > 35:
            score -= 1

    if fund.peg_ratio is not None:
        if fund.peg_ratio < 1.5:
            score += 1
        elif fund.peg_ratio > 2.5:
            score -= 1

    if fund.revenue_growth is not None:
        if fund.revenue_growth > 0.10:
            score += 1
        elif fund.revenue_growth < 0:
            score -= 1

    if fund.profit_margin is not None:
        if fund.profit_margin > 0.15:
            score += 1
        elif fund.profit_margin < 0.05:
            score -= 1

    if fund.debt_to_equity is not None and fund.debt_to_equity > 2.0:
        score -= 1

    return score


def _score_sentiment(sentiment: SentimentReport | None) -> int:
    if not sentiment:
        return 0

    score = 0
    if sentiment.analyst_consensus in ("strong_buy", "buy"):
        score += 1
    elif sentiment.analyst_consensus in ("strong_sell", "sell"):
        score -= 1

    if sentiment.overall_score >= 0.2:
        score += 1
    elif sentiment.overall_score <= -0.2:
        score -= 1

    return score


def _deduplicate_caveats(caveats: list[str]) -> list[str]:
    """Remove duplicate caveats, including semantically similar ones (e.g. P/E warnings)."""
    seen_keys: set[str] = set()
    result: list[str] = []

    for caveat in caveats:
        # Normalize for comparison: lowercase, strip punctuation differences
        normalized = caveat.lower().strip()

        # Collapse P/E variants into one key
        if "p/e" in normalized or "pe ratio" in normalized or "pe (" in normalized:
            key = "pe_extreme"
        elif "volatility" in normalized and ("high" in normalized or "extreme" in normalized or "atr" in normalized):
            key = "volatility_high"
        elif "debt" in normalized or "leverage" in normalized:
            key = "leverage_high"
        elif "technical" in normalized and "fundamental" in normalized:
            key = "tech_fund_disagree"
        else:
            key = normalized

        if key not in seen_keys:
            seen_keys.add(key)
            result.append(caveat)

    return result


def _fallback_synthesizer_recommendation(
    ticker: str,
    company_name: str,
    market_data: MarketDataReport | None,
    sentiment: SentimentReport | None,
    market_conditions: dict | None = None,
    extra_caveats: list[str] | None = None,
) -> FinalRecommendation:
    """Combine agent signals deterministically when the LLM is unavailable."""
    caveats = list(extra_caveats or [])
    technical_score = _score_market_technical(market_data)
    fundamental_score = _score_market_fundamental(market_data)
    sentiment_score = _score_sentiment(sentiment)
    category_scores = [technical_score, fundamental_score, sentiment_score]

    bullish_categories = sum(score > 0 for score in category_scores)
    bearish_categories = sum(score < 0 for score in category_scores)
    total_score = sum(category_scores)

    if bullish_categories >= 2 and total_score > 0:
        recommendation = "Buy"
    elif bearish_categories >= 2 and total_score < 0:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    confidence = 50 + min(35, abs(total_score) * 7)
    if bullish_categories and bearish_categories:
        confidence = min(confidence, 70)
    if recommendation == "Hold":
        confidence = min(confidence, 65)

    if market_data:
        if market_data.technical.volatility in ("high", "extreme") and recommendation == "Buy":
            confidence -= 15
            caveats.append(f"{market_data.technical.volatility.title()} volatility")
        if market_data.fundamental.pe_ratio is not None and market_data.fundamental.pe_ratio < 0:
            caveats.append("Negative P/E indicates profitability risk")

    market_conditions = market_conditions or {}
    if market_conditions:
        if market_conditions.get("regime") == "risk_off" and recommendation == "Buy":
            confidence -= 10
        if market_conditions.get("risk_level") == "high" and recommendation == "Buy":
            confidence -= 10
        if market_conditions.get("regime") == "risk_off" or market_conditions.get("risk_level") == "high":
            caveats.append(
                f"Broad market conditions are {market_conditions.get('regime')} "
                f"with {market_conditions.get('risk_level')} risk"
            )

    if sentiment:
        total_analysts = sentiment.analyst_buy + sentiment.analyst_hold + sentiment.analyst_sell
        if total_analysts == 0:
            caveats.append("Limited analyst coverage")
            confidence -= 5  # Less data = less certainty
    else:
        caveats.append("Sentiment data unavailable")
        confidence -= 10  # Penalize for incomplete data

    if not market_data:
        confidence -= 10  # Penalize for missing market data

    confidence = _clamp_int(int(confidence), 0, 100)
    sentiment_line = sentiment.summary if sentiment else "Sentiment data unavailable."

    rationale_parts = []
    if market_data:
        rationale_parts.append(
            f"Technical signals are {market_data.technical.trend} with "
            f"{market_data.technical.macd_crossover} MACD crossover and RSI "
            f"{market_data.technical.rsi}."
        )
        rationale_parts.append(
            f"Fundamental signals include P/E {market_data.fundamental.pe_ratio} "
            f"and revenue growth {market_data.fundamental.revenue_growth}."
        )
    if sentiment:
        rationale_parts.append(
            f"Sentiment shows {sentiment.analyst_consensus} analyst consensus "
            f"and overall score {sentiment.overall_score}."
        )
    rationale_parts.append(
        f"The deterministic signal balance is technical {technical_score}, "
        f"fundamental {fundamental_score}, and sentiment {sentiment_score}."
    )

    key_metrics = {}
    if market_data:
        key_metrics["price"] = market_data.technical.current_price
        key_metrics["rsi"] = market_data.technical.rsi
        key_metrics["trend"] = market_data.technical.trend
        key_metrics["pe_ratio"] = market_data.fundamental.pe_ratio
        key_metrics["peg_ratio"] = market_data.fundamental.peg_ratio
    if sentiment:
        total_analysts = sentiment.analyst_buy + sentiment.analyst_hold + sentiment.analyst_sell
        if total_analysts > 0:
            key_metrics["analyst_consensus"] = (
                f"{round(sentiment.analyst_buy / total_analysts * 100)}% Buy"
            )
    if market_data:
        # Compute stock's own return from SMA200 as proxy for ~10mo ago price
        current = market_data.technical.current_price
        sma200 = market_data.technical.sma_200
        if sma200 and sma200 > 0:
            stock_return = round(((current / sma200) - 1) * 100, 2)
            key_metrics["stock_return"] = f"{stock_return}%"
    if market_conditions:
        key_metrics["market_condition"] = market_conditions.get("regime")
        key_metrics["market_risk"] = market_conditions.get("risk_level")

    return FinalRecommendation(
        ticker=ticker,
        company_name=company_name,
        recommendation=recommendation,
        confidence=confidence,
        sentiment=sentiment_line,
        rationale=" ".join(rationale_parts),
        caveats=_deduplicate_caveats(caveats),
        key_metrics=key_metrics,
    )


def run_synthesizer_agent(
    ticker: str,
    company_name: str,
    market_data: MarketDataReport | None,
    sentiment: SentimentReport | None,
    market_conditions: dict | None = None,
    extra_caveats: list[str] | None = None,
) -> FinalRecommendation:
    """Synthesizer Agent — hybrid approach.

    1. Deterministic scoring produces recommendation, confidence, caveats (reproducible).
    2. LLM generates only the natural-language rationale and sentiment summary.
    """
    # ------------------------------------------------------------------
    # STEP 1: Deterministic scoring (always the same for the same data)
    # ------------------------------------------------------------------
    caveats = list(extra_caveats or [])
    technical_score = _score_market_technical(market_data)
    fundamental_score = _score_market_fundamental(market_data)
    sentiment_score = _score_sentiment(sentiment)
    category_scores = [technical_score, fundamental_score, sentiment_score]

    bullish_categories = sum(score > 0 for score in category_scores)
    bearish_categories = sum(score < 0 for score in category_scores)
    total_score = sum(category_scores)

    # Decision thresholds
    if bullish_categories >= 2 and total_score > 0:
        recommendation = "Buy"
    elif bearish_categories >= 2 and total_score < 0:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    # Confidence calculation
    confidence = 50 + min(35, abs(total_score) * 7)
    if bullish_categories and bearish_categories:
        confidence = min(confidence, 70)
    if recommendation == "Hold":
        confidence = min(confidence, 65)

    # Risk adjustments
    if market_data:
        if market_data.technical.volatility in ("high", "extreme") and recommendation == "Buy":
            confidence -= 15
            caveats.append(f"{market_data.technical.volatility.title()} volatility")
        if market_data.fundamental.pe_ratio is not None and market_data.fundamental.pe_ratio < 0:
            caveats.append("Negative P/E indicates profitability risk")

    market_conditions = market_conditions or {}
    if market_conditions:
        if market_conditions.get("regime") == "risk_off" and recommendation == "Buy":
            confidence -= 10
        if market_conditions.get("risk_level") == "high" and recommendation == "Buy":
            confidence -= 10
        if market_conditions.get("regime") == "risk_off" or market_conditions.get("risk_level") == "high":
            caveats.append(
                f"Broad market conditions are {market_conditions.get('regime')} "
                f"with {market_conditions.get('risk_level')} risk"
            )

    if sentiment:
        total_analysts = sentiment.analyst_buy + sentiment.analyst_hold + sentiment.analyst_sell
        if total_analysts == 0:
            caveats.append("Limited analyst coverage")
            confidence -= 5  # Less data = less certainty
    else:
        caveats.append("Sentiment data unavailable")
        confidence -= 10  # Penalize for incomplete data

    if not market_data:
        confidence -= 10  # Penalize for missing market data

    confidence = _clamp_int(int(confidence), 0, 100)

    # ------------------------------------------------------------------
    # STEP 2: Build key metrics (deterministic)
    # ------------------------------------------------------------------
    key_metrics = {}
    if market_data:
        key_metrics["price"] = market_data.technical.current_price
        key_metrics["rsi"] = market_data.technical.rsi
        key_metrics["trend"] = market_data.technical.trend
        key_metrics["pe_ratio"] = market_data.fundamental.pe_ratio
        key_metrics["peg_ratio"] = market_data.fundamental.peg_ratio
    if sentiment:
        total_analysts = sentiment.analyst_buy + sentiment.analyst_hold + sentiment.analyst_sell
        if total_analysts > 0:
            key_metrics["analyst_consensus"] = (
                f"{round(sentiment.analyst_buy / total_analysts * 100)}% Buy"
            )
    if market_data:
        current = market_data.technical.current_price
        sma200 = market_data.technical.sma_200
        if sma200 and sma200 > 0:
            stock_return = round(((current / sma200) - 1) * 100, 2)
            key_metrics["stock_return"] = f"{stock_return}%"
    if market_conditions:
        key_metrics["market_condition"] = market_conditions.get("regime")
        key_metrics["market_risk"] = market_conditions.get("risk_level")

    # ------------------------------------------------------------------
    # STEP 3: LLM generates narrative only (non-critical path)
    # ------------------------------------------------------------------
    # Build context for LLM — includes the decision so it can explain it
    parts = [
        f"Ticker: {ticker} ({company_name})\n",
        f"MODEL DECISION: {recommendation} with {confidence}% confidence\n",
        f"SIGNAL SCORES: Technical={technical_score}, Fundamental={fundamental_score}, Sentiment={sentiment_score}\n",
    ]

    if market_data:
        tech = market_data.technical
        fund = market_data.fundamental
        parts.append(
            f"\nTECHNICAL DATA:\n"
            f"- Price: ${tech.current_price}, RSI: {tech.rsi} ({tech.rsi_signal})\n"
            f"- MACD: {tech.macd_crossover}, Trend: {tech.trend}\n"
            f"- Price vs SMA50: {tech.price_vs_sma50}, vs SMA200: {tech.price_vs_sma200}\n"
            f"- Volatility: {tech.volatility} (ATR: {tech.atr_percent}%)\n"
            f"\nFUNDAMENTAL DATA:\n"
            f"- P/E: {fund.pe_ratio}, PEG: {fund.peg_ratio}\n"
            f"- Revenue Growth: {fund.revenue_growth}, Margin: {fund.profit_margin}\n"
            f"- Debt/Equity: {fund.debt_to_equity}\n"
        )

    if sentiment:
        parts.append(
            f"\nSENTIMENT DATA:\n"
            f"- Analysts: {sentiment.analyst_buy} Buy / {sentiment.analyst_hold} Hold / {sentiment.analyst_sell} Sell\n"
            f"- Consensus: {sentiment.analyst_consensus}\n"
            f"- News: {sentiment.news_sentiment} (score: {sentiment.news_score})\n"
        )

    if market_conditions:
        parts.append(
            f"\nMARKET CONDITIONS: {market_conditions.get('regime')} regime, "
            f"{market_conditions.get('risk_level')} risk, "
            f"SPY return {market_conditions.get('return_percent')}%\n"
        )

    if caveats:
        parts.append(f"\nCAVEATS: {', '.join(caveats)}\n")

    context = "\n".join(parts)

    # Attempt LLM narrative — fallback to deterministic text if it fails
    sentiment_line = sentiment.summary if sentiment else "Sentiment data unavailable."
    rationale_parts = []
    if market_data:
        rationale_parts.append(
            f"Technical signals are {market_data.technical.trend} with "
            f"{market_data.technical.macd_crossover} MACD crossover and RSI "
            f"{market_data.technical.rsi}."
        )
        rationale_parts.append(
            f"Fundamental signals include P/E {market_data.fundamental.pe_ratio} "
            f"and revenue growth {market_data.fundamental.revenue_growth}."
        )
    if sentiment:
        rationale_parts.append(
            f"Sentiment shows {sentiment.analyst_consensus} analyst consensus "
            f"and overall score {sentiment.overall_score}."
        )
    rationale_parts.append(
        f"Signal balance: technical {technical_score}, "
        f"fundamental {fundamental_score}, sentiment {sentiment_score}."
    )
    fallback_rationale = " ".join(rationale_parts)

    try:
        text = _invoke_llm(SYNTHESIZER_SYSTEM_PROMPT, context)
        fields = _parse_labeled_response(text)

        llm_sentiment = fields.get("SENTIMENT", "").strip()
        llm_rationale = fields.get("RATIONALE", "").strip()

        if llm_rationale:
            rationale = llm_rationale
        else:
            rationale = fallback_rationale

        if llm_sentiment:
            sentiment_line = llm_sentiment
    except Exception as e:
        logger.warning("LLM narrative failed for %s (using fallback text): %s", ticker, e)
        rationale = fallback_rationale

    return FinalRecommendation(
        ticker=ticker,
        company_name=company_name,
        recommendation=recommendation,
        confidence=confidence,
        sentiment=sentiment_line,
        rationale=rationale,
        caveats=_deduplicate_caveats(caveats),
        key_metrics=key_metrics,
    )
