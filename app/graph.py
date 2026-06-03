"""LangGraph orchestration — deterministic orchestrator with parallel agent dispatch.

Graph Structure:
    START → validate_input → [market_data_agent ║ sentiment_agent] → 
    collect_results → synthesizer_agent → validate_output → END

The orchestrator is deterministic Python logic. LLM is used only inside agents.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Annotated, Any

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.agents import run_market_data_agent, run_sentiment_agent, run_synthesizer_agent
from app.guardrails import (
    detect_anomalies,
    sanitize_output,
    validate_output,
    validate_ticker,
)
from app.models import FinalRecommendation, MarketDataReport, SentimentReport

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Graph State (TypedDict for LangGraph)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State passed through the LangGraph nodes."""

    ticker: str
    company_name: str
    is_valid: bool
    error: str
    market_data: MarketDataReport | None
    sentiment: SentimentReport | None
    market_conditions: dict
    anomalies: list[str]
    caveats: list[str]
    recommendation: FinalRecommendation | None
    retry_count: int


# ---------------------------------------------------------------------------
# Node: Input Validation
# ---------------------------------------------------------------------------


def validate_input_node(state: GraphState) -> GraphState:
    """Orchestrator Step 1: Validate the ticker input."""
    ticker = state.get("ticker", "")
    is_valid, cleaned, error = validate_ticker(ticker)

    if not is_valid:
        return {**state, "is_valid": False, "error": error}

    return {**state, "ticker": cleaned, "is_valid": True, "error": ""}


# ---------------------------------------------------------------------------
# Node: Parallel Agent Dispatch (Market Data + Sentiment)
# ---------------------------------------------------------------------------


def research_agents_node(state: GraphState) -> GraphState:
    """Orchestrator Step 2: Dispatch Market Data + Sentiment agents in parallel."""
    ticker = state["ticker"]
    market_data = None
    sentiment = None
    market_conditions = {}
    caveats = list(state.get("caveats", []))

    # Run agents/data checks in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        market_future = executor.submit(_safe_market_data, ticker)
        sentiment_future = executor.submit(_safe_sentiment, ticker)
        market_conditions_future = executor.submit(_safe_market_conditions)

        market_data, market_error = market_future.result()
        sentiment, sentiment_error = sentiment_future.result()
        market_conditions, market_conditions_error = market_conditions_future.result()

    # Handle failures gracefully
    if market_error:
        caveats.append(f"Technical/fundamental data limited: {market_error}")
    if sentiment_error:
        caveats.append(f"Sentiment data limited: {sentiment_error}")
    if market_conditions_error:
        caveats.append(f"Market condition data limited: {market_conditions_error}")
    elif market_conditions.get("risk_level") in ("moderate", "high"):
        caveats.append(
            f"Broad market conditions are {market_conditions['regime']} "
            f"with {market_conditions['risk_level']} risk"
        )

    # Both failed = cannot proceed
    if market_data is None and sentiment is None:
        return {
            **state,
            "error": "Unable to fetch any market data. Please try again later.",
            "is_valid": False,
        }

    company_name = ""
    if market_data:
        company_name = market_data.company_name
    else:
        # Try to get company name from basic info
        try:
            from app.tools import get_stock_info
            info = get_stock_info(ticker)
            company_name = info.get("company_name", ticker)
        except Exception:
            company_name = ticker

    return {
        **state,
        "market_data": market_data,
        "sentiment": sentiment,
        "market_conditions": market_conditions,
        "company_name": company_name,
        "caveats": caveats,
    }


def _safe_market_data(ticker: str) -> tuple[MarketDataReport | None, str]:
    """Run market data agent with error handling."""
    try:
        logger.debug(f"Running Market Data Agent for {ticker}")
        result = run_market_data_agent(ticker)
        logger.debug(f"Market Data Agent completed for {ticker}")
        return result, ""
    except Exception as e:
        logger.error(f"Market Data Agent failed: {str(e)}", exc_info=True)
        return None, str(e)[:200]


def _safe_sentiment(ticker: str) -> tuple[SentimentReport | None, str]:
    """Run sentiment agent with error handling."""
    try:
        logger.debug(f"Running Sentiment Agent for {ticker}")
        result = run_sentiment_agent(ticker)
        logger.debug(f"Sentiment Agent completed for {ticker}")
        return result, ""
    except Exception as e:
        logger.error(f"Sentiment Agent failed: {str(e)}", exc_info=True)
        return None, str(e)[:200]


def _safe_market_conditions() -> tuple[dict, str]:
    """Fetch broad market conditions with error handling."""
    try:
        from app.tools import get_market_conditions

        logger.debug("Fetching broad market conditions")
        result = get_market_conditions()
        logger.debug("Broad market conditions fetched")
        return result, ""
    except Exception as e:
        logger.error(f"Market conditions fetch failed: {str(e)}", exc_info=True)
        return {}, str(e)[:200]


# ---------------------------------------------------------------------------
# Node: Collect Results + Anomaly Detection
# ---------------------------------------------------------------------------


def collect_results_node(state: GraphState) -> GraphState:
    """Orchestrator Step 3: Detect anomalies and prepare for synthesis."""
    market_data = state.get("market_data")
    sentiment = state.get("sentiment")

    anomalies = detect_anomalies(market_data, sentiment)
    caveats = list(state.get("caveats", []))

    # Convert anomalies into caveats for the synthesizer
    caveats.extend(anomalies)

    return {**state, "anomalies": anomalies, "caveats": caveats}


# ---------------------------------------------------------------------------
# Node: Synthesizer Agent
# ---------------------------------------------------------------------------


def synthesizer_node(state: GraphState) -> GraphState:
    """Orchestrator Step 4: Run synthesizer to produce recommendation."""
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    market_data = state.get("market_data")
    sentiment = state.get("sentiment")
    market_conditions = state.get("market_conditions", {})
    caveats = state.get("caveats", [])

    try:
        recommendation = run_synthesizer_agent(
            ticker=ticker,
            company_name=company_name,
            market_data=market_data,
            sentiment=sentiment,
            market_conditions=market_conditions,
            extra_caveats=caveats,
        )
    except Exception as e:
        return {
            **state,
            "error": f"Synthesis failed: {str(e)[:100]}",
            "is_valid": False,
        }

    return {**state, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# Node: Output Validation
# ---------------------------------------------------------------------------


def validate_output_node(state: GraphState) -> GraphState:
    """Orchestrator Step 5: Validate and sanitize the final output."""
    recommendation = state.get("recommendation")

    if recommendation is None:
        return {**state, "error": "No recommendation generated.", "is_valid": False}

    is_valid, issues = validate_output(recommendation)

    if not is_valid:
        # Sanitize and accept the result
        recommendation = sanitize_output(recommendation)

    return {**state, "recommendation": recommendation, "is_valid": True}


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------


def route_after_validation(state: GraphState) -> str:
    """Route after input validation."""
    if state.get("is_valid"):
        return "research"
    return "end"


def route_after_output(state: GraphState) -> str:
    """Route after output validation — retry or finish."""
    if state.get("recommendation") is None and state.get("retry_count", 0) < 1:
        return "retry"
    return "end"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the research orchestration graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("research_agents", research_agents_node)
    graph.add_node("collect_results", collect_results_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("validate_output", validate_output_node)

    # Add edges
    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"research": "research_agents", "end": END},
    )
    graph.add_edge("research_agents", "collect_results")
    graph.add_edge("collect_results", "synthesizer")
    graph.add_edge("synthesizer", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_research(ticker: str) -> FinalRecommendation | str:
    """Run the full research pipeline for a ticker.

    Returns:
        FinalRecommendation on success, or error string on failure.
    """
    logger.info(f"Starting research pipeline for {ticker}")
    try:
        graph = build_graph()
        result = graph.invoke(
            {"ticker": ticker, "retry_count": 0, "caveats": []},
            {"recursion_limit": 25},
        )

        logger.debug(f"Graph execution result keys: {result.keys()}")
        
        if result.get("error") and not result.get("recommendation"):
            error_msg = result["error"]
            logger.warning(f"Research failed with error: {error_msg}")
            return error_msg

        recommendation = result.get("recommendation")
        if recommendation:
            logger.info(f"Research completed successfully for {ticker}")
            return recommendation
        else:
            logger.warning(f"No recommendation generated for {ticker}")
            return "No recommendation could be generated. Please check Ollama is running and try again."
    except Exception as e:
        logger.error(f"Research pipeline failed: {str(e)}", exc_info=True)
        return f"Research pipeline error: {str(e)}"
