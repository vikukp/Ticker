"""Consistency evals — logic alignment and signal-recommendation coherence."""

import pytest

import app.agents as agents
from app.agents import run_synthesizer_agent


# ---------------------------------------------------------------------------
# Signal-Recommendation Alignment
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify that recommendations are logically consistent with inputs."""

    @pytest.mark.integration
    def test_all_bearish_not_buy(self, bearish_market_data, bearish_sentiment):
        """When all signals are bearish, recommendation must NOT be Buy."""
        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bearish_market_data,
            sentiment=bearish_sentiment,
        )
        assert rec.recommendation != "Buy", (
            f"Got Buy with all-bearish inputs. Confidence: {rec.confidence}"
        )

    @pytest.mark.integration
    def test_all_bullish_not_sell(self, bullish_market_data, bullish_sentiment):
        """When all signals are bullish, recommendation must NOT be Sell."""
        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bullish_market_data,
            sentiment=bullish_sentiment,
        )
        assert rec.recommendation != "Sell", (
            f"Got Sell with all-bullish inputs. Confidence: {rec.confidence}"
        )

    @pytest.mark.integration
    def test_conflicting_signals_lower_confidence(
        self, bullish_market_data, bearish_sentiment
    ):
        """When signals conflict, confidence should be moderate (< 70)."""
        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bullish_market_data,
            sentiment=bearish_sentiment,
        )
        assert rec.confidence <= 70, (
            f"Confidence {rec.confidence} too high for conflicting signals"
        )

    @pytest.mark.integration
    def test_confidence_always_in_bounds(self, bullish_market_data, bullish_sentiment):
        """Confidence must always be between 0 and 100."""
        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bullish_market_data,
            sentiment=bullish_sentiment,
        )
        assert 0 <= rec.confidence <= 100

    @pytest.mark.integration
    def test_recommendation_is_valid_enum(self, bullish_market_data, bullish_sentiment):
        """Recommendation must be exactly Buy, Sell, or Hold."""
        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bullish_market_data,
            sentiment=bullish_sentiment,
        )
        assert rec.recommendation in ("Buy", "Sell", "Hold")


class TestAgentFallbacks:
    """Regression coverage for deterministic behavior around the LLM boundary."""

    def test_strong_sell_consensus_reachable(self, monkeypatch):
        """A sell ratio above 70% should be classified as strong_sell."""
        monkeypatch.setattr(
            agents,
            "get_sentiment_data",
            lambda _: {
                "headlines": [],
                "headline_count": 0,
                "analyst_buy": 1,
                "analyst_hold": 2,
                "analyst_sell": 10,
            },
        )
        monkeypatch.setattr(
            agents,
            "_invoke_llm",
            lambda *_: (
                "NEWS_SCORE: -0.4\n"
                "NEWS_SENTIMENT: negative\n"
                "OVERALL_SCORE: -0.7\n"
                "SUMMARY: Negative analyst tilt."
            ),
        )

        report = agents.run_sentiment_agent("TEST")

        assert report.analyst_consensus == "strong_sell"

    def test_synthesizer_falls_back_when_llm_unavailable(
        self, monkeypatch, bullish_market_data, bullish_sentiment
    ):
        """Synthesizer should still return a valid recommendation if LLM fails."""
        monkeypatch.setattr(
            agents,
            "_invoke_llm",
            lambda *_: (_ for _ in ()).throw(ConnectionError("Ollama unavailable")),
        )

        rec = run_synthesizer_agent(
            ticker="TEST",
            company_name="Test Corp",
            market_data=bullish_market_data,
            sentiment=bullish_sentiment,
        )

        assert rec.recommendation in ("Buy", "Sell", "Hold")
        assert rec.recommendation != "Sell"
        assert rec.rationale
