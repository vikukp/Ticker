"""Streamlit UI — Ticker Research Platform frontend."""

import streamlit as st

from app.graph import run_research
from app.models import FinalRecommendation
from app.pdf_report import generate_pdf


@st.cache_data(ttl=300, show_spinner=False)
def _cached_research(ticker: str):
    """Cache research results for 5 minutes to avoid redundant API calls."""
    return run_research(ticker)


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ticker Research Platform",
    page_icon="📈",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📈 Ticker Research Platform")
st.caption("Powered by AI")
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

# Session state for two-phase form locking
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "pending_ticker" not in st.session_state:
    st.session_state.pending_ticker = ""
if "result" not in st.session_state:
    st.session_state.result = None

is_busy = st.session_state.analyzing

with st.form("ticker_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Enter a stock ticker symbol",
            placeholder="e.g., AAPL, MSFT, TSLA",
            max_chars=10,
            disabled=is_busy,
        )
    with col2:
        st.write("")  # spacer
        st.write("")
        analyze_btn = st.form_submit_button(
            "🔍 Analyzing..." if is_busy else "🔍 Analyze",
            type="secondary", use_container_width=True,
            disabled=is_busy,
        )

# Phase 1: User clicked Analyze — lock the form and rerun
if analyze_btn and ticker_input and not is_busy:
    st.session_state.analyzing = True
    st.session_state.pending_ticker = ticker_input.strip().upper()
    st.session_state.result = None
    st.rerun()

# Phase 2: Form is locked — run the analysis
if st.session_state.analyzing and st.session_state.pending_ticker:
    with st.spinner("Running multi-agent research pipeline..."):
        st.session_state.result = _cached_research(st.session_state.pending_ticker)
    st.session_state.analyzing = False
    st.rerun()

# ---------------------------------------------------------------------------
# Analysis Results
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is not None:
    if isinstance(result, str):
        # Error case
        st.error(f"❌ {result}")
    elif isinstance(result, FinalRecommendation):
        rec = result

        # --- Recommendation Header ---
        st.divider()

        # Company name + ticker
        st.subheader(f"{rec.company_name} ({rec.ticker})")

        # Recommendation badge
        color_map = {"Buy": "green", "Sell": "red", "Hold": "orange"}
        emoji_map = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}
        rec_color = color_map.get(rec.recommendation, "gray")
        rec_emoji = emoji_map.get(rec.recommendation, "⚪")

        col_rec, col_conf = st.columns(2)
        with col_rec:
            st.metric("Recommendation", f"{rec_emoji} {rec.recommendation}")
        with col_conf:
            st.metric("Confidence", f"{rec.confidence}%")

        # --- Sentiment ---
        st.markdown(f"**Sentiment:** {rec.sentiment}")
        st.divider()

        # --- Key Metrics ---
        if rec.key_metrics:
            st.subheader("Key Metrics")
            # Friendly display names for metric keys
            display_names = {
                "price": "Price",
                "rsi": "RSI",
                "trend": "Trend",
                "pe_ratio": "PE Ratio",
                "peg_ratio": "PEG Ratio",
                "analyst_consensus": "Analyst Consensus",
                "market_condition": "Market Condition",
                "market_risk": "Market Risk",
                "stock_return": "Stock Return",
            }
            metric_cols = st.columns(min(len(rec.key_metrics), 4))
            for i, (key, value) in enumerate(rec.key_metrics.items()):
                with metric_cols[i % len(metric_cols)]:
                    label = display_names.get(key, key.replace("_", " ").title())
                    st.metric(label, value)
            st.divider()

        # --- Rationale ---
        st.subheader("Rationale")
        st.write(rec.rationale)

        # --- Caveats ---
        if rec.caveats:
            st.subheader("⚠️ Risk Caveats")
            for caveat in rec.caveats:
                st.warning(caveat)

        # --- PDF Download ---
        st.divider()
        pdf_bytes = generate_pdf(rec)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{rec.ticker}_research_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # --- Disclaimer ---
        st.caption(
            "⚠️ This report is AI-generated for educational purposes only. "
            "It does not constitute financial advice."
        )
    else:
        st.error("Unexpected result. Please try again.")

elif analyze_btn and not ticker_input:
    st.warning("Please enter a ticker symbol.")
