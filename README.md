# Ticker Research Platform

AI-powered multi-agent stock research tool that produces Buy/Sell/Hold recommendations with confidence scores, rationale, and risk caveats.

## Architecture

```
User Input → Input Validation → [Market Data Agent ║ Sentiment Agent ║ Market Conditions] → Anomaly Detection → Synthesizer Agent → Output Validation → Result
```

- **Orchestrator**: LangGraph (deterministic Python — no LLM in control flow)
- **Agents**: Market Data, Sentiment, Synthesizer (3 agents, parallel execution)
- **Hybrid Scoring**: Deterministic math for recommendation/confidence, LLM for narrative only
- **Frontend**: Streamlit with caching and PDF export

## Agents

| Agent | Role | LLM Usage |
|-------|------|-----------|
| Market Data | Fetches price history, computes 18 technical indicators + fundamentals | Summarizes data into narrative |
| Sentiment | Fetches analyst ratings + news headlines, scores sentiment | Scores news headlines |
| Synthesizer | Combines all signals → Buy/Sell/Hold + confidence + rationale | Generates explanation text only |

## Key Design Decisions

1. **Deterministic scoring** — Recommendation and confidence are computed via rule-based scoring (reproducible across systems)
2. **LLM for narrative only** — The LLM explains the decision but doesn't make it
3. **Parallel execution** — Market Data + Sentiment + Market Conditions run concurrently
4. **Graceful degradation** — If any agent fails, pipeline continues with caveats
5. **Multi-layer guardrails** — Input validation → anomaly detection → output validation → sanitization

## Guardrails

- **Input**: Regex validation, ticker existence check, injection prevention
- **Anomaly Detection**: Extreme RSI, negative P/E, high leverage, extreme volatility
- **Output**: Confidence bounds (0-100), prohibited phrases, rationale length check
- **Sanitization**: Strips prohibited financial advice language

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-key-here
   MODEL_NAME=gpt-4o-mini
   ```

3. Run:
   ```bash
   python run.py
   ```

   The app opens at http://localhost:8501

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `azure`, or `ollama` |
| `MODEL_NAME` | `gpt-4o-mini` | Model name for OpenAI/Azure |
| `TEMPERATURE` | `0.1` | LLM temperature |
| `OLLAMA_MODEL` | `mistral` | Model for local Ollama |

## Testing

```bash
pytest
```

- `test_guardrails.py` — 12 input validation cases (including injection attacks)
- `test_correctness.py` — Output schema and bounds validation
- `test_consistency.py` — Reproducibility checks

## Tech Stack

- **LangGraph** — Orchestration
- **LangChain + OpenAI** — LLM integration
- **yfinance** — Market data
- **ta** — Technical indicators (RSI, MACD, Bollinger, ATR)
- **Pydantic** — Data contracts
- **Streamlit** — Frontend
- **fpdf2** — PDF report generation
