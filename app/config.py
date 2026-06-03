"""Configuration — loads environment variables and defines constants."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM Configuration (supports OpenAI, Azure OpenAI, and local Ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai", "azure", or "ollama"
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Azure OpenAI (legacy, kept for backward compatibility)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    # Ollama (local LLM)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")  # Change to your Ollama model
    
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 2

    # Data
    PRICE_HISTORY_PERIOD: str = "6mo"
    REQUEST_TIMEOUT: int = 15  # seconds

    # Guardrails
    MAX_TICKER_LENGTH: int = 10
    MIN_HISTORY_DAYS: int = 30
    CONFIDENCE_MIN: int = 0
    CONFIDENCE_MAX: int = 100

    PROHIBITED_PHRASES: list[str] = [
        "you should",
        "guaranteed",
        "risk-free",
        "definitely will",
        "100% certain",
        "financial advice",
        "i recommend you",
    ]


config = Config()
