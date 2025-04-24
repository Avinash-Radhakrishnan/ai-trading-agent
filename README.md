AI Trading Agent for Webull
Overview
This project develops an AI trading agent for Webull to trade U.S. stocks using a 50-day/200-day SMA crossover strategy, with dynamic strategy generation via an LLM and RAG. The RAG pipeline retrieves information from a knowledge base of technical indicators and trading strategies to generate or refine trading rules.
Features

Trading Strategy: Initial 50-day/200-day SMA crossover, extensible via LLM+RAG.
LLM+RAG Pipeline:
Indexes multiple documents (e.g., indicators, strategies) using FAISS.
Generates detailed trading strategies with entry/exit rules, risk management, and suitability.
Saves FAISS index for efficient reuse.
Uses updated LangChain libraries (langchain-community, langchain-huggingface) to resolve deprecation warnings.
Configures max_new_tokens for stable LLM text generation.


Supports Webull’s OpenAPI for trading and market data.
Configurable via config.yaml for strategy and LLM parameters.

Setup

Clone Repository:git clone <your-repo-url>
cd ai-trading-agent


Create Virtual Environment:python -m venv trading_env
.\trading_env\Scripts\activate


Install Dependencies:pip install -r requirements.txt


Configure Environment:
Copy config/.env.example to config/.env.
Add Webull API keys:WEBULL_APP_KEY=your_app_key
WEBULL_APP_SECRET=your_app_secret




Run:python main.py



Structure

config/: API keys and configuration.
data/: Knowledge base for RAG (indicators, strategies).
src/: Core logic (data, strategy, LLM, execution, monitoring).
tests/: Unit tests.

Notes

On Windows, ignore or suppress HuggingFace symlink warnings by setting HF_HUB_DISABLE_SYMLINKS_WARNING=true.
Ensure langchain-community and langchain-huggingface are installed for updated imports.

Next Steps

Fetch historical data (Phase 2).
Implement SMA crossover strategy.
Enhance LLM+RAG with backtesting results.

