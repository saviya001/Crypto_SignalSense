import os
import json
import requests
from typing import Dict, Any
from agents.protocol import AgentMessage, AgentRole, MessageType
from utils.data_fetch import fetch_ohlc_data, fetch_coin_price
from utils.indicators import calculate_technical_summary

class TechnicalAgent:
    """
    Worker Agent 2: Technical Chart & Candlestick Analyst.
    Implements Tool-Use Pattern (fetches OHLC data & computes math indicators)
    and uses Groq Llama 3.1 8B for fast technical interpretation.
    """
    def __init__(self, api_key: str = None):
        self.role = AgentRole.TECHNICAL_AGENT
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def process_message(self, message: AgentMessage) -> AgentMessage:
        symbol = message.payload.get("symbol", "BTC").upper()
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Groq API Key is mandatory for TechnicalAgent chart interpretation.")

        # Tool Use 1: Fetch real OHLC candle data & live price
        price_info = fetch_coin_price(symbol)
        df_ohlc = fetch_ohlc_data(symbol)
        
        # Tool Use 2: Compute technical indicators (RSI, MACD, SMA, Support/Resistance)
        tech_summary = calculate_technical_summary(df_ohlc)
        tech_summary["price_info"] = price_info

        # Model Selection: Fast Groq Llama 3.1 8B model for technical interpretation
        interpretation = self._interpret_with_groq(symbol, tech_summary)

        payload = {
            "symbol": symbol,
            "metrics": tech_summary,
            "interpretation": interpretation
        }

        return AgentMessage(
            sender=self.role,
            receiver=AgentRole.SIGNAL_AGENT,
            message_type=MessageType.RESPONSE,
            payload=payload
        )

    def _interpret_with_groq(self, symbol: str, metrics: Dict[str, Any]) -> str:
        """Calls Groq API for rapid technical summary."""
        if not self.api_key:
            rsi = metrics.get("rsi", 50)
            bias = metrics.get("trend_bias", "BULLISH")
            price = metrics.get("current_price", 0.0)
            return f"{symbol} trades at ${price:,.2f} with RSI of {rsi}. Overall trend bias is {bias}."

        prompt = f"""Summarize technical indicator findings for {symbol}:
Price: ${metrics.get('current_price')}
RSI: {metrics.get('rsi')}
Trend Bias: {metrics.get('trend_bias')}
Support: ${metrics.get('support')}
Resistance: ${metrics.get('resistance')}

Provide a concise 2-sentence technical chart summary."""

        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 120
            }
            resp = requests.post(url, headers=headers, json=body, timeout=8)
            if resp.status_code == 200:
                res_data = resp.json()
                return res_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq API call error in TechnicalAgent: {e}")

        return f"{symbol} price shows steady technical structure with RSI at {metrics.get('rsi')} and support holding at ${metrics.get('support')}."
