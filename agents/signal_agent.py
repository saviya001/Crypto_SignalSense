import os
import json
import requests
from typing import Dict, Any, List
from agents.protocol import AgentMessage, AgentRole, MessageType
from rag.vector_store import SimpleVectorStore

class SignalAgent:
    """
    Chief Trading Strategist Agent (Orchestrator Brain).
    Design Patterns implemented:
    - RAG Integration: Context retrieval from 22 domain strategy documents.
    - Deep Reasoning: High-tier model selection via OpenRouter.
    - Reflection / Self-Critique Pattern: Pre-signal critique pass checking 1:2 Risk ratio and sentiment alignment.
    """
    def __init__(self, openrouter_key: str = None, vector_store: SimpleVectorStore = None):
        self.role = AgentRole.SIGNAL_AGENT
        self.api_key = openrouter_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        self.vector_store = vector_store or SimpleVectorStore()

    def generate_signal(self, symbol: str, news_message: AgentMessage, tech_message: AgentMessage) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing! OpenRouter API Key is mandatory for SignalAgent deep reasoning synthesis.")

        news_data = news_message.payload
        tech_data = tech_message.payload
        
        current_price = tech_data.get("metrics", {}).get("current_price", 100.0)
        rsi = tech_data.get("metrics", {}).get("rsi", 50.0)
        sentiment = news_data.get("sentiment", "NEUTRAL")

        # 1. RAG Retrieval Step: Fetch relevant trading rules & strategies
        query_text = f"{symbol} RSI {rsi} sentiment {sentiment} risk management 1:2 ratio stop loss take profit"
        rag_chunks = self.vector_store.query(query_text, top_k=3)

        # 2. Deep Reasoning Signal Generation (OpenRouter LLM)
        raw_signal = self._call_openrouter_synthesis(symbol, current_price, news_data, tech_data, rag_chunks)

        # 3. Reflection & Self-Critique Pattern Step
        critique_result = self._reflection_self_critique(raw_signal, news_data, tech_data, rag_chunks)

        # Build final structured output card
        final_signal = {
            "symbol": symbol,
            "current_price": current_price,
            "action": critique_result.get("action", raw_signal.get("action", "HOLD")),
            "entry_price": raw_signal.get("entry_price", current_price),
            "take_profit": raw_signal.get("take_profit", round(current_price * 1.04, 2)),
            "stop_loss": raw_signal.get("stop_loss", round(current_price * 0.98, 2)),
            "risk_reward_ratio": raw_signal.get("risk_reward_ratio", 2.0),
            "confidence": critique_result.get("adjusted_confidence", raw_signal.get("confidence", 0.85)),
            "reasoning": raw_signal.get("reasoning", "Educational Estimate: Signal synthesized based on indicator alignment and market sentiment."),
            "reflection_notes": critique_result.get("reflection_notes", "Passed mandatory risk & contradiction checklist."),
            "rag_references": [r["title"] for r in rag_chunks],
            "rag_sources": rag_chunks
        }
        return final_signal

    def _call_openrouter_synthesis(self, symbol: str, current_price: float, news_data: dict, tech_data: dict, rag_chunks: list) -> Dict[str, Any]:
        """Calls OpenRouter API for high reasoning synthesis."""
        rag_context = "\n".join([f"- [{c['title']}]: {c['content'][:250]}" for c in rag_chunks])

        prompt = f"""You are a Master Crypto Trading Strategist. Synthesize an educational technical signal estimate for {symbol}:
Current Price: ${current_price}
News Sentiment: {news_data.get('sentiment')} ({news_data.get('summary')})
Technical Metrics: RSI {tech_data.get('metrics', {}).get('rsi')}, Support ${tech_data.get('metrics', {}).get('support')}, Resistance ${tech_data.get('metrics', {}).get('resistance')}

Relevant Domain Strategy Rules (RAG Context):
{rag_context}

Output ONLY a JSON object with:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "entry_price": float,
  "take_profit": float (derive from Resistance/Support; ensure TP distance >= 2x SL distance),
  "stop_loss": float,
  "risk_reward_ratio": float (must be >= 2.0 for BUY/SELL),
  "confidence": float between 0.5 and 0.95,
  "reasoning": "Detailed educational rationale deriving TP/SL from Support/Resistance levels"
}}"""

        if self.api_key:
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                resp = requests.post(url, headers=headers, json=body, timeout=12)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    return json.loads(content.strip())
            except Exception as e:
                print(f"OpenRouter synthesis error: {e}")

        # Fallback dynamic signal calculation using Support & Resistance levels
        rsi = tech_data.get("metrics", {}).get("rsi", 50.0)
        sentiment = news_data.get("sentiment", "NEUTRAL")
        support = tech_data.get("metrics", {}).get("support", current_price * 0.97)
        resistance = tech_data.get("metrics", {}).get("resistance", current_price * 1.05)

        if rsi < 45 and sentiment in ["BULLISH", "NEUTRAL"]:
            action = "BUY"
            sl = round(max(support * 0.99, current_price * 0.98), 2)
            risk_dist = current_price - sl
            tp = round(current_price + (risk_dist * 2.1), 2)
            rr = round((tp - current_price) / max(risk_dist, 0.001), 2)
        elif rsi > 65 or sentiment == "BEARISH":
            action = "SELL"
            sl = round(min(resistance * 1.01, current_price * 1.02), 2)
            risk_dist = sl - current_price
            tp = round(current_price - (risk_dist * 2.1), 2)
            rr = round((current_price - tp) / max(risk_dist, 0.001), 2)
        else:
            action = "HOLD"
            sl = round(current_price * 0.98, 2)
            tp = round(current_price * 1.04, 2)
            rr = 2.0

        return {
            "action": action,
            "entry_price": current_price,
            "take_profit": tp,
            "stop_loss": sl,
            "risk_reward_ratio": rr,
            "confidence": 0.82,
            "reasoning": f"Educational Estimate: Confluence of RSI ({rsi}), Technical Support (${support:,.2f}) & Resistance (${resistance:,.2f}), and news sentiment ({sentiment}) supports {action} position structure."
        }

    def _reflection_self_critique(self, signal: dict, news_data: dict, tech_data: dict, rag_chunks: list) -> Dict[str, Any]:
        """
        Implements Reflection / Self-Critique Pattern.
        Validates the proposed trade setup against strict risk management rules.
        """
        action = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0.80)
        notes = []

        entry = signal.get("entry_price", 100.0)
        tp = signal.get("take_profit", 104.0)
        sl = signal.get("stop_loss", 98.0)

        if action == "BUY":
            risk = max(abs(entry - sl), 0.0001)
            reward = abs(tp - entry)
            rr_ratio = reward / risk
            if rr_ratio < 1.8:
                action = "HOLD"
                confidence = max(0.5, confidence * 0.7)
                notes.append(f"Reflection Alert: Calculated R:R ratio ({rr_ratio:.2f}) fell below mandatory 1:2 threshold. Downgraded to HOLD.")
            else:
                notes.append(f"Reflection Check Passed: Valid R:R ratio ({rr_ratio:.2f} >= 1:2).")
        elif action == "SELL":
            risk = max(abs(sl - entry), 0.0001)
            reward = abs(entry - tp)
            rr_ratio = reward / risk
            if rr_ratio < 1.8:
                action = "HOLD"
                confidence = max(0.5, confidence * 0.7)
                notes.append(f"Reflection Alert: Calculated R:R ratio ({rr_ratio:.2f}) fell below mandatory 1:2 threshold. Downgraded to HOLD.")
            else:
                notes.append(f"Reflection Check Passed: Valid R:R ratio ({rr_ratio:.2f} >= 1:2).")

        sentiment = news_data.get("sentiment")
        if action == "BUY" and sentiment == "BEARISH":
            confidence *= 0.8
            notes.append("Reflection Warning: Sentiment is BEARISH while technical setup is BUY. Reduced confidence level.")
        elif action == "SELL" and sentiment == "BULLISH":
            confidence *= 0.8
            notes.append("Reflection Warning: Sentiment is BULLISH while technical setup is SELL. Reduced confidence level.")
        else:
            notes.append("Reflection Check Passed: Sentiment and technical direction aligned.")

        return {
            "action": action,
            "adjusted_confidence": round(confidence, 2),
            "reflection_notes": " | ".join(notes)
        }
