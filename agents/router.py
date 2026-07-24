from typing import Dict, Any, List
from agents.protocol import AgentMessage, AgentRole, MessageType
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.signal_agent import SignalAgent

class RouterAgent:
    """
    Orchestrator & Router Agent.
    Implements Router & Orchestrator-Worker Patterns.
    Routes structured requests to NewsAgent and TechnicalAgent, then passes
    the combined message payloads to SignalAgent.
    """
    def __init__(self, news_agent: NewsAgent = None, tech_agent: TechnicalAgent = None, signal_agent: SignalAgent = None):
        self.role = AgentRole.ROUTER
        self.news_agent = news_agent or NewsAgent()
        self.tech_agent = tech_agent or TechnicalAgent()
        self.signal_agent = signal_agent or SignalAgent()
        self.message_history: List[AgentMessage] = []

    def run_analysis(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        self.message_history.clear()

        # Step 1: Create dispatch request messages for worker agents
        msg_to_news = AgentMessage(
            sender=self.role,
            receiver=AgentRole.NEWS_AGENT,
            message_type=MessageType.REQUEST,
            payload={"symbol": symbol}
        )
        msg_to_tech = AgentMessage(
            sender=self.role,
            receiver=AgentRole.TECHNICAL_AGENT,
            message_type=MessageType.REQUEST,
            payload={"symbol": symbol}
        )

        self.message_history.append(msg_to_news)
        self.message_history.append(msg_to_tech)

        # Step 2: Execute Worker Agents
        res_news = self.news_agent.process_message(msg_to_news)
        res_tech = self.tech_agent.process_message(msg_to_tech)

        self.message_history.append(res_news)
        self.message_history.append(res_tech)

        # Step 3: SignalAgent RAG + Synthesis + Reflection
        signal_output = self.signal_agent.generate_signal(symbol, res_news, res_tech)

        msg_final = AgentMessage(
            sender=AgentRole.SIGNAL_AGENT,
            receiver=self.role,
            message_type=MessageType.RESPONSE,
            payload={"final_signal": signal_output}
        )
        self.message_history.append(msg_final)

        return {
            "symbol": symbol,
            "signal": signal_output,
            "news_analysis": res_news.payload,
            "technical_analysis": res_tech.payload,
            "message_flow": [m.to_dict() for m in self.message_history]
        }
