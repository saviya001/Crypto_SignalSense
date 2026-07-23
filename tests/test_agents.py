import unittest
from agents.protocol import AgentMessage, AgentRole, MessageType
from agents.router import RouterAgent

class TestAgentOrchestration(unittest.TestCase):
    def test_message_protocol(self):
        msg = AgentMessage(
            sender=AgentRole.ROUTER,
            receiver=AgentRole.NEWS_AGENT,
            message_type=MessageType.REQUEST,
            payload={"symbol": "BTC"}
        )
        self.assertEqual(msg.sender, AgentRole.ROUTER)
        self.assertEqual(msg.receiver, AgentRole.NEWS_AGENT)
        self.assertEqual(msg.payload["symbol"], "BTC")
        d = msg.to_dict()
        self.assertEqual(d["sender"], "router")

    def test_message_timestamp(self):
        msg = AgentMessage(
            sender=AgentRole.NEWS_AGENT,
            receiver=AgentRole.SIGNAL_AGENT,
            message_type=MessageType.RESPONSE,
            payload={"sentiment": "BULLISH"}
        )
        self.assertGreater(msg.timestamp, 0.0)
        self.assertTrue(msg.message_id.startswith("msg_"))

    def test_router_end_to_end_flow(self):
        from agents.news_agent import NewsAgent
        from agents.technical_agent import TechnicalAgent
        from agents.signal_agent import SignalAgent
        
        # Test with dummy key to verify validation or mock pass
        news_agent = NewsAgent(api_key="test_groq_key")
        tech_agent = TechnicalAgent(api_key="test_groq_key")
        signal_agent = SignalAgent(openrouter_key="test_openrouter_key")
        
        router = RouterAgent(news_agent=news_agent, tech_agent=tech_agent, signal_agent=signal_agent)
        result = router.run_analysis("BTC")
        self.assertIn("signal", result)
        self.assertIn("message_flow", result)
        self.assertGreaterEqual(len(result["message_flow"]), 5)
        
        signal = result["signal"]
        self.assertIn(signal["action"], ["BUY", "SELL", "HOLD"])
        self.assertGreater(signal["take_profit"], 0)
        self.assertGreater(signal["stop_loss"], 0)

if __name__ == "__main__":
    unittest.main()
