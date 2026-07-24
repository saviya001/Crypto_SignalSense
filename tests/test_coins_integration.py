import unittest
from agents.router import RouterAgent

class TestCoinsIntegration(unittest.TestCase):
    def setUp(self):
        from agents.news_agent import NewsAgent
        from agents.technical_agent import TechnicalAgent
        from agents.signal_agent import SignalAgent
        
        news_agent = NewsAgent(api_key="test_groq_key")
        tech_agent = TechnicalAgent(api_key="test_groq_key")
        signal_agent = SignalAgent(openrouter_key="test_openrouter_key")
        self.router = RouterAgent(news_agent=news_agent, tech_agent=tech_agent, signal_agent=signal_agent)

    def test_all_five_coins(self):
        coins = ["BTC", "ETH", "BNB", "SOL", "XRP"]
        for coin in coins:
            res = self.router.run_analysis(coin)
            self.assertEqual(res["symbol"], coin)
            self.assertIn(res["signal"]["action"], ["BUY", "SELL", "HOLD"])
            self.assertGreater(res["signal"]["entry_price"], 0.0)
            self.assertGreaterEqual(len(res["message_flow"]), 5)

if __name__ == "__main__":
    unittest.main()
