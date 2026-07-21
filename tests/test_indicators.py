import unittest
import pandas as pd
from utils.indicators import calculate_rsi, calculate_macd, calculate_technical_summary

class TestIndicators(unittest.TestCase):
    def test_rsi_calculation(self):
        prices = [float(10 + i * 2) for i in range(25)]
        df = pd.DataFrame({'close': prices})
        rsi = calculate_rsi(df, period=14)
        self.assertGreater(rsi, 50.0)

    def test_macd_calculation(self):
        prices = [float(i) for i in range(10, 40)]
        df = pd.DataFrame({'close': prices})
        macd = calculate_macd(df)
        self.assertIn("macd", macd)
        self.assertIn("signal", macd)
        self.assertIn("crossover", macd)

    def test_technical_summary(self):
        records = []
        for i in range(30):
            records.append({
                "timestamp": i,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 500
            })
        df = pd.DataFrame(records)
        summary = calculate_technical_summary(df)
        self.assertGreater(summary["current_price"], 0)
        self.assertTrue(0 <= summary["rsi"] <= 100)
        self.assertLessEqual(summary["support"], summary["resistance"])

if __name__ == "__main__":
    unittest.main()
