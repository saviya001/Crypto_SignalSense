import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates Relative Strength Index (RSI)."""
    if df.empty or len(df) <= period:
        return 50.0

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Prevent division by zero when loss is zero
    loss_safe = loss.replace(0, 1e-10)
    rs = gain / loss_safe
    rsi_series = 100 - (100 / (1 + rs))
    latest_rsi = rsi_series.iloc[-1]
    
    if np.isnan(latest_rsi):
        return 50.0
    return float(np.clip(latest_rsi, 0.0, 100.0))

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculates Moving Average Convergence Divergence (MACD)."""
    if df.empty or len(df) < slow:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0, "crossover": "NEUTRAL"}

    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    latest_hist = float(histogram.iloc[-1])

    crossover = "NEUTRAL"
    if len(histogram) >= 2:
        prev_hist = float(histogram.iloc[-2])
        if prev_hist < 0 and latest_hist > 0:
            crossover = "BULLISH_CROSSOVER"
        elif prev_hist > 0 and latest_hist < 0:
            crossover = "BEARISH_CROSSOVER"

    return {
        "macd": latest_macd,
        "signal": latest_signal,
        "histogram": latest_hist,
        "crossover": crossover
    }

def calculate_sma(df: pd.DataFrame, window: int = 20) -> float:
    """Calculates Simple Moving Average (SMA) for specified window."""
    if df.empty or len(df) < window:
        return float(df['close'].iloc[-1]) if not df.empty else 0.0
    return float(df['close'].tail(window).mean())

def calculate_technical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates comprehensive technical metrics for a candle dataframe."""
    if df.empty:
        return {
            "current_price": 0.0,
            "rsi": 50.0,
            "sma_20": 0.0,
            "sma_50": 0.0,
            "macd": {"macd": 0.0, "signal": 0.0, "histogram": 0.0, "crossover": "NEUTRAL"},
            "support": 0.0,
            "resistance": 0.0,
            "trend_bias": "NEUTRAL"
        }

    current_price = float(df['close'].iloc[-1])
    rsi = calculate_rsi(df)
    macd = calculate_macd(df)
    
    sma_20 = calculate_sma(df, 20)
    sma_50 = calculate_sma(df, 50) if len(df) >= 50 else sma_20

    support = float(df['low'].tail(20).min())
    resistance = float(df['high'].tail(20).max())

    return {
        "current_price": current_price,
        "rsi": round(rsi, 2),
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "macd": macd,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "trend_bias": "BULLISH" if current_price >= sma_20 else "BEARISH"
    }
