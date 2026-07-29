import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Crypto SignalSense — Next-Gen AI Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Next-Level CSS (Google Fonts + Cyberpunk Glassmorphism + Glowing Micro-animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* Main Container Deep Cyberpunk Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 40%, #070a12 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Animated Glowing Signal Badges */
    .badge-buy {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 16px 32px;
        border-radius: 14px;
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: pulse-green 2s infinite;
    }
    .badge-sell {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        padding: 16px 32px;
        border-radius: 14px;
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: pulse-red 2s infinite;
    }
    .badge-hold {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: #ffffff;
        padding: 16px 32px;
        border-radius: 14px;
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: pulse-amber 2s infinite;
    }

    @keyframes pulse-green {
        0% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 35px rgba(16, 185, 129, 0.8); }
        100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.8); }
        100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    }
    @keyframes pulse-amber {
        0% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
        50% { box-shadow: 0 0 35px rgba(245, 158, 11, 0.8); }
        100% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
    }

    /* Metric Boxes Custom Styling */
    .stat-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }

    /* RAG Pills & Tags */
    .rag-pill {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Step Workflow Node */
    .workflow-step {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Agent Imports
from agents.router import RouterAgent
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.signal_agent import SignalAgent
from rag.vector_store import SimpleVectorStore
from utils.data_fetch import fetch_coin_price, fetch_ohlc_data

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚡ SignalSense Control Panel")
    st.markdown("---")
    
    # Secrets / Env resolution
    groq_default = os.getenv("GROQ_API_KEY", "")
    try:
        groq_default = st.secrets.get("GROQ_API_KEY", groq_default)
    except Exception:
        pass

    openrouter_default = os.getenv("OPENROUTER_API_KEY", "")
    try:
        openrouter_default = st.secrets.get("OPENROUTER_API_KEY", openrouter_default)
    except Exception:
        pass

    groq_api_key = st.text_input("🔑 Groq API Key (Worker Agents)", value=groq_default, type="password", help="Required for NewsAgent & TechnicalAgent LLM calls")
    openrouter_api_key = st.text_input("🔑 OpenRouter API Key (Signal Agent)", value=openrouter_default, type="password", help="Required for SignalAgent Deep Reasoning LLM calls")
    
    st.markdown("---")
    st.markdown("### 🧠 Multi-Agent Network")
    st.markdown("""
    - 🟢 **RouterAgent**: Message Dispatcher
    - 🔵 **NewsAgent**: Sentiment AI (*Groq Llama 3.1*)
    - 🟣 **TechnicalAgent**: Chart Math AI (*Groq Llama 3.1*)
    - 🟡 **SignalAgent**: Strategy Brain (*OpenRouter DeepSeek*)
    - 📚 **RAG Engine**: Vector Store (22 Strategy PDFs/Docs)
    """)
    st.markdown("---")
    st.caption("IT41043 — Intelligent Systems | Horizon Campus")

# Main Banner Header
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="font-size: 42px; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ⚡ Crypto SignalSense AI
    </h1>
    <p style="font-size: 16px; color: #94a3b8; max-width: 700px; margin: 0 auto;">
        Autonomous Multi-Agent Cryptocurrency Trading System powered by Real-Time Technical Indicators, News Sentiment, and RAG Domain Strategy Vectors.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Coin Selector & Live Quick-Metrics Header Row
st.markdown("### 🪙 Select Cryptocurrency Asset")

cols_coin = st.columns([1, 1, 1, 1, 1])
coins = ["BTC", "ETH", "BNB", "SOL", "XRP"]

if "selected_coin" not in st.session_state:
    st.session_state["selected_coin"] = "BTC"

for i, coin in enumerate(coins):
    with cols_coin[i]:
        pinfo = fetch_coin_price(coin)
        change_col = "#10b981" if pinfo["change_24h"] >= 0 else "#ef4444"
        symbol_prefix = "+" if pinfo["change_24h"] >= 0 else ""
        
        btn_label = f"{coin}\n${pinfo['price']:,.2f}\n{symbol_prefix}{pinfo['change_24h']:.2f}%"
        if st.button(f"{coin} (${pinfo['price']:,.2f})", key=f"btn_{coin}", use_container_width=True):
            st.session_state["selected_coin"] = coin

selected_coin = st.session_state["selected_coin"]

st.info(f"Selected Target Asset: **{selected_coin}**")

col_action = st.columns([1, 2, 1])
with col_action[1]:
    analyze_clicked = st.button("🚀 RUN MULTI-AGENT AI ANALYSIS", type="primary", use_container_width=True)

# Main Analysis Execution & Rendering
if analyze_clicked or "last_result" in st.session_state:
    if analyze_clicked:
        if not groq_api_key or not openrouter_api_key:
            st.error("⚠️ **API Keys Required!** Both `GROQ_API_KEY` and `OPENROUTER_API_KEY` are mandatory to execute the multi-agent AI system. Please enter your API keys in the sidebar or configure `.streamlit/secrets.toml`.")
            st.stop()

        with st.spinner(f"🤖 Agents assembling for {selected_coin} market analysis..."):
            news_agent = NewsAgent(api_key=groq_api_key)
            tech_agent = TechnicalAgent(api_key=groq_api_key)
            vector_store = SimpleVectorStore()
            signal_agent = SignalAgent(openrouter_key=openrouter_api_key, vector_store=vector_store)
            
            router = RouterAgent(news_agent=news_agent, tech_agent=tech_agent, signal_agent=signal_agent)
            
            result = router.run_analysis(selected_coin)
            st.session_state["last_result"] = result
            st.session_state["last_coin"] = selected_coin
    else:
        result = st.session_state["last_result"]

    signal = result["signal"]
    tech = result["technical_analysis"]
    news = result["news_analysis"]

    st.markdown("---")

    # 1. Next-Gen Signal Card Row
    st.markdown(f"## 🎯 Synthesized Signal Card — {result['symbol']}")
    
    action = signal.get("action", "HOLD")
    badge_class = "badge-buy" if action == "BUY" else ("badge-sell" if action == "SELL" else "badge-hold")
    
    c_card1, c_card2 = st.columns([2, 3])
    
    with c_card1:
        st.markdown(f'<div class="{badge_class}">{action}</div>', unsafe_allow_html=True)
        st.write("")
        
        m_a, m_b = st.columns(2)
        with m_a:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Confidence Level</div>
                <div class="stat-value">{signal.get('confidence', 0.85)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m_b:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Risk-to-Reward</div>
                <div class="stat-value">{signal.get('risk_reward_ratio', 2.0):.2f}:1</div>
            </div>
            """, unsafe_allow_html=True)

    with c_card2:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Entry Price</div>
                <div class="stat-value">${signal.get('entry_price', 0.0):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Take Profit (TP)</div>
                <div class="stat-value" style="color: #10b981;">${signal.get('take_profit', 0.0):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Stop Loss (SL)</div>
                <div class="stat-value" style="color: #ef4444;">${signal.get('stop_loss', 0.0):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 💡 Strategic AI Rationale:")
        st.info(signal.get("reasoning", ""))
        
        st.markdown("##### 🛡️ Reflection & Self-Critique Verification:")
        st.success(signal.get("reflection_notes", ""))

    st.markdown("---")

    # 2. Price Action Chart & Worker Agents Row
    st.markdown("## 📊 Live Market Data & Worker Agent Findings")
    
    col_chart, col_workers = st.columns([3, 2])
    
    with col_chart:
        st.markdown("#### 📈 Price Action Chart (Hourly Candles)")
        df_ohlc = fetch_ohlc_data(result['symbol'], limit=40)
        if not df_ohlc.empty:
            chart_df = df_ohlc.set_index('timestamp')[['close', 'open', 'high', 'low']]
            st.line_chart(chart_df[['close']], color="#38bdf8", use_container_width=True)
            
            p_curr = tech.get("metrics", {}).get("current_price", 0)
            sup = tech.get("metrics", {}).get("support", 0)
            res = tech.get("metrics", {}).get("resistance", 0)
            st.caption(f"📍 Current Price: **${p_curr:,.2f}** | Support: **${sup:,.2f}** | Resistance: **${res:,.2f}**")

    with col_workers:
        st.markdown("#### 📊 Technical Agent")
        metrics = tech.get("metrics", {})
        st.markdown(f"**RSI (14):** `{metrics.get('rsi', 50)}` | **Trend Bias:** `{metrics.get('trend_bias', 'NEUTRAL')}`")
        st.write(tech.get("interpretation", ""))

        st.markdown("#### 📰 News & Sentiment Agent")
        sent_tag = news.get("sentiment", "NEUTRAL")
        st.markdown(f"**Sentiment Tag:** `{sent_tag}` | **Confidence:** `{news.get('confidence', 0.8)*100:.0f}%`")
        st.write(news.get("summary", ""))

    st.markdown("---")

    # 3. RAG Knowledge Vector Store Context Drawer
    st.markdown("## 📚 RAG Knowledge Base Vector References")
    
    sources = signal.get("rag_sources", [])
    if sources:
        for s in sources:
            st.markdown(f'<span class="rag-pill">📖 {s["title"]} ({s["source"]})</span>', unsafe_allow_html=True)

    with st.expander(f"Inspect Retrieved RAG Strategy Chunks ({len(sources)} documents)"):
        for src in sources:
            st.markdown(f"#### 📄 {src['title']} (`{src['source']}`)")
            st.code(src["content"][:400] + "...", language="markdown")

    st.markdown("---")

    # 4. Agent-to-Agent Communication Protocol Visualizer
    st.markdown("## 💬 Agent-to-Agent Message Protocol Trace")
    st.caption("Demonstrates decoupled JSON message passing between Router, Worker Agents, and Signal Strategist.")
    
    with st.expander("Inspect Live Agent Communication Log"):
        for idx, msg in enumerate(result.get("message_flow", []), 1):
            sender = msg['sender'].upper()
            receiver = msg['receiver'].upper()
            mtype = msg['message_type']
            
            st.markdown(f"""
            <div class="workflow-step">
                <b>Step {idx}:</b> <code>{sender}</code> ➔ <code>{receiver}</code> | Type: <code>{mtype}</code>
            </div>
            """, unsafe_allow_html=True)
            st.json(msg["payload"])

# Educational Disclaimer Footer
st.markdown("---")
st.warning("⚠️ **Academic & Financial Disclaimer:** Signals, Entry prices, Take Profit, and Stop Loss estimates generated by this AI system are educational technical estimates calculated from Support/Resistance levels and ATR volatility bands for IT41043 coursework evaluation. They do not constitute financial advice, backtested execution guarantees, or commercial trading signals.")
st.caption("⚡ Crypto SignalSense | IT41043 Intelligent Systems (Agentic AI) | Developed for Horizon Campus Submission | Live OpenRouter & Groq Multi-Agent Integration")
