import streamlit as st
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Crypto SignalSense — Multi-Agent AI Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Dark glassmorphism container styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .signal-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    }
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
    }
    .agent-msg-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    .rag-tag {
        background: #334155;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Imports after env loading
from agents.router import RouterAgent
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.signal_agent import SignalAgent
from rag.vector_store import SimpleVectorStore

# Sidebar Configuration
with st.sidebar:
    st.title("⚡ SignalSense Settings")
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

    groq_api_key = st.text_input("Groq API Key (Worker Agents)", value=groq_default, type="password")
    openrouter_api_key = st.text_input("OpenRouter API Key (Signal Agent)", value=openrouter_default, type="password")
    
    st.markdown("---")
    st.markdown("### 🤖 Agentic Architecture")
    st.markdown("""
    - **Router:** Dispatch Orchestrator
    - **Worker 1:** News & Sentiment Agent (*Groq*)
    - **Worker 2:** Technical Analyst Agent (*Groq*)
    - **Brain:** Signal Strategist (*OpenRouter + RAG*)
    """)
    st.markdown("---")
    st.caption("IT41043 — Intelligent Systems Assignment")

# Main Dashboard Header
st.title("📈 Crypto SignalSense — Agentic AI Signal Generator")
st.markdown("Autonomous multi-agent crypto analysis powered by RAG, technical indicators, and real-time news sentiment.")

# Coin Selection Row
col_coin, col_btn = st.columns([3, 2])

with col_coin:
    selected_coin = st.selectbox(
        "Select Cryptocurrency to Analyze (1 of 5):",
        options=["BTC", "ETH", "BNB", "SOL", "XRP"],
        index=0,
        help="Choose a cryptocurrency asset to trigger multi-agent analysis."
    )

with col_btn:
    st.write("") # Spacer
    st.write("") 
    analyze_clicked = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

if analyze_clicked or "last_result" in st.session_state:
    if analyze_clicked:
        if not groq_api_key or not openrouter_api_key:
            st.error("⚠️ **API Keys Required!** Both `GROQ_API_KEY` and `OPENROUTER_API_KEY` are mandatory to execute the multi-agent AI system. Please enter your API keys in the sidebar or configure `.streamlit/secrets.toml`.")
            st.stop()

        with st.spinner(f"Agents assembling for {selected_coin} analysis..."):
            # Initialize Agents
            news_agent = NewsAgent(api_key=groq_api_key)
            tech_agent = TechnicalAgent(api_key=groq_api_key)
            vector_store = SimpleVectorStore()
            signal_agent = SignalAgent(openrouter_key=openrouter_api_key, vector_store=vector_store)
            
            router = RouterAgent(news_agent=news_agent, tech_agent=tech_agent, signal_agent=signal_agent)
            
            # Run multi-agent pipeline
            result = router.run_analysis(selected_coin)
            st.session_state["last_result"] = result
            st.session_state["last_coin"] = selected_coin
    else:
        result = st.session_state["last_result"]

    signal = result["signal"]
    tech = result["technical_analysis"]
    news = result["news_analysis"]

    st.markdown("---")

    # 1. Final Signal Card Section
    st.subheader(f"🎯 Final Signal Card — {result['symbol']}")
    
    action = signal.get("action", "HOLD")
    badge_class = "signal-buy" if action == "BUY" else ("signal-sell" if action == "SELL" else "signal-hold")
    
    col_sig, col_metrics = st.columns([2, 3])
    
    with col_sig:
        st.markdown(f'<div class="{badge_class}">{action}</div>', unsafe_allow_html=True)
        st.write("")
        st.metric("Confidence Level", f"{signal.get('confidence', 0.85)*100:.1f}%")
        st.metric("Risk-to-Reward Ratio", f"{signal.get('risk_reward_ratio', 2.0):.2f}:1")

    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Entry Price", f"${signal.get('entry_price', 0.0):,.2f}")
        m2.metric("Take Profit (TP)", f"${signal.get('take_profit', 0.0):,.2f}")
        m3.metric("Stop Loss (SL)", f"${signal.get('stop_loss', 0.0):,.2f}")

        st.markdown("**Strategic Rationale:**")
        st.info(signal.get("reasoning", ""))
        
        st.markdown("**🛡️ Reflection & Self-Critique Notes:**")
        st.success(signal.get("reflection_notes", ""))

    st.markdown("---")

    # 2. Worker Agent Findings (2 Columns)
    c_tech, c_news = st.columns(2)

    with c_tech:
        st.subheader("📊 Technical Agent Findings")
        metrics = tech.get("metrics", {})
        t1, t2, t3 = st.columns(3)
        t1.metric("Current Price", f"${metrics.get('current_price', 0):,.2f}")
        t2.metric("RSI (14)", f"{metrics.get('rsi', 50)}")
        t3.metric("Trend Bias", f"{metrics.get('trend_bias', 'NEUTRAL')}")

        st.markdown("**Technical Interpretation:**")
        st.write(tech.get("interpretation", ""))
        
        st.caption(f"Support Zone: ${metrics.get('support', 0):,.2f} | Resistance Zone: ${metrics.get('resistance', 0):,.2f}")

    with c_news:
        st.subheader("📰 News & Sentiment Agent Findings")
        sentiment_tag = news.get("sentiment", "NEUTRAL")
        st.markdown(f"**Overall News Sentiment:** `{sentiment_tag}` (Confidence: {news.get('confidence', 0.8)*100:.0f}%)")
        st.write(news.get("summary", ""))

        with st.expander("View Recent Headlines Analyzed"):
            for h in news.get("headlines", []):
                st.markdown(f"- {h}")

    st.markdown("---")

    # 3. RAG References Accordion
    st.subheader("📚 RAG Knowledge Base References")
    with st.expander(f"View Retrieved Strategy Documents ({len(signal.get('rag_sources', []))} chunks)"):
        for src in signal.get("rag_sources", []):
            st.markdown(f"#### 📖 {src['title']} ({src['source']})")
            st.code(src["content"][:400] + "...", language="markdown")

    st.markdown("---")

    # 4. Agent-to-Agent Message Flow Visualizer
    st.subheader("💬 Agent-to-Agent Message Exchange Flow")
    st.caption("Demonstrates structured JSON message protocol exchange between Router, Worker Agents, and Signal Strategist.")
    
    with st.expander("Expand Agent Communication Log"):
        for idx, msg in enumerate(result.get("message_flow", []), 1):
            sender = msg['sender'].upper()
            receiver = msg['receiver'].upper()
            mtype = msg['message_type']
            st.markdown(f"**Step {idx}:** `{sender}` ➔ `{receiver}` | Type: `{mtype}`")
            st.json(msg["payload"])

else:
    st.info("👈 Select a cryptocurrency above and click **Run Multi-Agent Analysis** to trigger the AI system.")

# Footer
st.markdown("---")
st.caption("⚡ Crypto SignalSense | IT41043 Intelligent Systems (Agentic AI) | Developed for Horizon Campus Submission | Live OpenRouter & Groq Integration")
