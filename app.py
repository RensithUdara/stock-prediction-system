import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
import time
from datetime import timedelta

# Configure page
st.set_page_config(
    page_title="🚀 Advanced Stock Prediction Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main .block-container {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced metric cards with gradients */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.25);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.95;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Alternative colorful metric cards */
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card-purple {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card-blue:hover, .metric-card-green:hover, 
    .metric-card-orange:hover, .metric-card-purple:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    /* Enhanced prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .up-prediction {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
    }
    
    .down-prediction {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        border: none;
    }
    
    .prediction-box h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .prediction-box h2 {
        margin: 0 0 1rem 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Enhanced info box */
    .info-box {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        color: #234e52;
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(45, 55, 72, 0.08);
        border-left: 5px solid #38b2ac;
    }
    
    .info-box h4 {
        color: #2c7a7b;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Tab styling with animations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 8px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        color: #475569;
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-color: #cbd5e0;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        color: #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px solid #667eea;
        color: white;
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        z-index: 10;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* Add glowing effect for active tab */
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 100%);
        pointer-events: none;
        border-radius: 10px;
    }
    
    /* Add subtle animation for tab content */
    .stTabs [role="tabpanel"] {
        animation: fadeIn 0.5s ease-in-out;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Enhanced section header styling */
    .section-header {
        text-align: center;
        margin: 3rem 0 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Metric component styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def load_models():
    """Load all ML models with caching"""
    try:
        mlp_cls_model = load_model("models/mlp_classification.h5", compile=False)
        lstm_cls_model = load_model("models/lstm_classification.h5", compile=False)
        mlp_reg_model = load_model("models/mlp_regression.h5", compile=False)
        lstm_reg_model = load_model("models/lstm_regression.h5", compile=False)
        return mlp_cls_model, lstm_cls_model, mlp_reg_model, lstm_reg_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Load models
mlp_cls_model, lstm_cls_model, mlp_reg_model, lstm_reg_model = load_models()

# Enhanced data processing function
@st.cache_data(ttl=300)
def get_processed_data(ticker, period="2y"):
    """Enhanced data processing with technical indicators"""
    try:
        # Download data with configurable period
        df = yf.download(ticker, period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)

        # Add technical indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        # Price change and volatility
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Drop NaN values after adding indicators
        df.dropna(inplace=True)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]].values)

        y_cls = (df["Close"].shift(-1) > df["Close"]).astype(int).values[:-1]
        X_cls = X_scaled[:-1]
        y_reg = df["Close"].shift(-1).values[:-1]
        X_reg = X_scaled[:-1]

        seq_len = 10
        X_seq = []
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_len:i])
        X_seq = np.array(X_seq)

        return X_cls[-1:], X_seq[-1:], y_cls[-1], y_reg[-1], scaler, df
    except Exception as e:
        st.error(f"Error processing data for {ticker}: {str(e)}")
        return None, None, None, None, None, None

def get_stock_info(ticker):
    """Get additional stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'current_price': info.get('currentPrice', 0)
        }
    except:
        return {'name': ticker, 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 0, 'dividend_yield': 0, 'beta': 0, 'current_price': 0}

def create_candlestick_chart(df, ticker):
    """Create interactive candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Chart with Technical Indicators', 'Volume', 'RSI'),
        row_width=[0.2, 0.1, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12', line=dict(color='purple')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='blue')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} - Comprehensive Technical Analysis',
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

def create_prediction_charts(pred_cls_mlp, pred_cls_lstm, pred_reg_mlp, pred_reg_lstm, current_price):
    """Create enhanced prediction visualization charts"""
    
    # Classification predictions chart
    fig_cls = go.Figure(data=[
        go.Bar(
            x=['MLP Model', 'LSTM Model'],
            y=[pred_cls_mlp, pred_cls_lstm],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f'{pred_cls_mlp:.3f}', f'{pred_cls_lstm:.3f}'],
            textposition='auto',
        )
    ])
    
    fig_cls.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
    fig_cls.update_layout(
        title='📊 Price Direction Prediction Confidence',
        yaxis_title='Confidence Score (0-1)',
        template='plotly_white',
        height=400
    )
    
    # Price prediction chart
    fig_price = go.Figure(data=[
        go.Bar(
            x=['Current Price', 'MLP Prediction', 'LSTM Prediction'],
            y=[current_price, pred_reg_mlp, pred_reg_lstm],
            marker_color=['#2ca02c', '#d62728', '#9467bd'],
            text=[f'${current_price:.2f}', f'${pred_reg_mlp:.2f}', f'${pred_reg_lstm:.2f}'],
            textposition='auto',
        )
    ])
    
    fig_price.update_layout(
        title='💰 Price Prediction Comparison',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400
    )
    
    return fig_cls, fig_price

# Main App UI with enhanced header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="main-header">🚀 Advanced Stock Prediction Hub</h1>
    <p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">
        AI-Powered Stock Analysis & Price Prediction Platform
    </p>
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 4px; width: 200px; margin: 1rem auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Stock selection with search
    st.subheader("📊 Stock Selection")
    stock_categories = {
        "🔥 Popular Stocks": ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META"],
        "🏦 Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "🛒 Consumer": ["WMT", "KO", "PG", "NKE", "MCD", "SBUX"],
        "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "TMO", "DHR"],
        "🎬 Entertainment": ["DIS", "NFLX", "CMCSA", "T", "VZ"],
        "⚡ Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
        "🏭 Industrial": ["BA", "CAT", "GE", "MMM", "UPS", "FDX"]
    }
    
    # Category selection
    selected_category = st.selectbox("Choose Category", list(stock_categories.keys()))
    ticker = st.selectbox("🔍 Select Stock", stock_categories[selected_category])
    
    # Custom ticker input
    custom_ticker = st.text_input("Or enter custom ticker:", placeholder="e.g., AAPL")
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    # Time period selection
    st.subheader("📅 Analysis Period")
    period = st.selectbox("Select Period", 
                         ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                         index=3)
    
    # Advanced settings
    st.subheader("⚙️ Advanced Settings")
    show_technical = st.checkbox("Show Technical Indicators", True)
    show_sentiment = st.checkbox("Show Market Sentiment", True)
    auto_refresh = st.checkbox("Auto Refresh (30s)", False)
    
    # Model confidence threshold
    confidence_threshold = st.slider("Prediction Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# Check if models are loaded
if all([mlp_cls_model, lstm_cls_model, mlp_reg_model, lstm_reg_model]):
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📈 Analysis for {ticker}")
        
    with col2:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
    with col3:
        if st.button("📊 Analyze Stock", type="secondary", use_container_width=True):
            st.session_state.analyze = True
    
    # Get stock information
    stock_info = get_stock_info(ticker)
    
    # Display enhanced stock information cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-blue">
            <h4>🏢 Company Info</h4>
            <p><strong>{stock_info['name'][:25]}{'...' if len(stock_info['name']) > 25 else ''}</strong></p>
            <p>Sector: {stock_info['sector']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        market_cap_b = stock_info['market_cap'] / 1e9 if stock_info['market_cap'] else 0
        st.markdown(f"""
        <div class="metric-card-green">
            <h4>💰 Market Cap</h4>
            <p><strong>${market_cap_b:.1f}B</strong></p>
            <p>Beta: {stock_info['beta']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-orange">
            <h4>📊 Valuation</h4>
            <p><strong>P/E: {stock_info['pe_ratio']:.2f}</strong></p>
            <p>Current: ${stock_info['current_price']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dividend_pct = (stock_info['dividend_yield'] * 100) if stock_info['dividend_yield'] else 0
        st.markdown(f"""
        <div class="metric-card-purple">
            <h4>💵 Dividend</h4>
            <p><strong>{dividend_pct:.2f}%</strong></p>
            <p>Yield Rate</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction section with enhanced loading
    if st.button("🎯 Generate Predictions", type="primary", use_container_width=True) or st.session_state.get('analyze', False):
        
        # Custom loading animation
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="display: inline-block; animation: spin 2s linear infinite;">
                🔄
            </div>
            <h3 style="color: #667eea; margin: 1rem 0;">AI is analyzing market data...</h3>
            <p style="color: #666;">Fetching real-time data • Running ML models • Calculating predictions</p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """, unsafe_allow_html=True)
        
        with st.spinner(""):
            
            # Process data
            X_cls, X_seq, true_cls, true_price, scaler, df = get_processed_data(ticker, period)
            
            if df is not None and len(df) > 0:
                # Make predictions
                pred_cls_mlp = mlp_cls_model.predict(X_cls, verbose=0)[0][0]
                pred_cls_lstm = lstm_cls_model.predict(X_seq, verbose=0)[0][0]
                pred_reg_mlp = mlp_reg_model.predict(X_cls, verbose=0)[0][0]
                pred_reg_lstm = lstm_reg_model.predict(X_seq, verbose=0)[0][0]
                
                current_price = df['Close'].iloc[-1]
                
                # Clear loading animation
                loading_placeholder.empty()
                
                # Reset analyze state
                if 'analyze' in st.session_state:
                    del st.session_state.analyze
                
                # Display success message
                st.success("🎉 Predictions generated successfully!")
                
                # Display predictions in an attractive layout
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <h2 style="color: #667eea; margin-bottom: 0.5rem;">🎯 AI Predictions Dashboard</h2>
                    <p style="color: #666;">Advanced neural network analysis results</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Price direction predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    mlp_direction = "🔼 UP" if pred_cls_mlp > confidence_threshold else "🔽 DOWN"
                    mlp_confidence = max(pred_cls_mlp, 1-pred_cls_mlp)
                    box_class = "up-prediction" if pred_cls_mlp > confidence_threshold else "down-prediction"
                    
                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                        <h3>🤖 MLP Model Prediction</h3>
                        <h2>{mlp_direction}</h2>
                        <p>Confidence: {mlp_confidence:.1%}</p>
                        <p>Score: {pred_cls_mlp:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    lstm_direction = "🔼 UP" if pred_cls_lstm > confidence_threshold else "🔽 DOWN"
                    lstm_confidence = max(pred_cls_lstm, 1-pred_cls_lstm)
                    box_class = "up-prediction" if pred_cls_lstm > confidence_threshold else "down-prediction"
                    
                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                        <h3>🧠 LSTM Model Prediction</h3>
                        <h2>{lstm_direction}</h2>
                        <p>Confidence: {lstm_confidence:.1%}</p>
                        <p>Score: {pred_cls_lstm:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Price predictions
                st.subheader("💰 Price Forecasts")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    mlp_change = pred_reg_mlp - current_price
                    st.metric("MLP Prediction", f"${pred_reg_mlp:.2f}", f"{mlp_change:+.2f}")
                
                with col3:
                    lstm_change = pred_reg_lstm - current_price
                    st.metric("LSTM Prediction", f"${pred_reg_lstm:.2f}", f"{lstm_change:+.2f}")
                
                with col4:
                    avg_prediction = (pred_reg_mlp + pred_reg_lstm) / 2
                    avg_change = avg_prediction - current_price
                    st.metric("Average Prediction", f"${avg_prediction:.2f}", f"{avg_change:+.2f}")
                
                # Interactive charts with enhanced header
                st.markdown("""
                <div style="text-align: center; margin: 3rem 0 2rem 0;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                                background-clip: text; display: inline-block;">
                        <h2 style="margin: 0; font-size: 2.2rem; font-weight: 700;">
                            📊 Interactive Visualizations
                        </h2>
                    </div>
                    <p style="color: #666; margin-top: 0.5rem; font-size: 1.1rem;">
                        Comprehensive technical analysis and prediction charts
                    </p>
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                height: 3px; width: 150px; margin: 1rem auto; border-radius: 2px;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🕯️ Technical Chart", 
                    "📈 Predictions", 
                    "📊 Model Comparison", 
                    "📋 Analysis Summary"
                ])
                
                with tab1:
                    if show_technical and len(df) > 20:
                        fig_tech = create_candlestick_chart(df, ticker)
                        st.plotly_chart(fig_tech, use_container_width=True)
                    else:
                        # Simple price chart
                        fig_simple = go.Figure()
                        fig_simple.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
                        fig_simple.update_layout(title=f'{ticker} Price History', template='plotly_white')
                        st.plotly_chart(fig_simple, use_container_width=True)
                
                with tab2:
                    fig_cls, fig_price = create_prediction_charts(pred_cls_mlp, pred_cls_lstm, pred_reg_mlp, pred_reg_lstm, current_price)
                    st.plotly_chart(fig_cls, use_container_width=True)
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with tab3:
                    # Model performance comparison
                    model_data = {
                        'Model': ['MLP', 'LSTM'],
                        'Direction_Confidence': [max(pred_cls_mlp, 1-pred_cls_mlp), max(pred_cls_lstm, 1-pred_cls_lstm)],
                        'Price_Change_Prediction': [mlp_change, lstm_change],
                        'Direction_Prediction': [mlp_direction, lstm_direction]
                    }
                    
                    df_models = pd.DataFrame(model_data)
                    st.dataframe(df_models, use_container_width=True)
                    
                    # Ensemble prediction
                    ensemble_direction = "🔼 UP" if (pred_cls_mlp + pred_cls_lstm) / 2 > confidence_threshold else "🔽 DOWN"
                    ensemble_confidence = abs((pred_cls_mlp + pred_cls_lstm) / 2 - 0.5) + 0.5
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>🎯 Ensemble Prediction (Combined Models)</h4>
                        <p><strong>Direction:</strong> {ensemble_direction}</p>
                        <p><strong>Confidence:</strong> {ensemble_confidence:.1%}</p>
                        <p><strong>Average Price Target:</strong> ${avg_prediction:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab4:
                    # Technical analysis summary
                    if len(df) > 20:
                        latest = df.iloc[-1]
                        rsi_signal = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
                        ma_signal = "Bullish" if latest['Close'] > latest['SMA_20'] else "Bearish"
                        
                        st.markdown(f"""
                        ### 📋 Technical Analysis Summary
                        
                        **Current Indicators:**
                        - **RSI (14):** {latest['RSI']:.1f} - {rsi_signal}
                        - **Price vs SMA(20):** {ma_signal}
                        - **Volatility:** {latest['Volatility']:.4f}
                        - **Volume vs Average:** {(latest['Volume'] / latest['Volume_SMA']):.2f}x
                        
                        **Key Levels:**
                        - **Bollinger Upper:** ${latest['BB_upper']:.2f}
                        - **Bollinger Lower:** ${latest['BB_lower']:.2f}
                        - **Support/Resistance Range:** ${latest['BB_lower']:.2f} - ${latest['BB_upper']:.2f}
                        """)
                    
                    # Risk assessment
                    risk_score = 0
                    if stock_info['beta'] > 1.5:
                        risk_score += 2
                    elif stock_info['beta'] > 1:
                        risk_score += 1
                    
                    if len(df) > 20 and df.iloc[-1]['Volatility'] > 0.03:
                        risk_score += 2
                    elif len(df) > 20 and df.iloc[-1]['Volatility'] > 0.02:
                        risk_score += 1
                    
                    risk_level = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"
                    risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                    
                    volatility_text = f"{df.iloc[-1]['Volatility']:.4f}" if len(df) > 20 else "N/A"
                    
                    st.markdown(f"""
                    ### ⚠️ Risk Assessment
                    
                    **Overall Risk Level:** <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                    
                    **Risk Factors:**
                    - **Beta:** {stock_info['beta']:.2f} (Market sensitivity)
                    - **Recent Volatility:** {volatility_text}
                    
                    **Investment Considerations:**
                    - This is for educational purposes only
                    - Past performance doesn't guarantee future results
                    - Consider diversification and risk tolerance
                    """, unsafe_allow_html=True)
            
            else:
                st.error("❌ Unable to fetch data for this ticker. Please check the symbol and try again.")

else:
    st.error("❌ Models could not be loaded. Please check that the model files exist in the 'models/' directory.")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="text-align: center;">
        <h3>🚀 <strong>Advanced Stock Prediction Hub</strong></h3>
        <p style="font-size: 1.1rem; margin: 1rem 0;">Built with ❤️ using Streamlit & AI Neural Networks</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 1.5rem 0;">
            <div>📊 <strong>Data Source:</strong> Yahoo Finance</div>
            <div>🤖 <strong>AI Models:</strong> MLP & LSTM</div>
            <div>📈 <strong>Technical Analysis:</strong> Advanced Indicators</div>
        </div>
        <p style="color: #ffd700; font-weight: 600; font-size: 1rem;">
            ⚠️ This tool is for educational purposes only. Not financial advice.
        </p>
        <p style="opacity: 0.8; margin-top: 1rem;">
            Always consult with a qualified financial advisor before making investment decisions.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
