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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .up-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .down-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
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

# Main App UI
st.markdown('<h1 class="main-header">🚀 Advanced Stock Prediction Hub</h1>', unsafe_allow_html=True)

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
    
    # Display stock information cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏢 {stock_info['name']}</h4>
            <p><strong>Sector:</strong> {stock_info['sector']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        market_cap_b = stock_info['market_cap'] / 1e9 if stock_info['market_cap'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Market Cap</h4>
            <p><strong>${market_cap_b:.1f}B</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 P/E Ratio</h4>
            <p><strong>{stock_info['pe_ratio']:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dividend_pct = (stock_info['dividend_yield'] * 100) if stock_info['dividend_yield'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>💵 Dividend Yield</h4>
            <p><strong>{dividend_pct:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction section
    if st.button("🎯 Generate Predictions", type="primary", use_container_width=True) or st.session_state.get('analyze', False):
        with st.spinner("🔄 Fetching data and generating AI predictions..."):
            
            # Process data
            X_cls, X_seq, true_cls, true_price, scaler, df = get_processed_data(ticker, period)
            
            if df is not None and len(df) > 0:
                # Make predictions
                pred_cls_mlp = mlp_cls_model.predict(X_cls, verbose=0)[0][0]
                pred_cls_lstm = lstm_cls_model.predict(X_seq, verbose=0)[0][0]
                pred_reg_mlp = mlp_reg_model.predict(X_cls, verbose=0)[0][0]
                pred_reg_lstm = lstm_reg_model.predict(X_seq, verbose=0)[0][0]
                
                current_price = df['Close'].iloc[-1]
                
                # Reset analyze state
                if 'analyze' in st.session_state:
                    del st.session_state.analyze
                
                # Display predictions in an attractive layout
                st.markdown("---")
                st.subheader("🎯 AI Predictions Dashboard")
                
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
                
                # Interactive charts
                st.subheader("📊 Interactive Visualizations")
                
                tab1, tab2, tab3, tab4 = st.tabs(["🕯️ Technical Chart", "📈 Predictions", "📊 Model Comparison", "📋 Analysis Summary"])
                
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
                    
                    st.markdown(f"""
                    ### ⚠️ Risk Assessment
                    
                    **Overall Risk Level:** <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                    
                    **Risk Factors:**
                    - **Beta:** {stock_info['beta']:.2f} (Market sensitivity)
                    - **Recent Volatility:** {df.iloc[-1]['Volatility']:.4f if len(df) > 20 else 'N/A'}
                    
                    **Investment Considerations:**
                    - This is for educational purposes only
                    - Past performance doesn't guarantee future results
                    - Consider diversification and risk tolerance
                    """, unsafe_allow_html=True)
            
            else:
                st.error("❌ Unable to fetch data for this ticker. Please check the symbol and try again.")

else:
    st.error("❌ Models could not be loaded. Please check that the model files exist in the 'models/' directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 <strong>Advanced Stock Prediction Hub</strong> | Built with ❤️ using Streamlit & AI</p>
    <p>⚠️ <em>This tool is for educational purposes only. Not financial advice.</em></p>
    <p>📊 Data powered by Yahoo Finance | 🤖 AI Models: MLP & LSTM Neural Networks</p>
</div>
""", unsafe_allow_html=True)
