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

# Preprocess function
def get_processed_data(ticker):
    df = yf.download(ticker, start="2022-01-01", end=datetime.date.today())
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)

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

# UI
st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("📈 Real-Time Stock Price Prediction App")

stocks = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META", "NFLX", "JPM", "WMT", "XOM", "DIS", "PFE"]
ticker = st.selectbox("🔍 Select a Stock", stocks)

if st.button("Predict"):
    with st.spinner("⏳ Fetching data and making predictions..."):
        X_cls, X_seq, true_cls, true_price, scaler, df = get_processed_data(ticker)

        # Predictions
        pred_cls_mlp = mlp_cls_model.predict(X_cls)[0][0]
        pred_cls_lstm = lstm_cls_model.predict(X_seq)[0][0]
        pred_reg_mlp = mlp_reg_model.predict(X_cls)[0][0]
        pred_reg_lstm = lstm_reg_model.predict(X_seq)[0][0]

        # Display classification predictions
        st.subheader("📊 Will the Price Go Up Tomorrow?")
        st.markdown(f"**MLP Classification:** {'🔼 Up' if pred_cls_mlp > 0.5 else '🔽 Down'} ({pred_cls_mlp:.2f})")
        st.markdown(f"**LSTM Classification:** {'🔼 Up' if pred_cls_lstm > 0.5 else '🔽 Down'} ({pred_cls_lstm:.2f})")

        # Visualize predicted price movement
        fig, ax = plt.subplots()
        ax.bar(['MLP', 'LSTM'], [pred_cls_mlp, pred_cls_lstm], color=['blue', 'orange'])
        ax.set_ylabel('Prediction Score')
        ax.set_title('Predicted Price Movement (Up or Down)')
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # Display predicted next close price
        st.subheader("💰 Predicted Next-Day Closing Price")
        st.markdown(f"**MLP Regression:** ${float(pred_reg_mlp):.2f}")
        st.markdown(f"**LSTM Regression:** ${float(pred_reg_lstm):.2f}")

        # Visualize predicted next-day closing price
        fig, ax = plt.subplots()
        ax.bar(['MLP Predicted', 'LSTM Predicted'], [pred_reg_mlp, pred_reg_lstm], color=['green', 'red'])
        ax.set_ylabel('Price ($)')
        ax.set_title('Predicted Next-Day Closing Price')
        st.pyplot(fig)

        # Historical Close Price chart
        st.subheader("📈 Historical Closing Price")
        st.line_chart(df["Close"])

        # Visualize
        today_close = df["Close"].iloc[-1]
        tomorrow_close_mlp = pred_reg_mlp
        tomorrow_close_lstm = pred_reg_lstm
        
        fig, ax = plt.subplots()
        ax.bar(['Today', 'Tomorrow (MLP)', 'Tomorrow (LSTM)'], [today_close, tomorrow_close_mlp, tomorrow_close_lstm], color=['blue', 'green', 'red'])
        ax.set_ylabel('Price ($)')
        ax.set_title('Today\'s and Predicted Tomorrow\'s Closing Price')
        st.pyplot(fig)
