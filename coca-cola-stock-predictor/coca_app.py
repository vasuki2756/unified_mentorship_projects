import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load stock history data
data = pd.read_csv("E:\\projects\\coca cola\\data\\Coca-Cola_stock_history.csv")
data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)
data.sort_values('Date', inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)

# Feature Engineering
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

# ML Features and Target
features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits',
            'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
X = data[features]
y = data['Close']

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
latest_input = X.iloc[[-1]]
latest_prediction = model.predict(latest_input)

# Load static financial info
info = pd.read_csv("E:\\projects\\coca cola\\data\\Coca-Cola_stock_info.csv").dropna().reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="Coca-Cola Stock Prediction", layout="wide")
st.title("📈 Coca-Cola Stock Price Prediction Dashboard")

st.subheader("📊 Closing Price & Moving Averages")
st.line_chart(data.set_index('Date')[['Close', 'MA_20', 'MA_50']])

st.subheader("🔮 Predicted Closing Price (Most Recent Day)")
st.metric(label="Predicted Close Price", value=f"${latest_prediction[0]:.2f}")

st.sidebar.title("🏢 Company Financial Overview")
important_keys = [
    'dividendYield', 'returnOnEquity', 'marketCap', 'payoutRatio',
    'operatingMargins', 'grossMargins', 'profitMargins'
]

for key in important_keys:
    row = info[info['Key'] == key]
    if not row.empty:
        st.sidebar.write(f"**{key}**: {row['Value'].values[0]}")

with st.expander("📄 Show Raw Stock Data"):
    st.dataframe(data.tail(100))

with st.expander("📄 Show Full Company Info"):
    st.dataframe(info)
