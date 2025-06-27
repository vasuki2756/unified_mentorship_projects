# 📘 COFFEE SALES STREAMLIT APP

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error
)

# App Title
st.title("☕ Coffee Sales Prediction App")

# Load Data
data = pd.read_csv("coffee_sales.csv")
data.columns = data.columns.str.strip().str.lower()
data["money"] = data["money"].fillna(data["money"].median())
data["cash_type"] = data["cash_type"].fillna("Unknown")
data["coffee_name"] = data["coffee_name"].fillna("Unknown")
data.drop(columns=["card"], inplace=True)
data["date"] = pd.to_datetime(data["date"])
data["datetime"] = pd.to_datetime(data["datetime"])
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_year"] = data["date"].dt.dayofyear
data["hour"] = data["datetime"].dt.hour

# Encode Categories
le_cash = LabelEncoder()
le_coffee = LabelEncoder()
data["cash_type_enc"] = le_cash.fit_transform(data["cash_type"])
data["coffee_name_enc"] = le_coffee.fit_transform(data["coffee_name"])

X = data[["cash_type_enc", "coffee_name_enc", "day_of_week", "day_of_year", "hour"]]
y = data["money"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
model = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
model.fit(X_scaled, y)

# Sidebar Inputs
st.sidebar.header("Input Parameters")

cash_input = st.sidebar.selectbox(
    "Payment Type", le_cash.classes_
)
coffee_input = st.sidebar.selectbox(
    "Coffee Type", le_coffee.classes_
)
day_of_week_input = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
day_of_year_input = st.sidebar.slider("Day of Year", 1, 365, 100)
hour_input = st.sidebar.slider("Hour of Day", 0, 23, 10)

# Encode & Scale Input
input_df = pd.DataFrame({
    "cash_type_enc": [le_cash.transform([cash_input])[0]],
    "coffee_name_enc": [le_coffee.transform([coffee_input])[0]],
    "day_of_week": [day_of_week_input],
    "day_of_year": [day_of_year_input],
    "hour": [hour_input]
})

input_scaled = scaler.transform(input_df)

# Predict
predicted = model.predict(input_scaled)[0]
st.write(f"### 💰 Predicted Sale: **{predicted:.2f}**")

# Evaluation Metrics
y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mape = mean_absolute_percentage_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.subheader("Model Evaluation")
st.write(f"- MAE: {mae:.2f}")
st.write(f"- RMSE: {rmse:.2f}")
st.write(f"- MAPE: {mape*100:.2f}%")
st.write(f"- R²: {r2:.2f}")

# Plots
st.subheader("Monthly Sales Trend")
monthly = data.groupby(data["date"].dt.to_period("M"))["money"].mean().reset_index()
monthly["date"] = monthly["date"].astype(str)
st.line_chart(monthly.set_index("date"))
