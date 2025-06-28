# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Forecast Comparison: Prophet vs ARIMA", layout="wide")
st.title("📈 Forecast Comparison App: Prophet vs ARIMA")

# Upload data
file = st.file_uploader("Upload your CSV file", type=["csv"])
if file:
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Target column selection
    target_col = st.selectbox("Select the target column for forecasting:", [col for col in df.columns if col != 'date'])

    # Group by date and clean
    ts_df = df[['date', target_col]].copy().groupby('date').mean().reset_index()
    ts_df = ts_df.rename(columns={'date': 'ds', target_col: 'y'})

    # Remove outliers and apply smoothing
    q_low = ts_df['y'].quantile(0.01)
    q_high = ts_df['y'].quantile(0.99)
    ts_df = ts_df[(ts_df['y'] >= q_low) & (ts_df['y'] <= q_high)]
    ts_df['y'] = ts_df['y'].rolling(window=3, center=True).mean()
    ts_df = ts_df.dropna()
    ts_df['ds'] = pd.to_datetime(ts_df['ds']).dt.tz_localize(None)

    st.subheader("📊 Prophet Forecast")
    with st.spinner("Training Prophet model..."):
        prophet_df = ts_df.copy()
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=26, freq='W')
        forecast = model.predict(future)
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    st.subheader("📉 ARIMA Forecast")
    with st.spinner("Training ARIMA model..."):
        arima_series = ts_df.set_index('ds')['y']
        arima_model = ARIMA(arima_series, order=(2, 1, 2))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=26)

        fig2, ax = plt.subplots(figsize=(10, 5))
        arima_series[-100:].plot(ax=ax, label='Historical')
        arima_forecast.index = pd.date_range(start=arima_series.index[-1] + pd.Timedelta(weeks=1), periods=26, freq='W')
        arima_forecast.plot(ax=ax, label='ARIMA Forecast', color='orange')
        ax.set_title("ARIMA Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig2)

    st.subheader("📈 Performance Comparison (Last 26 Points)")
    # Match Prophet forecast to historical range
    merged = pd.merge(ts_df, forecast[['ds', 'yhat']], on='ds', how='inner')
    actual = merged['y'].values
    pred_prophet = merged['yhat'].values

# ARIMA prediction is only for future steps; compare against ARIMA in-sample prediction
    arima_fitted = arima_result.fittedvalues
    actual_arima = arima_series[-len(arima_fitted):]
    pred_arima = arima_fitted.values


    def metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    prophet_metrics = metrics(actual, pred_prophet)
    arima_metrics = metrics(actual, pred_arima)

    st.write("**Prophet Metrics:**", prophet_metrics)
    st.write("**ARIMA Metrics:**", arima_metrics)

    best_model = "Prophet" if prophet_metrics['R2'] > arima_metrics['R2'] else "ARIMA"
    st.success(f"✅ Best Performing Model (based on R²): {best_model}")

    st.subheader("🔍 Scenario Analysis")

    scenario_type = st.selectbox("Select scenario:", ["No Change", "Weekly Growth (+5%)", "Weekly Drop (-5%)"])
    future_weeks = 26

    if scenario_type == "Weekly Growth (+5%)":
        adjustment = np.array([(1.05) ** i for i in range(future_weeks)])
    elif scenario_type == "Weekly Drop (-5%)":
        adjustment = np.array([(0.95) ** i for i in range(future_weeks)])
    else:
        adjustment = np.ones(future_weeks)

    forecast_scenario = forecast.tail(future_weeks).copy()
    forecast_scenario['yhat_adjusted'] = forecast_scenario['yhat'].values * adjustment

    # Plot scenario
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(forecast_scenario['ds'], forecast_scenario['yhat'], label='Original Forecast')
    ax3.plot(forecast_scenario['ds'], forecast_scenario['yhat_adjusted'], label='Adjusted Scenario', linestyle='--')
    ax3.set_title(f"📈 Scenario Forecast: {scenario_type}")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
