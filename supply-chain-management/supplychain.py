import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("📦 Supply Chain Demand Forecasting with Stacking Regressor")

# Sidebar upload
st.sidebar.header("📂 Upload CSV")
file = st.sidebar.file_uploader("Upload your supply_chain_data.csv", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df.drop(columns=["SKU"], inplace=True)
    df.dropna(inplace=True)

    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())

    # Feature Engineering
    df["Profit_Margin"] = df["Revenue_generated"] - df["Manufacturing_costs"] - df["Shipping_costs"]
    df["Cost_per_Unit"] = df["Manufacturing_costs"] / (df["Order_quantities"] + 1)

    # Encode categorical
    cat_cols = df.select_dtypes(include="object").columns
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Split and scale
    X = df.drop(columns=["Number_of_products_sold"])
    y = df["Number_of_products_sold"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define Stacking Regressor
    base_models = [
        ('ridge', Ridge()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    final_estimator = HistGradientBoostingRegressor(random_state=42)
    model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Units Sold", int(df["Number_of_products_sold"].sum()))
    col2.metric("💰 Total Revenue", f"${df['Revenue_generated'].sum():,.2f}")
    col3.metric("🔧 Avg Defect Rate", f"{df['Defect_rates'].mean():.2f} %")

    st.subheader("📈 Model Performance")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")

    # 🔍 True vs Predicted
    st.subheader("🧠 True vs Predicted Sales")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=preds, ax=ax1)
    ax1.set_xlabel("True Sales")
    ax1.set_ylabel("Predicted Sales")
    ax1.set_title("True vs Predicted")
    st.pyplot(fig1)

    # 📊 Distribution of Sales
    st.subheader("📊 Distribution of Number of Products Sold")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Number_of_products_sold"], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    # 💹 Revenue vs Manufacturing Costs
    st.subheader("💹 Revenue vs Manufacturing Costs")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df["Manufacturing_costs"], y=df["Revenue_generated"], ax=ax3)
    ax3.set_xlabel("Manufacturing Costs")
    ax3.set_ylabel("Revenue Generated")
    st.pyplot(fig3)

    # 📦 Stock Levels by Product Type
    st.subheader("📦 Stock Levels by Product Type")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=df["Product_type"], y=df["Stock_levels"], ax=ax4)
    st.pyplot(fig4)

    # 🚛 Shipping Costs by Carrier
    st.subheader("🚛 Shipping Costs by Carrier")
    fig5, ax5 = plt.subplots()
    sns.barplot(x=df["Shipping_carriers"], y=df["Shipping_costs"], estimator=np.mean, ax=ax5)
    ax5.set_title("Avg Shipping Cost by Carrier")
    st.pyplot(fig5)

    # 🌍 Sales by Location
    st.subheader("🌍 Sales by Location")
    fig6, ax6 = plt.subplots()
    sns.barplot(x=df["Location"], y=df["Number_of_products_sold"], estimator=np.sum, ax=ax6)
    st.pyplot(fig6)

else:
    st.info("👈 Upload a valid `supply_chain_data.csv` file to get started.")
