import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from models import forecast_arima, forecast_prophet

# DB Connection
engine = create_engine("sqlite:///econ_forecast.db")

st.title("📈 Economic Forecasting Dashboard")
st.write("Forecast unemployment, CPI, and GDP using ARIMA & Prophet models.")

# Sidebar selection
indicator = st.sidebar.selectbox("Select Indicator", ["Unemployment Rate", "Consumer Price Index", "Gross Domestic Product"])
model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "Prophet"])
steps = st.sidebar.slider("Forecast Horizon (months)", 6, 24, 12)

# Load data
query = f"SELECT * FROM economic_data WHERE indicator='{indicator}'"
df = pd.read_sql(query, engine, parse_dates=["date"])
df = df.sort_values("date")

# Forecast
if model_choice == "ARIMA":
    forecast = forecast_arima(df, steps)
    forecast_df = pd.DataFrame({"date": pd.date_range(df.index[-1], periods=steps+1, freq="M")[1:], "forecast": forecast})
else:
    forecast = forecast_prophet(df, steps)
    forecast_df = forecast[["ds","yhat"]].rename(columns={"ds":"date","yhat":"forecast"})

# Plot
fig = px.line(df, x="date", y="value", title=f"{indicator} Forecast ({model_choice})")
fig.add_scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines", name="Forecast")
st.plotly_chart(fig)

st.dataframe(forecast_df.tail(10))
