import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ARIMA model
def forecast_arima(df, steps=12):
    model = ARIMA(df["value"], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Prophet model
def forecast_prophet(df, periods=12):
    df = df.reset_index().rename(columns={"date":"ds","value":"y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    return forecast
