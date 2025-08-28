import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def prepare_data_for_arima(df):
    """Prepare data for ARIMA model"""
    # Ensure data is sorted by date
    df = df.sort_values('date')
    
    # Set date as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Handle missing values
    df['value'] = df['value'].interpolate(method='linear')
    
    return df

def check_stationarity(timeseries):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary

def find_best_arima_order(data, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA order using AIC"""
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order

def forecast_arima(df, steps=12):
    """
    Forecast using ARIMA model with automatic order selection
    """
    try:
        # Prepare data
        df_processed = prepare_data_for_arima(df.copy())
        
        if len(df_processed) < 10:
            raise ValueError("Not enough data points for ARIMA modeling")
        
        # Find best ARIMA order
        best_order = find_best_arima_order(df_processed['value'])
        
        # Fit ARIMA model
        model = ARIMA(df_processed['value'], order=best_order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast_result = model_fit.forecast(steps=steps)
        
        # Create forecast dates
        last_date = df_processed.index[-1]
        if df_processed.index.freq is None:
            # Infer frequency from the data
            freq = pd.infer_freq(df_processed.index)
            if freq is None:
                freq = 'MS'  # Default to month start
        else:
            freq = df_processed.index.freq
            
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=steps + 1, 
            freq=freq
        )[1:]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_result,
            'model': 'ARIMA'
        })
        
        # Get confidence intervals
        conf_int = model_fit.get_forecast(steps=steps).conf_int()
        forecast_df['lower_ci'] = conf_int.iloc[:, 0]
        forecast_df['upper_ci'] = conf_int.iloc[:, 1]
        
        return forecast_df, model_fit
        
    except Exception as e:
        print(f"ARIMA Error: {str(e)}")
        # Return dummy forecast if error occurs
        last_date = pd.to_datetime(df['date']).max()
        forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        last_value = df['value'].iloc[-1]
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [last_value] * steps,
            'model': 'ARIMA (fallback)',
            'lower_ci': [last_value * 0.95] * steps,
            'upper_ci': [last_value * 1.05] * steps
        }), None

def forecast_prophet(df, periods=12):
    """
    Forecast using Prophet model
    """
    try:
        # Prepare data for Prophet
        prophet_df = df.copy()
        prophet_df['date'] = pd.to_datetime(prophet_df['date'])
        prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
        prophet_df = prophet_df.sort_values('ds')
        
        # Remove any NaN values
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 10:
            raise ValueError("Not enough data points for Prophet modeling")
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='MS')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract only the future predictions
        forecast_future = forecast.tail(periods).copy()
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_future['ds'],
            'forecast': forecast_future['yhat'],
            'model': 'Prophet',
            'lower_ci': forecast_future['yhat_lower'],
            'upper_ci': forecast_future['yhat_upper']
        })
        
        return forecast_df, model
        
    except Exception as e:
        print(f"Prophet Error: {str(e)}")
        # Return dummy forecast if error occurs
        last_date = pd.to_datetime(df['date']).max()
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
        last_value = df['value'].iloc[-1]
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [last_value] * periods,
            'model': 'Prophet (fallback)',
            'lower_ci': [last_value * 0.95] * periods,
            'upper_ci': [last_value * 1.05] * periods
        }), None

def get_model_metrics(actual, predicted):
    """Calculate model performance metrics"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE': round(mape, 2)
    }