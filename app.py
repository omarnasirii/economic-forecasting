import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from models import forecast_arima, forecast_prophet
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Economic Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border: 1px solid #e1e5e9;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.stAlert {
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data from SQLite database with caching"""
    try:
        engine = create_engine("sqlite:///econ_forecast.db")
        
        with engine.connect() as conn:
            # Check if table exists and has data
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='economic_data'"))
            if not result.fetchone():
                st.error("Database table 'economic_data' not found. Please run fetch_data.py first.")
                return pd.DataFrame()
            
            # Load all data
            df = pd.read_sql("SELECT * FROM economic_data ORDER BY date", conn, parse_dates=['date'])
            
            if df.empty:
                st.error("No data found in database. Please run fetch_data.py first.")
                return pd.DataFrame()
            
            return df
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_forecast_plot(historical_df, forecast_df, indicator_name, model_name):
    """Create interactive forecast plot with confidence intervals"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['value'],
        mode='lines',
        name='Historical Data',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines',
        name=f'{model_name} Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Confidence intervals if available
    if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_ci'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_ci'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255,127,14,0.2)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{indicator_name} Forecast ({model_name})',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # Title and description
    st.title("📈 Economic Forecasting Dashboard")
    st.markdown("Forecast key economic indicators using ARIMA and Prophet models with real-time FRED data.")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please ensure you have run `python fetch_data.py` to fetch data from FRED.")
        st.info("📝 **Setup Instructions:**\n1. Create a `.env` file with your FRED API key\n2. Run `python fetch_data.py` to fetch data\n3. Refresh this dashboard")
        return
    
    # Sidebar configuration
    st.sidebar.header("📊 Forecast Configuration")
    
    # Get available indicators
    available_indicators = df['indicator'].unique()
    
    if len(available_indicators) == 0:
        st.error("No indicators found in the database.")
        return
    
    # User selections
    selected_indicator = st.sidebar.selectbox(
        "Select Economic Indicator",
        available_indicators,
        help="Choose the economic indicator to forecast"
    )
    
    model_choice = st.sidebar.selectbox(
        "Select Forecasting Model",
        ["ARIMA", "Prophet", "Both"],
        help="Choose the forecasting model to use"
    )
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (months)",
        min_value=3,
        max_value=24,
        value=12,
        help="Number of months to forecast into the future"
    )
    
    # Filter data for selected indicator
    indicator_data = df[df['indicator'] == selected_indicator].copy()
    indicator_data = indicator_data.sort_values('date')
    
    if len(indicator_data) < 10:
        st.error(f"Not enough data for {selected_indicator}. Need at least 10 data points.")
        return
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Data Points", len(indicator_data))
    
    with col2:
        start_date = indicator_data['date'].min().strftime('%Y-%m-%d')
        st.metric("🗓️ Start Date", start_date)
    
    with col3:
        end_date = indicator_data['date'].max().strftime('%Y-%m-%d')
        st.metric("🗓️ End Date", end_date)
    
    with col4:
        latest_value = indicator_data['value'].iloc[-1]
        st.metric("📈 Latest Value", f"{latest_value:.2f}")
    
    # Generate forecasts
    st.header("🔮 Forecasting Results")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if model_choice in ["ARIMA", "Both"]:
        status_text.text("Running ARIMA forecast...")
        progress_bar.progress(25)
        
        try:
            arima_forecast, arima_model = forecast_arima(indicator_data, steps=forecast_horizon)
            
            # Display ARIMA results
            st.subheader("🎯 ARIMA Forecast")
            
            # Create and display plot
            arima_fig = create_forecast_plot(indicator_data, arima_forecast, selected_indicator, "ARIMA")
            st.plotly_chart(arima_fig, use_container_width=True)
            
            # Display forecast table
            with st.expander("📋 ARIMA Forecast Data"):
                st.dataframe(arima_forecast[['date', 'forecast', 'lower_ci', 'upper_ci']].round(4))
            
            progress_bar.progress(50)
            
        except Exception as e:
            st.error(f"ARIMA model failed: {str(e)}")
    
    if model_choice in ["Prophet", "Both"]:
        status_text.text("Running Prophet forecast...")
        progress_bar.progress(75)
        
        try:
            prophet_forecast, prophet_model = forecast_prophet(indicator_data, periods=forecast_horizon)
            
            # Display Prophet results
            st.subheader("🔮 Prophet Forecast")
            
            # Create and display plot
            prophet_fig = create_forecast_plot(indicator_data, prophet_forecast, selected_indicator, "Prophet")
            st.plotly_chart(prophet_fig, use_container_width=True)
            
            # Display forecast table
            with st.expander("📋 Prophet Forecast Data"):
                st.dataframe(prophet_forecast[['date', 'forecast', 'lower_ci', 'upper_ci']].round(4))
            
        except Exception as e:
            st.error(f"Prophet model failed: {str(e)}")
    
    progress_bar.progress(100)
    status_text.text("✅ Forecasting complete!")
    
    # Model comparison if both models were run
    if model_choice == "Both":
        st.header("⚖️ Model Comparison")
        
        try:
            # Create comparison plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=indicator_data['date'],
                y=indicator_data['value'],
                mode='lines',
                name='Historical Data',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # ARIMA forecast
            if 'arima_forecast' in locals():
                fig.add_trace(go.Scatter(
                    x=arima_forecast['date'],
                    y=arima_forecast['forecast'],
                    mode='lines',
                    name='ARIMA Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
            
            # Prophet forecast
            if 'prophet_forecast' in locals():
                fig.add_trace(go.Scatter(
                    x=prophet_forecast['date'],
                    y=prophet_forecast['forecast'],
                    mode='lines',
                    name='Prophet Forecast',
                    line=dict(color='#2ca02c', width=2, dash='dot')
                ))
            
            fig.update_layout(
                title=f'{selected_indicator} - Model Comparison',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating comparison plot: {str(e)}")
    
    # Historical data visualization
    st.header("📊 Historical Data Analysis")
    
    # Time series plot
    historical_fig = px.line(
        indicator_data, 
        x='date', 
        y='value',
        title=f'{selected_indicator} - Historical Trend'
    )
    historical_fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(historical_fig, use_container_width=True)
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Summary Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                indicator_data['value'].mean(),
                indicator_data['value'].median(),
                indicator_data['value'].std(),
                indicator_data['value'].min(),
                indicator_data['value'].max()
            ]
        })
        st.dataframe(stats_df.round(4))
    
    with col2:
        st.subheader("📊 Recent Trend")
        recent_data = indicator_data.tail(12)
        trend_change = recent_data['value'].iloc[-1] - recent_data['value'].iloc[0]
        trend_pct = (trend_change / recent_data['value'].iloc[0]) * 100
        
        st.metric(
            "12-Month Change", 
            f"{trend_change:.2f}",
            f"{trend_pct:.2f}%"
        )
        
        # Show recent data
        st.dataframe(recent_data[['date', 'value']].tail(5))

if __name__ == "__main__":
    main()