# 📈 Economic Forecasting with Time Series

An interactive **time series forecasting dashboard** that predicts key U.S. economic indicators (Unemployment, CPI, GDP) using **ARIMA** and **Prophet** models.  
Built with **Python, Streamlit, SQL, and FRED API**.

---

## 🚀 Features
- Fetch real-world macroeconomic data directly from **FRED**.
- Store time series data in a **SQLite database**.
- Forecast indicators using **ARIMA (statsmodels)** and **Prophet (Meta)**.
- Interactive **Streamlit dashboard** with Plotly visualizations.
- Adjustable forecast horizon (6–24 months).

---

## 📂 Project Structure

## ⚙️ Setup Instructions

### 1. Clone Repository
git clone https://github.com/omarnasirii/economic-forecasting.git

cd economic-forecasting

### 2. Create Virtual Environment
python -m venv venv

source venv/bin/activate   # Mac/Linux

.\venv\Scripts\activate    # Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Get a FRED API Key

Sign up here: [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html?utm_source=chatgpt.com)

Create a .env file in the project folder:

FRED_API_KEY=your_api_key_here

### 5. Fetch Economic Data
python fetch_data.py

This will download data and save it into econ_forecast.db (SQLite).

### 6. Run Dashboard
streamlit run app.py

## 📊 Example Dashboard

Select an indicator: Unemployment, CPI, or GDP

Choose model: ARIMA or Prophet

Adjust forecast horizon: 6–24 months

View interactive line charts + latest forecast values.

## 🛡️ Notes

.env and econ_forecast.db are excluded with .gitignore.

Each user generates their own database with fetch_data.py.

## 🔮 Future Improvements

Add seasonality analysis.

Compare multiple forecasting models side by side.

Deploy to Streamlit Cloud for easy sharing.
