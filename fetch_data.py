import os
import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# SQLite DB
engine = create_engine("sqlite:///econ_forecast.db")

# Select indicators
indicators = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "GDP": "Gross Domestic Product"
}

def fetch_and_store():
    for code, name in indicators.items():
        print(f"Fetching {name}...")
        series = fred.get_series(code)
        df = pd.DataFrame(series, columns=["value"])
        df.index.name = "date"
        df["indicator"] = name
        df.to_sql("economic_data", engine, if_exists="append")

if __name__ == "__main__":
    fetch_and_store()
    print("Data stored in econ_forecast.db")
