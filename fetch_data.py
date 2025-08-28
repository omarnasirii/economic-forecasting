import os
import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_database():
    """Initialize database and create tables if they don't exist"""
    engine = create_engine("sqlite:///econ_forecast.db")
    
    # Create table with proper schema
    create_table_query = """
    CREATE TABLE IF NOT EXISTS economic_data (
        date TEXT,
        value REAL,
        indicator TEXT,
        indicator_code TEXT,
        PRIMARY KEY (date, indicator_code)
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()
    
    return engine

def fetch_and_store():
    """Fetch economic data from FRED and store in SQLite database"""
    
    # Check for API key
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not found in .env file")
        return False
    
    try:
        fred = Fred(api_key=api_key)
        engine = setup_database()
        
        # Economic indicators to fetch
        indicators = {
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index", 
            "GDP": "Gross Domestic Product",
            "FEDFUNDS": "Federal Funds Rate",
            "PAYEMS": "Total Nonfarm Payrolls"
        }
        
        for code, name in indicators.items():
            try:
                logger.info(f"Fetching {name} ({code})...")
                
                # Fetch data from FRED
                series = fred.get_series(code, limit=1000)  # Limit to recent 1000 observations
                
                if series.empty:
                    logger.warning(f"No data found for {code}")
                    continue
                
                # Prepare DataFrame
                df = pd.DataFrame({
                    'date': series.index.strftime('%Y-%m-%d'),
                    'value': series.values,
                    'indicator': name,
                    'indicator_code': code
                })
                
                # Remove any NaN values
                df = df.dropna()
                
                # Clear existing data for this indicator
                with engine.connect() as conn:
                    conn.execute(text("DELETE FROM economic_data WHERE indicator_code = :code"), 
                               {"code": code})
                    conn.commit()
                
                # Store in database
                df.to_sql("economic_data", engine, if_exists="append", index=False)
                logger.info(f"Stored {len(df)} records for {name}")
                
            except Exception as e:
                logger.error(f"Error fetching {code}: {str(e)}")
                continue
        
        logger.info("Data fetching completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database setup error: {str(e)}")
        return False

def verify_data():
    """Verify data was stored correctly"""
    try:
        engine = create_engine("sqlite:///econ_forecast.db")
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT indicator, COUNT(*) as count FROM economic_data GROUP BY indicator"))
            records = result.fetchall()
            
            print("\nData verification:")
            print("-" * 40)
            for record in records:
                print(f"{record[0]}: {record[1]} records")
            
            # Show sample data
            sample = conn.execute(text("SELECT * FROM economic_data LIMIT 5"))
            sample_data = sample.fetchall()
            
            print("\nSample data:")
            print("-" * 40)
            for row in sample_data:
                print(f"Date: {row[0]}, Value: {row[1]}, Indicator: {row[2]}")
                
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")

if __name__ == "__main__":
    print("Starting data fetch process...")
    success = fetch_and_store()
    
    if success:
        verify_data()
        print("\n✅ Data fetching completed successfully!")
    else:
        print("\n❌ Data fetching failed. Please check your FRED API key and try again.")