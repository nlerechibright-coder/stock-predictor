import yfinance as yf
import pandas as pd
import os

print("--- Starting AAPL Data Acquisition ---")

# 1. Download Data
ticker = "AAPL"
print(f"Downloading 5 years of historical data for {ticker}...")
# We use period='5y' to get a robust dataset for training
data = yf.download(ticker, period="5y", auto_adjust=True)

# Handle yfinance multi-index columns (common in newer versions)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. Save Raw Data
os.makedirs("ml_pipeline/artifacts", exist_ok=True)
raw_path = "ml_pipeline/artifacts/aapl_raw.csv"
data.to_csv(raw_path)
print(f"Raw data saved to {raw_path} ({len(data)} records)")

# 3. Clean and Validate Data
print("\n--- Validating and Cleaning Data ---")
df = data.copy()

# Reset index to make Date a column
df = df.reset_index()

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort chronologically
df = df.sort_values(by='Date')

# Drop any rows with missing values
initial_count = len(df)
df = df.dropna()
dropped_count = initial_count - len(df)
print(f"Dropped {dropped_count} rows with missing values.")

# Remove duplicate dates
df = df.drop_duplicates(subset=['Date'])

# Save Cleaned Data
clean_path = "ml_pipeline/artifacts/aapl_clean.csv"
df.to_csv(clean_path, index=False)

# 4. Data Quality Report
print("\n--- Data Quality Report ---")
print(f"Total Records: {len(df)}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Columns: {list(df.columns)}")
print(f"Latest Close Price: ${df['Close'].iloc[-1]:.2f}")
print(f"Oldest Close Price: ${df['Close'].iloc[0]:.2f}")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\n✅ Phase 1 Complete: Data acquired and validated.")
