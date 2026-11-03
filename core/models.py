
import yfinance as yf
import pandas as pd
from datetime import datetime
from core.models import Stock, HistoricalPrice

print("Starting seeding process...")
stock, _ = Stock.objects.get_or_create(ticker='AAPL', defaults={'name': 'Apple Inc.', 'exchange': 'NASDAQ'})

print("Downloading data...")
data = yf.download('AAPL', period='6mo', interval='1d', progress=False)

# Clear existing data to avoid duplicates
HistoricalPrice.objects.filter(stock=stock).delete()

count = 0
for date, row in data.iterrows():
    try:
        # Safely extract values, handling potential pandas Series issues
        o = float(row['Open'].iloc[0] if isinstance(row['Open'], pd.Series) else row['Open'])
        h = float(row['High'].iloc[0] if isinstance(row['High'], pd.Series) else row['High'])
        l = float(row['Low'].iloc[0] if isinstance(row['Low'], pd.Series) else row['Low'])
        c = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
        v = float(row['Volume'].iloc[0] if isinstance(row['Volume'], pd.Series) else row['Volume'])
        
        # Convert pandas timestamp to Python date object
        date_obj = date.date() if isinstance(date, pd.Timestamp) else datetime.strptime(str(date), '%Y-%m-%d').date()
        
        HistoricalPrice.objects.create(
            stock=stock, 
            date=date_obj, 
            open_price=o,  # <-- FIXED: Matches your models.py exactly
            high=h, 
            low=l, 
            close=c, 
            volume=int(v) if pd.notna(v) else 0
        )
        count += 1
    except Exception as e:
        print(f"Skipping {date}: {e}")

print(f"✅ Successfully seeded {count} days of data into Neon!")
