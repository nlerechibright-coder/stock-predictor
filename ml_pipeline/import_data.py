import os
import django
import pandas as pd
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
django.setup()

from core.models import Stock, HistoricalPrice

print("--- Starting Data Import ---")

# 1. Create the AAPL Stock record
stock, created = Stock.objects.get_or_create(
    ticker='AAPL',
    defaults={'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'currency': 'USD'}
)
print(f"Stock record: {'Created' if created else 'Already exists'}")

# 2. Load and import historical prices
data_path = "ml_pipeline/artifacts/aapl_clean.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date']).dt.date

imported_count = 0
for index, row in df.iterrows():
    _, created = HistoricalPrice.objects.update_or_create(
        stock=stock,
        date=row['Date'],
        defaults={
            'open_price': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': int(row['Volume'])
        }
    )
    if created:
        imported_count += 1

print(f"Imported {imported_count} new historical price records.")
print(f"Total records in database: {HistoricalPrice.objects.count()}")
print("✅ Phase 5 Complete: Historical data integrated into Django.")
