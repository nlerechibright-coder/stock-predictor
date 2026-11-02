import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from django.utils import timezone
from core.models import Stock, HistoricalPrice, PredictionRecord

# Load artifacts once at module level (cached in memory)
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_pipeline', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'feature_cols.joblib')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    MODEL_VERSION = "XGBoost_v1" if "XGB" in str(type(model)) else "RandomForest_v1"
except FileNotFoundError as e:
    raise RuntimeError(f"Model artifacts not found. Please run train_model.py first. Error: {e}")

def get_next_business_day(current_date):
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day

def generate_5_day_forecast(stock_ticker='AAPL'):
    print(f"--- Generating 5-Day Forecast for {stock_ticker} ---")
    
    # 1. Get the latest historical data (need at least 35 days for MA_30 and Lag_3)
    stock = Stock.objects.get(ticker=stock_ticker)
    history = HistoricalPrice.objects.filter(stock=stock).order_by('-date')[:40]
    
    if len(history) < 35:
        raise ValueError("Insufficient historical data. Need at least 35 days.")
    
    # Reverse to chronological order
    history = list(reversed(history))
    
    # 2. Create a working dataframe
    df = pd.DataFrame([{
        'Date': h.date,
        'Open': h.open_price,
        'High': h.high,
        'Low': h.low,
        'Close': h.close,
        'Volume': h.volume
    } for h in history])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.reset_index(drop=True)
    
    last_known_date = df['Date'].iloc[-1]
    last_known_close = df['Close'].iloc[-1]
    avg_volume = int(df['Volume'].tail(5).mean())
    
    predictions = []
    
    # 3. Autoregressive Forecasting Loop (5 days)
    for day in range(1, 6):
        # Calculate features for the current state of the dataframe
        df['Return'] = df['Close'].pct_change()
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['Volatility_7'] = df['Return'].rolling(window=7).std()
        df['Lag_1'] = df['Close'].shift(1)
        df['Lag_2'] = df['Close'].shift(2)
        df['Lag_3'] = df['Close'].shift(3)
        
        # Get the last row's features
        last_row = df.iloc[-1][feature_cols].to_frame().T
        
        # Scale and predict
        last_row_scaled = scaler.transform(last_row)
        next_close_pred = float(model.predict(last_row_scaled)[0])
        
        # Determine next business day
        next_date = get_next_business_day(last_known_date) if day == 1 else get_next_business_day(predictions[-1]['date_obj'])
        
        # Store prediction
        predictions.append({
            'date_obj': next_date,
            'date_str': next_date.strftime('%Y-%m-%d'),
            'close': round(next_close_pred, 2)
        })
        
        # Append predicted day to dataframe for the next iteration's feature calculation
        # We use the predicted close as Open/High/Low and average volume to maintain feature continuity
        new_row = {
            'Date': next_date,
            'Open': next_close_pred,
            'High': next_close_pred,
            'Low': next_close_pred,
            'Close': next_close_pred,
            'Volume': avg_volume
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        last_known_date = next_date

    # 4. Save to Database
    record = PredictionRecord.objects.create(
        stock=stock,
        model_version=MODEL_VERSION,
        day_1_close=predictions[0]['close'],
        day_2_close=predictions[1]['close'],
        day_3_close=predictions[2]['close'],
        day_4_close=predictions[3]['close'],
        day_5_close=predictions[4]['close'],
        mae_at_time=2.50 # Placeholder: In a real scenario, you'd calculate this from the validation set
    )
    
    print(f"✅ Forecast generated and saved. ID: {record.id}")
    return record, predictions, last_known_close
