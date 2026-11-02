import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("--- Starting Machine Learning Pipeline ---")

# 1. Load Data
data_path = "ml_pipeline/artifacts/aapl_clean.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Loaded {len(df)} records for feature engineering.")

# 2. Feature Engineering
print("Engineering features...")
df['Return'] = df['Close'].pct_change()
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['Volatility_7'] = df['Return'].rolling(window=7).std()
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['Target'] = df['Close'].shift(-1)

df = df.dropna().reset_index(drop=True)
print(f"Records after feature engineering: {len(df)}")

# 3. Define Features and Target
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'MA_7', 'MA_30', 'Volatility_7', 'Lag_1', 'Lag_2', 'Lag_3']
X = df[feature_cols]
y = df['Target']

# 4. Chronological Train/Test Split (NO SHUFFLING)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# 5. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Models
print("Training Random Forest (Baseline)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

print("Training XGBoost (Primary)...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

# 7. Evaluate Models
def evaluate_model(model, X_test_scaled, X_test_df, y_test, model_name):
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Directional Accuracy: Did the model correctly predict if price goes up or down vs previous day?
    prev_close = X_test_df['Lag_1'].values
    actual_up = y_test.values > prev_close
    pred_up = preds > prev_close
    dir_acc = np.mean(actual_up == pred_up) * 100
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    return mae, rmse, dir_acc

rf_mae, rf_rmse, rf_dir = evaluate_model(rf_model, X_test_scaled, X_test, y_test, "Random Forest")
xgb_mae, xgb_rmse, xgb_dir = evaluate_model(xgb_model, X_test_scaled, X_test, y_test, "XGBoost")

# 8. Select Best Model and Save Artifacts
if xgb_mae < rf_mae:
    best_model = xgb_model
    model_name = "XGBoost"
else:
    best_model = rf_model
    model_name = "RandomForest"

print(f"\n✅ Selected Best Model: {model_name}")

os.makedirs("ml_pipeline/artifacts", exist_ok=True)
joblib.dump(best_model, "ml_pipeline/artifacts/best_model.joblib")
joblib.dump(scaler, "ml_pipeline/artifacts/scaler.joblib")
joblib.dump(feature_cols, "ml_pipeline/artifacts/feature_cols.joblib")

print("Artifacts saved to ml_pipeline/artifacts/")
print("✅ Phase 2 Complete: Model trained, evaluated, and saved.")
