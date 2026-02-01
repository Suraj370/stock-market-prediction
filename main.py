import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from dotenv import load_dotenv
from polygon import RESTClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================================================
# 1. Load environment variables
# =========================================================
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ POLYGON_API_KEY not found in .env")

# =========================================================
# 2. Fetch NVDA data using Polygon SDK
# =========================================================
TICKER = "MSFT"
START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

print(f"\n📥 Fetching {TICKER} data from {START_DATE} to {END_DATE}...")

client = RESTClient(API_KEY)

aggs = client.get_aggs(
    ticker=TICKER,
    multiplier=1,
    timespan="day",
    from_=START_DATE,
    to=END_DATE,
    adjusted=True,
    sort="asc",
    limit=50000
)

records = []
for bar in aggs:
    records.append({
        "date": datetime.fromtimestamp(bar.timestamp / 1000),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    })

df = pl.DataFrame(records)
print(f"✅ Loaded {len(df)} rows")

# =========================================================
# 3. Feature engineering
# =========================================================
df = df.with_columns([
    pl.col("close").pct_change().alias("return"),
    pl.col("close").rolling_mean(5).alias("ma_5"),
    pl.col("close").rolling_mean(20).alias("ma_20"),
    pl.col("close").rolling_std(20).alias("volatility_20"),
    pl.col("volume").rolling_mean(5).alias("vol_ma_5"),
])

# Target = next day's close
df = df.with_columns(
    pl.col("close").shift(-1).alias("target")
)

df = df.drop_nulls()

# =========================================================
# 4. Train / test split (time-series safe)
# =========================================================
FEATURES = ["return", "ma_5", "ma_20", "volatility_20", "vol_ma_5"]

X = df.select(FEATURES).to_numpy()
y = df.select("target").to_numpy().ravel()
dates = df.select("date").to_numpy().ravel()

split_idx = int(len(df) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

# =========================================================
# 5. Train model
# =========================================================
print("\n🌲 Training Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================================================
# 6. Evaluate model
# =========================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Evaluation")
print("=" * 40)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print("=" * 40)

# =========================================================
# 7. Predict next trading day close
# =========================================================
latest_features = df.select(FEATURES).tail(1).to_numpy()
next_close = model.predict(latest_features)[0]

print("\n🔮 Next-Day Close Price Prediction")
print("=" * 40)
print(f"Predicted Close Price: ${next_close:.2f}")
print("=" * 40)
# =========================================================
# 8. Plot Actual vs Predicted Close Prices
# =========================================================
plt.figure()
plt.plot(dates_test, y_test, label="Actual Close")
plt.plot(dates_test, y_pred, label="Predicted Close")
plt.title("NVDA Actual vs Predicted Close Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


