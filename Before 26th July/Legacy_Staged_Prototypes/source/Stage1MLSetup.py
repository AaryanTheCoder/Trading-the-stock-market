# The RL code sets up the environment for a reinforcement learning model to make market decisions, and learn 
# This code will instead use Machine Learning to predict the next day's price movement, SIMPLY UP OR DOWN NO MORE 
# NEED TO START SIMPLE!!! TO UNDERSTAND AND BUILD COMPLEXITY LATER ON!!

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# STEP 1: DOWNLOAD APPLE HISTORICAL DATA
# ==========================================
print("Fetching Apple (AAPL) market data via yfinance...")
# We fetch from late 2018 to calculate technical indicators cleanly for Jan 1, 2019
raw_data = yf.download("AAPL", start="2018-11-01", end="2025-12-31")

# Flatting multi-level columns if returned by newer yfinance versions
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

df = raw_data.copy()

# ==========================================
# STEP 2: FEATURE ENGINEERING (Pattern Creation)
# ==========================================
# Target: Did the market go UP next day? (1 = Up, 0 = Down/Flat)
df['Next_Day_Close'] = df['Close'].shift(-1)
df['Target'] = (df['Next_Day_Close'] > df['Close']).astype(int)

# Structural Indicators (The patterns our machine learning model will analyze)
df['Return_1d'] = df['Close'].pct_change()
df['Return_5d'] = df['Close'].pct_change(5)
df['Vol_Change_1d'] = df['Volume'].pct_change()

# Moving Average Ratios (Capturing market momentum trends)
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Close_to_MA10'] = df['Close'] / df['MA10']
df['Close_to_MA50'] = df['Close'] / df['MA50']

# High-Low spread (Volatility indicator)
df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

# Drop all invalid rows caused by lookbacks/shifts
df = df.dropna()

# ==========================================
# STEP 3: SPLIT TRAIN (2019-2024) & SIMULATION (2025)
# ==========================================
# Segment data cleanly by timestamps to avoid look-ahead bias
train_df = df[(df.index >= '2019-01-01') & (df.index <= '2024-12-31')]
sim_df = df[(df.index >= '2025-01-01') & (df.index <= '2025-12-31')]

# Define the features the ML brain uses to recognize structural patterns
features = ['Return_1d', 'Return_5d', 'Vol_Change_1d', 'Close_to_MA10', 'Close_to_MA50', 'HL_Spread']

X_train = train_df[features]
y_train = train_df['Target']

X_sim = sim_df[features]
y_sim = sim_df['Target']

print(f"Training Dataset Range: 2019 - 2024 ({len(X_train)} trading days)")
print(f"Simulation Dataset Range: 2025 ({len(X_sim)} trading days)")

# ==========================================
# STEP 4: TRAINING THE MACHINE LEARNING MODEL
# ==========================================
# Random Forest matches patterns by building 100 localized decision tree structures
model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
model.fit(X_train, y_train)
print("Model training successfully completed.")

# ==========================================
# STEP 5: SIMULATING THE 2025 LIVE MARKET
# ==========================================
# The model attempts to predict the targets across the unseen 2025 simulation space
predictions = model.predict(X_sim)

# Calculate direction accuracy statistics
correct_guesses = np.sum(predictions == y_sim)
total_guesses = len(y_sim)
accuracy_percent = (correct_guesses / total_guesses) * 100

# ==========================================
# STEP 6: PERFORMANCE SUMMARY TERMINAL PRINT
# ==========================================
print("\n" + "="*45)
print("          2025 SIMULATION RESULTS          ")
print("="*45)
print(f"Total Simulation Trading Days : {total_guesses}")
print(f"Successful Direction Guesses   : {correct_guesses}")
print(f"Directional Prediction Accuracy: {accuracy_percent:.2f}%")
print("="*45)

# Benchmarking performance against market realities
if accuracy_percent > 55.0:
    print("Result: Outstanding. The model found an exploitable statistical anomaly.")
elif accuracy_percent >= 50.0:
    print("Result: Normal. The model performs slightly better than an arbitrary coin toss.")
else:
    print("Result: Subpar. The model fell into an overfitting trap or market conditions shifted.")













