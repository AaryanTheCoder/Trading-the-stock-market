#Same as stage 2 but with SMA features added to the observation space for the RL agent to learn from. 
# SMA -- Simple Moving Average -- is a common technical indicator used in trading to smooth out price data and identify trends over a specific period. By adding SMA features, the RL agent can better understand market trends and make more informed trading decisions.
# 20 day and 50 day SMAs are commonly used to identify short-term and long-term trends, respectively. The agent can use these features to determine whether the market is in an uptrend or downtrend, which can help it decide when to enter or exit trades.

import gymnasium as gym
import gym_trading_env
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO

print("🚀 Step 1: Downloading Apple stock data...")
# Downloading 2020 to 2025 data
df = yf.download("AAPL", start="2020-01-01", end="2025-12-31", multi_level_index=False)
df.columns = df.columns.str.lower()

# --- ADDING THE NEW SMA FEATURES ---
# 1. Daily percentage change (what you had before)
df["feature_close_pct"] = df["close"].pct_change() 

# 2. Short-term Trend: Current price divided by the 20-day moving average
df["feature_sma_20"] = df["close"] / df["close"].rolling(window=20).mean()

# 3. Long-term Trend: Current price divided by the 50-day moving average
df["feature_sma_50"] = df["close"] / df["close"].rolling(window=50).mean()

# Drop rows with NaN values (the first 50 days won't have a 50-day average yet!)
df.dropna(inplace=True) 

print("📊 Step 2: Creating the custom trading environment...")
env = gym.make(
    "TradingEnv",
    df=df,
    positions=[-1, 0, 1], # Short, Cash, Long
    trading_fees=0.01     # 1% execution fee
)

print("🧠 Step 3: Initializing the PPO Brain...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("🏋️‍♂️ Step 4: Training the AI with 200,000 steps...")
# Bumping it up to 200k steps so it has time to digest the new features
model.learn(total_timesteps=200000)

print("💾 Step 5: Saving the upgraded AI model...")
model.save("ppo_apple_trader_v2")
print("🎉 Success! Your upgraded AI is saved as 'ppo_apple_trader_v2.zip'.")