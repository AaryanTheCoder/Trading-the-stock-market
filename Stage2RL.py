#Now the AI Actually makes the decision, for all days of simulation 
# Awarded + points for good decision, - for bad
# by end = Good Policy!

import gymnasium as gym
import gym_trading_env
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO

print("🚀 Step 1: Downloading Apple stock data...")
df = yf.download("AAPL", start="2020-01-01", end="2025-12-31", multi_level_index=False)
df.columns = df.columns.str.lower()

# Calculate daily percentage change so the AI can read market movements cleanly
df["feature_close"] = df["close"].pct_change() 
df.dropna(inplace=True) # Remove the very first day since it doesn't have a "yesterday" to compare to

print("📊 Step 2: Creating the custom trading environment...")
env = gym.make(
    "TradingEnv",
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.01     # Enforcing your 1% transaction fee rule!
)

print("🧠 Step 3: Initializing the PPO Brain...")
# "MlpPolicy" means a standard neural network built for tabular data (rows and columns)
# We use the CPU here because for small charts, it's faster than sending tiny bits of data to the GPU!
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("🏋️‍♂️ Step 4: Training the AI (Playing the trading game)...")
# We tell the AI to take 20,000 steps (trading days) to learn your rules
model.learn(total_timesteps=75000)

print("💾 Step 5: Saving the trained AI model...")
# Save its brain weights to a file so we can load it later without retraining it
model.save("ppo_apple_trader")
print("🎉 Success! Your AI is trained and saved as 'ppo_apple_trader.zip'.")




























































