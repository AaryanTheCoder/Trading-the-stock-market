# Description of stage 1: 
# Setting up the environement and importing necessary libraries for data processing and analysis.
# Yfinance is used to fetch financial data, 

import gymnasium as gym  # Import the Gymnasium library for RL environments
import gym_trading_env  # Import the custom trading environment package
import yfinance as yf  # Import Yahoo Finance to download stock market data
import pandas as pd  # Import pandas to work with tabular data
import tensorflow as tf  # Import TensorFlow for building and training neural networks
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting and visualization


print("🚀 Step 1: Downloading Apple stock data from Yahoo Finance...")  # Show that the script is starting the data download step
# FIX: Added multi_level_index=False to flatten the table structure
# Download Apple stock data for Apple (AAPL) between the given dates
# The result is stored in a pandas DataFrame called df
# multi_level_index=False makes the data come back in a simpler flat format
# so it is easier to work with in this project
df = yf.download("AAPL", start="2020-01-01", end="2025-12-31", multi_level_index=False)

# Convert all column names to lowercase so they are easier to reference consistently
# Example: 'Close' becomes 'close'
df.columns = df.columns.str.lower()

# Create a new column named feature_close that copies the close price column
# This gives the trading environment a simple feature to use as input for the agent
df["feature_close"] = df["close"]

print("📊 Step 2: Creating the custom trading environment...")  # Show that the environment is being built
# Create the trading environment using Gymnasium
# The environment gets the stock data, allowed actions, and trading fees
# The environment name is TradingEnv, which comes from the imported gym_trading_env package
env = gym.make(
    "TradingEnv",  # Name of the custom trading environment
    df=df,  # Pass the stock data into the environment
    positions=[-1, 0, 1],  # Allowed actions: -1 = short, 0 = hold, 1 = long
    trading_fees=0.01  # Charge a 1% fee for each trade
)

print("🏁 Step 3: Resetting the environment to the first day of trading...")  # Show that the environment is being initialized
# Reset the environment to the starting state so the agent can begin from the first day
# This returns the first observation and some extra info
observation, info = env.reset()

print("\n🧠 Let's look at what the AI actually 'sees' on Day 1:")  # Explain that we are showing the agent's initial input
# Print the shape of the observation so we can see how much data the agent receives
print(f"Observation Shape: {observation.shape}")
# Print the actual observation values that the agent sees at the beginning
print(f"Raw Observation Data:\n{observation}")