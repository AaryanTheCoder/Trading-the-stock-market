import gymnasium as gym
import gym_trading_env
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO

STARTING_CASH = 100000

print("🚀 Step 1: Downloading 2025 Evaluation Data...")
# Note: We pull from late 2024 to make sure the 50-day SMA is already calculated by Jan 1, 2025
df = yf.download("AAPL", start="2024-10-01", end="2025-12-31", multi_level_index=False)
df.columns = df.columns.str.lower()

# Calculate Indicators
df["feature_close_pct"] = df["close"].pct_change() 
df["feature_sma_20"] = df["close"] / df["close"].rolling(window=20).mean()
df["feature_sma_50"] = df["close"] / df["close"].rolling(window=50).mean()
df.dropna(inplace=True)

# Filter out 2024 so the simulator strictly starts on the first trading day of 2025
sim_df = df.loc["2025-01-01":].copy()
print(f"📊 2025 Simulator Ready with {len(sim_df)} trading days.")

print("🧠 Step 2: Loading Trained AI Policy...")
try:
    model = PPO.load("ppo_apple_trader_v3")
except FileNotFoundError:
    print("❌ Error: Could not find 'ppo_apple_trader_v3.zip'. Run the training script first!")
    exit()

print("🎮 Step 3: Setting up the 2025 Live Market Simulator...")
sim_env = gym.make(
    "TradingEnv",
    df=sim_df,
    positions=[-1, 0, 1],
    trading_fees=0.00,
    initial_position=0 # Start in Cash
)

# Reset environment and start loop
obs, info = sim_env.reset()
done = False
truncated = False

print("📈 Step 4: AI is trading through 2025...")
while not (done or truncated):
    # deterministic=True ensures the AI relies strictly on policy, no random exploration
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = sim_env.step(action)

# ==========================================
# 📊 PERFORMANCE LOGS & BALANCE METRICS
# ==========================================
print("\n" + "="*45)
print("🏁 FINAL 2025 SIMULATOR RESULTS")
print("="*45)

# Calculate final portfolio values based on your $100,000 budget
portfolio_multiplier = sim_env.unwrapped.historical_info[-1]['portfolio_valuation'] / sim_env.unwrapped.historical_info[0]['portfolio_valuation']

# FIX: Use 'data_close' to find the pure market/stock price return
market_multiplier = sim_env.unwrapped.historical_info[-1]['data_close'] / sim_env.unwrapped.historical_info[0]['data_close']

final_ai_balance = STARTING_CASH * portfolio_multiplier
final_market_balance = STARTING_CASH * market_multiplier

ai_return_pct = (portfolio_multiplier - 1) * 100
market_return_pct = (market_multiplier - 1) * 100

print(f"💵 Starting Balance:          ${STARTING_CASH:,.2f}")
print(f"🤖 Final AI Trader Balance:   ${final_ai_balance:,.2f} ({ai_return_pct:+.2f}%)")
print(f"🍏 Final Buy & Hold Balance:  ${final_market_balance:,.2f} ({market_return_pct:+.2f}%)")
print("="*45)

if final_ai_balance > final_market_balance:
    print("🏆 OUTPERFORMED! The AI beat a pure buy-and-hold strategy!")
else:
    print("📉 UNDERPERFORMED: Hold outperformed the AI's active trading strategy.")