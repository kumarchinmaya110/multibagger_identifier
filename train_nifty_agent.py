import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath, train_split=0.8):
    """
    Loads Nifty historical data and preprocesses it.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    print("Preprocessing data...")
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    # Handle missing values if any
    df = df.ffill()

    # Simple normalization/scaling could be added here,
    # but for PPO and typical financial data, we might want to normalize on the fly
    # or pass raw differences. For simplicity, we will pass raw OHLCV and
    # rely on stable-baselines3's VecNormalize later if needed,
    # but for now we keep the actual prices for accurate PnL.

    # Split into train and test
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df

def train_agent():
    from nifty_trading_env import NiftyTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import os

    train_df, test_df = load_and_preprocess_data("nifty_data.csv")

    # Save test data for mock trading
    test_df.to_csv("nifty_test_data.csv", index=False)

    # Create environment
    env = NiftyTradingEnv(df=train_df, window_size=10)

    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    print("Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048, batch_size=64)

    print("Training model (this will run for a short time for demonstration)...")
    # In a real scenario, this would be much higher, e.g., 1_000_000
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)

    model_path = "nifty_ppo_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train_agent()
