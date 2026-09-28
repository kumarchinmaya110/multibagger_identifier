import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from nifty_rl_trader.env.nifty_futures_env import NiftyFuturesEnv
from nifty_rl_trader.features.technical_features import load_config

def evaluate_model():
    config = load_config()
    data_dir = config['settings']['data_paths']['processed_data']
    model_dir = "nifty_rl_trader/saved_models"

    print("Loading test dataset...")
    test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

    def make_env():
        return NiftyFuturesEnv(test_df, config, is_training=False)

    env = DummyVecEnv([make_env])

    # Load VecNormalize statistics
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # Frame stacking
    n_stack = config['settings']['rl_agent']['frame_stack']
    env = VecFrameStack(env, n_stack=n_stack)

    # Load model
    model_path = os.path.join(model_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "ppo_nifty_final.zip")

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Starting evaluation...")
    obs = env.reset()
    done = False

    equities = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        equities.append(infos[0]['equity'])
        done = dones[0]

    # Calculate metrics
    equities = np.array(equities)
    returns = np.diff(equities) / equities[:-1]

    total_return = (equities[-1] - equities[0]) / equities[0]

    # Sharpe (annualized, assuming 375 mins/day, 252 days/year)
    minutes_per_year = 375 * 252
    if np.std(returns) > 0:
        sharpe = np.sqrt(minutes_per_year) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak
    max_dd = np.min(drawdown)

    print("=== Evaluation Results ===")
    print(f"Initial Capital : ₹{equities[0]:,.2f}")
    print(f"Final Capital   : ₹{equities[-1]:,.2f}")
    print(f"Total Return    : {total_return*100:.2f}%")
    print(f"Sharpe Ratio    : {sharpe:.2f}")
    print(f"Max Drawdown    : {max_dd*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
