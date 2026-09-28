import argparse
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from nifty_rl_trader.features.technical_features import load_config
from nifty_rl_trader.env.nifty_futures_env import NiftyFuturesEnv
from nifty_rl_trader.paper_trade.excel_logger import ExcelLogger
from nifty_rl_trader.paper_trade.live_feed import LiveFeedPoller
from nifty_rl_trader.paper_trade.broker_mock import MockBroker
from nifty_rl_trader.models.train import train_agent
from nifty_rl_trader.models.evaluate import evaluate_model

def run_paper_trading():
    config = load_config()
    model_dir = "nifty_rl_trader/saved_models"
    data_path = os.path.join(config['settings']['data_paths']['processed_data'], "test.parquet")

    print("Initializing Paper Trading Engine...")
    logger = ExcelLogger("nifty_rl_trader/trades_log.xlsx")
    feed = LiveFeedPoller(data_path)
    broker = MockBroker(config, logger)

    # Setup dummy env purely for observation preprocessing using stable baselines wrappers
    dummy_df = pd.read_parquet(data_path).iloc[:1] # Just for initialization
    def make_env():
        return NiftyFuturesEnv(dummy_df, config, is_training=False)

    env = DummyVecEnv([make_env])

    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    n_stack = config['settings']['rl_agent']['frame_stack']
    env = VecFrameStack(env, n_stack=n_stack)

    model_path = os.path.join(model_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "ppo_nifty_final.zip")

    print(f"Loading Model: {model_path}")
    model = PPO.load(model_path)

    print("Starting Live Feed Polling...")

    # Initialize observation stack
    # We must manually step through the feed and maintain the stacked observations
    # For simplicity in this mock, we will update the internal env dataframe and call step
    # A true live production system would manually maintain a deque of normalized observations.

    full_test_df = pd.read_parquet(data_path)
    real_env = NiftyFuturesEnv(full_test_df, config, is_training=False)

    wrapped_env = DummyVecEnv([lambda: real_env])
    if os.path.exists(vec_norm_path):
        wrapped_env = VecNormalize.load(vec_norm_path, wrapped_env)
        wrapped_env.training = False
        wrapped_env.norm_reward = False
    wrapped_env = VecFrameStack(wrapped_env, n_stack=n_stack)

    obs = wrapped_env.reset()

    current_idx = 0
    while feed.has_next():
        current_bar = feed.get_next_bar(delay=0.0) # set delay > 0 to simulate real time
        next_bar = full_test_df.iloc[current_idx + 1] if current_idx + 1 < len(full_test_df) else None

        # RL Agent Predicts Action based on current state
        action, _states = model.predict(obs, deterministic=True)
        action_scalar = action[0] if isinstance(action, np.ndarray) else action

        # Broker executes based on signal
        broker.process_signal(current_bar, next_bar, action_scalar)

        # Step env to get next observation (syncing the agent's view)
        obs, rewards, dones, infos = wrapped_env.step(action)

        # Mirror broker position into env to keep states synchronized
        # (Real world requires careful synchronization between broker actual state and agent perceived state)
        real_env.position = broker.position
        real_env.entry_price = broker.entry_price
        real_env.trade_duration = broker.trade_duration
        real_env.unrealized_pnl = 0 if broker.position == 0 else (next_bar['close'] - broker.entry_price) * broker.lot_size * broker.position if next_bar is not None else 0

        current_idx += 1

        # Stop if EOD / end of stream
        if dones[0]:
             break

    print("Paper trading finished. Logs saved to Excel.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty RL Trading Agent")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "paper"], default="train", help="Mode to run the agent in")

    args, unknown = parser.parse_known_args() # Prevent issues in notebook environments

    if args.mode == "train":
        train_agent()
    elif args.mode == "backtest":
        evaluate_model()
    elif args.mode == "paper":
        run_paper_trading()
