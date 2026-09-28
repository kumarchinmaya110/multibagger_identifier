import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from nifty_rl_trader.env.nifty_futures_env import NiftyFuturesEnv
from nifty_rl_trader.features.technical_features import load_config

def make_env(df, config, is_training=True):
    def _init():
        env = NiftyFuturesEnv(df, config, is_training=is_training)
        return env
    return _init

def train_agent():
    config = load_config()
    data_dir = config['settings']['data_paths']['processed_data']

    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))

    num_envs = 4
    # Split train_df into chunks for each env
    chunk_size = len(train_df) // num_envs
    env_dfs = [train_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_envs)]

    # Create vectorized environments
    train_env = SubprocVecEnv([make_env(env_dfs[i], config, is_training=True) for i in range(num_envs)])
    val_env = DummyVecEnv([make_env(val_df, config, is_training=False)])

    # Normalize observations (but NOT rewards to keep monetary value meaningful for tracking, though normalization can help PPO)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    val_env.training = False

    # Frame stacking
    n_stack = config['settings']['rl_agent']['frame_stack']
    train_env = VecFrameStack(train_env, n_stack=n_stack)
    val_env = VecFrameStack(val_env, n_stack=n_stack)

    # Model parameters
    params = config['settings']['rl_agent']['hyperparameters']

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        gamma=params['gamma'],
        gae_lambda=params['gae_lambda'],
        clip_range=params['clip_range'],
        ent_coef=params['ent_coef'],
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    # Callbacks
    os.makedirs("nifty_rl_trader/saved_models", exist_ok=True)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="nifty_rl_trader/saved_models/best_model",
        log_path="nifty_rl_trader/saved_models/logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    # Train for a limited number of timesteps for demonstration/testing (scale up in production)
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    print("Saving final model and vec_normalize stats...")
    model.save("nifty_rl_trader/saved_models/ppo_nifty_final")
    train_env.save("nifty_rl_trader/saved_models/vec_normalize.pkl")

if __name__ == "__main__":
    train_agent()
