import pandas as pd
from stable_baselines3 import PPO
from nifty_trading_env import NiftyTradingEnv

def run_mock_trade():
    print("Loading test data...")
    test_df = pd.read_csv("nifty_test_data.csv")
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Create env purely for simulation
    env = NiftyTradingEnv(df=test_df, window_size=10)

    print("Loading trained model...")
    model = PPO.load("nifty_ppo_model")

    obs, info = env.reset()

    all_trades = []

    print("Running mock trade simulation...")
    # For demonstration, we might just run a portion if it's too long, but we will run the whole test set
    # Using a progress counter
    total_steps = env.max_steps - env.window_size

    for i in range(total_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10000 == 0 and i > 0:
            print(f"Processed {i}/{total_steps} steps...")

        if terminated:
            break

    # Read the complete history directly from the environment to avoid losing flipped positions
    for trade in env.history:
        step_idx = trade['step']
        trade['date'] = test_df.iloc[step_idx]['date']
        all_trades.append(trade)

    return all_trades

if __name__ == "__main__":
    trades = run_mock_trade()
    print(f"Total trades executed: {len(trades)}")

    if trades:
        print("Exporting trades to mock_trades_log.xlsx...")
        trades_df = pd.DataFrame(trades)

        # Ensure 'date' is a timezone unaware string or datetime for Excel
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Reorder columns for readability if needed
        cols = ['date', 'type', 'price', 'pnl', 'step']
        # If 'pnl' is not in all dicts (open trades don't have PnL), it will be NaN
        for col in cols:
            if col not in trades_df.columns:
                trades_df[col] = None

        trades_df = trades_df[cols]

        trades_df.to_excel("mock_trades_log.xlsx", index=False)
        print("Export complete.")
