import time
import os
import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Configuration
TICKER = "^NSEI"
WINDOW_SIZE = 10
MODEL_PATH = "nifty_ppo_model"
LOG_FILE = "live_mock_trades_log.xlsx"
POLL_INTERVAL = 60 # seconds

def get_latest_data(window_size):
    """
    Fetches the latest 1-minute interval data for Nifty 50.
    Requires at least `window_size` rows.
    """
    try:
        # Fetch last 1 day of 1-minute data to ensure we have enough history
        ticker_data = yf.Ticker(TICKER)
        df = ticker_data.history(period="1d", interval="1m")
        if df.empty or len(df) < window_size:
            return None

        # Standardize column names to lowercase to match training data
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # We need the last `window_size` rows
        df_window = df.iloc[-window_size:][['open', 'high', 'low', 'close', 'volume']]

        # yfinance index is the datetime
        latest_time = df_window.index[-1]

        return df_window, latest_time

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

def format_observation(df_window):
    """
    Formats the pandas dataframe into a numpy array for the model.
    """
    obs = df_window.values.astype(np.float32)
    # The model expects batched input for dummyvecenv or just the correct shape
    # For a single env, shape should match the observation space, which is (WINDOW_SIZE, n_features)
    # PPO.predict usually wants an unbatched observation if not using VecEnv, or batched.
    # We'll pass it as is, predict will handle it or we expand dims.
    return obs

def log_trade(trade_info):
    """
    Appends a single trade record to the Excel log file.
    """
    df_new = pd.DataFrame([trade_info])
    df_new['date'] = df_new['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    cols = ['date', 'type', 'price', 'pnl']
    for col in cols:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[cols]

    if os.path.exists(LOG_FILE):
        df_existing = pd.read_excel(LOG_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(LOG_FILE, index=False)
    print(f"Trade logged: {trade_info}")

def run_live_loop():
    print(f"Loading trained model from {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    current_position = 0 # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0

    print(f"Starting real-time mock trading loop for {TICKER}. Polling every {POLL_INTERVAL} seconds...")

    while True:
        try:
            df_window, latest_time = get_latest_data(WINDOW_SIZE)

            if df_window is None:
                print("Could not fetch enough data. Retrying next cycle...")
                time.sleep(POLL_INTERVAL)
                continue

            obs = format_observation(df_window)

            # Predict action: 0: Short, 1: Flat, 2: Long
            action, _ = model.predict(obs, deterministic=True)
            # Ensure action is an int if it's a 0d numpy array
            if hasattr(action, 'item'):
                action = action.item()

            target_position = action - 1
            current_price = df_window.iloc[-1]['close']

            # Execute logic if position needs to change
            if current_position != target_position:
                # Close existing position if any
                if current_position == 1:
                    pnl = current_price - entry_price
                    log_trade({'type': 'close_long', 'price': current_price, 'pnl': pnl, 'date': latest_time})
                elif current_position == -1:
                    pnl = entry_price - current_price
                    log_trade({'type': 'close_short', 'price': current_price, 'pnl': pnl, 'date': latest_time})

                # Open new position if any
                if target_position == 1:
                    entry_price = current_price
                    log_trade({'type': 'open_long', 'price': current_price, 'date': latest_time})
                elif target_position == -1:
                    entry_price = current_price
                    log_trade({'type': 'open_short', 'price': current_price, 'date': latest_time})

                current_position = target_position

            print(f"[{latest_time}] Price: {current_price:.2f} | Current Pos: {current_position} | Target Pos: {target_position}")

            # Wait for the next 1-minute candle
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopping live mock trading...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run_live_loop()
