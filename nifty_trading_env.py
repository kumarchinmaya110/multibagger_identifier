import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class NiftyTradingEnv(gym.Env):
    """
    A custom trading environment for Nifty 50 futures based on spot data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=10, render_mode=None):
        super(NiftyTradingEnv, self).__init__()

        # Ensure df is properly formatted with OHLCV features and sorted by date
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.render_mode = render_mode

        # Actions: 0: Short, 1: Flat, 2: Long
        self.action_space = spaces.Discrete(3)

        # Features: open, high, low, close, volume (normalized typically)
        # Assuming df has these columns exactly in some specific order
        self.n_features = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )

        self.current_step = self.window_size
        # Position: -1 (Short), 0 (Flat), 1 (Long)
        self.current_position = 0
        self.entry_price = 0.0

        self.max_steps = len(self.df) - 1
        self.history = []

    def _get_observation(self):
        # Retrieve the past 'window_size' rows for OHLCV
        # Ensure we just get the values for the 5 features
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step][['open', 'high', 'low', 'close', 'volume']].values
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.current_position = 0
        self.entry_price = 0.0
        self.history = []

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # action is 0: Short, 1: Flat, 2: Long

        # Map action to target position (-1, 0, 1)
        target_position = action - 1

        current_price = self.df.iloc[self.current_step]['close']

        reward = 0.0
        trade_info = None

        # Calculate PnL if we are closing or changing position
        if self.current_position != target_position:
            # We are changing position

            # Close existing position if any
            if self.current_position == 1:
                # Close Long
                pnl = current_price - self.entry_price
                prev_price = self.df.iloc[self.current_step - 1]['close']
                reward += (current_price - prev_price)  # Only add final step delta to avoid double counting
                trade_info = {'type': 'close_long', 'price': current_price, 'pnl': pnl, 'step': self.current_step}
                self.history.append(trade_info)
            elif self.current_position == -1:
                # Close Short
                pnl = self.entry_price - current_price
                prev_price = self.df.iloc[self.current_step - 1]['close']
                reward += (prev_price - current_price) # Only add final step delta to avoid double counting
                trade_info = {'type': 'close_short', 'price': current_price, 'pnl': pnl, 'step': self.current_step}
                self.history.append(trade_info)

            # Open new position if any
            if target_position == 1:
                # Open Long
                self.entry_price = current_price
                trade_info = {'type': 'open_long', 'price': current_price, 'step': self.current_step}
                self.history.append(trade_info)
            elif target_position == -1:
                # Open Short
                self.entry_price = current_price
                trade_info = {'type': 'open_short', 'price': current_price, 'step': self.current_step}
                self.history.append(trade_info)

            self.current_position = target_position
        else:
            # Holding position, can give small step reward based on unrealized PnL change
            if self.current_position == 1:
                prev_price = self.df.iloc[self.current_step - 1]['close']
                reward += (current_price - prev_price)
            elif self.current_position == -1:
                prev_price = self.df.iloc[self.current_step - 1]['close']
                reward += (prev_price - current_price)

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if terminated and self.current_position != 0:
            # Force close at the end
            if self.current_position == 1:
                pnl = current_price - self.entry_price
                reward += pnl
                self.history.append({'type': 'force_close_long', 'price': current_price, 'pnl': pnl, 'step': self.current_step-1})
            elif self.current_position == -1:
                pnl = self.entry_price - current_price
                reward += pnl
                self.history.append({'type': 'force_close_short', 'price': current_price, 'pnl': pnl, 'step': self.current_step-1})
            self.current_position = 0

        obs = self._get_observation()

        info = {
            'current_price': current_price,
            'position': self.current_position,
            'trade_info': trade_info
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
