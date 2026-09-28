import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class NiftyFuturesEnv(gym.Env):
    """
    Custom Environment for trading Nifty Futures.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, config, is_training=True):
        super(NiftyFuturesEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.config = config['settings']['env']
        self.is_training = is_training

        # Friction params
        self.slippage = self.config['slippage']
        self.brokerage = self.config['brokerage_per_order']
        self.stt_rate = self.config['stt_rate_sell']
        self.exch_charge = self.config['exchange_charges']
        self.gst_rate = self.config['gst_rate']
        self.sebi_charge = self.config['sebi_charges']
        self.lot_size = self.config['lot_size']
        self.initial_balance = self.config['initial_balance']
        self.max_dd_pts = self.config['max_drawdown_pts']

        # Actions: 0 = Flat, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)

        # Features to extract from df
        self.feature_cols = [col for col in self.df.columns if col not in ['date', 'time', 'open', 'high', 'low', 'close', 'volume']]

        # State: feature_cols + position(1) + unrealized_pnl(1) + duration(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.position = 0 # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.trade_duration = 0

        self.balance = self.initial_balance
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.peak_balance = self.initial_balance

        self.is_done = False
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        state_extras = np.array([
            self.position,
            self.unrealized_pnl / self.initial_balance, # Normalized unrealized PnL
            self.trade_duration / 375.0 # Normalized duration (max 375 mins)
        ], dtype=np.float32)
        return np.concatenate([obs, state_extras])

    def _calculate_transaction_costs(self, price, action_type):
        """
        action_type: 'buy' or 'sell'
        Returns total cost per lot in INR.
        """
        turnover = price * self.lot_size
        brokerage = self.brokerage
        exch_fee = turnover * self.exch_charge
        sebi_fee = turnover * self.sebi_charge
        stt = turnover * self.stt_rate if action_type == 'sell' else 0.0
        gst = (brokerage + exch_fee) * self.gst_rate

        total_charges = brokerage + exch_fee + sebi_fee + stt + gst
        return total_charges

    def step(self, action):
        if self.is_done:
            return self._get_obs(), 0, True, False, {}

        current_time_str = str(self.df.loc[self.current_step, 'time'])
        is_eod = current_time_str >= self.config['intraday_square_off']

        # Mapping action: 0 -> 0, 1 -> 1, 2 -> -1
        desired_position = 0
        if action == 1: desired_position = 1
        elif action == 2: desired_position = -1

        # Force EOD square off
        if is_eod:
            desired_position = 0

        # Get next candle's open for execution to avoid look-ahead bias
        # If we are at the last step, use current close instead (edge case)
        if self.current_step + 1 < len(self.df):
            exec_price = self.df.loc[self.current_step + 1, 'open']
        else:
            exec_price = self.df.loc[self.current_step, 'close']
            desired_position = 0 # Force exit on last tick

        reward = 0.0
        transaction_cost = 0.0

        # Handle position changes
        if self.position != desired_position:
            # 1. Close existing position
            if self.position != 0:
                # Close Long
                if self.position == 1:
                    exit_price = exec_price - self.slippage
                    gross_pnl = (exit_price - self.entry_price) * self.lot_size
                    cost = self._calculate_transaction_costs(exit_price, 'sell')
                # Close Short
                elif self.position == -1:
                    exit_price = exec_price + self.slippage
                    gross_pnl = (self.entry_price - exit_price) * self.lot_size
                    cost = self._calculate_transaction_costs(exit_price, 'buy')

                transaction_cost += cost
                net_pnl = gross_pnl - cost

                self.balance += net_pnl
                self.realized_pnl += net_pnl

                # Reward is net_pnl minus what was already given as unrealized_pnl during the hold
                reward += (net_pnl - self.unrealized_pnl)

                self.position = 0
                self.entry_price = 0.0
                self.trade_duration = 0
                self.unrealized_pnl = 0.0

            # 2. Open new position
            if desired_position != 0:
                if desired_position == 1: # Open Long
                    self.entry_price = exec_price + self.slippage
                    cost = self._calculate_transaction_costs(self.entry_price, 'buy')
                elif desired_position == -1: # Open Short
                    self.entry_price = exec_price - self.slippage
                    cost = self._calculate_transaction_costs(self.entry_price, 'sell')

                transaction_cost += cost
                self.balance -= cost # deduct cost immediately
                reward -= cost

                self.position = desired_position
                self.trade_duration = 1

        # Update state for hold
        if self.position != 0:
            self.trade_duration += 1

            # Use next candle's close for unrealized PnL (since we move step)
            if self.current_step + 1 < len(self.df):
                next_close = self.df.loc[self.current_step + 1, 'close']
            else:
                next_close = exec_price

            if self.position == 1:
                new_unrealized = (next_close - self.entry_price) * self.lot_size
            else:
                new_unrealized = (self.entry_price - next_close) * self.lot_size

            delta_unrealized = new_unrealized - self.unrealized_pnl
            reward += delta_unrealized
            self.unrealized_pnl = new_unrealized

            # Penalize excessive churn (turnover penalty) - implicitly handled by high trans costs

        # Drawdown check
        current_equity = self.balance + self.unrealized_pnl
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity

        drawdown_pts = (self.peak_balance - current_equity) / self.lot_size
        if drawdown_pts > self.max_dd_pts:
            reward -= 5000 # Heavy penalty
            self.is_done = True

        # Time step forward
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.is_done = True

        info = {
            'balance': self.balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'equity': current_equity,
            'drawdown_pts': drawdown_pts
        }

        return self._get_obs(), reward, self.is_done, False, info
