import pandas as pd
from nifty_rl_trader.env.nifty_futures_env import NiftyFuturesEnv

class MockBroker:
    """
    Mock broker engine that interfaces with the LiveFeed and manages positions, margins, and logging.
    Re-uses the transaction logic from NiftyFuturesEnv.
    """
    def __init__(self, config, logger):
        self.config = config['settings']['env']
        self.logger = logger

        self.lot_size = self.config['lot_size']
        self.slippage = self.config['slippage']
        self.initial_balance = self.config['initial_balance']
        self.eod_time = self.config['intraday_square_off']

        # State
        self.balance = self.initial_balance
        self.position = 0 # 0: flat, 1: long, -1: short
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_duration = 0

        # Env for cost calculations
        # We dummy instantiate just to access the cost method easily, though we can duplicate it here.
        self.dummy_env = NiftyFuturesEnv(pd.DataFrame({'close':[0]}), config, is_training=False)

        # Daily tracking
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_peak_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.current_date = None

    def _calc_cost(self, price, action_type):
        return self.dummy_env._calculate_transaction_costs(price, action_type)

    def process_signal(self, current_bar, next_bar, action):
        """
        Process the action from the RL agent. Execution happens on the OPEN of next_bar.
        If next_bar is None, force close.
        """
        if next_bar is None:
            # End of stream, force close if open
            if self.position != 0:
                self._execute_trade(current_bar, current_bar['close'], 0, "End_Of_Stream")
            return

        current_time_str = str(current_bar['time'])
        is_eod = current_time_str >= self.eod_time

        desired_position = 0
        if action == 1: desired_position = 1
        elif action == 2: desired_position = -1

        if is_eod:
            desired_position = 0
            exit_reason = "EOD_SquareOff"
        else:
            exit_reason = "Signal"

        exec_price = next_bar['open']

        if self.position != desired_position:
            # 1. Close current position
            if self.position != 0:
                self._execute_trade(next_bar, exec_price, 0, exit_reason)

            # 2. Open new position
            if desired_position != 0:
                self._execute_trade(next_bar, exec_price, desired_position, "Signal")

        # Track daily reset and summary
        bar_date = str(current_bar['date'].date())
        if self.current_date is None:
            self.current_date = bar_date

        if bar_date != self.current_date:
            self._log_daily_summary()
            self.current_date = bar_date
            self.daily_start_balance = self.balance
            self.daily_peak_balance = self.balance
            self.daily_trades = 0
            self.daily_wins = 0

        # Update duration and peak
        if self.position != 0:
            self.trade_duration += 1
            # Current equity approx
            unrealized = (current_bar['close'] - self.entry_price) * self.lot_size * self.position
            equity = self.balance + unrealized
            if equity > self.daily_peak_balance:
                self.daily_peak_balance = equity

    def _execute_trade(self, bar, exec_price, target_pos, reason):
        exec_time = str(bar['date'])

        # Close position
        if target_pos == 0 and self.position != 0:
            if self.position == 1:
                exit_price = exec_price - self.slippage
                gross_pnl = (exit_price - self.entry_price) * self.lot_size
                cost = self._calc_cost(exit_price, 'sell')
                trade_type = 'LONG'
            else:
                exit_price = exec_price + self.slippage
                gross_pnl = (self.entry_price - exit_price) * self.lot_size
                cost = self._calc_cost(exit_price, 'buy')
                trade_type = 'SHORT'

            net_pnl = gross_pnl - cost
            self.balance += net_pnl

            # Log Trade
            self.logger.log_trade(
                entry_time=str(self.entry_time),
                exit_time=exec_time,
                trade_type=trade_type,
                entry_price=self.entry_price,
                exit_price=exit_price,
                qty=self.lot_size,
                gross_pnl=gross_pnl,
                charges=cost + self.entry_cost, # Combine entry and exit cost for display
                net_pnl=net_pnl - self.entry_cost,
                exit_reason=reason,
                cum_pnl=self.balance - self.initial_balance
            )

            self.daily_trades += 1
            if (net_pnl - self.entry_cost) > 0:
                self.daily_wins += 1

            self.position = 0
            self.entry_price = 0.0
            self.entry_time = None
            self.trade_duration = 0
            self.entry_cost = 0.0

        # Open position
        elif target_pos != 0 and self.position == 0:
            self.position = target_pos
            self.entry_time = exec_time
            if target_pos == 1:
                self.entry_price = exec_price + self.slippage
                self.entry_cost = self._calc_cost(self.entry_price, 'buy')
            else:
                self.entry_price = exec_price - self.slippage
                self.entry_cost = self._calc_cost(self.entry_price, 'sell')

            self.balance -= self.entry_cost
            self.trade_duration = 1

    def _log_daily_summary(self):
        if self.daily_trades > 0:
            win_rate = self.daily_wins / self.daily_trades
        else:
            win_rate = 0.0

        max_dd = (self.daily_peak_balance - self.balance) / self.lot_size
        net_profit = self.balance - self.daily_start_balance

        self.logger.log_daily_summary(
            date_str=self.current_date,
            total_trades=self.daily_trades,
            win_rate=win_rate,
            max_dd_pts=max_dd,
            net_profit=net_profit,
            ending_capital=self.balance
        )
