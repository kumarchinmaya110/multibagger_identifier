import time
import pandas as pd
from datetime import datetime

class LiveFeedPoller:
    """
    Simulates a live feed by sequentially yielding rows from the historical test dataset.
    """
    def __init__(self, data_path):
        self.df = pd.read_parquet(data_path)
        self.current_idx = 0

    def get_next_bar(self, delay=0.0):
        """
        Yields the next 1-minute OHLCV bar and its features.
        delay: optional sleep to simulate real-time polling wait.
        """
        if self.current_idx < len(self.df):
            if delay > 0:
                time.sleep(delay)
            bar = self.df.iloc[self.current_idx].copy()
            self.current_idx += 1
            return bar
        return None

    def has_next(self):
        return self.current_idx < len(self.df)
