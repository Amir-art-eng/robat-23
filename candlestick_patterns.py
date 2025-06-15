import pandas as pd
import numpy as np

# Attempt to import data_fetcher for example usage.
try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False
    print("data_fetcher.py not found. Example will use manually created data or a simple fallback.")

# Attempt to import the actual config for the main example section,
# but individual functions will rely on the passed 'config_module'.
try:
    import config as actual_config
except ImportError:
    actual_config = None
    print("Warning: Full config.py not found for candlestick_patterns.py main example. Using MockConfig for all tests.")


def is_doji(data: pd.DataFrame, body_tolerance_ratio: float) -> pd.Series: # Removed default
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")
    candle_range = data['High'] - data['Low']
    body_size = abs(data['Open'] - data['Close'])
    is_doji_series = pd.Series(False, index=data.index)
    non_zero_range_mask = candle_range != 0
    is_doji_series[non_zero_range_mask] = \
        (body_size[non_zero_range_mask] / candle_range[non_zero_range_mask]) < body_tolerance_ratio
    zero_range_mask = candle_range == 0
    # For zero-range bars, it's a Doji if the body is also effectively zero.
    # Using a small part of High as tolerance, or simply body_size == 0.
    is_doji_series[zero_range_mask] = body_size[zero_range_mask] < (data['High'][zero_range_mask] * (body_tolerance_ratio / 2.0))
    return is_doji_series

def is_marubozu(data: pd.DataFrame, body_min_ratio: float, wick_max_ratio: float) -> pd.Series: # Removed defaults
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")
    candle_range = data['High'] - data['Low']
    body_size = abs(data['Open'] - data['Close'])
    upper_wick = data['High'] - np.maximum(data['Open'], data['Close'])
    lower_wick = np.minimum(data['Open'], data['Close']) - data['Low']
    total_wick = upper_wick + lower_wick
    is_marubozu_series = pd.Series(False, index=data.index)
    valid_range_mask = candle_range > 0
    condition1 = (body_size[valid_range_mask] / candle_range[valid_range_mask]) >= body_min_ratio
    condition2 = (total_wick[valid_range_mask] / candle_range[valid_range_mask]) < wick_max_ratio
    is_marubozu_series[valid_range_mask] = condition1 & condition2
    return is_marubozu_series

def is_outside_bar(data: pd.DataFrame) -> pd.Series:
    if not all(col in data.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns.")
    is_outside_series = pd.Series(False, index=data.index)
    if len(data) < 2: return is_outside_series
    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)
    is_outside_series = (data['High'] > prev_high) & (data['Low'] < prev_low)
    is_outside_series.iloc[0] = False
    return is_outside_series

def is_inside_bar(data: pd.DataFrame) -> pd.Series:
    if not all(col in data.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns.")
    is_inside_series = pd.Series(False, index=data.index)
    if len(data) < 2: return is_inside_series
    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)
    is_inside_series = (data['High'] < prev_high) & (data['Low'] > prev_low)
    is_inside_series.iloc[0] = False
    return is_inside_series

# --- Two-Bar Reversal Pattern ---
def get_two_bar_reversal_signal(df: pd.DataFrame, i: int, config_module) -> int:
    """
    Checks for a 2-Bar Reversal pattern ending at index i.
    'i' is the integer location (iloc) of the current (second) candle.
    Returns 1 for bullish 2-bar reversal, -1 for bearish, 0 otherwise.
    """
    lookback_period = getattr(config_module, 'CP_2BAR_LOOKBACK_PERIOD', 5)
    body_similarity_pct = getattr(config_module, 'CP_2BAR_BODY_SIMILARITY_PCT', 0.25)
    close_extreme_pct_2nd_bar = getattr(config_module, 'CP_2BAR_SECOND_BAR_CLOSE_EXTREME_PCT', 0.80)

    if i < 1: return 0

    bar2 = df.iloc[i]
    bar2_body_size = abs(bar2['Close'] - bar2['Open'])
    # Use a small fraction of High price as threshold for negligible body
    if bar2_body_size < (bar2['High'] * 0.0005): return 0

    # Check for Bullish 2-Bar Reversal (Bar1 Bearish, Bar2 Bullish)
    if bar2['Close'] > bar2['Open']: # Bar2 is Bullish
        bar2_hl_range = bar2['High'] - bar2['Low']
        bar2_closes_strong = False
        if bar2_hl_range > 0:
            bar2_close_loc_ratio = (bar2['Close'] - bar2['Low']) / bar2_hl_range
            if bar2_close_loc_ratio >= close_extreme_pct_2nd_bar:
                bar2_closes_strong = True
        elif bar2_body_size > 0 and bar2['Close'] == bar2['High']: # Bullish marubozu-like zero-range bar
             bar2_closes_strong = True

        if bar2_closes_strong:
            for j in range(max(0, i - lookback_period), i):
                bar1 = df.iloc[j]
                if bar1['Close'] < bar1['Open']: # Bar1 is Bearish
                    bar1_body_size = abs(bar1['Close'] - bar1['Open'])
                    if bar1_body_size < (bar1['High'] * 0.0005): continue

                    larger_body = max(bar1_body_size, bar2_body_size)
                    if larger_body == 0: continue
                    if abs(bar1_body_size - bar2_body_size) / larger_body <= body_similarity_pct:
                        return 1

    # Check for Bearish 2-Bar Reversal (Bar1 Bullish, Bar2 Bearish)
    if bar2['Close'] < bar2['Open']: # Bar2 is Bearish
        bar2_hl_range = bar2['High'] - bar2['Low']
        bar2_closes_strong = False
        if bar2_hl_range > 0:
            bar2_close_loc_ratio = (bar2['Close'] - bar2['Low']) / bar2_hl_range
            if bar2_close_loc_ratio <= (1.0 - close_extreme_pct_2nd_bar): # e.g. closes in bottom 20%
                bar2_closes_strong = True
        elif bar2_body_size > 0 and bar2['Close'] == bar2['Low']: # Bearish marubozu-like zero-range bar
            bar2_closes_strong = True

        if bar2_closes_strong:
            for j in range(max(0, i - lookback_period), i):
                bar1 = df.iloc[j]
                if bar1['Close'] > bar1['Open']: # Bar1 is Bullish
                    bar1_body_size = abs(bar1['Close'] - bar1['Open'])
                    if bar1_body_size < (bar1['High'] * 0.0005): continue

                    larger_body = max(bar1_body_size, bar2_body_size)
                    if larger_body == 0: continue
                    if abs(bar1_body_size - bar2_body_size) / larger_body <= body_similarity_pct:
                        return -1
    return 0

def add_candlestick_patterns(data_df: pd.DataFrame, config_module) -> pd.DataFrame:
    """
    Adds candlestick pattern columns to the OHLCV DataFrame.
    Uses config_module to get parameters for various patterns.
    """
    df = data_df.copy()

    doji_tol_ratio = getattr(config_module, 'CP_DOJI_BODY_TOLERANCE_RATIO', 0.05)
    maru_body_min_ratio = getattr(config_module, 'CP_MARUBOZU_BODY_MIN_RATIO', 0.8)
    maru_wick_max_ratio = getattr(config_module, 'CP_MARUBOZU_WICK_MAX_RATIO', 0.1)

    df['is_doji'] = is_doji(df, body_tolerance_ratio=doji_tol_ratio)
    df['is_marubozu'] = is_marubozu(df, body_min_ratio=maru_body_min_ratio, wick_max_ratio=maru_wick_max_ratio)
    df['is_outside_bar'] = is_outside_bar(df)
    df['is_inside_bar'] = is_inside_bar(df)

    # Add 2-Bar Reversal signals
    if 'get_two_bar_reversal_signal' in globals():
        df_len = len(df)
        two_bar_rev_values = [get_two_bar_reversal_signal(df, k, config_module) for k in range(df_len)]
        df['two_bar_reversal'] = two_bar_rev_values
    else:
        if 'two_bar_reversal' not in df.columns: df['two_bar_reversal'] = 0

    return df

# --- Best Signal Bar Functions ---
def is_best_bull_signal_bar(df: pd.DataFrame, i: int, config_module) -> bool:
    if i == 0: return False
    current_bar = df.iloc[i]; prev_bar = df.iloc[i-1]
    if not (current_bar['Close'] > current_bar['Open']): return False
    hl_range = current_bar['High'] - current_bar['Low']
    if hl_range == 0: return False

    lower_wick_min_r = getattr(config_module, 'CP_BEST_SIG_LOWER_WICK_MIN_RATIO_BULL', 0.25)
    lower_wick_max_r = getattr(config_module, 'CP_BEST_SIG_LOWER_WICK_MAX_RATIO_BULL', 0.60)
    upper_wick_max_r = getattr(config_module, 'CP_BEST_SIG_UPPER_WICK_MAX_SIZE_RATIO_BULL', 0.10)
    max_overlap_r = getattr(config_module, 'CP_BEST_SIG_MAX_BODY_OVERLAP_PREV_BAR_RATIO', 0.5)
    extreme_n_bars = getattr(config_module, 'CP_BEST_SIG_CLOSE_EXTREME_N_BARS', 3)

    lower_wick = min(current_bar['Open'], current_bar['Close']) - current_bar['Low']
    if not (lower_wick_min_r <= (lower_wick / hl_range) <= lower_wick_max_r): return False
    upper_wick = current_bar['High'] - max(current_bar['Open'], current_bar['Close'])
    if (upper_wick / hl_range) > upper_wick_max_r: return False

    current_body_low = min(current_bar['Open'], current_bar['Close'])
    current_body_high = max(current_bar['Open'], current_bar['Close'])
    overlap_with_prev_range = 0
    if prev_bar['High'] > current_body_low and prev_bar['Low'] < current_body_high:
        overlap_amount = min(current_body_high, prev_bar['High']) - max(current_body_low, prev_bar['Low'])
        current_body_size = abs(current_bar['Close'] - current_bar['Open'])
        if current_body_size > 0: overlap_with_prev_range = overlap_amount / current_body_size
    if overlap_with_prev_range > max_overlap_r: return False

    if i < extreme_n_bars: return False
    if not (current_bar['Close'] > df['High'].iloc[i-extreme_n_bars : i].max()): return False
    return True

def is_best_bear_signal_bar(df: pd.DataFrame, i: int, config_module) -> bool:
    if i == 0: return False
    current_bar = df.iloc[i]; prev_bar = df.iloc[i-1]
    if not (current_bar['Close'] < current_bar['Open']): return False
    hl_range = current_bar['High'] - current_bar['Low']
    if hl_range == 0: return False

    upper_wick_min_r = getattr(config_module, 'CP_BEST_SIG_UPPER_WICK_MIN_RATIO_BEAR', 0.25)
    upper_wick_max_r = getattr(config_module, 'CP_BEST_SIG_UPPER_WICK_MAX_RATIO_BEAR', 0.60)
    lower_wick_max_r = getattr(config_module, 'CP_BEST_SIG_LOWER_WICK_MAX_SIZE_RATIO_BEAR', 0.10)
    max_overlap_r = getattr(config_module, 'CP_BEST_SIG_MAX_BODY_OVERLAP_PREV_BAR_RATIO', 0.5)
    extreme_n_bars = getattr(config_module, 'CP_BEST_SIG_CLOSE_EXTREME_N_BARS', 3)

    upper_wick = current_bar['High'] - max(current_bar['Open'], current_bar['Close'])
    if not (upper_wick_min_r <= (upper_wick / hl_range) <= upper_wick_max_r): return False
    lower_wick = min(current_bar['Open'], current_bar['Close']) - current_bar['Low']
    if (lower_wick / hl_range) > lower_wick_max_r: return False

    current_body_low = min(current_bar['Open'], current_bar['Close'])
    current_body_high = max(current_bar['Open'], current_bar['Close'])
    overlap_with_prev_range = 0
    if prev_bar['High'] > current_body_low and prev_bar['Low'] < current_body_high:
        overlap_amount = min(current_body_high, prev_bar['High']) - max(current_body_low, prev_bar['Low'])
        current_body_size = abs(current_bar['Close'] - current_bar['Open'])
        if current_body_size > 0: overlap_with_prev_range = overlap_amount / current_body_size
    if overlap_with_prev_range > max_overlap_r: return False

    if i < extreme_n_bars: return False
    if not (current_bar['Close'] < df['Low'].iloc[i-extreme_n_bars : i].min()): return False
    return True

if __name__ == "__main__":
    print("Running candlestick_patterns.py example...")

    # --- Mock Config for all tests in this main block ---
    class MockConfigFull:
        CP_DOJI_BODY_TOLERANCE_RATIO = 0.05
        CP_MARUBOZU_BODY_MIN_RATIO = 0.8
        CP_MARUBOZU_WICK_MAX_RATIO = 0.1

        CP_BEST_SIG_LOWER_WICK_MIN_RATIO_BULL = 0.25
        CP_BEST_SIG_LOWER_WICK_MAX_RATIO_BULL = 0.60
        CP_BEST_SIG_UPPER_WICK_MAX_SIZE_RATIO_BULL = 0.10
        CP_BEST_SIG_MAX_BODY_OVERLAP_PREV_BAR_RATIO = 0.5
        CP_BEST_SIG_CLOSE_EXTREME_N_BARS = 3
        CP_BEST_SIG_UPPER_WICK_MIN_RATIO_BEAR = 0.25
        CP_BEST_SIG_UPPER_WICK_MAX_RATIO_BEAR = 0.60
        CP_BEST_SIG_LOWER_WICK_MAX_SIZE_RATIO_BEAR = 0.10

        CP_2BAR_LOOKBACK_PERIOD = 5
        CP_2BAR_BODY_SIMILARITY_PCT = 0.25
        CP_2BAR_SECOND_BAR_CLOSE_EXTREME_PCT = 0.80 # Top/Bottom 20%

        EX_LSTM_TRAINER_SYMBOL = 'SPY'
        PRIMARY_INTERVAL_LSTM = '1d'
        PRIMARY_PERIOD_LSTM = '6mo'

    mock_config_for_tests = MockConfigFull()
    cfg_to_use = actual_config if actual_config else mock_config_for_tests

    sample_ohlcv_data = None
    example_symbol = getattr(cfg_to_use, 'EX_LSTM_TRAINER_SYMBOL', 'SPY')
    example_interval = getattr(cfg_to_use, 'PRIMARY_INTERVAL_LSTM', '1d')
    example_period = getattr(cfg_to_use, 'PRIMARY_PERIOD_LSTM', '6mo')

    if DATA_FETCHER_AVAILABLE:
        print(f"Attempting to fetch sample data for '{example_symbol}'...")
        sample_ohlcv_data = fetch_price_data(symbol=example_symbol, interval=example_interval, period=example_period)

    if sample_ohlcv_data is None or sample_ohlcv_data.empty:
        print("Using manually created sample OHLCV data as fallback for general tests.")
        data_dict = {
            'Open':  [100,102,101,105,103,108,108,100,105,103,100,100,100,105,102, 105,100], # Added more for 2-bar
            'High':  [105,103,101.5,110,103.5,110,109,106,105.5,104,102,100.1,103,108,102.5,106,106],
            'Low':   [98,101,100.5,103,102.5,105,107,98,104.5,102,98,99.9,97,100,101.5, 99,100],
            'Close': [102,101.5,101,109,103,106,107.5,105,105,102.5,101,100,98,101,102, 100,105], # Bar 15 Bear, Bar 16 Bull
            'Volume':[1000,1500,1200,1800,1100,2000,1300,1700,900,1600,1400,800,1900,2200,1000,1000,1000]
        }
        sample_ohlcv_data = pd.DataFrame(data_dict, index=pd.date_range(start='2023-01-01', periods=len(data_dict['Open'])))

    if sample_ohlcv_data is not None and not sample_ohlcv_data.empty:
        print("\nOriginal data (sample head):")
        print(sample_ohlcv_data.head())

        data_with_patterns = add_candlestick_patterns(sample_ohlcv_data, config_module=cfg_to_use)

        print("\nData with patterns (sample head with pattern columns):")
        cols_to_show = ['Open', 'Close', 'is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar', 'two_bar_reversal']
        print(data_with_patterns[[col for col in cols_to_show if col in data_with_patterns.columns]].head())

        pattern_columns = ['is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar'] # bool patterns
        identified_bool_patterns_df = data_with_patterns[data_with_patterns[pattern_columns].any(axis=1)]

        if not identified_bool_patterns_df.empty:
            print(f"\nIdentified boolean patterns ({len(identified_bool_patterns_df)} instances):")
            print(identified_bool_patterns_df[['Open', 'Close'] + pattern_columns])
        else:
            print("\nNo boolean candlestick patterns identified in the sample data.")

        identified_2bar_rev = data_with_patterns[data_with_patterns['two_bar_reversal'] != 0]
        if not identified_2bar_rev.empty:
            print(f"\nIdentified 2-Bar Reversals ({len(identified_2bar_rev)} instances):")
            print(identified_2bar_rev[['Open', 'Close', 'two_bar_reversal']])
        else:
            print("\nNo 2-Bar Reversals identified in the sample data.")
    else:
        print("No data available to process for general candlestick patterns.")

    # --- Test Best Signal Bar Functions (using mock_config_for_tests for these specific params) ---
    print("\n--- Testing Best Signal Bar Functions ---")
    bull_test_data = {'Open':[100,101,102,103,104],'High':[101,102,103,104,110],'Low':[99,100,101,102,100],'Close':[101,102,103,102.5,109.5]}
    bull_test_df = pd.DataFrame(bull_test_data)
    print("\nTesting Best Bull Signal Bar:"); print(bull_test_df)
    is_best_bull = is_best_bull_signal_bar(bull_test_df, 4, mock_config_for_tests)
    print(f"Bar at iloc 4 is Best Bull Signal Bar: {is_best_bull} (Expected: True)")

    bear_test_data = {'Open':[110,109,108,107,106],'High':[111,110,109,108,110],'Low':[109,108,107,106,100],'Close':[109,108,107,107.5,100.5]}
    bear_test_df = pd.DataFrame(bear_test_data)
    print("\nTesting Best Bear Signal Bar:"); print(bear_test_df)
    is_best_bear = is_best_bear_signal_bar(bear_test_df, 4, mock_config_for_tests)
    print(f"Bar at iloc 4 is Best Bear Signal Bar: {is_best_bear} (Expected: True)")

    # --- Test 2-Bar Reversal with specific data ---
    print("\n--- Testing 2-Bar Reversal Signal (Specific Data) ---")
    # Bar 1 (Bearish), Bar 2 (Bullish, strong close, similar body) -> Bullish 2-bar @ iloc 2
    # Bar 4 (Bullish), Bar 5 (Bearish, strong close, similar body) -> Bearish 2-bar @ iloc 5
    two_bar_data = {
        'Open':  [100, 105, 100.5, 100, 100.5, 105.5],
        'High':  [102, 106, 105,   101, 106,   106],
        'Low':   [98,  100, 99.5,  99,  99.5,  100],
        'Close': [99,  100.5, 104.5, 100, 105,   100.5]
    } # Expected: Bullish at iloc 2, Bearish at iloc 5
    two_bar_df_specific = pd.DataFrame(two_bar_data)
    print("Specific 2-Bar Reversal Test DataFrame:")
    print(two_bar_df_specific)

    df_with_specific_2bar = add_candlestick_patterns(two_bar_df_specific.copy(), config_module=mock_config_for_tests)
    print("\nDataFrame with 'two_bar_reversal' (Specific Data):")
    print(df_with_specific_2bar[['Open', 'Close', 'two_bar_reversal']])
    print(f"Signal at iloc 2: {get_two_bar_reversal_signal(two_bar_df_specific, 2, mock_config_for_tests)} (Expected: 1)")
    print(f"Signal at iloc 5: {get_two_bar_reversal_signal(two_bar_df_specific, 5, mock_config_for_tests)} (Expected: -1)")

    print("\ncandlestick_patterns.py all examples finished.")
