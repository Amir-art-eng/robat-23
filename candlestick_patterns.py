import pandas as pd
import numpy as np # Will be used for NaN handling if necessary

# Attempt to import data_fetcher for example usage.
try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False
    print("data_fetcher.py not found. Example will use manually created data or a simple fallback.")

try:
    import config
except ImportError:
    print("Warning: config.py not found in candlestick_patterns.py. Using internal defaults for pattern params.")
    # Define a fallback config class or dictionary if config.py is not found
    class config:
        CP_DOJI_BODY_TOLERANCE_RATIO = 0.05
        CP_MARUBOZU_BODY_MIN_RATIO = 0.8
        CP_MARUBOZU_WICK_MAX_RATIO = 0.1
        # Add any other config variables this script might use directly in its main block
        EX_LSTM_TRAINER_SYMBOL = 'SPY' # Fallback for example
        PRIMARY_INTERVAL_LSTM = '1d' # Fallback for example
        PRIMARY_PERIOD_LSTM = '6mo' # Fallback for example


def is_doji(data: pd.DataFrame, body_tolerance_ratio: float = 0.05) -> pd.Series:
    """
    Identifies Doji candles in OHLC data.
    A Doji is where abs(Open - Close) / (High - Low) is less than body_tolerance_ratio.

    Args:
        data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        body_tolerance_ratio (float): Maximum ratio of body to range for a Doji.

    Returns:
        pd.Series: Boolean Series indicating if a candle is a Doji.
    """
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

    candle_range = data['High'] - data['Low']
    body_size = abs(data['Open'] - data['Close'])

    # Handle potential division by zero if High == Low (candle_range is 0)
    # In such cases, if body_size is also 0 (Open == Close), it's a perfect Doji (four-price Doji).
    # If candle_range is 0 but body_size is not (should not happen with valid OHLC), it's not a Doji.
    # We'll consider candle_range = 0 as a Doji if body_size is also very small (effectively zero).
    # A common approach: if candle_range is zero, it's a doji if open and close are also equal.

    # Initialize result Series
    is_doji_series = pd.Series(False, index=data.index)

    # Where candle_range is not zero
    non_zero_range_mask = candle_range != 0
    is_doji_series[non_zero_range_mask] = \
        (body_size[non_zero_range_mask] / candle_range[non_zero_range_mask]) < body_tolerance_ratio

    # Where candle_range is zero (High == Low)
    zero_range_mask = candle_range == 0
    # It's a Doji if Open == Close (body_size is zero or very small)
    is_doji_series[zero_range_mask] = body_size[zero_range_mask] < (data['High'][zero_range_mask] * body_tolerance_ratio) # Check against a small absolute value if range is 0
    # More simply, if range is 0, it's a doji if body is also 0.
    # is_doji_series[zero_range_mask] = (body_size[zero_range_mask] == 0) # Alternative for zero range

    return is_doji_series

def is_marubozu(data: pd.DataFrame, body_min_ratio: float = 0.8, wick_max_ratio: float = 0.1) -> pd.Series:
    """
    Identifies Marubozu candles (strong bullish/bearish body).
    The body should be at least body_min_ratio of the total candle range.
    The combined wicks should be less than wick_max_ratio of the total candle range.

    Args:
        data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        body_min_ratio (float): Minimum ratio of body to range.
        wick_max_ratio (float): Maximum ratio of combined wicks to range.

    Returns:
        pd.Series: Boolean Series indicating if a candle is a Marubozu.
    """
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

    candle_range = data['High'] - data['Low']
    body_size = abs(data['Open'] - data['Close'])

    upper_wick = data['High'] - np.maximum(data['Open'], data['Close'])
    lower_wick = np.minimum(data['Open'], data['Close']) - data['Low']
    total_wick = upper_wick + lower_wick

    # Initialize result Series
    is_marubozu_series = pd.Series(False, index=data.index)

    # Avoid division by zero for zero-range candles
    # A Marubozu typically implies a significant range, so H != L.
    # If High == Low, candle_range is 0. Such candles are not typical Marubozu.
    valid_range_mask = candle_range > 0 # Ensure candle_range is positive

    condition1 = (body_size[valid_range_mask] / candle_range[valid_range_mask]) >= body_min_ratio
    condition2 = (total_wick[valid_range_mask] / candle_range[valid_range_mask]) < wick_max_ratio

    is_marubozu_series[valid_range_mask] = condition1 & condition2

    return is_marubozu_series

def is_outside_bar(data: pd.DataFrame) -> pd.Series:
    """
    Identifies if the current bar is an Outside Bar.
    High > previous_High AND Low < previous_Low.

    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low' columns.

    Returns:
        pd.Series: Boolean Series indicating if a candle is an Outside Bar.
    """
    if not all(col in data.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns.")

    is_outside_series = pd.Series(False, index=data.index)
    if len(data) < 2:
        return is_outside_series # Not enough data for comparison

    # Shift High and Low to get previous bar's values
    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)

    is_outside_series = (data['High'] > prev_high) & (data['Low'] < prev_low)
    is_outside_series.iloc[0] = False # First bar cannot be an outside bar
    return is_outside_series

def is_inside_bar(data: pd.DataFrame) -> pd.Series:
    """
    Identifies if the current bar is an Inside Bar.
    High < previous_High AND Low > previous_Low.

    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low' columns.

    Returns:
        pd.Series: Boolean Series indicating if a candle is an Inside Bar.
    """
    if not all(col in data.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns.")

    is_inside_series = pd.Series(False, index=data.index)
    if len(data) < 2:
        return is_inside_series

    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)

    is_inside_series = (data['High'] < prev_high) & (data['Low'] > prev_low)
    is_inside_series.iloc[0] = False # First bar cannot be an inside bar
    return is_inside_series

def add_candlestick_patterns(data: pd.DataFrame,
                             doji_tolerance_ratio: float = config.CP_DOJI_BODY_TOLERANCE_RATIO if 'config' in globals() else 0.05,
                             marubozu_body_min_ratio: float = config.CP_MARUBOZU_BODY_MIN_RATIO if 'config' in globals() else 0.8,
                             marubozu_wick_max_ratio: float = config.CP_MARUBOZU_WICK_MAX_RATIO if 'config' in globals() else 0.1
                             ) -> pd.DataFrame:
    """
    Adds candlestick pattern columns to the OHLCV DataFrame.

    Args:
        data (pd.DataFrame): Input OHLCV DataFrame.
        doji_tolerance_ratio (float): Max body ratio for Doji.
        marubozu_body_min_ratio (float): Min body ratio for Marubozu.
        marubozu_wick_max_ratio (float): Max wick ratio for Marubozu.

    Returns:
        pd.DataFrame: DataFrame with added boolean pattern columns.
    """
    df = data.copy()
    df['is_doji'] = is_doji(df, body_tolerance_ratio=doji_tolerance_ratio)
    df['is_marubozu'] = is_marubozu(df, body_min_ratio=marubozu_body_min_ratio, wick_max_ratio=marubozu_wick_max_ratio)
    # Assuming is_bullish_engulfing, is_bearish_engulfing etc. are added elsewhere or not part of this config change directly
    # For now, only adding patterns that are explicitly configured for ratios here.
    # Other patterns like engulfing, hammer, etc., if they were to be made configurable, would need similar treatment.
    # The original subtask was to make doji and marubozu ratios configurable.
    # Adding other common patterns that don't have such ratio configs yet:
    df['is_outside_bar'] = is_outside_bar(df)
    df['is_inside_bar'] = is_inside_bar(df)
    # Placeholder for other patterns that might be added later if needed by config.LSTM_FEATURE_COLS
    # For example, if 'is_bullish_engulfing' is in config.LSTM_FEATURE_COLS, it should be generated here.
    # This function should aim to generate all boolean candlestick columns listed in config.LSTM_FEATURE_COLS if possible.
    # For now, sticking to the ones explicitly defined or modified by this subtask.
    return df

if __name__ == "__main__":
    print("Running candlestick_patterns.py example...")

    sample_ohlcv_data = None
    # Use config for fetching example data if config is available
    example_symbol = config.EX_LSTM_TRAINER_SYMBOL if 'config' in globals() else 'SPY'
    example_interval = config.PRIMARY_INTERVAL_LSTM if 'config' in globals() else '1d'
    example_period = config.PRIMARY_PERIOD_LSTM if 'config' in globals() else '6mo'

    if DATA_FETCHER_AVAILABLE:
        print(f"Attempting to fetch sample data using data_fetcher for '{example_symbol}'...")
        sample_ohlcv_data = fetch_price_data(symbol=example_symbol, interval=example_interval, period=example_period)
        if sample_ohlcv_data is None:
            print(f"Failed to fetch {example_symbol} data.")
        else:
            print(f"Successfully fetched {len(sample_ohlcv_data)} entries of {example_symbol} data.")

    if sample_ohlcv_data is None:
        print("Using manually created sample OHLCV data as fallback.")
        # Manually create a diverse set of candles for testing
        data_dict = {
            'Open':  [100, 102, 101, 105, 103, 108, 108, 100, 105, 103, 100, 100, 100, 105, 102],
            'High':  [105, 103, 101.5,110, 103.5,110, 109, 106, 105.5,104, 102, 100.1,103, 108, 102.5],
            'Low':   [98,  101, 100.5,103, 102.5,105, 107, 98,  104.5,102, 98,  99.9, 97,  100, 101.5],
            'Close': [102, 101.5,101, 109, 103, 106, 107.5,105, 105, 102.5,101, 100,  98, 101, 102],
            'Volume':[1000,1500,1200,1800,1100,2000,1300,1700,900,1600,1400,800,1900,2200,1000]
        }
        sample_ohlcv_data = pd.DataFrame(data_dict, index=pd.date_range(start='2023-01-01', periods=len(data_dict['Open'])))
        print(f"Created manual dataset with {len(sample_ohlcv_data)} candles.")

    if sample_ohlcv_data is not None and not sample_ohlcv_data.empty:
        print("\nOriginal data (first 5 rows):")
        print(sample_ohlcv_data.head())

        # Add candlestick patterns using parameters from config
        data_with_patterns = add_candlestick_patterns(
            sample_ohlcv_data,
            doji_tolerance_ratio=config.CP_DOJI_BODY_TOLERANCE_RATIO if 'config' in globals() else 0.05,
            marubozu_body_min_ratio=config.CP_MARUBOZU_BODY_MIN_RATIO if 'config' in globals() else 0.8,
            marubozu_wick_max_ratio=config.CP_MARUBOZU_WICK_MAX_RATIO if 'config' in globals() else 0.1
        )

        print("\nData with patterns (first 5 rows with pattern columns):")
        print(data_with_patterns[['Open', 'High', 'Low', 'Close', 'is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar']].head())

        # Print rows where any pattern is True
        pattern_columns = ['is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar']
        identified_patterns_df = data_with_patterns[data_with_patterns[pattern_columns].any(axis=1)]

        if not identified_patterns_df.empty:
            print(f"\nIdentified patterns ({len(identified_patterns_df)} instances):")
            print(identified_patterns_df[['Open', 'High', 'Low', 'Close'] + pattern_columns])

            print("\nCounts of each pattern type:")
            for pattern in pattern_columns:
                count = data_with_patterns[pattern].sum()
                print(f"  {pattern}: {count}")
        else:
            print("\nNo candlestick patterns identified in the sample data with current settings.")
            print("This might be expected depending on the data and parameters.")
            # print("\nFull data with patterns for review:")
            # print(data_with_patterns[['Open', 'High', 'Low', 'Close'] + pattern_columns])

    else:
        print("No data available to process for candlestick patterns.")

    print("\ncandlestick_patterns.py example finished.")
