import pandas as pd
import numpy as np

# --- Import functions from other project modules ---
try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    print("Warning: data_fetcher.py not found. Example usage will be limited.")
    DATA_FETCHER_AVAILABLE = False

try:
    from candlestick_patterns import add_candlestick_patterns
    CANDLESTICK_PATTERNS_AVAILABLE = True
except ImportError:
    print("Warning: candlestick_patterns.py not found. Candlestick features will be skipped.")
    CANDLESTICK_PATTERNS_AVAILABLE = False

try:
    from market_structure import find_swing_highs_lows, identify_pullbacks, identify_breakouts
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    print("Warning: market_structure.py not found. Market structure features will be skipped.")
    MARKET_STRUCTURE_AVAILABLE = False

try:
    import config # Import the configuration file
except ImportError:
    print("Error: config.py not found. Using fallback internal defaults for lstm_feature_engineer.")
    class config: # Minimal fallback
        EX_LSTM_TRAINER_SYMBOL = 'AAPL'; EX_LSTM_TRAINER_INTERVAL = '1h'; EX_LSTM_TRAINER_PERIOD = '200d';
        LSTM_FE_FUTURE_N_BARS = 5; LSTM_FE_SWING_WINDOW = 10;
        LSTM_FE_BREAKOUT_LOOKBACK = 24; LSTM_FE_PULLBACK_TREND_LOOKBACK = 2;


def create_lstm_features(ohlcv_df: pd.DataFrame, future_n_bars: int = 5,
                         swing_window: int = 5, breakout_lookback: int = 20,
                         pullback_trend_lookback: int = 2) -> pd.DataFrame:
    """
    Creates features and a target variable from OHLCV data for LSTM modeling.
    """
    if not (CANDLESTICK_PATTERNS_AVAILABLE and MARKET_STRUCTURE_AVAILABLE):
        print("Error: Modules for candlestick or market structure features missing. Feature set will be incomplete.")

    if not all(col in ohlcv_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        raise ValueError("Input DataFrame must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.")

    df = ohlcv_df.copy()

    print("Calculating price-derived features...")
    df['return'] = df['Close'].pct_change()
    df['body_abs'] = abs(df['Close'] - df['Open'])
    df['hl_range'] = df['High'] - df['Low']

    df['body_range_norm'] = np.where(df['hl_range'] == 0, 0.0, df['body_abs'] / df['hl_range'])
    df['body_range_norm'] = df['body_range_norm'].fillna(0.0)

    df['hl_range_norm'] = np.where(df['Close'] == 0, 0.0, df['hl_range'] / df['Close'])
    df['hl_range_norm'] = df['hl_range_norm'].fillna(0.0)


    print("Adding candlestick pattern features...")
    if CANDLESTICK_PATTERNS_AVAILABLE:
        df = add_candlestick_patterns(df)
    else:
        for p_col in ['is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar']: df[p_col] = False

    print("Adding market structure features...")
    if MARKET_STRUCTURE_AVAILABLE:
        if 'is_swing_high' not in df.columns: df['is_swing_high'] = False
        if 'is_swing_low' not in df.columns: df['is_swing_low'] = False

        df = find_swing_highs_lows(df, window=swing_window)
        df = identify_pullbacks(df, trend_lookback_swings=pullback_trend_lookback)
        df = identify_breakouts(df, lookback_period=breakout_lookback)
    else:
        for ms_col in ['is_swing_high', 'is_swing_low', 'is_pullback_bar',
                       'is_bullish_breakout', 'is_bearish_breakout']: df[ms_col] = False

    print("Defining target variable...")
    df['future_close'] = df['Close'].shift(-future_n_bars)
    df['target_price_direction'] = np.where(
        df['future_close'].notna(),
        (df['future_close'] > df['Close']).astype(int),
        np.nan
    )

    feature_cols = [ # This list should align with config.LSTM_FEATURE_COLS
        'return', 'body_range_norm', 'hl_range_norm', 'Volume',
        'is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar',
        'is_swing_high', 'is_swing_low', 'is_pullback_bar',
        'is_bullish_breakout', 'is_bearish_breakout'
    ]
    # If config is loaded and LSTM_FEATURE_COLS is defined, prefer that for consistency
    if 'config' in globals() and hasattr(config, 'LSTM_FEATURE_COLS'):
        final_feature_cols = [col for col in config.LSTM_FEATURE_COLS if col in df.columns]
    else: # Fallback to local definition
        final_feature_cols = [col for col in feature_cols if col in df.columns]


    lstm_df_cols = final_feature_cols + ['target_price_direction']
    lstm_df = df[lstm_df_cols].copy()

    print("Handling NaNs...")
    lstm_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    lstm_df.dropna(inplace=True)

    print(f"LSTM feature DataFrame created with shape: {lstm_df.shape}")
    return lstm_df

if __name__ == "__main__":
    if not DATA_FETCHER_AVAILABLE:
        print("Data fetcher not available. Cannot run lstm_feature_engineer.py example.")
    else:
        print("Running lstm_feature_engineer.py example using config.py...")

        symbol = config.EX_LSTM_TRAINER_SYMBOL
        interval = config.EX_LSTM_TRAINER_INTERVAL
        period = config.EX_LSTM_TRAINER_PERIOD

        print(f"\nFetching data for {symbol}, interval {interval}, period {period}...")
        ohlcv_data = fetch_price_data(symbol=symbol, interval=interval, period=period)

        if ohlcv_data is not None and not ohlcv_data.empty:
            print(f"\nSuccessfully fetched {len(ohlcv_data)} data points for {symbol}.")
            initial_rows = len(ohlcv_data)

            lstm_features_df = create_lstm_features(
                ohlcv_data,
                future_n_bars=config.LSTM_FE_FUTURE_N_BARS,
                swing_window=config.LSTM_FE_SWING_WINDOW,
                breakout_lookback=config.LSTM_FE_BREAKOUT_LOOKBACK,
                pullback_trend_lookback=config.LSTM_FE_PULLBACK_TREND_LOOKBACK # Ensured this uses the correct config name
            )

            if not lstm_features_df.empty:
                print(f"Initial data rows: {initial_rows}, Feature DataFrame rows: {len(lstm_features_df)}")
                print("\n--- LSTM Features DataFrame ---")
                print("Head:")
                print(lstm_features_df.head())
                print("\nTail:")
                print(lstm_features_df.tail())

                print("\nInfo:")
                lstm_features_df.info()

                print("\nDescribe:")
                print(lstm_features_df.describe())

                print("\nTarget Variable Distribution (target_price_direction):")
                print(lstm_features_df['target_price_direction'].value_counts(normalize=True))
            else:
                print("\nFailed to create LSTM features or DataFrame is empty after NaN handling.")
        else:
            print(f"Could not fetch data for {symbol}. Cannot create features.")

        print("\nlstm_feature_engineer.py example finished.")
