import pandas as pd
import numpy as np
import joblib # For loading the scaler
import os # For checking file paths

# TensorFlow imports
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow is not installed. LSTM model filtering will not be possible.")
    TENSORFLOW_AVAILABLE = False


# --- Import functions from other project modules ---
try:
    from candlestick_patterns import add_candlestick_patterns
    CANDLESTICK_PATTERNS_AVAILABLE = True
except ImportError:
    print("Error: candlestick_patterns.py not found. Candlestick features will be skipped if generating features here.")
    CANDLESTICK_PATTERNS_AVAILABLE = False

try:
    from market_structure import find_swing_highs_lows, identify_simple_trend, find_support_resistance_from_swings, identify_pullbacks, identify_breakouts
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    print("Error: market_structure.py not found. Market structure features will be skipped if generating features here.")
    MARKET_STRUCTURE_AVAILABLE = False

try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    print("Error: data_fetcher.py not found. Cannot run example without it.")
    DATA_FETCHER_AVAILABLE = False

try:
    import config # Import the configuration file
except ImportError:
    print("Error: config.py not found. Using fallback internal defaults for signal_generator.")
    class config:
        SYMBOL = 'AAPL'; PRIMARY_INTERVAL_BACKTEST = '1h'; PRIMARY_PERIOD_BACKTEST = '60d'; HIGHER_TF_INTERVAL = '1d';
        MS_SWING_WINDOW = 10; MS_PULLBACK_TREND_LOOKBACK = 2; MS_BREAKOUT_LOOKBACK_PERIOD = 20;
        MS_SR_MIN_TOUCHES = 2; MS_SR_RELATIVE_TOLERANCE = 0.015;
        PULLBACK_TREND_LOOKBACK_SWINGS = MS_PULLBACK_TREND_LOOKBACK # For compatibility if used directly
        SR_MIN_TOUCHES = MS_SR_MIN_TOUCHES # For compatibility
        SR_RELATIVE_TOLERANCE = MS_SR_RELATIVE_TOLERANCE # For compatibility
        SWING_WINDOW = MS_SWING_WINDOW # For compatibility
        BREAKOUT_LOOKBACK_PERIOD = MS_BREAKOUT_LOOKBACK_PERIOD # For compatibility

        SG_SR_NEARBY_PERCENTAGE = 0.02; SG_STOP_LOSS_BUFFER_PERCENTAGE = 0.005; SG_REWARD_RATIO = 2.0;
        LSTM_MODEL_PATH = 'best_lstm_model.keras'; LSTM_SCALER_PATH = 'lstm_scaler.gz';
        SG_LSTM_SEQUENCE_LENGTH = 20; SG_LSTM_BUY_THRESHOLD = 0.55; SG_LSTM_SELL_THRESHOLD = 0.45;
        SG_USE_LSTM_FILTER = True; MODEL_DIR = ".";
        EX_SG_PRIMARY_SYMBOL = SYMBOL; EX_SG_PRIMARY_INTERVAL = PRIMARY_INTERVAL_BACKTEST;
        EX_SG_PRIMARY_PERIOD = PRIMARY_PERIOD_BACKTEST; EX_SG_HIGHER_TF_INTERVAL = HIGHER_TF_INTERVAL;
        EX_PREPROC_SWING_WINDOW = MS_SWING_WINDOW; EX_PREPROC_BREAKOUT_LOOKBACK = MS_BREAKOUT_LOOKBACK_PERIOD;
        EX_PREPROC_PULLBACK_LOOKBACK = MS_PULLBACK_TREND_LOOKBACK;
        EX_PREPROC_SR_MIN_TOUCHES = MS_SR_MIN_TOUCHES; EX_PREPROC_SR_TOLERANCE = MS_SR_RELATIVE_TOLERANCE;
        LSTM_FEATURE_COLS = [ # Fallback LSTM_FEATURE_COLS
            'return', 'body_range_norm', 'hl_range_norm', 'Volume', 'is_doji', 'is_marubozu',
            'is_outside_bar', 'is_inside_bar', 'is_swing_high', 'is_swing_low', 'is_pullback_bar',
            'is_bullish_breakout', 'is_bearish_breakout']


# Feature columns list - must match what the LSTM model was trained on
LSTM_FEATURE_COLS = config.LSTM_FEATURE_COLS


def pre_process_data_for_signals(ohlcv_df,
                                 swing_window=config.MS_SWING_WINDOW,
                                 breakout_lookback=config.MS_BREAKOUT_LOOKBACK_PERIOD,
                                 pullback_trend_lookback=config.MS_PULLBACK_TREND_LOOKBACK,
                                 sr_min_touches=config.MS_SR_MIN_TOUCHES,
                                 sr_tolerance=config.MS_SR_RELATIVE_TOLERANCE
                                 ):
    """Applies all necessary feature engineering steps before signal generation."""
    if ohlcv_df is None or ohlcv_df.empty: return pd.DataFrame()
    df = ohlcv_df.copy()
    print("Pre-processing: Calculating base OHLCV features...")
    df['return'] = df['Close'].pct_change()
    df['body_abs'] = abs(df['Close'] - df['Open'])
    df['hl_range'] = df['High'] - df['Low']

    df['body_range_norm'] = np.where(df['hl_range'] == 0, 0.0, df['body_abs'] / df['hl_range'])
    df['body_range_norm'] = df['body_range_norm'].fillna(0.0)

    df['hl_range_norm'] = np.where(df['Close'] == 0, 0.0, df['hl_range'] / df['Close'])
    df['hl_range_norm'] = df['hl_range_norm'].fillna(0.0)

    if 'Volume' not in df.columns:
        print("Warning: 'Volume' column missing in input for pre_process. Adding dummy column with zeros.")
        df['Volume'] = 0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

    if CANDLESTICK_PATTERNS_AVAILABLE:
        print("Pre-processing: Adding candlestick patterns using config parameters...")
        df = add_candlestick_patterns(
            df,
            doji_tolerance_ratio=config.CP_DOJI_BODY_TOLERANCE_RATIO if hasattr(config, 'CP_DOJI_BODY_TOLERANCE_RATIO') else 0.05,
            marubozu_body_min_ratio=config.CP_MARUBOZU_BODY_MIN_RATIO if hasattr(config, 'CP_MARUBOZU_BODY_MIN_RATIO') else 0.8,
            marubozu_wick_max_ratio=config.CP_MARUBOZU_WICK_MAX_RATIO if hasattr(config, 'CP_MARUBOZU_WICK_MAX_RATIO') else 0.1
        )
    else:
        # Fallback: if candlestick_patterns module is not available, create dummy columns for features listed in config
        # This ensures that the dataframe has the columns expected by later stages, e.g. LSTM model prediction
        print("Warning: candlestick_patterns.py module not available. Creating dummy candlestick columns based on config.LSTM_FEATURE_COLS.")
        candlestick_cols_from_config = [
            'is_doji', 'is_bullish_marubozu', 'is_bearish_marubozu', 'is_marubozu', # Ensure marubozu split is handled or base is present
            'is_bullish_engulfing', 'is_bearish_engulfing',
            'is_shooting_star', 'is_hanging_man', 'is_hammer', 'is_inverted_hammer',
            'is_inside_bar', 'is_outside_bar'
        ]
        # Check which of these are actually in LSTM_FEATURE_COLS
        relevant_candlestick_cols = [col for col in candlestick_cols_from_config if col in config.LSTM_FEATURE_COLS]
        for p_col in relevant_candlestick_cols:
            df[p_col] = False
        # Special handling for is_marubozu if its components are used
        if 'is_marubozu' in relevant_candlestick_cols and 'is_bullish_marubozu' not in relevant_candlestick_cols and 'is_bearish_marubozu' not in relevant_candlestick_cols:
            pass # is_marubozu would be a direct column
        elif 'is_bullish_marubozu' in relevant_candlestick_cols or 'is_bearish_marubozu' in relevant_candlestick_cols:
             if 'is_marubozu' not in df.columns: df['is_marubozu'] = False # Ensure base is_marubozu if components are used by LSTM


    if MARKET_STRUCTURE_AVAILABLE:
        print("Pre-processing: Adding market structure features (swings, pullbacks, breakouts)...")
        df = find_swing_highs_lows(df, window=swing_window)
        df = identify_pullbacks(df, trend_lookback_swings=pullback_trend_lookback)
        df = identify_breakouts(df, lookback_period=breakout_lookback,
                                sr_min_touches=sr_min_touches, sr_tolerance=sr_tolerance)
    else:
        for ms_col in ['is_swing_high', 'is_swing_low', 'is_pullback_bar',
                       'is_bullish_breakout', 'is_bearish_breakout']: df[ms_col] = False

    df.dropna(subset=['return'], inplace=True)
    return df

def generate_initial_signals(
    primary_df_input: pd.DataFrame,
    primary_symbol: str,
    primary_interval: str,
    higher_tf_df_input: pd.DataFrame | None,
    sr_nearby_percentage: float = config.SG_SR_NEARBY_PERCENTAGE,
    stop_loss_buffer_percentage: float = config.SG_STOP_LOSS_BUFFER_PERCENTAGE,
    reward_ratio: float = config.SG_REWARD_RATIO,
    lstm_model_path: str = config.LSTM_MODEL_PATH,
    scaler_path: str = config.LSTM_SCALER_PATH,
    lstm_sequence_length: int = config.SG_LSTM_SEQUENCE_LENGTH,
    lstm_buy_threshold: float = config.SG_LSTM_BUY_THRESHOLD,
    lstm_sell_threshold: float = config.SG_LSTM_SELL_THRESHOLD,
    use_lstm_filter: bool = config.SG_USE_LSTM_FILTER
) -> tuple[pd.DataFrame | None, str | None, str | None, dict | None]:
    """
    Generates trading signals, optionally filtered by an LSTM model.
    """

    primary_df = primary_df_input.copy()

    model = None; scaler = None; actual_use_lstm_filter = use_lstm_filter

    if actual_use_lstm_filter:
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available, disabling LSTM filter."); actual_use_lstm_filter = False
        else:
            if not (os.path.exists(lstm_model_path) and os.path.exists(scaler_path)):
                print(f"Warning: LSTM model ({lstm_model_path}) or scaler ({scaler_path}) file not found. Disabling LSTM filter.")
                actual_use_lstm_filter = False
            else:
                try:
                    model = load_model(lstm_model_path); print(f"LSTM model loaded from {lstm_model_path}")
                    scaler = joblib.load(scaler_path); print(f"Scaler loaded from {scaler_path}")
                except Exception as e:
                    print(f"Warning: Error loading LSTM model/scaler. Error: {e}. Disabling LSTM filter."); actual_use_lstm_filter = False
        if not actual_use_lstm_filter: model = None; scaler = None # Ensure they are None if filter disabled

    overall_higher_tf_trend = "Unavailable"
    if higher_tf_df_input is not None and not higher_tf_df_input.empty and MARKET_STRUCTURE_AVAILABLE:
        print(f"Analyzing {len(higher_tf_df_input)} higher timeframe data points...")
        htf_df_analyzed = find_swing_highs_lows(higher_tf_df_input.copy(), window=config.MS_SWING_WINDOW)
        overall_higher_tf_trend = identify_simple_trend(htf_df_analyzed, lookback_swings=config.MS_PULLBACK_TREND_LOOKBACK) if htf_df_analyzed['is_swing_high'].any() or htf_df_analyzed['is_swing_low'].any() else "Undetermined"
    elif higher_tf_df_input is None or higher_tf_df_input.empty:
        overall_higher_tf_trend = None; print("Warning: No HTF data. HTF trend context is None.")
    else: overall_higher_tf_trend = None; print("Warning: Market structure module unavailable for HTF. HTF trend is None.")
    print(f"Overall identified trend for HIGHER TIMEFRAME: {overall_higher_tf_trend}")

    primary_trend = "Unavailable"
    required_cols = ['is_swing_high', 'is_swing_low']
    if MARKET_STRUCTURE_AVAILABLE and all(col in primary_df.columns and isinstance(primary_df[col], pd.Series) and primary_df[col].dtype == 'bool' for col in required_cols):
        primary_trend = identify_simple_trend(primary_df.copy(), lookback_swings=config.MS_PULLBACK_TREND_LOOKBACK) if primary_df['is_swing_high'].any() or primary_df['is_swing_low'].any() else "Undetermined"
        sr_levels = find_support_resistance_from_swings(primary_df.copy(), min_touches=config.MS_SR_MIN_TOUCHES, relative_tolerance=config.MS_SR_RELATIVE_TOLERANCE)
    else:
        primary_trend = "Undetermined"; sr_levels = {'support': [], 'resistance': []}
        print(f"Warning: Swing columns missing/invalid or market_structure module not available for primary_df. Primary trend/S&R set to defaults. Found columns: {primary_df.columns.tolist()}")


    print(f"Overall identified trend for PRIMARY TIMEFRAME ({primary_interval}): {primary_trend}")
    support_levels = sr_levels.get('support', []); resistance_levels = sr_levels.get('resistance', [])
    print(f"Identified Support Levels (Primary TF): {support_levels}")
    print(f"Identified Resistance Levels (Primary TF): {resistance_levels}")

    primary_df['signal'] = 0; primary_df['stop_loss'] = np.nan; primary_df['take_profit'] = np.nan
    primary_df['lstm_prediction'] = np.nan

    if 'is_bullish_marubozu' not in primary_df.columns: primary_df['is_bullish_marubozu'] = primary_df.get('is_marubozu', False) & (primary_df['Close'] > primary_df['Open'])
    if 'is_bearish_marubozu' not in primary_df.columns: primary_df['is_bearish_marubozu'] = primary_df.get('is_marubozu', False) & (primary_df['Close'] < primary_df['Open'])

    print("\nGenerating signals with SL/TP, potentially filtered by LSTM...")
    for current_bar_iloc in range(len(primary_df)):
        index = primary_df.index[current_bar_iloc]; row = primary_df.iloc[current_bar_iloc]
        lstm_confirms_buy = False; lstm_confirms_sell = False; lstm_prediction_made_this_bar = False

        if actual_use_lstm_filter:
            sequence_start_iloc = current_bar_iloc - lstm_sequence_length + 1
            if sequence_start_iloc >= 0:
                sequence_df_iloc = primary_df.iloc[sequence_start_iloc : current_bar_iloc + 1]
                if len(sequence_df_iloc) == lstm_sequence_length:
                    sequence_data_for_lstm = sequence_df_iloc[LSTM_FEATURE_COLS].copy()
                    all_cols_present = True
                    for col_name in LSTM_FEATURE_COLS:
                        if col_name not in sequence_data_for_lstm: all_cols_present = False; break
                        if sequence_data_for_lstm[col_name].dtype == 'bool': sequence_data_for_lstm[col_name] = sequence_data_for_lstm[col_name].astype(int)
                        if sequence_data_for_lstm[col_name].dtype == 'object':
                             try: sequence_data_for_lstm[col_name] = pd.to_numeric(sequence_data_for_lstm[col_name])
                             except ValueError: sequence_data_for_lstm[col_name] = 0; print(f"Warning: Col {col_name} to numeric failed.")

                    if not all_cols_present: print(f"Warning: Missing LSTM features at index {index}. LSTM filter skipped.")
                    elif sequence_data_for_lstm.isnull().values.any(): pass
                    else:
                        try:
                            sequence_features_numeric = sequence_data_for_lstm.astype(np.float64).values
                            scaled_sequence_features = scaler.transform(sequence_features_numeric)
                            reshaped_features = np.reshape(scaled_sequence_features, (1, lstm_sequence_length, len(LSTM_FEATURE_COLS)))
                            lstm_pred_value = model.predict(reshaped_features, verbose=0)[0][0]
                            primary_df.loc[index, 'lstm_prediction'] = lstm_pred_value
                            lstm_prediction_made_this_bar = True
                            if lstm_pred_value >= lstm_buy_threshold: lstm_confirms_buy = True
                            if lstm_pred_value <= lstm_sell_threshold: lstm_confirms_sell = True
                        except Exception as e: print(f"Error during LSTM prediction for bar {index}: {e}.")

        rule_based_buy_signal = False; rule_based_sell_signal = False
        if primary_trend == "Uptrend":
            if overall_higher_tf_trend == "Uptrend" or overall_higher_tf_trend is None or overall_higher_tf_trend in ["Unavailable", "Undetermined"]:
                if row.get('is_bullish_marubozu', False):
                    is_near_support = False if support_levels else True
                    for s_level in support_levels:
                        if s_level > 0 and abs(row['Low'] - s_level) / s_level <= sr_nearby_percentage: is_near_support = True; break
                    if is_near_support: rule_based_buy_signal = True
        elif primary_trend == "Downtrend":
            if overall_higher_tf_trend == "Downtrend" or overall_higher_tf_trend is None or overall_higher_tf_trend in ["Unavailable", "Undetermined"]:
                if row.get('is_bearish_marubozu', False):
                    is_near_resistance = False if resistance_levels else True
                    for r_level in resistance_levels:
                        if r_level > 0 and abs(row['High'] - r_level) / r_level <= sr_nearby_percentage: is_near_resistance = True; break
                    if is_near_resistance: rule_based_sell_signal = True

        final_signal = 0
        if rule_based_buy_signal:
            if actual_use_lstm_filter: final_signal = 1 if lstm_prediction_made_this_bar and lstm_confirms_buy else 0
            else: final_signal = 1
        elif rule_based_sell_signal:
            if actual_use_lstm_filter: final_signal = -1 if lstm_prediction_made_this_bar and lstm_confirms_sell else 0
            else: final_signal = -1

        if final_signal != 0:
            primary_df.loc[index, 'signal'] = final_signal; entry_price = row['Close']
            if final_signal == 1:
                calculated_sl = row['Low'] * (1 - stop_loss_buffer_percentage)
                primary_df.loc[index, 'stop_loss'] = calculated_sl
                if calculated_sl < entry_price :
                    risk_amount = entry_price - calculated_sl
                    if risk_amount > 1e-9: primary_df.loc[index, 'take_profit'] = entry_price + (risk_amount * reward_ratio)
                    else: primary_df.loc[index, 'signal'] = 0; primary_df.loc[index, 'stop_loss'] = np.nan
                else: primary_df.loc[index, 'signal'] = 0; primary_df.loc[index, 'stop_loss'] = np.nan
            elif final_signal == -1:
                calculated_sl = row['High'] * (1 + stop_loss_buffer_percentage)
                primary_df.loc[index, 'stop_loss'] = calculated_sl
                if calculated_sl > entry_price:
                    risk_amount = calculated_sl - entry_price
                    if risk_amount > 1e-9: primary_df.loc[index, 'take_profit'] = entry_price - (risk_amount * reward_ratio)
                    else: primary_df.loc[index, 'signal'] = 0; primary_df.loc[index, 'stop_loss'] = np.nan
                else: primary_df.loc[index, 'signal'] = 0; primary_df.loc[index, 'stop_loss'] = np.nan
    return primary_df, overall_higher_tf_trend, primary_trend, sr_levels

if __name__ == "__main__":
    if not DATA_FETCHER_AVAILABLE:
        print("Data fetcher not available. Cannot run signal_generator example.")
    else:
        print("Running signal_generator.py MTF example with LSTM filter and SL/TP using config...")
        print("\n--- Configuration Parameters ---")
        print(f"Symbol for Signal Generation: {config.EX_SG_PRIMARY_SYMBOL}")
        print(f"Primary Interval: {config.EX_SG_PRIMARY_INTERVAL}")
        print(f"Primary Period: {config.EX_SG_PRIMARY_PERIOD}")
        print(f"Higher Timeframe Interval: {config.EX_SG_HIGHER_TF_INTERVAL}")
        print(f"Use LSTM Filter: {config.SG_USE_LSTM_FILTER}")
        if config.SG_USE_LSTM_FILTER:
            print(f"  LSTM Model Path: {config.LSTM_MODEL_PATH}")
            print(f"  LSTM Scaler Path: {config.LSTM_SCALER_PATH}")
            print(f"  LSTM Buy Threshold: {config.SG_LSTM_BUY_THRESHOLD}")
            print(f"  LSTM Sell Threshold: {config.SG_LSTM_SELL_THRESHOLD}")
        print("---------------------------------")

        ohlcv_data_main = fetch_price_data(
            symbol=config.EX_SG_PRIMARY_SYMBOL,
            interval=config.EX_SG_PRIMARY_INTERVAL,
            period=config.EX_SG_PRIMARY_PERIOD )

        htf_df_main = None
        if ohlcv_data_main is not None and not ohlcv_data_main.empty:
            if not isinstance(ohlcv_data_main.index, pd.DatetimeIndex):
                ohlcv_data_main.index = pd.to_datetime(ohlcv_data_main.index)
            htf_start = ohlcv_data_main.index.min().strftime('%Y-%m-%d')
            htf_end = (ohlcv_data_main.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            htf_df_main = fetch_price_data(
                symbol=config.EX_SG_PRIMARY_SYMBOL, interval=config.EX_SG_HIGHER_TF_INTERVAL,
                start_date=htf_start, end_date=htf_end )

        if ohlcv_data_main is not None and not ohlcv_data_main.empty:
            print("\nPre-processing primary data to add all features for LSTM and rules...")
            enriched_primary_df = pre_process_data_for_signals(
                ohlcv_data_main,
                swing_window=config.EX_PREPROC_SWING_WINDOW,
                breakout_lookback=config.EX_PREPROC_BREAKOUT_LOOKBACK,
                pullback_trend_lookback=config.EX_PREPROC_PULLBACK_LOOKBACK,
                sr_min_touches=config.EX_PREPROC_SR_MIN_TOUCHES,
                sr_tolerance=config.EX_PREPROC_SR_TOLERANCE )
            print(f"Enriched primary DataFrame shape: {enriched_primary_df.shape}")

            if 'Volume' in enriched_primary_df.columns:
                enriched_primary_df['Volume'] = pd.to_numeric(enriched_primary_df['Volume'], errors='coerce').fillna(0)

            missing_feature_cols_for_lstm = [col for col in LSTM_FEATURE_COLS if col not in enriched_primary_df.columns]
            if missing_feature_cols_for_lstm:
                print(f"FATAL ERROR: Enriched DF is missing required LSTM feature columns: {missing_feature_cols_for_lstm}")
            else:
                for col in LSTM_FEATURE_COLS:
                    if enriched_primary_df[col].isnull().any():
                        if enriched_primary_df[col].dtype == 'bool':
                             enriched_primary_df[col] = enriched_primary_df[col].fillna(False)
                        else:
                             print(f"Warning: Numeric/Object column '{col}' has NaNs post pre-processing. Filling with 0.")
                             enriched_primary_df[col] = enriched_primary_df[col].fillna(0)
                signals_result_tuple = generate_initial_signals(
                    primary_df_input=enriched_primary_df,
                    primary_symbol=config.EX_SG_PRIMARY_SYMBOL,
                    primary_interval=config.EX_SG_PRIMARY_INTERVAL,
                    higher_tf_df_input=htf_df_main
                    # Rely on function defaults for other parameters which now pull from config
                )
                signals_df, htf_trend, ptf_trend, ptf_sr_levels = signals_result_tuple

                if signals_df is not None:
                    print(f"\n--- Analysis Summary ---")
                    print(f"Overall Primary Trend ({config.EX_SG_PRIMARY_INTERVAL}): {ptf_trend}")
                    print(f"Overall Higher Timeframe Trend ({config.EX_SG_HIGHER_TF_INTERVAL}): {htf_trend}")
                    if ptf_sr_levels:
                        print(f"Identified Support Levels (Primary TF): {ptf_sr_levels.get('support', [])}")
                        print(f"Identified Resistance Levels (Primary TF): {ptf_sr_levels.get('resistance', [])}")
                    else:
                        print("Support/Resistance levels not available from signal generation output.")

                    output_df = signals_df.copy()
                    cols_to_show = [
                        'Open', 'High', 'Low', 'Close', 'Volume', 'signal',
                        'lstm_prediction', 'stop_loss', 'take_profit',
                        'is_marubozu', 'is_pullback_bar',
                        'is_bullish_breakout', 'is_bearish_breakout' # Corrected: is_bullish_breakout, is_bearish_breakout
                    ]
                    # Ensure is_marubozu exists (it might be split into bullish/bearish in some versions)
                    if 'is_marubozu' not in output_df.columns and 'is_bullish_marubozu' in output_df.columns : # crude check
                         output_df['is_marubozu'] = output_df['is_bullish_marubozu'] | output_df['is_bearish_marubozu']

                    relevant_cols = [col for col in cols_to_show if col in output_df.columns]

                    recent_activity_df = output_df[output_df['signal'] != 0][relevant_cols]

                    if recent_activity_df.empty:
                        print("\nNo signals generated in this run. Showing last 15 bars with key features:")
                        # Ensure 'lstm_prediction' is included if filter was on
                        cols_for_no_signal_view = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if 'lstm_prediction' in output_df.columns:
                            cols_for_no_signal_view.append('lstm_prediction')

                        # Add a few other key boolean features if available
                        for bool_col in ['is_marubozu', 'is_pullback_bar', 'is_bullish_breakout', 'is_bearish_breakout']:
                            if bool_col in output_df.columns:
                                cols_for_no_signal_view.append(bool_col)

                        print(output_df[cols_for_no_signal_view].tail(15).to_string())
                        if 'lstm_prediction' not in output_df.columns and config.SG_USE_LSTM_FILTER:
                            print("(Note: LSTM prediction column was not found in the output.)")
                    else:
                        print("\n--- Generated Signals ---")
                        print(recent_activity_df.to_string())
                else:
                    print("\nFailed to generate signals (signals_df is None).")
        else:
            print(f"Could not fetch primary data for {config.EX_SG_PRIMARY_SYMBOL}. Signal generation aborted.")
        print("\nsignal_generator.py MTF example with LSTM filter and SL/TP finished.")
