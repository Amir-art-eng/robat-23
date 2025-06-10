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
    from market_structure import (
        find_swing_highs_lows, identify_simple_trend,
        find_support_resistance_from_swings, identify_pullbacks,
        identify_breakouts, calculate_atr, # Existing
        find_double_tops_bottoms, calculate_daily_pivot_points # New
    )
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    print("Error: market_structure.py (or specific functions like find_double_tops_bottoms, calculate_daily_pivot_points) not found. "
          "Market structure features, ATR, DT/DB, and Pivot calculations might be skipped if generating features here.")
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
    class config: # Fallback config class
        SYMBOL = 'AAPL'; PRIMARY_INTERVAL_BACKTEST = '1h'; PRIMARY_PERIOD_BACKTEST = '60d'; HIGHER_TF_INTERVAL = '1d';
        MS_SWING_WINDOW = 10; MS_PULLBACK_TREND_LOOKBACK = 2; MS_BREAKOUT_LOOKBACK_PERIOD = 20;
        MS_SR_MIN_TOUCHES = 2; MS_SR_RELATIVE_TOLERANCE = 0.015;
        CP_DOJI_BODY_TOLERANCE_RATIO = 0.05; CP_MARUBOZU_BODY_MIN_RATIO = 0.8; CP_MARUBOZU_WICK_MAX_RATIO = 0.1;
        SG_SR_NEARBY_PERCENTAGE = 0.02; SG_STOP_LOSS_BUFFER_PERCENTAGE = 0.005; SG_REWARD_RATIO = 2.0;
        LSTM_MODEL_PATH = 'best_lstm_model.keras'; LSTM_SCALER_PATH = 'lstm_scaler.gz';
        SG_LSTM_SEQUENCE_LENGTH = 20; SG_LSTM_BUY_THRESHOLD = 0.55; SG_LSTM_SELL_THRESHOLD = 0.45;
        SG_USE_LSTM_FILTER = True; MODEL_DIR = ".";
        EX_SG_PRIMARY_SYMBOL = SYMBOL; EX_SG_PRIMARY_INTERVAL = PRIMARY_INTERVAL_BACKTEST;
        EX_SG_PRIMARY_PERIOD = PRIMARY_PERIOD_BACKTEST; EX_SG_HIGHER_TF_INTERVAL = HIGHER_TF_INTERVAL;
        EX_PREPROC_SWING_WINDOW = MS_SWING_WINDOW; EX_PREPROC_BREAKOUT_LOOKBACK = MS_BREAKOUT_LOOKBACK_PERIOD;
        EX_PREPROC_PULLBACK_LOOKBACK = MS_PULLBACK_TREND_LOOKBACK;
        EX_PREPROC_SR_MIN_TOUCHES = MS_SR_MIN_TOUCHES; EX_PREPROC_SR_TOLERANCE = MS_SR_RELATIVE_TOLERANCE;
        SG_USE_ATR_STOP = False; SG_ATR_PERIOD = 14; SG_ATR_MULTIPLIER = 2.0; # ATR Fallbacks
        LSTM_FEATURE_COLS = [
            'return', 'body_range_norm', 'hl_range_norm', 'Volume', 'is_doji', 'is_marubozu',
            'is_outside_bar', 'is_inside_bar', 'is_swing_high', 'is_swing_low', 'is_pullback_bar',
            'is_bullish_breakout', 'is_bearish_breakout']

LSTM_FEATURE_COLS = config.LSTM_FEATURE_COLS


def pre_process_data_for_signals(ohlcv_df, config_module):
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
        df['Volume'] = 0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

    cp_doji_tol = getattr(config_module, 'CP_DOJI_BODY_TOLERANCE_RATIO', 0.05)
    cp_maru_body_min = getattr(config_module, 'CP_MARUBOZU_BODY_MIN_RATIO', 0.8)
    cp_maru_wick_max = getattr(config_module, 'CP_MARUBOZU_WICK_MAX_RATIO', 0.1)
    ms_sw_window = getattr(config_module, 'MS_SWING_WINDOW', 10)
    ms_bo_lookback = getattr(config_module, 'MS_BREAKOUT_LOOKBACK_PERIOD', 20)
    ms_pt_lookback = getattr(config_module, 'MS_PULLBACK_TREND_LOOKBACK', 2)
    ms_sr_touches = getattr(config_module, 'MS_SR_MIN_TOUCHES', 2)
    ms_sr_tol = getattr(config_module, 'MS_SR_RELATIVE_TOLERANCE', 0.015)

    if CANDLESTICK_PATTERNS_AVAILABLE:
        df = add_candlestick_patterns(df, doji_tolerance_ratio=cp_doji_tol,
                                      marubozu_body_min_ratio=cp_maru_body_min,
                                      marubozu_wick_max_ratio=cp_maru_wick_max)
    else:
        lstm_features_list = getattr(config_module, 'LSTM_FEATURE_COLS', [])
        candlestick_cols = [col for col in ['is_doji', 'is_marubozu', 'is_bullish_marubozu', 'is_bearish_marubozu',
                                           'is_bullish_engulfing', 'is_bearish_engulfing', 'is_shooting_star',
                                           'is_hanging_man', 'is_hammer', 'is_inverted_hammer',
                                           'is_inside_bar', 'is_outside_bar'] if col in lstm_features_list]
        for p_col in candlestick_cols: df[p_col] = False

    if MARKET_STRUCTURE_AVAILABLE:
        df = find_swing_highs_lows(df, window=ms_sw_window)
        df = identify_pullbacks(df, trend_lookback_swings=ms_pt_lookback)
        df = identify_breakouts(df, lookback_period=ms_bo_lookback,
                                sr_min_touches=ms_sr_touches, sr_tolerance=ms_sr_tol)
    else:
        for ms_col in ['is_swing_high', 'is_swing_low', 'is_pullback_bar',
                       'is_bullish_breakout', 'is_bearish_breakout']: df[ms_col] = False

    df.dropna(subset=['return'], inplace=True)

    # Double Tops/Bottoms calculation
    if MARKET_STRUCTURE_AVAILABLE and 'find_double_tops_bottoms' in globals():
        dtb_lookback = getattr(config_module, 'MS_DTB_LOOKBACK_SWINGS', 3)
        dtb_sim_pct = getattr(config_module, 'MS_DTB_SIMILARITY_PCT', 0.03)
        dtb_conf_ratio = getattr(config_module, 'MS_DTB_CONFIRMATION_RATIO', 0.3)
        print(f"Pre-processing: Calculating Double Tops/Bottoms with lookback={dtb_lookback}, sim_pct={dtb_sim_pct}, conf_ratio={dtb_conf_ratio}...")
        df = find_double_tops_bottoms(df,
                                      lookback_swings_for_pattern=dtb_lookback,
                                      price_similarity_pct=dtb_sim_pct,
                                      confirmation_break_ratio=dtb_conf_ratio)
    else:
        if 'is_double_top_confirmed' not in df.columns: df['is_double_top_confirmed'] = False
        if 'is_double_bottom_confirmed' not in df.columns: df['is_double_bottom_confirmed'] = False


    # Calculate ATR if SG_USE_ATR_STOP is true OR if ATR is intended as an LSTM feature
    should_calculate_atr = (
        getattr(config_module, 'SG_USE_ATR_STOP', False) or \
        getattr(config_module, 'LSTM_USE_ATR_FEATURE', False)
    )

    if should_calculate_atr and MARKET_STRUCTURE_AVAILABLE and 'calculate_atr' in globals():
        atr_period = getattr(config_module, 'SG_ATR_PERIOD', 14)
        print(f"Pre-processing: Calculating ATR with period {atr_period}...")
        df = calculate_atr(df, period=atr_period)
    elif 'atr' not in df.columns:
        print("Pre-processing: ATR calculation not requested or not possible. Ensuring 'atr' column exists with NaNs.")
        df['atr'] = np.nan

    # Daily pivots are not added as columns here; they are fetched once in generate_initial_signals
    # and looked up per bar, so no changes for pivots in pre_process_data_for_signals.
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
    use_lstm_filter: bool = config.SG_USE_LSTM_FILTER,
    use_atr_stop: bool = getattr(config, 'SG_USE_ATR_STOP', False),
    atr_multiplier: float = getattr(config, 'SG_ATR_MULTIPLIER', 2.0)
) -> tuple[pd.DataFrame | None, str | None, str | None, dict | None]:

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
        if not actual_use_lstm_filter: model = None; scaler = None

    overall_higher_tf_trend = "Unavailable"
    if higher_tf_df_input is not None and not higher_tf_df_input.empty and MARKET_STRUCTURE_AVAILABLE:
        htf_df_analyzed = find_swing_highs_lows(higher_tf_df_input.copy(), window=getattr(config, 'MS_SWING_WINDOW', 10))
        overall_higher_tf_trend = identify_simple_trend(htf_df_analyzed, lookback_swings=getattr(config, 'MS_PULLBACK_TREND_LOOKBACK', 2)) if htf_df_analyzed['is_swing_high'].any() or htf_df_analyzed['is_swing_low'].any() else "Undetermined"
    elif higher_tf_df_input is None or higher_tf_df_input.empty:
        overall_higher_tf_trend = None
    print(f"Overall identified trend for HIGHER TIMEFRAME: {overall_higher_tf_trend}")

    primary_trend = "Unavailable"; sr_levels = {'support': [], 'resistance': []}
    if MARKET_STRUCTURE_AVAILABLE and all(col in primary_df.columns for col in ['is_swing_high', 'is_swing_low']):
        primary_trend = identify_simple_trend(primary_df.copy(), lookback_swings=getattr(config, 'MS_PULLBACK_TREND_LOOKBACK',2)) if primary_df['is_swing_high'].any() or primary_df['is_swing_low'].any() else "Undetermined"
        sr_levels = find_support_resistance_from_swings(primary_df.copy(),
                                                        min_touches=getattr(config, 'MS_SR_MIN_TOUCHES',2),
                                                        relative_tolerance=getattr(config, 'MS_SR_RELATIVE_TOLERANCE',0.015))
    print(f"Overall identified trend for PRIMARY TIMEFRAME ({primary_interval}): {primary_trend}")
    print(f"Identified Support Levels (Primary TF): {sr_levels.get('support', [])}")
    print(f"Identified Resistance Levels (Primary TF): {sr_levels.get('resistance', [])}")

    primary_df['signal'] = 0; primary_df['stop_loss'] = np.nan; primary_df['take_profit'] = np.nan
    primary_df['lstm_prediction'] = np.nan

    if 'is_bullish_marubozu' not in primary_df.columns: primary_df['is_bullish_marubozu'] = primary_df.get('is_marubozu', False) & (primary_df['Close'] > primary_df['Open'])
    if 'is_bearish_marubozu' not in primary_df.columns: primary_df['is_bearish_marubozu'] = primary_df.get('is_marubozu', False) & (primary_df['Close'] < primary_df['Open'])
    if 'is_double_top_confirmed' not in primary_df.columns: primary_df['is_double_top_confirmed'] = False
    if 'is_double_bottom_confirmed' not in primary_df.columns: primary_df['is_double_bottom_confirmed'] = False

    # --- Pivot Filter Setup ---
    daily_pivots_dict = {}
    use_pivot_filter_config = getattr(config, 'SG_USE_PIVOT_FILTER', False)

    if use_pivot_filter_config and MARKET_STRUCTURE_AVAILABLE and DATA_FETCHER_AVAILABLE and 'calculate_daily_pivot_points' in globals():
        if not isinstance(primary_df.index, pd.DatetimeIndex):
            try:
                primary_df.index = pd.to_datetime(primary_df.index)
                print("Converted primary_df index to DatetimeIndex for pivot calculations.")
            except Exception as e:
                print(f"Warning: Could not convert primary_df index to DatetimeIndex. Disabling pivot filter. Error: {e}")
                use_pivot_filter_config = False

        if use_pivot_filter_config: # Re-check after potential conversion failure
            start_date_pivots = primary_df.index.min().strftime('%Y-%m-%d')
            end_date_pivots = (primary_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # Ensure last day is included
            print(f"Fetching daily data for pivots for {primary_symbol} from {start_date_pivots} to {end_date_pivots}...")

            daily_ohlc_for_pivots = fetch_price_data(
                symbol=primary_symbol, # Use the same symbol as primary data
                interval='1d',
                start_date=start_date_pivots,
                end_date=end_date_pivots
            )

            if daily_ohlc_for_pivots is not None and not daily_ohlc_for_pivots.empty:
                daily_pivots_dict = calculate_daily_pivot_points(daily_ohlc_for_pivots)
                if not daily_pivots_dict:
                    print("Warning: Failed to calculate daily pivot points. Disabling pivot filter for this run.")
                    use_pivot_filter_config = False
                else:
                    print(f"Successfully calculated {len(daily_pivots_dict)} days of pivot points.")
            else:
                print(f"Warning: Failed to fetch daily data for pivot points for {primary_symbol}. Disabling pivot filter for this run.")
                use_pivot_filter_config = False
    elif use_pivot_filter_config: # If any dependency was missing
        print("Warning: Pivot filter was enabled in config, but dependencies (market_structure, data_fetcher) are missing. Disabling pivot filter.")
        use_pivot_filter_config = False


    print("\nGenerating signals with SL/TP, DT/DB logic, Pivot filter, and potentially LSTM filter...")
    for current_bar_iloc in range(len(primary_df)):
        index = primary_df.index[current_bar_iloc]; row = primary_df.iloc[current_bar_iloc]

        final_signal = 0
        final_signal_priority = 0 # 0=None, 1=Marubozu/Rule, 2=DT/DB

        # --- 1. Double Top/Bottom Signals (Highest Priority) ---
        use_dtb_signals_config = getattr(config, 'SG_USE_DOUBLE_TOP_BOTTOM_SIGNALS', False)
        if use_dtb_signals_config:
            if row.get('is_double_top_confirmed', False):
                final_signal = -1
                final_signal_priority = 2
                # print(f"Debug: Double Top confirmed at {index}, signal: {final_signal}")
            elif row.get('is_double_bottom_confirmed', False):
                final_signal = 1
                final_signal_priority = 2
                # print(f"Debug: Double Bottom confirmed at {index}, signal: {final_signal}")

        # --- 2. Marubozu/Rule-based Signals (Lower Priority) ---
        if final_signal_priority < 2: # Only if DT/DB didn't fire
            potential_rule_signal = 0
            rule_based_buy_signal = False
            rule_based_sell_signal = False

            if primary_trend == "Uptrend":
                if overall_higher_tf_trend == "Uptrend" or overall_higher_tf_trend is None or overall_higher_tf_trend in ["Unavailable", "Undetermined"]:
                    if row.get('is_bullish_marubozu', False):
                        is_near_support = not sr_levels['support'] # True if no support levels defined (less restrictive)
                        for s_level in sr_levels['support']:
                            if s_level > 0 and abs(row['Low'] - s_level) / s_level <= sr_nearby_percentage:
                                is_near_support = True; break
                        if is_near_support: rule_based_buy_signal = True
            elif primary_trend == "Downtrend":
                if overall_higher_tf_trend == "Downtrend" or overall_higher_tf_trend is None or overall_higher_tf_trend in ["Unavailable", "Undetermined"]:
                    if row.get('is_bearish_marubozu', False):
                        is_near_resistance = not sr_levels['resistance'] # True if no resistance levels
                        for r_level in sr_levels['resistance']:
                            if r_level > 0 and abs(row['High'] - r_level) / r_level <= sr_nearby_percentage:
                                is_near_resistance = True; break
                        if is_near_resistance: rule_based_sell_signal = True

            if rule_based_buy_signal: potential_rule_signal = 1
            elif rule_based_sell_signal: potential_rule_signal = -1

            # --- Pivot Filter Application (Only for Marubozu/Rule signals) ---
            if potential_rule_signal != 0 and use_pivot_filter_config and daily_pivots_dict:
                current_bar_date_str = index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index).split(' ')[0]
                pivots_for_day = daily_pivots_dict.get(current_bar_date_str)

                passed_pivot_filter = False
                if pivots_for_day:
                    pivot_proximity_pct = getattr(config, 'SG_PIVOT_PROXIMITY_PCT', 0.005)
                    pivot_levels_to_check = getattr(config, 'SG_CONSIDER_PIVOT_LEVELS', ['PP', 'S1', 'R1'])

                    if potential_rule_signal == 1: # Buy signal, check support pivots
                        for level_key in pivot_levels_to_check:
                            if level_key.startswith('S') or level_key == 'PP':
                                level_price = pivots_for_day.get(level_key)
                                # Price bounced off pivot: Low is near/below pivot, Close is above/near pivot
                                if level_price and row['Low'] <= level_price * (1 + pivot_proximity_pct) and \
                                   row['Close'] >= level_price * (1 - pivot_proximity_pct):
                                    passed_pivot_filter = True; break
                    elif potential_rule_signal == -1: # Sell signal, check resistance pivots
                        for level_key in pivot_levels_to_check:
                            if level_key.startswith('R') or level_key == 'PP':
                                level_price = pivots_for_day.get(level_key)
                                # Price rejected at pivot: High is near/above pivot, Close is below/near pivot
                                if level_price and row['High'] >= level_price * (1 - pivot_proximity_pct) and \
                                   row['Close'] <= level_price * (1 + pivot_proximity_pct):
                                    passed_pivot_filter = True; break
                else: # No pivots for this day (e.g. holiday, data issue for that day's pivots)
                    # print(f"Debug: No pivots for day {current_bar_date_str}. Defaulting to pass filter.")
                    passed_pivot_filter = True # Default to pass if no pivots to check against for that specific day

                if passed_pivot_filter:
                    final_signal = potential_rule_signal
                    final_signal_priority = 1
                # else: final_signal remains 0 from this path (pivot filter failed)

            elif potential_rule_signal != 0: # Pivot filter not used or not applicable
                final_signal = potential_rule_signal
                final_signal_priority = 1

        # --- 3. LSTM Filter (Apply if signal is from rules (prio 1), not DT/DB (prio 2)) ---
        if final_signal_priority == 1 and actual_use_lstm_filter and model and scaler:
            lstm_prediction_value_this_bar = np.nan
            lstm_prediction_made_this_bar = False
            sequence_start_iloc = current_bar_iloc - lstm_sequence_length + 1
            if sequence_start_iloc >= 0:
                sequence_df_iloc = primary_df.iloc[sequence_start_iloc : current_bar_iloc + 1]
                if len(sequence_df_iloc) == lstm_sequence_length:
                    sequence_data_for_lstm = sequence_df_iloc[LSTM_FEATURE_COLS].copy()
                    all_cols_present_lstm = all(col in sequence_data_for_lstm for col in LSTM_FEATURE_COLS)

                    if all_cols_present_lstm and not sequence_data_for_lstm.isnull().values.any(): # Ensure no NaNs after selection
                        try:
                            # Ensure dtypes are correct (bools to int, objects to numeric)
                            for col_name_lstm in LSTM_FEATURE_COLS:
                                if sequence_data_for_lstm[col_name_lstm].dtype == 'bool':
                                    sequence_data_for_lstm[col_name_lstm] = sequence_data_for_lstm[col_name_lstm].astype(int)
                                # Objects should have been handled in pre_processing or are errors

                            sequence_features_numeric = sequence_data_for_lstm.astype(np.float64).values
                            scaled_sequence_features = scaler.transform(sequence_features_numeric)
                            reshaped_features = np.reshape(scaled_sequence_features, (1, lstm_sequence_length, len(LSTM_FEATURE_COLS)))
                            lstm_prediction_value_this_bar = model.predict(reshaped_features, verbose=0)[0][0]
                            primary_df.loc[index, 'lstm_prediction'] = lstm_prediction_value_this_bar
                            lstm_prediction_made_this_bar = True
                        except Exception as e:
                            print(f"Error during LSTM sequence processing/prediction for bar {index}: {e}.")
                    # else: print(f"Debug: LSTM sequence for {index} had NaNs or missing columns.")

            if lstm_prediction_made_this_bar:
                if not ((final_signal == 1 and lstm_prediction_value_this_bar >= lstm_buy_threshold) or \
                        (final_signal == -1 and lstm_prediction_value_this_bar <= lstm_sell_threshold)):
                    final_signal = 0 # LSTM filter rejects the rule-based signal
                    # print(f"Debug: LSTM rejected rule-based signal at {index}. Pred: {lstm_prediction_value_this_bar}, Signal: {primary_df.loc[index, 'signal_before_lstm']}")
            elif final_signal != 0: # LSTM prediction failed for a bar that had a rule-based signal
                print(f"Warning: LSTM prediction failed for bar {index} which had a rule signal {final_signal}. Signal voided.")
                final_signal = 0


        # --- SL/TP Calculation (Common for all signals that pass) ---
        if final_signal != 0:
            primary_df.loc[index, 'signal'] = final_signal
            entry_price = row['Close'] # Assuming entry at close of signal bar
            calculated_sl = np.nan

            # Determine reward ratio: specific for DT/DB if defined, else general
            current_reward_ratio = reward_ratio # General reward ratio
            if final_signal_priority == 2 and hasattr(config, 'SG_DBL_TOP_BOTTOM_REWARD_RATIO'):
                 current_reward_ratio = getattr(config, 'SG_DBL_TOP_BOTTOM_REWARD_RATIO', reward_ratio)

            if use_atr_stop and 'atr' in row and pd.notna(row['atr']) and row['atr'] > 0:
                current_atr = row['atr']
                if final_signal == 1:
                    calculated_sl = entry_price - (current_atr * atr_multiplier)
                elif final_signal == -1:
                    calculated_sl = entry_price + (current_atr * atr_multiplier)
            else:
                if use_atr_stop:
                    print(f"Warning: ATR stop enabled but ATR value not available/valid at {index} (ATR: {row.get('atr', 'N/A')}). Falling back to % stop for signal {final_signal}.")
                # Fallback to percentage-based stop loss
                if final_signal == 1:
                    calculated_sl = row['Low'] * (1 - stop_loss_buffer_percentage)
                elif final_signal == -1:
                    calculated_sl = row['High'] * (1 + stop_loss_buffer_percentage)

            primary_df.loc[index, 'stop_loss'] = calculated_sl

            # Validate trade and set Take Profit
            valid_trade = False
            if not pd.isna(calculated_sl):
                if final_signal == 1 and calculated_sl < entry_price: # Buy signal valid if SL is below entry
                    risk_amount = entry_price - calculated_sl
                    if risk_amount > 1e-9: # Ensure risk is meaningful
                        primary_df.loc[index, 'take_profit'] = entry_price + (risk_amount * current_reward_ratio)
                        valid_trade = True
                elif final_signal == -1 and calculated_sl > entry_price: # Sell signal valid if SL is above entry
                    risk_amount = calculated_sl - entry_price
                    if risk_amount > 1e-9: # Ensure risk is meaningful
                        primary_df.loc[index, 'take_profit'] = entry_price - (risk_amount * current_reward_ratio)
                        valid_trade = True

            if not valid_trade: # If SL/TP logic makes trade invalid (e.g., SL too close or on wrong side)
                primary_df.loc[index, 'signal'] = 0 # Reset signal
                primary_df.loc[index, 'stop_loss'] = np.nan
                primary_df.loc[index, 'take_profit'] = np.nan

    return primary_df, overall_higher_tf_trend, primary_trend, sr_levels

if __name__ == "__main__":
    if 'config' not in globals():
        print("Critical: config.py could not be imported in signal_generator __main__.")
        exit()

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
        print(f"Use ATR Stop: {getattr(config, 'SG_USE_ATR_STOP', False)}")
        if getattr(config, 'SG_USE_ATR_STOP', False):
            print(f"  ATR Period: {getattr(config, 'SG_ATR_PERIOD', 14)}")
            print(f"  ATR Multiplier: {getattr(config, 'SG_ATR_MULTIPLIER', 2.0)}")

        # New features config printout
        print(f"Use Pivot Filter: {getattr(config, 'SG_USE_PIVOT_FILTER', False)}")
        if getattr(config, 'SG_USE_PIVOT_FILTER', False):
            print(f"  Pivot Proximity Pct: {getattr(config, 'SG_PIVOT_PROXIMITY_PCT', 0.005)}")
            print(f"  Pivot Levels to Consider: {getattr(config, 'SG_CONSIDER_PIVOT_LEVELS', ['PP', 'S1', 'R1'])}")

        print(f"Use Double Top/Bottom Signals: {getattr(config, 'SG_USE_DOUBLE_TOP_BOTTOM_SIGNALS', False)}")
        if getattr(config, 'SG_USE_DOUBLE_TOP_BOTTOM_SIGNALS', False):
            print(f"  DT/DB Reward Ratio (if ATR not used for SL): {getattr(config, 'SG_DBL_TOP_BOTTOM_REWARD_RATIO', 2.0)}")
            # Mention related MS parameters for DT/DB feature calculation
            print(f"  DT/DB Lookback Swings (MS_DTB_LOOKBACK_SWINGS): {getattr(config, 'MS_DTB_LOOKBACK_SWINGS', 3)}")
            print(f"  DT/DB Price Similarity Pct (MS_DTB_SIMILARITY_PCT): {getattr(config, 'MS_DTB_SIMILARITY_PCT', 0.03)}")
            print(f"  DT/DB Confirmation Ratio (MS_DTB_CONFIRMATION_RATIO): {getattr(config, 'MS_DTB_CONFIRMATION_RATIO', 0.3)}")
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
            enriched_primary_df = pre_process_data_for_signals(ohlcv_data_main, config_module=config)
            print(f"Enriched primary DataFrame shape: {enriched_primary_df.shape}")
            if 'atr' in enriched_primary_df.columns:
                print("ATR column found in enriched_primary_df. Last 5 values:")
                print(enriched_primary_df['atr'].tail())

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
                             enriched_primary_df[col] = enriched_primary_df[col].fillna(0)

                signals_result_tuple = generate_initial_signals(
                    primary_df_input=enriched_primary_df,
                    primary_symbol=config.EX_SG_PRIMARY_SYMBOL,
                    primary_interval=config.EX_SG_PRIMARY_INTERVAL,
                    higher_tf_df_input=htf_df_main
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
                        'Open', 'High', 'Low', 'Close', 'Volume', 'signal', 'atr',
                        'lstm_prediction', 'stop_loss', 'take_profit',
                        'is_marubozu', 'is_pullback_bar',
                        'is_bullish_breakout', 'is_bearish_breakout',
                        'is_double_top_confirmed', 'is_double_bottom_confirmed' # Added DT/DB columns
                    ]
                    if 'is_marubozu' not in output_df.columns and 'is_bullish_marubozu' in output_df.columns:
                         output_df['is_marubozu'] = output_df['is_bullish_marubozu'] | output_df['is_bearish_marubozu']

                    # Ensure all columns in cols_to_show actually exist in output_df before selecting
                    relevant_cols = [col for col in cols_to_show if col in output_df.columns]
                    recent_activity_df = output_df[output_df['signal'] != 0][relevant_cols]

                    if recent_activity_df.empty:
                        print("\nNo signals generated in this run. Showing last 15 bars with key features:")
                        cols_for_no_signal_view = ['Open', 'High', 'Low', 'Close', 'Volume', 'atr']
                        if 'lstm_prediction' in output_df.columns: # Add LSTM prediction if available
                            cols_for_no_signal_view.append('lstm_prediction')
                        # Add boolean features for context
                        for bool_col in ['is_marubozu', 'is_pullback_bar',
                                         'is_bullish_breakout', 'is_bearish_breakout',
                                         'is_double_top_confirmed', 'is_double_bottom_confirmed']: # Added DT/DB
                            if bool_col in output_df.columns:
                                cols_for_no_signal_view.append(bool_col)

                        final_cols_to_view = [col for col in cols_for_no_signal_view if col in output_df.columns]
                        print(output_df[final_cols_to_view].tail(15).to_string())

                        if 'lstm_prediction' not in output_df.columns and getattr(config, 'SG_USE_LSTM_FILTER', False):
                            print("(Note: LSTM prediction column was not found in the output, though LSTM filter is ON.)")
                    else:
                        print("\n--- Generated Signals ---")
                        print(recent_activity_df.to_string())
                else:
                    print("\nFailed to generate signals (signals_df is None).")
        else:
            print(f"Could not fetch primary data for {config.EX_SG_PRIMARY_SYMBOL}. Signal generation aborted.")
        print("\nsignal_generator.py MTF example with LSTM filter and SL/TP finished.")
