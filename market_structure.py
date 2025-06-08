import pandas as pd
import numpy as np

# Attempt to import data_fetcher for example usage.
# If it fails, the example will use manually created data.
try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False
    print("data_fetcher.py not found. Example will use manually created data.")

try:
    import config
except ImportError:
    print("Warning: config.py not found in market_structure.py. Using internal defaults.")
    class config: # Fallback config
        EX_LSTM_TRAINER_SYMBOL = 'AAPL' # Symbol for example data
        PRIMARY_PERIOD_LSTM = '1y'    # Period for example data
        PRIMARY_INTERVAL_LSTM = '1d'  # Interval for example data
        MS_SWING_WINDOW = 5
        MS_PULLBACK_TREND_LOOKBACK = 2
        MS_SR_MIN_TOUCHES = 2
        MS_SR_RELATIVE_TOLERANCE = 0.015
        MS_BREAKOUT_LOOKBACK_PERIOD = 20


def find_swing_highs_lows(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Identifies swing highs and swing lows in price data.
    A swing high is a price peak strictly higher than 'window' bars on each side.
    A swing low is a price trough strictly lower than 'window' bars on each side.
    Adds 'is_swing_high' and 'is_swing_low' columns to the DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    if not all(col in data.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("'window' must be a positive integer.")

    df = data.copy()
    df['is_swing_high'] = False
    df['is_swing_low'] = False

    for i in range(window, len(df) - window):
        # Check for Swing High
        is_potential_sh = True
        for j in range(1, window + 1): # Check left
            if df['High'].iloc[i] <= df['High'].iloc[i-j]:
                is_potential_sh = False; break

        if is_potential_sh: # Only if left passed, check right
            for j in range(1, window + 1):
                if df['High'].iloc[i] <= df['High'].iloc[i+j]:
                    is_potential_sh = False; break

        if is_potential_sh: # If both left and right conditions passed
            df.loc[df.index[i], 'is_swing_high'] = True

        # Check for Swing Low (independent of SH check)
        is_potential_sl = True
        for j in range(1, window + 1): # Check left
            if df['Low'].iloc[i] >= df['Low'].iloc[i-j]:
                is_potential_sl = False; break

        if is_potential_sl: # Only if left passed, check right
            for j in range(1, window + 1):
                if df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                    is_potential_sl = False; break

        if is_potential_sl: # If both left and right conditions passed
            df.loc[df.index[i], 'is_swing_low'] = True

    data['is_swing_high'] = df['is_swing_high']
    data['is_swing_low'] = df['is_swing_low']
    return data

def identify_simple_trend(data: pd.DataFrame, lookback_swings: int = 2) -> str:
    # ... (remains unchanged) ...
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    if not all(col in data.columns for col in ['High', 'Low', 'is_swing_high', 'is_swing_low']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'is_swing_high', and 'is_swing_low' columns.")
    if not isinstance(lookback_swings, int) or lookback_swings < 2:
        raise ValueError("'lookback_swings' must be an integer greater than or equal to 2.")
    sh_points = data[data['is_swing_high']]['High']
    sl_points = data[data['is_swing_low']]['Low']
    if len(sh_points) < lookback_swings or len(sl_points) < lookback_swings:
        return "Undetermined"
    last_n_sh = sh_points.tail(lookback_swings).values
    last_n_sl = sl_points.tail(lookback_swings).values
    is_uptrend = True
    for i in range(lookback_swings - 1):
        if not (last_n_sh[i+1] > last_n_sh[i] and last_n_sl[i+1] > last_n_sl[i]):
            is_uptrend = False; break
    if is_uptrend: return "Uptrend"
    is_downtrend = True
    for i in range(lookback_swings - 1):
        if not (last_n_sh[i+1] < last_n_sh[i] and last_n_sl[i+1] < last_n_sl[i]):
            is_downtrend = False; break
    if is_downtrend: return "Downtrend"
    return "Trading Range / Chop"

def find_support_resistance_from_swings(data: pd.DataFrame, min_touches: int = 2, relative_tolerance: float = 0.01) -> dict:
    # ... (remains unchanged) ...
    if not isinstance(data, pd.DataFrame): raise ValueError("Input 'data' must be a pandas DataFrame.")
    if not all(col in data.columns for col in ['High', 'Low', 'is_swing_high', 'is_swing_low']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'is_swing_high', and 'is_swing_low' columns.")
    if not isinstance(min_touches, int) or min_touches < 1: raise ValueError("'min_touches' must be a positive integer >= 1.")
    if not isinstance(relative_tolerance, float) or not (0 < relative_tolerance < 1):
        raise ValueError("'relative_tolerance' must be a float between 0 and 1 (exclusive).")
    potential_resistance_points = sorted(data[data['is_swing_high']]['High'].tolist(), reverse=True)
    potential_support_points = sorted(data[data['is_swing_low']]['Low'].tolist())
    identified_resistance_levels = []
    identified_support_levels = []
    processed_resistance_points = [False] * len(potential_resistance_points)
    for i, p_high in enumerate(potential_resistance_points):
        if processed_resistance_points[i]: continue
        current_cluster = [p_high]
        processed_resistance_points[i] = True
        for j in range(i + 1, len(potential_resistance_points)):
            if processed_resistance_points[j]: continue
            p_next_high = potential_resistance_points[j]
            if p_next_high >= p_high * (1 - relative_tolerance):
                current_cluster.append(p_next_high); processed_resistance_points[j] = True
            else: break
        if len(current_cluster) >= min_touches:
            identified_resistance_levels.append(sum(current_cluster) / len(current_cluster))
    processed_support_points = [False] * len(potential_support_points)
    for i, p_low in enumerate(potential_support_points):
        if processed_support_points[i]: continue
        current_cluster = [p_low]
        processed_support_points[i] = True
        for j in range(i + 1, len(potential_support_points)):
            if processed_support_points[j]: continue
            p_next_low = potential_support_points[j]
            if p_next_low <= p_low * (1 + relative_tolerance):
                current_cluster.append(p_next_low); processed_support_points[j] = True
            else: break
        if len(current_cluster) >= min_touches:
            identified_support_levels.append(sum(current_cluster) / len(current_cluster))
    return {'support': sorted(list(set(identified_support_levels))), 'resistance': sorted(list(set(identified_resistance_levels)), reverse=True)}

def identify_pullbacks(data_df: pd.DataFrame, trend_lookback_swings: int = 2) -> pd.DataFrame:
    # ... (remains unchanged) ...
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input 'data_df' must be a pandas DataFrame.")
    if not all(col in data_df.columns for col in ['High', 'Low', 'is_swing_high', 'is_swing_low']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'is_swing_high', and 'is_swing_low' columns.")
    if trend_lookback_swings < 2:
        raise ValueError("'trend_lookback_swings' must be at least 2.")
    df = data_df.copy()
    df['is_pullback_bar'] = False
    all_sh_points_series = df[df['is_swing_high']]['High']
    all_sl_points_series = df[df['is_swing_low']]['Low']
    df['prev_High'] = df['High'].shift(1)
    df['prev_Low'] = df['Low'].shift(1)
    start_iteration_index = 0
    if len(all_sl_points_series) >= trend_lookback_swings and len(all_sh_points_series) >= trend_lookback_swings:
        idx_nth_sl = all_sl_points_series.index[trend_lookback_swings -1]
        idx_nth_sh = all_sh_points_series.index[trend_lookback_swings -1]
        latest_of_nth_swings = max(idx_nth_sl, idx_nth_sh)
        try:
            start_iteration_index = df.index.get_loc(latest_of_nth_swings) + 1
        except KeyError:
            start_iteration_index = trend_lookback_swings * 3
    if start_iteration_index >= len(df):
        data_df['is_pullback_bar'] = df['is_pullback_bar']
        return data_df
    for i in range(start_iteration_index, len(df)):
        current_index_label = df.index[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        prev_high_val = df['prev_High'].iloc[i]
        prev_low_val = df['prev_Low'].iloc[i]
        if pd.isna(prev_high_val): continue
        local_sh_points = all_sh_points_series[all_sh_points_series.index < current_index_label]
        local_sl_points = all_sl_points_series[all_sl_points_series.index < current_index_label]
        if len(local_sh_points) < trend_lookback_swings or len(local_sl_points) < trend_lookback_swings:
            continue
        last_n_sh_values = local_sh_points.tail(trend_lookback_swings).values
        last_n_sl_values = local_sl_points.tail(trend_lookback_swings).values
        is_local_uptrend = True
        for k in range(trend_lookback_swings - 1):
            if not (last_n_sh_values[k+1] > last_n_sh_values[k] and last_n_sl_values[k+1] > last_n_sl_values[k]):
                is_local_uptrend = False; break
        if is_local_uptrend:
            if current_high < prev_high_val and current_low > last_n_sl_values[-1]:
                df.loc[current_index_label, 'is_pullback_bar'] = True
        else:
            is_local_downtrend = True
            for k in range(trend_lookback_swings - 1):
                if not (last_n_sh_values[k+1] < last_n_sh_values[k] and last_n_sl_values[k+1] < last_n_sl_values[k]):
                    is_local_downtrend = False; break
            if is_local_downtrend:
                if current_low > prev_low_val and current_high < last_n_sh_values[-1]:
                     df.loc[current_index_label, 'is_pullback_bar'] = True
    data_df['is_pullback_bar'] = df['is_pullback_bar']
    return data_df

def identify_breakouts(data_df: pd.DataFrame, lookback_period: int = 20, sr_min_touches: int = 2, sr_tolerance: float = 0.01) -> pd.DataFrame:
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input 'data_df' must be a pandas DataFrame.")
    if not all(col in data_df.columns for col in ['Open', 'High', 'Low', 'Close', 'is_swing_high', 'is_swing_low']):
        raise ValueError("DataFrame must contain OHLC and swing columns.")

    df = data_df.copy()
    df['is_bullish_breakout'] = False
    df['is_bearish_breakout'] = False

    sr_levels = find_support_resistance_from_swings(df.copy(), min_touches=sr_min_touches, relative_tolerance=sr_tolerance)
    resistance_levels = sr_levels.get('resistance', [])
    support_levels = sr_levels.get('support', [])

    # print(f"DEBUG: Breakouts - Resistance Levels: {resistance_levels}")
    # print(f"DEBUG: Breakouts - Support Levels: {support_levels}")

    for i in range(lookback_period, len(df)):
        current_index_label = df.index[i]
        current_row = df.iloc[i]

        lookback_data = df.iloc[i-lookback_period : i]

        recent_swing_highs = lookback_data[lookback_data['is_swing_high']]['High']
        recent_highest_sh = recent_swing_highs.max() if not recent_swing_highs.empty else np.nan

        recent_swing_lows = lookback_data[lookback_data['is_swing_low']]['Low']
        recent_lowest_sl = recent_swing_lows.min() if not recent_swing_lows.empty else np.nan

        bullish_breakout_triggered = False
        if not pd.isna(recent_highest_sh) and current_row['Close'] > recent_highest_sh:
            bullish_breakout_triggered = True

        if not bullish_breakout_triggered:
            for r_level in resistance_levels:
                if current_row['Low'] < r_level < current_row['Close']:
                    bullish_breakout_triggered = True; break
        if bullish_breakout_triggered:
            df.loc[current_index_label, 'is_bullish_breakout'] = True

        bearish_breakout_triggered = False
        if not pd.isna(recent_lowest_sl) and current_row['Close'] < recent_lowest_sl:
            bearish_breakout_triggered = True

        if not bearish_breakout_triggered:
            for s_level in support_levels:
                if current_row['High'] > s_level > current_row['Close']:
                    bearish_breakout_triggered = True; break
        if bearish_breakout_triggered:
            df.loc[current_index_label, 'is_bearish_breakout'] = True

    data_df['is_bullish_breakout'] = df['is_bullish_breakout']
    data_df['is_bearish_breakout'] = df['is_bearish_breakout']
    return data_df

if __name__ == "__main__":
    print("Running market_structure.py example with Breakout Identification...")
    main_sample_data = None

    # Use config for fetching example data
    example_symbol = config.EX_LSTM_TRAINER_SYMBOL if hasattr(config, 'EX_LSTM_TRAINER_SYMBOL') else 'AAPL'
    example_interval = config.PRIMARY_INTERVAL_LSTM if hasattr(config, 'PRIMARY_INTERVAL_LSTM') else '1d'
    example_period = config.PRIMARY_PERIOD_LSTM if hasattr(config, 'PRIMARY_PERIOD_LSTM') else '1y'

    if DATA_FETCHER_AVAILABLE:
        print(f"\nAttempting to fetch sample data using data_fetcher for {example_symbol} ({example_period})...")
        main_sample_data = fetch_price_data(symbol=example_symbol, interval=example_interval, period=example_period)
        if main_sample_data is None: print(f"Failed to fetch {example_symbol} data.")
        else: print(f"Successfully fetched {example_symbol} data for {main_sample_data.index.nunique()} days.")

    if main_sample_data is None:
        print(f"Using fallback manual data for main test as {example_symbol} fetch failed.")
        data_dict_fallback_values = {
            'Open':  [100,102,101,105,103,108,105,104,107,110,108,105,107,100,103,100,100,100,100,100],
            'High':  [105,103,102,110,108,110,109,106,110,112,110,109,108,104,105,100,100,100,100,100],
            'Low':   [98,101,100,103,102,105,103,102,105,108,104,103,102,99,100,100,100,100,100,100],
            'Close': [102,101,101,109,105,109,104,103,109,111,105,104,103,102,104,100,100,100,100,100],
            'Volume':[1000]*20 }
        main_sample_data = pd.DataFrame(data_dict_fallback_values, index=pd.date_range(start='2023-01-01', periods=20))

    if main_sample_data is not None and not main_sample_data.empty:
        print("\n--- Analyzing Fetched/Fallback Data ---")
        data_analyzed = main_sample_data.copy()

        # Use parameters from config
        sw_window = config.MS_SWING_WINDOW if hasattr(config, 'MS_SWING_WINDOW') else 5
        pt_lookback = config.MS_PULLBACK_TREND_LOOKBACK if hasattr(config, 'MS_PULLBACK_TREND_LOOKBACK') else 2
        sr_touches = config.MS_SR_MIN_TOUCHES if hasattr(config, 'MS_SR_MIN_TOUCHES') else 2
        sr_tol = config.MS_SR_RELATIVE_TOLERANCE if hasattr(config, 'MS_SR_RELATIVE_TOLERANCE') else 0.015
        bo_lookback = config.MS_BREAKOUT_LOOKBACK_PERIOD if hasattr(config, 'MS_BREAKOUT_LOOKBACK_PERIOD') else 20

        data_analyzed = find_swing_highs_lows(data_analyzed, window=sw_window)
        overall_trend = identify_simple_trend(data_analyzed.copy(), lookback_swings=pt_lookback)
        print(f"Overall Identified Trend: {overall_trend}")

        sr_levels_main = find_support_resistance_from_swings(data_analyzed.copy(), min_touches=sr_touches, relative_tolerance=sr_tol)
        print(f"S/R Levels (from main analysis) - Support: {sr_levels_main.get('support')}, Resistance: {sr_levels_main.get('resistance')}")

        data_analyzed = identify_pullbacks(data_analyzed.copy(), trend_lookback_swings=pt_lookback)
        data_analyzed = identify_breakouts(data_analyzed.copy(), lookback_period=bo_lookback, sr_min_touches=sr_touches, sr_tolerance=sr_tol)

        pullback_bars = data_analyzed[data_analyzed['is_pullback_bar']]
        if not pullback_bars.empty:
            print(f"\n--- Pullback Bars ({len(pullback_bars)} instances) ---")
            print(pullback_bars[['Open', 'High', 'Low', 'Close', 'is_pullback_bar']].head())
        else: print("\nNo pullback bars identified in fetched/fallback data.")

        breakout_bars = data_analyzed[data_analyzed['is_bullish_breakout'] | data_analyzed['is_bearish_breakout']]
        if not breakout_bars.empty:
            print(f"\n--- Breakout Bars ({len(breakout_bars)} instances) ---")
            print(breakout_bars[['Open', 'High', 'Low', 'Close', 'is_bullish_breakout', 'is_bearish_breakout']].head())
        else: print("\nNo breakout bars identified in fetched/fallback data.")

    # --- Specific Manual Tests for Breakouts (using window=1 for swings) ---
    print("\n\n--- Specific Manual Test for Breakout (Above Recent SH) ---")
    # SH at 105 (idx 1). Breakout bar at idx 3 (Close=106)
    breakout_sh_values = {
        'Open':  [100, 103, 100, 105.5, 106], 'High':  [101, 105, 101, 106,   107],
        'Low':   [99,  102, 99,  104,   105], 'Close': [100.5,104,100.5,106,106.5], 'Volume':[100]*5 }
    breakout_sh_df = pd.DataFrame(breakout_sh_values, index=pd.date_range(start='2023-03-01', periods=5))
    breakout_sh_df = find_swing_highs_lows(breakout_sh_df.copy(), window=1)
    print("Manual Data for SH Breakout (Swings marked):")
    print(breakout_sh_df[['High','Low','is_swing_high','is_swing_low']]) # Print all rows to check swings
    breakout_sh_df = identify_breakouts(breakout_sh_df.copy(), lookback_period=3, sr_min_touches=1, sr_tolerance=0.01)
    breakout_sh_bars = breakout_sh_df[breakout_sh_df['is_bullish_breakout']]
    if not breakout_sh_bars.empty: print(f"--- MANUAL Bullish Breakout (SH) ---\n{breakout_sh_bars[['Open','High','Low','Close', 'is_bullish_breakout']]}")
    else: print("No manual bullish SH breakout identified.")

    print("\n\n--- Specific Manual Test for Breakout (Below Recent SL) ---")
    # SL at 95 (idx 1). Breakout bar at idx 3 (Close=94)
    breakout_sl_values = {
        'Open':  [100, 97, 100, 94.5, 93], 'High':  [101, 98, 101, 95,   94],
        'Low':   [99,  95, 96,  94,   92], 'Close': [99.5,96, 99.5,94,   93], 'Volume':[100]*5 }
    breakout_sl_df = pd.DataFrame(breakout_sl_values, index=pd.date_range(start='2023-04-01', periods=5))
    breakout_sl_df = find_swing_highs_lows(breakout_sl_df.copy(), window=1)
    print("Manual Data for SL Breakout (Swings marked):")
    print(breakout_sl_df[['High','Low','is_swing_high','is_swing_low']])
    breakout_sl_df = identify_breakouts(breakout_sl_df.copy(), lookback_period=3, sr_min_touches=1, sr_tolerance=0.01)
    breakout_sl_bars = breakout_sl_df[breakout_sl_df['is_bearish_breakout']]
    if not breakout_sl_bars.empty: print(f"--- MANUAL Bearish Breakout (SL) ---\n{breakout_sl_bars[['Open','High','Low','Close', 'is_bearish_breakout']]}")
    else: print("No manual bearish SL breakout identified.")

    print("\n\n--- Specific Manual Test for Breakout (Above Resistance Level) ---")
    # R level at 102. Breakout bar idx 5 (Close=103)
    breakout_res_values = {
        'Open':  [100, 101, 100, 101, 100, 101.5], 'High':  [101, 102, 101, 102, 101, 104], # SHs at 102 (idx1), 102 (idx3)
        'Low':   [99,  100, 99,  100, 99,  101],   'Close': [100.5,101,100.5,101,100, 103], 'Volume':[100]*6 }
    breakout_res_df = pd.DataFrame(breakout_res_values, index=pd.date_range(start='2023-05-01', periods=6))
    breakout_res_df = find_swing_highs_lows(breakout_res_df.copy(), window=1)
    print("Manual Data for Resistance Breakout (Swings marked for S/R calc):")
    print(breakout_res_df[['High','Low','is_swing_high','is_swing_low']])
    breakout_res_df = identify_breakouts(breakout_res_df.copy(), lookback_period=5, sr_min_touches=2, sr_tolerance=0.005)
    breakout_res_bars = breakout_res_df[breakout_res_df['is_bullish_breakout']]
    if not breakout_res_bars.empty: print(f"--- MANUAL Bullish Breakout (Resistance) ---\n{breakout_res_bars[['Open','High','Low','Close', 'is_bullish_breakout']]}")
    else: print("No manual bullish resistance breakout identified.")

    print("\n\n--- Specific Manual Test for Breakout (Below Support Level) ---")
    # S level at 98. Breakout bar idx 5 (Close=97)
    breakout_sup_values = {
        'Open':  [100, 99,  100, 99,  100, 98.5], 'High':  [101, 100, 101, 100, 101, 99],
        'Low':   [99,  98,  99,  98,  99,  96],   'Close': [99.5,99,  99.5,99, 98.6, 97], 'Volume':[100]*6 } # SLs at 98 (idx1), 98 (idx3)
    breakout_sup_df = pd.DataFrame(breakout_sup_values, index=pd.date_range(start='2023-06-01', periods=6))
    breakout_sup_df = find_swing_highs_lows(breakout_sup_df.copy(), window=1)
    print("Manual Data for Support Breakout (Swings marked for S/R calc):")
    print(breakout_sup_df[['High','Low','is_swing_high','is_swing_low']])
    breakout_sup_df = identify_breakouts(breakout_sup_df.copy(), lookback_period=5, sr_min_touches=2, sr_tolerance=0.005)
    breakout_sup_bars = breakout_sup_df[breakout_sup_df['is_bearish_breakout']]
    if not breakout_sup_bars.empty: print(f"--- MANUAL Bearish Breakout (Support) ---\n{breakout_sup_bars[['Open','High','Low','Close', 'is_bearish_breakout']]}")
    else: print("No manual bearish support breakout identified.")

    print("\nmarket_structure.py example finished.")
