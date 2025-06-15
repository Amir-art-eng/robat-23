import pandas as pd
import numpy as np # Ensure numpy is imported

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
    print("Warning: config.py not found in market_structure.py. Using internal defaults for example parameters if not overridden.")
    pass # Fallback to direct defaults in __main__ if config not found or attribute missing


def find_swing_highs_lows(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
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
        is_potential_sh = True
        for j in range(1, window + 1):
            if df['High'].iloc[i] <= df['High'].iloc[i-j]:
                is_potential_sh = False; break
        if is_potential_sh:
            for j in range(1, window + 1):
                if df['High'].iloc[i] <= df['High'].iloc[i+j]:
                    is_potential_sh = False; break
        if is_potential_sh:
            df.loc[df.index[i], 'is_swing_high'] = True

        is_potential_sl = True
        for j in range(1, window + 1):
            if df['Low'].iloc[i] >= df['Low'].iloc[i-j]:
                is_potential_sl = False; break
        if is_potential_sl:
            for j in range(1, window + 1):
                if df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                    is_potential_sl = False; break
        if is_potential_sl:
            df.loc[df.index[i], 'is_swing_low'] = True

    data['is_swing_high'] = df['is_swing_high']
    data['is_swing_low'] = df['is_swing_low']
    return data

def identify_simple_trend(data: pd.DataFrame, lookback_swings: int = 2) -> str:
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

def calculate_daily_pivot_points(daily_data_df: pd.DataFrame) -> dict:
    if not isinstance(daily_data_df.index, pd.DatetimeIndex):
        try:
            daily_data_df.index = pd.to_datetime(daily_data_df.index)
        except Exception as e:
            print(f"Error converting index to DatetimeIndex: {e}. Please ensure index is datetime-like.")
            return {}

    pivot_points_data = {}
    if len(daily_data_df) < 2:
        print("Not enough data to calculate pivot points (need at least 2 days).")
        return pivot_points_data

    for i in range(1, len(daily_data_df)):
        prev_high = daily_data_df['High'].iloc[i-1]
        prev_low = daily_data_df['Low'].iloc[i-1]
        prev_close = daily_data_df['Close'].iloc[i-1]

        pp = (prev_high + prev_low + prev_close) / 3.0
        s1 = (pp * 2) - prev_high
        r1 = (pp * 2) - prev_low
        s2 = pp - (prev_high - prev_low)
        r2 = pp + (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pp)
        r3 = prev_high + 2 * (pp - prev_low)

        current_day_date_str = daily_data_df.index[i].strftime('%Y-%m-%d')
        pivot_points_data[current_day_date_str] = {
            'PP': round(pp, 2), 'S1': round(s1, 2), 'R1': round(r1, 2),
            'S2': round(s2, 2), 'R2': round(r2, 2),
            'S3': round(s3, 2), 'R3': round(r3, 2)
        }
    return pivot_points_data

def get_iloc_or_nan(df_index, label_index):
    try:
        return df_index.get_loc(label_index)
    except KeyError:
        return np.nan

def find_trend_line_segments(data_df: pd.DataFrame, lookback_swings: int = 3, min_swing_separation: int = 3) -> dict:
    if 'is_swing_low' not in data_df.columns or 'is_swing_high' not in data_df.columns:
        print("Error: DataFrame must contain 'is_swing_low' and 'is_swing_high' columns.")
        print("Please run find_swing_highs_lows() first.")
        return {'uptrend_lines': [], 'downtrend_lines': []}

    swing_lows_df = data_df[data_df['is_swing_low']].copy()
    swing_highs_df = data_df[data_df['is_swing_high']].copy()

    sl_points = []
    for idx, row in swing_lows_df.iterrows():
        sl_points.append({'index': idx, 'price': row['Low'], 'iloc_pos': get_iloc_or_nan(data_df.index, idx)})

    sh_points = []
    for idx, row in swing_highs_df.iterrows():
        sh_points.append({'index': idx, 'price': row['High'], 'iloc_pos': get_iloc_or_nan(data_df.index, idx)})

    sl_points = [p for p in sl_points if not np.isnan(p['iloc_pos'])]
    sh_points = [p for p in sh_points if not np.isnan(p['iloc_pos'])]

    sl_points.sort(key=lambda x: x['iloc_pos'])
    sh_points.sort(key=lambda x: x['iloc_pos'])

    recent_sl = sl_points[-lookback_swings:]
    recent_sh = sh_points[-lookback_swings:]

    uptrend_lines = []
    if len(recent_sl) >= 2:
        for i in range(len(recent_sl)):
            for j in range(i + 1, len(recent_sl)):
                sl1 = recent_sl[i]
                sl2 = recent_sl[j]
                if sl1['price'] < sl2['price'] and (sl2['iloc_pos'] - sl1['iloc_pos'] >= min_swing_separation):
                    uptrend_lines.append( ((sl1['index'], sl1['price']), (sl2['index'], sl2['price'])) )

    downtrend_lines = []
    if len(recent_sh) >= 2:
        for i in range(len(recent_sh)):
            for j in range(i + 1, len(recent_sh)):
                sh1 = recent_sh[i]
                sh2 = recent_sh[j]
                if sh1['price'] > sh2['price'] and (sh2['iloc_pos'] - sh1['iloc_pos'] >= min_swing_separation):
                    downtrend_lines.append( ((sh1['index'], sh1['price']), (sh2['index'], sh2['price'])) )

    return {'uptrend_lines': uptrend_lines, 'downtrend_lines': downtrend_lines}

def find_basic_channels(data_df: pd.DataFrame, trend_lines_dict: dict) -> dict:
    channels = {'uptrend_channels': [], 'downtrend_channels': []}
    df_main_index = data_df.index

    def get_channel_line_price_at_x(x_iloc_target, anchor_point_iloc, anchor_point_price, slope):
        return slope * (x_iloc_target - anchor_point_iloc) + anchor_point_price

    for main_line in trend_lines_dict.get('uptrend_lines', []):
        (sl1_idx, sl1_p), (sl2_idx, sl2_p) = main_line
        sl1_iloc = get_iloc_or_nan(df_main_index, sl1_idx)
        sl2_iloc = get_iloc_or_nan(df_main_index, sl2_idx)

        if np.isnan(sl1_iloc) or np.isnan(sl2_iloc) or sl1_iloc == sl2_iloc: continue

        start_iloc_lookup = min(sl1_iloc, sl2_iloc)
        end_iloc_lookup = max(sl1_iloc, sl2_iloc)
        segment_df = data_df.iloc[int(start_iloc_lookup) : int(end_iloc_lookup) + 1]
        relevant_swing_highs = segment_df[segment_df['is_swing_high']]

        if not relevant_swing_highs.empty:
            sh_c_series = relevant_swing_highs.loc[relevant_swing_highs['High'].idxmax()]
            sh_c_idx = sh_c_series.name
            sh_c_p = sh_c_series['High']
            sh_c_iloc = get_iloc_or_nan(df_main_index, sh_c_idx)
            if np.isnan(sh_c_iloc): continue

            slope = (sl2_p - sl1_p) / (sl2_iloc - sl1_iloc)
            cl1_p = get_channel_line_price_at_x(sl1_iloc, sh_c_iloc, sh_c_p, slope)
            cl2_p = get_channel_line_price_at_x(sl2_iloc, sh_c_iloc, sh_c_p, slope)
            channel_line_segment = ((sl1_idx, cl1_p), (sl2_idx, cl2_p))
            channels['uptrend_channels'].append((main_line, channel_line_segment))

    for main_line in trend_lines_dict.get('downtrend_lines', []):
        (sh1_idx, sh1_p), (sh2_idx, sh2_p) = main_line
        sh1_iloc = get_iloc_or_nan(df_main_index, sh1_idx)
        sh2_iloc = get_iloc_or_nan(df_main_index, sh2_idx)

        if np.isnan(sh1_iloc) or np.isnan(sh2_iloc) or sh1_iloc == sh2_iloc: continue

        start_iloc_lookup = min(sh1_iloc, sh2_iloc)
        end_iloc_lookup = max(sh1_iloc, sh2_iloc)
        segment_df = data_df.iloc[int(start_iloc_lookup) : int(end_iloc_lookup) + 1]
        relevant_swing_lows = segment_df[segment_df['is_swing_low']]

        if not relevant_swing_lows.empty:
            sl_c_series = relevant_swing_lows.loc[relevant_swing_lows['Low'].idxmin()]
            sl_c_idx = sl_c_series.name
            sl_c_p = sl_c_series['Low']
            sl_c_iloc = get_iloc_or_nan(df_main_index, sl_c_idx)
            if np.isnan(sl_c_iloc): continue

            slope = (sh2_p - sh1_p) / (sh2_iloc - sh1_iloc)
            cl1_p = get_channel_line_price_at_x(sh1_iloc, sl_c_iloc, sl_c_p, slope)
            cl2_p = get_channel_line_price_at_x(sh2_iloc, sl_c_iloc, sl_c_p, slope)
            channel_line_segment = ((sh1_idx, cl1_p), (sh2_idx, cl2_p))
            channels['downtrend_channels'].append((main_line, channel_line_segment))

    return channels

def calculate_line_params(p1_iloc, p1_price, p2_iloc, p2_price):
    if p2_iloc == p1_iloc:
        return np.inf, p1_iloc
    slope = (p2_price - p1_price) / (p2_iloc - p1_iloc)
    intercept = p1_price - slope * p1_iloc
    return slope, intercept

def calculate_line_intersection(m1, c1, m2, c2):
    if m1 == m2:
        return None, None
    if np.isinf(m1):
        apex_x = c1
        apex_y = m2 * apex_x + c2
        return apex_x, apex_y
    if np.isinf(m2):
        apex_x = c2
        apex_y = m1 * apex_x + c1
        return apex_x, apex_y

    apex_x = (c2 - c1) / (m1 - m2)
    apex_y = m1 * apex_x + c1
    return apex_x, apex_y

def find_symmetrical_triangles(data_df: pd.DataFrame, trend_lines_dict: dict,
                             min_line_duration_bars: int = 5,
                             max_apex_distance_bars: int = 100) -> list:
    triangles = []
    df_main_index = data_df.index

    uptrend_lines = trend_lines_dict.get('uptrend_lines', [])
    downtrend_lines = trend_lines_dict.get('downtrend_lines', [])

    for up_line in uptrend_lines:
        (ul_p1_idx, ul_p1_price), (ul_p2_idx, ul_p2_price) = up_line
        ul_p1_iloc = get_iloc_or_nan(df_main_index, ul_p1_idx)
        ul_p2_iloc = get_iloc_or_nan(df_main_index, ul_p2_idx)

        if np.isnan(ul_p1_iloc) or np.isnan(ul_p2_iloc) or (ul_p2_iloc - ul_p1_iloc < min_line_duration_bars):
            continue

        m_up, c_up = calculate_line_params(ul_p1_iloc, ul_p1_price, ul_p2_iloc, ul_p2_price)
        if np.isinf(m_up): continue

        for down_line in downtrend_lines:
            (dl_p1_idx, dl_p1_price), (dl_p2_idx, dl_p2_price) = down_line
            dl_p1_iloc = get_iloc_or_nan(df_main_index, dl_p1_idx)
            dl_p2_iloc = get_iloc_or_nan(df_main_index, dl_p2_idx)

            if np.isnan(dl_p1_iloc) or np.isnan(dl_p2_iloc) or (dl_p2_iloc - dl_p1_iloc < min_line_duration_bars):
                continue

            m_down, c_down = calculate_line_params(dl_p1_iloc, dl_p1_price, dl_p2_iloc, dl_p2_price)
            if np.isinf(m_down): continue

            if m_up <= 0 or m_down >= 0:
                continue

            apex_x_iloc, apex_y_price = calculate_line_intersection(m_up, c_up, m_down, c_down)

            if apex_x_iloc is None:
                continue

            latest_start_iloc = max(ul_p1_iloc, dl_p1_iloc)
            earliest_end_iloc = min(ul_p2_iloc, dl_p2_iloc)
            latest_end_iloc = max(ul_p2_iloc, dl_p2_iloc)

            if apex_x_iloc > earliest_end_iloc and (apex_x_iloc - latest_end_iloc) <= max_apex_distance_bars:
                price_up_at_mouth_end = m_up * earliest_end_iloc + c_up
                price_down_at_mouth_end = m_down * earliest_end_iloc + c_down

                if price_up_at_mouth_end < price_down_at_mouth_end :
                    triangles.append({
                        'uptrend_line': up_line,
                        'downtrend_line': down_line,
                        'apex_iloc': apex_x_iloc,
                        'apex_price': apex_y_price,
                        'formation_end_iloc': earliest_end_iloc
                    })
    return triangles

def identify_gaps(data_df: pd.DataFrame, min_gap_percentage: float = 0.001) -> pd.DataFrame:
    df = data_df.copy()

    if 'Close' not in df.columns or 'Open' not in df.columns:
        print("Error: DataFrame must contain 'Open' and 'Close' columns.")
        data_df['gap_type'] = 'No Gap'
        data_df['gap_size_pct'] = 0.0
        return data_df

    df['previous_close'] = df['Close'].shift(1)

    is_gap_up = df['Open'] > df['previous_close']
    is_gap_down = df['Open'] < df['previous_close']

    df['abs_gap_size'] = np.abs(df['Open'] - df['previous_close'])
    gap_pct_series = (df['abs_gap_size'] / df['previous_close']).fillna(0)
    df['gap_size_pct'] = np.where(df['previous_close'] == 0, 0, gap_pct_series)

    df['gap_type'] = 'No Gap'
    significant_gap = df['gap_size_pct'] >= min_gap_percentage

    conditions = []
    choices = []

    if 'is_bullish_breakout' in df.columns and 'is_bearish_breakout' in df.columns:
        conditions.extend([
            (is_gap_up & significant_gap & df['is_bullish_breakout']),
            (is_gap_down & significant_gap & df['is_bearish_breakout'])
        ])
        choices.extend([
            'Potential Breakaway Gap Up',
            'Potential Breakaway Gap Down'
        ])
    elif 'is_bullish_breakout' not in df.columns or 'is_bearish_breakout' not in df.columns:
        print("Warning: Breakout columns ('is_bullish_breakout', 'is_bearish_breakout') "
              "not found in DataFrame. Breakaway gap classification will be skipped.")

    conditions.extend([
        (is_gap_up & significant_gap),
        (is_gap_down & significant_gap)
    ])
    choices.extend([
        'Gap Up',
        'Gap Down'
    ])

    df['gap_type'] = np.select(conditions, choices, default='No Gap')
    df.loc[~significant_gap | (df['gap_type'] == 'No Gap'), 'gap_size_pct'] = 0.0

    data_df['gap_type'] = df['gap_type']
    data_df['gap_size_pct'] = df['gap_size_pct']
    return data_df

def identify_strong_trend(data_df: pd.DataFrame,
                                  pressure_window: int = 10,
                                  min_consecutive_bar_strength: int = 3,
                                  dominance_ratio_threshold: float = 0.6, # e.g. 60% of bars in one direction
                                  body_ratio_threshold: float = 0.6, # e.g. body is 60% of H-L range for a "strong bar"
                                  close_extreme_pct_threshold: float = 0.75 # Closes in top/bottom 25% (1-0.75)
                                 ) -> pd.DataFrame:
    """
    Identifies strong trend segments based on user's "buying/selling pressure" criteria,
    analyzing a rolling window of bars.
    Adds a 'strong_trend_status' column: 'Strong Uptrend', 'Strong Downtrend', or 'None'.
    """
    df = data_df.copy() # Work on a copy
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        print("Error: DataFrame must contain OHLC columns for identify_strong_trend.")
        # Ensure column exists even if we return early
        if 'strong_trend_status' not in data_df.columns:
            data_df['strong_trend_status'] = 'None'
        return data_df

    # Initialize the column in the original DataFrame structure that will be returned
    data_df['strong_trend_status'] = 'None'

    # Calculate bar properties on the copy
    df['is_bull_bar'] = df['Close'] > df['Open']
    df['is_bear_bar'] = df['Close'] < df['Open']
    df['body_size'] = np.abs(df['Close'] - df['Open'])
    df['hl_range'] = df['High'] - df['Low']
    # Replace 0 hl_range with NaN to prevent division by zero, then fill resulting NaNs if any
    df['hl_range_no_zero'] = df['hl_range'].replace(0, np.nan)

    df['body_hl_ratio'] = (df['body_size'] / df['hl_range_no_zero']).fillna(0)
    df['close_loc_norm'] = ((df['Close'] - df['Low']) / df['hl_range_no_zero']).fillna(0.5)


    # Iterate to apply rolling window logic
    # Start from index where a full window is available
    for i in range(pressure_window - 1, len(df)):
        window_slice = df.iloc[i - pressure_window + 1 : i + 1]

        num_bull_bars = window_slice['is_bull_bar'].sum()
        num_bear_bars = window_slice['is_bear_bar'].sum()

        consecutive_bull = 0
        temp_cb = 0
        for k_bull in range(len(window_slice)): # Check within the window
            if window_slice['is_bull_bar'].iloc[k_bull]:
                temp_cb += 1
            else:
                consecutive_bull = max(consecutive_bull, temp_cb)
                temp_cb = 0
        consecutive_bull = max(consecutive_bull, temp_cb) # Final check

        consecutive_bear = 0
        temp_cbear = 0
        for k_bear in range(len(window_slice)):
            if window_slice['is_bear_bar'].iloc[k_bear]:
                temp_cbear += 1
            else:
                consecutive_bear = max(consecutive_bear, temp_cbear)
                temp_cbear = 0
        consecutive_bear = max(consecutive_bear, temp_cbear)

        strong_bull_bar_count = window_slice[
            (window_slice['is_bull_bar']) & \
            (window_slice['body_hl_ratio'] >= body_ratio_threshold) & \
            (window_slice['close_loc_norm'] >= close_extreme_pct_threshold)
        ].shape[0]

        strong_bear_bar_count = window_slice[
            (window_slice['is_bear_bar']) & \
            (window_slice['body_hl_ratio'] >= body_ratio_threshold) & \
            (window_slice['close_loc_norm'] <= (1 - close_extreme_pct_threshold))
        ].shape[0]

        is_strong_uptrend = False
        if num_bull_bars / pressure_window >= dominance_ratio_threshold and \
           (consecutive_bull >= min_consecutive_bar_strength or \
            strong_bull_bar_count / pressure_window >= (dominance_ratio_threshold * 0.4)): # Relaxed a bit
            is_strong_uptrend = True

        is_strong_downtrend = False
        if num_bear_bars / pressure_window >= dominance_ratio_threshold and \
           (consecutive_bear >= min_consecutive_bar_strength or \
            strong_bear_bar_count / pressure_window >= (dominance_ratio_threshold * 0.4)):
            is_strong_downtrend = True

        # Get the original index label for the current bar (end of window)
        current_bar_original_index = df.index[i]

        if is_strong_uptrend and not is_strong_downtrend:
            data_df.loc[current_bar_original_index, 'strong_trend_status'] = 'Strong Uptrend'
        elif is_strong_downtrend and not is_strong_uptrend:
            data_df.loc[current_bar_original_index, 'strong_trend_status'] = 'Strong Downtrend'
        # else, it remains 'None' as initialized in data_df

    return data_df

def find_double_tops_bottoms(data_df: pd.DataFrame,
                             lookback_swings_for_pattern: int = 3,
                             price_similarity_pct: float = 0.02,
                             confirmation_break_ratio: float = 0.3) -> pd.DataFrame:
    df = data_df.copy()
    if not all(col in df.columns for col in ['is_swing_high', 'is_swing_low', 'High', 'Low', 'Close', 'Open']):
        print("Error: Required columns not found. Run find_swing_highs_lows first.")
        df['is_double_top_confirmed'] = False
        df['is_double_bottom_confirmed'] = False
        return df

    df['is_double_top_confirmed'] = False
    df['is_double_bottom_confirmed'] = False

    sh_points = df[df['is_swing_high']].copy()
    sl_points = df[df['is_swing_low']].copy()

    if len(sh_points) < 2 or len(sl_points) < 1: # Need at least 2 SH and 1 SL for a double top
        # For double bottom, need 2 SL and 1 SH. Check done before specific loop.
        pass # Will return df with False columns if not enough points for either.

    # Double Top: SH1 -> SL_valley -> SH2 (SH1 ~ SH2 in price) -> Break below SL_valley
    for i in range(len(sh_points) - 1):
        sh1 = sh_points.iloc[i]
        sh1_idx = sh_points.index[i]

        sh2 = sh_points.iloc[i+1] # Next available swing high
        sh2_idx = sh_points.index[i+1]

        valley_candidates = sl_points[(sl_points.index > sh1_idx) & (sl_points.index < sh2_idx)]
        if valley_candidates.empty: continue

        valley = valley_candidates.loc[valley_candidates['Low'].idxmin()]
        valley_idx = valley.name
        valley_price = valley['Low']

        if abs(sh1['High'] - sh2['High']) / sh1['High'] <= price_similarity_pct:
            height = abs(max(sh1['High'], sh2['High']) - valley_price)
            if height == 0: continue # Avoid division by zero or meaningless pattern
            confirmation_level = valley_price - (height * confirmation_break_ratio)

            bars_after_sh2 = df[df.index > sh2_idx]
            for bar_idx, bar_row in bars_after_sh2.iterrows():
                if bar_row['Low'] < confirmation_level:
                    df.loc[bar_idx, 'is_double_top_confirmed'] = True
                    break

    # Double Bottom: SL1 -> SH_peak -> SL2 (SL1 ~ SL2 in price) -> Break above SH_peak
    if len(sl_points) < 2 or len(sh_points) < 1: # Check specific needs for double bottom
        data_df[['is_double_top_confirmed', 'is_double_bottom_confirmed']] = df[['is_double_top_confirmed', 'is_double_bottom_confirmed']]
        return data_df # Return if not enough points for double bottom specifically

    for i in range(len(sl_points) - 1):
        sl1 = sl_points.iloc[i]
        sl1_idx = sl_points.index[i]

        sl2 = sl_points.iloc[i+1] # Next available swing low
        sl2_idx = sl_points.index[i+1]

        peak_candidates = sh_points[(sh_points.index > sl1_idx) & (sh_points.index < sl2_idx)]
        if peak_candidates.empty: continue

        peak = peak_candidates.loc[peak_candidates['High'].idxmax()]
        peak_idx = peak.name
        peak_price = peak['High']

        if abs(sl1['Low'] - sl2['Low']) / sl1['Low'] <= price_similarity_pct:
            height = abs(peak_price - min(sl1['Low'], sl2['Low']))
            if height == 0: continue
            confirmation_level = peak_price + (height * confirmation_break_ratio)

            bars_after_sl2 = df[df.index > sl2_idx]
            for bar_idx, bar_row in bars_after_sl2.iterrows():
                if bar_row['High'] > confirmation_level:
                    df.loc[bar_idx, 'is_double_bottom_confirmed'] = True
                    break

    data_df[['is_double_top_confirmed', 'is_double_bottom_confirmed']] = df[['is_double_top_confirmed', 'is_double_bottom_confirmed']]
    return data_df

def calculate_atr(data_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates Average True Range (ATR).
    Adds 'tr' (True Range) and 'atr' columns to the DataFrame.
    Uses Wilder's smoothing for ATR.
    """
    df = data_df.copy()
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'High', 'Low', 'Close' columns for ATR calculation.")
        if 'atr' not in df.columns: df['atr'] = np.nan
        if 'tr' not in df.columns: df['tr'] = np.nan
        # If essential columns are missing, return original df with potentially empty ATR/TR columns
        # Assign potentially empty series to the original DataFrame's columns
        data_df['tr'] = df.get('tr', pd.Series(index=df.index, dtype=float))
        data_df['atr'] = df.get('atr', pd.Series(index=df.index, dtype=float))
        return data_df

    df['prev_close'] = df['Close'].shift(1)

    df['h_minus_l'] = df['High'] - df['Low']
    df['h_minus_pc'] = np.abs(df['High'] - df['prev_close'])
    df['l_minus_pc'] = np.abs(df['Low'] - df['prev_close'])

    df['tr'] = df[['h_minus_l', 'h_minus_pc', 'l_minus_pc']].max(axis=1)

    df['atr'] = np.nan

    if len(df) >= period:
        df.loc[df.index[period - 1], 'atr'] = df['tr'].iloc[:period].mean()

        for i in range(period, len(df)):
            current_tr = df['tr'].iloc[i]
            prev_atr = df['atr'].iloc[i-1]
            # Ensure prev_atr is not nan; if it is, ATR calculation cannot proceed with Wilder's method.
            if pd.isna(prev_atr): # This might happen if initial TRs were all NaN leading to NaN mean
                df.loc[df.index[i], 'atr'] = df['tr'].iloc[i-period+1 : i+1].mean() # Fallback to SMA for current window
            else:
                 df.loc[df.index[i], 'atr'] = (prev_atr * (period - 1) + current_tr) / period
    else:
        print(f"Not enough data to calculate ATR for period {period}. ATR will be NaN.")

    data_df['tr'] = df['tr']
    data_df['atr'] = df['atr']
    return data_df

if __name__ == "__main__":
    print("Running market_structure.py example with Breakout Identification...")
    main_sample_data = None

    example_symbol_main = 'AAPL'; example_interval_main = '1d'; example_period_main = '1y'
    sw_window_main = 5; pt_lookback_main = 2; sr_touches_main = 2; sr_tol_main = 0.015; bo_lookback_main = 20

    config_available = 'config' in globals() and config is not None
    if config_available:
        example_symbol_main = getattr(config, 'EX_LSTM_TRAINER_SYMBOL', example_symbol_main)
        example_interval_main = getattr(config, 'PRIMARY_INTERVAL_LSTM', example_interval_main)
        example_period_main = getattr(config, 'PRIMARY_PERIOD_LSTM', example_period_main)
        sw_window_main = getattr(config, 'MS_SWING_WINDOW', sw_window_main)
        pt_lookback_main = getattr(config, 'MS_PULLBACK_TREND_LOOKBACK', pt_lookback_main)
        sr_touches_main = getattr(config, 'MS_SR_MIN_TOUCHES', sr_touches_main)
        sr_tol_main = getattr(config, 'MS_SR_RELATIVE_TOLERANCE', sr_tol_main)
        bo_lookback_main = getattr(config, 'MS_BREAKOUT_LOOKBACK_PERIOD', bo_lookback_main)

    if DATA_FETCHER_AVAILABLE:
        print(f"\nAttempting to fetch sample data for market structure analysis: {example_symbol_main} ({example_period_main})...")
        main_sample_data = fetch_price_data(symbol=example_symbol_main, interval=example_interval_main, period=example_period_main)
        if main_sample_data is None: print(f"Failed to fetch {example_symbol_main} data.")
        else: print(f"Successfully fetched {example_symbol_main} data for {main_sample_data.index.nunique()} days.")

    if main_sample_data is None:
        print(f"Using fallback manual data for market structure analysis as {example_symbol_main} fetch failed.")
        data_dict_fallback_values = {
            'Open':  [100,102,101,105,103,108,105,104,107,110,108,105,107,100,103,100,100,100,100,100],
            'High':  [105,103,102,110,108,110,109,106,110,112,110,109,108,104,105,100,100,100,100,100],
            'Low':   [98,101,100,103,102,105,103,102,105,108,104,103,102,99,100,100,100,100,100,100],
            'Close': [102,101,101,109,105,109,104,103,109,111,105,104,103,102,104,100,100,100,100,100],
            'Volume':[1000]*20 }
        main_sample_data = pd.DataFrame(data_dict_fallback_values, index=pd.date_range(start='2023-01-01', periods=20))

    daily_ohlc_data_for_pivots_and_lines = None
    if main_sample_data is not None and not main_sample_data.empty:
        daily_ohlc_data_for_pivots_and_lines = main_sample_data.copy()
        print("\n--- Analyzing Market Structure (Swings, Trend, S/R, Pullbacks, Breakouts) ---")
        data_analyzed = main_sample_data.copy()
        data_analyzed = find_swing_highs_lows(data_analyzed, window=sw_window_main)
        overall_trend = identify_simple_trend(data_analyzed.copy(), lookback_swings=pt_lookback_main)
        print(f"Overall Identified Trend: {overall_trend}")
        sr_levels_main = find_support_resistance_from_swings(data_analyzed.copy(), min_touches=sr_touches_main, relative_tolerance=sr_tol_main)
        print(f"S/R Levels - Support: {sr_levels_main.get('support')}, Resistance: {sr_levels_main.get('resistance')}")
        data_analyzed = identify_pullbacks(data_analyzed.copy(), trend_lookback_swings=pt_lookback_main)
        data_for_gap_test_breakouts = identify_breakouts(data_analyzed.copy(), lookback_period=bo_lookback_main, sr_min_touches=sr_touches_main, sr_tolerance=sr_tol_main) # Store for gap test

        pullback_bars = data_for_gap_test_breakouts[data_for_gap_test_breakouts['is_pullback_bar']]
        if not pullback_bars.empty:
            print(f"\n--- Pullback Bars ({len(pullback_bars)} instances) ---")
            print(pullback_bars[['Open', 'High', 'Low', 'Close', 'is_pullback_bar']].head())
        else: print("\nNo pullback bars identified.")

        breakout_bars = data_for_gap_test_breakouts[data_for_gap_test_breakouts['is_bullish_breakout'] | data_for_gap_test_breakouts['is_bearish_breakout']]
        if not breakout_bars.empty:
            print(f"\n--- Breakout Bars ({len(breakout_bars)} instances) ---")
            print(breakout_bars[['Open', 'High', 'Low', 'Close', 'is_bullish_breakout', 'is_bearish_breakout']].head())
        else: print("\nNo breakout bars identified.")

    # Manual breakout tests... (omitted for brevity in this thought block, but they are in the full code)
    print("\n\n--- Specific Manual Test for Breakout (Above Recent SH) ---") # ... and so on for all manual tests
    breakout_sh_values = {
        'Open':  [100, 103, 100, 105.5, 106], 'High':  [101, 105, 101, 106,   107],
        'Low':   [99,  102, 99,  104,   105], 'Close': [100.5,104,100.5,106,106.5], 'Volume':[100]*5 }
    breakout_sh_df = pd.DataFrame(breakout_sh_values, index=pd.date_range(start='2023-03-01', periods=5))
    breakout_sh_df = find_swing_highs_lows(breakout_sh_df.copy(), window=1)
    print("Manual Data for SH Breakout (Swings marked):")
    print(breakout_sh_df[['High','Low','is_swing_high','is_swing_low']])
    breakout_sh_df = identify_breakouts(breakout_sh_df.copy(), lookback_period=3, sr_min_touches=1, sr_tolerance=0.01)
    breakout_sh_bars = breakout_sh_df[breakout_sh_df['is_bullish_breakout']]
    if not breakout_sh_bars.empty: print(f"--- MANUAL Bullish Breakout (SH) ---\n{breakout_sh_bars[['Open','High','Low','Close', 'is_bullish_breakout']]}")
    else: print("No manual bullish SH breakout identified.")

    print("\n\n--- Specific Manual Test for Breakout (Below Recent SL) ---")
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
    breakout_res_values = {
        'Open':  [100, 101, 100, 101, 100, 101.5], 'High':  [101, 102, 101, 102, 101, 104],
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
    breakout_sup_values = {
        'Open':  [100, 99,  100, 99,  100, 98.5], 'High':  [101, 100, 101, 100, 101, 99],
        'Low':   [99,  98,  99,  98,  99,  96],   'Close': [99.5,99,  99.5,99, 98.6, 97], 'Volume':[100]*6 }
    breakout_sup_df = pd.DataFrame(breakout_sup_values, index=pd.date_range(start='2023-06-01', periods=6))
    breakout_sup_df = find_swing_highs_lows(breakout_sup_df.copy(), window=1)
    print("Manual Data for Support Breakout (Swings marked for S/R calc):")
    print(breakout_sup_df[['High','Low','is_swing_high','is_swing_low']])
    breakout_sup_df = identify_breakouts(breakout_sup_df.copy(), lookback_period=5, sr_min_touches=2, sr_tolerance=0.005)
    breakout_sup_bars = breakout_sup_df[breakout_sup_df['is_bearish_breakout']]
    if not breakout_sup_bars.empty: print(f"--- MANUAL Bearish Breakout (Support) ---\n{breakout_sup_bars[['Open','High','Low','Close', 'is_bearish_breakout']]}")
    else: print("No manual bearish support breakout identified.")

    print("\nMarket structure examples (breakouts etc.) finished.")

    # --- Daily Pivot Points Example ---
    print("\n--- Daily Pivot Points Example ---")
    daily_symbol_test_pivots = 'MSFT'
    daily_period_test_pivots = '70d'
    data_for_pivots_ref = None
    current_pivot_symbol_test = daily_symbol_test_pivots

    if DATA_FETCHER_AVAILABLE:
        if daily_ohlc_data_for_pivots_and_lines is not None and not daily_ohlc_data_for_pivots_and_lines.empty and \
           (daily_ohlc_data_for_pivots_and_lines.index.to_series().diff().dt.days.mode().iloc[0] == 1 if len(daily_ohlc_data_for_pivots_and_lines) > 1 else True) :
            print(f"Re-using already fetched daily data for {example_symbol_main} for pivot points.")
            data_for_pivots_ref = daily_ohlc_data_for_pivots_and_lines.copy()
            current_pivot_symbol_test = example_symbol_main
        else:
            print(f"Fetching new daily data for {daily_symbol_test_pivots} for pivot points...")
            data_for_pivots_ref = fetch_price_data(symbol=daily_symbol_test_pivots, interval='1d', period=daily_period_test_pivots)

        if data_for_pivots_ref is not None and not data_for_pivots_ref.empty:
            if len(data_for_pivots_ref) >= 2:
                daily_pivots = calculate_daily_pivot_points(data_for_pivots_ref)
                if daily_pivots:
                    print(f"Calculated Daily Pivot Points for {current_pivot_symbol_test}:")
                    keys_to_print = sorted(daily_pivots.keys())
                    for date_str in keys_to_print[-3:]:
                        print(f"For {date_str}: {daily_pivots[date_str]}")
                else:
                    print("Could not calculate daily pivots from the data (function returned empty).")
            else:
                print(f"Fetched only {len(data_for_pivots_ref)} day(s) of data for {current_pivot_symbol_test}, need at least 2 for pivots.")
        else:
            print(f"Could not fetch daily OHLC data for {current_pivot_symbol_test} to test pivots.")
    else:
        print("Data fetcher not available, cannot run pivot points example with fetched data.")

    # --- Trend Line Segments Example ---
    print("\n--- Trend Line Segments Example ---")
    data_for_trendlines_ref = None
    trendline_symbol_test = example_symbol_main
    data_with_swings_for_lines_ref = None

    if data_for_pivots_ref is not None and not data_for_pivots_ref.empty and \
       (data_for_pivots_ref.index.to_series().diff().dt.days.mode().iloc[0] == 1 if len(data_for_pivots_ref) > 1 else True):
        print(f"Using data fetched for pivot points ({current_pivot_symbol_test}) for trend line analysis.")
        data_for_trendlines_ref = data_for_pivots_ref.copy()
        trendline_symbol_test = current_pivot_symbol_test
    elif DATA_FETCHER_AVAILABLE:
        trendline_symbol_test = 'SPY'
        print(f"Fetching new daily data for {trendline_symbol_test} for trend line analysis (period: 1y)...")
        data_for_trendlines_ref = fetch_price_data(symbol=trendline_symbol_test, interval='1d', period='1y')

    trend_lines_result = None
    if data_for_trendlines_ref is not None and not data_for_trendlines_ref.empty:
        swing_window_for_lines = 10
        if config_available:
             swing_window_for_lines = getattr(config, 'MS_SWING_WINDOW', swing_window_for_lines)

        data_with_swings_for_lines_ref = find_swing_highs_lows(data_for_trendlines_ref.copy(), window=swing_window_for_lines)
        trend_lines_result = find_trend_line_segments(data_with_swings_for_lines_ref, lookback_swings=10, min_swing_separation=10)

        print(f"\nPotential Uptrend Line Segments for {trendline_symbol_test} (connecting last 10 swings, min 10 bars apart):")
        if trend_lines_result['uptrend_lines']:
            for line in trend_lines_result['uptrend_lines'][-5:]:
                p1_idx_str = str(line[0][0]).split(' ')[0]
                p2_idx_str = str(line[1][0]).split(' ')[0]
                print(f"  Line from ({p1_idx_str}, {line[0][1]:.2f}) to ({p2_idx_str}, {line[1][1]:.2f})")
        else:
            print("  No significant uptrend line segments found.")

        print(f"\nPotential Downtrend Line Segments for {trendline_symbol_test} (connecting last 10 swings, min 10 bars apart):")
        if trend_lines_result['downtrend_lines']:
            for line in trend_lines_result['downtrend_lines'][-5:]:
                p1_idx_str = str(line[0][0]).split(' ')[0]
                p2_idx_str = str(line[1][0]).split(' ')[0]
                print(f"  Line from ({p1_idx_str}, {line[0][1]:.2f}) to ({p2_idx_str}, {line[1][1]:.2f})")
        else:
            print("  No significant downtrend line segments found.")
    else:
        print(f"Could not fetch/use data for {trendline_symbol_test} to test trend lines.")

    # --- Basic Channels Example ---
    print("\n--- Basic Channels Example ---")
    if data_with_swings_for_lines_ref is not None and trend_lines_result is not None:
        if not data_with_swings_for_lines_ref.empty and (trend_lines_result['uptrend_lines'] or trend_lines_result['downtrend_lines']):
            basic_channels = find_basic_channels(data_with_swings_for_lines_ref, trend_lines_result)

            print(f"Potential Uptrend Channels for {trendline_symbol_test}:")
            if basic_channels['uptrend_channels']:
                for main_l, chan_l in basic_channels['uptrend_channels'][-3:]:
                    p1_main_idx_str = str(main_l[0][0]).split(' ')[0]
                    p2_main_idx_str = str(main_l[1][0]).split(' ')[0]
                    p1_chan_idx_str = str(chan_l[0][0]).split(' ')[0]
                    p2_chan_idx_str = str(chan_l[1][0]).split(' ')[0]
                    print(f"  Main Line: ({p1_main_idx_str}, {main_l[0][1]:.2f}) to ({p2_main_idx_str}, {main_l[1][1]:.2f})")
                    print(f"  Chan Line: ({p1_chan_idx_str}, {chan_l[0][1]:.2f}) to ({p2_chan_idx_str}, {chan_l[1][1]:.2f})")
            else:
                print("  No significant uptrend channels found.")

            print(f"Potential Downtrend Channels for {trendline_symbol_test}:")
            if basic_channels['downtrend_channels']:
                for main_l, chan_l in basic_channels['downtrend_channels'][-3:]:
                    p1_main_idx_str = str(main_l[0][0]).split(' ')[0]
                    p2_main_idx_str = str(main_l[1][0]).split(' ')[0]
                    p1_chan_idx_str = str(chan_l[0][0]).split(' ')[0]
                    p2_chan_idx_str = str(chan_l[1][0]).split(' ')[0]
                    print(f"  Main Line: ({p1_main_idx_str}, {main_l[0][1]:.2f}) to ({p2_main_idx_str}, {main_l[1][1]:.2f})")
                    print(f"  Chan Line: ({p1_chan_idx_str}, {chan_l[0][1]:.2f}) to ({p2_chan_idx_str}, {chan_l[1][1]:.2f})")
            else:
                print("  No significant downtrend channels found.")
        else:
            print("Data or trend lines not available for channel calculation (empty or no lines found).")
    else:
        print("Trend lines data or source data not available for channel calculation. Run Trend Line example first.")

    # --- Symmetrical Triangles Example ---
    print("\n--- Symmetrical Triangles Example ---")
    if data_with_swings_for_lines_ref is not None and trend_lines_result is not None:
        if not data_with_swings_for_lines_ref.empty:
            triangles = find_symmetrical_triangles(data_with_swings_for_lines_ref, trend_lines_result,
                                                   min_line_duration_bars=10,
                                                   max_apex_distance_bars=50)

            print(f"Potential Symmetrical Triangles for {trendline_symbol_test}:")
            if triangles:
                for tri in triangles[-3:]:
                    ul_p1_idx_str = str(tri['uptrend_line'][0][0]).split(' ')[0]
                    ul_p2_idx_str = str(tri['uptrend_line'][1][0]).split(' ')[0]
                    dl_p1_idx_str = str(tri['downtrend_line'][0][0]).split(' ')[0]
                    dl_p2_idx_str = str(tri['downtrend_line'][1][0]).split(' ')[0]
                    apex_date_str = "N/A"
                    apex_iloc_rounded = int(round(tri['apex_iloc']))
                    if 0 <= apex_iloc_rounded < len(data_with_swings_for_lines_ref.index):
                       apex_date_str = str(data_with_swings_for_lines_ref.index[apex_iloc_rounded]).split(' ')[0]

                    print(f"  Triangle formed by: ")
                    print(f"    Up Line: ({ul_p1_idx_str}, {tri['uptrend_line'][0][1]:.2f}) to ({ul_p2_idx_str}, {tri['uptrend_line'][1][1]:.2f})")
                    print(f"    Dn Line: ({dl_p1_idx_str}, {tri['downtrend_line'][0][1]:.2f}) to ({dl_p2_idx_str}, {tri['downtrend_line'][1][1]:.2f})")
                    print(f"    Apex (iloc, price, approx_date): ({tri['apex_iloc']:.0f}, {tri['apex_price']:.2f}, {apex_date_str})")
                    print(f"    Formation ends (iloc): {tri['formation_end_iloc']}")
            else:
                print("  No significant symmetrical triangles found with current settings.")
        else:
            print("Data not available for triangle calculation.")
    else:
        print("Trend lines data or source data not available for triangle calculation. Run Trend Line example first.")

    # --- Gap Identification Example ---
    print("\n--- Gap Identification Example ---")
    gap_test_symbol = 'MSFT'
    gap_test_period = '100d'
    if config_available:
        gap_test_symbol = getattr(config, 'SYMBOL', gap_test_symbol) # Using SYMBOL from general config for gap test
        gap_test_period = getattr(config, 'PRIMARY_PERIOD_BACKTEST', gap_test_period)

    data_for_gap_test = None
    if DATA_FETCHER_AVAILABLE:
        data_for_gap_test = fetch_price_data(symbol=gap_test_symbol, interval='1d', period=gap_test_period)

    if data_for_gap_test is not None and not data_for_gap_test.empty:
        sw_window_gap = 10; breakout_lookback_gap = 20; sr_min_touches_gap = 2; sr_tolerance_gap = 0.015
        if config_available:
            sw_window_gap = getattr(config, 'MS_SWING_WINDOW', sw_window_gap)
            breakout_lookback_gap = getattr(config, 'MS_BREAKOUT_LOOKBACK_PERIOD', breakout_lookback_gap)
            sr_min_touches_gap = getattr(config, 'MS_SR_MIN_TOUCHES', sr_min_touches_gap)
            sr_tolerance_gap = getattr(config, 'MS_SR_RELATIVE_TOLERANCE', sr_tolerance_gap)

        data_for_gap_test_swings = find_swing_highs_lows(data_for_gap_test.copy(), window=sw_window_gap)
        data_for_gap_test_breakouts = identify_breakouts(data_for_gap_test_swings.copy(),
                                                         lookback_period=breakout_lookback_gap,
                                                         sr_min_touches=sr_min_touches_gap,
                                                         sr_tolerance=sr_tolerance_gap)

        data_for_gap_test_breakouts = identify_gaps(data_for_gap_test_breakouts, min_gap_percentage=0.001)
        gaps_found = data_for_gap_test_breakouts[data_for_gap_test_breakouts['gap_type'] != 'No Gap']

        print(f"Identified Gaps for {gap_test_symbol} (min 0.1%):")
        if not gaps_found.empty:
            print(gaps_found[['Open', 'Close', 'gap_type', 'gap_size_pct']].tail(10).to_string())
        else:
            print("  No significant gaps found with current settings.")
    else:
        print(f"Data not available for {gap_test_symbol} to test gap identification.")

    # --- Strong Trend Identification Example (Pressure Based) ---
    print("\n--- Strong Trend Identification Example (Pressure Based) ---")
    # Attempt to import config for parameters, with fallbacks
    try:
        import config # Re-import or ensure it's available if main example already imported it
        trend_test_symbol_cfg = getattr(config, 'SYMBOL', 'BTC-USD') # Example, use a relevant symbol
        trend_test_interval_cfg = getattr(config, 'PRIMARY_INTERVAL_BACKTEST', '1h')
        trend_test_period_cfg = getattr(config, 'PRIMARY_PERIOD_BACKTEST', '90d')

        pressure_win_cfg = getattr(config, 'MS_PRESSURE_WINDOW', 10)
        min_consecutive_cfg = getattr(config, 'MS_MIN_CONSECUTIVE_STRENGTH', 3)
        dominance_ratio_cfg = getattr(config, 'MS_DOMINANCE_RATIO', 0.6)
        body_ratio_cfg = getattr(config, 'MS_BODY_RATIO_STRONG_BAR', 0.6)
        close_extreme_pct_cfg = getattr(config, 'MS_CLOSE_EXTREME_PCT', 0.75)
    except ImportError:
        print("config.py not found for strong trend example. Using default parameters.")
        trend_test_symbol_cfg = 'BTC-USD'
        trend_test_interval_cfg = '1h'
        trend_test_period_cfg = '90d'
        pressure_win_cfg = 10
        min_consecutive_cfg = 3
        dominance_ratio_cfg = 0.6
        body_ratio_cfg = 0.6
        close_extreme_pct_cfg = 0.75

    data_for_pressure_trend = None
    if DATA_FETCHER_AVAILABLE:
        data_for_pressure_trend = fetch_price_data(symbol=trend_test_symbol_cfg,
                                                 interval=trend_test_interval_cfg,
                                                 period=trend_test_period_cfg)
    else:
        print(f"Data fetcher not available, cannot fetch data for {trend_test_symbol_cfg} for pressure trend test.")


    if data_for_pressure_trend is not None and not data_for_pressure_trend.empty:
        # The function identify_strong_trend modifies the DataFrame in place by adding a column.
        # It returns the same DataFrame that was passed, now with the 'strong_trend_status' column.
        strong_trend_df_pressure = identify_strong_trend(
            data_for_pressure_trend, # Pass the original df to be modified
            pressure_window=pressure_win_cfg,
            min_consecutive_bar_strength=min_consecutive_cfg,
            dominance_ratio_threshold=dominance_ratio_cfg,
            body_ratio_threshold=body_ratio_cfg,
            close_extreme_pct_threshold=close_extreme_pct_cfg
        )

        print(f"Strong Trend Status for {trend_test_symbol_cfg} (Pressure Based):")
        # Now strong_trend_df_pressure is the same as data_for_pressure_trend but with the new column
        strong_segments = strong_trend_df_pressure[strong_trend_df_pressure['strong_trend_status'] != 'None']
        if not strong_segments.empty:
            print(strong_segments[['Close', 'strong_trend_status']].tail(20))
            print(f"Found {len(strong_segments[strong_segments['strong_trend_status'] == 'Strong Uptrend'])} 'Strong Uptrend' bars.")
            print(f"Found {len(strong_segments[strong_segments['strong_trend_status'] == 'Strong Downtrend'])} 'Strong Downtrend' bars.")
        else:
            print("  No strong trend segments identified with current pressure settings.")
    else:
        print(f"Data not available for {trend_test_symbol_cfg} to test pressure-based strong trend.")

    # --- Double Top/Bottom Example ---
    print("\n--- Double Top/Bottom Example ---")
    data_for_dbt = data_for_gap_test_breakouts # Re-use data that has swings
    if data_for_dbt is not None and not data_for_dbt.empty:
        if not all(col in data_for_dbt.columns for col in ['is_swing_high', 'is_swing_low']):
            print("Re-calculating swings for double top/bottom analysis as they are missing.")
            sw_window_dbt = 10
            if config_available: sw_window_dbt = getattr(config, 'MS_SWING_WINDOW', sw_window_dbt)
            data_for_dbt = find_swing_highs_lows(data_for_dbt.copy(), window=sw_window_dbt)

        dbt_df = find_double_tops_bottoms(data_for_dbt.copy(),
                                          lookback_swings_for_pattern=3,
                                          price_similarity_pct=0.03,
                                          confirmation_break_ratio=0.3)

        confirmed_d_tops = dbt_df[dbt_df['is_double_top_confirmed']]
        confirmed_d_bottoms = dbt_df[dbt_df['is_double_bottom_confirmed']]

        print(f"Confirmed Double Tops for {gap_test_symbol}:")
        if not confirmed_d_tops.empty:
            print(confirmed_d_tops[['Open', 'High', 'Low', 'Close', 'is_double_top_confirmed']].tail().to_string())
        else:
            print("  No confirmed double tops found.")

        print(f"Confirmed Double Bottoms for {gap_test_symbol}:")
        if not confirmed_d_bottoms.empty:
            print(confirmed_d_bottoms[['Open', 'High', 'Low', 'Close', 'is_double_bottom_confirmed']].tail().to_string())
        else:
            print("  No confirmed double bottoms found.")
    else:
        print(f"Data not available for {gap_test_symbol} to test double top/bottom.")

    # --- ATR Calculation Example ---
    print("\n--- ATR Calculation Example ---")
    atr_example_symbol_name = "used_from_previous_step" # Placeholder
    data_for_atr_calc = None

    # Re-use 'data_for_gap_test_breakouts' if it exists and is suitable (daily data)
    if 'data_for_gap_test_breakouts' in locals() and data_for_gap_test_breakouts is not None and not data_for_gap_test_breakouts.empty:
         data_for_atr_calc = data_for_gap_test_breakouts.copy() # Use daily data if available
         # Try to get the symbol name from the context if possible, otherwise use placeholder
         if 'gap_test_symbol' in locals():
             atr_example_symbol_name = gap_test_symbol
         print(f"Using data for '{atr_example_symbol_name}' from previous gap test example for ATR calculation.")

    if data_for_atr_calc is None: # Fallback to fetching fresh daily data
        atr_example_symbol_name = 'MSFT' # Default symbol for fresh fetch
        print(f"Fetching fresh daily data for ATR example ({atr_example_symbol_name}, 100d)")
        if DATA_FETCHER_AVAILABLE:
            data_for_atr_calc = fetch_price_data(symbol=atr_example_symbol_name, interval='1d', period='100d')
        else:
            print("DATA_FETCHER_AVAILABLE is False. Cannot fetch fresh data for ATR example.")

    if data_for_atr_calc is not None and not data_for_atr_calc.empty:
        atr_period_test = 14
        # Ensure 'High', 'Low', 'Close' are present
        if all(c in data_for_atr_calc.columns for c in ['High','Low','Close']):
            atr_df_example = calculate_atr(data_for_atr_calc, period=atr_period_test) # Operates on a copy
            print(f"ATR({atr_period_test}) for {atr_example_symbol_name}:")
            # Using .to_string() to ensure full tail(10) display if it's wide
            print(atr_df_example[['Close', 'tr', 'atr']].tail(10).to_string())
        else:
            print(f"ATR example skipped: Data for {atr_example_symbol_name} does not contain High, Low, Close columns.")
    else:
        print(f"Data not available for ATR calculation example (symbol: {atr_example_symbol_name}).")

    print("\nmarket_structure.py all examples finished.")
