import pandas as pd
import numpy as np
import os # For joining paths

# --- Import functions from other project modules ---
try:
    from signal_generator import generate_initial_signals, pre_process_data_for_signals
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError:
    print("Error: signal_generator.py not found or its functions could not be imported.")
    SIGNAL_GENERATOR_AVAILABLE = False

try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    print("Error: data_fetcher.py not found or fetch_price_data could not be imported.")
    DATA_FETCHER_AVAILABLE = False

try:
    import config # Import the configuration file
    LSTM_FEATURE_COLS = config.LSTM_FEATURE_COLS # Import specifically for create_manual_test_df
except ImportError:
    print("Error: config.py not found. Please ensure it exists in the same directory.")
    LSTM_FEATURE_COLS = [] # Fallback
    class config: # Minimal fallback
        EX_BT_PRIMARY_SYMBOL = 'AAPL'; EX_BT_PRIMARY_INTERVAL = '1h'; EX_BT_PRIMARY_PERIOD = '60d';
        EX_BT_HIGHER_TF_INTERVAL = '1d';
        EX_PREPROC_SWING_WINDOW = 10; EX_PREPROC_BREAKOUT_LOOKBACK = 20; EX_PREPROC_PULLBACK_LOOKBACK = 2;
        EX_PREPROC_SR_MIN_TOUCHES = 2; EX_PREPROC_SR_TOLERANCE = 0.015;
        SG_SR_NEARBY_PERCENTAGE = 0.02; SG_STOP_LOSS_BUFFER_PERCENTAGE = 0.005; SG_REWARD_RATIO = 2.0;
        LSTM_MODEL_PATH = os.path.join('trading_bot_artifacts','best_lstm_model.keras') # Adjusted for MODEL_DIR
        LSTM_SCALER_PATH = os.path.join('trading_bot_artifacts','lstm_scaler.gz')    # Adjusted for MODEL_DIR
        SG_LSTM_SEQUENCE_LENGTH = 20; SG_LSTM_BUY_THRESHOLD = 0.55; SG_LSTM_SELL_THRESHOLD = 0.45;
        SG_USE_LSTM_FILTER = True;
        BT_INITIAL_CAPITAL = 10000.0; BT_TRADE_SIZE_PERCENTAGE = 0.1;
        MODEL_DIR = "trading_bot_artifacts" # Needs to be part of fallback if used in path construction above

def run_backtest(processed_df: pd.DataFrame, initial_capital: float = 10000.0, trade_size_percentage: float = 0.10) -> dict:
    """
    Runs a simple event-driven backtest on data with trading signals.
    """
    if not all(col in processed_df.columns for col in ['Open', 'High', 'Low', 'Close', 'signal', 'stop_loss', 'take_profit']):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close', 'signal', 'stop_loss', 'take_profit' columns.")

    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    active_trade = None
    peak_equity = initial_capital
    max_drawdown = 0.0

    print(f"\nStarting backtest with initial capital: ${initial_capital:,.2f}")

    for index, row in processed_df.iterrows():
        current_bar_high_price = row['High']
        current_bar_low_price = row['Low']
        current_bar_close_price = row['Close']

        exit_price_for_this_bar = None

        if active_trade:
            profit = 0.0; trade_closed_this_bar = False; exit_reason = None
            if active_trade['type'] == 'buy':
                if current_bar_low_price <= active_trade['stop_loss']:
                    profit = (active_trade['stop_loss'] - active_trade['entry_price']) * active_trade['shares']
                    exit_price_for_this_bar = active_trade['stop_loss']; exit_reason = "SL"; trade_closed_this_bar = True
                elif current_bar_high_price >= active_trade['take_profit']:
                    profit = (active_trade['take_profit'] - active_trade['entry_price']) * active_trade['shares']
                    exit_price_for_this_bar = active_trade['take_profit']; exit_reason = "TP"; trade_closed_this_bar = True
            elif active_trade['type'] == 'sell':
                if current_bar_high_price >= active_trade['stop_loss']:
                    profit = (active_trade['entry_price'] - active_trade['stop_loss']) * active_trade['shares']
                    exit_price_for_this_bar = active_trade['stop_loss']; exit_reason = "SL"; trade_closed_this_bar = True
                elif current_bar_low_price <= active_trade['take_profit']:
                    profit = (active_trade['entry_price'] - active_trade['take_profit']) * active_trade['shares']
                    exit_price_for_this_bar = active_trade['take_profit']; exit_reason = "TP"; trade_closed_this_bar = True

            if trade_closed_this_bar:
                capital += profit
                trades.append({
                    'entry_date': active_trade['entry_index'], 'exit_date': index,
                    'type': active_trade['type'], 'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price_for_this_bar, 'shares': active_trade['shares'],
                    'profit': profit, 'reason': exit_reason })
                active_trade = None
                peak_equity = max(peak_equity, capital)
                drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

        if active_trade is None and row['signal'] != 0:
            if not pd.isna(row['stop_loss']) and not pd.isna(row['take_profit']):
                entry_price = current_bar_close_price
                stop_loss = row['stop_loss']; take_profit = row['take_profit']
                valid_sl = (row['signal'] == 1 and stop_loss < entry_price) or \
                           (row['signal'] == -1 and stop_loss > entry_price)
                if valid_sl:
                    position_size_value = capital * trade_size_percentage
                    shares_to_trade = position_size_value / entry_price
                    if shares_to_trade > 0:
                        active_trade = {
                            'type': 'buy' if row['signal'] == 1 else 'sell',
                            'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                            'entry_index': index, 'shares': shares_to_trade }
        equity_curve.append(capital)

    if active_trade:
        closing_price_at_end = processed_df['Close'].iloc[-1]; profit = 0.0
        if active_trade['type'] == 'buy': profit = (closing_price_at_end - active_trade['entry_price']) * active_trade['shares']
        elif active_trade['type'] == 'sell': profit = (active_trade['entry_price'] - closing_price_at_end) * active_trade['shares']
        capital += profit
        trades.append({
            'entry_date': active_trade['entry_index'], 'exit_date': processed_df.index[-1],
            'type': active_trade['type'], 'entry_price': active_trade['entry_price'],
            'exit_price': closing_price_at_end, 'shares': active_trade['shares'],
            'profit': profit, 'reason': 'EoD Close' })
        active_trade = None
        peak_equity = max(peak_equity, capital)
        drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['profit'] > 0)
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0
    net_profit = capital - initial_capital
    net_profit_percentage = (net_profit / initial_capital) * 100 if initial_capital > 0 else 0
    avg_win_amount = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss_amount = gross_loss / losing_trades if losing_trades > 0 else 0
    avg_trade_profit = net_profit / total_trades if total_trades > 0 else 0
    returns_series = pd.Series(equity_curve).pct_change().dropna()
    sharpe_ratio_annualized = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() != 0 and len(returns_series) > 1 else 0

    return {
        'initial_capital': initial_capital, 'final_capital': capital, 'net_profit': net_profit,
        'net_profit_percentage': net_profit_percentage, 'total_trades': total_trades,
        'winning_trades': winning_trades, 'losing_trades': losing_trades, 'win_rate_percentage': win_rate,
        'gross_profit': gross_profit, 'gross_loss': gross_loss, 'profit_factor': profit_factor,
        'average_win_amount': avg_win_amount, 'average_loss_amount': avg_loss_amount,
        'average_trade_profit': avg_trade_profit, 'max_drawdown_percentage': max_drawdown * 100,
        'sharpe_ratio_annualized (approx)': sharpe_ratio_annualized,
        'equity_curve': equity_curve, 'trades_log': trades
    }

def create_manual_test_df_for_backtester():
    data = {
        'Open':  [100, 101, 100, 98,  99,  102, 103, 101, 100, 98 ],
        'High':  [102, 103, 101, 99,  100, 104, 105, 103, 101, 99 ],
        'Low':   [99,  100, 98,  97,  98,  100, 102, 100, 99,  97 ],
        'Close': [101, 102, 99,  97,  100, 103, 102, 100, 99,  97 ],
        'signal':      [1,  0,   0,  0,   -1,  0,   0,   0,   0,  0  ],
        'stop_loss':   [99.0, np.nan, np.nan, np.nan, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'take_profit': [103.0,np.nan, np.nan, np.nan, 98.0,  np.nan, np.nan, np.nan, np.nan, np.nan] }
    idx = pd.date_range(start='2023-01-01', periods=len(data['Open']), freq='D')
    df = pd.DataFrame(data, index=idx)

    # Use LSTM_FEATURE_COLS from config (imported at the top of the file)
    # Fallback if LSTM_FEATURE_COLS is empty (e.g. config not loaded properly)
    cols_to_add = config.LSTM_FEATURE_COLS if LSTM_FEATURE_COLS else [
        'return', 'body_range_norm', 'hl_range_norm', 'Volume', 'is_doji', 'is_marubozu',
        'is_outside_bar', 'is_inside_bar', 'is_swing_high', 'is_swing_low',
        'is_pullback_bar', 'is_bullish_breakout', 'is_bearish_breakout']

    for col in cols_to_add:
        if col not in df.columns:
            # For simplicity, add as float if it's a known numeric type, else int (for bools)
            if col in ['return', 'body_range_norm', 'hl_range_norm']:
                df[col] = 0.0
            else: # Includes Volume and all boolean features
                df[col] = 0
    return df

if __name__ == "__main__":
    if not (SIGNAL_GENERATOR_AVAILABLE and DATA_FETCHER_AVAILABLE and 'config' in globals()): # Check if config was imported
        print("Signal generator, Data fetcher, or Config not available. Cannot run backtester example.")
    else:
        print("Running backtester.py example using config.py...")
        print("\n--- Configuration Parameters ---")
        print(f"Symbol for Backtest: {config.EX_BT_PRIMARY_SYMBOL}")
        print(f"Primary Interval for Backtest: {config.EX_BT_PRIMARY_INTERVAL}")
        print(f"Primary Period for Backtest: {config.EX_BT_PRIMARY_PERIOD}")
        print(f"Higher Timeframe Interval: {config.EX_BT_HIGHER_TF_INTERVAL}")
        print(f"Initial Capital: ${config.BT_INITIAL_CAPITAL:,.2f}")
        print(f"Trade Size Percentage: {config.BT_TRADE_SIZE_PERCENTAGE*100:.2f}%")
        print(f"Use LSTM Filter for Signals: {config.SG_USE_LSTM_FILTER}")
        if hasattr(config, 'MODEL_DIR'): # Check if MODEL_DIR is defined
            print(f"Model Directory: {config.MODEL_DIR}")
        print(f"LSTM Model Path: {config.LSTM_MODEL_PATH}")
        print(f"LSTM Scaler Path: {config.LSTM_SCALER_PATH}")
        # Print ATR Stop specific config from config file
        print(f"Use ATR Stop for SL: {getattr(config, 'SG_USE_ATR_STOP', False)}")
        if getattr(config, 'SG_USE_ATR_STOP', False):
            print(f"  ATR Period: {getattr(config, 'SG_ATR_PERIOD', 14)}")
            print(f"  ATR Multiplier: {getattr(config, 'SG_ATR_MULTIPLIER', 2.0)}")

        # New features config printout for Pivots and DT/DB
        print(f"Use Pivot Filter for Rule-Based Signals: {getattr(config, 'SG_USE_PIVOT_FILTER', False)}")
        if getattr(config, 'SG_USE_PIVOT_FILTER', False):
            print(f"  Pivot Proximity Pct: {getattr(config, 'SG_PIVOT_PROXIMITY_PCT', 0.005)}")
            print(f"  Pivot Levels to Consider: {getattr(config, 'SG_CONSIDER_PIVOT_LEVELS', ['PP', 'S1', 'R1'])}")

        print(f"Use Double Top/Bottom Pattern Signals: {getattr(config, 'SG_USE_DOUBLE_TOP_BOTTOM_SIGNALS', False)}")
        if getattr(config, 'SG_USE_DOUBLE_TOP_BOTTOM_SIGNALS', False):
            # These MS_DTB parameters are used in pre_process_data_for_signals but are relevant to the DT/DB strategy
            print(f"  DT/DB Lookback Swings (MS_DTB_LOOKBACK_SWINGS): {getattr(config, 'MS_DTB_LOOKBACK_SWINGS', 3)}")
            print(f"  DT/DB Price Similarity Pct (MS_DTB_SIMILARITY_PCT): {getattr(config, 'MS_DTB_SIMILARITY_PCT', 0.03)}")
            print(f"  DT/DB Confirmation Ratio (MS_DTB_CONFIRMATION_RATIO): {getattr(config, 'MS_DTB_CONFIRMATION_RATIO', 0.3)}")
            print(f"  DT/DB Specific Reward Ratio (if ATR not used): {getattr(config, 'SG_DBL_TOP_BOTTOM_REWARD_RATIO', config.SG_REWARD_RATIO)}") # Defaults to general R:R

        print(f"Use Best Signal Bar Filter: {getattr(config, 'SG_USE_BEST_SIGNAL_BAR_FILTER', False)}")
        if getattr(config, 'SG_USE_BEST_SIGNAL_BAR_FILTER', False):
            # These CP_BEST_SIG... parameters are used in candlestick_patterns.py but are relevant to the overall strategy settings
            print(f"  Best Bull Bar Lower Wick Min/Max Ratio: {getattr(config, 'CP_BEST_SIG_LOWER_WICK_MIN_RATIO_BULL', 0.25)} / {getattr(config, 'CP_BEST_SIG_LOWER_WICK_MAX_RATIO_BULL', 0.60)}")
            print(f"  Best Bull Bar Upper Wick Max Ratio: {getattr(config, 'CP_BEST_SIG_UPPER_WICK_MAX_SIZE_RATIO_BULL', 0.10)}")
            print(f"  Best Bear Bar Upper Wick Min/Max Ratio: {getattr(config, 'CP_BEST_SIG_UPPER_WICK_MIN_RATIO_BEAR', 0.25)} / {getattr(config, 'CP_BEST_SIG_UPPER_WICK_MAX_RATIO_BEAR', 0.60)}")
            print(f"  Best Bear Bar Lower Wick Max Ratio: {getattr(config, 'CP_BEST_SIG_LOWER_WICK_MAX_SIZE_RATIO_BEAR', 0.10)}")
            print(f"  Best Signal Bar Max Body Overlap with Prev Bar: {getattr(config, 'CP_BEST_SIG_MAX_BODY_OVERLAP_PREV_BAR_RATIO', 0.5)}")
            print(f"  Best Signal Bar Close N-Bar Extreme: {getattr(config, 'CP_BEST_SIG_CLOSE_EXTREME_N_BARS', 3)}")

        print(f"Use 2-Bar Reversal Signals: {getattr(config, 'SG_USE_2BAR_REVERSAL_SIGNALS', False)}")
        if getattr(config, 'SG_USE_2BAR_REVERSAL_SIGNALS', False):
            # These CP_2BAR... parameters are used in candlestick_patterns.py but are relevant
            print(f"  2-Bar Reversal Lookback Period (CP_2BAR_LOOKBACK_PERIOD): {getattr(config, 'CP_2BAR_LOOKBACK_PERIOD', 5)}")
            print(f"  2-Bar Body Similarity Pct (CP_2BAR_BODY_SIMILARITY_PCT): {getattr(config, 'CP_2BAR_BODY_SIMILARITY_PCT', 0.25)}")
            print(f"  2-Bar Second Bar Close Extreme Pct (CP_2BAR_SECOND_BAR_CLOSE_EXTREME_PCT): {getattr(config, 'CP_2BAR_SECOND_BAR_CLOSE_EXTREME_PCT', 0.80)}")
            print(f"  2-Bar Reversal Reward Ratio (SG_2BAR_REVERSAL_REWARD_RATIO): {getattr(config, 'SG_2BAR_REVERSAL_REWARD_RATIO', config.SG_REWARD_RATIO)}") # Defaults to general R:R
        print("---------------------------------")

        # --- Test with Manually Crafted Data First ---
        # This manual test might not be as relevant if full live data processing is the focus
        # but can be kept for basic backtester logic verification.
        print("\n--- Testing Backtester with Manual Data ---")
        manual_df = create_manual_test_df_for_backtester()
        manual_results = run_backtest(
            manual_df,
            initial_capital=config.BT_INITIAL_CAPITAL,
            trade_size_percentage=config.BT_TRADE_SIZE_PERCENTAGE
        )
        print("\n--- Manual Backtest Results ---")
        for key, value in manual_results.items():
            if key not in ['equity_curve', 'trades_log']:
                # Pre-format value if it's a float for consistency
                display_value = f"{value:,.2f}" if isinstance(value, float) else value
                print(f"{key.replace('_', ' ').title()}: {display_value}")
        print("\nTrades Log (Manual):")
        manual_trades_log_df = pd.DataFrame(manual_results['trades_log'])
        if not manual_trades_log_df.empty:
            print(manual_trades_log_df.to_string(index=False))
        else:
            print("No trades in manual backtest.")


        # --- Test with Live Data ---
        print(f"\n\n--- Testing Backtester with Live Data ({config.EX_BT_PRIMARY_SYMBOL}) ---")

        ohlcv_data_live = fetch_price_data(
            symbol=config.EX_BT_PRIMARY_SYMBOL,
            interval=config.EX_BT_PRIMARY_INTERVAL,
            period=config.EX_BT_PRIMARY_PERIOD
        )

        htf_df_live = None
        if ohlcv_data_live is not None and not ohlcv_data_live.empty:
            if not isinstance(ohlcv_data_live.index, pd.DatetimeIndex):
                ohlcv_data_live.index = pd.to_datetime(ohlcv_data_live.index)
            htf_start_live = ohlcv_data_live.index.min().strftime('%Y-%m-%d')
            htf_end_live = (ohlcv_data_live.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            htf_df_live = fetch_price_data(
                symbol=config.EX_BT_PRIMARY_SYMBOL,
                interval=config.EX_BT_HIGHER_TF_INTERVAL,
                start_date=htf_start_live,
                end_date=htf_end_live
            )

        if ohlcv_data_live is not None and not ohlcv_data_live.empty:
            print("\nPre-processing live data for signals...")
            # Updated call to pass config_module, assuming pre_process_data_for_signals
            # now sources its parameters from the config_module internally.
            enriched_live_df = pre_process_data_for_signals(
                ohlcv_data_live,
                config_module=config
            )
            print(f"Enriched live DataFrame shape: {enriched_live_df.shape}")

            missing_cols = [col for col in config.LSTM_FEATURE_COLS if col not in enriched_live_df.columns]
            if missing_cols:
                print(f"FATAL: Enriched DF for live backtest is missing required LSTM feature columns: {missing_cols}")
            else:
                for col in config.LSTM_FEATURE_COLS:
                    if enriched_live_df[col].isnull().any():
                        is_bool_col_in_lstm_features = enriched_live_df[col].dtype == 'bool'
                        if is_bool_col_in_lstm_features:
                             enriched_live_df[col] = enriched_live_df[col].fillna(False)
                        else:
                             print(f"Warning: Live data, column '{col}' has NaNs post pre-processing. Filling with 0.")
                             enriched_live_df[col] = enriched_live_df[col].fillna(0)

                signals_df_live, htf_trend_live, ptf_trend_live, ptf_sr_levels_live = generate_initial_signals(
                    primary_df_input=enriched_live_df,
                    primary_symbol=config.EX_BT_PRIMARY_SYMBOL,
                    primary_interval=config.EX_BT_PRIMARY_INTERVAL,
                    higher_tf_df_input=htf_df_live,
                    # Parameters below will use defaults from generate_initial_signals, which are already linked to config
                    # sr_nearby_percentage=config.SG_SR_NEARBY_PERCENTAGE,
                    # stop_loss_buffer_percentage=config.SG_STOP_LOSS_BUFFER_PERCENTAGE,
                    # reward_ratio=config.SG_REWARD_RATIO,
                    # lstm_model_path=config.LSTM_MODEL_PATH,
                    # scaler_path=config.LSTM_SCALER_PATH,
                    # lstm_sequence_length=config.SG_LSTM_SEQUENCE_LENGTH,
                    # lstm_buy_threshold=config.SG_LSTM_BUY_THRESHOLD,
                    # lstm_sell_threshold=config.SG_LSTM_SELL_THRESHOLD,
                    # use_lstm_filter=config.SG_USE_LSTM_FILTER
                )

                if signals_df_live is not None and not signals_df_live.empty:
                    backtest_results_live = run_backtest(
                        signals_df_live,
                        initial_capital=config.BT_INITIAL_CAPITAL,
                        trade_size_percentage=config.BT_TRADE_SIZE_PERCENTAGE
                    )
                    print("\n--- Live Data Backtest Results ---")
                    for key, value in backtest_results_live.items():
                        if key not in ['equity_curve', 'trades_log']:
                            display_value = f"{value:,.2f}" if isinstance(value, float) else value
                            print(f"  {key.replace('_', ' ').title()}: {display_value}")

                    # Save Equity Curve
                    if 'equity_curve' in backtest_results_live and signals_df_live is not None:
                        # The equity curve has one extra initial point for the starting capital
                        # Align with signal_df index, ensuring to handle length difference
                        equity_index = signals_df_live.index[:len(backtest_results_live['equity_curve'])-1]
                        # Prepend a placeholder for the initial capital point if needed, or adjust index
                        # For simplicity, if lengths mismatch significantly, use a simple range index.
                        # Assuming one equity point per bar after initial capital.
                        if len(backtest_results_live['equity_curve']) == len(signals_df_live.index) + 1:
                             # Use a combined index: one before start for initial, then the df index
                             # This is tricky. Let's use the simpler approach:
                             # equity_curve_df = pd.DataFrame({'Equity': backtest_results_live['equity_curve']})
                             # Or, if signals_df_live.index is used, ensure equity_curve starts from first bar's end equity
                             # The current equity_curve includes initial capital as first point.
                             # So, its length is num_bars + 1.
                             # If signals_df_live has N bars, equity_curve has N+1 points.
                             # We can associate the equity points with the *end* of each bar,
                             # with the first point being initial capital *before* any bars.

                             # Let's use the index of the signals_df for the equity points *after* the initial capital.
                             # So, skip the first equity point (initial capital) for alignment with bar data.
                            equity_curve_for_df = backtest_results_live['equity_curve'][1:]
                            if len(equity_curve_for_df) == len(signals_df_live.index):
                                equity_curve_df = pd.DataFrame(
                                    {'Equity': equity_curve_for_df},
                                    index=signals_df_live.index
                                )
                            else: # Fallback if lengths still don't align as expected
                                print(f"Warning: Equity curve length ({len(equity_curve_for_df)}) and signal DF length ({len(signals_df_live.index)}) mismatch. Using range index for equity curve.")
                                equity_curve_df = pd.DataFrame({'Equity': backtest_results_live['equity_curve']})
                        else: # Fallback for unexpected length
                            print(f"Warning: Equity curve length ({len(backtest_results_live['equity_curve'])}) not N+1 of signal DF length ({len(signals_df_live.index)}). Using range index for equity curve.")
                            equity_curve_df = pd.DataFrame({'Equity': backtest_results_live['equity_curve']})

                        equity_curve_filename = "backtest_equity_curve.csv"
                        equity_curve_df.to_csv(equity_curve_filename)
                        print(f"\nEquity curve saved to {equity_curve_filename}")

                    # Save Trades Log
                    if 'trades_log' in backtest_results_live:
                        trades_log_df = pd.DataFrame(backtest_results_live['trades_log'])
                        trades_log_filename = "backtest_trades_log.csv"
                        if not trades_log_df.empty:
                            trades_log_df.to_csv(trades_log_filename, index=False)
                            print(f"Trades log saved to {trades_log_filename}")
                            print("\n  Trades Log (Live Data - First 5 if many):")
                            print(trades_log_df.head().to_string(index=False))
                            if len(trades_log_df) > 5: print("    ... (more trades in CSV)")
                        else:
                            print("\n  Trades Log (Live Data): No trades executed.")
                    else:
                        print("\nTrades log not found in backtest results.")

                else:
                    print("\nNo signals generated from live data; cannot run backtest.")
        else:
            print(f"Could not fetch data for {config.EX_BT_PRIMARY_SYMBOL}. Backtest on live data aborted.")

        print("\nbacktester.py example finished.")
