import pandas as pd
import time
from datetime import datetime, timedelta

# Attempt to import project-specific modules
try:
    import config
    from data_fetcher import fetch_coinmarketcap_data
    # Assuming pre_process_data_for_signal_generation is globally accessible
    # from signal_generator module or a utils module.
    # For this subtask, we'll assume it's in signal_generator.
    from signal_generator import pre_process_data_for_signals, generate_initial_signals
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure all required files (config.py, data_fetcher.py, signal_generator.py) are present "
          "and correctly structured.")
    exit()

def get_seconds_for_interval(interval_str: str) -> int:
    if 'm' in interval_str:
        return int(interval_str.replace('m', '')) * 60
    elif 'h' in interval_str:
        return int(interval_str.replace('h', '')) * 60 * 60
    elif 'd' in interval_str:
        return int(interval_str.replace('d', '')) * 60 * 60 * 24
    # Default to 5 minutes if format is unknown, for safety, though config should be validated.
    print(f"Warning: Unknown interval format '{interval_str}'. Defaulting to 300 seconds.")
    return 300

def run_live_poc():
    print("Starting Live Trader POC (Proof of Concept)...")

    # --- Configuration ---
    if not hasattr(config, 'COINMARKETCAP_API_KEY') or \
       config.COINMARKETCAP_API_KEY == 'YOUR_API_KEY_HERE' or \
       not config.COINMARKETCAP_API_KEY:
        print("CRITICAL ERROR: CoinMarketCap API key not found or not set in config.py.")
        print("Please set COINMARKETCAP_API_KEY in config.py to run this script.")
        return

    api_key = config.COINMARKETCAP_API_KEY
    symbol = config.SYMBOL
    # Use PRIMARY_INTERVAL_BACKTEST as the live interval for consistency with config naming
    live_interval = config.PRIMARY_INTERVAL_BACKTEST
    seconds_per_interval = get_seconds_for_interval(live_interval)

    # Number of historical bars to fetch initially for indicators to warm up
    # Should be enough for all lookbacks (swing, breakouts, LSTM sequence)
    # e.g., max(SWING_WINDOW, BREAKOUT_LOOKBACK_PERIOD, LSTM_SEQUENCE_LENGTH) + some buffer
    # For LSTM_SEQUENCE_LENGTH = 20, BREAKOUT_LOOKBACK_PERIOD = 20, a count of 50-100 might be okay.
    # Let's use a slightly larger number for safety in PoC.
    initial_bar_count = getattr(config, 'LSTM_TRAIN_SEQUENCE_LENGTH', 20) + \
                          getattr(config, 'MS_BREAKOUT_LOOKBACK_PERIOD', 20) + 30 # Buffer. Used MS_BREAKOUT_LOOKBACK_PERIOD from config

    # Max bars to keep in live_data_df to prevent memory issues
    max_bars_to_keep = initial_bar_count + 50

    print(f"Symbol: {symbol}, Interval: {live_interval} ({seconds_per_interval}s)")
    print(f"Initial bars to fetch: {initial_bar_count}")
    print(f"Max bars to keep in memory: {max_bars_to_keep}")
    print(f"Using LSTM Filter: {config.SG_USE_LSTM_FILTER}")
    if config.SG_USE_LSTM_FILTER:
        print(f"LSTM Model: {config.LSTM_MODEL_PATH}, Scaler: {config.LSTM_SCALER_PATH}")


    # --- Initial Data Load ---
    print(f"Fetching initial {initial_bar_count} bars of historical data for {symbol}...")
    live_data_df = fetch_coinmarketcap_data(symbol=symbol,
                                            interval=live_interval,
                                            count=initial_bar_count,
                                            api_key=api_key)

    if live_data_df is None or live_data_df.empty:
        print("Failed to fetch initial historical data. Exiting.")
        return

    print(f"Initial data fetched successfully. Last timestamp: {live_data_df.index[-1]}")

    # --- Main Loop ---
    while True:
        loop_start_time = time.time()
        print(f"\n--- Loop Iteration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        try:
            # Fetch recent data (e.g., last 3-5 bars to catch up if there was a delay)
            print(f"Fetching latest market data for {symbol} ({live_interval})...")
            recent_data_df = fetch_coinmarketcap_data(symbol=symbol,
                                                      interval=live_interval,
                                                      count=5, # Fetch a few recent bars
                                                      api_key=api_key)

            if recent_data_df is not None and not recent_data_df.empty:
                # Identify new bars
                new_bars = recent_data_df[~recent_data_df.index.isin(live_data_df.index)]

                if not new_bars.empty:
                    print(f"Found {len(new_bars)} new bar(s). Appending to live data.")
                    # Append new bars and re-sort (though CMC data should be sorted)
                    live_data_df = pd.concat([live_data_df, new_bars]).sort_index()
                    # Remove duplicates that might arise if count overlaps perfectly
                    live_data_df = live_data_df[~live_data_df.index.duplicated(keep='last')]

                    # Trim old data
                    if len(live_data_df) > max_bars_to_keep:
                        live_data_df = live_data_df.iloc[-max_bars_to_keep:]

                    print(f"Live data updated. Total bars: {len(live_data_df)}. Last timestamp: {live_data_df.index[-1]}")
                else:
                    print("No new bars found in the latest fetch.")
            else:
                print("Failed to fetch recent data or no data returned.")
                # Continue to next iteration after sleep, maybe API is temporarily down

            # --- Run Analysis and Signal Generation ---
            if not live_data_df.empty:
                print("Processing data and generating signals...")
                # Create a copy for processing
                processing_df = live_data_df.copy()

                # Pre-process data (adds features, candlestick patterns, market structure)
                # This function needs the config object
                enriched_df = pre_process_data_for_signals(processing_df, config_module=config)

                if enriched_df is None or enriched_df.empty:
                    print("Error during data pre-processing. Skipping signal generation for this iteration.")
                else:
                    # Generate signals (includes LSTM filter if enabled in config)
                    # This function also needs many parameters from config
                    # Higher TF interval needs to be sourced from config_module for generate_initial_signals
                    higher_tf_interval_val = getattr(config, 'HIGHER_TF_INTERVAL', '1d') # Default if not found

                    signals_result_tuple = generate_initial_signals(
                        primary_df_input=enriched_df,
                        primary_symbol=symbol,
                        primary_interval=live_interval,
                        higher_tf_df_input=None, # For PoC, HTF data not fetched in loop. generate_initial_signals can handle None.
                                                 # A more advanced version would fetch/update HTF data too.
                        # Parameters below will use defaults from generate_initial_signals, which pull from its own global 'config' import.
                        # So, no need to pass them all explicitly if generate_initial_signals is self-sufficient with its config import.
                        # However, the subtask description passes them, so adhering to that:
                        sr_nearby_percentage=config.SG_SR_NEARBY_PERCENTAGE,
                        stop_loss_buffer_percentage=config.SG_STOP_LOSS_BUFFER_PERCENTAGE,
                        reward_ratio=config.SG_REWARD_RATIO,
                        lstm_model_path=config.LSTM_MODEL_PATH,
                        scaler_path=config.LSTM_SCALER_PATH,
                        lstm_sequence_length=config.SG_LSTM_SEQUENCE_LENGTH,
                        lstm_buy_threshold=config.SG_LSTM_BUY_THRESHOLD,
                        lstm_sell_threshold=config.SG_LSTM_SELL_THRESHOLD,
                        use_lstm_filter=config.SG_USE_LSTM_FILTER
                    )

                    signals_df_final = signals_result_tuple[0]

                    if signals_df_final is not None and not signals_df_final.empty:
                        latest_bar_analysis = signals_df_final.iloc[-1]
                        print("\n--- Latest Bar Analysis & Signal ---")
                        print(f"Timestamp:         {latest_bar_analysis.name}")
                        print(f"Close Price:       {latest_bar_analysis['Close']:.2f}")

                        signal_text = "HOLD"
                        if latest_bar_analysis['signal'] == 1:
                            signal_text = "BUY"
                        elif latest_bar_analysis['signal'] == -1:
                            signal_text = "SELL"
                        print(f"Signal:            {signal_text}")

                        if 'lstm_prediction' in latest_bar_analysis and pd.notna(latest_bar_analysis['lstm_prediction']):
                            print(f"LSTM Prediction:   {latest_bar_analysis['lstm_prediction']:.4f}")

                        if signal_text != "HOLD":
                            print(f"Stop Loss:         {latest_bar_analysis['stop_loss']:.2f}")
                            print(f"Take Profit:       {latest_bar_analysis['take_profit']:.2f}")
                        print("------------------------------------")
                    else:
                        print("Signal generation did not produce output or an error occurred.")
            else:
                print("Live data DataFrame is empty. Cannot process.")

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()

        # --- Sleep ---
        loop_duration = time.time() - loop_start_time
        sleep_time = max(0, seconds_per_interval - loop_duration)
        print(f"Loop duration: {loop_duration:.2f}s. Sleeping for {sleep_time:.2f}s...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    run_live_poc()
