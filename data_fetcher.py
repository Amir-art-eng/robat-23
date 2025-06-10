import yfinance as yf
import pandas as pd
import os
# Note: requests and json are for the CoinMarketCap function, which is out of scope for this specific subtask version.
# import requests
# import json
# from datetime import datetime

# DEFAULT_DATA_DIRECTORY = "market_data" # Related to caching, not used in this simplified version

# Caching functions are preserved for potential future use but not integrated into fetch_price_data for this subtask
# def save_data_to_csv(data: pd.DataFrame, symbol: str, interval: str, data_directory: str = DEFAULT_DATA_DIRECTORY):
#     """
#     Saves the DataFrame to a CSV file.
#     """
#     if data is None or data.empty:
#         print(f"No data provided for {symbol}_{interval}, skipping save.")
#         return
#     try:
#         os.makedirs(data_directory, exist_ok=True)
#         filename = f"{symbol.upper()}_{interval}.csv"
#         filepath = os.path.join(data_directory, filename)
#         data.to_csv(filepath)
#         print(f"Data for {symbol}_{interval} saved to {filepath}")
#     except Exception as e:
#         print(f"Error saving data for {symbol}_{interval} to CSV: {e}")

# def load_data_from_csv(symbol: str, interval: str, data_directory: str = DEFAULT_DATA_DIRECTORY) -> pd.DataFrame | None:
#     """
#     Loads data from a CSV file if it exists.
#     """
#     filename = f"{symbol.upper()}_{interval}.csv"
#     filepath = os.path.join(data_directory, filename)
#     if os.path.exists(filepath):
#         try:
#             data = pd.read_csv(filepath, index_col=0, parse_dates=True)
#             print(f"Data for {symbol}_{interval} loaded from {filepath}")
#             return data
#         except Exception as e:
#             print(f"Error loading data for {symbol}_{interval} from CSV: {e}")
#             return None
#     else:
#         return None

def fetch_price_data(symbol: str, interval: str, period: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame | None:
    """
    Fetches historical stock price data using yfinance.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL').
        interval (str): The data interval (e.g., '1m', '5m', '15m', '1h', '1d').
                       Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
                       Intraday data (e.g., '1m') is limited to the last 7 days by default if period is not specified,
                       or up to 60 days if start/end dates are used.
        period (str, optional): How far back to fetch data (e.g., '1d', '5d', '1mo', '1y', 'max').
                                This is used if start_date and end_date are None.
        start_date (str, optional): Start date string in 'YYYY-MM-DD' format. Used if period is None.
        end_date (str, optional): End date string in 'YYYY-MM-DD' format. Used if period is None.

    Returns:
        pandas.DataFrame: OHLCV data with a DatetimeIndex, or None if an error occurs or no data is found.
    """
    print(f"Fetching data for {symbol} ({interval}) from yfinance...")
    try:
        ticker = yf.Ticker(symbol)

        # Parameters for yfinance .history()
        history_params = {'interval': interval}

        if period:
            history_params['period'] = period
            print(f"Using period: {period}")
        elif start_date and end_date:
            history_params['start'] = start_date
            history_params['end'] = end_date
            print(f"Using start_date: {start_date}, end_date: {end_date}")
        else:
            # If neither period nor start/end dates are provided,
            # yfinance defaults: 'max' for daily/weekly, limited history for intraday (e.g., 7d for 1m if no period specified).
            # This behavior is fine as per the function's purpose.
            print("No period or start/end dates specified. yfinance will use its default duration for the interval.")

        data = ticker.history(**history_params)

        if data.empty:
            print(f"No data found for symbol '{symbol}' with parameters: {history_params}")
            return None

        print(f"Successfully fetched {len(data)} rows of data for {symbol} ({interval}).")
        return data

    except Exception as e:
        # Common errors: invalid symbol, network issues, changes in yfinance API.
        print(f"Error fetching data for {symbol} ({interval}) from yfinance: {e}")
        return None

if __name__ == "__main__":
    print("Starting yfinance data fetcher examples...")

    # Example 1: Fetch 1-minute data for 'MSFT' for the last 1 day.
    # For '1m' interval, 'period="1d"' typically means data for the most recent available trading day.
    # yfinance limits '1m' data retrieval; 'period' cannot exceed 7 days.
    print("\n--- Example 1: MSFT 1-minute data for the last 1 day ---")
    msft_1m = fetch_price_data(symbol='MSFT', interval='1m', period='1d')
    if msft_1m is not None:
        print(f"MSFT 1m data (period='1d') (shape: {msft_1m.shape}):")
        print(msft_1m.head())
        if len(msft_1m) > 5: # Show tail only if more than 5 rows
            print(msft_1m.tail())
    else:
        print(f"Failed to fetch MSFT 1m data for period '1d'. Error message should be above.")

    # Example 2: Fetch 5-minute data for 'AAPL' for the period '5d'.
    print("\n--- Example 2: AAPL 5-minute data for '5d' period ---")
    aapl_5m = fetch_price_data(symbol='AAPL', interval='5m', period='5d')
    if aapl_5m is not None:
        print(f"AAPL 5m data (period='5d') (shape: {aapl_5m.shape}):")
        print(aapl_5m.head())
        if len(aapl_5m) > 5:
            print(aapl_5m.tail())
    else:
        print(f"Failed to fetch AAPL 5m data for period '5d'. Error message should be above.")

    # Example 3: Fetch daily data for 'GOOGL' from '2023-01-01' to '2023-12-31'.
    print("\n--- Example 3: GOOGL daily data from '2023-01-01' to '2023-12-31' ---")
    googl_daily = fetch_price_data(symbol='GOOGL', interval='1d', start_date='2023-01-01', end_date='2023-12-31')
    if googl_daily is not None:
        print(f"GOOGL 1d data (2023-01-01 to 2023-12-31) (shape: {googl_daily.shape}):")
        print(googl_daily.head())
        if len(googl_daily) > 5:
            print(googl_daily.tail())
    else:
        print(f"Failed to fetch GOOGL 1d data for the specified date range. Error message should be above.")

    # Example 4: Fetch 1-hour data for an invalid symbol like 'NONEXISTENTICKER' to demonstrate error handling.
    # Using period='1d' as a valid period parameter for the attempt.
    print("\n--- Example 4: Error handling for 'NONEXISTENTICKER' (1h data, period='1d') ---")
    invalid_data = fetch_price_data(symbol='NONEXISTENTICKER', interval='1h', period='1d')
    if invalid_data is None:
        # The function fetch_price_data should have printed an error message.
        print("Fetching data for 'NONEXISTENTICKER' failed as expected.")
    else:
        # This case should ideally not be reached if the symbol is truly invalid.
        print(f"Unexpectedly received data for 'NONEXISTENTICKER' (shape: {invalid_data.shape}):")
        print(invalid_data.head())

    print("\nAll yfinance data fetcher examples finished.")

# The CoinMarketCap related code is preserved below but commented out or made non-operational
# to keep this script focused on the yfinance requirements of the current subtask.

# def fetch_coinmarketcap_data(symbol: str, interval: str, count: int, api_key: str, convert: str = 'USD'):
#     """
#     Fetches historical OHLCV data from CoinMarketCap API.
#     (This function is out of scope for the current subtask and is not actively used)
#     """
#     # BASE_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical'
#     # headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': api_key}
#     # parameters = {'symbol': symbol, 'count': count, 'interval': interval, 'convert': convert}
#     # print(f"Attempting to fetch {count} intervals of {interval} data for {symbol} from CoinMarketCap...")
#     # try:
#     #     response = requests.get(BASE_URL, headers=headers, params=parameters)
#     #     response.raise_for_status()
#     #     data = response.json()
#     #     # ... (parsing logic would follow) ...
#     #     # ohlcv_list = []
#     #     # ...
#     #     # df = pd.DataFrame(ohlcv_list)
#     #     # df.set_index('Timestamp', inplace=True)
#     #     # df.sort_index(inplace=True)
#     #     # print(f"Successfully fetched and processed {len(df)} data points from CoinMarketCap.")
#     #     # return df
#     #     print("CoinMarketCap fetch function is currently a placeholder/deactivated for this subtask.")
#     return None
#     # except requests.exceptions.RequestException as e:
#     #     print(f"Error fetching data from CoinMarketCap: {e}")
#     #     return None
#     # except (KeyError, TypeError, ValueError) as e:
#     #     print(f"Error parsing CoinMarketCap JSON response: {e}")
#     #     return None

# if __name__ == "__main__":
    # ... (previous yfinance examples) ...

    # --- CoinMarketCap Fetcher Example (Commented out for this subtask) ---
    # print("\n--- CoinMarketCap Fetcher Example (Currently Deactivated) ---")
    # print("CoinMarketCap examples are commented out to focus on yfinance for this subtask.")
    # try:
    #     import config
    #     if hasattr(config, 'COINMARKETCAP_API_KEY') and \
    #        config.COINMARKETCAP_API_KEY != 'YOUR_API_KEY_HERE' and \
    #        config.COINMARKETCAP_API_KEY:
    #         cmc_api_key = config.COINMARKETCAP_API_KEY
    #         # btc_5m_cmc = fetch_coinmarketcap_data(symbol='BTC', interval='5m', count=10, api_key=cmc_api_key)
    #         # if btc_5m_cmc is not None:
    #         #     print("\nBTC 5-minute data from CoinMarketCap (last 10 intervals):")
    #         #     print(btc_5m_cmc.head())
    #     else:
    #         # print("CoinMarketCap API key not found or not set in config.py. Skipping CMC examples.")
    #         pass # Silently pass for this version
    # except ImportError:
    #     # print("config.py not found. Skipping CoinMarketCap examples.")
    #     pass # Silently pass for this version
    # except Exception as e:
    #     # print(f"An error occurred in CoinMarketCap example section: {e}")
    #     pass

    # print("\nFull data_fetcher.py script finished (yfinance part executed).")
