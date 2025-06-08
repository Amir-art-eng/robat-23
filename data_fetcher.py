import yfinance as yf
import pandas as pd
import os

DEFAULT_DATA_DIRECTORY = "market_data"

def save_data_to_csv(data: pd.DataFrame, symbol: str, interval: str, data_directory: str = DEFAULT_DATA_DIRECTORY):
    """
    Saves the DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): The pandas DataFrame to save.
        symbol (str): The stock symbol.
        interval (str): The data interval.
        data_directory (str): The directory where data files will be stored.
    """
    if data is None or data.empty:
        print(f"No data provided for {symbol}_{interval}, skipping save.")
        return

    try:
        os.makedirs(data_directory, exist_ok=True)
        filename = f"{symbol.upper()}_{interval}.csv"
        filepath = os.path.join(data_directory, filename)
        data.to_csv(filepath)
        print(f"Data for {symbol}_{interval} saved to {filepath}")
    except Exception as e:
        print(f"Error saving data for {symbol}_{interval} to CSV: {e}")

def load_data_from_csv(symbol: str, interval: str, data_directory: str = DEFAULT_DATA_DIRECTORY) -> pd.DataFrame | None:
    """
    Loads data from a CSV file if it exists.

    Args:
        symbol (str): The stock symbol.
        interval (str): The data interval.
        data_directory (str): The directory where data files are stored.

    Returns:
        pd.DataFrame | None: Loaded DataFrame or None if file doesn't exist or error occurs.
    """
    filename = f"{symbol.upper()}_{interval}.csv"
    filepath = os.path.join(data_directory, filename)

    if os.path.exists(filepath):
        try:
            # Assuming the index is the first column and needs to be parsed as dates/datetime
            # yfinance typically uses a DatetimeIndex
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Data for {symbol}_{interval} loaded from {filepath}")
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}_{interval} from CSV: {e}")
            return None
    else:
        # print(f"No cached data found for {symbol}_{interval} at {filepath}")
        return None

def fetch_price_data(symbol: str, interval: str, period: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame | None:
    """
    Fetches historical stock price data using yfinance, with CSV caching.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL').
        interval (str): The data interval (e.g., '1m', '5m', '15m', '1h', '1d').
        period (str, optional): How far back to fetch data.
        start_date (str, optional): Start date string in 'YYYY-MM-DD' format.
        end_date (str, optional): End date string in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: OHLCV data, or None if an error occurs.
    """
    force_refetch = period is not None or (start_date is not None and end_date is not None)
    cached_data = load_data_from_csv(symbol, interval)

    if cached_data is not None:
        if force_refetch:
            print(f"Loaded data for {symbol}_{interval} from cache. However, specific period/dates were requested, so refetching.")
        else:
            # If no specific period/dates, and cache exists, return cached data
            print(f"Using cached data for {symbol}_{interval} as no specific period/date range was requested.")
            return cached_data

    # If cache not found, or if force_refetch is True
    print(f"Fetching fresh data for {symbol}_{interval} from yfinance...")
    try:
        ticker = yf.Ticker(symbol)

        # Determine fetch parameters for yfinance
        # yfinance's .history() is flexible. If period is given, it's prioritized.
        # If start/end are given, they are used.
        # If NEITHER period nor start/end are given, yfinance defaults to 'max' period for daily,
        # or a shorter period for intraday (e.g., last 60 days for many intervals if period is not 'max').
        # For our cache saving, we want to save the most comprehensive data possible if no specific range is given.

        fetch_params = {'interval': interval}
        if period:
            fetch_params['period'] = period
        elif start_date and end_date:
            fetch_params['start'] = start_date
            fetch_params['end'] = end_date
        else:
            # No specific period or date range from user.
            # For caching, let's attempt to get a good amount of data.
            # yfinance default for daily is 'max'. For intraday, it's often '60d' or '730d' for '1h'.
            # To make the cache more generally useful when no period is specified,
            # we might fetch 'max' for daily, or a substantial period for intraday.
            # However, 'max' for 1m data can be huge and slow.
            # Let's stick to yfinance defaults if no period/dates are given by user,
            # but be aware of what this means for the saved CSV.
            # If we want a "full" cache, we might explicitly set a long period here.
            # For now, let yfinance decide the default if no specific range.
            # The problem states: "If ONLY symbol and interval are provided (meaning the user implies they want 'whatever is generally available or cached for this')"
            # This means if we fetch, we fetch what yfinance gives by default.
            pass # yfinance will use its default period based on interval

        data = ticker.history(**fetch_params)

        if data.empty:
            print(f"No data found for symbol {symbol} with parameters: {fetch_params}")
            return None

        print(f"Successfully fetched {len(data)} rows of new data for {symbol}_{interval}.")
        # Save the newly fetched data
        # If user requested a specific period/date, the CSV will store that specific slice.
        # If user did not specify period/date, CSV stores yfinance's default (e.g. 'max' for daily)
        save_data_to_csv(data, symbol, interval)
        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}_{interval} from yfinance: {e}")
        return None

if __name__ == "__main__":
    # Ensure the data directory exists for the examples, useful for caching if implemented
    os.makedirs(DEFAULT_DATA_DIRECTORY, exist_ok=True)

    print("\n--- Example 1: Fetch 1-minute data for 'MSFT' for the last 1 day ---")
    # Note: 1m data is usually available for the last 7 days if period is '1d'.
    # yfinance behavior: '1d' period for intraday data might mean 'data of the last 1 day available'
    # or 'data up to 1 day ago from today'. For '1m', yfinance usually limits 'period' to 7 days.
    # So, using period='1d' for 1m interval should give data for the most recent trading day.
    msft_1m = fetch_price_data(symbol='MSFT', interval='1m', period='1d')
    if msft_1m is not None:
        print(f"MSFT 1-minute data for last 1 day (shape {msft_1m.shape}):")
        print(msft_1m.head())
        print(msft_1m.tail()) # Good to see the end too for period checks
    else:
        print("Failed to fetch MSFT 1m data for 1d period.")

    print("\n--- Example 2: Fetch 5-minute data for 'AAPL' for the period '5d' ---")
    aapl_5m = fetch_price_data(symbol='AAPL', interval='5m', period='5d')
    if aapl_5m is not None:
        print(f"AAPL 5-minute data for last 5 days (shape {aapl_5m.shape}):")
        print(aapl_5m.head())
        print(aapl_5m.tail())
    else:
        print("Failed to fetch AAPL 5m data for 5d period.")

    print("\n--- Example 3: Fetch daily data for 'GOOGL' from '2023-01-01' to '2023-12-31' ---")
    googl_daily = fetch_price_data(symbol='GOOGL', interval='1d', start_date='2023-01-01', end_date='2023-12-31')
    if googl_daily is not None:
        print(f"GOOGL daily data from 2023-01-01 to 2023-12-31 (shape {googl_daily.shape}):")
        print(googl_daily.head())
        print(googl_daily.tail())
    else:
        print("Failed to fetch GOOGL daily data for the specified date range.")

    print("\n--- Example 4: Fetch 1-hour data for 'NONEXISTENTICKER' ---")
    invalid_data = fetch_price_data(symbol='NONEXISTENTICKER', interval='1h', period='1d')
    if invalid_data is None:
        print("Correctly handled non-existent ticker 'NONEXISTENTICKER'. An error message should have been printed by the function.")
    else:
        print("Unexpectedly received data for 'NONEXISTENTICKER'.")
        print(invalid_data.head())

    print("\n--- Check content of market_data directory (if caching is active) ---")
    if os.path.exists(DEFAULT_DATA_DIRECTORY):
        print(f"Files in '{DEFAULT_DATA_DIRECTORY}': {os.listdir(DEFAULT_DATA_DIRECTORY)}")
    else:
        print(f"Directory '{DEFAULT_DATA_DIRECTORY}' not found or caching not active.")

    print("\nData fetcher example usage finished.")
