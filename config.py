# config.py
import os

# === Data Fetching & General ===
SYMBOL = 'AAPL'
# For backtester.py and signal_generator.py example:
PRIMARY_INTERVAL_BACKTEST = '1h'
PRIMARY_PERIOD_BACKTEST = '250d' # For fetching data for backtesting
# For lstm_model_trainer.py example:
PRIMARY_INTERVAL_LSTM = '1h'
PRIMARY_PERIOD_LSTM = '300d' # More data might be better for LSTM training

HIGHER_TF_INTERVAL = '1d' # Higher timeframe for MTF context in signal generation

# === Market Structure Analysis (applied in pre_processing within main scripts) ===
MS_SWING_WINDOW = 10             # Default for find_swing_highs_lows (e.g., 10 for 1h/1d, 5 for 5m)
MS_PULLBACK_TREND_LOOKBACK = 2
MS_BREAKOUT_LOOKBACK_PERIOD = 20 # e.g., 20 bars
MS_SR_MIN_TOUCHES = 2
MS_SR_RELATIVE_TOLERANCE = 0.015 # 1.5% for S/R level clustering in general market_structure examples

# === Candlestick Patterns (candlestick_patterns.py defaults, can be overridden if needed) ===
CP_DOJI_BODY_TOLERANCE_RATIO = 0.05
CP_MARUBOZU_BODY_MIN_RATIO = 0.8
CP_MARUBOZU_WICK_MAX_RATIO = 0.1

# === Signal Generation (signal_generator.py) ===
SG_SR_NEARBY_PERCENTAGE = 0.02       # How close to S/R for signal
SG_STOP_LOSS_BUFFER_PERCENTAGE = 0.005 # 0.5% buffer for SL
SG_REWARD_RATIO = 2.0                # e.g., 2:1 reward to risk

# === LSTM Configuration ===
# Feature columns list - must match what the LSTM model was trained on
LSTM_FEATURE_COLS = [
    'return', 'body_range_norm', 'hl_range_norm', 'Volume',
    'is_doji', 'is_marubozu', 'is_outside_bar', 'is_inside_bar',
    'is_swing_high', 'is_swing_low', 'is_pullback_bar',
    'is_bullish_breakout', 'is_bearish_breakout'
]

# For Feature Engineering (lstm_feature_engineer.py - used by trainer)
LSTM_FE_FUTURE_N_BARS = 6       # Predict direction 6 bars (e.g., 6 hours for 1h data)
LSTM_FE_SWING_WINDOW = MS_SWING_WINDOW
LSTM_FE_BREAKOUT_LOOKBACK = MS_BREAKOUT_LOOKBACK_PERIOD
LSTM_FE_PULLBACK_TREND_LOOKBACK = MS_PULLBACK_TREND_LOOKBACK # Corrected: Use the MS_ version

# For Model Training (lstm_model_trainer.py)
LSTM_TRAIN_SEQUENCE_LENGTH = 20
LSTM_TRAIN_TEST_SIZE = 0.2
LSTM_TRAIN_EPOCHS = 15          # Reduced for example runs, increase for real training (e.g., 50-100)
LSTM_TRAIN_BATCH_SIZE = 32
LSTM_TRAIN_UNITS = 50
LSTM_TRAIN_DROPOUT_RATE = 0.2

# For Signal Generation (signal_generator.py - LSTM filter part)
SG_USE_LSTM_FILTER = True
SG_LSTM_SEQUENCE_LENGTH = LSTM_TRAIN_SEQUENCE_LENGTH
SG_LSTM_BUY_THRESHOLD = 0.55
SG_LSTM_SELL_THRESHOLD = 0.45

MODEL_DIR = "trading_bot_artifacts"
LSTM_MODEL_NAME = 'best_lstm_model.keras'
LSTM_SCALER_NAME = 'lstm_scaler.gz'
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, LSTM_MODEL_NAME)
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, LSTM_SCALER_NAME)

# === Backtesting (backtester.py) ===
BT_INITIAL_CAPITAL = 10000.0
BT_TRADE_SIZE_PERCENTAGE = 0.10

# === Specific settings for script examples ===
EX_LSTM_TRAINER_SYMBOL = SYMBOL
EX_LSTM_TRAINER_INTERVAL = PRIMARY_INTERVAL_LSTM
EX_LSTM_TRAINER_PERIOD = PRIMARY_PERIOD_LSTM

EX_SG_PRIMARY_SYMBOL = SYMBOL
EX_SG_PRIMARY_INTERVAL = PRIMARY_INTERVAL_BACKTEST
EX_SG_PRIMARY_PERIOD = PRIMARY_PERIOD_BACKTEST
EX_SG_HIGHER_TF_INTERVAL = HIGHER_TF_INTERVAL

EX_BT_PRIMARY_SYMBOL = SYMBOL
EX_BT_PRIMARY_INTERVAL = PRIMARY_INTERVAL_BACKTEST
EX_BT_PRIMARY_PERIOD = PRIMARY_PERIOD_BACKTEST
EX_BT_HIGHER_TF_INTERVAL = HIGHER_TF_INTERVAL

EX_PREPROC_SWING_WINDOW = MS_SWING_WINDOW # Use general MS settings for preprocessing in examples
EX_PREPROC_BREAKOUT_LOOKBACK = MS_BREAKOUT_LOOKBACK_PERIOD
EX_PREPROC_PULLBACK_LOOKBACK = MS_PULLBACK_TREND_LOOKBACK
EX_PREPROC_SR_MIN_TOUCHES = MS_SR_MIN_TOUCHES
EX_PREPROC_SR_TOLERANCE = MS_SR_RELATIVE_TOLERANCE
