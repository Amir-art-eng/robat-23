import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow is not installed. LSTM model training will not be possible.")
    TENSORFLOW_AVAILABLE = False

try:
    from lstm_feature_engineer import create_lstm_features
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    print("Warning: lstm_feature_engineer.py not found. LSTM model training will be limited.")
    FEATURE_ENGINEER_AVAILABLE = False

try:
    from data_fetcher import fetch_price_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    print("Warning: data_fetcher.py not found. Example usage will be limited.")
    DATA_FETCHER_AVAILABLE = False

try:
    import config
except ImportError:
    print("Error: config.py not found. Using fallback internal defaults for lstm_model_trainer.")
    class config:
        EX_LSTM_TRAINER_SYMBOL = 'AAPL'; EX_LSTM_TRAINER_INTERVAL = '1h'; EX_LSTM_TRAINER_PERIOD = '250d';
        LSTM_FE_FUTURE_N_BARS = 5; LSTM_FE_SWING_WINDOW = 10;
        LSTM_FE_BREAKOUT_LOOKBACK = 24; LSTM_FE_PULLBACK_TREND_LOOKBACK = 2;
        LSTM_TRAIN_SEQUENCE_LENGTH = 20; LSTM_TRAIN_TEST_SIZE = 0.2;
        MODEL_DIR = "."; LSTM_SCALER_NAME = "lstm_scaler.gz"; LSTM_MODEL_NAME = "best_lstm_model.keras";
        LSTM_TRAIN_UNITS = 50; LSTM_TRAIN_DROPOUT_RATE = 0.2;
        LSTM_TRAIN_EPOCHS = 15; LSTM_TRAIN_BATCH_SIZE = 32;
        LSTM_SCALER_PATH = os.path.join(MODEL_DIR, LSTM_SCALER_NAME) # Added for fallback
        LSTM_MODEL_PATH = os.path.join(MODEL_DIR, LSTM_MODEL_NAME)   # Added for fallback


def prepare_data_for_lstm(feature_df: pd.DataFrame, sequence_length: int, test_size: float = 0.2):
    if feature_df.empty or len(feature_df) < sequence_length:
        print("Error: Feature DataFrame is empty or too short for the given sequence length.")
        return None, None, None, None, None

    X_raw = feature_df.drop('target_price_direction', axis=1)
    y_raw = feature_df['target_price_direction']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_sequences = []
    y_sequences = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_sequences.append(X_scaled[i : i + sequence_length])
        y_sequences.append(y_raw.iloc[i + sequence_length - 1])
    if not X_sequences:
        print("Error: Not enough data to create sequences after processing.")
        return None, None, None, None, None
    X_sequences = np.array(X_sequences); y_sequences = np.array(y_sequences)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=test_size, shuffle=False )
    print(f"Data prepared: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape: tuple, lstm_units: int, dropout_rate: float) -> Sequential | None: # Removed defaults to ensure config is used
    if not TENSORFLOW_AVAILABLE: return None
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    if not all([DATA_FETCHER_AVAILABLE, FEATURE_ENGINEER_AVAILABLE, TENSORFLOW_AVAILABLE, 'config' in globals()]):
        print("Essential modules or config missing. Cannot run LSTM trainer example.")
    else:
        print("Running lstm_model_trainer.py example using config.py...")
        print("\n--- LSTM Training Configuration ---")
        print(f"Symbol for Training: {config.EX_LSTM_TRAINER_SYMBOL}")
        print(f"Interval for Training Data: {config.EX_LSTM_TRAINER_INTERVAL}")
        print(f"Period for Training Data: {config.EX_LSTM_TRAINER_PERIOD}")
        print(f"LSTM Sequence Length: {config.LSTM_TRAIN_SEQUENCE_LENGTH}")
        print(f"Future N Bars (Target): {config.LSTM_FE_FUTURE_N_BARS}")
        print(f"Test Split Size: {config.LSTM_TRAIN_TEST_SIZE}")
        print(f"Epochs: {config.LSTM_TRAIN_EPOCHS}")
        print(f"Batch Size: {config.LSTM_TRAIN_BATCH_SIZE}")
        print(f"LSTM Units: {config.LSTM_TRAIN_UNITS}")
        print(f"Dropout Rate: {config.LSTM_TRAIN_DROPOUT_RATE}")
        if hasattr(config, 'MODEL_DIR'): # Check if MODEL_DIR is defined
             print(f"Model Directory: {config.MODEL_DIR}")
        print(f"Model Save Path: {config.LSTM_MODEL_PATH}")
        print(f"Scaler Save Path: {config.LSTM_SCALER_PATH}")
        print("--- Feature Engineering Params Used ---")
        print(f"  Swing Window: {config.LSTM_FE_SWING_WINDOW}")
        print(f"  Breakout Lookback: {config.LSTM_FE_BREAKOUT_LOOKBACK}")
        print(f"  Pullback Trend Lookback: {config.LSTM_FE_PULLBACK_TREND_LOOKBACK}")
        print("------------------------------------")

        SEQ_LENGTH = config.LSTM_TRAIN_SEQUENCE_LENGTH
        FUTURE_N_BARS = config.LSTM_FE_FUTURE_N_BARS
        EPOCHS = config.LSTM_TRAIN_EPOCHS
        BATCH_SIZE = config.LSTM_TRAIN_BATCH_SIZE
        TEST_SPLIT_SIZE = config.LSTM_TRAIN_TEST_SIZE
        LSTM_UNITS = config.LSTM_TRAIN_UNITS
        DROPOUT_RATE = config.LSTM_TRAIN_DROPOUT_RATE

        symbol = config.EX_LSTM_TRAINER_SYMBOL
        interval = config.EX_LSTM_TRAINER_INTERVAL
        period = config.EX_LSTM_TRAINER_PERIOD

        fe_swing_window = config.LSTM_FE_SWING_WINDOW
        fe_breakout_lookback = config.LSTM_FE_BREAKOUT_LOOKBACK
        fe_pullback_trend_lookback = config.LSTM_FE_PULLBACK_TREND_LOOKBACK

        # Ensure MODEL_DIR exists
        model_dir = getattr(config, 'MODEL_DIR', 'trading_bot_artifacts') # Default if not in config
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        scaler_filename = config.LSTM_SCALER_PATH
        model_checkpoint_path = config.LSTM_MODEL_PATH # This is now the full path from config

        print(f"\nFetching data for {symbol}, interval {interval}, period {period}...")
        ohlcv_df = fetch_price_data(symbol=symbol, interval=interval, period=period)

        if ohlcv_df is not None and not ohlcv_df.empty:
            print(f"Successfully fetched {len(ohlcv_df)} data points for {symbol}.")
            print("\nCreating LSTM features...")
            feature_df = create_lstm_features(
                ohlcv_df,
                future_n_bars=FUTURE_N_BARS,
                swing_window=fe_swing_window,
                breakout_lookback=fe_breakout_lookback,
                pullback_trend_lookback=fe_pullback_trend_lookback )

            if feature_df is not None and not feature_df.empty:
                print(f"Feature engineering complete. Shape: {feature_df.shape}")
                print("\nPreparing data for LSTM...")
                X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(
                    feature_df, sequence_length=SEQ_LENGTH, test_size=TEST_SPLIT_SIZE )

                if X_train is not None and len(X_train) > 0:
                    joblib.dump(scaler, scaler_filename)
                    print(f"Scaler saved to {scaler_filename}")
                    print(f"Data prepared: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
                    print("\nBuilding LSTM model...")
                    model = build_lstm_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE )

                    if model:
                        model.summary()
                        print("\nTraining LSTM model...")
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
                        model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
                        history = model.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_split=0.1, callbacks=[early_stopping, model_checkpoint], verbose=1 )
                        print("Model training finished.")
                        print("\n--- Model Evaluation & Save ---")
                        final_model_to_evaluate = None
                        try:
                            # Load the best model saved by ModelCheckpoint
                            if os.path.exists(model_checkpoint_path):
                                final_model_to_evaluate = tf.keras.models.load_model(model_checkpoint_path)
                                print(f"Successfully loaded best model from {model_checkpoint_path} for evaluation.")
                            else:
                                print(f"Warning: Best model checkpoint not found at {model_checkpoint_path}. Evaluating with the model from the last epoch.")
                                final_model_to_evaluate = model # Fallback to model from last epoch

                            test_loss, test_accuracy = final_model_to_evaluate.evaluate(X_test, y_test, verbose=0)
                            print(f"Test Loss: {test_loss:.4f}")
                            print(f"Test Accuracy: {test_accuracy*100:.2f}%")
                            # ModelCheckpoint already saved the best model, so no explicit save here unless it's a different path/name
                            print(f"Trained model (best during training) saved to: {model_checkpoint_path}")
                            print(f"Scaler saved to: {scaler_filename}")

                        except Exception as e:
                            print(f"Error during model evaluation or loading best model: {e}.")
                            if model: # If training happened but loading best failed, try evaluating 'model' instance
                                print("Evaluating model from last epoch due to error with best_model loading...")
                                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                                print(f"Test Loss (last epoch model): {test_loss:.4f}")
                                print(f"Test Accuracy (last epoch model): {test_accuracy*100:.2f}%")
                                print(f"Note: Model from last epoch might not be the best performing one.")
                                print(f"Scaler was saved to: {scaler_filename}")
                                print(f"Model (last epoch, if not overwritten by checkpoint) would be implicitly at: {model_checkpoint_path} if checkpointing failed early.")
                            else:
                                print("No model available for evaluation.")
                else:
                    print("Not enough data to create training/test sequences for LSTM.")
            else:
                print("Feature DataFrame is empty after feature engineering. Cannot proceed.")
        else:
            print(f"Could not fetch data for {symbol}. LSTM training aborted.")
        print("\nlstm_model_trainer.py example finished.")
