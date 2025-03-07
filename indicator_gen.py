#!/usr/bin/env python3
"""
Complete working code for training a neural network (LSTM) for predicting mean reversion
with volatility awareness. Data is loaded from a PostgreSQL database.
"""

import pandas as pd
import numpy as np
import psycopg2  # psycopg2-binary should be installed
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

# -----------------------------
# Database Configuration & Data Loading
# -----------------------------

DATABASE_URL = "postgresql://trader:trader@localhost:5432/tradingdb"
engine = create_engine(DATABASE_URL)


def fetch_symbols():
    """Fetch distinct stock symbols from the database."""
    with engine.connect() as conn:
        query = "SELECT DISTINCT(symbol) FROM instrument_trades WHERE exchange='ASX'"
        return pd.read_sql(query, conn)["symbol"].tolist()


def fetch_data():
    """Fetch all trading data for ASX from the database."""
    with engine.connect() as conn:
        query = """
            SELECT symbol, date, open, high, low, close
            FROM instrument_trades
            WHERE exchange='ASX'
            ORDER BY symbol, date
        """
        df = pd.read_sql(query, conn)
    # Ensure date is in datetime format and sort properly
    df["date"] = pd.to_datetime(df["date"])
    return df


# -----------------------------
# Data Preprocessing Functions
# -----------------------------


def preprocess_data(df):
    """
    For each symbol, compute log-transformed prices and ATR (over 252 days).
    Returns a new DataFrame with additional columns.
    """
    processed_dfs = []
    for symbol in df["symbol"].unique():
        df_symbol = (
            df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        )
        # Compute log prices
        df_symbol["log_open"] = np.log(df_symbol["open"])
        df_symbol["log_high"] = np.log(df_symbol["high"])
        df_symbol["log_low"] = np.log(df_symbol["low"])
        df_symbol["log_close"] = np.log(df_symbol["close"])

        # Compute previous close for True Range calculation
        df_symbol["prev_close"] = df_symbol["close"].shift(1)

        # Compute True Range for each day
        def compute_true_range(row):
            if pd.isna(row["prev_close"]):
                return row["high"] - row["low"]
            return max(
                row["high"] - row["low"],
                abs(row["high"] - row["prev_close"]),
                abs(row["low"] - row["prev_close"]),
            )

        df_symbol["true_range"] = df_symbol.apply(compute_true_range, axis=1)

        # Compute ATR over a 252-day rolling window (simple moving average of true range)
        df_symbol["ATR_252"] = df_symbol["true_range"].rolling(window=252).mean()
        processed_dfs.append(df_symbol)

    df_processed = pd.concat(processed_dfs, ignore_index=True)
    return df_processed


def construct_samples(df, window_size=11):
    """
    Constructs training samples from the preprocessed DataFrame.
    For each symbol, an input sample is an 11-day window (each day: [log_open, log_high, log_low, log_close])
    and the target is the volatility-adjusted deviation of the current day's log_close from the mean of the previous 10 days.
    Ensures numerical consistency (dividing by exactly 10) and skips samples with missing/zero ATR.
    """
    X_samples = []
    y_samples = []

    # Process each symbol separately
    for symbol in df["symbol"].unique():
        df_symbol = (
            df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        )
        # Ensure there are enough rows (at least window_size and 252 rows for ATR)
        if len(df_symbol) < max(window_size, 252):
            continue

        # Loop from index where ATR is available and window can be formed:
        # ATR is computed starting at index 251 (0-indexed), so start from i = max(window_size-1, 251)
        start_idx = max(window_size - 1, 251)
        for i in range(start_idx, len(df_symbol)):
            # Input window: from i-window_size+1 to i (inclusive)
            window_data = df_symbol.iloc[i - window_size + 1 : i + 1]
            # Skip if the window is not complete (shouldn't happen)
            if len(window_data) < window_size:
                continue

            # Extract input features: log_open, log_high, log_low, log_close
            input_features = window_data[
                ["log_open", "log_high", "log_low", "log_close"]
            ].values.astype(np.float32)

            # Compute target:
            # Use the current day's log_close (last row) and average log_close of previous 10 days (window_size - 1)
            current_log_close = window_data.iloc[-1]["log_close"]
            # Average over previous window_size - 1 days (ensuring numerical consistency, denominator = window_size - 1)
            avg_log_close = window_data.iloc[:-1]["log_close"].mean()
            # Get the ATR for the current day; skip sample if ATR is missing or zero
            atr_value = df_symbol.iloc[i]["ATR_252"]
            if pd.isna(atr_value) or atr_value == 0:
                continue

            # Compute target as volatility-adjusted deviation (denom is window_size - 1)
            target = (current_log_close - avg_log_close) / atr_value

            X_samples.append(input_features)
            y_samples.append(target)

    X_arr = np.array(X_samples)  # Shape: (num_samples, window_size, 4)
    y_arr = np.array(y_samples).astype(np.float32)  # Shape: (num_samples,)
    return X_arr, y_arr


# -----------------------------
# PyTorch Dataset Definition
# -----------------------------


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array of shape (num_samples, window_size, num_features)
        y: numpy array of shape (num_samples,)
        """
        self.X = torch.from_numpy(X)  # Convert to torch tensor (float32 by default)
        self.y = torch.from_numpy(y).unsqueeze(1)  # (num_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# LSTM Model Definition using PyTorch
# -----------------------------


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_layers=3, dropout=0.15):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: batch_first=True so input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size, hidden_size=100, num_layers=2, batch_first=True, dropout=dropout
        )
        # Fully-connected layer for regression output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states with zeros (using CPU)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Pass through LSTM layer; output shape: (batch, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Use the last time step's output for regression
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# -----------------------------
# Training Function with Early Stopping
# -----------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    patience=10,
    learning_rate=0.003,
    device="cpu",
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            best_model_state = model.state_dict()  # Save best model state
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)  # restore best state
                break
    return model


# -----------------------------
# Main Pipeline
# -----------------------------


def main():
    # Set device to CPU (Mac M1 without MPS support)
    device = torch.device("cpu")

    # 1. Load data from database
    print("Fetching data from database...")
    df_raw = fetch_data()

    # 2. Preprocess data (compute log prices and ATR)
    print("Preprocessing data...")
    df_processed = preprocess_data(df_raw)

    # 3. Construct training samples (window_size=11 days)
    print("Constructing training samples...")
    X, y = construct_samples(df_processed, window_size=11)
    if X.size == 0:
        print("No valid samples found. Check data quality and ATR computation.")
        return

    print(f"Total samples: {X.shape[0]}")

    # 4. Create PyTorch Dataset and split into training and validation sets (90/10 split)
    dataset = TimeSeriesDataset(X, y)
    total_samples = len(dataset)
    val_samples = int(total_samples * 0.1)
    train_samples = total_samples - val_samples
    train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])

    # 5. Create DataLoaders (batch size = 64)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 6. Build the LSTM model
    input_size = 4  # [log_open, log_high, log_low, log_close]
    model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=1, dropout=0.3)
    model.to(device)

    # 7. Train the model with early stopping
    print("Training the model...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        patience=10,
        learning_rate=0.003,
        device=device,
    )

    # 8. Evaluate final performance on validation set
    model.eval()
    criterion = nn.MSELoss()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"Final Validation Loss: {avg_val_loss:.6f}")

    # 9. Save the trained model to a file for later use
    model_save_path = "lstm_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
