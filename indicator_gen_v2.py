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
    df["date"] = pd.to_datetime(df["date"])
    return df


# -----------------------------
# Data Preprocessing Functions
# -----------------------------


def preprocess_data(df):
    """
    For each symbol, compute log-transformed prices and ATR (over 252 days) using log-prices.
    Returns a new DataFrame with additional columns.
    """
    processed_dfs = []
    for symbol in df["symbol"].unique():
        df_symbol = (
            df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        )
        df_symbol["log_open"] = np.log(df_symbol["open"])
        df_symbol["log_high"] = np.log(df_symbol["high"])
        df_symbol["log_low"] = np.log(df_symbol["low"])
        df_symbol["log_close"] = np.log(df_symbol["close"])
        df_symbol["prev_log_close"] = df_symbol["log_close"].shift(1)

        def compute_log_true_range(row):
            if pd.isna(row["prev_log_close"]):
                return row["log_high"] - row["log_low"]
            return max(
                row["log_high"] - row["log_low"],
                abs(row["log_high"] - row["prev_log_close"]),
                abs(row["log_low"] - row["prev_log_close"]),
            )

        df_symbol["log_true_range"] = df_symbol.apply(compute_log_true_range, axis=1)
        df_symbol["ATR_252"] = df_symbol["log_true_range"].rolling(window=252).mean()
        processed_dfs.append(df_symbol)
    df_processed = pd.concat(processed_dfs, ignore_index=True)
    return df_processed


def construct_samples(df, window_size=20, horizon=10):
    """
    Constructs training samples from the preprocessed DataFrame.
    For each symbol, an input sample is a 'window_size'-day window of log-prices
    (each day: [log_open, log_high, log_low, log_close]).

    Define the equilibrium as the average log_close over the input window.
    The target is defined as:

      target = (future_avg_log_close - equilibrium) / ATR_252

    where future_avg_log_close is the average log_close over the next 'horizon' days.
    """
    X_samples = []
    y_samples = []
    for symbol in df["symbol"].unique():
        df_symbol = (
            df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        )
        if len(df_symbol) < max(window_size, 252 + horizon):
            continue
        start_idx = max(window_size - 1, 251)
        for i in range(start_idx, len(df_symbol) - horizon):
            window_data = df_symbol.iloc[i - window_size + 1 : i + 1]
            if len(window_data) < window_size:
                continue
            input_features = window_data[
                ["log_open", "log_high", "log_low", "log_close"]
            ].values.astype(np.float32)
            equilibrium = window_data[
                "log_close"
            ].mean()  # average over the input window
            future_window = df_symbol.iloc[i + 1 : i + 1 + horizon]
            if len(future_window) < horizon:
                continue
            future_avg = future_window["log_close"].mean()
            atr_value = df_symbol.iloc[i]["ATR_252"]
            if pd.isna(atr_value) or atr_value == 0:
                continue
            target = (future_avg - equilibrium) / atr_value
            X_samples.append(input_features)
            y_samples.append(target)
    X_arr = np.array(X_samples)
    y_arr = np.array(y_samples).astype(np.float32)
    return X_arr, y_arr


# -----------------------------
# PyTorch Dataset Definition
# -----------------------------


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# LSTM Model Definition using PyTorch
# -----------------------------


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_layers=3, dropout=0.20):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# -----------------------------
# Loss Function
# -----------------------------


def mean_reversion_loss(predictions, targets):
    """
    Custom loss function: Mean Squared Error.
    """
    loss = torch.mean((predictions - targets) ** 2)
    return loss


# -----------------------------
# Training Function with Early Stopping
# -----------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    patience=10,
    learning_rate=0.002,
    device="cpu",
):
    criterion = mean_reversion_loss
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break
    return model


# -----------------------------
# Main Pipeline
# -----------------------------


def main():
    device = torch.device("cpu")  # Mac M1 without MPS support
    print("Fetching data from database...")
    df_raw = fetch_data()
    print("Preprocessing data...")
    df_processed = preprocess_data(df_raw)
    print("Constructing training samples...")
    X, y = construct_samples(df_processed, window_size=20, horizon=10)
    if X.size == 0:
        print("No valid samples found. Check data quality and ATR computation.")
        return
    print(f"Total samples: {X.shape[0]}")
    dataset = TimeSeriesDataset(X, y)
    total_samples = len(dataset)
    val_samples = int(total_samples * 0.1)
    train_samples = total_samples - val_samples
    train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    model = LSTMModel(input_size=4, hidden_size=100, num_layers=3, dropout=0.20)
    model.to(device)
    print("Training the model...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        patience=15,
        learning_rate=0.003,
        device=device,
    )
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
    model_save_path = "lstm_model.pth"
    metadata = {
        "input_window_size": 20,
        "atr_period": 252,
        "feature_columns": ["log_open", "log_high", "log_low", "log_close"],
        "target_calculation": "(avg(log_close of input window) - avg(log_close of next 10 days)) / ATR_252",
        "model_type": "LSTMModel",
        "training_date": datetime.now().isoformat(),
    }
    model_package = {"state_dict": model.state_dict(), "metadata": metadata}
    torch.save(model_package, model_save_path)
    print(f"Model saved to {model_save_path} with embedded metadata.")


if __name__ == "__main__":
    main()
