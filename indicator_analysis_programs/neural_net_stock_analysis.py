#!/usr/bin/env python3
"""
Script to load a trained LSTM model (with embedded metadata) and analyze a given stock.
Usage: python analyze_stock.py <model_path> <exchange:symbol>
Example: python analyze_stock.py lstm_model.pth ASX:XYZ
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# -----------------------------
# Database Configuration
# -----------------------------
DATABASE_URL = "postgresql://trader:trader@localhost:5432/tradingdb"
engine = create_engine(DATABASE_URL)


# -----------------------------
# Model Definition (same as training code)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_layers=3, dropout=0.15):
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
# Data Fetching and Preprocessing
# -----------------------------
def fetch_stock_data(exchange, symbol):
    """
    Fetch trading data for the specified stock from the database.
    """
    query = f"""
        SELECT date, open, high, low, close
        FROM instrument_trades
        WHERE exchange = '{exchange}' AND symbol = '{symbol}'
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def preprocess_stock_data(df):
    """
    Preprocess stock data:
      - Sort by date.
      - Compute log-transformed open, high, low, and close.
      - Compute the previous day's log_close.
      - Compute the log True Range and ATR based on the atr_period defined in metadata.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["log_open"] = np.log(df["open"])
    df["log_high"] = np.log(df["high"])
    df["log_low"] = np.log(df["low"])
    df["log_close"] = np.log(df["close"])
    df["prev_log_close"] = df["log_close"].shift(1)

    def compute_log_true_range(row):
        if pd.isna(row["prev_log_close"]):
            return row["log_high"] - row["log_low"]
        return max(
            row["log_high"] - row["log_low"],
            abs(row["log_high"] - row["prev_log_close"]),
            abs(row["log_low"] - row["prev_log_close"]),
        )

    df["log_true_range"] = df.apply(compute_log_true_range, axis=1)
    return df


def compute_atr(df, atr_period):
    """
    Computes the ATR over the specified period from metadata.
    """
    df["ATR"] = df["log_true_range"].rolling(window=atr_period).mean()
    return df


def construct_samples(df, window_size):
    """
    Constructs prediction samples from the preprocessed data using the window size from metadata.
    Returns:
      - dates: list of dates corresponding to each sample (the last day in the window)
      - inputs_arr: numpy array of shape (num_samples, window_size, num_features)
      - log_close_values: numpy array of the current day's log_close for each sample.
    """
    dates = []
    inputs = []
    log_close_values = []

    start_idx = max(window_size - 1, df["ATR"].first_valid_index() or 0)

    for i in range(start_idx, len(df)):
        window_data = df.iloc[i - window_size + 1 : i + 1]
        if len(window_data) < window_size:
            continue

        atr_value = df.iloc[i]["ATR"]
        if pd.isna(atr_value) or atr_value == 0:
            continue

        sample = window_data[
            ["log_open", "log_high", "log_low", "log_close"]
        ].values.astype(np.float32)
        inputs.append(sample)
        dates.append(df.iloc[i]["date"])
        log_close_values.append(df.iloc[i]["log_close"])

    if len(inputs) == 0:
        return None, None, None

    inputs_arr = np.array(inputs)
    return dates, inputs_arr, np.array(log_close_values)


# -----------------------------
# Metrics: Entropy and IQR Ratio
# -----------------------------
def compute_entropy(data, bins=20):
    """
    Compute the Shannon entropy of the distribution of the data.
    """
    counts, _ = np.histogram(data, bins=bins)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    entropy_val = -np.sum(probabilities * np.log(probabilities))
    return entropy_val


def compute_iqr_ratio(data):
    """
    Compute the IQR ratio defined as:
        IQR_ratio = IQR(data1) / IQR(data2)
    where IQR is the difference between the 75th and 25th percentiles.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return float("nan")
    return (data.max() - data.min()) / iqr


# -----------------------------
# Main Analysis Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Neural Net Stock Analysis")
    parser.add_argument("model_path", help="Path to the saved neural net model file")
    parser.add_argument(
        "stock", help="Stock to analyze in the format exchange:symbol (e.g. ASX:XYZ)"
    )
    args = parser.parse_args()

    try:
        exchange, symbol = args.stock.split(":")
    except ValueError:
        print(
            "Error: Stock argument must be in the format exchange:symbol (e.g. ASX:XYZ)"
        )
        return

    print(f"Fetching data for {exchange}:{symbol}...")
    df = fetch_stock_data(exchange, symbol)
    if df.empty:
        print("No data found for the specified stock.")
        return

    print("Preprocessing data...")
    df = preprocess_stock_data(df)

    print(f"Loading model from {args.model_path}...")
    model_package = torch.load(args.model_path, map_location="cpu")

    if isinstance(model_package, dict) and "metadata" in model_package:
        state_dict = model_package["state_dict"]
        metadata = model_package["metadata"]
    else:
        print("Error: Model file does not contain metadata.")
        return

    window_size = metadata.get("input_window_size", 11)
    atr_period = metadata.get("atr_period", 252)

    df = compute_atr(df, atr_period)

    print("Constructing samples...")
    dates, X_samples, log_close_values = construct_samples(df, window_size)
    if dates is None:
        print("Not enough data to construct samples.")
        return

    model = LSTMModel(input_size=4, hidden_size=100, num_layers=3, dropout=0.15)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    nn_outputs = []
    with torch.no_grad():
        for sample in X_samples:
            sample_tensor = torch.tensor(sample).unsqueeze(0).to("cpu")
            output = model(sample_tensor)
            nn_outputs.append(output.item())

    nn_outputs = np.array(nn_outputs)
    print(f"{nn_outputs}")

    data = {
        "Date": dates,
        "Neural_Net_Output": nn_outputs,
        "Log_Close": log_close_values,
    }

    df_output = pd.DataFrame(data)

    # Save to CSV
    output_filename = f"{exchange}_{symbol}_analysis.csv"
    output_path = os.path.join(os.getcwd(), output_filename)

    df_output.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

    nn_entropy = compute_entropy(nn_outputs, bins=20)
    iqr_ratio = compute_iqr_ratio(nn_outputs)

    print(f"Neural Net Output Entropy: {nn_entropy:.4f}")
    print(f"IQR Ratio (NN outputs / log_close): {iqr_ratio:.4f}")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Log Close Prices", "Neural Net Output"),
    )

    fig.add_trace(
        go.Scatter(x=dates, y=log_close_values, mode="lines", name="Log Close Prices"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=nn_outputs, mode="lines", name="Neural Net Output"),
        row=2,
        col=1,
    )

    fig.update_layout(title=f"{exchange}:{symbol} Analysis", hovermode="x unified")
    fig.show()


if __name__ == "__main__":
    main()
