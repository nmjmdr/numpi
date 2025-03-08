#!/usr/bin/env python3
"""
Script to load a trained LSTM model and analyze a given stock.
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
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the last time step
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
    # Ensure date column is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    return df


def preprocess_stock_data(df):
    """
    Preprocess stock data:
      - Sort by date.
      - Compute log-transformed open, high, low, and close.
      - Compute the previous day's log_close.
      - Compute the log True Range and ATR over a 252-day rolling window.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["log_open"] = np.log(df["open"])
    df["log_high"] = np.log(df["high"])
    df["log_low"] = np.log(df["low"])
    df["log_close"] = np.log(df["close"])
    df["prev_log_close"] = df["log_close"].shift(1)

    # Define the log true range calculation
    def compute_log_true_range(row):
        if pd.isna(row["prev_log_close"]):
            return row["log_high"] - row["log_low"]
        return max(
            row["log_high"] - row["log_low"],
            abs(row["log_high"] - row["prev_log_close"]),
            abs(row["log_low"] - row["prev_log_close"]),
        )

    df["log_true_range"] = df.apply(compute_log_true_range, axis=1)
    # ATR over 252 days: simple moving average of log_true_range
    df["ATR_252"] = df["log_true_range"].rolling(window=252).mean()
    return df


def construct_samples(df, window_size=11):
    """
    Constructs prediction samples from the preprocessed data.
    For each valid day (ensuring we have at least window_size days of data
    and a non-zero ATR for the current day), build an input sample using
    the 11-day window of features [log_open, log_high, log_low, log_close].

    Returns:
      - dates: list of dates corresponding to each sample (the last day in the window)
      - inputs_arr: numpy array of shape (num_samples, window_size, 4)
      - log_close_values: numpy array of the current day's log_close for each sample.
    """
    dates = []
    inputs = []
    log_close_values = []

    # The starting index must allow for both the window and the ATR computation.
    start_idx = max(window_size - 1, 251)
    for i in range(start_idx, len(df)):
        window_data = df.iloc[i - window_size + 1 : i + 1]
        if len(window_data) < window_size:
            continue
        # Ensure ATR is available and non-zero for the current day
        atr_value = df.iloc[i]["ATR_252"]
        if pd.isna(atr_value) or atr_value == 0:
            continue

        # Construct the input sample from the 11-day window (4 features per day)
        sample = window_data[
            ["log_open", "log_high", "log_low", "log_close"]
        ].values.astype(np.float32)
        inputs.append(sample)
        dates.append(df.iloc[i]["date"])
        log_close_values.append(df.iloc[i]["log_close"])

    if len(inputs) == 0:
        return None, None, None
    inputs_arr = np.array(inputs)  # shape: (num_samples, window_size, 4)
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

    # Fetch and preprocess data for the stock
    print(f"Fetching data for {exchange}:{symbol}...")
    df = fetch_stock_data(exchange, symbol)
    if df.empty:
        print("No data found for the specified stock.")
        return
    print("Preprocessing data...")
    df_processed = preprocess_stock_data(df)

    # Construct samples (using a window_size of 11 days)
    window_size = 11
    print("Constructing samples...")
    dates, X_samples, log_close_values = construct_samples(
        df_processed, window_size=window_size
    )
    if dates is None:
        print(
            "Not enough data to construct samples. Check the dataset and ATR computation."
        )
        return

    # Load the trained neural net model
    device = torch.device("cpu")
    model = LSTMModel(input_size=4, hidden_size=100, num_layers=3, dropout=0.15)
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Compute neural net outputs for each sample
    nn_outputs = []
    with torch.no_grad():
        for sample in X_samples:
            sample_tensor = torch.tensor(sample).unsqueeze(0).to(device)  # (1, 11, 4)
            output = model(sample_tensor)
            nn_outputs.append(output.item())
    nn_outputs = np.array(nn_outputs)

    # Compute the entropy of the NN output distribution
    nn_entropy = compute_entropy(nn_outputs, bins=20)
    # Compute the IQR ratio: (IQR of NN outputs) / (IQR of log_close values)
    iqr_ratio = compute_iqr_ratio(nn_outputs)

    print(f"Neural Net Output Entropy: {nn_entropy:.4f}")
    print(f"IQR Ratio (NN outputs / log_close): {iqr_ratio:.4f}")

    # -----------------------------
    # Plotting: Create subplots with shared x-axis
    # -----------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Log Close Prices", "Neural Net Output"),
    )

    # Subplot 1: Log Close Prices
    fig.add_trace(
        go.Scatter(x=dates, y=log_close_values, mode="lines", name="Log Close Prices"),
        row=1,
        col=1,
    )

    # Subplot 2: Neural Net Outputs
    fig.add_trace(
        go.Scatter(x=dates, y=nn_outputs, mode="lines", name="Neural Net Output"),
        row=2,
        col=1,
    )

    # Compute the overall x-axis range from the data
    min_date = min(dates)
    max_date = max(dates)

    # Update x-axis for both subplots so that they share the same range.
    fig.update_xaxes(range=[min_date, max_date], row=1, col=1, showticklabels=True)
    fig.update_xaxes(
        range=[min_date, max_date],
        row=2,
        col=1,
        showticklabels=True,
        rangeslider=dict(visible=True),
    )

    # Ensure both x-axes match (this forces them to use the same tick settings)
    fig.update_xaxes(matches="x", row=1, col=1)
    fig.update_xaxes(matches="x", row=2, col=1)

    # Update layout with common titles and labels
    fig.update_layout(
        title=f"{exchange}:{symbol} Analysis<br>Entropy: {nn_entropy:.4f}, IQR Ratio: {iqr_ratio:.4f}",
        xaxis_title="Date",
        yaxis_title="Log Close Prices",
        yaxis2_title="NN Output",
        hovermode="x unified",
    )

    # Display the interactive plot
    fig.show()


if __name__ == "__main__":
    main()
