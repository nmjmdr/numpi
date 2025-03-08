#!/usr/bin/env python3
"""
This program prepares the data required by the neural net,
loads the trained LSTM model (with embedded metadata) from a .pth file,
computes the neural net outputs (r_series) for each test day,
and then uses the profit_factor_module to find the optimal thresholds and profit factors.
It then builds an interactive Plotly graph:
  - The top subplot shows the close price series with markers at long and short entries/exits.
  - The bottom subplot shows the neural net outputs (r_series) with horizontal lines for l_theta and s_theta.
The graph is scrollable and zoomable.
The program accepts a stock parameter (--stock) in the format <exchange>:<symbol> to analyze a specific stock.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from profit_factor_module import find_optimal_thresholds
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Database Configuration and Data Fetching
# -----------------------------
DATABASE_URL = "postgresql://trader:trader@localhost:5432/tradingdb"
engine = create_engine(DATABASE_URL)


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
      - Compute the log True Range and ATR over a 252-day rolling window.
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
    df["ATR_252"] = df["log_true_range"].rolling(window=252).mean()
    return df


def construct_test_samples(df, test_start_date, test_end_date, window_size):
    """
    For each day between test_start_date and test_end_date, constructs an input sample
    consisting of an 11-day window of features (as defined in metadata).

    Returns:
      - test_dates: List of dates corresponding to each sample (the last day in the window).
      - samples: NumPy array of shape (num_samples, window_size, 4).
      - close_series: List of close prices for the test dates.
    """
    df = df.sort_values("date").reset_index(drop=True)
    # Ensure enough data: start 300 days before test_start_date.
    min_date = pd.to_datetime(test_start_date) - pd.Timedelta(days=300)
    df_test = df[df["date"] >= min_date].reset_index(drop=True)

    test_dates = []
    samples = []
    close_series = []

    for i in range(window_size - 1, len(df_test)):
        current_date = df_test.loc[i, "date"]
        if current_date < pd.to_datetime(
            test_start_date
        ) or current_date > pd.to_datetime(test_end_date):
            continue
        if pd.isna(df_test.loc[i, "ATR_252"]):
            continue
        window_data = df_test.iloc[i - window_size + 1 : i + 1]
        if len(window_data) < window_size:
            continue
        sample = window_data[
            ["log_open", "log_high", "log_low", "log_close"]
        ].values.astype(np.float32)
        samples.append(sample)
        test_dates.append(current_date)
        close_series.append(df_test.loc[i, "close"])

    samples = np.array(samples)
    return test_dates, samples, close_series


# -----------------------------
# Neural Network Model Definition (same as used in training)
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


def generate_trade_markers(
    r_series, dates, close_series, l_theta, s_theta, is_positive_corr
):
    """
    Generate trade entry and exit markers.
    For is_positive_corr False:
      - Long entry when r_value <= l_theta, exit when r_value > l_theta.
      - Short entry when r_value >= s_theta, exit when r_value < s_theta.
    Returns lists of (date, price) for long entries, long exits, short entries, and short exits.
    """
    long_entries = []
    long_exits = []
    short_entries = []
    short_exits = []
    long_open = False
    short_open = False
    for i in range(len(r_series)):
        r_val = r_series[i]
        date = dates[i]
        price = close_series[i]
        if not long_open and (r_val <= l_theta):
            long_entries.append((date, price))
            long_open = True
        if long_open and (r_val > l_theta):
            long_exits.append((date, price))
            long_open = False
        if not short_open and (r_val >= s_theta):
            short_entries.append((date, price))
            short_open = True
        if short_open and (r_val < s_theta):
            short_exits.append((date, price))
            short_open = False
    return long_entries, long_exits, short_entries, short_exits


def main():
    parser = argparse.ArgumentParser(
        description="Profit Factor Analysis using Neural Net"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="lstm_model.pth",
        help="Path to the saved neural net model",
    )
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Stock to analyze in the format <exchange>:<symbol>",
    )
    parser.add_argument(
        "--test_start_date",
        type=str,
        default="2020-01-01",
        help="Test start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test_end_date",
        type=str,
        default="2020-12-31",
        help="Test end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min_num_trades",
        type=int,
        default=5,
        help="Minimum number of trades required",
    )
    args = parser.parse_args()

    try:
        exchange, symbol = args.stock.split(":")
    except ValueError:
        print(
            "Error: Stock must be provided in the format <exchange>:<symbol> (e.g., ASX:XYZ)"
        )
        return

    print(f"Fetching data for {exchange}:{symbol} ...")
    df = fetch_stock_data(exchange, symbol)
    if df.empty:
        print("No data found for the specified stock.")
        return
    print("Preprocessing data...")
    df_processed = preprocess_stock_data(df)

    # Load the model package (state_dict + metadata)
    device = torch.device("cpu")
    print(f"Loading model from {args.model_path}...")
    model_package = torch.load(args.model_path, map_location=device)
    if isinstance(model_package, dict) and "metadata" in model_package:
        state_dict = model_package["state_dict"]
        metadata = model_package["metadata"]
    else:
        state_dict = model_package
        metadata = {
            "input_window_size": 11,
            "atr_period": 252,
            "feature_columns": ["log_open", "log_high", "log_low", "log_close"],
            "is_positive_corr": False,
        }

    # Use metadata to set parameters.
    window_size = metadata.get("input_window_size", 11)
    is_positive_corr = metadata.get("is_positive_corr", False)

    print("Constructing test samples...")
    test_dates, test_samples, close_series = construct_test_samples(
        df_processed, args.test_start_date, args.test_end_date, window_size
    )
    if len(test_samples) == 0:
        print(
            "No test samples constructed. Check the date range and data availability."
        )
        return

    model = LSTMModel(input_size=4, hidden_size=100, num_layers=3, dropout=0.15)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Compute neural net outputs (r_series)
    r_series = []
    with torch.no_grad():
        for sample in test_samples:
            sample_tensor = torch.tensor(sample).unsqueeze(0).to(device)
            output = model(sample_tensor)
            r_series.append(output.item())
    r_series = np.array(r_series)

    # Prepare x_series for profit factor module.
    x_series = pd.DataFrame({"date": test_dates, "close": close_series})
    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    print("Computing optimal thresholds and profit factors...")
    l_theta, long_pf, s_theta, short_pf = find_optimal_thresholds(
        r_series, x_series, bins, is_positive_corr, args.min_num_trades
    )

    print("\nOptimal Thresholds and Profit Factors:")
    print(f"Long Position (l_theta): {l_theta}, Profit Factor: {long_pf}")
    print(f"Short Position (s_theta): {s_theta}, Profit Factor: {short_pf}")

    long_entries, long_exits, short_entries, short_exits = generate_trade_markers(
        r_series, test_dates, close_series, l_theta, s_theta, is_positive_corr
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "Close Price with Trade Markers",
            "Neural Net Output (r_series)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=test_dates, y=close_series, mode="lines", name="Close Price"),
        row=1,
        col=1,
    )
    if long_entries:
        le_dates, le_prices = zip(*long_entries)
        fig.add_trace(
            go.Scatter(
                x=le_dates,
                y=le_prices,
                mode="markers+text",
                name="Long Entry",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                text=["T"] * len(le_dates),
                textposition="top center",
            ),
            row=1,
            col=1,
        )
    if long_exits:
        lx_dates, lx_prices = zip(*long_exits)
        fig.add_trace(
            go.Scatter(
                x=lx_dates,
                y=lx_prices,
                mode="markers+text",
                name="Long Exit",
                marker=dict(symbol="triangle-down", size=10, color="green"),
                text=["T"] * len(lx_dates),
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )
    if short_entries:
        se_dates, se_prices = zip(*short_entries)
        fig.add_trace(
            go.Scatter(
                x=se_dates,
                y=se_prices,
                mode="markers+text",
                name="Short Entry",
                marker=dict(symbol="triangle-up", size=10, color="red"),
                text=["T"] * len(se_dates),
                textposition="top center",
            ),
            row=1,
            col=1,
        )
    if short_exits:
        sx_dates, sx_prices = zip(*short_exits)
        fig.add_trace(
            go.Scatter(
                x=sx_dates,
                y=sx_prices,
                mode="markers+text",
                name="Short Exit",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                text=["T"] * len(sx_dates),
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=r_series,
            mode="lines",
            name="r_series",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=l_theta,
        line_dash="dash",
        line_color="green",
        annotation_text=f"l_theta = {l_theta}",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=s_theta,
        line_dash="dash",
        line_color="red",
        annotation_text=f"s_theta = {s_theta}",
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{exchange}:{symbol} Analysis<br>Long PF: {long_pf:.2f}, Short PF: {short_pf:.2f}",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        hovermode="x unified",
        margin=dict(t=50, b=50),
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    fig.update_yaxes(title_text="r_series", row=2, col=1)
    fig.show()


if __name__ == "__main__":
    main()
