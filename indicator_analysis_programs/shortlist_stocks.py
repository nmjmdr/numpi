#!/usr/bin/env python3
"""
This program loads a neural net model (with embedded metadata) and, for each stock in the database:
  - Loads all available trading data,
  - Prepares data based on the model’s metadata (input_window_size, atr_period, etc.),
  - Runs inference to get an r_series,
  - Computes the IQR ratio of the r_series; if it is ≥ 5, it adjusts the r_series
    (using a signed log transformation with progressive clipping),
  - Computes optimal long and short thresholds (l_theta and s_theta) and profit factors
    via the profit_factor_module (using the is_positive_corr flag from metadata),
  - If either profit factor is at least the configured threshold (pf_theta), the stock is shortlisted.
Finally, it prints a table ranking the shortlisted stocks by max profit factor, entropy (higher is better), and IQR ratio (lower is better).
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from datetime import datetime
from scipy.stats import entropy as scipy_entropy
from profit_factor_module import find_optimal_thresholds


# -----------------------------
# Entropy and IQR Adjustment Functions
# -----------------------------
def calculate_entropy(data, n_bins=None):
    """
    Calculate Shannon entropy of the data using equal-width binning.
    If n_bins is not provided, uses Sturges' Rule.
    """
    if n_bins is None:
        n_bins = int(np.ceil(np.log2(len(data)) + 1))
    data_min, data_max = np.min(data), np.max(data)
    if data_max == data_min:
        return 0.0
    bin_width = (data_max - data_min) / n_bins
    bin_edges = np.arange(data_min, data_max + bin_width, bin_width)
    hist, _ = np.histogram(data, bins=bin_edges, density=True)
    hist = hist[hist > 0]
    return scipy_entropy(hist, base=2)


def adjust_iqr_ratio(
    data, target_IQR_ratio=5, deviation_tolerance=0.5, entropy_tolerance=0.1
):
    """
    Adjusts the IQR ratio of a time series (data) to be below target_IQR_ratio by applying
    a signed log transformation (x -> sign(x)*log(1+|x|)) and progressive clipping.
    Returns the adjusted data, final IQR ratio, and final entropy.
    """
    # Use signed log transformation.
    data_transformed = np.sign(data) * np.log(1 + np.abs(data))

    Q1 = np.percentile(data_transformed, 25)
    Q3 = np.percentile(data_transformed, 75)
    IQR = Q3 - Q1
    median_val = np.median(data_transformed)
    IQR_ratio = IQR / (np.abs(median_val) + 1e-6)
    print(f"Initial IQR Ratio (after signed log): {IQR_ratio}")

    if IQR_ratio <= target_IQR_ratio + deviation_tolerance:
        print("IQR Ratio is already below target. No adjustment needed.")
        return data_transformed, IQR_ratio, calculate_entropy(data_transformed)

    print(f"IQR Ratio after signed log transformation: {IQR_ratio}")

    def clip_data(data, lower_percentile=1, upper_percentile=99):
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        return np.clip(data, lower_bound, upper_bound)

    initial_entropy = calculate_entropy(data_transformed)
    print(f"Initial Entropy: {initial_entropy}")
    clip_range = (99, 1)

    while IQR_ratio > target_IQR_ratio + deviation_tolerance:
        data_transformed = clip_data(
            data_transformed,
            lower_percentile=clip_range[1],
            upper_percentile=clip_range[0],
        )
        Q1 = np.percentile(data_transformed, 25)
        Q3 = np.percentile(data_transformed, 75)
        IQR = Q3 - Q1
        median_val = np.median(data_transformed)
        IQR_ratio = IQR / (np.abs(median_val) + 1e-6)
        current_entropy = calculate_entropy(data_transformed)
        entropy_drop = initial_entropy - current_entropy
        if entropy_drop > entropy_tolerance:
            print(f"Entropy dropped too much ({entropy_drop}). Stopping clipping.")
            break
        print(
            f"Current IQR Ratio: {IQR_ratio} (Clipped at {clip_range[0]}-{clip_range[1]} percentile)"
        )
        print(f"Current Entropy: {current_entropy} (Entropy drop: {entropy_drop})")
        clip_range = (clip_range[0] - 1, clip_range[1] + 1)
        if IQR_ratio <= target_IQR_ratio + deviation_tolerance:
            break

    final_entropy = calculate_entropy(data_transformed)
    print(f"Final IQR Ratio: {IQR_ratio}")
    print(f"Final Entropy: {final_entropy}")
    return data_transformed, IQR_ratio, final_entropy


# -----------------------------
# Database Functions
# -----------------------------
DATABASE_URL = "postgresql://trader:trader@localhost:5432/tradingdb"
engine = create_engine(DATABASE_URL)


def fetch_symbols():
    """Fetch distinct stock symbols from the database."""
    query = "SELECT DISTINCT(symbol) FROM instrument_trades WHERE exchange='ASX'"
    with engine.connect() as conn:
        symbols = pd.read_sql(query, conn)["symbol"].tolist()
    return symbols


def fetch_stock_data(exchange, symbol):
    """Fetch trading data for a given stock."""
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
    """Sort by date, compute log prices, and compute log true range."""
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
    """Compute ATR over atr_period (simple moving average of log_true_range)."""
    df["ATR_252"] = df["log_true_range"].rolling(window=atr_period).mean()
    return df


def construct_test_samples(df, test_start_date, test_end_date, window_size):
    """
    For each day between test_start_date and test_end_date, construct an input sample
    using a window of 'window_size' days.
    Returns:
      - test_dates: List of dates corresponding to each sample (the last day in the window).
      - samples: NumPy array of shape (num_samples, window_size, 4).
      - close_series: List of raw close prices for the sample day.
    """
    df = df.sort_values("date").reset_index(drop=True)
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
# Neural Net Model Definition
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
# Trade Marker Generation
# -----------------------------
def generate_trade_markers(
    r_series, dates, close_series, l_theta, s_theta, is_positive_corr
):
    """
    Generate trade entry and exit markers.
    For is_positive_corr False:
      - Long entry when r_value <= l_theta; exit when r_value > l_theta.
      - Short entry when r_value >= s_theta; exit when r_value < s_theta.
    Returns four lists of (date, price) tuples.
    """
    long_entries, long_exits, short_entries, short_exits = [], [], [], []
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


# -----------------------------
# Main Processing Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Shortlist stocks based on neural net outputs and profit factors"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the neural net model file (with metadata)",
    )
    parser.add_argument(
        "--pf_theta",
        type=float,
        default=1.2,
        help="Profit factor threshold for shortlisting stocks",
    )
    args = parser.parse_args()

    # Load model package and metadata.
    device = torch.device("cpu")
    model_package = torch.load(args.model_path, map_location=device)
    if isinstance(model_package, dict) and "metadata" in model_package:
        state_dict = model_package["state_dict"]
        metadata = model_package["metadata"]
    else:
        print("Error: Model file does not contain metadata.")
        return

    window_size = metadata.get("input_window_size", 11)
    atr_period = metadata.get("atr_period", 252)
    is_positive_corr = metadata.get("is_positive_corr", False)

    # Initialize model.
    model = LSTMModel(input_size=4, hidden_size=100, num_layers=3, dropout=0.15)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Candidate bins for threshold search.
    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    symbols = fetch_symbols()
    results = []

    for symbol in symbols:
        print(f"\nProcessing stock: {symbol}")
        df = fetch_stock_data("ASX", symbol)
        if df.empty:
            print("No data found; skipping.")
            continue
        df = preprocess_stock_data(df)
        df = compute_atr(df, atr_period)
        # Here we use a fixed test date range covering all available years.
        test_dates, test_samples, close_values = construct_test_samples(
            df, "2000-01-01", "2050-12-31", window_size
        )
        if test_dates is None or len(test_samples) == 0:
            print("Not enough data for samples; skipping.")
            continue

        # Run inference to compute r_series.
        r_series = []
        with torch.no_grad():
            for sample in test_samples:
                sample_tensor = torch.tensor(sample).unsqueeze(0).to(device)
                output = model(sample_tensor)
                r_series.append(output.item())
        r_series = np.array(r_series)

        if r_series.size == 0:
            print(f"No valid r_series computed for {symbol}; skipping.")
            continue

        print(
            f"r_series for {symbol}: min={np.min(r_series)}, max={np.max(r_series)}, median={np.median(r_series)}"
        )

        Q1 = np.percentile(r_series, 25)
        Q3 = np.percentile(r_series, 75)
        IQR_value = Q3 - Q1
        median_val = np.median(r_series)
        orig_iqr_ratio = IQR_value / (np.abs(median_val) + 1e-6)
        print(f"Original IQR Ratio for {symbol}: {orig_iqr_ratio}")

        if orig_iqr_ratio >= 5:
            print("IQR ratio is high; adjusting r_series...")
            r_series_adjusted, final_iqr_ratio, final_entropy = adjust_iqr_ratio(
                r_series, target_IQR_ratio=5
            )
        else:
            r_series_adjusted = r_series
            final_iqr_ratio = orig_iqr_ratio
            final_entropy = calculate_entropy(r_series)

        # Prepare x_series for profit factor module.
        x_series = pd.DataFrame({"date": test_dates, "close": close_values})

        l_theta, long_pf, s_theta, short_pf = find_optimal_thresholds(
            r_series_adjusted, x_series, bins, is_positive_corr, min_num_trades=4
        )
        # If both profit factors are -inf, skip this stock.
        if long_pf == float("-inf") and short_pf == float("-inf"):
            print(f"No valid trades for {symbol}; skipping.")
            continue

        print(
            f"Stock {symbol}: Long PF = {long_pf}, l_theta = {l_theta}; Short PF = {short_pf}, s_theta = {s_theta}"
        )
        max_pf = max(long_pf, short_pf)
        if max_pf >= args.pf_theta:
            results.append(
                {
                    "Stock": symbol,
                    "Long PF": long_pf,
                    "l_theta": l_theta,
                    "Short PF": short_pf,
                    "s_theta": s_theta,
                    "Max PF": max_pf,
                    "IQR Ratio": final_iqr_ratio,
                    "Entropy": final_entropy,
                }
            )

    if not results:
        print("No stocks met the profit factor threshold.")
        return

    df_results = pd.DataFrame(results)
    df_results.sort_values(
        by=["Max PF", "Entropy", "IQR Ratio"],
        ascending=[False, False, True],
        inplace=True,
    )
    print("\nShortlisted Stocks:")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
