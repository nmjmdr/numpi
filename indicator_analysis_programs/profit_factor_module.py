# profit_factor_module.py
import numpy as np


def long_condition(r_value, threshold, is_positive_corr):
    """
    Determines if a long position should be entered.
    For positive correlation: enter when r_value >= threshold.
    For negative correlation: enter when r_value <= threshold.
    """
    return r_value >= threshold if is_positive_corr else r_value <= threshold


def short_condition(r_value, threshold, is_positive_corr):
    """
    Determines if a short position should be entered.
    For positive correlation: enter when r_value < threshold.
    For negative correlation: enter when r_value >= threshold.
    """
    return r_value < threshold if is_positive_corr else r_value >= threshold


def simulate_trades(r_series, close_series, threshold, is_positive_corr, trade_type):
    """
    Simulates trades based on a candidate threshold.

    Parameters:
      r_series: Array of indicator values.
      close_series: Array of close prices corresponding to r_series.
      threshold: The candidate threshold value.
      is_positive_corr: Boolean flag indicating correlation direction.
      trade_type: "long" or "short".

    Returns:
      profit_factor: Ratio (sum of winning returns) / (sum of absolute losing returns)
      num_trades: Number of trades executed.
    """
    trades = []
    i = 0
    n = len(r_series)

    while i < n - 1:  # Ensure there's at least one day available for return calculation
        if trade_type == "long":
            entry_condition = long_condition(r_series[i], threshold, is_positive_corr)
        elif trade_type == "short":
            entry_condition = short_condition(r_series[i], threshold, is_positive_corr)
        else:
            raise ValueError("trade_type must be 'long' or 'short'")

        if entry_condition:
            trade_returns = []
            j = i + 1
            # Maintain the trade until the condition fails.
            while j < n:
                if trade_type == "long":
                    if not long_condition(r_series[j], threshold, is_positive_corr):
                        break
                    # For long trades, use normal log return.
                    daily_return = np.log(close_series[j] / close_series[j - 1])
                else:  # short trade
                    if not short_condition(r_series[j], threshold, is_positive_corr):
                        break
                    # For short trades, invert the log return.
                    daily_return = -np.log(close_series[j] / close_series[j - 1])
                trade_returns.append(daily_return)
                j += 1
            trades.append(trade_returns)
            i = j  # Jump to the day after trade exit
        else:
            i += 1

    total_wins = 0.0
    total_losses = 0.0
    for trade in trades:
        for ret in trade:
            if ret > 0:
                total_wins += ret
            elif ret < 0:
                total_losses += abs(ret)

    if total_losses == 0:
        profit_factor = float("inf") if total_wins > 0 else 0.0
    else:
        profit_factor = total_wins / total_losses

    num_trades = len(trades)
    return profit_factor, num_trades


def find_optimal_thresholds(r_series, x_series, bins, is_positive_corr, min_num_trades):
    """
    Finds the optimal thresholds for long (l_theta) and short (s_theta) positions separately.

    Parameters:
      r_series: NumPy array of indicator values (e.g., neural net outputs) for each day.
      x_series: DataFrame or dict-like object containing at least a 'close' column corresponding to r_series.
      bins: List of candidate threshold values.
      is_positive_corr: Boolean flag (for our case, typically False).
      min_num_trades: Minimum number of trades required for a candidate threshold to be valid.

    Returns:
      l_theta, long_profit_factor, s_theta, short_profit_factor
    """
    # Extract close prices from x_series (assumes a 'close' column)
    close_series = (
        x_series["close"].values
        if hasattr(x_series, "columns")
        else np.array(x_series["close"])
    )
    r_series = np.array(r_series)

    best_long_profit_factor = -np.inf
    l_theta = None
    best_short_profit_factor = -np.inf
    s_theta = None

    # Evaluate candidate thresholds for long positions independently.
    for candidate in bins:
        pf, num_trades = simulate_trades(
            r_series, close_series, candidate, is_positive_corr, "long"
        )
        if num_trades >= min_num_trades and pf > best_long_profit_factor:
            best_long_profit_factor = pf
            l_theta = candidate

    # Evaluate candidate thresholds for short positions independently.
    for candidate in bins:
        pf, num_trades = simulate_trades(
            r_series, close_series, candidate, is_positive_corr, "short"
        )
        if num_trades >= min_num_trades and pf > best_short_profit_factor:
            best_short_profit_factor = pf
            s_theta = candidate

    return l_theta, best_long_profit_factor, s_theta, best_short_profit_factor
