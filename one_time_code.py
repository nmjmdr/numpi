#!/usr/bin/env python3
import torch
from datetime import datetime
import argparse


def update_model_metadata(model_path):
    # Load the existing model file
    model_data = torch.load(model_path, map_location="cpu")
    # If already a package with metadata, then do nothing.
    if isinstance(model_data, dict) and "metadata" in model_data:
        print("Model already contains metadata. No update performed.")
        return

    # Otherwise, assume model_data is the state_dict.
    state_dict = model_data
    # Define the common metadata dictionary.
    metadata = {
        "input_window_size": 11,
        "atr_period": 252,
        "feature_columns": ["log_open", "log_high", "log_low", "log_close"],
        "target_calculation": "(current_log_close - average(log_close of previous 10 days)) / ATR_252",
        "model_type": "LSTMModel",
        "training_date": datetime.now().isoformat(),
        "is_positive_corr": False,
    }
    # Create the package.
    model_package = {"state_dict": state_dict, "metadata": metadata}
    # Save the package back to the same file.
    torch.save(model_package, model_path)
    print(f"Updated model saved to {model_path} with embedded metadata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update existing .pth model file with metadata."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the existing .pth model file.",
    )
    args = parser.parse_args()
    update_model_metadata(args.model_path)
